"""Strict historical inputs and the audited movement-evaluation cohort ledger."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import pandas as pd

from predictor_v5.movement_evaluation.contract import (
    INPUT_IDENTITIES,
    POSITIONS,
    ContractRefusal,
    KtcMode,
    key_sha256,
    validate_ktc_mode,
)

LABEL_COLUMNS = (
    "player_id", "season", "start_value", "end_value", "position", "age", "games", "points",
    "start_evidence", "end_evidence", "start_trades_window", "end_trades_window",
    "end_value_next_board", "end_evidence_next_board",
)
ARCHIVE_COLUMNS = ("player_id", "season", "position", "age", "points", "games", "years_exp")
_NUMERIC_LABEL_COLUMNS = (
    "start_value", "end_value", "age", "points", "start_evidence", "end_evidence",
    "start_trades_window", "end_trades_window", "end_value_next_board", "end_evidence_next_board",
)


@dataclass(frozen=True)
class HistoricalBundle:
    archive: pd.DataFrame
    labels: pd.DataFrame
    provenance: dict[str, object]
    identities: dict[str, str]


@dataclass(frozen=True)
class EvaluationData:
    joined: pd.DataFrame
    primary_scored: pd.DataFrame
    observed_zero_end: pd.DataFrame
    thin_endpoint: pd.DataFrame
    coverage: dict[str, object]


def _refuse(message: str, cause: BaseException | None = None) -> NoReturn:
    if cause is None:
        raise ContractRefusal(message)
    raise ContractRefusal(message) from cause


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _refuse("duplicate JSON object key")
        result[key] = value
    return result


def _json_bytes(raw: bytes) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: _refuse("non-finite JSON constant"),
        )
    except ContractRefusal:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _refuse("invalid JSON", exc)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _optional_float(value: Any, field: str) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return float("nan")
    if not _finite(value):
        _refuse(f"{field} must be finite")
    return float(value)


def _integer(value: Any, field: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not float(value).is_integer():
        _refuse(f"{field} must be an integer")
    return int(value)


def _check_year(value: Any) -> int:
    year = _integer(value, "season")
    assert year is not None
    if year < 2020 or year > 2025:
        _refuse("season is outside the historical contract")
    return year


def parse_archive_bytes(raw: bytes, *, include_ktc: bool) -> tuple[pd.DataFrame, str]:
    """Validate and parse the sole pinned training archive member."""
    if not isinstance(raw, bytes):
        _refuse("archive input must be captured bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.namelist()
            if members != ["training-data.json"]:
                _refuse("archive must contain exactly training-data.json")
            member = archive.read("training-data.json")
    except ContractRefusal:
        raise
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        _refuse("invalid training archive", exc)
    member_hash = hashlib.sha256(member).hexdigest()
    payload = _json_bytes(member)
    if not isinstance(payload, dict) or set(payload) != {"players"} or not isinstance(payload["players"], list):
        _refuse("archive JSON must contain a players envelope")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for player in payload["players"]:
        if not isinstance(player, dict):
            _refuse("archive player must be an object")
        player_id = player.get("player_id")
        if player_id is None:
            _refuse("player_id must be a nonempty string")
        player_id = str(player_id)
        if not player_id:
            _refuse("player_id must be a nonempty string")
        position = player.get("position")
        if position not in POSITIONS:
            _refuse("archive position is unsupported")
        seasons = player.get("seasons")
        if not isinstance(seasons, list):
            _refuse("archive seasons must be a list")
        for season_data in seasons:
            if not isinstance(season_data, dict):
                _refuse("archive season must be an object")
            season = _check_year(season_data.get("year"))
            key = (season, player_id)
            if key in seen:
                _refuse("duplicate archive player-season key")
            seen.add(key)
            years_exp_raw = season_data.get("years_exp")
            years_exp = -1 if years_exp_raw is None else _integer(years_exp_raw, "years_exp")
            assert years_exp is not None
            if years_exp < -1:
                _refuse("years_exp is outside its supported range")
            games = _integer(season_data.get("games_played"), "games_played", allow_none=True)
            if games is not None and games < 0:
                _refuse("games_played cannot be negative")
            row: dict[str, Any] = {
                "player_id": player_id,
                "season": season,
                "position": position,
                "age": _optional_float(season_data.get("age"), "age"),
                "points": _optional_float(season_data.get("fantasy_points"), "fantasy_points"),
                "games": games if games is not None else float("nan"),
                "years_exp": years_exp,
            }
            if include_ktc:
                for field in ("start_ktc", "end_ktc"):
                    value = _optional_float(season_data.get(field), field)
                    if value == 0:
                        value = float("nan")
                    elif math.isfinite(value) and not 0 < value <= 9999:
                        _refuse(f"{field} must be positive and at most 9999")
                    row[field] = value
            rows.append(row)
    columns = list(ARCHIVE_COLUMNS) + (["start_ktc", "end_ktc"] if include_ktc else [])
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(["season", "player_id"], kind="mergesort").reset_index(drop=True)
    return frame, member_hash


def parse_label_bytes(raw: bytes) -> pd.DataFrame:
    """Validate the physical label schema and parse its captured bytes."""
    if not isinstance(raw, bytes):
        _refuse("label input must be captured bytes")
    try:
        header = next(csv.reader(io.StringIO(raw.decode("utf-8"))))
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        _refuse("invalid label CSV", exc)
    if len(header) != len(set(header)) or tuple(header) != LABEL_COLUMNS:
        _refuse("label CSV has unexpected physical columns")
    try:
        frame = pd.read_csv(io.BytesIO(raw), dtype={"player_id": "string"}, keep_default_na=True)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        _refuse("invalid label CSV", exc)
    if tuple(frame.columns) != LABEL_COLUMNS:
        _refuse("label CSV has unexpected physical columns")
    frame["player_id"] = frame["player_id"].astype("string")
    if frame["player_id"].isna().any() or (frame["player_id"].str.len() == 0).any():
        _refuse("player_id must be a nonempty string")
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    if frame["season"].isna().any() or (~(frame["season"] % 1 == 0)).any():
        _refuse("season must be an integer")
    frame["season"] = frame["season"].astype(int)
    if ((frame["season"] < 2020) | (frame["season"] > 2025)).any():
        _refuse("season is outside the historical contract")
    if frame.duplicated(["season", "player_id"]).any():
        _refuse("duplicate CSV player-season key")
    if (~frame["position"].isin(POSITIONS)).any():
        _refuse("CSV position is unsupported")
    for column in _NUMERIC_LABEL_COLUMNS:
        original = frame[column].copy()
        frame[column] = pd.to_numeric(original, errors="coerce")
        malformed = original.notna() & original.astype("string").str.strip().ne("") & frame[column].isna()
        if malformed.any():
            _refuse(f"{column} must be numeric")
        values = frame[column].dropna().to_numpy()
        if not all(math.isfinite(float(value)) for value in values):
            _refuse(f"{column} must be finite")
    for column in ("start_value", "end_value", "start_trades_window", "end_trades_window"):
        if (frame[column].dropna() < 0).any():
            _refuse(f"{column} cannot be negative")
    for column in ("start_trades_window", "end_trades_window", "games"):
        if column in frame:
            original = frame[column].copy()
            frame[column] = pd.to_numeric(original, errors="coerce")
            malformed = original.notna() & original.astype("string").str.strip().ne("") & frame[column].isna()
            if malformed.any():
                _refuse(f"{column} must be numeric")
            values = frame[column].dropna()
            if not (values % 1 == 0).all():
                _refuse(f"{column} must be an integer")
    if frame["games"].notna().any() and (frame["games"].dropna() < 0).any():
        _refuse("games cannot be negative")
    return frame.sort_values(["season", "player_id"], kind="mergesort").reset_index(drop=True)


def _path_bytes(path: Path, expected_hash: str, label: str) -> bytes:
    try:
        stat = path.lstat()
    except OSError as exc:
        _refuse(f"missing pinned {label}", exc)
    if not stat or not os.path.isfile(path) or path.is_symlink():
        _refuse(f"pinned {label} must be a regular non-symlink file")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_hash:
        _refuse(f"pinned {label} hash mismatch")
    return raw


def load_pinned_bundle(repo_root: Path, *, ktc_mode: KtcMode) -> HistoricalBundle:
    """Read exactly the three fixed historical inputs after byte validation."""
    mode = validate_ktc_mode(ktc_mode)
    root = Path(repo_root)
    label_spec = INPUT_IDENTITIES["labels"]
    provenance_spec = INPUT_IDENTITIES["provenance"]
    archive_spec = INPUT_IDENTITIES["archive"]
    labels_raw = _path_bytes(root / label_spec["path"], label_spec["sha256"], "labels")
    provenance_raw = _path_bytes(root / provenance_spec["path"], provenance_spec["sha256"], "provenance")
    archive_raw = _path_bytes(root / archive_spec["path"], archive_spec["sha256"], "archive")
    provenance = _json_bytes(provenance_raw)
    if not isinstance(provenance, dict):
        _refuse("provenance must be a JSON object")
    label_file = provenance.get("label_file")
    training_data = provenance.get("training_data")
    if not isinstance(label_file, dict) or label_file.get("file") != Path(label_spec["path"]).name or label_file.get("sha256") != label_spec["sha256"]:
        _refuse("provenance label cross-hash mismatch")
    if not isinstance(training_data, dict) or training_data.get("file") != "training-data.zip" or training_data.get("sha256") != archive_spec["sha256"]:
        _refuse("provenance archive cross-hash mismatch")
    labels = parse_label_bytes(labels_raw)
    archive, member_hash = parse_archive_bytes(archive_raw, include_ktc=mode == "archive")
    if member_hash != archive_spec["member_sha256"]:
        _refuse("archive member hash mismatch")
    identities = {
        "labels_sha256": label_spec["sha256"], "provenance_sha256": provenance_spec["sha256"],
        "archive_sha256": archive_spec["sha256"], "archive_member_sha256": member_hash,
        "labels_path": label_spec["path"], "provenance_path": provenance_spec["path"],
        "archive_path": archive_spec["path"], "archive_member": archive_spec["member"],
    }
    return HistoricalBundle(archive=archive, labels=labels, provenance=provenance, identities=identities)


def _validate_bundle_frames(archive: pd.DataFrame, labels: pd.DataFrame) -> None:
    if not isinstance(archive, pd.DataFrame) or not isinstance(labels, pd.DataFrame):
        _refuse("bundle frames must be DataFrames")
    for frame, required, label in ((archive, ARCHIVE_COLUMNS, "archive"), (labels, LABEL_COLUMNS, "labels")):
        if not set(required).issubset(frame.columns):
            _refuse(f"{label} frame has missing columns")
    if archive.duplicated(["season", "player_id"]).any() or labels.duplicated(["season", "player_id"]).any():
        _refuse("duplicate player-season key")
    if set(archive["position"]) - set(POSITIONS) or set(labels["position"]) - set(POSITIONS):
        _refuse("unsupported position")
    for column in ("age", "points", "games"):
        values = pd.to_numeric(archive[column], errors="coerce")
        original = archive[column]
        malformed = original.notna() & original.astype("string").str.strip().ne("") & values.isna()
        if malformed.any():
            _refuse(f"archive {column} must be numeric")
        if values.notna().any() and not values.dropna().map(lambda value: math.isfinite(float(value))).all():
            _refuse(f"archive {column} must be finite")
    for column in ("age", "points", "games", "start_value", "end_value", "start_evidence", "end_evidence", "start_trades_window", "end_trades_window"):
        if column in labels:
            values = pd.to_numeric(labels[column], errors="coerce")
            original = labels[column]
            malformed = original.notna() & original.astype("string").str.strip().ne("") & values.isna()
            if malformed.any():
                _refuse(f"labels {column} must be numeric")
            if values.notna().any() and not values.dropna().map(lambda value: math.isfinite(float(value))).all():
                _refuse(f"labels {column} must be finite")
    years = pd.to_numeric(archive["years_exp"], errors="coerce")
    if years.isna().any() or (years % 1 != 0).any() or (years < -1).any():
        _refuse("years_exp must be an integer")
    games = pd.to_numeric(archive["games"], errors="coerce").dropna()
    if not (games % 1 == 0).all() or (games < 0).any():
        _refuse("archive games must be a nonnegative integer")
    for frame in (archive, labels):
        for value in frame["season"]:
            _check_year(value)
        for column in ("start_value", "end_value", "start_trades_window", "end_trades_window"):
            if column in frame and (pd.to_numeric(frame[column], errors="coerce").dropna() < 0).any():
                _refuse(f"{column} cannot be negative")


def _population(frame: pd.DataFrame) -> dict[str, object]:
    keys = [(int(s), str(p)) for s, p in frame[["season", "player_id"]].itertuples(index=False, name=None)]
    by_season = {str(season): int((frame["season"] == season).sum()) for season in range(2020, 2026)}
    by_position = {position: int((frame["position"] == position).sum()) for position in POSITIONS}
    return {
        "count": len(frame), "distinct_players": frame["player_id"].nunique(),
        "key_sha256": key_sha256(keys), "by_season": by_season, "by_position": by_position,
    }


def _distribution(frame: pd.DataFrame, column: str) -> dict[str, object]:
    if column not in frame or frame[column].dropna().empty:
        return {"count": 0, "min": None, "max": None, "mean": None}
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return {"count": int(values.size), "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean())}


def prepare_evaluation_data(bundle: HistoricalBundle) -> EvaluationData:
    """Reconcile labels to archive stats and build the fixed eligibility ledger."""
    if not isinstance(bundle, HistoricalBundle):
        _refuse("expected HistoricalBundle")
    archive = bundle.archive.copy()
    labels = bundle.labels.copy()
    _validate_bundle_frames(archive, labels)
    archive_stats = archive.rename(columns={column: f"archive_{column}" for column in ("position", "age", "points", "games", "years_exp")})
    label_frame = labels.rename(columns={column: f"label_{column}" for column in labels.columns if column not in ("season", "player_id")})
    joined = label_frame.merge(archive_stats, on=["season", "player_id"], how="outer", indicator=True, validate="one_to_one")
    joined["join_status"] = joined["_merge"].map({"both": "matched", "left_only": "label_only", "right_only": "archive_only"})
    joined.drop(columns=["_merge"], inplace=True)
    for column in ("position", "age", "points", "games", "years_exp"):
        joined[column] = joined[f"archive_{column}"]
    for column in labels.columns:
        if column not in ("season", "player_id", "position", "age", "points", "games"):
            joined[column] = joined[f"label_{column}"]

    matched = joined["join_status"] == "matched"
    if (matched & (joined["label_position"].astype(str) != joined["archive_position"].astype(str))).any():
        _refuse("CSV/archive position mismatch")
    label_games = pd.to_numeric(joined.loc[matched, "label_games"], errors="coerce")
    archive_games = pd.to_numeric(joined.loc[matched, "archive_games"], errors="coerce")
    if ((label_games.isna() ^ archive_games.isna()) | (label_games.notna() & archive_games.notna() & (label_games != archive_games))).any():
        _refuse("CSV/archive games mismatch")
    for field in ("age", "points"):
        left = pd.to_numeric(joined.loc[matched, f"label_{field}"], errors="coerce")
        right = pd.to_numeric(joined.loc[matched, f"archive_{field}"], errors="coerce")
        both_finite = left.notna() & right.notna()
        if ((left.isna() ^ right.isna()) | (both_finite & ((left - right).abs() > 1e-9))).any():
            _refuse(f"CSV/archive {field} mismatch")

    joined["is_pre_nfl"] = (joined["years_exp"] == 0) & (joined["games"] == 0)
    joined["pre_nfl"] = joined["is_pre_nfl"]
    joined["not_joined_to_archive"] = joined["join_status"] != "matched"
    joined["unknown_experience"] = joined["years_exp"] == -1
    finite = lambda series: series.map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
    joined["missing_age"] = ~finite(pd.to_numeric(joined["age"], errors="coerce"))
    joined["missing_start"] = ~finite(pd.to_numeric(joined["label_start_value"], errors="coerce"))
    joined["nonpositive_start"] = (~joined["missing_start"]) & (joined["label_start_value"] <= 0)
    joined["missing_start_liquidity"] = joined["label_start_trades_window"].isna()
    joined["low_start_liquidity"] = (~joined["missing_start_liquidity"]) & (joined["label_start_trades_window"] < 5)
    joined["missing_points"] = ~finite(pd.to_numeric(joined["points"], errors="coerce"))
    joined["missing_end"] = ~finite(pd.to_numeric(joined["label_end_value"], errors="coerce"))
    joined["zero_end"] = (~joined["missing_end"]) & (joined["label_end_value"] == 0)
    joined["missing_end_liquidity"] = joined["label_end_trades_window"].isna()
    joined["low_end_liquidity"] = (~joined["missing_end_liquidity"]) & (joined["label_end_trades_window"] < 5)
    stages = (
        ("not_joined_to_archive", joined["join_status"] != "matched"), ("pre_nfl", joined["is_pre_nfl"]),
        ("missing_age", joined["missing_age"]), ("missing_start", joined["missing_start"]),
        ("nonpositive_start", joined["nonpositive_start"]), ("missing_start_liquidity", joined["missing_start_liquidity"]),
        ("low_start_liquidity", joined["low_start_liquidity"]), ("missing_points", joined["missing_points"]),
        ("missing_end", joined["missing_end"]), ("zero_end", joined["zero_end"]),
        ("missing_end_liquidity", joined["missing_end_liquidity"]), ("low_end_liquidity", joined["low_end_liquidity"]),
    )
    for name, flag in stages:
        joined[f"reason_{name}"] = flag
    joined["first_failure_reason"] = "eligible"
    for name, flag in reversed(stages):
        joined.loc[flag, "first_failure_reason"] = name
    joined["first_failure"] = joined["first_failure_reason"]
    start_eligible = joined["first_failure_reason"].isin(("eligible", "missing_points", "missing_end", "zero_end", "missing_end_liquidity", "low_end_liquidity"))
    joined["start_eligible"] = start_eligible
    joined["primary_eligible"] = start_eligible & ~joined["missing_points"] & ~joined["missing_end"] & ~joined["zero_end"] & ~joined["missing_end_liquidity"] & ~joined["low_end_liquidity"]
    primary = joined.loc[joined["primary_eligible"]].copy()
    zero_end = joined.loc[start_eligible & ~joined["missing_points"] & joined["zero_end"]].copy()
    thin = joined.loc[start_eligible & ~joined["missing_points"] & ~joined["missing_end"] & (joined["label_end_value"] > 0) & ~joined["missing_end_liquidity"] & joined["low_end_liquidity"]].copy()
    for frame in (joined, primary, zero_end, thin):
        frame.sort_values(["season", "player_id"], kind="mergesort", inplace=True)
        frame.reset_index(drop=True, inplace=True)

    label_keys = set(zip(labels["season"].astype(int), labels["player_id"].astype(str)))
    archive_keys = set(zip(archive["season"].astype(int), archive["player_id"].astype(str)))
    archive_only = sorted(archive_keys - label_keys)
    label_only = sorted(label_keys - archive_keys)
    available_labels = joined.loc[joined["join_status"] != "archive_only"]
    by_population = {
        "archive": _population(archive), "available_labels": _population(available_labels),
        "start_eligible": _population(joined.loc[start_eligible]), "primary_scored": _population(primary),
    }
    reasons = {name: int(flag.sum()) for name, flag in stages}
    coverage: dict[str, object] = {
        "upstream_provenance": bundle.provenance,
        "ledger": {"first_failure": {name: int((joined["first_failure_reason"] == name).sum()) for name, _ in stages}, "start_eligible": int(start_eligible.sum()), "primary_scored": len(primary)},
        "reasons": reasons,
        "independent_flags": {name: int(flag.sum()) for name, flag in stages} | {"unknown_experience": int(joined["unknown_experience"].sum())},
        "by_population": by_population,
        "coverage_by_season_position": by_population,
        "archive_population": by_population["archive"], "available_label_population": by_population["available_labels"],
        "available_start_eligible": by_population["start_eligible"], "primary_scored_population": by_population["primary_scored"],
        "key_hashes": {
            "archive_only": {"count": len(archive_only), "sha256": key_sha256(archive_only)},
            "label_only": {"count": len(label_only), "sha256": key_sha256(label_only)},
            "common": {"count": len(archive_keys & label_keys), "sha256": key_sha256(sorted(archive_keys & label_keys))},
        },
        "archive_only_key_count": len(archive_only), "label_only_key_count": len(label_only),
        "archive_only_key_sha256": key_sha256(archive_only), "label_only_key_sha256": key_sha256(label_only),
        "start_evidence": _distribution(joined.loc[start_eligible], "label_start_evidence"),
        "end_evidence": _distribution(primary, "label_end_evidence"),
        "start_trades": _distribution(joined, "label_start_trades_window"),
        "end_trades": _distribution(joined, "label_end_trades_window"),
        "negative_point_count": int((pd.to_numeric(joined["points"], errors="coerce") < 0).sum()),
        "observed_zero_end": {"count": len(zero_end), "absolute_inputs": ["start_value", "end_value", "points"]},
        "thin_endpoint": {"count": len(thin), "trades_range": [0, 4]},
        "scored_rate_within_available_labels": (len(primary) / len(joined)) if len(joined) else None,
        "start_eligible_rate_within_available_labels": (int(start_eligible.sum()) / len(joined)) if len(joined) else None,
        "full_preseason_population_count": {"value": None, "reason": "not_recoverable_from_historical_export"},
        "full_preseason_coverage": {"value": None, "reason": "not_recoverable_from_historical_export"},
    }
    return EvaluationData(joined=joined, primary_scored=primary, observed_zero_end=zero_end, thin_endpoint=thin, coverage=coverage)
