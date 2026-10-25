"""Offline verification of registered KTC observations."""

import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import pandas as pd

from predictor_v5.replacement_contract import canonical_bytes, validate_identity
from predictor_v5.replacement_starts import RELATIVE_PATH, SHA256, check_target_starts

RAW_COLUMNS = ["format", "ktc_id", "sleeper_id", "date", "sf_value"]
TARGET_COLUMNS = [
    "player_id",
    "season",
    "start_value",
    "end_value",
    "position",
    "age",
    "games",
    "points",
]
MAPPING_COLUMNS = ["format", "ktc_id", "sleeper_id"]


def _csv_frame(data: bytes, columns: list[str]) -> pd.DataFrame:
    """Check physical records before pandas can infer an implicit index."""
    try:
        records = csv.reader(io.StringIO(data.decode("utf-8"), newline=""), strict=True)
        if next(records, None) != columns:
            raise ValueError("invalid CSV columns")
        if any(len(record) != len(columns) for record in records):
            raise ValueError("invalid CSV record width")
        return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    except (UnicodeError, csv.Error, pd.errors.ParserError) as exc:
        raise ValueError("invalid CSV") from exc


def mapping_snapshot(data: bytes) -> dict[tuple[str, str], str]:
    """Decode the canonical, one-to-one pre-outcome dynasty mapping rows.

    Canonical CSV uses UTF-8, minimal quoting, LF records, and numeric ktc_id
    order. The snapshot covers every non-null dynasty mapping at review time;
    future raw exports must match it, including rows excluded from scoring.
    """
    frame = _csv_frame(data, MAPPING_COLUMNS)
    if (
        frame.empty
        or not frame["format"].eq("dynasty").all()
        or not frame["ktc_id"].str.fullmatch(r"[1-9][0-9]*").all()
        or frame["sleeper_id"].map(lambda value: not value or value != value.strip()).any()
        or frame.duplicated(["format", "ktc_id"]).any()
        or frame["sleeper_id"].duplicated().any()
    ):
        raise ValueError("invalid mapping snapshot rows")
    rows = sorted(
        frame.itertuples(index=False, name=None),
        key=lambda row: (row[0], int(row[1]), row[2]),
    )
    canonical = io.StringIO(newline="")
    writer = csv.writer(canonical, lineterminator="\n")
    writer.writerow(MAPPING_COLUMNS)
    writer.writerows(rows)
    if canonical.getvalue().encode("utf-8") != data:
        raise ValueError("noncanonical mapping snapshot")
    return {(format_, ktc_id): sleeper_id for format_, ktc_id, sleeper_id in rows}


def select_end_observations(raw: pd.DataFrame) -> pd.DataFrame:
    if list(raw.columns) != RAW_COLUMNS:
        raise ValueError("invalid raw columns")
    df = raw.copy()
    for col in ("format", "ktc_id", "sleeper_id", "date"):
        if (
            df[col].isna().any()
            or df[col].map(lambda v: not isinstance(v, str) or not v.strip()).any()
        ):
            raise ValueError("invalid raw identity")
    if not df["format"].eq("dynasty").all():
        raise ValueError("invalid format")
    dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    if (
        dates.isna().any()
        or not dates.between("2026-12-28", "2027-01-10").all()
        or not dates.dt.strftime("%Y-%m-%d").eq(df["date"]).all()
    ):
        raise ValueError("invalid observation date")
    if df["sf_value"].map(lambda x: isinstance(x, bool)).any():
        raise ValueError("invalid value")
    if (df.groupby(["format", "ktc_id"])["sleeper_id"].nunique() > 1).any():
        raise ValueError("source identity remapped")
    values = pd.to_numeric(df["sf_value"], errors="coerce")
    if not values.between(0, 9999, inclusive="neither").all():
        raise ValueError("invalid value")
    df["sf_value"] = values.astype(float)
    for keys in (["format", "ktc_id", "date"], ["sleeper_id", "date"]):
        grouped = df.groupby(keys, dropna=False)
        if (grouped["sf_value"].nunique() > 1).any() or (
            grouped["sleeper_id"].nunique() > 1
        ).any():
            raise ValueError("conflicting observations")
    df = df.sort_values(["sleeper_id", "date"]).drop_duplicates(
        "sleeper_id", keep="last"
    )
    return df.rename(columns={"sleeper_id": "player_id", "sf_value": "end_value"})[
        ["player_id", "date", "end_value"]
    ].reset_index(drop=True)


def _read(root, identity):
    validate_identity(identity)
    root = Path(root).resolve()
    path = (root / identity["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("path escapes root")
    try:
        data = path.read_bytes()
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("cannot read registered file") from exc
    if hashlib.sha256(data).hexdigest() != identity["sha256"]:
        raise ValueError("registered digest differs")
    return data


def _canonical(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        obj = json.loads(data, object_pairs_hook=pairs)
        if canonical_bytes(obj) != data:
            raise ValueError("noncanonical JSON")
        return obj
    except (UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("invalid JSON") from exc


def _archive_rows(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if z.namelist() != ["training-data.json"]:
            raise ValueError("invalid stats archive members")
        envelope = json.loads(z.read("training-data.json"))
    if not isinstance(envelope, dict) or not isinstance(envelope.get("players"), list):
        raise ValueError("invalid players envelope")  # noqa: TRY004
    rows, seen = [], set()
    for player in envelope["players"]:
        if not isinstance(player, dict):
            raise ValueError("invalid player")  # noqa: TRY004
        player_id, position, seasons = (
            player.get("player_id"),
            player.get("position"),
            player.get("seasons"),
        )
        if (
            not isinstance(player_id, str)
            or not player_id.strip()
            or position not in ("QB", "RB", "TE", "WR")
            or not isinstance(seasons, list)
        ):
            raise ValueError("invalid player facts")
        for season in seasons:
            if (
                not isinstance(season, dict)
                or type(season.get("year")) is not int
                or season["year"] != 2026
            ):
                raise ValueError("invalid stats season")
            key = (player_id, 2026)
            if key in seen:
                raise ValueError("duplicate archive player season")
            seen.add(key)
            age = season.get("age")
            games = season.get("games_played")
            points = season.get("fantasy_points")
            games = 0 if games is None else games
            points = 0 if points is None else points
            if any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (age, games, points)
            ):
                raise ValueError("invalid numeric stats")
            if games < 0 or int(games) != games:
                raise ValueError("invalid games")
            rows.append(
                {
                    "player_id": player_id,
                    "season": 2026,
                    "position": position,
                    "age": age,
                    "games": int(games),
                    "points": points,
                }
            )
    return rows


def _validate_coverage(value):
    if not isinstance(value, dict) or set(value) != {
        "raw_players",
        "target_players",
        "missing_start_keys",
        "excluded_raw_players",
    }:
        raise ValueError("invalid coverage")
    if any(
        type(value[k]) is not int or value[k] < 0
        for k in ("raw_players", "target_players")
    ):
        raise ValueError("invalid coverage count")
    missing, excluded = value["missing_start_keys"], value["excluded_raw_players"]
    if not isinstance(missing, list) or not isinstance(excluded, list):
        raise ValueError("invalid coverage lists")  # noqa: TRY004
    if any(
        not isinstance(k, list)
        or len(k) != 2
        or type(k[0]) is not int
        or k[0] != 2026
        or not isinstance(k[1], str)
        or not k[1].strip()
        for k in missing
    ):
        raise ValueError("invalid missing key")
    if any(not isinstance(k, str) or not k.strip() for k in excluded):
        raise ValueError("invalid excluded player")


def load_ktc_target(
    registration: dict, starts: pd.DataFrame, stats: pd.DataFrame, root: Path
) -> pd.DataFrame:
    ids = registration["ktc"]
    keys = {
        "raw_end",
        "target_csv",
        "target_stats",
        "target_sidecar",
        "sql",
        "extractor",
        "mapping_code",
        "mapping_data",
    }
    if set(ids) != keys:
        raise ValueError("invalid KTC registration")
    data = {key: _read(root, ids[key]) for key in keys}
    mapping = mapping_snapshot(data["mapping_data"])
    side = _canonical(data["target_sidecar"])
    if not isinstance(side, dict) or set(side) != {
        "schema_version",
        "provider",
        "reviewer",
        "starts",
        "identities",
        "observation_dates",
        "coverage",
    }:
        raise ValueError("invalid sidecar")
    if (
        type(side["schema_version"]) is not int
        or side["schema_version"] != 1
        or side["provider"] != "local-thelab"
        or not isinstance(side["reviewer"], str)
        or not side["reviewer"].strip()
    ):
        raise ValueError("invalid sidecar provenance")
    if side["starts"] != {"path": RELATIVE_PATH, "sha256": SHA256} or side[
        "identities"
    ] != {k: ids[k] for k in keys - {"target_sidecar"}}:
        raise ValueError("sidecar identity mismatch")
    _validate_coverage(side["coverage"])
    raw = _csv_frame(data["raw_end"], RAW_COLUMNS)
    target = _csv_frame(data["target_csv"], TARGET_COLUMNS)
    if any(
        mapping.get((row.format, row.ktc_id)) != row.sleeper_id
        for row in raw.itertuples(index=False)
    ):
        raise ValueError("raw mapping absent from or conflicts with frozen snapshot")
    try:
        archive_rows = _archive_rows(data["target_stats"])
    except (ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid registered data") from exc
    if list(target.columns) != TARGET_COLUMNS:
        raise ValueError("invalid target columns")
    if (
        not target["season"].eq("2026").all()
        or not target["games"].str.fullmatch(r"[0-9]+").all()
    ):
        raise ValueError("invalid integer fields")
    if (
        target["player_id"].map(lambda x: not isinstance(x, str) or not x.strip()).any()
        or target.duplicated(["player_id", "season"]).any()
    ):
        raise ValueError("invalid target identity")
    for col in ("season", "start_value", "end_value", "age", "games", "points"):
        target[col] = pd.to_numeric(target[col], errors="coerce")
    if (
        not target["season"].eq(2026).all()
        or not target["position"].isin(["QB", "RB", "TE", "WR"]).all()
    ):
        raise ValueError("invalid target season or position")
    import numpy as np

    if (
        not np.isfinite(
            target[["start_value", "end_value", "age", "games", "points"]].to_numpy()
        ).all()
        or not target["start_value"].between(0, 9999, inclusive="neither").all()
        or not target["end_value"].between(0, 9999, inclusive="neither").all()
        or not (
            target["games"].ge(0) & target["games"].eq(target["games"].astype(int))
        ).all()
    ):
        raise ValueError("invalid target values")
    target["season"] = target["season"].astype(int)
    target["games"] = target["games"].astype(int)
    for col in ("start_value", "end_value", "age", "points"):
        target[col] = target[col].astype(float)
    selected = select_end_observations(raw)
    merged = target.merge(
        selected,
        on="player_id",
        how="left",
        suffixes=("", "_raw"),
        validate="one_to_one",
    )
    if (
        merged["date"].isna().any()
        or not merged["end_value"].eq(merged["end_value_raw"]).all()
    ):
        raise ValueError("raw observations disagree")
    cols = ["player_id", "season", "position", "age", "games", "points"]
    for frame in (stats, pd.DataFrame(archive_rows, columns=cols)):
        if frame.duplicated(["player_id", "season"]).any():
            raise ValueError("duplicate stats")
        joined = target[cols].merge(
            frame[cols],
            on=["player_id", "season"],
            how="left",
            suffixes=("", "_stats"),
            validate="one_to_one",
        )
        if any(not joined[c].eq(joined[c + "_stats"]).all() for c in cols[2:]):
            raise ValueError("stats mismatch")
    coverage = check_target_starts(starts, target, "ktc")
    dates = selected[["player_id", "date"]].sort_values("player_id").to_dict("records")
    expected_coverage = {
        "raw_players": len(selected),
        "target_players": len(target),
        "missing_start_keys": [list(k) for k in coverage["missing_keys"]],
        "excluded_raw_players": sorted(
            set(selected["player_id"]) - set(target["player_id"])
        ),
    }
    if side["observation_dates"] != dates or side["coverage"] != expected_coverage:
        raise ValueError("sidecar coverage or dates mismatch")
    target.attrs["coverage"] = expected_coverage
    return target.sort_values(["season", "player_id"]).reset_index(drop=True)
