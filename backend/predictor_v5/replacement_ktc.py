"""Offline verification of registered KTC observations."""

import hashlib
import io
import json
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
DESCRIPTOR = {
    "format": "dynasty",
    "join_keys": ["format", "ktc_id"],
    "nonnull_source_id": True,
    "provider": "local-thelab",
    "row_evidence": "registered-raw-end-export",
    "schema_version": 1,
    "source_id": "sleeper_id",
    "source_table": "ktc_values",
    "target_id": "player_id",
}


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
    except OSError as exc:
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
    if data["mapping_data"] != canonical_bytes(DESCRIPTOR):
        raise ValueError("invalid mapping descriptor")
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
    try:
        raw = pd.read_csv(io.BytesIO(data["raw_end"]), dtype=str, keep_default_na=False)
        target = pd.read_csv(
            io.BytesIO(data["target_csv"]),
            dtype=str,
            keep_default_na=False,
        )
        with zipfile.ZipFile(io.BytesIO(data["target_stats"])) as z:
            if z.namelist() != ["training-data.json"]:
                raise ValueError("invalid stats archive members")
            players = json.loads(z.read("training-data.json"))["players"]
        archive_rows = []
        for p in players:
            for s in p.get("seasons", []):
                if type(s["year"]) is not int or s["year"] != 2026:
                    raise ValueError("invalid stats season")
                archive_rows.append(
                    {
                        "player_id": str(p["player_id"]),
                        "season": s["year"],
                        "position": p["position"],
                        "age": s["age"],
                        "games": s.get("games_played") or 0,
                        "points": s.get("fantasy_points") or 0,
                    }
                )
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
    for frame in (stats, pd.DataFrame(archive_rows)):
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
