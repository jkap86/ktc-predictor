"""Validate the independently registered MFL target and its statistics."""

import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from predictor_v5.replacement_ktc import _canonical, _read
from predictor_v5.replacement_starts import check_target_starts

STAT_COLUMNS = ["player_id", "season", "position", "age", "games", "points"]
TARGET_COLUMNS = [
    "player_id",
    "season",
    "start_value",
    "end_value",
    "position",
    "age",
    "games",
    "points",
    "start_evidence",
    "end_evidence",
    "start_trades_window",
    "end_trades_window",
    "end_value_next_board",
    "end_evidence_next_board",
]


def _pairs(items):
    obj = {}
    for key, value in items:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def _stats_bytes(data):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if z.namelist() != ["training-data.json"]:
                raise ValueError("invalid archive members")
            envelope = json.loads(
                z.read("training-data.json"),
                object_pairs_hook=_pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(
                    ValueError("invalid constant")
                ),
            )
    except (zipfile.BadZipFile, UnicodeError, json.JSONDecodeError, KeyError) as exc:
        raise ValueError("invalid stats archive") from exc
    if not isinstance(envelope, dict) or not isinstance(envelope.get("players"), list):
        raise ValueError("invalid players envelope")  # noqa: TRY004
    rows, seen = [], set()
    for player in envelope["players"]:
        if not isinstance(player, dict):
            raise ValueError("invalid player")  # noqa: TRY004
        pid, position, seasons = (
            player.get("player_id"),
            player.get("position"),
            player.get("seasons"),
        )
        if (
            not isinstance(pid, str)
            or not pid.strip()
            or pid in seen
            or position not in ("QB", "RB", "TE", "WR")
            or not isinstance(seasons, list)
        ):
            raise ValueError("invalid player facts")
        seen.add(pid)
        if len(seasons) > 1:
            raise ValueError("duplicate season")
        for season in seasons:
            if (
                not isinstance(season, dict)
                or type(season.get("year")) is not int
                or season["year"] != 2026
            ):
                raise ValueError("invalid season")
            age = season.get("age")
            games = season.get("games_played")
            points = season.get("fantasy_points")
            games = 0 if games is None else games
            points = 0 if points is None else points
            if (
                any(
                    type(v) not in (int, float) or not math.isfinite(v)
                    for v in (age, games, points)
                )
                or games < 0
                or int(games) != games
            ):
                raise ValueError("invalid stats")
            rows.append((pid, 2026, position, float(age), int(games), float(points)))
    frame = pd.DataFrame(rows, columns=STAT_COLUMNS).astype(
        {
            "player_id": "object",
            "season": "int64",
            "position": "object",
            "age": "float64",
            "games": "int64",
            "points": "float64",
        }
    )
    return frame.sort_values(["season", "player_id"]).reset_index(drop=True)


def load_stats(path: Path) -> pd.DataFrame:
    try:
        return _stats_bytes(Path(path).read_bytes())
    except OSError as exc:
        raise ValueError("cannot read stats") from exc


def load_mfl_target(
    registration: dict, starts: pd.DataFrame, stats: pd.DataFrame, root: Path
) -> pd.DataFrame:
    ids = registration.get("mfl")
    keys = {"target_csv", "target_stats", "target_sidecar", "sql", "build"}
    if not isinstance(ids, dict) or set(ids) != keys:
        raise ValueError("invalid MFL registration")
    data = {key: _read(root, ids[key]) for key in keys}
    side = _canonical(data["target_sidecar"])
    expected = {
        "schema_version": 1,
        "season": 2026,
        "model_version": "baseline-hl28-v1",
        "verified": True,
        "source": "export",
        "problems": [],
        "start_cutoff": "2026-09-09",
        "end_cutoff": "2027-01-10",
        "timezone": "America/New_York",
        **{key: ids[key] for key in keys - {"target_sidecar"}},
    }
    if (
        not isinstance(side, dict)
        or side != expected
        or type(side["schema_version"]) is not int
        or type(side["season"]) is not int
        or side["verified"] is not True
    ):
        raise ValueError("invalid MFL sidecar")
    archive = _stats_bytes(data["target_stats"])
    try:
        target = pd.read_csv(
            io.BytesIO(data["target_csv"]), dtype=str, keep_default_na=False
        )
    except (ValueError, pd.errors.ParserError) as exc:
        raise ValueError("invalid target CSV") from exc
    if (
        list(target.columns) != TARGET_COLUMNS
        or target["player_id"].map(lambda x: not x.strip()).any()
        or target.duplicated(["player_id", "season"]).any()
        or not target["season"].eq("2026").all()
        or not target["position"].isin(["QB", "RB", "TE", "WR"]).all()
    ):
        raise ValueError("invalid target schema or identity")
    for col in ("games", "start_trades_window", "end_trades_window"):
        if not target[col].str.fullmatch(r"[0-9]+").all():
            raise ValueError("invalid integer")
    optional = ["end_value_next_board", "end_evidence_next_board"]
    numeric = [
        "season",
        "start_value",
        "end_value",
        "age",
        "games",
        "points",
        "start_evidence",
        "end_evidence",
        "start_trades_window",
        "end_trades_window",
        *optional,
    ]
    for col in numeric:
        original = target[col]
        values = pd.to_numeric(original, errors="coerce")
        valid = np.isfinite(values) | ((original == "") if col in optional else False)
        if not valid.all():
            raise ValueError("invalid numeric value")
        target[col] = values
    if (
        not (target[["start_value", "end_value"]] > 0).all().all()
        or not (target[["start_evidence", "end_evidence"]] >= 0).all().all()
        or not (target[["start_trades_window", "end_trades_window"]] >= 5).all().all()
        or any((target[c].dropna() < 0).any() for c in optional)
    ):
        raise ValueError("invalid values or insufficient trades")
    for col in ("season", "games", "start_trades_window", "end_trades_window"):
        target[col] = target[col].astype(int)
    for frame in (archive, stats):
        if frame.duplicated(["player_id", "season"]).any():
            raise ValueError("duplicate stats")
        joined = target[STAT_COLUMNS].merge(
            frame[STAT_COLUMNS],
            on=["player_id", "season"],
            how="left",
            suffixes=("", "_stats"),
            validate="one_to_one",
        )
        if any(not joined[c].eq(joined[c + "_stats"]).all() for c in STAT_COLUMNS[2:]):
            raise ValueError("stats mismatch")
    coverage = check_target_starts(starts, target, "mfl")
    if not starts["bc_model_version"].eq("baseline-hl28-v1").all():
        raise ValueError("invalid frozen model version")
    target = target.sort_values(["season", "player_id"]).reset_index(drop=True)
    target.attrs["coverage"] = coverage
    return target
