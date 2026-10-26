"""Frozen, side-effect-free protocol constants for movement evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import pandas as pd


class ContractRefusal(ValueError):
    """Raised when an evaluator input or protocol value violates the contract."""


KtcMode = Literal["off", "archive"]
BASE_COMMIT = "c4270804379c26bcb340f32a07e507153b4c7342"
TARGET_ID = "backchannel-mfl-v1-reference-same-season-log-movement-2020-2025-v1"
POSITIONS = ("QB", "RB", "TE", "WR")
SCALE_MAX = 10000.0

ARCHIVE_SHA256 = "b67e7b03e600ae9ad83b5b179b416f273b7e169c8d47eff7345ef257cb034a86"
ARCHIVE_MEMBER_SHA256 = "6dfa1e3eaed486d5a37880217c74fb3b139f19504aedb58b39a38013466a5388"
MODEL_SOURCE_SHA256 = "01674fa50df69a8e27f221f46eafa2e55b459df9e74c2c2357d6e03e733cade1"

HGB_PARAMS = MappingProxyType(
    {
        "loss": "absolute_error",
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "random_state": 0,
    }
)
FEATURES = ("log_start", "age", "points", "position_code")
MONOTONIC = (0, 0, 1, 0)
POINTS_GRID_STEP = 1.0
FLAT_BAND = math.log(1.10)
LARGE_MOVE_BAND = math.log(1.50)
CALIBRATION_EDGES = (
    -math.log(2),
    -math.log(1.25),
    -math.log(1.10),
    0.0,
    math.log(1.10),
    math.log(1.25),
    math.log(2),
)

INPUT_IDENTITIES = MappingProxyType(
    {
        "labels": MappingProxyType(
            {
                "path": "backend/predictor_v5/data/backchannel_labels_2019_2025_mfl.csv",
                "sha256": "3e57be0b6e6ccb8f626a8b4ee1e726eaad75f635174d147514c3aa8ccf3c8e89",
            }
        ),
        "provenance": MappingProxyType(
            {
                "path": "backend/predictor_v5/data/backchannel_labels_2019_2025_mfl.provenance.json",
                "sha256": "5258f1247b982b371e96125aa122d58f8afae3738b3e24d4fa39cacc05ee709a",
            }
        ),
        "archive": MappingProxyType(
            {
                "path": "backend/data/training-data.zip",
                "sha256": ARCHIVE_SHA256,
                "member": "training-data.json",
                "member_sha256": ARCHIVE_MEMBER_SHA256,
            }
        ),
        "model": MappingProxyType({"path": "backend/predictor_v5/model.py", "sha256": MODEL_SOURCE_SHA256}),
    }
)

_DATES = (
    (2020, ("2020-09-10", "2021-01-03")),
    (2021, ("2021-09-09", "2022-01-09")),
    (2022, ("2022-09-08", "2023-01-08")),
    (2023, ("2023-09-07", "2024-01-07")),
    (2024, ("2024-09-05", "2025-01-05")),
    (2025, ("2025-09-04", "2026-01-04")),
)
_GENERATIONS = (
    (2020, (("start", (9381, 3)), ("end", (9496, 3)))),
    (2021, (("start", (9719, 3)), ("end", (9841, 3)))),
    (2022, (("start", (10026, 3)), ("end", (10148, 3)))),
    (2023, (("start", (10350, 3)), ("end", (10472, 3)))),
    (2024, (("start", (10665, 3)), ("end", (10787, 3)))),
    (2025, (("start", (10983, 3)), ("end", (11105, 3)))),
)


@dataclass(frozen=True)
class FoldSpec:
    test_season: int
    train_seasons: tuple[int, ...]
    short_history: bool


FOLDS = (
    FoldSpec(2021, (2020,), True),
    FoldSpec(2022, (2020, 2021), False),
    FoldSpec(2023, (2020, 2021, 2022), False),
    FoldSpec(2024, (2020, 2021, 2022, 2023), False),
    FoldSpec(2025, (2020, 2021, 2022, 2023, 2024), False),
)


def validate_ktc_mode(value: str) -> KtcMode:
    if isinstance(value, bool) or value not in ("off", "archive"):
        raise ContractRefusal("ktc diagnostic must be exactly 'off' or 'archive'")
    return value


def canonical_bytes(value: object) -> bytes:
    """Encode a JSON value using the protocol's deterministic JSON primitive."""
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractRefusal("value is not canonical JSON") from exc
    return raw + b"\n"


def sorted_keys(frame: pd.DataFrame) -> tuple[tuple[int, str], ...]:
    if not isinstance(frame, pd.DataFrame) or not {"season", "player_id"}.issubset(frame.columns):
        raise ContractRefusal("key frame requires season and player_id columns")
    keys: list[tuple[int, str]] = []
    for season, player_id in frame[["season", "player_id"]].itertuples(index=False, name=None):
        if isinstance(season, bool) or not isinstance(season, (int, float)) or not float(season).is_integer():
            raise ContractRefusal("season must be an integer")
        season = int(season)
        if season == 2026 or season < 2020 or season > 2025:
            raise ContractRefusal("season is outside the historical contract")
        if not isinstance(player_id, str) or not player_id:
            raise ContractRefusal("player_id must be a nonempty string")
        keys.append((season, player_id))
    result = tuple(sorted(keys))
    if len(result) != len(set(result)):
        raise ContractRefusal("duplicate season/player_id key")
    return result


def key_sha256(keys: Iterable[tuple[int, str]]) -> str:
    material: list[list[object]] = []
    for key in keys:
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            raise ContractRefusal("keys must be season/player_id pairs")
        season, player_id = key
        if isinstance(season, bool) or not isinstance(season, int) or season == 2026 or not 2020 <= season <= 2025:
            raise ContractRefusal("season is outside the historical contract")
        if not isinstance(player_id, str) or not player_id:
            raise ContractRefusal("player_id must be a nonempty string")
        material.append([season, player_id])
    ordered = sorted(material, key=lambda pair: (pair[0], pair[1]))
    if len(ordered) != len({(pair[0], pair[1]) for pair in ordered}):
        raise ContractRefusal("duplicate season/player_id key")
    return hashlib.sha256(canonical_bytes(ordered)).hexdigest()


def contract_payload(mode: KtcMode) -> dict[str, object]:
    mode = validate_ktc_mode(mode)
    model = {
        "features": FEATURES,
        "hgb_params": dict(HGB_PARAMS),
        "monotonic": MONOTONIC,
        "scale_max": SCALE_MAX,
        "points_grid_step": POINTS_GRID_STEP,
        "early_stopping": "auto",
        "early_stopping_note": "inactive for these small folds",
    }
    return {
        "base_commit": BASE_COMMIT,
        "target": {
            "id": TARGET_ID,
            "formula": "log(end_value / start_value)",
            "scale": SCALE_MAX,
            "columns": {"start": "start_value", "end": "end_value"},
            "positions": POSITIONS,
            "dates": {season: list(dates) for season, dates in _DATES},
            "generations": {
                season: {kind: {"id": values[0], "revision": values[1]} for kind, values in records}
                for season, records in _GENERATIONS
            },
        },
        "inputs": {key: dict(value) for key, value in INPUT_IDENTITIES.items()},
        "models": {"production": model, "no_production": {**model, "points": 0.0}},
        "folds": [
            {"test_season": fold.test_season, "train_seasons": list(fold.train_seasons), "short_history": fold.short_history}
            for fold in FOLDS
        ],
        "metrics": {
            "flat_band": FLAT_BAND,
            "large_move_band": LARGE_MOVE_BAND,
            "calibration_edges": CALIBRATION_EDGES,
        },
        "bootstrap": {"seed": 20261025, "resamples": 2000, "percentiles": [2.5, 97.5]},
        "ktc_mode": mode,
    }
