"""Strict, sealed artifact validation for the 2026 model confirmation."""

import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import tempfile
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from predictor_v5 import label_export, model
from predictor_v5.data import build_dataset
from predictor_v5.evaluate import summarize_exact
from predictor_v5.labels import CsvLabels
from predictor_v5.model import WhatIfModel

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path("backend/predictor_v5/data/model_2026_confirmation_manifest.json")
REGISTRY_PATH = Path("backend/predictor_v5/data/model_2026_holdout_registry.json")

_SOURCE_PATHS = (
    "backend/predictor_v5/model.py",
    "backend/predictor_v5/sql/backchannel_labels.sql",
    "backend/predictor_v5/label_export.py",
    "backend/predictor_v5/data.py",
    "backend/predictor_v5/labels.py",
    "backend/predictor_v5/evaluate.py",
    "backend/predictor_v5/confirmation.py",
    "backend/predictor_v5/__main__.py",
)
_LABEL_COLUMNS = (
    "player_id", "season", "start_value", "end_value", "position", "age", "games", "points",
    "start_evidence", "end_evidence", "start_trades_window", "end_trades_window",
    "end_value_next_board", "end_evidence_next_board",
)
_RAW_MARKET_COLUMNS = (
    "player_id", "season", "start_value", "end_value", "start_evidence", "end_evidence",
    "start_trades_window", "end_trades_window", "end_value_next_board", "end_evidence_next_board",
)


class PreflightError(ValueError):
    """A deterministic refusal which a future CLI must classify as exit 2."""


def confirmation_exit_code(error: BaseException) -> int:
    """Keep CLI classification testable without adding a CLI in this task."""
    return 2 if isinstance(error, PreflightError) else 1


@dataclass
class ConfirmationConfig:
    """Only filesystem and loading dependencies may be injected for confirmation."""

    repo_root: Path = REPOSITORY_ROOT
    manifest_path: Path = MANIFEST_PATH
    registry_path: Path = REGISTRY_PATH
    input_registration_path: Path = Path("backend/predictor_v5/data/model_2026_confirmation_inputs.json")
    target_loader: Callable[[Path, Path], pd.DataFrame] | None = None
    output_dir: Path = Path("backend/predictor_v5/artifacts/confirmations")

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root)
        self.manifest_path = Path(self.manifest_path)
        self.registry_path = Path(self.registry_path)
        self.input_registration_path = Path(self.input_registration_path)
        self.output_dir = Path(self.output_dir)
        if self.target_loader is None:
            self.target_loader = _default_target_loader


@dataclass
class PreflightContext:
    manifest: dict
    manifest_sha256: str
    input_registration: dict
    input_registration_sha256: str
    holdout_registry: dict
    holdout_registry_sha256: str
    baseline_csv: Path
    candidate_csv: Path
    training_archive: Path
    target_csv: Path
    target_sidecar: Path
    target_stats: Path
    output_path: Path
    train_seasons: tuple[int, ...]
    positions: tuple[str, ...]
    label_filter: dict
    target_recipe: dict
    target_loader: Callable[[Path, Path], pd.DataFrame]
    target_frame: pd.DataFrame | None = field(default=None, init=False)
    training_frames: dict | None = field(default=None, init=False)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the protocol's sole JSON representation."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_repo_relative_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("repository-relative path must be a nonempty string")
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if "\\" in value or path.is_absolute() or any(part in ("", ".", "..") for part in raw_parts):
        raise ValueError("path must be repository-relative and contain no traversal")
    return Path(*path.parts)


def _no_duplicate_keys(pairs: list) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def load_canonical_json(path: Path, schema: Callable[[dict[str, Any]], None]) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON artifact: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("artifact root must be an object")  # noqa: TRY004 -- protocol failures are ValueError
    if raw != canonical_json_bytes(value):
        raise ValueError("artifact bytes are not canonical JSON")
    schema(value)
    return value, sha256_bytes(raw)


def _fields(value: Any, required: set, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")  # noqa: TRY004 -- protocol failures are ValueError
    missing = required - set(value)
    unknown = set(value) - required
    if missing:
        raise ValueError(f"{context} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {sorted(unknown)}")
    return value


def _nonempty(value: Any, context: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string")


def _timestamp(value: Any, context: str) -> datetime:
    _nonempty(value, context)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{context} must be a canonical timestamp") from error
    if parsed.tzinfo is None or value != parsed.isoformat(timespec="seconds"):
        raise ValueError(f"{context} must be a canonical timestamp with seconds and UTC offset")
    return parsed


def _review_metadata(value: dict, context: str) -> datetime:
    for name in ("reviewer", "evidence_commit", "evidence_reference"):
        _nonempty(value[name], f"{context} {name}")
    reviewed = _timestamp(value["reviewed_at"], context + " reviewed_at")
    if value["transition_source"] != "sealed":
        raise ValueError(context + " transition_source must be sealed")
    return reviewed


def _sha(value: Any, context: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{context} must be a lowercase SHA-256")


def _identity(value: Any, context: str) -> None:
    entry = _fields(value, {"path", "sha256"}, context)
    validate_repo_relative_path(entry["path"])
    _sha(entry["sha256"], context + ".sha256")


def _drift(context: str, expected: Any, actual: Any) -> ValueError:
    return ValueError(f"{context} drift: expected {expected!r}; actual {actual!r}")


def _first_difference(expected: Any, actual: Any, context: str) -> ValueError:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                return _drift(context + "." + key, expected.get(key), actual.get(key))
            if expected[key] != actual[key]:
                return _first_difference(expected[key], actual[key], context + "." + key)
    return _drift(context, expected, actual)


def _validate_live_label_identity(name: str, entry: dict[str, Any], repo_root: Path = REPOSITORY_ROOT) -> None:
    path = validate_repo_relative_path(entry["path"])
    absolute = repo_root / path
    live_sha = file_sha256(absolute)
    if entry["sha256"] != live_sha:
        raise _drift(str(path) + " sha256", live_sha, entry["sha256"])
    helper = getattr(label_export, "label_identity", None)
    if not callable(helper):
        raise _drift("label_export.label_identity", "callable(path, manifest_path=...)", type(helper).__name__)
    parameters = list(inspect.signature(helper).parameters.values())
    if len(parameters) != 2 or parameters[0].name != "path" or parameters[1].name != "manifest_path" or parameters[1].default is inspect.Parameter.empty:
        raise _drift("label_export.label_identity signature", "(path, manifest_path=<default>)", str(inspect.signature(helper)))
    identity = helper(absolute)
    expected_keys = {"sha256", "model_version", "verified", "source", "problems"}
    if not isinstance(identity, dict) or set(identity) != expected_keys:
        raise _drift(name + " label_identity schema", sorted(expected_keys), sorted(identity) if isinstance(identity, dict) else type(identity).__name__)
    expected_provenance = {"model_version": "baseline-hl28-v1", "verified": True}
    actual_provenance = {"model_version": identity["model_version"], "verified": identity["verified"]}
    if actual_provenance != expected_provenance:
        raise _first_difference(expected_provenance, actual_provenance, name + " provenance")
    if identity["sha256"] != live_sha:
        raise _drift(str(path) + " label_identity.sha256", live_sha, identity["sha256"])
    if not isinstance(identity["source"], str) or not identity["source"] or identity["problems"] != []:
        raise _drift(name + " label_identity verification", "nonempty source and [] problems", {"source": identity["source"], "problems": identity["problems"]})


def validate_manifest(manifest: dict[str, Any], repo_root: Path = REPOSITORY_ROOT) -> None:
    required = {
        "schema_version", "protocol", "sealing_commit", "sealed_at", "baseline", "candidate", "training_data",
        "market_model_version", "train_seasons", "positions", "label_filter", "model", "source_identities",
        "target_recipe", "gate", "walk_forward", "bootstrap", "registry_path", "input_registration_path", "output_template",
    }
    _fields(manifest, required, "manifest")
    if manifest["schema_version"] != 2 or manifest["protocol"] != "model-2026-confirmation":
        raise ValueError("manifest protocol identity drift")
    _nonempty(manifest["sealing_commit"], "manifest sealing_commit")
    _timestamp(manifest["sealed_at"], "manifest sealed_at")
    for name in ("baseline", "candidate"):
        entry = _fields(manifest[name], {"path", "sha256", "label_columns", "provenance"}, name)
        validate_repo_relative_path(entry["path"])
        _sha(entry["sha256"], name + ".sha256")
        if entry["label_columns"] != list(_LABEL_COLUMNS):
            raise ValueError(name + " label column contract drift")
        provenance = _fields(entry["provenance"], {"model_version", "verified"}, name + ".provenance")
        expected_provenance = {"model_version": "baseline-hl28-v1", "verified": True}
        if provenance != expected_provenance:
            raise _first_difference(expected_provenance, provenance, name + " declared provenance")
        _validate_live_label_identity(name, entry, repo_root)
    _identity(manifest["training_data"], "training_data")
    if manifest["market_model_version"] != "baseline-hl28-v1":
        raise ValueError("market model version drift")
    if manifest["train_seasons"] != [2021, 2022, 2023, 2024, 2025]:
        raise ValueError("training seasons drift")
    if manifest["positions"] != ["QB", "RB", "TE", "WR"]:
        raise ValueError("position contract drift")
    if manifest["label_filter"] != {"min_trades": 5, "positive_value_columns": ["start_value", "end_value"]}:
        raise ValueError("label filter drift")
    expected_model = {"features": model.FEATURES, "hgb_params": model.HGB_PARAMS, "monotonic": model.MONOTONIC, "points_grid_step": model.POINTS_GRID_STEP, "scale_max": 10000.0}
    if manifest["model"] != expected_model:
        raise _first_difference(expected_model, manifest["model"], "model")
    sources = manifest["source_identities"]
    if not isinstance(sources, dict) or set(sources) != set(_SOURCE_PATHS):
        raise _drift("source identities paths", sorted(_SOURCE_PATHS), sorted(sources) if isinstance(sources, dict) else type(sources).__name__)
    for source_path in _SOURCE_PATHS:
        identity = sources[source_path]
        live_hash = file_sha256(repo_root / validate_repo_relative_path(source_path))
        if identity != live_hash:
            raise _drift(source_path + " sha256", live_hash, identity)
    recipe = _fields(manifest["target_recipe"], {"sql", "build", "output_columns", "provenance_contract", "stats_contract"}, "target_recipe")
    _identity(recipe["sql"], "target_recipe.sql")
    _identity(recipe["build"], "target_recipe.build")
    expected_recipe_paths = {"sql": "backend/predictor_v5/sql/backchannel_labels.sql", "build": "backend/predictor_v5/label_export.py"}
    actual_recipe_paths = {name: recipe[name]["path"] for name in expected_recipe_paths}
    if actual_recipe_paths != expected_recipe_paths:
        raise _first_difference(expected_recipe_paths, actual_recipe_paths, "target recipe paths")
    for recipe_name in ("sql", "build"):
        identity = recipe[recipe_name]
        recipe_path = validate_repo_relative_path(identity["path"])
        live_hash = file_sha256(repo_root / recipe_path)
        if identity["sha256"] != live_hash:
            raise _drift(str(recipe_path) + " recipe sha256", live_hash, identity["sha256"])
    if recipe["output_columns"] != list(_LABEL_COLUMNS):
        raise ValueError("target recipe output column drift")
    if recipe["provenance_contract"] != {"helper": "registered_sidecar", "schema_version": 1, "model_version": "baseline-hl28-v1", "verified": True}:
        raise ValueError("label export provenance contract drift")
    if recipe["stats_contract"] != {"registration_field": "target_stats", "format": "training-players-zip", "seasons": [2026], "verified_columns": ["position", "age", "games", "points"]}:
        raise ValueError("target stats contract drift")
    if manifest["gate"] != {"minimum_gain": 0.02, "minimum_position_count": 30, "maximum_position_drop": 0.05}:
        raise ValueError("gate drift")
    if manifest["walk_forward"] != {"folds": {"2022": [2021], "2023": [2021, 2022], "2024": [2021, 2022, 2023], "2025": [2021, 2022, 2023, 2024]}, "no_shortening": True}:
        raise ValueError("walk-forward contract drift")
    if manifest["bootstrap"] != {"seed": 20261024, "resamples": 10000, "percentiles": [2.5, 97.5]}:
        raise ValueError("bootstrap contract drift")
    if manifest["registry_path"] != str(REGISTRY_PATH) or manifest["input_registration_path"] != "backend/predictor_v5/data/model_2026_confirmation_inputs.json":
        raise ValueError("protocol artifact path drift")
    if manifest["output_template"] != "backend/predictor_v5/artifacts/confirmations/2026-<manifest-sha256>.json":
        raise ValueError("output template drift")


def validate_registry(registry: dict[str, Any]) -> None:
    _fields(registry, {"schema_version", "attestation", "entries"}, "registry")
    if registry["schema_version"] != 2 or registry["attestation"] != "review and Git evidence; not proof of non-observation":
        raise ValueError("registry identity drift")
    if not isinstance(registry["entries"], list):
        raise ValueError("registry entries must be a list")  # noqa: TRY004 -- protocol failures are ValueError
    keys = set()
    for entry in registry["entries"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("status"), str) or entry["status"] not in {"sealed", "untouched", "reused"}:
            raise ValueError("invalid registry lifecycle entry")
        status = entry["status"]
        fields = {"season", "manifest_sha256", "status", "sealing_commit", "sealed_at"}
        if status in {"untouched", "reused"}:
            fields |= {"reviewer", "evidence_reference", "evidence_commit", "reviewed_at", "transition_source"}
        _fields(entry, fields, "registry entry")
        _nonempty(entry["sealing_commit"], "registry sealing_commit")
        sealed = _timestamp(entry["sealed_at"], "registry sealed_at")
        if status != "sealed" and _review_metadata(entry, "registry") < sealed:
            raise ValueError("registry reviewed_at predates sealing")
        if not isinstance(entry["season"], int) or entry["season"] < 1:
            raise ValueError("registry season must be a positive integer")
        _sha(entry["manifest_sha256"], "registry manifest_sha256")
        key = (entry["season"], entry["manifest_sha256"])
        if key in keys:
            raise ValueError("duplicate registry key")
        keys.add(key)


def validate_input_registration(registration: dict[str, Any]) -> None:
    """Validate the post-season artifact without reading its target files."""
    required = {
        "schema_version", "season", "manifest_sha256", "target_csv", "target_sidecar", "target_stats", "provenance",
        "target_contract", "recipe", "holdout_attestation",
    }
    _fields(registration, required, "input registration")
    if registration["schema_version"] != 2 or registration["season"] != 2026:
        raise ValueError("input registration identity drift")
    _sha(registration["manifest_sha256"], "input registration manifest_sha256")
    _identity(registration["target_csv"], "target_csv")
    _identity(registration["target_sidecar"], "target_sidecar")
    _identity(registration["target_stats"], "target_stats")
    provenance = _fields(registration["provenance"], {"model_version", "verified", "identity"}, "provenance")
    if provenance["model_version"] != "baseline-hl28-v1" or provenance["verified"] is not True or not isinstance(provenance["identity"], str) or not provenance["identity"]:
        raise ValueError("target provenance drift")
    contract = _fields(registration["target_contract"], {"raw_market_columns", "joined_model_columns", "label_filter"}, "target_contract")
    if contract["raw_market_columns"] != list(_RAW_MARKET_COLUMNS):
        raise ValueError("target raw market columns drift")
    if contract["joined_model_columns"] != ["player_id", "season", "position", "age", "points", "start_value", "end_value"]:
        raise ValueError("target joined model columns drift")
    if contract["label_filter"] != {"positive_values": True, "min_trades": 5}:
        raise ValueError("target label filter drift")
    recipe = _fields(registration["recipe"], {"sql_path", "sql_sha256", "build_path", "build_sha256"}, "recipe")
    if recipe["sql_path"] != "backend/predictor_v5/sql/backchannel_labels.sql" or recipe["build_path"] != "backend/predictor_v5/label_export.py":
        raise ValueError("target recipe path drift")
    _sha(recipe["sql_sha256"], "recipe.sql_sha256")
    _sha(recipe["build_sha256"], "recipe.build_sha256")
    attestation = _fields(
        registration["holdout_attestation"],
        {"registry_key", "status", "reviewer", "evidence_commit", "evidence_reference", "reviewed_at", "transition_source"},
        "holdout_attestation",
    )
    if attestation["registry_key"] != "2026:" + registration["manifest_sha256"]:
        raise ValueError("holdout attestation key mismatch")
    if attestation["status"] != "untouched":
        raise ValueError("holdout attestation must be untouched")
    _review_metadata(attestation, "holdout attestation")


def _repo_file(root: Path, value: str, context: str) -> Path:
    relative = validate_repo_relative_path(value)
    root = root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"{context} is not a regular repository file")
    return path


def _config_file(config: ConfirmationConfig, value: Path, context: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        if not path.is_file():
            raise ValueError(f"{context} is not a regular file")
        return path
    return _repo_file(config.repo_root, str(path), context)


def _default_target_loader(target_csv: Path, target_stats: Path) -> pd.DataFrame:
    values = pd.read_csv(target_csv, dtype={"player_id": str})
    label_export.validate_market_export(values, expected_seasons=(2026,), model_version="baseline-hl28-v1")
    stats = _read_registered_stats_archive(target_stats, "target stats")
    if set(stats["season"]) != {2026}:
        raise ValueError("target stats must contain only season 2026")
    joined = label_export.build_label_frame(values, stats)
    keys = ["player_id", "season"]
    checked = values.merge(joined, on=keys, how="left", suffixes=("_csv", "_stats"), validate="one_to_one", indicator=True)
    missing = checked[checked["_merge"] != "both"]
    if not missing.empty:
        raise ValueError(f"target stats missing keys count={len(missing)} sample={missing[keys].head().to_dict('records')}")
    for column in ("position", "age", "games", "points"):
        bad = checked[checked[column + "_csv"] != checked[column + "_stats"]]
        if not bad.empty:
            raise ValueError(f"target stats mismatch {column} count={len(bad)} sample={bad[keys].head().to_dict('records')}")
    return joined


def _validate_registered_sidecar(sidecar: dict, registration: dict) -> None:
    _fields(sidecar, {"schema_version", "target_csv", "target_stats", "model_version", "verified", "problems", "source", "identity"}, "target sidecar")
    if type(sidecar["schema_version"]) is not int or sidecar["schema_version"] != 1:
        raise ValueError("target sidecar schema_version must be 1")
    for name in ("target_csv", "target_stats"):
        _identity(sidecar[name], "target sidecar " + name)
        if sidecar[name] != registration[name]:
            raise ValueError(f"target sidecar {name} mismatch")
    expected = {"model_version": "baseline-hl28-v1", "verified": True, "problems": [], "source": "export", "identity": registration["provenance"]["identity"]}
    for name, value in expected.items():
        if type(sidecar[name]) is not type(value) or sidecar[name] != value:
            raise ValueError(f"target sidecar {name} mismatch")
    if sidecar["identity"] != "sidecar":
        raise ValueError("target sidecar identity must be sidecar")


def _matching_entry(entries: list, season: int, manifest_sha: str, context: str) -> dict:
    found = [entry for entry in entries if entry["season"] == season and entry["manifest_sha256"] == manifest_sha]
    if len(found) != 1:
        raise ValueError(f"{context} must contain exactly one matching entry")
    return found[0]


def _validate_target_frame(frame: pd.DataFrame, context: PreflightContext) -> None:
    required = ["player_id", "season", "position", "age", "points", "start_value", "end_value", "start_trades_window", "end_trades_window"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"joined target missing columns: {missing}")
    data = frame.copy()
    data["player_id"] = data["player_id"].astype(str)
    _reject_duplicate_rows(data, "joined target")
    if set(data["season"].astype(int)) - {2026}:
        raise ValueError("target contains training-season leakage")
    filtered = data[(data["start_value"] > 0) & (data["end_value"] > 0) & (data["start_trades_window"] >= 5) & (data["end_trades_window"] >= 5)]
    context.target_frame = filtered.sort_values(["season", "player_id"], kind="stable").reset_index(drop=True)


def _reject_duplicate_rows(data: pd.DataFrame, source: str) -> None:
    bad = data[data.duplicated(["player_id", "season"], keep=False)]
    if not bad.empty:
        raise PreflightError(f"{source} duplicate keys count={len(bad)} sample={bad[['player_id', 'season']].head().to_dict('records')}")


def _reject_future_rows(data: pd.DataFrame, source: str) -> None:
    bad = data[pd.to_numeric(data["season"], errors="raise") >= 2026]
    if not bad.empty:
        raise PreflightError(f"{source} future leakage count={len(bad)} sample={bad[['player_id', 'season']].head().to_dict('records')}")


def _read_registered_stats_archive(path: Path, purpose: str) -> pd.DataFrame:
    """Read a hash-verified stats ZIP, classifying only known input defects."""
    try:
        with zipfile.ZipFile(path) as archive:
            member = next((name for name in archive.namelist() if name.endswith(".json")), None)
            if member is None:
                raise ValueError(f"{purpose}: missing JSON member")
            try:
                raw = archive.read(member)
            except zlib.error as error:
                raise ValueError(f"{purpose}: invalid ZIP archive ({error})") from error
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ValueError(f"{purpose}: invalid ZIP archive ({error})") from error
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{purpose}: malformed JSON ({error})") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("players"), list):
        raise ValueError(f"{purpose}: invalid stats structure (players must be a list)")  # noqa: TRY004 -- input refusal
    rows = []
    try:
        for player in payload["players"]:
            if not isinstance(player, dict):
                raise TypeError("player must be an object")
            position = player["position"]
            player_id = player["player_id"]
            seasons = player["seasons"]
            if isinstance(player_id, bool) or not isinstance(player_id, (str, int)):
                raise TypeError("player_id must be a string or integer")
            if not isinstance(position, str):
                raise TypeError("position must be a string")
            if not isinstance(seasons, list):
                raise TypeError("seasons must be a list")
            if position not in ("QB", "RB", "TE", "WR"):
                continue
            for season in seasons:
                if not isinstance(season, dict):
                    raise TypeError("season must be an object")
                rows.append({
                    "player_id": str(player_id),
                    "name": player.get("name"),
                    "season": int(season["year"]),
                    "position": position,
                    "age": float(season["age"]),
                    "points": float(season.get("fantasy_points") or 0.0),
                    "games": int(season.get("games_played") or 0),
                    "years_exp": int(season["years_exp"]) if season.get("years_exp") is not None else -1,
                })
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{purpose}: invalid stats row ({error})") from error
    return pd.DataFrame(rows, columns=(
        "player_id", "name", "season", "position", "age", "points", "games", "years_exp",
    ))


def _require_coverage(data: pd.DataFrame, seasons, positions, source: str) -> None:
    actual = set(zip(data["season"], data["position"]))
    missing = [(season, position) for season in seasons for position in positions if (season, position) not in actual]
    if missing:
        raise PreflightError(f"{source} coverage missing count={len(missing)} sample={missing[:5]}")


def _validate_folds(data: pd.DataFrame, seasons, positions, folds: dict, source: str) -> None:
    _require_coverage(data, seasons, positions, source + " trainable")
    for test_season, history in folds.items():
        _require_coverage(data, history, positions, f"{source} fold {test_season} train")
        _require_coverage(data, [int(test_season)], positions, f"{source} fold {test_season} test")


def _prepare_training(context: PreflightContext) -> dict:
    stats = _read_registered_stats_archive(context.training_archive, "training archive")
    _reject_future_rows(stats, "training archive raw")
    raw, filtered, joined, trainable = {}, {}, {}, {}
    # Validate both complete raw sources before any filtering can hide defects.
    for name, path in (("baseline", context.baseline_csv), ("candidate", context.candidate_csv)):
        raw[name] = pd.read_csv(path, dtype={"player_id": str})
        _reject_duplicate_rows(raw[name], name + " raw")
        _reject_future_rows(raw[name], name + " raw")
    for name, path in (("baseline", context.baseline_csv), ("candidate", context.candidate_csv)):
        filtered[name] = CsvLabels(path, name=name, scale_max=10000.0, min_trades=5).load()
        # Check the unfiltered join first; positive-value filtering must not hide duplicates.
        unfiltered_join = stats.merge(filtered[name], on=["player_id", "season"], how="inner")
        _reject_duplicate_rows(unfiltered_join, name + " joined")
        joined[name] = build_dataset(stats, _LoadedLabels(name, 10000.0, filtered[name]))
        data = joined[name]
        numeric = data[["age", "points", "start_value", "end_value"]].to_numpy(float)
        trainable[name] = data[
            data["season"].isin(context.train_seasons)
            & data["position"].isin(context.positions)
            & np.isfinite(numeric).all(axis=1)
        ].sort_values(["season", "player_id"], kind="stable").reset_index(drop=True)
        _validate_folds(trainable[name], context.train_seasons, context.positions,
                        context.manifest["walk_forward"]["folds"], name)
    return {"raw": raw, "filtered": filtered, "joined": joined, "trainable": trainable}


def preflight(config: ConfirmationConfig) -> PreflightContext:
    """Verify every sealed input before allowing the target loader to execute."""
    try:
        return _preflight(config)
    except (ValueError, OSError) as error:
        raise PreflightError(str(error)) from error


def _preflight(config: ConfirmationConfig) -> PreflightContext:
    manifest_path = _config_file(config, config.manifest_path, "manifest")
    manifest, manifest_sha = load_canonical_json(manifest_path, lambda value: validate_manifest(value, config.repo_root))
    output_dir = config.output_dir if config.output_dir.is_absolute() else config.repo_root / config.output_dir
    output_path = output_dir / ("2026-" + manifest_sha + ".json")
    if output_path.exists():
        raise ValueError("confirmation output already exists")
    registry_path = _config_file(config, config.registry_path, "registry")
    registry, registry_sha = load_canonical_json(registry_path, validate_registry)
    registry_entry = _matching_entry(registry["entries"], 2026, manifest_sha, "registry")
    for name in ("sealing_commit", "sealed_at"):
        if registry_entry[name] != manifest[name]:
            raise ValueError(f"registry {name} mismatch with manifest")
    if registry_entry["status"] != "untouched":
        raise ValueError("2026 registry entry must be untouched")
    registration_path = _config_file(config, config.input_registration_path, "input registration")
    registration, registration_sha = load_canonical_json(registration_path, validate_input_registration)
    if registration["manifest_sha256"] != manifest_sha:
        raise ValueError("input registration manifest hash mismatch")
    attestation = registration["holdout_attestation"]
    if attestation["registry_key"] != "2026:" + manifest_sha or attestation["status"] != registry_entry["status"]:
        raise ValueError("input registration does not attest the registry entry")
    for key in ("reviewer", "evidence_commit", "evidence_reference", "reviewed_at", "transition_source"):
        if registry_entry.get(key) != attestation[key]:
            raise ValueError(f"registry attestation {key} mismatch")
    target_csv = _repo_file(config.repo_root, registration["target_csv"]["path"], "target CSV")
    target_sidecar = _repo_file(config.repo_root, registration["target_sidecar"]["path"], "target sidecar")
    target_stats = _repo_file(config.repo_root, registration["target_stats"]["path"], "target stats")
    for path, entry, name in ((target_csv, registration["target_csv"], "target CSV"), (target_sidecar, registration["target_sidecar"], "target sidecar"), (target_stats, registration["target_stats"], "target stats")):
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"{name} hash mismatch")
    load_canonical_json(target_sidecar, lambda value: _validate_registered_sidecar(value, registration))
    recipe = registration["recipe"]
    for name, path_key, hash_key in (("sql", "sql_path", "sql_sha256"), ("build", "build_path", "build_sha256")):
        path = _repo_file(config.repo_root, recipe[path_key], "recipe " + name)
        if file_sha256(path) != recipe[hash_key] or recipe[hash_key] != manifest["target_recipe"][name]["sha256"]:
            raise ValueError(f"recipe {name} hash mismatch")
    training_archive = _repo_file(config.repo_root, manifest["training_data"]["path"], "training archive")
    if file_sha256(training_archive) != manifest["training_data"]["sha256"]:
        raise ValueError("training archive hash mismatch")
    baseline_csv = _repo_file(config.repo_root, manifest["baseline"]["path"], "baseline CSV")
    candidate_csv = _repo_file(config.repo_root, manifest["candidate"]["path"], "candidate CSV")
    for path, entry, name in ((baseline_csv, manifest["baseline"], "baseline CSV"), (candidate_csv, manifest["candidate"], "candidate CSV")):
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"{name} hash mismatch")
    context = PreflightContext(manifest=manifest, manifest_sha256=manifest_sha, input_registration=registration,
        input_registration_sha256=registration_sha, holdout_registry=registry, holdout_registry_sha256=registry_sha,
        baseline_csv=baseline_csv, candidate_csv=candidate_csv,
        training_archive=training_archive, target_csv=target_csv, target_sidecar=target_sidecar, target_stats=target_stats, output_path=output_path,
        train_seasons=tuple(manifest["train_seasons"]), positions=tuple(manifest["positions"]),
        label_filter=registration["target_contract"]["label_filter"], target_recipe=registration["recipe"], target_loader=config.target_loader)
    context.training_frames = _prepare_training(context)
    raw = pd.read_csv(target_csv, dtype={"player_id": str})
    if list(raw.columns) != list(_LABEL_COLUMNS):
        raise ValueError("target CSV build-labels column contract drift")
    # Derived stat fields are verified against the independently registered stats.
    label_export.validate_market_export(raw.loc[:, _RAW_MARKET_COLUMNS], expected_seasons=(2026,))
    _validate_target_frame(config.target_loader(target_csv, target_stats), context)
    return context


def run_confirmation(config: ConfirmationConfig) -> dict:
    """Preflight every sealed input, then execute the fixed confirmation."""
    context = preflight(config)
    return fit_and_score(context)


def report_path(config: ConfirmationConfig) -> Path:
    """Derive the sealed output path without accepting a caller-selected name."""
    manifest_path = _config_file(config, config.manifest_path, "manifest")
    _, manifest_sha = load_canonical_json(
        manifest_path, lambda value: validate_manifest(value, config.repo_root)
    )
    output_dir = config.output_dir if config.output_dir.is_absolute() else config.repo_root / config.output_dir
    return output_dir / f"2026-{manifest_sha}.json"


def _git_revision(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision if len(revision) == 40 else None


def validate_report(wrapper: dict[str, Any]) -> None:
    """Validate the exact outer report schema and its payload-only digest."""
    fields = _fields(wrapper, {"schema_version", "payload", "payload_sha256"}, "report")
    if fields["schema_version"] != 1:
        raise ValueError("report schema_version must be 1")
    if not isinstance(fields["payload"], dict):
        raise ValueError("report payload must be an object")  # noqa: TRY004
    _sha(fields["payload_sha256"], "report payload_sha256")
    expected = sha256_bytes(canonical_json_bytes(fields["payload"]))
    if fields["payload_sha256"] != expected:
        raise ValueError("report payload_sha256 mismatch")


def _sync_directory(path: Path) -> None:
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    except Exception as error:
        try:
            os.close(directory)
        except OSError as cleanup_error:
            error.add_note(f"failed to close directory after sync failure: {cleanup_error}")
        raise
    else:
        os.close(directory)


def write_report(payload: dict, final_path: Path) -> dict:
    """Write one canonical report atomically, without replacing an existing report."""
    payload = dict(payload)
    decision = payload.get("decision")
    if not isinstance(decision, dict) or decision.get("promotion_authorized") is not False:
        raise ValueError("report decision must set promotion_authorized to false")
    payload["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "git_revision": _git_revision(REPOSITORY_ROOT),
    }
    wrapper = {
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }
    validate_report(wrapper)
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        raise FileExistsError(f"confirmation output already exists: {final_path}")
    temporary: Path | None = None
    created_final = False
    active_error: BaseException | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(wrapper))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, final_path)
        created_final = True
        temporary.unlink()
        temporary = None
        _sync_directory(final_path.parent)
    except Exception as error:
        active_error = error
        if created_final:
            try:
                final_path.unlink()
            except OSError as cleanup_error:
                error.add_note(f"failed to roll back confirmation output: {cleanup_error}")
            else:
                try:
                    _sync_directory(final_path.parent)
                except OSError as cleanup_error:
                    error.add_note(f"failed to sync confirmation directory after rollback: {cleanup_error}")
        raise
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                if active_error is None:
                    raise
                active_error.add_note(f"failed to remove temporary confirmation output: {cleanup_error}")
    return wrapper


_SCORED_COLUMNS = [
    "player_id", "season", "position", "start_value", "end_value", "pred_value",
    "abs_error", "no_change_abs_error", "residual",
]
_DIAGNOSTIC_TAGS = {"used_for_configuration": False, "used_for_promotion": False}


@dataclass(frozen=True)
class _LoadedLabels:
    """Expose one already-loaded label frame through the LabelSource contract."""

    name: str
    scale_max: float
    frame: pd.DataFrame

    def load(self) -> pd.DataFrame:
        return self.frame.copy()


def score_predictions(
    predictions: pd.DataFrame,
    target: pd.DataFrame,
    keys: list[tuple[str, int]],
) -> pd.DataFrame:
    """Apply model-relative changes to one shared target value scale."""
    normalized_keys = sorted((str(player_id), int(season)) for player_id, season in keys)
    index = pd.MultiIndex.from_tuples(normalized_keys, names=["player_id", "season"])
    own = predictions.assign(
        player_id=predictions["player_id"].astype(str), season=predictions["season"].astype(int)
    ).set_index(["player_id", "season"])
    truth = target.assign(
        player_id=target["player_id"].astype(str), season=target["season"].astype(int)
    ).set_index(["player_id", "season"])
    missing = [key for key in normalized_keys if key not in own.index or key not in truth.index]
    if missing:
        raise KeyError(f"keys missing from predictions or target: {missing[:5]}")
    own = own.loc[index]
    truth = truth.loc[index]
    pred_value = (
        truth["start_value"].to_numpy(float)
        * own["pred_value"].to_numpy(float)
        / own["start_value"].to_numpy(float)
    )
    end_value = truth["end_value"].to_numpy(float)
    start_value = truth["start_value"].to_numpy(float)
    scored = pd.DataFrame({
        "player_id": [key[0] for key in normalized_keys],
        "season": [key[1] for key in normalized_keys],
        "position": truth["position"].astype(str).to_numpy(),
        "start_value": start_value,
        "end_value": end_value,
        "pred_value": pred_value,
        "abs_error": np.abs(pred_value - end_value),
        "no_change_abs_error": np.abs(start_value - end_value),
        "residual": pred_value - end_value,
    })
    return scored.loc[:, _SCORED_COLUMNS]


def summarize_metrics(scored: pd.DataFrame) -> dict:
    """Return full-precision error and residual diagnostics."""
    if scored.empty:
        return {
            "n": 0, "mae": None, "median_absolute_error": None, "p90_absolute_error": None,
            "signed_residual_mean": None,
            "residual_summary": {"min": None, "p10": None, "median": None, "p90": None, "max": None},
            **_DIAGNOSTIC_TAGS,
        }
    errors = scored["abs_error"].to_numpy(float)
    residuals = scored["residual"].to_numpy(float)
    return {
        "n": len(scored),
        "mae": float(np.mean(errors)),
        "median_absolute_error": float(np.median(errors)),
        "p90_absolute_error": float(np.percentile(errors, 90)),
        "signed_residual_mean": float(np.mean(residuals)),
        "residual_summary": {
            "min": float(np.min(residuals)),
            "p10": float(np.percentile(residuals, 10)),
            "median": float(np.median(residuals)),
            "p90": float(np.percentile(residuals, 90)),
            "max": float(np.max(residuals)),
        },
        **_DIAGNOSTIC_TAGS,
    }


def _coverage_counts(frame: pd.DataFrame) -> dict:
    by_season = frame.groupby("season", sort=True).size() if "season" in frame else pd.Series(dtype=int)
    by_position = frame.groupby("position", sort=True).size() if "position" in frame else pd.Series(dtype=int)
    return {
        "total": len(frame),
        "by_season": {str(int(key)): int(value) for key, value in by_season.items()},
        "by_position": {str(key): int(value) for key, value in by_position.items()},
    }


def coverage_report(raw, filtered, joined, trainable, one_sided, matched, missing) -> dict:
    """Count per-source training funnels and unique target cohorts."""
    training = {}
    for source in ("baseline", "candidate"):
        training[source] = {
            "raw": _coverage_counts(raw[source]),
            "filtered": _coverage_counts(filtered[source]),
            "joined": _coverage_counts(joined[source]),
            "trainable": _coverage_counts(trainable[source]),
        }
    return {
        "training": training,
        "target": {
            "one_sided": _coverage_counts(one_sided),
            "matched": _coverage_counts(matched),
            "missing": _coverage_counts(missing),
        },
        **_DIAGNOSTIC_TAGS,
    }


def bootstrap_delta(
    scored: pd.DataFrame,
    seed: int = 20261024,
    resamples: int = 10000,
    percentile: tuple[float, float] = (2.5, 97.5),
) -> dict:
    """Bootstrap paired candidate-minus-baseline absolute-error deltas."""
    ordered = scored.copy()
    sort_columns = [column for column in ("season", "player_id") if column in ordered]
    if sort_columns:
        ordered = ordered.sort_values(sort_columns, kind="stable")
    delta = (
        ordered["candidate_abs_error"].to_numpy(float)
        - ordered["baseline_abs_error"].to_numpy(float)
    )
    if not len(delta):
        interval = {"lower": None, "upper": None, "percentile": [float(percentile[0]), float(percentile[1])]}
        return {
            "n": 0, "mean_delta": None, "median_delta": None, "candidate_wins": 0, "ties": 0,
            "candidate_losses": 0, "interval": interval, "seed": int(seed), "resamples": int(resamples),
            **_DIAGNOSTIC_TAGS,
        }
    rng = np.random.default_rng(seed)
    sampled = delta[rng.integers(0, len(delta), size=(resamples, len(delta)))].mean(axis=1)
    bounds = np.percentile(sampled, percentile)
    return {
        "n": len(delta),
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "candidate_wins": int(np.sum(delta < 0)),
        "ties": int(np.sum(delta == 0)),
        "candidate_losses": int(np.sum(delta > 0)),
        "interval": {"lower": float(bounds[0]), "upper": float(bounds[1]), "percentile": [float(percentile[0]), float(percentile[1])]},
        "seed": int(seed),
        "resamples": int(resamples),
        **_DIAGNOSTIC_TAGS,
    }


def _score_summary(scored: pd.DataFrame) -> dict:
    return {
        "overall": summarize_exact(scored),
        "by_position": {
            str(position): summarize_exact(group)
            for position, group in scored.groupby("position", sort=True)
        },
    }


def evaluate_gate(
    baseline: dict,
    candidate: dict,
    positions: tuple[str, ...],
    minimum_gain: float = 0.02,
    maximum_position_drop: float = 0.05,
    minimum_position_count: int = 30,
) -> dict:
    """Evaluate the preregistered promotion gate without display rounding."""
    reasons = []
    base_overall, candidate_overall = baseline["overall"], candidate["overall"]

    def valid_count(value) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0

    def finite(value) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)

    for name, summary in (("baseline", base_overall), ("candidate", candidate_overall)):
        count = summary.get("n")
        if not valid_count(count):
            reasons.append(f"invalid_overall_n:{name}")
        elif count == 0:
            reasons.append("empty_overlap")
        no_change_mae = summary.get("no_change_mae")
        if not finite(no_change_mae):
            reasons.append(f"invalid_overall_no_change_mae:{name}")
        elif no_change_mae <= 0:
            reasons.append("zero_no_change_mae")
        if not finite(summary.get("skill")):
            reasons.append(f"invalid_overall_skill:{name}")
    base_skill, candidate_skill = base_overall.get("skill"), candidate_overall.get("skill")
    gain = candidate_skill - base_skill if finite(base_skill) and finite(candidate_skill) else None
    if gain is not None and not finite(gain):
        reasons.append("invalid_gain")
        gain = None
    elif gain is not None and gain < minimum_gain:
        reasons.append("minimum_gain_not_met")
    position_deltas = {}
    for position in positions:
        base_position = baseline.get("by_position", {}).get(position)
        candidate_position = candidate.get("by_position", {}).get(position)
        if base_position is None or candidate_position is None:
            reasons.append(f"missing_position:{position}")
            continue
        position_valid = True
        for name, summary in (("baseline", base_position), ("candidate", candidate_position)):
            count = summary.get("n")
            if not valid_count(count):
                reasons.append(f"invalid_position_n:{name}:{position}")
                position_valid = False
            elif count < minimum_position_count:
                reasons.append(f"insufficient_position_count:{position}")
            no_change_mae = summary.get("no_change_mae")
            if not finite(no_change_mae) or no_change_mae <= 0:
                reasons.append(f"invalid_position_no_change_mae:{name}:{position}")
                position_valid = False
            if not finite(summary.get("skill")):
                reasons.append(f"invalid_position_skill:{name}:{position}")
                position_valid = False
        left, right = base_position.get("skill"), candidate_position.get("skill")
        if not position_valid or not finite(left) or not finite(right):
            continue
        position_deltas[position] = right - left
        if not finite(position_deltas[position]):
            reasons.append(f"invalid_position_delta:{position}")
            del position_deltas[position]
        elif position_deltas[position] < -maximum_position_drop:
            reasons.append(f"maximum_position_drop_exceeded:{position}")
    return {
        "passed": not reasons,
        "gain": gain,
        "position_skill_deltas": position_deltas,
        "thresholds": {
            "minimum_gain": minimum_gain,
            "maximum_position_drop": maximum_position_drop,
            "minimum_position_count": minimum_position_count,
        },
        "reasons": reasons,
    }


def _fit_label_set(name: str, dataset: pd.DataFrame, target: pd.DataFrame, train_seasons: tuple[int, ...], folds: dict) -> tuple[dict, pd.DataFrame, dict]:
    _validate_folds(dataset, train_seasons, ("QB", "RB", "TE", "WR"), folds, name)
    fold_results = {}
    for test_season, fold_train_seasons in sorted((int(key), value) for key, value in folds.items()):
        train = dataset[dataset["season"].isin(fold_train_seasons)]
        test = dataset[dataset["season"] == test_season].copy()
        fitted = WhatIfModel(label_source=name, scale_max=10000.0).fit(train)
        test["pred_value"] = fitted.predict(test)
        fold_scored = score_predictions(test, test, list(zip(test["player_id"].astype(str), test["season"].astype(int))))
        fold_results[str(test_season)] = {"train_seasons": list(fold_train_seasons), "metrics": summarize_metrics(fold_scored), **_DIAGNOSTIC_TAGS}
    training = dataset[dataset["season"].isin(train_seasons)].sort_values(["season", "player_id"], kind="stable")
    fitted = WhatIfModel(label_source=name, scale_max=10000.0).fit(training)
    predictions = target.copy().sort_values(["season", "player_id"], kind="stable")
    predictions["pred_value"] = fitted.predict(predictions) if not predictions.empty else pd.Series(dtype=float)
    fit = {"label_source": name, "train_seasons": list(train_seasons), "n_train": len(training), "scale_max": 10000.0}
    return fit, predictions, fold_results


def fit_and_score(context: PreflightContext) -> dict:
    """Fit both preregistered models and score their common 2026 cohort."""
    prepared = context.training_frames if context.training_frames is not None else _prepare_training(context)
    raw_labels, filtered_labels = prepared["raw"], prepared["filtered"]
    joined_labels, datasets = prepared["joined"], prepared["trainable"]
    target = context.target_frame.copy().loc[:, ["player_id", "season", "position", "age", "points", "start_value", "end_value"]]
    target["player_id"] = target["player_id"].astype(str)
    target = target.sort_values(["season", "player_id"], kind="stable").reset_index(drop=True)
    folds = context.manifest["walk_forward"]["folds"]
    fits, predictions, walk_forward = {}, {}, {}
    for name in ("baseline", "candidate"):
        fits[name], predictions[name], walk_forward[name] = _fit_label_set(name, datasets[name], target, context.train_seasons, folds)
    key_sets = {
        name: set(zip(frame["player_id"].astype(str), frame["season"].astype(int)))
        for name, frame in predictions.items()
    }
    keys = sorted(key_sets["baseline"] & key_sets["candidate"])
    scored = {name: score_predictions(predictions[name], target, keys) for name in ("baseline", "candidate")}
    summaries = {name: _score_summary(scored[name]) for name in ("baseline", "candidate")}
    gate_config = context.manifest["gate"]
    gate = evaluate_gate(
        summaries["baseline"], summaries["candidate"], context.positions,
        minimum_gain=gate_config["minimum_gain"],
        maximum_position_drop=gate_config["maximum_position_drop"],
        minimum_position_count=gate_config["minimum_position_count"],
    )
    paired = scored["baseline"][["player_id", "season", "abs_error"]].rename(columns={"abs_error": "baseline_abs_error"}).merge(
        scored["candidate"][["player_id", "season", "abs_error"]].rename(columns={"abs_error": "candidate_abs_error"}),
        on=["player_id", "season"], validate="one_to_one",
    )
    bootstrap = context.manifest["bootstrap"]
    prediction_key_sets = key_sets
    one_sided_keys = prediction_key_sets["baseline"] ^ prediction_key_sets["candidate"]
    missing_keys = set(zip(target["player_id"], target["season"])) - set(keys)
    coverage = coverage_report(
        raw_labels, filtered_labels, joined_labels, datasets,
        target.loc[[key in one_sided_keys for key in zip(target.player_id, target.season)]],
        target.loc[[key in set(keys) for key in zip(target.player_id, target.season)]],
        target.loc[[key in missing_keys for key in zip(target.player_id, target.season)]],
    )
    diagnostics = {
        "walk_forward": {"folds": {str(key): list(value) for key, value in folds.items()}, "results": walk_forward, **_DIAGNOSTIC_TAGS},
        "baseline_metrics": {"overall": summarize_metrics(scored["baseline"]), "by_position": {p: summarize_metrics(g) for p, g in scored["baseline"].groupby("position", sort=True)}, **_DIAGNOSTIC_TAGS},
        "candidate_metrics": {"overall": summarize_metrics(scored["candidate"]), "by_position": {p: summarize_metrics(g) for p, g in scored["candidate"].groupby("position", sort=True)}, **_DIAGNOSTIC_TAGS},
        "coverage": coverage,
        "bootstrap_delta": bootstrap_delta(paired, seed=bootstrap["seed"], resamples=bootstrap["resamples"], percentile=tuple(bootstrap["percentiles"])),
    }
    passed = gate["passed"]
    decision = {
        "status": "recommend_promotion" if passed else "retain",
        "gate_evaluated": True,
        "gate_passed": passed,
        "promotion_authorized": False,
        "reason": "gate_passed" if passed else ";".join(gate["reasons"]),
        "manifest_sha256": context.manifest_sha256,
        "input_registration_sha256": context.input_registration_sha256,
        "holdout_registry_sha256": context.holdout_registry_sha256,
    }
    return {"fits": fits, "cohort": {"n": len(keys), **summaries}, "gate": gate, "diagnostics": diagnostics, "decision": decision}
