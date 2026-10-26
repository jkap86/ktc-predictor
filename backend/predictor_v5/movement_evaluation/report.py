"""Canonical assembly and crash-safe publication for movement reports."""

from __future__ import annotations

import fcntl
import hashlib
import math
import os
import platform
import re
import secrets
import stat
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn

from .contract import (
    BASE_COMMIT,
    FOLDS,
    POSITIONS,
    ContractRefusal,
    KtcMode,
    contract_payload,
    key_sha256,
    validate_ktc_mode,
)
from .contract import canonical_bytes as _contract_canonical_bytes
from .data import EvaluationData, HistoricalBundle
from .models import FoldResult

_PROTOCOL = "market-movement-development"
_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_THREAD_VARIABLES = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_SOURCE_PATHS = (
    "model.py",
    "movement_evaluation/__init__.py",
    "movement_evaluation/__main__.py",
    "movement_evaluation/contract.py",
    "movement_evaluation/data.py",
    "movement_evaluation/metrics.py",
    "movement_evaluation/models.py",
    "movement_evaluation/report.py",
)
_OPTIONAL_KTC_FIELDS = (
    "status",
    "provenance_limits",
    "cohorts",
    "fits",
    "scores",
    "predictions",
)
_PREDICTION_FIELDS = (
    "season",
    "player_id",
    "position",
    "S",
    "E",
    "r",
    "points",
    "age",
    "start_trades_window",
    "end_trades_window",
    "value_tier",
    "production_E_hat",
    "no_production_E_hat",
    "no_change_E_hat",
)
_FOLD_SCORE_FIELDS = (
    "models",
    "contrasts",
    "direction_baselines",
    "slices",
    "uncertainty",
)


@dataclass
class _LocalPublicationState:
    lock: threading.Lock
    owner: int | None = None


@dataclass(frozen=True)
class _DirectoryChain:
    root_path: Path
    root: int
    backend: int
    package: int
    artifacts: int
    output: int
    output_path: Path


_LOCAL_PUBLICATION_STATES: dict[tuple[int, int], _LocalPublicationState] = {}
_LOCAL_PUBLICATION_STATES_GUARD = threading.Lock()


def _refuse(message: str, cause: BaseException | None = None) -> None:
    if cause is None:
        raise ContractRefusal(message)
    raise ContractRefusal(message) from cause


def _json_value(value: object, *, path: str = "$" ) -> object:
    """Return plain finite JSON data while rejecting ambiguous coercions."""
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                _refuse(f"canonical JSON object key at {path} must be a string")
            result[key] = _json_value(child, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(child, path=f"{path}[{index}]") for index, child in enumerate(value)]
    if isinstance(value, np.generic):
        return _json_value(value.item(), path=path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _refuse(f"canonical JSON number at {path} must be finite")
        return value
    _refuse(f"value at {path} is not canonical JSON")


def canonical_bytes(value: object) -> bytes:
    """Encode strict JSON with the protocol's shared canonical primitive."""
    return _contract_canonical_bytes(_json_value(value))


def runtime_identity() -> dict[str, object]:
    """Describe dependency and thread settings that delimit reproducibility."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "thread_settings": {
            name: os.environ.get(name, "unset") for name in _THREAD_VARIABLES
        },
    }


def _regular_source(path: Path) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        _refuse(f"missing source identity file: {path.name}", exc)
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        _refuse(f"source identity path must be a regular non-symlink file: {path.name}")
    try:
        return path.read_bytes()
    except OSError as exc:
        _refuse(f"could not read source identity file: {path.name}", exc)


def source_identities(package_root: Path) -> dict[str, str]:
    """Hash the seven evaluator sources and protected model dependency."""
    root = Path(package_root)
    try:
        info = root.lstat()
    except OSError as exc:
        _refuse("predictor_v5 package root is missing", exc)
    if root.is_symlink() or not stat.S_ISDIR(info.st_mode):
        _refuse("predictor_v5 package root must be a regular non-symlink directory")
    return {
        relative: hashlib.sha256(_regular_source(root / relative)).hexdigest()
        for relative in _SOURCE_PATHS
    }


def _identity_mapping(
    value: Mapping[str, str], *, name: str, exact_keys: set[str] | None = None
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        _refuse(f"{name} must be a mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str) or not item:
            _refuse(f"{name} keys and values must be nonempty strings")
        if key.endswith("sha256") and _SHA256.fullmatch(item) is None:
            _refuse(f"{name} hash {key} must be lowercase SHA-256")
        result[key] = item
    if exact_keys is not None and set(result) != exact_keys:
        _refuse(f"{name} must contain the exact source identity set")
    return {key: result[key] for key in sorted(result)}


def build_run_identity(
    *,
    inputs: Mapping[str, str],
    sources: Mapping[str, str],
    mode: KtcMode,
    runtime: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    """Build and hash the complete deterministic run identity."""
    validated_mode = validate_ktc_mode(mode)
    source_values = _identity_mapping(sources, name="sources", exact_keys=set(_SOURCE_PATHS))
    for key, value in source_values.items():
        if _SHA256.fullmatch(value) is None:
            _refuse(f"source identity {key} must be lowercase SHA-256")
    input_values = _identity_mapping(inputs, name="inputs")
    required_inputs = {
        "labels_sha256",
        "provenance_sha256",
        "archive_sha256",
        "archive_member_sha256",
        "labels_path",
        "provenance_path",
        "archive_path",
        "archive_member",
    }
    if set(input_values) != required_inputs:
        _refuse("inputs must contain exact input, member, and path identities")
    contract = contract_payload(validated_mode)
    runtime_value = _json_value(runtime, path="$.runtime")
    if not isinstance(runtime_value, dict) or set(runtime_value) != {
        "python",
        "numpy",
        "pandas",
        "scikit_learn",
        "scipy",
        "thread_settings",
    }:
        _refuse("runtime identity is incomplete")
    thread_settings = runtime_value["thread_settings"]
    if not isinstance(thread_settings, dict) or set(thread_settings) != set(_THREAD_VARIABLES):
        _refuse("runtime thread settings are incomplete")
    identity = {
        "protocol": _PROTOCOL,
        "schema_version": _SCHEMA_VERSION,
        "base_commit": BASE_COMMIT,
        "inputs": input_values,
        "sources": source_values,
        "folds": _json_value(contract["folds"]),
        "models": _json_value(contract["models"]),
        "metrics": _json_value(contract["metrics"]),
        "bootstrap": _json_value(contract["bootstrap"]),
        "ktc_mode": validated_mode,
        "runtime": runtime_value,
    }
    digest = hashlib.sha256(canonical_bytes(identity)).hexdigest()
    return identity, digest


def _tier_edges(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(POSITIONS):
        _refuse("fold tier edges must name all positions")
    result: dict[str, object] = {}
    for position in POSITIONS:
        edges = value[position]
        if edges is None:
            result[position] = {
                "q1": None,
                "q2": None,
                "reason": "no_eligible_training_starts",
            }
            continue
        if not isinstance(edges, (tuple, list)) or len(edges) != 2:
            _refuse(f"tier edges for {position} must be a two-value pair")
        q1, q2 = (_json_value(item) for item in edges)
        if isinstance(q1, bool) or isinstance(q2, bool) or not isinstance(q1, (int, float)) or not isinstance(q2, (int, float)):
            _refuse(f"tier edges for {position} must be numeric")
        result[position] = {"q1": q1, "q2": q2, "reason": None}
    return result


def _target_payload(value: Mapping[str, object]) -> dict[str, object]:
    """Convert semantic season keys to their canonical JSON representation."""
    target = dict(value)
    for field in ("dates", "generations"):
        seasons = target.get(field)
        if not isinstance(seasons, Mapping):
            _refuse(f"target {field} must be a season mapping")
        target[field] = {str(season): payload for season, payload in seasons.items()}
    result = _json_value(target, path="$.target")
    if not isinstance(result, dict):
        _refuse("target payload must be an object")
    return result


def _prediction_records(fold: FoldResult) -> list[dict[str, object]]:
    frame = fold.predictions
    if not isinstance(frame, pd.DataFrame):
        _refuse("fold predictions must be a pandas DataFrame")
    missing = sorted(set(_PREDICTION_FIELDS) - set(frame.columns))
    if missing:
        _refuse(f"fold predictions are missing fields: {', '.join(missing)}")
    ordered = frame.sort_values(["season", "player_id"], kind="mergesort").reset_index(drop=True)
    keys = tuple(
        (int(season), str(player_id))
        for season, player_id in ordered[["season", "player_id"]].itertuples(
            index=False, name=None
        )
    )
    if keys != fold.test_keys or len(keys) != len(set(keys)):
        _refuse("fold prediction keys must exactly match unique held-out test keys")
    if any(season != fold.spec.test_season for season, _ in keys):
        _refuse("fold predictions contain a row outside the held-out season")
    records: list[dict[str, object]] = []
    for values in ordered.loc[:, _PREDICTION_FIELDS].itertuples(index=False, name=None):
        source = dict(zip(_PREDICTION_FIELDS, values, strict=True))
        record = {
            "season": int(source["season"]),
            "player_id": str(source["player_id"]),
            "position": str(source["position"]),
            "S": source["S"],
            "E": source["E"],
            "r": source["r"],
            "points": source["points"],
            "age": source["age"],
            "start_trades": source["start_trades_window"],
            "end_trades": source["end_trades_window"],
            "value_tier": str(source["value_tier"]),
            "production_E_hat": source["production_E_hat"],
            "no_production_E_hat": source["no_production_E_hat"],
            "no_change_E_hat": source["no_change_E_hat"],
        }
        records.append(_json_value(record, path="$.predictions"))
    return records


def _ordered_optional_ktc(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(_OPTIONAL_KTC_FIELDS):
        _refuse("optional KTC payload must contain its complete exact schema")
    result = _json_value(value, path="$.optional_ktc")
    if not isinstance(result, dict):
        _refuse("optional KTC payload must be an object")

    def ordered_list(name: str, key: Any) -> None:
        rows = result[name]
        if not isinstance(rows, list):
            _refuse(f"optional KTC {name} must be an array")
        try:
            result[name] = sorted(rows, key=key)
        except (KeyError, TypeError, ValueError) as exc:
            _refuse(f"optional KTC {name} cannot be deterministically ordered", exc)

    ordered_list("cohorts", lambda row: int(row["test_season"]))
    ordered_list("fits", lambda row: (int(row["test_season"]), str(row["name"])))
    ordered_list("scores", lambda row: int(row["test_season"])) if isinstance(result["scores"], list) else None
    ordered_list(
        "predictions", lambda row: (int(row["season"]), str(row["player_id"]))
    )
    return result


def _cell_value(value: object) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("value") if "value" in value else value.get("estimate")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _refuse("interpretation evidence value must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        _refuse("interpretation evidence value must be finite")
    return numeric


def _point_status(value: float | None) -> str:
    if value is None:
        return "undefined"
    if value > 0:
        return "improved"
    if value < 0:
        return "worse"
    return "tied"


def _contrast_value(container: Mapping[str, object], contrast: str) -> float | None:
    contrasts = container.get("contrasts")
    if not isinstance(contrasts, Mapping):
        return None
    payload = contrasts.get(contrast)
    if not isinstance(payload, Mapping):
        return None
    return _cell_value(payload.get("positive_mae_log_improvement"))


def _evidence_flags(
    pooled: Mapping[str, object], fold_reports: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    incremental = _contrast_value(pooled, "production_vs_no_production")
    no_change = _contrast_value(pooled, "production_vs_no_change")
    fold_values = [
        _contrast_value(fold, "production_vs_no_production") for fold in fold_reports
    ]
    by_position = pooled.get("uncertainty", {})
    if isinstance(by_position, Mapping):
        by_position = by_position.get("by_position", {})
    position_values: list[float | None] = []
    for position in POSITIONS:
        value = by_position.get(position, {}) if isinstance(by_position, Mapping) else {}
        position_values.append(
            _contrast_value(value, "production_vs_no_production")
            if isinstance(value, Mapping)
            else None
        )
    return {
        "incremental_production_point_estimate": {
            "status": _point_status(incremental),
            "value": incremental,
            "reason": "undefined_primary_contrast" if incremental is None else None,
        },
        "no_change_point_estimate": {
            "status": _point_status(no_change),
            "value": no_change,
            "reason": "undefined_no_change_contrast" if no_change is None else None,
        },
        "improved_folds": {
            "count": sum(value is not None and value > 0 for value in fold_values),
            "defined": sum(value is not None for value in fold_values),
            "total": len(FOLDS),
        },
        "improved_positions": {
            "count": sum(value is not None and value > 0 for value in position_values),
            "defined": sum(value is not None for value in position_values),
            "total": len(POSITIONS),
        },
    }


def _validate_nullable_metrics(value: object, *, path: str = "$" ) -> None:
    if isinstance(value, Mapping):
        if value.get("value", object()) is None and "value" in value:
            is_nonmetric_container = "cells" in value
            if not is_nonmetric_container and not value.get("reason"):
                _refuse(f"nullable metric at {path} requires an explicit reason")
        if (
            value.get("estimate", object()) is None
            and "estimate" in value
            and not value.get("reason")
        ):
            _refuse(f"nullable estimate at {path} requires an explicit reason")
        for key, child in value.items():
            _validate_nullable_metrics(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_nullable_metrics(child, path=f"{path}[{index}]")


def build_report(
    *,
    bundle: HistoricalBundle,
    data: EvaluationData,
    folds: Sequence[FoldResult],
    fold_scores: Sequence[Mapping[str, object]],
    pooled: Mapping[str, object],
    optional_ktc: Mapping[str, object],
    sources: Mapping[str, str],
    mode: KtcMode,
    runtime: Mapping[str, object],
) -> dict[str, object]:
    """Assemble one deterministic report from explicitly whitelisted values."""
    if not isinstance(bundle, HistoricalBundle) or not isinstance(data, EvaluationData):
        _refuse("report assembly requires HistoricalBundle and EvaluationData")
    validated_mode = validate_ktc_mode(mode)
    run_identity, run_sha256 = build_run_identity(
        inputs=bundle.identities,
        sources=sources,
        mode=validated_mode,
        runtime=runtime,
    )
    fold_list = sorted(folds, key=lambda fold: fold.spec.test_season)
    if not all(isinstance(fold, FoldResult) for fold in fold_list):
        _refuse("folds must contain FoldResult values")
    if tuple(fold.spec for fold in fold_list) != FOLDS:
        _refuse("report requires the complete fixed fold sequence")
    score_by_season: dict[int, Mapping[str, object]] = {}
    for score in fold_scores:
        if not isinstance(score, Mapping) or isinstance(score.get("test_season"), bool):
            _refuse("fold score requires a test season")
        try:
            season = int(score["test_season"])
        except (KeyError, TypeError, ValueError) as exc:
            _refuse("fold score requires an integer test season", exc)
        if season in score_by_season:
            _refuse("duplicate fold score season")
        score_by_season[season] = score
    if set(score_by_season) != {fold.test_season for fold in FOLDS}:
        _refuse("report requires one score for every fixed fold")

    fold_reports: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for fold in fold_list:
        score = score_by_season[fold.spec.test_season]
        fold_predictions = _prediction_records(fold)
        if score.get("n") != len(fold_predictions):
            _refuse("fold score support must match held-out predictions")
        fold_report: dict[str, object] = {
            "test_season": fold.spec.test_season,
            "train_seasons": list(fold.spec.train_seasons),
            "short_history": fold.spec.short_history,
            "train_key_hash": key_sha256(fold.train_keys),
            "test_key_hash": key_sha256(fold.test_keys),
            "n_train": len(fold.train_keys),
            "n_test": len(fold.test_keys),
            "training_majority": str(fold.training_majority),
            "tier_edges": _tier_edges(fold.tier_edges),
            "model_parameters": _json_value(fold.model_parameters),
            "clipping": _json_value(fold.clipping),
        }
        for field in _FOLD_SCORE_FIELDS:
            if field not in score:
                _refuse(f"fold score is missing {field}")
            fold_report[field] = _json_value(score[field], path=f"$.folds.{field}")
        fold_reports.append(fold_report)
        predictions.extend(fold_predictions)
    prediction_keys = [
        (int(row["season"]), str(row["player_id"])) for row in predictions
    ]
    if prediction_keys != sorted(prediction_keys) or len(prediction_keys) != len(
        set(prediction_keys)
    ):
        _refuse("held-out predictions must be globally sorted and unique")

    contract = contract_payload(validated_mode)
    target = _target_payload(contract["target"])
    target["input_identities"] = _json_value(bundle.identities)
    report_contract = {
        "models": _json_value(contract["models"]),
        "folds": _json_value(contract["folds"]),
        "eligibility": {
            "positions": list(POSITIONS),
            "start_value": "strictly_positive",
            "end_value": "strictly_positive_for_primary_log_scoring",
            "start_trades_minimum": 5,
            "end_trades_minimum": 5,
            "zero_points_valid": True,
            "pre_nfl": "years_exp == 0 and games == 0",
        },
        "metrics": _json_value(contract["metrics"]),
        "bins": {
            "calibration_log_edges": _json_value(contract["metrics"]["calibration_edges"]),
            "liquidity": ["5-9", "10-24", "25+"],
            "value_tiers": "train_position_quantiles_linear_one_third_two_thirds",
        },
        "bootstrap": _json_value(contract["bootstrap"]),
        "ktc_mode": validated_mode,
    }
    pooled_value = _json_value(pooled, path="$.pooled")
    optional_value = _ordered_optional_ktc(optional_ktc)
    report: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "protocol": _PROTOCOL,
        "run_identity": run_identity,
        "run_sha256": run_sha256,
        "evidence": {
            "class": "historical_development",
            "historical_exposure": "architecture_and_data_choices_informed_by_prior_historical_experiments",
            "2025_status": "reused_development",
            "conditional_points": True,
            "historical_asof_validity": "unverified",
            "measurement_limits": "generated_backchannel_market_measurement_not_objective_exchange_value",
            "promotion_authorized": False,
            "independent_confirmation_complete": False,
        },
        "target": target,
        "contract": report_contract,
        "coverage": _json_value(data.coverage, path="$.coverage"),
        "folds": fold_reports,
        "pooled": pooled_value,
        "predictions": predictions,
        "optional_ktc": optional_value,
        "interpretation": {
            "evidence_flags": _evidence_flags(pooled_value, fold_reports),
            "caveats": [
                "Historical development evidence is reused, including 2025; it is not untouched confirmation.",
                "The product estimate is conditional on realized season points, not a validated preseason points forecast.",
                "This retrospective association is not a causal estimate of scoring more points.",
                "Backchannel is a generated market measurement, not objective exchange value or raw MFL trades.",
                "Historical as-of validity is unverified; hashes and revisions establish identity, not historical availability.",
                "Endpoint dates and full-season points lack evidence of aligned intraday availability.",
                "Primary scoring selects liquid observations at both endpoints and does not establish full-preseason coverage.",
                "Observed zero-value collapses are excluded from primary logarithmic scoring and reported separately.",
                "Full preseason population and coverage are not recoverable from the historical export.",
                "One-way bootstrap intervals omit joint player, team, season, and league uncertainty.",
                "Family-wide multiplicity is uncontrolled; slices are exploratory.",
                "Optional KTC is provenance-limited diagnostic evidence and cannot alter primary evidence flags.",
                "No score, flag, or interval authorizes promotion, shipping, replacement, or registration.",
            ],
        },
    }
    _validate_nullable_metrics(report)
    canonical_bytes(report)
    return report


def _validated_sha256(value: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _refuse("run SHA-256 must be exactly 64 lowercase hexadecimal characters")
    return value


def report_path(repo_root: Path, run_sha256: str) -> Path:
    """Return the sole permitted destination for a run identity."""
    digest = _validated_sha256(run_sha256)
    return (
        Path(repo_root)
        / "backend"
        / "predictor_v5"
        / "artifacts"
        / "market_movement"
        / f"development-2020-2025-{digest}.json"
    )


def _open_root_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        _refuse("repository root must be an existing non-symlink directory", exc)
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        _refuse("repository root must be a directory")
    return descriptor


def _open_child_directory(parent: int, name: str, label: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
    except OSError as exc:
        _refuse(f"{label} must be an existing non-symlink directory", exc)
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        _refuse(f"{label} must be a directory")
    return descriptor


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _validate_directory_entry(parent: int, name: str, child: int, label: str) -> None:
    try:
        linked = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except OSError as exc:
        _refuse(f"{label} changed during publication", exc)
    opened = os.fstat(child)
    if not stat.S_ISDIR(linked.st_mode) or not _same_inode(linked, opened):
        _refuse(f"{label} changed during publication")


def _validate_directory_chain(chain: _DirectoryChain) -> None:
    try:
        linked_root = chain.root_path.lstat()
    except OSError as exc:
        _refuse("repository root changed during publication", exc)
    if stat.S_ISLNK(linked_root.st_mode) or not _same_inode(
        linked_root, os.fstat(chain.root)
    ):
        _refuse("repository root changed during publication")
    _validate_directory_entry(chain.root, "backend", chain.backend, "backend directory")
    _validate_directory_entry(
        chain.backend, "predictor_v5", chain.package, "predictor_v5 directory"
    )
    _validate_directory_entry(
        chain.package, "artifacts", chain.artifacts, "artifacts directory"
    )
    _validate_directory_entry(
        chain.artifacts,
        "market_movement",
        chain.output,
        "market movement output directory",
    )


def _rmdir_owned(parent: int, name: str, identity: tuple[int, int]) -> bool:
    try:
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if (info.st_dev, info.st_ino) != identity or not stat.S_ISDIR(info.st_mode):
        return False
    try:
        os.rmdir(name, dir_fd=parent)
    except OSError:
        return False
    return True


@contextmanager
def _anchored_output_directory(repo_root: Path):
    root_path = Path(repo_root)
    descriptors: list[int] = []
    created = False
    parent_durable = False
    output_identity: tuple[int, int] | None = None
    yielded = False
    try:
        root = _open_root_directory(root_path)
        descriptors.append(root)
        backend = _open_child_directory(root, "backend", "backend")
        descriptors.append(backend)
        package = _open_child_directory(backend, "predictor_v5", "predictor_v5")
        descriptors.append(package)
        artifacts = _open_child_directory(package, "artifacts", "artifacts")
        descriptors.append(artifacts)
        try:
            os.mkdir("market_movement", 0o755, dir_fd=artifacts)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            _refuse("could not create market movement output directory", exc)
        output = _open_child_directory(
            artifacts, "market_movement", "market movement output"
        )
        descriptors.append(output)
        output_info = os.fstat(output)
        output_identity = (output_info.st_dev, output_info.st_ino)
        try:
            os.fsync(artifacts)
            if created:
                parent_durable = True
        except OSError:
            if created:
                _rmdir_owned(artifacts, "market_movement", output_identity)
                try:
                    os.fsync(artifacts)
                except OSError:
                    pass
            raise
        chain = _DirectoryChain(
            root_path=root_path,
            root=root,
            backend=backend,
            package=package,
            artifacts=artifacts,
            output=output,
            output_path=(
                root_path
                / "backend"
                / "predictor_v5"
                / "artifacts"
                / "market_movement"
            ),
        )
        _validate_directory_chain(chain)
        yielded = True
        yield chain
    except Exception:
        if (
            created
            and not yielded
            and output_identity is not None
            and parent_durable
            and _rmdir_owned(artifacts, "market_movement", output_identity)
        ):
            try:
                os.fsync(artifacts)
            except OSError:
                pass
        raise
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


@contextmanager
def _publication_guard(directory: int):
    info = os.fstat(directory)
    identity = (info.st_dev, info.st_ino)
    thread_id = threading.get_ident()
    with _LOCAL_PUBLICATION_STATES_GUARD:
        state = _LOCAL_PUBLICATION_STATES.setdefault(
            identity, _LocalPublicationState(lock=threading.Lock())
        )
        if state.owner == thread_id:
            _refuse("reentrant publication cannot accept an in-progress report")
    state.lock.acquire()
    locked = False
    try:
        with _LOCAL_PUBLICATION_STATES_GUARD:
            state.owner = thread_id
        fcntl.flock(directory, fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        if locked:
            fcntl.flock(directory, fcntl.LOCK_UN)
        with _LOCAL_PUBLICATION_STATES_GUARD:
            if state.owner == thread_id:
                state.owner = None
        state.lock.release()


def _write_all(descriptor: int, raw: bytes) -> None:
    remaining = memoryview(raw)
    while remaining:
        try:
            written = os.write(descriptor, remaining)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("report write made no progress")
        remaining = remaining[written:]


def _unlink_owned(directory: int, name: str, identity: tuple[int, int]) -> bool:
    try:
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if (info.st_dev, info.st_ino) != identity:
        return False
    os.unlink(name, dir_fd=directory)
    return True


def _read_regular(directory: int, name: str) -> bytes | None:
    try:
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        _refuse("existing report destination must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
    except OSError as exc:
        _refuse("could not open existing report destination safely", exc)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            _refuse("existing report destination changed during validation")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def publish_report(report: Mapping[str, object], repo_root: Path) -> Path:
    """Publish canonical report bytes atomically and without replacement."""
    if not isinstance(report, Mapping):
        _refuse("report must be a mapping")
    run_identity = report.get("run_identity")
    run_sha256 = report.get("run_sha256")
    if not isinstance(run_identity, Mapping) or not isinstance(run_sha256, str):
        _refuse("report is missing run identity metadata")
    digest = hashlib.sha256(canonical_bytes(run_identity)).hexdigest()
    if run_sha256 != digest:
        _refuse("report run SHA-256 does not match its canonical identity")
    raw = canonical_bytes(report)
    with _anchored_output_directory(Path(repo_root)) as chain:
        destination = report_path(Path(repo_root), run_sha256)
        if destination.parent != chain.output_path:
            _refuse("report destination escapes its output namespace")
        directory = chain.output
        with _publication_guard(directory):
            temporary = f".{run_sha256}.tmp.{secrets.token_hex(12)}"
            descriptor: int | None = None
            identity: tuple[int, int] | None = None
            temporary_created = False
            published = False
            try:
                existing = _read_regular(directory, destination.name)
                if existing is not None:
                    if existing == raw:
                        _validate_directory_chain(chain)
                        return destination
                    _refuse("same run identity already exists with different bytes")
                flags = (
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                try:
                    descriptor = os.open(temporary, flags, 0o600, dir_fd=directory)
                except FileExistsError as exc:
                    _refuse("exclusive temporary report name collision", exc)
                temporary_created = True
                info = os.fstat(descriptor)
                identity = (info.st_dev, info.st_ino)
                os.fchmod(descriptor, 0o600)
                _write_all(descriptor, raw)
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = None
                try:
                    os.link(
                        temporary,
                        destination.name,
                        src_dir_fd=directory,
                        dst_dir_fd=directory,
                        follow_symlinks=False,
                    )
                    published = True
                except FileExistsError:
                    existing = _read_regular(directory, destination.name)
                    if existing != raw:
                        _refuse("same run identity already exists with different bytes")
                if not _unlink_owned(directory, temporary, identity):
                    raise OSError("temporary report identity changed during publication")
                temporary_created = False
                os.fsync(directory)
                _validate_directory_chain(chain)
                return destination
            except Exception:
                if descriptor is not None:
                    os.close(descriptor)
                if identity is not None and published:
                    _unlink_owned(directory, destination.name, identity)
                if identity is not None and temporary_created:
                    _unlink_owned(directory, temporary, identity)
                try:
                    os.fsync(directory)
                except OSError:
                    pass
                raise
