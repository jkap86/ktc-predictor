"""Strict, sealed artifact validation for the 2026 model confirmation."""

import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Tuple

from predictor_v5 import label_export, model


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path("backend/predictor_v5/data/model_2026_confirmation_manifest.json")
REGISTRY_PATH = Path("backend/predictor_v5/data/model_2026_holdout_registry.json")

_SOURCE_PATHS = (
    "backend/predictor_v5/model.py",
    "backend/predictor_v5/sql/backchannel_labels.sql",
    "backend/predictor_v5/label_export.py",
)
_LABEL_COLUMNS = (
    "player_id", "season", "start_value", "end_value", "position", "age", "games", "points",
    "start_evidence", "end_evidence", "start_trades_window", "end_trades_window",
    "end_value_next_board", "end_evidence_next_board",
)


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


def _no_duplicate_keys(pairs: list) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key: %s" % key)
        value[key] = item
    return value


def load_canonical_json(path: Path, schema: Callable[[Dict[str, Any]], None]) -> Tuple[Dict[str, Any], str]:
    raw = Path(path).read_bytes()
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid JSON artifact: %s" % error) from error
    if not isinstance(value, dict):
        raise ValueError("artifact root must be an object")
    if raw != canonical_json_bytes(value):
        raise ValueError("artifact bytes are not canonical JSON")
    schema(value)
    return value, sha256_bytes(raw)


def _fields(value: Any, required: set, context: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("%s must be an object" % context)
    missing = required - set(value)
    unknown = set(value) - required
    if missing:
        raise ValueError("%s missing fields: %s" % (context, sorted(missing)))
    if unknown:
        raise ValueError("%s has unknown fields: %s" % (context, sorted(unknown)))
    return value


def _sha(value: Any, context: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("%s must be a lowercase SHA-256" % context)


def _identity(value: Any, context: str) -> None:
    entry = _fields(value, {"path", "sha256"}, context)
    validate_repo_relative_path(entry["path"])
    _sha(entry["sha256"], context + ".sha256")


def _drift(context: str, expected: Any, actual: Any) -> ValueError:
    return ValueError("%s drift: expected %r; actual %r" % (context, expected, actual))


def _first_difference(expected: Any, actual: Any, context: str) -> ValueError:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                return _drift(context + "." + key, expected.get(key), actual.get(key))
            if expected[key] != actual[key]:
                return _first_difference(expected[key], actual[key], context + "." + key)
    return _drift(context, expected, actual)


def _validate_live_label_identity(name: str, entry: Dict[str, Any]) -> None:
    path = validate_repo_relative_path(entry["path"])
    absolute = REPOSITORY_ROOT / path
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


def validate_manifest(manifest: Dict[str, Any]) -> None:
    required = {
        "schema_version", "protocol", "sealing_commit", "sealed_at", "baseline", "candidate", "training_data",
        "market_model_version", "train_seasons", "positions", "label_filter", "model", "source_identities",
        "target_recipe", "gate", "walk_forward", "bootstrap", "registry_path", "input_registration_path", "output_template",
    }
    _fields(manifest, required, "manifest")
    if manifest["schema_version"] != 1 or manifest["protocol"] != "model-2026-confirmation":
        raise ValueError("manifest protocol identity drift")
    if not isinstance(manifest["sealing_commit"], str) or not manifest["sealing_commit"]:
        raise ValueError("manifest sealing_commit must be nonempty")
    if not isinstance(manifest["sealed_at"], str) or not manifest["sealed_at"]:
        raise ValueError("manifest sealed_at must be nonempty")
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
        _validate_live_label_identity(name, entry)
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
        live_hash = file_sha256(REPOSITORY_ROOT / validate_repo_relative_path(source_path))
        if identity != live_hash:
            raise _drift(source_path + " sha256", live_hash, identity)
    recipe = _fields(manifest["target_recipe"], {"sql", "build", "output_columns", "provenance_contract"}, "target_recipe")
    _identity(recipe["sql"], "target_recipe.sql")
    _identity(recipe["build"], "target_recipe.build")
    expected_recipe_paths = {"sql": "backend/predictor_v5/sql/backchannel_labels.sql", "build": "backend/predictor_v5/label_export.py"}
    actual_recipe_paths = {name: recipe[name]["path"] for name in expected_recipe_paths}
    if actual_recipe_paths != expected_recipe_paths:
        raise _first_difference(expected_recipe_paths, actual_recipe_paths, "target recipe paths")
    for recipe_name in ("sql", "build"):
        identity = recipe[recipe_name]
        recipe_path = validate_repo_relative_path(identity["path"])
        live_hash = file_sha256(REPOSITORY_ROOT / recipe_path)
        if identity["sha256"] != live_hash:
            raise _drift(str(recipe_path) + " recipe sha256", live_hash, identity["sha256"])
    if recipe["output_columns"] != list(_LABEL_COLUMNS):
        raise ValueError("target recipe output column drift")
    if recipe["provenance_contract"] != {"helper": "label_identity", "model_version": "baseline-hl28-v1", "verified": True}:
        raise ValueError("label export provenance contract drift")
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


def validate_registry(registry: Dict[str, Any]) -> None:
    _fields(registry, {"schema_version", "attestation", "entries"}, "registry")
    if registry["schema_version"] != 1 or registry["attestation"] != "review and Git evidence; not proof of non-observation":
        raise ValueError("registry identity drift")
    if not isinstance(registry["entries"], list):
        raise ValueError("registry entries must be a list")
    keys = set()
    for entry in registry["entries"]:
        if not isinstance(entry, dict) or entry.get("status") not in {"sealed", "untouched", "reused"}:
            raise ValueError("invalid registry lifecycle entry")
        status = entry["status"]
        fields = {"season", "manifest_sha256", "status", "sealing_commit", "sealed_at"}
        if status in {"untouched", "reused"}:
            fields |= {"reviewer", "evidence_reference", "evidence_commit", "reviewed_at", "transition_source"}
        _fields(entry, fields, "registry entry")
        if not isinstance(entry["season"], int) or entry["season"] < 1:
            raise ValueError("registry season must be a positive integer")
        _sha(entry["manifest_sha256"], "registry manifest_sha256")
        key = (entry["season"], entry["manifest_sha256"])
        if key in keys:
            raise ValueError("duplicate registry key")
        keys.add(key)


def validate_input_registration(registration: Dict[str, Any]) -> None:
    """Validate the post-season artifact without reading its target files."""
    required = {
        "schema_version", "season", "manifest_sha256", "target", "provenance", "target_recipe",
        "holdout_attestation",
    }
    _fields(registration, required, "input registration")
    if registration["schema_version"] != 1 or registration["season"] != 2026:
        raise ValueError("input registration identity drift")
    _sha(registration["manifest_sha256"], "input registration manifest_sha256")
    target = _fields(registration["target"], {"csv", "sidecar", "label_columns", "label_filter"}, "target")
    _identity(target["csv"], "target.csv")
    _identity(target["sidecar"], "target.sidecar")
    if target["label_columns"] != list(_LABEL_COLUMNS):
        raise ValueError("target label columns drift")
    if target["label_filter"] != {"min_trades": 5, "positive_value_columns": ["start_value", "end_value"]}:
        raise ValueError("target label filter drift")
    if registration["provenance"] != {"model_version": "baseline-hl28-v1", "verified": True}:
        raise ValueError("target provenance drift")
    recipe = _fields(registration["target_recipe"], {"sql", "build"}, "input target_recipe")
    _identity(recipe["sql"], "input target_recipe.sql")
    _identity(recipe["build"], "input target_recipe.build")
    attestation = _fields(
        registration["holdout_attestation"],
        {"season", "manifest_sha256", "status", "reviewer", "evidence_commit", "reviewed_at"},
        "holdout_attestation",
    )
    if attestation["season"] != 2026 or attestation["manifest_sha256"] != registration["manifest_sha256"]:
        raise ValueError("holdout attestation key mismatch")
    if attestation["status"] != "untouched":
        raise ValueError("holdout attestation must be untouched")
