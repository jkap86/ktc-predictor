"""Strict, sealed artifact validation for the 2026 model confirmation."""

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Tuple

import pandas as pd

from predictor_v5 import label_export, model
from predictor_v5.data import read_season_stats


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
    output_path: Path
    train_seasons: tuple[int, ...]
    positions: tuple[str, ...]
    label_filter: dict
    target_recipe: dict
    target_loader: Callable[[Path, Path], pd.DataFrame]
    target_frame: pd.DataFrame | None = field(default=None, init=False)


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


def _validate_live_label_identity(name: str, entry: Dict[str, Any], repo_root: Path = REPOSITORY_ROOT) -> None:
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


def validate_manifest(manifest: Dict[str, Any], repo_root: Path = REPOSITORY_ROOT) -> None:
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
        live_hash = file_sha256(repo_root / recipe_path)
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
        "schema_version", "season", "manifest_sha256", "target_csv", "target_sidecar", "provenance",
        "target_contract", "recipe", "holdout_attestation",
    }
    _fields(registration, required, "input registration")
    if registration["schema_version"] != 1 or registration["season"] != 2026:
        raise ValueError("input registration identity drift")
    _sha(registration["manifest_sha256"], "input registration manifest_sha256")
    _identity(registration["target_csv"], "target_csv")
    _identity(registration["target_sidecar"], "target_sidecar")
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
    for name in ("reviewer", "evidence_commit", "evidence_reference", "reviewed_at", "transition_source"):
        if not isinstance(attestation[name], str) or not attestation[name]:
            raise ValueError("holdout attestation %s must be nonempty" % name)


def _repo_file(root: Path, value: str, context: str) -> Path:
    relative = validate_repo_relative_path(value)
    root = root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("%s is not a regular repository file" % context)
    return path


def _config_file(config: ConfirmationConfig, value: Path, context: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        if not path.is_file():
            raise ValueError("%s is not a regular file" % context)
        return path
    return _repo_file(config.repo_root, str(path), context)


def _default_target_loader(target_csv: Path, training_archive: Path) -> pd.DataFrame:
    values = pd.read_csv(target_csv, dtype={"player_id": str})
    label_export.validate_market_export(values, expected_seasons=(2026,), model_version="baseline-hl28-v1")
    stats = read_season_stats(training_archive)
    return label_export.build_label_frame(values, stats)


def _matching_entry(entries: list, season: int, manifest_sha: str, context: str) -> dict:
    found = [entry for entry in entries if entry["season"] == season and entry["manifest_sha256"] == manifest_sha]
    if len(found) != 1:
        raise ValueError("%s must contain exactly one matching entry" % context)
    return found[0]


def _validate_target_frame(frame: pd.DataFrame, context: PreflightContext) -> None:
    required = ["player_id", "season", "position", "age", "points", "start_value", "end_value", "start_trades_window", "end_trades_window"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError("joined target missing columns: %s" % missing)
    data = frame.copy()
    data["player_id"] = data["player_id"].astype(str)
    if data.duplicated(["player_id", "season"]).any():
        raise ValueError("joined target has duplicate player-season keys")
    if set(data["season"].astype(int)) - {2026}:
        raise ValueError("target contains training-season leakage")
    filtered = data[(data["start_value"] > 0) & (data["end_value"] > 0) & (data["start_trades_window"] >= 5) & (data["end_trades_window"] >= 5)]
    if filtered.empty:
        raise ValueError("target has no rows after the registered label filter")
    context.target_frame = filtered.sort_values(["season", "player_id"], kind="stable").reset_index(drop=True)


def _validate_training_coverage(path: Path, seasons: tuple[int, ...], name: str) -> None:
    data = pd.read_csv(path, usecols=["season"])
    missing = sorted(set(seasons) - set(data["season"].astype(int)))
    if missing:
        raise ValueError("%s is missing required training seasons %s" % (name, missing))


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
            raise ValueError("registry attestation %s mismatch" % key)
    target_csv = _repo_file(config.repo_root, registration["target_csv"]["path"], "target CSV")
    target_sidecar = _repo_file(config.repo_root, registration["target_sidecar"]["path"], "target sidecar")
    for path, entry, name in ((target_csv, registration["target_csv"], "target CSV"), (target_sidecar, registration["target_sidecar"], "target sidecar")):
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("%s hash mismatch" % name)
    load_canonical_json(target_sidecar, lambda value: _fields(value, set(value), "target sidecar"))
    recipe = registration["recipe"]
    for name, path_key, hash_key in (("sql", "sql_path", "sql_sha256"), ("build", "build_path", "build_sha256")):
        path = _repo_file(config.repo_root, recipe[path_key], "recipe " + name)
        if file_sha256(path) != recipe[hash_key] or recipe[hash_key] != manifest["target_recipe"][name]["sha256"]:
            raise ValueError("recipe %s hash mismatch" % name)
    raw = pd.read_csv(target_csv, dtype={"player_id": str})
    if list(raw.columns) != list(_LABEL_COLUMNS):
        raise ValueError("target CSV build-labels column contract drift")
    # Derived stat fields are deliberately not presented as market-export
    # columns to the validator; the loader re-derives and verifies their join.
    label_export.validate_market_export(raw.loc[:, _RAW_MARKET_COLUMNS], expected_seasons=(2026,))
    training_archive = _repo_file(config.repo_root, manifest["training_data"]["path"], "training archive")
    if file_sha256(training_archive) != manifest["training_data"]["sha256"]:
        raise ValueError("training archive hash mismatch")
    baseline_csv = _repo_file(config.repo_root, manifest["baseline"]["path"], "baseline CSV")
    candidate_csv = _repo_file(config.repo_root, manifest["candidate"]["path"], "candidate CSV")
    for path, entry, name in ((baseline_csv, manifest["baseline"], "baseline CSV"), (candidate_csv, manifest["candidate"], "candidate CSV")):
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("%s hash mismatch" % name)
        _validate_training_coverage(path, tuple(manifest["train_seasons"]), name)
    context = PreflightContext(manifest=manifest, manifest_sha256=manifest_sha, input_registration=registration,
        input_registration_sha256=registration_sha, holdout_registry=registry, holdout_registry_sha256=registry_sha,
        baseline_csv=baseline_csv, candidate_csv=candidate_csv,
        training_archive=training_archive, target_csv=target_csv, target_sidecar=target_sidecar, output_path=output_path,
        train_seasons=tuple(manifest["train_seasons"]), positions=tuple(manifest["positions"]),
        label_filter=registration["target_contract"]["label_filter"], target_recipe=registration["recipe"], target_loader=config.target_loader)
    _validate_target_frame(config.target_loader(target_csv, training_archive), context)
    return context


def run_confirmation(config: ConfirmationConfig) -> dict:
    """Reserve confirmation execution behind preflight; scoring is intentionally separate."""
    context = preflight(config)
    return {"status": "preflight-passed", "output_path": str(context.output_path)}
