"""Canonical, closed contracts for the replacement evaluation protocol."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


def canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("invalid JSON") from exc


def load_canonical(path, required: set[str]) -> tuple[dict, str]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
        if (
            not isinstance(value, dict)
            or set(value) != required
            or canonical_bytes(value) != raw
        ):
            raise ValueError("noncanonical object")
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON") from exc
    return value, hashlib.sha256(raw).hexdigest()


def _fields(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("unexpected fields")


def _path(value):
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("invalid path")


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid digest")


def validate_identity(value: dict) -> None:
    _fields(value, {"path", "sha256"})
    _path(value["path"])
    _hash(value["sha256"])


def _commit(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("invalid commit")


def _string(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("blank string")


def _timestamp(value):
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() is None or parsed.isoformat(timespec="seconds") != value:
            raise ValueError("invalid timestamp")
        return parsed
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid timestamp") from exc


def _fixed(value, expected):
    if isinstance(expected, dict):
        _fields(value, expected)
        for key, item in expected.items():
            _fixed(value[key], item)
    elif isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise ValueError("invalid list")
        for actual, item in zip(value, expected):
            _fixed(actual, item)
    elif type(value) is not type(expected) or value != expected:
        raise ValueError("invalid fixed value")


def validate_manifest(value: dict) -> None:
    keys = {
        "schema_version",
        "protocol",
        "sealing_commit",
        "sealed_at",
        "registry_path",
        "input_registration_path",
        "output_template",
        "claim_template",
        "source_identities",
        "training",
        "starts",
        "model",
        "cutoffs",
        "filters",
        "recipes",
        "gate",
        "diagnostics",
    }
    _fields(value, keys)
    _fixed(value["schema_version"], 1)
    _fixed(value["protocol"], "ktc-replacement-2026")
    _commit(value["sealing_commit"])
    _timestamp(value["sealed_at"])
    for key in ("registry_path", "input_registration_path"):
        _path(value[key])
    for key, suffix in (
        ("output_template", ".json"),
        ("claim_template", ".claim.json"),
    ):
        _fixed(
            value[key],
            "backend/predictor_v5/artifacts/replacements/2026-<manifest-sha256>"
            + suffix,
        )
    sources = value["source_identities"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("empty sources")
    for identity in sources:
        validate_identity(identity)
    paths = [item["path"] for item in sources]
    if paths != sorted(set(paths)):
        raise ValueError("source paths not unique and sorted")
    training = value["training"]
    _fields(training, {"archive", "ktc_member", "mfl_labels", "seasons", "positions"})
    for key in ("archive", "mfl_labels"):
        validate_identity(training[key])
    _path(training["ktc_member"])
    _fixed(training["seasons"], [2021, 2022, 2023, 2024, 2025])
    _fixed(training["positions"], ["QB", "RB", "TE", "WR"])
    _fixed(
        value["starts"],
        {
            "identity": {
                "path": "backend/predictor_v5/data/start_values_2026.csv",
                "sha256": "6366b85300c4f9532f0866c602c6d262baaccf6c3a84b9252a6d2e8d46c541bb",
            },
            "git_blob": "ead24b508102ca2a1071d07f3f665e9ea920aa96",
            "introduced_commit": "c415177",
            "cutoff": "2026-09-09",
            "provenance_limit": "raw start provenance not independently verified",
        },
    )
    _fixed(
        value["model"],
        {
            "features": ["log_start", "age", "points", "position_code"],
            "monotonic": [0, 0, 1, 0],
            "points_grid_step": 1,
            "hgb_params": {
                "loss": "absolute_error",
                "learning_rate": 0.05,
                "max_iter": 300,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 20,
                "l2_regularization": 1,
                "random_state": 0,
            },
            "ktc_scale_max": 9999,
            "mfl_scale_max": 10000,
            "mfl_model_version": "baseline-hl28-v1",
        },
    )
    _fixed(
        value["cutoffs"],
        {
            "start": "2026-09-09",
            "end": "2027-01-10",
            "timezone": "America/New_York",
            "ktc_end_window_start": "2026-12-28",
            "ktc_start_lookback_days": 14,
        },
    )
    _fixed(
        value["filters"],
        {
            "ktc": {"minimum_exclusive": 0, "maximum_exclusive": 9999},
            "mfl": {
                "minimum_exclusive": 0,
                "minimum_start_trades": 5,
                "minimum_end_trades": 5,
            },
        },
    )
    _fixed(
        value["gate"],
        {
            "minimum_overall_delta": -0.01,
            "minimum_position_delta": -0.05,
            "minimum_position_count": 30,
        },
    )
    _fixed(
        value["diagnostics"],
        {
            "bootstrap": {
                "seed": 20261024,
                "resamples": 10000,
                "percentiles": [2.5, 97.5],
            },
            "rank": {
                "order": "descending",
                "ties": "average",
                "correlation": "spearman",
                "error": "mean_absolute_rank_error",
            },
            "used_for_configuration": False,
            "used_for_promotion": False,
        },
    )
    recipes = value["recipes"]
    _fields(recipes, {"ktc", "mfl"})
    base_columns = [
        "player_id",
        "season",
        "start_value",
        "end_value",
        "position",
        "age",
        "games",
        "points",
    ]
    for name, identity_keys, extra in (
        (
            "ktc",
            ("sql", "extractor", "mapping_code", "mapping_data"),
            {
                "raw_columns": ["format", "ktc_id", "sleeper_id", "date", "sf_value"],
                "target_columns": base_columns,
                "stats_format": "training-players-zip",
            },
        ),
        (
            "mfl",
            ("sql", "build"),
            {
                "target_columns": base_columns
                + [
                    "start_evidence",
                    "end_evidence",
                    "start_trades_window",
                    "end_trades_window",
                    "end_value_next_board",
                    "end_evidence_next_board",
                ],
                "stats_format": "training-players-zip",
                "model_version": "baseline-hl28-v1",
                "verified": True,
            },
        ),
    ):
        recipe = recipes[name]
        _fields(recipe, set(identity_keys) | set(extra))
        for key in identity_keys:
            validate_identity(recipe[key])
            if key != "mapping_data" and recipe[key] not in sources:
                raise ValueError("recipe source missing")
        for key, expected in extra.items():
            _fixed(recipe[key], expected)


def validate_joint_attestation(value: dict) -> None:
    _fields(
        value,
        {
            "schema_version",
            "season",
            "replacement_manifest_sha256",
            "confirmation_manifest_sha256",
            "reviewer",
            "evidence_commit",
            "evidence_reference",
            "reviewed_at",
            "recipes_committed_before_outcomes",
            "evidence_untouched",
        },
    )
    _fixed(value["schema_version"], 1)
    _fixed(value["season"], 2026)
    for key in ("replacement_manifest_sha256", "confirmation_manifest_sha256"):
        _hash(value[key])
    _string(value["reviewer"])
    _commit(value["evidence_commit"])
    _path(value["evidence_reference"])
    _timestamp(value["reviewed_at"])
    _fixed(value["recipes_committed_before_outcomes"], True)
    _fixed(value["evidence_untouched"], True)


def _review(value, transition):
    _string(value["reviewer"])
    _commit(value["evidence_commit"])
    _path(value["evidence_reference"])
    _timestamp(value["reviewed_at"])
    _fixed(value["transition_source"], transition)
    validate_identity(value["joint_review"])


def validate_registry(value: dict) -> None:
    _fields(value, {"schema_version", "attestation", "entries"})
    _fixed(value["schema_version"], 1)
    _fixed(
        value["attestation"], "review and Git evidence; not proof of non-observation"
    )
    if not isinstance(value["entries"], list):
        raise ValueError("entries must be a list")  # noqa: TRY004 - public contract
    seen = set()
    for entry in value["entries"]:
        if not isinstance(entry, dict) or entry.get("status") not in (
            "sealed",
            "untouched",
            "reused",
            "used",
        ):
            raise ValueError("invalid state")
        state = entry["status"]
        fields = {"season", "manifest_sha256", "status", "sealing_commit", "sealed_at"}
        if state != "sealed":
            fields |= {
                "reviewer",
                "evidence_commit",
                "evidence_reference",
                "reviewed_at",
                "transition_source",
                "joint_review",
            }
        if state == "used":
            fields |= {"claim", "report"}
        _fields(entry, fields)
        _fixed(entry["season"], 2026)
        _hash(entry["manifest_sha256"])
        _commit(entry["sealing_commit"])
        sealed = _timestamp(entry["sealed_at"])
        key = (entry["season"], entry["manifest_sha256"])
        if key in seen:
            raise ValueError("duplicate registry key")
        seen.add(key)
        if state != "sealed":
            _review(entry, "untouched" if state == "used" else "sealed")
            if _timestamp(entry["reviewed_at"]) < sealed:
                raise ValueError("review predates seal")
        if state == "used":
            validate_identity(entry["claim"])
            if entry["report"] is not None:
                validate_identity(entry["report"])


def validate_registration(value: dict) -> None:
    _fields(
        value,
        {
            "schema_version",
            "season",
            "manifest_sha256",
            "ktc",
            "mfl",
            "holdout_attestation",
        },
    )
    _fixed(value["schema_version"], 1)
    _fixed(value["season"], 2026)
    _hash(value["manifest_sha256"])
    for name, keys in (
        (
            "ktc",
            {
                "raw_end",
                "target_csv",
                "target_stats",
                "target_sidecar",
                "sql",
                "extractor",
                "mapping_code",
                "mapping_data",
            },
        ),
        ("mfl", {"target_csv", "target_stats", "target_sidecar", "sql", "build"}),
    ):
        _fields(value[name], keys)
        for identity in value[name].values():
            validate_identity(identity)
    review = value["holdout_attestation"]
    _fields(
        review,
        {
            "registry_key",
            "status",
            "reviewer",
            "evidence_commit",
            "evidence_reference",
            "reviewed_at",
            "transition_source",
            "joint_review",
        },
    )
    _fixed(review["registry_key"], "2026:" + value["manifest_sha256"])
    _fixed(review["status"], "untouched")
    _review(review, "sealed")


def validate_references(
    manifest: dict,
    registry: dict,
    registration: dict,
    attestation: dict,
    *,
    manifest_sha256: str,
    attestation_identity: dict,
    root: Path,
) -> dict:
    validate_manifest(manifest)
    validate_registry(registry)
    validate_registration(registration)
    validate_joint_attestation(attestation)
    validate_identity(attestation_identity)
    _hash(manifest_sha256)
    if hashlib.sha256(canonical_bytes(manifest)).hexdigest() != manifest_sha256:
        raise ValueError("manifest digest mismatch")
    if (
        hashlib.sha256(canonical_bytes(attestation)).hexdigest()
        != attestation_identity["sha256"]
    ):
        raise ValueError("review digest mismatch")
    entries = [
        e
        for e in registry["entries"]
        if e["season"] == 2026 and e["manifest_sha256"] == manifest_sha256
    ]
    if len(entries) != 1 or entries[0]["status"] != "untouched":
        raise ValueError("no eligible registry entry")
    entry = entries[0]
    review = registration["holdout_attestation"]
    if (
        registration["manifest_sha256"] != manifest_sha256
        or attestation["replacement_manifest_sha256"] != manifest_sha256
        or entry["sealing_commit"] != manifest["sealing_commit"]
        or _timestamp(entry["sealed_at"]) != _timestamp(manifest["sealed_at"])
        or _timestamp(attestation["reviewed_at"]) < _timestamp(manifest["sealed_at"])
    ):
        raise ValueError("manifest reference mismatch")
    if (
        entry["joint_review"] != attestation_identity
        or review["joint_review"] != attestation_identity
    ):
        raise ValueError("review identity mismatch")
    for key in ("reviewer", "evidence_commit", "evidence_reference", "reviewed_at"):
        values = (entry[key], review[key], attestation[key])
        if key == "reviewed_at":
            values = tuple(_timestamp(item) for item in values)
        if not values[0] == values[1] == values[2]:
            raise ValueError("review metadata mismatch")
    if attestation_identity["path"] != attestation["evidence_reference"]:
        raise ValueError("review path mismatch")
    for name, keys in (
        ("ktc", ("sql", "extractor", "mapping_code", "mapping_data")),
        ("mfl", ("sql", "build")),
    ):
        for key in keys:
            if registration[name][key] != manifest["recipes"][name][key]:
                raise ValueError("recipe mismatch")
    root = root.resolve()

    def paths(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in (
                    "path",
                    "registry_path",
                    "input_registration_path",
                    "evidence_reference",
                    "ktc_member",
                ):
                    _path(item)
                    if not (root / item).resolve().is_relative_to(root):
                        raise ValueError("path escapes root")
                else:
                    paths(item)
        elif isinstance(value, list):
            for item in value:
                paths(item)

    for value in (manifest, registry, registration, attestation, attestation_identity):
        paths(value)
    return entry
