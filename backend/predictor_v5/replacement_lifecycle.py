"""Metadata-only preflight and irreversible claim for replacement evaluation."""

from __future__ import annotations

import hashlib
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

from predictor_v5 import confirmation
from predictor_v5.replacement_cohort import prepare_training
from predictor_v5.replacement_contract import (
    canonical_bytes,
    load_canonical,
    validate_references,
)
from predictor_v5.replacement_starts import load_starts

REPLACEMENT_MANIFEST_PATH = Path(
    "backend/predictor_v5/data/ktc_replacement_2026_manifest.json"
)
CONFIRMATION_MANIFEST_PATH = Path(
    "backend/predictor_v5/data/model_2026_confirmation_manifest.json"
)
CONFIRMATION_MANIFEST_SHA256 = (
    "c376651de489d0ae5c6478c001d565bb635d6968b46df4a16e99875d9da7430e"
)

_MANIFEST_FIELDS = {
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
_REGISTRY_FIELDS = {"schema_version", "attestation", "entries"}
_REGISTRATION_FIELDS = {
    "schema_version",
    "season",
    "manifest_sha256",
    "ktc",
    "mfl",
    "holdout_attestation",
}
_JOINT_FIELDS = {
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
}
_CONTEXT_FIELDS = {
    "manifest",
    "manifest_sha256",
    "registry",
    "registry_sha256",
    "registration",
    "registration_sha256",
    "joint_review",
    "joint_review_sha256",
    "confirmation_manifest",
    "confirmation_manifest_sha256",
    "confirmation_registry",
    "confirmation_registry_sha256",
    "confirmation_registration",
    "confirmation_registration_sha256",
    "training_frames",
    "starts",
    "output_path",
    "claim_path",
}


def _root(root: Path) -> Path:
    value = Path(root)
    try:
        if value.is_symlink():
            raise ValueError("repository root must not be a symlink")
        resolved = value.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("invalid repository root") from exc
    if not resolved.is_dir():
        raise ValueError("repository root must be a directory")
    return resolved


def _relative(value: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("unsafe repository path")
    return Path(*value.split("/"))


def _beneath(root: Path, value: str, *, allow_missing_leaf: bool = False) -> Path:
    relative = _relative(value)
    current = root
    parts = relative.parts
    for index, part in enumerate(parts):
        current = current / part
        final = index == len(parts) - 1
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if final and allow_missing_leaf:
                return current
            raise ValueError("repository path does not exist") from None
        except OSError as exc:
            raise ValueError("cannot inspect repository path") from exc
        if stat.S_ISLNK(mode):
            raise ValueError("repository path contains a symlink")
        if not final and not stat.S_ISDIR(mode):
            raise ValueError("repository path parent is not a directory")
    return current


def _metadata_file(root: Path, identity: dict) -> Path:
    path = _beneath(root, identity["path"])
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError("cannot stat registered outcome") from exc
    if not stat.S_ISREG(mode):
        raise ValueError("registered outcome is not a regular file")
    return path


def _artifact_file(root: Path, value: str) -> Path:
    path = _beneath(root, value)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("protocol artifact is not a regular file")
    return path


def _identity_file(root: Path, identity: dict) -> Path:
    path = _artifact_file(root, identity["path"])
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError("cannot read sealed pre-outcome input") from exc
    if digest.hexdigest() != identity["sha256"]:
        raise ValueError("sealed pre-outcome identity mismatch")
    return path


def _absent(path: Path, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"cannot inspect {label}") from exc
    raise ValueError(f"{label} already exists")


def _matching_entry(registry: dict, digest: str, label: str) -> dict:
    entries = [
        entry
        for entry in registry["entries"]
        if entry["season"] == 2026 and entry["manifest_sha256"] == digest
    ]
    if len(entries) != 1 or entries[0]["status"] != "untouched":
        raise ValueError(f"{label} has no unique untouched entry")
    return entries[0]


def _load_confirmation(root: Path, joint: dict, joint_sha: str) -> dict:
    manifest_path = _artifact_file(root, str(CONFIRMATION_MANIFEST_PATH))
    manifest, manifest_sha = confirmation.load_canonical_json(
        manifest_path, lambda _value: None
    )
    if manifest_sha != CONFIRMATION_MANIFEST_SHA256:
        raise ValueError("protected confirmation manifest mismatch")
    if manifest_sha != joint["confirmation_manifest_sha256"]:
        raise ValueError("joint review confirmation manifest mismatch")

    registry_path = _artifact_file(root, manifest["registry_path"])
    registry, registry_sha = confirmation.load_canonical_json(
        registry_path, confirmation.validate_registry
    )
    entry = _matching_entry(registry, manifest_sha, "confirmation registry")
    if any(entry[key] != manifest[key] for key in ("sealing_commit", "sealed_at")):
        raise ValueError("confirmation registry seal mismatch")

    registration_path = _artifact_file(root, manifest["input_registration_path"])
    registration, registration_sha = confirmation.load_canonical_json(
        registration_path, confirmation.validate_input_registration
    )
    if registration["manifest_sha256"] != manifest_sha:
        raise ValueError("confirmation registration manifest mismatch")
    review = registration["holdout_attestation"]
    if review["registry_key"] != "2026:" + manifest_sha:
        raise ValueError("confirmation registration key mismatch")
    for key in (
        "status",
        "reviewer",
        "evidence_commit",
        "evidence_reference",
        "reviewed_at",
        "transition_source",
    ):
        if review[key] != entry[key]:
            raise ValueError("confirmation review metadata mismatch")
    for key in ("reviewer", "evidence_commit", "evidence_reference", "reviewed_at"):
        if review[key] != joint[key]:
            raise ValueError("joint confirmation review metadata mismatch")
    if review["evidence_reference"] != joint["evidence_reference"]:
        raise ValueError("confirmation joint review path mismatch")

    output_value = manifest["output_template"].replace(
        "<manifest-sha256>", manifest_sha
    )
    output_path = _beneath(root, output_value, allow_missing_leaf=True)
    _absent(output_path, "confirmation output")

    outcome_paths = {
        registration[key]["path"]
        for key in ("target_csv", "target_sidecar", "target_stats")
    }
    preoutcome_paths = {
        manifest["baseline"]["path"],
        manifest["candidate"]["path"],
        manifest["training_data"]["path"],
        manifest["target_recipe"]["sql"]["path"],
        manifest["target_recipe"]["build"]["path"],
        *manifest["source_identities"],
    }
    if outcome_paths & preoutcome_paths:
        raise ValueError("confirmation outcome aliases a pre-outcome input")
    confirmation.validate_manifest(manifest, root)
    for key in ("target_csv", "target_sidecar", "target_stats"):
        _metadata_file(root, registration[key])
    _identity_file(root, manifest["training_data"])
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "registry": registry,
        "registry_sha256": registry_sha,
        "registration": registration,
        "registration_sha256": registration_sha,
        "joint_review_sha256": joint_sha,
    }


def preflight(root: Path) -> dict:
    """Validate every pre-claim invariant without opening registered outcomes."""
    try:
        return _preflight(root)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("invalid lifecycle metadata") from exc


def _preflight(root: Path) -> dict:
    root = _root(root)
    manifest_path = _artifact_file(root, str(REPLACEMENT_MANIFEST_PATH))
    manifest, manifest_sha = load_canonical(manifest_path, _MANIFEST_FIELDS)

    output_value = manifest["output_template"].replace(
        "<manifest-sha256>", manifest_sha
    )
    claim_value = manifest["claim_template"].replace(
        "<manifest-sha256>", manifest_sha
    )
    output_path = _beneath(root, output_value, allow_missing_leaf=True)
    claim_path = _beneath(root, claim_value, allow_missing_leaf=True)
    if output_path.parent != claim_path.parent:
        raise ValueError("claim and output parents differ")
    _absent(output_path, "replacement report")
    _absent(claim_path, "replacement claim")

    registry_path = _artifact_file(root, manifest["registry_path"])
    registry, registry_sha = load_canonical(registry_path, _REGISTRY_FIELDS)
    registration_path = _artifact_file(root, manifest["input_registration_path"])
    registration, registration_sha = load_canonical(
        registration_path, _REGISTRATION_FIELDS
    )

    review_identity = registration["holdout_attestation"].get("joint_review")
    if not isinstance(review_identity, dict) or set(review_identity) != {"path", "sha256"}:
        raise ValueError("missing joint review identity")
    joint_path = _artifact_file(root, review_identity["path"])
    joint, joint_sha = load_canonical(joint_path, _JOINT_FIELDS)
    if joint_sha != review_identity["sha256"]:
        raise ValueError("joint review hash mismatch")
    validate_references(
        manifest,
        registry,
        registration,
        joint,
        manifest_sha256=manifest_sha,
        attestation_identity=review_identity,
        root=root,
    )

    confirmation_context = _load_confirmation(root, joint, joint_sha)

    outcome_keys = {
        "ktc": ("raw_end", "target_csv", "target_stats", "target_sidecar"),
        "mfl": ("target_csv", "target_stats", "target_sidecar"),
    }
    for source, keys in outcome_keys.items():
        for key in keys:
            _metadata_file(root, registration[source][key])

    identities = list(manifest["source_identities"])
    identities.extend(
        [
            manifest["training"]["archive"],
            manifest["training"]["mfl_labels"],
            manifest["starts"]["identity"],
            manifest["recipes"]["ktc"]["mapping_data"],
        ]
    )
    registered_outcome_paths = {
        registration[source][key]["path"]
        for source, keys in outcome_keys.items()
        for key in keys
    }
    if registered_outcome_paths & {identity["path"] for identity in identities}:
        raise ValueError("replacement outcome aliases a pre-outcome input")
    checked = set()
    for identity in identities:
        pair = (identity["path"], identity["sha256"])
        if pair not in checked:
            _identity_file(root, identity)
            checked.add(pair)

    training_frames = prepare_training(root, manifest)
    starts = load_starts(root / manifest["starts"]["identity"]["path"], root)

    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "registry": registry,
        "registry_sha256": registry_sha,
        "registration": registration,
        "registration_sha256": registration_sha,
        "joint_review": joint,
        "joint_review_sha256": joint_sha,
        "confirmation_manifest": confirmation_context["manifest"],
        "confirmation_manifest_sha256": confirmation_context["manifest_sha256"],
        "confirmation_registry": confirmation_context["registry"],
        "confirmation_registry_sha256": confirmation_context["registry_sha256"],
        "confirmation_registration": confirmation_context["registration"],
        "confirmation_registration_sha256": confirmation_context[
            "registration_sha256"
        ],
        "training_frames": training_frames,
        "starts": starts,
        "output_path": output_path,
        "claim_path": claim_path,
    }


def _claim_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _validated_claim_timestamp() -> str:
    value = _claim_timestamp()
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid claim timestamp") from exc
    if parsed.utcoffset() is None or parsed.isoformat(timespec="seconds") != value:
        raise ValueError("invalid claim timestamp")
    return value


def _validate_context(context: dict) -> None:
    if not isinstance(context, dict) or set(context) != _CONTEXT_FIELDS:
        raise ValueError("invalid preflight context")
    objects = (
        ("manifest", "manifest_sha256"),
        ("registry", "registry_sha256"),
        ("registration", "registration_sha256"),
        ("joint_review", "joint_review_sha256"),
        ("confirmation_manifest", "confirmation_manifest_sha256"),
        ("confirmation_registry", "confirmation_registry_sha256"),
        ("confirmation_registration", "confirmation_registration_sha256"),
    )
    for object_key, digest_key in objects:
        value = context[object_key]
        digest = context[digest_key]
        if (
            not isinstance(value, dict)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or hashlib.sha256(canonical_bytes(value)).hexdigest() != digest
        ):
            raise ValueError("preflight context hash mismatch")
    if (
        not isinstance(context["training_frames"], dict)
        or set(context["training_frames"]) != {"ktc", "mfl"}
        or not hasattr(context["starts"], "columns")
    ):
        raise ValueError("invalid preflight data context")
    manifest = context["manifest"]
    if manifest.get("protocol") != "ktc-replacement-2026":
        raise ValueError("invalid preflight manifest")
    expected_output = manifest.get("output_template", "").replace(
        "<manifest-sha256>", context["manifest_sha256"]
    )
    expected_claim = manifest.get("claim_template", "").replace(
        "<manifest-sha256>", context["manifest_sha256"]
    )
    for key, expected in (("output_path", expected_output), ("claim_path", expected_claim)):
        path = context[key]
        expected_parts = Path(expected).parts
        if (
            not isinstance(path, Path)
            or not path.is_absolute()
            or not expected
            or path.parts[-len(expected_parts) :] != expected_parts
        ):
            raise ValueError("preflight context path mismatch")


def _validate_absolute_parents(path: Path) -> None:
    for parent in path.parents:
        if parent == parent.parent:
            continue
        try:
            mode = parent.lstat().st_mode
        except OSError as exc:
            raise ValueError("claim parent is unavailable") from exc
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError("claim parent is unsafe")


def _claim_payload(context: dict) -> dict:
    _validate_context(context)
    return {
        "claimed_at": _validated_claim_timestamp(),
        "confirmation_manifest_sha256": context["confirmation_manifest_sha256"],
        "confirmation_registration_sha256": context[
            "confirmation_registration_sha256"
        ],
        "confirmation_registry_sha256": context["confirmation_registry_sha256"],
        "joint_review_sha256": context["joint_review_sha256"],
        "manifest_sha256": context["manifest_sha256"],
        "protocol": "ktc-replacement-2026",
        "registration_sha256": context["registration_sha256"],
        "registry_sha256": context["registry_sha256"],
        "schema_version": 1,
        "season": 2026,
    }


def _write_all(descriptor: int, raw: bytes) -> None:
    remaining = memoryview(raw)
    while remaining:
        try:
            written = os.write(descriptor, remaining)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("claim write made no progress")
        remaining = remaining[written:]


def _sync_parent(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def claim(context: dict) -> Path:
    """Irrevocably and durably claim this holdout before outcome-byte access."""
    payload = _claim_payload(context)
    claim_path = Path(context["claim_path"])
    output_path = Path(context["output_path"])
    if not claim_path.is_absolute() or not output_path.is_absolute():
        raise ValueError("claim paths must come from preflight")
    if claim_path.parent != output_path.parent:
        raise ValueError("claim and output parents differ")
    _validate_absolute_parents(claim_path)
    try:
        parent_mode = claim_path.parent.lstat().st_mode
    except OSError as exc:
        raise ValueError("claim parent is unavailable") from exc
    if stat.S_ISLNK(parent_mode) or not stat.S_ISDIR(parent_mode):
        raise ValueError("claim parent is unsafe")
    _absent(output_path, "replacement report")
    _absent(claim_path, "replacement claim")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(claim_path, flags, 0o600)
    except FileExistsError as exc:
        raise ValueError("replacement claim already exists") from exc
    try:
        _write_all(descriptor, canonical_bytes(payload))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _sync_parent(claim_path.parent)
    return claim_path
