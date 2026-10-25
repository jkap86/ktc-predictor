"""Metadata-only preflight and irreversible claim for replacement evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import secrets
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from predictor_v5 import confirmation
from predictor_v5.replacement_cohort import prepare_training
from predictor_v5.replacement_contract import (
    canonical_bytes,
    validate_joint_attestation,
    validate_manifest,
    validate_references,
    validate_registration,
    validate_registry,
)
from predictor_v5.replacement_starts import load_starts

REPLACEMENT_MANIFEST_PATH = "backend/predictor_v5/data/ktc_replacement_2026_manifest.json"
CONFIRMATION_MANIFEST_PATH = "backend/predictor_v5/data/model_2026_confirmation_manifest.json"
CONFIRMATION_MANIFEST_SHA256 = (
    "c376651de489d0ae5c6478c001d565bb635d6968b46df4a16e99875d9da7430e"
)
LABEL_MANIFEST_PATH = "backend/predictor_v5/data/label_manifest.json"

_MANIFEST_FIELDS = {
    "schema_version", "protocol", "sealing_commit", "sealed_at", "registry_path",
    "input_registration_path", "output_template", "claim_template",
    "source_identities", "training", "starts", "model", "cutoffs", "filters",
    "recipes", "gate", "diagnostics",
}
_REGISTRY_FIELDS = {"schema_version", "attestation", "entries"}
_REGISTRATION_FIELDS = {
    "schema_version", "season", "manifest_sha256", "ktc", "mfl",
    "holdout_attestation",
}
_JOINT_FIELDS = {
    "schema_version", "season", "replacement_manifest_sha256",
    "confirmation_manifest_sha256", "reviewer", "evidence_commit",
    "evidence_reference", "reviewed_at", "recipes_committed_before_outcomes",
    "evidence_untouched",
}
_CONFIRMATION_MANIFEST_FIELDS = {
    "schema_version", "protocol", "sealing_commit", "sealed_at", "baseline",
    "candidate", "training_data", "market_model_version", "train_seasons",
    "positions", "label_filter", "model", "source_identities", "target_recipe",
    "gate", "walk_forward", "bootstrap", "registry_path",
    "input_registration_path", "output_template",
}
_CONFIRMATION_REGISTRATION_FIELDS = {
    "schema_version", "season", "manifest_sha256", "target_csv",
    "target_sidecar", "target_stats", "provenance", "target_contract", "recipe",
    "holdout_attestation",
}
_CONTEXT_FIELDS = {
    "manifest", "manifest_sha256", "registry", "registry_sha256", "registration",
    "registration_sha256", "joint_review", "joint_review_sha256",
    "confirmation_manifest", "confirmation_manifest_sha256",
    "confirmation_registry", "confirmation_registry_sha256",
    "confirmation_registration", "confirmation_registration_sha256",
    "training_frames", "starts", "output_path", "claim_path",
}
_CAPABILITY = object()
_DIR_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)


@dataclass
class _Captured:
    path: str
    raw: bytes
    device: int
    inode: int
    size: int
    descriptor: int | None = None

    @property
    def identity(self) -> tuple[int, int]:
        return self.device, self.inode

    def close(self) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None


class _PreflightContext(dict):
    """Dict-compatible capability backed by pinned repository/output dirfds."""

    def __init__(
        self, values: dict, *, token: object, root_fd: int, parent_fd: int,
        root_path: Path, output_name: str, claim_name: str,
    ) -> None:
        super().__init__(values)
        self._token = token
        self._root_fd = root_fd
        self._parent_fd = parent_fd
        self._root_path = root_path
        self._parent_relative = "/".join(
            Path(values["output_path"]).relative_to(root_path).parts[:-1]
        )
        self._output_name = output_name
        self._claim_name = claim_name
        self._parent_identity = _inode(os.fstat(parent_fd))
        self._objects_digest = _objects_digest(values)
        self._training_digest = _frames_digest(values["training_frames"])
        self._starts_digest = _frame_digest(values["starts"])
        self._output_path = values["output_path"]
        self._claim_path = values["claim_path"]
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for name in ("_parent_fd", "_root_fd"):
            descriptor = getattr(self, name, -1)
            setattr(self, name, -1)
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def __enter__(self):
        if self.closed:
            raise ValueError("preflight capability is closed")
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _relative(value: str) -> tuple[str, ...]:
    if (
        not isinstance(value, str) or not value or value.startswith("/")
        or "\\" in value or "\x00" in value
    ):
        raise ValueError("unsafe repository path")
    parts = tuple(value.split("/"))
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("unsafe repository path")
    return parts


def _absolute_root(root: Path) -> tuple[Path, int]:
    path = Path(root)
    if not path.is_absolute():
        path = Path.cwd() / path
    if any(part == ".." for part in path.parts):
        raise ValueError("unsafe repository root")
    current = os.open("/", _DIR_FLAGS)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, _DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = next_fd
        if not stat.S_ISDIR(os.fstat(current).st_mode):
            raise ValueError("repository root is not a directory")
        return path, current
    except Exception:
        os.close(current)
        raise


def _open_parent(root_fd: int, value: str) -> tuple[int, str]:
    parts = _relative(value)
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, _DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = next_fd
        return current, parts[-1]
    except Exception:
        os.close(current)
        raise


def _open_directory(root_fd: int, value: str) -> int:
    parts = _relative(value)
    current = os.dup(root_fd)
    try:
        for part in parts:
            next_fd = os.open(part, _DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = next_fd
        return current
    except Exception:
        os.close(current)
        raise


def _inode(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _stat_relative(root_fd: int, value: str, *, required: bool = True):
    parent_fd, name = _open_parent(root_fd, value)
    try:
        try:
            result = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            if required:
                raise ValueError("repository path does not exist") from None
            return None
        if stat.S_ISLNK(result.st_mode) or not stat.S_ISREG(result.st_mode):
            raise ValueError("repository path is not a regular no-follow file")
        return result
    finally:
        os.close(parent_fd)


def _read_descriptor(descriptor: int) -> bytes:
    chunks = []
    while True:
        try:
            chunk = os.read(descriptor, 1024 * 1024)
        except InterruptedError:
            continue
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _capture(
    root_fd: int, value: str, outcome_inodes=frozenset(), *, keep_open=False,
) -> _Captured:
    parent_fd, name = _open_parent(root_fd, value)
    try:
        descriptor = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("repository path is not a regular file")
        if before.st_nlink != 1:
            raise ValueError("repository input has a physical link alias")
        if _inode(before) in outcome_inodes:
            raise ValueError("registered outcome aliases a readable input")
        raw = _read_descriptor(descriptor)
        after = os.fstat(descriptor)
        if (
            _inode(before) != _inode(after)
            or before.st_size != after.st_size
            or before.st_nlink != 1
            or after.st_nlink != 1
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise ValueError("repository file changed while captured")
        captured = _Captured(
            value, raw, before.st_dev, before.st_ino, before.st_size,
            descriptor if keep_open else None,
        )
        if keep_open:
            descriptor = -1
        return captured
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _recheck_capture(captured: _Captured, outcome_inodes: set[tuple[int, int]]) -> None:
    if captured.descriptor is None:
        raise ValueError("metadata descriptor is unavailable")
    info = os.fstat(captured.descriptor)
    if (
        _inode(info) != captured.identity
        or info.st_size != captured.size
        or info.st_nlink != 1
        or captured.identity in outcome_inodes
    ):
        raise ValueError("metadata link or identity changed during discovery")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _decode(
    raw: bytes, required: set[str], canonicalizer=canonical_bytes,
) -> tuple[dict, str]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid canonical JSON") from exc
    if (
        not isinstance(value, dict) or set(value) != required
        or canonicalizer(value) != raw
    ):
        raise ValueError("invalid canonical JSON")
    return value, hashlib.sha256(raw).hexdigest()


def _verify_identity(captured: _Captured, identity: dict) -> None:
    digest = hashlib.sha256(captured.raw).hexdigest()
    if captured.path != identity["path"] or digest != identity["sha256"]:
        raise ValueError("sealed pre-outcome identity mismatch")


def _matching_entry(registry: dict, digest: str, label: str) -> dict:
    entries = [
        entry for entry in registry["entries"]
        if entry["season"] == 2026 and entry["manifest_sha256"] == digest
    ]
    if len(entries) != 1 or entries[0]["status"] != "untouched":
        raise ValueError(f"{label} has no unique untouched entry")
    return entries[0]


def _absent_at(parent_fd: int, name: str, label: str) -> None:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"cannot inspect {label}") from exc
    raise ValueError(f"{label} already exists")


def _outcome_identities(registration: dict) -> list[dict]:
    return [
        registration[source][key]
        for source, keys in {
            "ktc": ("raw_end", "target_csv", "target_stats", "target_sidecar"),
            "mfl": ("target_csv", "target_stats", "target_sidecar"),
        }.items()
        for key in keys
    ]


def _confirmation_outcomes(registration: dict) -> list[dict]:
    return [registration[key] for key in ("target_csv", "target_sidecar", "target_stats")]


def _sidecar_path(value: str) -> str:
    path = Path(value)
    return (path.parent / (path.stem + ".provenance.json")).as_posix()


def _git_directory(root_fd: int, root_path: Path) -> tuple[Path, int]:
    info = os.stat(".git", dir_fd=root_fd, follow_symlinks=False)
    if stat.S_ISDIR(info.st_mode):
        descriptor = os.open(".git", _DIR_FLAGS, dir_fd=root_fd)
        return root_path / ".git", descriptor
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("unsafe Git administration path")
    descriptor = os.open(".git", _FILE_FLAGS, dir_fd=root_fd)
    try:
        before = os.fstat(descriptor)
        raw = _read_descriptor(descriptor)
        after = os.fstat(descriptor)
        if (
            _inode(before) != _inode(after)
            or before.st_size != after.st_size
            or before.st_nlink != 1
            or after.st_nlink != 1
        ):
            raise ValueError("Git administration file changed")
    finally:
        os.close(descriptor)
    try:
        text = raw.decode("utf-8").strip()
    except UnicodeError as exc:
        raise ValueError("invalid Git administration file") from exc
    if not text.startswith("gitdir: "):
        raise ValueError("invalid Git administration file")
    value = Path(text.removeprefix("gitdir: "))
    if not value.is_absolute():
        value = root_path / value
    value = Path(os.path.abspath(value))
    git_path, descriptor = _absolute_root(value)
    return git_path, descriptor


def _validate_outcomes(
    root_fd: int, expected: dict[str, tuple[tuple[int, int], int]],
) -> None:
    for path, (identity, size) in expected.items():
        info = _stat_relative(root_fd, path)
        if _inode(info) != identity or info.st_size != size or info.st_nlink != 1:
            raise ValueError("registered outcome identity changed during preflight")


def _claim_exists_beneath(root_fd: int, basename: str) -> bool:
    pending = [os.dup(root_fd)]
    seen = set()
    try:
        while pending:
            descriptor = pending.pop()
            identity = _inode(os.fstat(descriptor))
            if identity in seen:
                os.close(descriptor)
                continue
            seen.add(identity)
            try:
                for name in os.listdir(descriptor):
                    info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if name == basename:
                        return True
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(os.open(name, _DIR_FLAGS, dir_fd=descriptor))
            finally:
                os.close(descriptor)
        return False
    finally:
        for descriptor in pending:
            os.close(descriptor)


def _materialize(root: Path, captures: dict[str, _Captured]) -> None:
    for value, captured in captures.items():
        destination = root / value
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(captured.raw)


def _frame_digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pickle.dumps(frame, protocol=5)).hexdigest()


def _frames_digest(frames: dict) -> tuple[tuple[str, str], ...]:
    if not isinstance(frames, dict):
        raise ValueError("invalid training frames")  # noqa: TRY004 - protocol refusal
    return tuple((key, _frame_digest(frames[key])) for key in sorted(frames))


def _objects_digest(values: dict) -> str:
    keys = (
        "manifest", "manifest_sha256", "registry", "registry_sha256",
        "registration", "registration_sha256", "joint_review",
        "joint_review_sha256", "confirmation_manifest",
        "confirmation_manifest_sha256", "confirmation_registry",
        "confirmation_registry_sha256", "confirmation_registration",
        "confirmation_registration_sha256",
    )
    return hashlib.sha256(
        canonical_bytes({key: values[key] for key in keys})
    ).hexdigest()


def _snapshot_validation(
    captures: dict[str, _Captured], manifest: dict, registry: dict,
    registration: dict, joint: dict, manifest_sha: str, review_identity: dict,
    confirmation_manifest: dict, git_directory: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    with tempfile.TemporaryDirectory(prefix="ktc-replacement-preflight-") as directory:
        snapshot = Path(directory)
        _materialize(snapshot, captures)
        (snapshot / ".git").write_text(f"gitdir: {git_directory}\n")
        validate_references(
            manifest, registry, registration, joint,
            manifest_sha256=manifest_sha,
            attestation_identity=review_identity,
            root=snapshot,
        )
        original_identity = confirmation.label_export.label_identity

        def snapshot_identity(path, manifest_path=None):
            return original_identity(path, manifest_path=snapshot / LABEL_MANIFEST_PATH)

        confirmation.label_export.label_identity = snapshot_identity
        try:
            confirmation.validate_manifest(confirmation_manifest, snapshot)
        finally:
            confirmation.label_export.label_identity = original_identity
        training = prepare_training(snapshot, manifest)
        starts_path = snapshot / manifest["starts"]["identity"]["path"]
        starts = load_starts(starts_path, snapshot)
        return training, starts


def _review_confirmation(
    manifest: dict, manifest_sha: str, registry: dict, registration: dict,
    joint: dict,
) -> None:
    entry = _matching_entry(registry, manifest_sha, "confirmation registry")
    if any(entry[key] != manifest[key] for key in ("sealing_commit", "sealed_at")):
        raise ValueError("confirmation registry seal mismatch")
    review = registration["holdout_attestation"]
    if registration["manifest_sha256"] != manifest_sha:
        raise ValueError("confirmation registration manifest mismatch")
    for key in (
        "status", "reviewer", "evidence_commit", "evidence_reference",
        "reviewed_at", "transition_source",
    ):
        if review[key] != entry[key]:
            raise ValueError("confirmation review metadata mismatch")
    for key in ("reviewer", "evidence_commit", "evidence_reference", "reviewed_at"):
        if review[key] != joint[key]:
            raise ValueError("joint confirmation review metadata mismatch")


def preflight(root: Path) -> dict:
    """Validate every pre-claim invariant without opening registered outcomes."""
    root_path, root_fd = _absolute_root(root)
    parent_fd = None
    git_fd = None
    held_metadata = []
    try:
        manifest_capture = _capture(
            root_fd, REPLACEMENT_MANIFEST_PATH, keep_open=True
        )
        held_metadata.append(manifest_capture)
        manifest, manifest_sha = _decode(manifest_capture.raw, _MANIFEST_FIELDS)
        validate_manifest(manifest)
        output_value = manifest["output_template"].replace(
            "<manifest-sha256>", manifest_sha
        )
        claim_value = manifest["claim_template"].replace(
            "<manifest-sha256>", manifest_sha
        )
        output_parts, claim_parts = _relative(output_value), _relative(claim_value)
        if output_parts[:-1] != claim_parts[:-1]:
            raise ValueError("claim and output parents differ")
        if _claim_exists_beneath(root_fd, claim_parts[-1]):
            raise ValueError("replacement claim already exists beneath repository root")
        parent_fd = _open_directory(root_fd, "/".join(output_parts[:-1]))
        _absent_at(parent_fd, output_parts[-1], "replacement report")
        _absent_at(parent_fd, claim_parts[-1], "replacement claim")

        registry_capture = _capture(
            root_fd, manifest["registry_path"], keep_open=True
        )
        held_metadata.append(registry_capture)
        registry, registry_sha = _decode(registry_capture.raw, _REGISTRY_FIELDS)
        validate_registry(registry)
        registration_capture = _capture(
            root_fd, manifest["input_registration_path"], keep_open=True
        )
        held_metadata.append(registration_capture)
        registration, registration_sha = _decode(
            registration_capture.raw, _REGISTRATION_FIELDS
        )
        validate_registration(registration)
        review_identity = registration["holdout_attestation"]["joint_review"]
        joint_capture = _capture(root_fd, review_identity["path"], keep_open=True)
        held_metadata.append(joint_capture)
        joint, joint_sha = _decode(joint_capture.raw, _JOINT_FIELDS)
        validate_joint_attestation(joint)
        if joint_sha != review_identity["sha256"]:
            raise ValueError("joint review hash mismatch")

        confirmation_manifest_capture = _capture(
            root_fd, CONFIRMATION_MANIFEST_PATH, keep_open=True
        )
        held_metadata.append(confirmation_manifest_capture)
        confirmation_manifest, confirmation_manifest_sha = _decode(
            confirmation_manifest_capture.raw, _CONFIRMATION_MANIFEST_FIELDS,
            confirmation.canonical_json_bytes,
        )
        if confirmation_manifest_sha != CONFIRMATION_MANIFEST_SHA256:
            raise ValueError("protected confirmation manifest mismatch")
        if confirmation_manifest_sha != joint["confirmation_manifest_sha256"]:
            raise ValueError("joint review confirmation manifest mismatch")
        confirmation_registry_capture = _capture(
            root_fd, confirmation_manifest["registry_path"], keep_open=True
        )
        held_metadata.append(confirmation_registry_capture)
        confirmation_registry, confirmation_registry_sha = _decode(
            confirmation_registry_capture.raw, _REGISTRY_FIELDS,
            confirmation.canonical_json_bytes,
        )
        confirmation.validate_registry(confirmation_registry)
        confirmation_registration_capture = _capture(
            root_fd, confirmation_manifest["input_registration_path"], keep_open=True
        )
        held_metadata.append(confirmation_registration_capture)
        confirmation_registration, confirmation_registration_sha = _decode(
            confirmation_registration_capture.raw,
            _CONFIRMATION_REGISTRATION_FIELDS,
            confirmation.canonical_json_bytes,
        )
        confirmation.validate_input_registration(confirmation_registration)
        _review_confirmation(
            confirmation_manifest, confirmation_manifest_sha,
            confirmation_registry, confirmation_registration, joint,
        )

        confirmation_output = confirmation_manifest["output_template"].replace(
            "<manifest-sha256>", confirmation_manifest_sha
        )
        confirmation_output_parts = _relative(confirmation_output)
        confirmation_parent = _open_directory(
            root_fd, "/".join(confirmation_output_parts[:-1])
        )
        try:
            _absent_at(
                confirmation_parent, confirmation_output_parts[-1],
                "confirmation output",
            )
        finally:
            os.close(confirmation_parent)

        outcome_entries = _outcome_identities(registration) + _confirmation_outcomes(
            confirmation_registration
        )
        outcome_paths = {identity["path"] for identity in outcome_entries}
        if len(outcome_paths) != len(outcome_entries):
            raise ValueError("registered outcome aliases another outcome path")
        outcome_stats = [
            _stat_relative(root_fd, identity["path"]) for identity in outcome_entries
        ]
        if any(info.st_nlink != 1 for info in outcome_stats):
            raise ValueError("registered outcome has a physical alias")
        outcome_inodes = {_inode(info) for info in outcome_stats}
        outcome_expected = {
            identity["path"]: (_inode(info), info.st_size)
            for identity, info in zip(outcome_entries, outcome_stats, strict=True)
        }
        metadata_captures = {
            item.path: item
            for item in (
                manifest_capture, registry_capture, registration_capture,
                joint_capture, confirmation_manifest_capture,
                confirmation_registry_capture, confirmation_registration_capture,
            )
        }
        for captured in metadata_captures.values():
            _recheck_capture(captured, outcome_inodes)
        _validate_outcomes(root_fd, outcome_expected)
        for captured in held_metadata:
            captured.close()

        replacement_identities = [
            *manifest["source_identities"], manifest["training"]["archive"],
            manifest["training"]["mfl_labels"], manifest["starts"]["identity"],
            manifest["recipes"]["ktc"]["mapping_data"],
        ]
        confirmation_paths = {
            confirmation_manifest["baseline"]["path"],
            confirmation_manifest["candidate"]["path"],
            confirmation_manifest["training_data"]["path"],
            confirmation_manifest["target_recipe"]["sql"]["path"],
            confirmation_manifest["target_recipe"]["build"]["path"],
            *confirmation_manifest["source_identities"],
        }
        readable_paths = {
            *(identity["path"] for identity in replacement_identities),
            *confirmation_paths, LABEL_MANIFEST_PATH,
        }
        for label_path in (
            confirmation_manifest["baseline"]["path"],
            confirmation_manifest["candidate"]["path"],
        ):
            sidecar = _sidecar_path(label_path)
            if _stat_relative(root_fd, sidecar, required=False) is not None:
                readable_paths.add(sidecar)
        if outcome_paths & readable_paths:
            raise ValueError("registered outcome aliases a readable input path")

        captures = dict(metadata_captures)
        for value in sorted(readable_paths):
            captures[value] = _capture(root_fd, value, outcome_inodes)
            _validate_outcomes(root_fd, outcome_expected)
        for identity in replacement_identities:
            _verify_identity(captures[identity["path"]], identity)
        _verify_identity(
            captures[confirmation_manifest["training_data"]["path"]],
            confirmation_manifest["training_data"],
        )
        git_directory, git_fd = _git_directory(root_fd, root_path)
        training_frames, starts = _snapshot_validation(
            captures, manifest, registry, registration, joint, manifest_sha,
            review_identity, confirmation_manifest, git_directory,
        )
        if not stat.S_ISDIR(os.fstat(git_fd).st_mode):
            raise ValueError("Git administration directory changed")
        _validate_outcomes(root_fd, outcome_expected)
        values = {
            "manifest": manifest, "manifest_sha256": manifest_sha,
            "registry": registry, "registry_sha256": registry_sha,
            "registration": registration,
            "registration_sha256": registration_sha,
            "joint_review": joint, "joint_review_sha256": joint_sha,
            "confirmation_manifest": confirmation_manifest,
            "confirmation_manifest_sha256": confirmation_manifest_sha,
            "confirmation_registry": confirmation_registry,
            "confirmation_registry_sha256": confirmation_registry_sha,
            "confirmation_registration": confirmation_registration,
            "confirmation_registration_sha256": confirmation_registration_sha,
            "training_frames": training_frames, "starts": starts,
            "output_path": root_path / output_value,
            "claim_path": root_path / claim_value,
        }
        context = _PreflightContext(
            values, token=_CAPABILITY, root_fd=root_fd, parent_fd=parent_fd,
            root_path=root_path, output_name=output_parts[-1],
            claim_name=claim_parts[-1],
        )
        root_fd = -1
        parent_fd = None
        return context
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("invalid lifecycle metadata") from exc
    finally:
        for captured in held_metadata:
            if captured.descriptor is not None:
                captured.close()
        if git_fd is not None:
            os.close(git_fd)
        if parent_fd is not None:
            os.close(parent_fd)
        if root_fd >= 0:
            os.close(root_fd)


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


def _validate_context(context: dict) -> _PreflightContext:
    if (
        type(context) is not _PreflightContext
        or context._token is not _CAPABILITY
        or context.closed
        or set(context) != _CONTEXT_FIELDS
    ):
        raise ValueError("invalid preflight capability")
    if _inode(os.fstat(context._parent_fd)) != context._parent_identity:
        raise ValueError("preflight parent capability changed")
    current_parent = _open_directory(context._root_fd, context._parent_relative)
    try:
        if _inode(os.fstat(current_parent)) != context._parent_identity:
            raise ValueError("preflight parent path changed")
    finally:
        os.close(current_parent)
    if _objects_digest(context) != context._objects_digest:
        raise ValueError("preflight artifact context changed")
    if _frames_digest(context["training_frames"]) != context._training_digest:
        raise ValueError("preflight training frames changed")
    if _frame_digest(context["starts"]) != context._starts_digest:
        raise ValueError("preflight starts changed")
    if (
        context["output_path"] != context._output_path
        or context["claim_path"] != context._claim_path
    ):
        raise ValueError("preflight paths changed")
    return context


def _claim_payload(context: _PreflightContext) -> dict:
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
        "schema_version": 1, "season": 2026,
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


def _canonical_claim_matches(bound: _PreflightContext, identity) -> None:
    current_parent = _open_directory(bound._root_fd, bound._parent_relative)
    try:
        if _inode(os.fstat(current_parent)) != bound._parent_identity:
            raise ValueError("canonical claim parent changed after creation")
        info = os.stat(
            bound._claim_name, dir_fd=current_parent, follow_symlinks=False
        )
        if not stat.S_ISREG(info.st_mode) or _inode(info) != identity:
            raise ValueError("canonical claim identity changed after creation")
    finally:
        os.close(current_parent)


def _create_claim(bound: _PreflightContext) -> Path:
    payload = _claim_payload(bound)
    _absent_at(bound._parent_fd, bound._output_name, "replacement report")
    _absent_at(bound._parent_fd, bound._claim_name, "replacement claim")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            bound._claim_name, flags, 0o600, dir_fd=bound._parent_fd
        )
    except FileExistsError as exc:
        raise ValueError("replacement claim already exists") from exc
    try:
        created_identity = _inode(os.fstat(descriptor))
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, canonical_bytes(payload))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(bound._parent_fd)
    _canonical_claim_matches(bound, created_identity)
    return bound["claim_path"]


def claim(context: dict) -> Path:
    """Irrevocably and durably claim this holdout before outcome-byte access."""
    try:
        bound = _validate_context(context)
        return _create_claim(bound)
    finally:
        if type(context) is _PreflightContext:
            context.close()


def _validated_report_bytes(value) -> bytes:
    if type(value) is not bytes:
        raise ValueError("renderer must return immutable canonical report bytes")
    try:
        decoded = json.loads(value.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("renderer must return immutable canonical report bytes") from exc
    if not isinstance(decoded, dict) or canonical_bytes(decoded) != value:
        raise ValueError("renderer must return immutable canonical report bytes")
    return value


def _cleanup_report_publication(
    bound: _PreflightContext, temp_name: str | None, published: bool,
) -> None:
    error = None
    names = ([bound._output_name] if published else []) + (
        [temp_name] if temp_name is not None else []
    )
    for name in names:
        try:
            os.unlink(name, dir_fd=bound._parent_fd)
        except FileNotFoundError:
            pass
        except OSError as exc:
            error = error or exc
    try:
        os.fsync(bound._parent_fd)
    except OSError as exc:
        error = error or exc
    if error is not None:
        raise error


def _publish_report(bound: _PreflightContext, raw: bytes) -> Path:
    temp_name = f".{bound._output_name}.tmp.{secrets.token_hex(12)}"
    published = False
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temp_name, flags, 0o600, dir_fd=bound._parent_fd)
        try:
            os.fchmod(descriptor, 0o600)
            _write_all(descriptor, raw)
            report_identity = _inode(os.fstat(descriptor))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temp_name, bound._output_name,
                src_dir_fd=bound._parent_fd, dst_dir_fd=bound._parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise ValueError("replacement report already exists") from exc
        published = True
        os.unlink(temp_name, dir_fd=bound._parent_fd)
        temp_name = None
        os.fsync(bound._parent_fd)
        current_parent = _open_directory(bound._root_fd, bound._parent_relative)
        try:
            if _inode(os.fstat(current_parent)) != bound._parent_identity:
                raise ValueError("canonical report parent changed during publication")
            info = os.stat(
                bound._output_name, dir_fd=current_parent, follow_symlinks=False
            )
            if not stat.S_ISREG(info.st_mode) or _inode(info) != report_identity:
                raise ValueError("canonical report identity changed during publication")
        finally:
            os.close(current_parent)
        return bound["output_path"]
    except Exception:
        _cleanup_report_publication(bound, temp_name, published)
        raise


def run_claimed_stages(
    context: dict, *, validate_outcomes, fit, write_report,
):
    """Claim first, then render canonical bytes and publish them internally.

    ``write_report`` is a renderer: it receives only the fitted result and must
    return exact immutable canonical JSON bytes. It never receives the lifecycle
    context, an output path, or publication authority.
    """
    try:
        bound = _validate_context(context)
        _create_claim(bound)
        outcomes = validate_outcomes(bound)
        fitted = fit(bound, outcomes)
        _absent_at(bound._parent_fd, bound._output_name, "replacement report")
        raw = _validated_report_bytes(write_report(fitted))
        return _publish_report(bound, raw)
    finally:
        if type(context) is _PreflightContext:
            context.close()
