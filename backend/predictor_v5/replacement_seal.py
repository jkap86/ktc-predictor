"""Deterministic, pre-outcome sealing of an independently reviewed HEAD.

``build_manifest`` returns an untimestamped template. Only ``write_seal``
adds the explicitly reviewed UTC timestamp and validates the final manifest.
No function invents a timestamp or registers/reads future outcome inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from predictor_v5.label_export import label_identity
from predictor_v5.replacement_contract import (
    canonical_bytes,
    validate_manifest,
    validate_registry,
)
from predictor_v5.replacement_ktc import RAW_COLUMNS, TARGET_COLUMNS, mapping_snapshot
from predictor_v5.replacement_mfl import TARGET_COLUMNS as MFL_COLUMNS
from predictor_v5.replacement_starts import (
    GIT_BLOB,
    INTRODUCTION,
    RELATIVE_PATH,
    SHA256,
)

PREFIX = "backend/predictor_v5/"
MANIFEST_PATH = PREFIX + "data/ktc_replacement_2026_manifest.json"
REGISTRY_PATH = PREFIX + "data/ktc_replacement_2026_holdout_registry.json"
ARCHIVE_PATH = "backend/data/training-data.zip"
MFL_PATH = PREFIX + "data/backchannel_labels_2019_2025_mfl.csv"
MAPPING_PATH = PREFIX + "data/ktc_replacement_mapping_snapshot.csv"
SOURCE_PATHS = tuple(
    sorted(
        PREFIX + name
        for name in (
            "__init__.py",
            "__main__.py",
            "confirmation.py",
            "data.py",
            "evaluate.py",
            "label_export.py",
            "labels.py",
            "model.py",
            "ktc_replacement_2026.py",
            "replacement_contract.py",
            "replacement_starts.py",
            "replacement_ktc.py",
            "replacement_mfl.py",
            "replacement_cohort.py",
            "replacement_metrics.py",
            "replacement_lifecycle.py",
            "replacement_report.py",
            "replacement_seal.py",
            "sql/backchannel_labels.sql",
            "sql/ktc_replacement_end.sql",
            "data/label_manifest.json",
            "data/backchannel_labels_2019_2025_mfl.provenance.json",
        )
    )
)


def _git(root: Path, *args: str) -> bytes:
    # Do not permit inherited Git routing or replacement objects to redefine HEAD.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        return subprocess.check_output(
            ["git", "--no-optional-locks", *args],
            cwd=root,
            env=environment,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot verify committed Git identity") from exc


def _check_anchor(root: Path, anchor: str) -> None:
    if not isinstance(anchor, str) or re.fullmatch(r"[0-9a-f]{40}", anchor) is None:
        raise ValueError("source anchor must be an exact commit SHA")
    if (
        Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
        != root
    ):
        raise ValueError("root must be the repository root")
    if (
        _git(root, "cat-file", "-t", anchor).strip() != b"commit"
        or _git(root, "rev-parse", "HEAD").decode().strip() != anchor
    ):
        raise ValueError("source anchor must be committed HEAD")
    if _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--ignore-submodules=none",
    ):
        raise ValueError("tracked worktree and index must be clean")
    # status may hide modified bytes under these index flags. Refuse the flags
    # rather than mutating the reviewer's index or trusting a hidden worktree.
    if any(
        entry[:1] != b"H"
        for entry in _git(root, "ls-files", "-v", "-z").split(b"\0")
        if entry
    ):
        raise ValueError("tracked files must not hide changes with index flags")


def _safe_path(root: Path, relative: str) -> Path:
    current = root
    for part in Path(relative).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("symlink in sealed path")
    return current


def _identity(root: Path, anchor: str, relative: str) -> tuple[dict, bytes]:
    path = _safe_path(root, relative)
    entry = _git(root, "ls-tree", anchor, "--", relative).decode().strip()
    if not entry or entry.split()[0] not in {"100644", "100755"}:
        raise ValueError(f"missing regular committed identity: {relative}")
    raw = _git(root, "show", anchor + ":" + relative)
    try:
        if not path.is_file() or path.read_bytes() != raw:
            raise ValueError(f"working bytes differ from anchor: {relative}")
    except OSError as exc:
        raise ValueError(f"cannot read identity: {relative}") from exc
    return {"path": relative, "sha256": hashlib.sha256(raw).hexdigest()}, raw


def _timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
        if (
            parsed.utcoffset() != UTC.utcoffset(None)
            or parsed.isoformat(timespec="seconds") != value
        ):
            raise ValueError("invalid seal timestamp")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "seal timestamp must be explicit canonical UTC (+00:00)"
        ) from exc


def build_manifest(root: Path, source_anchor: str) -> dict:
    """Build a deterministic template without ``sealed_at``; do not write files.

    The caller must supply that field before final manifest validation or
    registry construction. A clean exact committed HEAD is mandatory.
    """
    root = Path(root).resolve()
    _check_anchor(root, source_anchor)
    identities, contents = {}, {}
    for relative in (
        *SOURCE_PATHS,
        ARCHIVE_PATH,
        MFL_PATH,
        RELATIVE_PATH,
        MAPPING_PATH,
    ):
        identities[relative], contents[relative] = _identity(
            root, source_anchor, relative
        )
    if (
        identities[RELATIVE_PATH]["sha256"] != SHA256
        or _git(root, "rev-parse", source_anchor + ":" + RELATIVE_PATH).decode().strip()
        != GIT_BLOB
    ):
        raise ValueError("frozen start Git/content identity differs")
    mapping_snapshot(contents[MAPPING_PATH])
    try:
        with zipfile.ZipFile(io.BytesIO(contents[ARCHIVE_PATH])) as archive:
            if archive.namelist() != ["training-data.json"]:
                raise ValueError("training archive member differs")
            archive.read("training-data.json")  # Also verify ZIP member integrity.
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise ValueError("invalid training archive") from exc
    provenance = label_identity(
        root / MFL_PATH, root / PREFIX / "data/label_manifest.json"
    )
    if not provenance["verified"] or provenance["model_version"] != "baseline-hl28-v1":
        raise ValueError("MFL training labels must have verified v1 provenance")
    if provenance["sha256"] != identities[MFL_PATH]["sha256"]:
        raise ValueError("MFL training bytes changed during verification")
    # All inputs above are historical training, frozen starts, or recipe sources.
    manifest = {
        "schema_version": 1,
        "protocol": "ktc-replacement-2026",
        "sealing_commit": source_anchor,
        "registry_path": REGISTRY_PATH,
        "input_registration_path": PREFIX + "data/ktc_replacement_2026_inputs.json",
        "output_template": PREFIX
        + "artifacts/replacements/2026-<manifest-sha256>.json",
        "claim_template": PREFIX
        + "artifacts/replacements/2026-<manifest-sha256>.claim.json",
        "source_identities": [identities[path] for path in SOURCE_PATHS],
        "training": {
            "archive": identities[ARCHIVE_PATH],
            "ktc_member": "training-data.json",
            "mfl_labels": identities[MFL_PATH],
            "seasons": [2021, 2022, 2023, 2024, 2025],
            "positions": ["QB", "RB", "TE", "WR"],
        },
        "starts": {
            "identity": identities[RELATIVE_PATH],
            "git_blob": GIT_BLOB,
            "introduced_commit": INTRODUCTION,
            "cutoff": "2026-09-09",
            "provenance_limit": "raw start provenance not independently verified",
        },
        "model": {
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
        "cutoffs": {
            "start": "2026-09-09",
            "end": "2027-01-10",
            "timezone": "America/New_York",
            "ktc_end_window_start": "2026-12-28",
            "ktc_start_lookback_days": 14,
        },
        "filters": {
            "ktc": {"minimum_exclusive": 0, "maximum_exclusive": 9999},
            "mfl": {
                "minimum_exclusive": 0,
                "minimum_start_trades": 5,
                "minimum_end_trades": 5,
            },
        },
        "recipes": {
            "ktc": {
                "sql": identities[PREFIX + "sql/ktc_replacement_end.sql"],
                "extractor": identities[PREFIX + "replacement_ktc.py"],
                "mapping_code": identities[PREFIX + "replacement_ktc.py"],
                "mapping_data": identities[MAPPING_PATH],
                "raw_columns": list(RAW_COLUMNS),
                "target_columns": list(TARGET_COLUMNS),
                "stats_format": "training-players-zip",
            },
            "mfl": {
                "sql": identities[PREFIX + "sql/backchannel_labels.sql"],
                "build": identities[PREFIX + "label_export.py"],
                "target_columns": list(MFL_COLUMNS),
                "stats_format": "training-players-zip",
                "model_version": "baseline-hl28-v1",
                "verified": True,
            },
        },
        "gate": {
            "minimum_overall_delta": -0.01,
            "minimum_position_delta": -0.05,
            "minimum_position_count": 30,
        },
        "diagnostics": {
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
    }
    _check_anchor(root, source_anchor)
    return manifest


def build_registry(manifest: dict, sealed_at: str) -> dict:
    """Bind one sealed entry to a complete, already timestamped manifest."""
    _timestamp(sealed_at)
    validate_manifest(manifest)
    if manifest["sealed_at"] != sealed_at:
        raise ValueError("registry timestamp must match final manifest")
    registry = {
        "schema_version": 1,
        "attestation": "review and Git evidence; not proof of non-observation",
        "entries": [
            {
                "season": 2026,
                "manifest_sha256": hashlib.sha256(
                    canonical_bytes(manifest)
                ).hexdigest(),
                "status": "sealed",
                "sealing_commit": manifest["sealing_commit"],
                "sealed_at": sealed_at,
            }
        ],
    }
    validate_registry(registry)
    return registry


def _unlink_owned(path: Path, identity: tuple[int, int]) -> None:
    try:
        info = path.lstat()
        if (info.st_dev, info.st_ino) == identity:
            path.unlink()
    except FileNotFoundError:
        pass


def write_seal(root: Path, source_anchor: str, sealed_at: str) -> None:
    """Publish only the canonical manifest and registry, without replacement.

    Validate both before publication. A caught publication failure rolls back
    only files created by this call. A process crash may leave a partial seal;
    preserve it for review, never silently replace or resume it.
    """
    root = Path(root).resolve()
    _timestamp(sealed_at)
    destinations = [_safe_path(root, path) for path in (MANIFEST_PATH, REGISTRY_PATH)]
    if any(os.path.lexists(path) for path in destinations):
        raise ValueError("seal output already exists")
    manifest = build_manifest(root, source_anchor) | {"sealed_at": sealed_at}
    registry = build_registry(manifest, sealed_at)
    parent = destinations[0].parent
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary, published = [], []
    try:
        for payload in (manifest, registry):
            fd, name = tempfile.mkstemp(prefix=".replacement-seal-", dir=parent)
            path = Path(name)
            info = os.fstat(fd)
            identity = (info.st_dev, info.st_ino)
            temporary.append((path, identity))
            with os.fdopen(fd, "wb") as stream:
                stream.write(canonical_bytes(payload))
                stream.flush()
                os.fsync(stream.fileno())
        # Recheck anchor and bytes immediately before creating either output.
        if build_manifest(root, source_anchor) | {"sealed_at": sealed_at} != manifest:
            raise ValueError("seal inputs changed")
        if any(os.path.lexists(path) for path in destinations):
            raise ValueError("seal output already exists")
        for (path, identity), destination in zip(temporary, destinations):
            try:
                os.link(path, destination, follow_symlinks=False)
            except FileExistsError as exc:
                raise ValueError("seal output already exists") from exc
            published.append((destination, identity))
        for path, identity in temporary:
            _unlink_owned(path, identity)
        os.fsync(directory)
    except BaseException:
        for path, identity in published:
            _unlink_owned(path, identity)
        raise
    finally:
        for path, identity in temporary:
            _unlink_owned(path, identity)
        os.close(directory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-anchor", required=True)
    parser.add_argument("--sealed-at", required=True)
    args = parser.parse_args()
    try:
        write_seal(
            Path(__file__).resolve().parents[2], args.source_anchor, args.sealed_at
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
