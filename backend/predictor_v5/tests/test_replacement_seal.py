"""Sealing tests use disposable committed repositories, never real outcomes."""

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from predictor_v5.replacement_contract import (
    canonical_bytes,
    validate_manifest,
    validate_registry,
)

ROOT = Path(__file__).resolve().parents[3]
BASELINE = "1157678526d3408850780c7ec7808a730d8c2dbc"
PREFIX = "backend/predictor_v5/"
MANIFEST = PREFIX + "data/ktc_replacement_2026_manifest.json"
REGISTRY = PREFIX + "data/ktc_replacement_2026_holdout_registry.json"
SEALED_AT = "2026-10-25T12:00:00+00:00"
SOURCES = [
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
]


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.PIPE)


@pytest.fixture
def committed(tmp_path):
    for relative in SOURCES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# synthetic reviewed source\n")
    labels = tmp_path / PREFIX / "data/backchannel_labels_2019_2025_mfl.csv"
    labels.write_bytes(b"player_id,season\nfixture,2025\n")
    labels.with_suffix(".provenance.json").write_bytes(
        canonical_bytes(
            {
                "label_file": {
                    "sha256": hashlib.sha256(labels.read_bytes()).hexdigest()
                },
                "model_version": "baseline-hl28-v1",
                "model_version_source": "export",
            }
        )
    )
    (tmp_path / PREFIX / "data/label_manifest.json").write_bytes(b'{"files":{}}\n')
    for name in ("start_values_2026.csv", "ktc_replacement_mapping_source.json"):
        relative = PREFIX + "data/" + name
        (tmp_path / relative).write_bytes((ROOT / relative).read_bytes())
    archive = tmp_path / "backend/data/training-data.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("training-data.json", '{"players":[]}')
    (tmp_path / "README.md").write_text("Synthetic tracked documentation\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "seal-test@example.invalid")
    git(tmp_path, "config", "user.name", "Seal test")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "Synthetic reviewed sources")
    return tmp_path, git(tmp_path, "rev-parse", "HEAD").decode().strip()


def module():
    return importlib.import_module("predictor_v5.replacement_seal")


def assert_absent(root):
    assert not os.path.lexists(root / MANIFEST)
    assert not os.path.lexists(root / REGISTRY)


def test_generator_deterministic_and_refuses_unreviewed_identity(committed):
    root, anchor = committed
    seal = module()
    first = seal.build_manifest(root, anchor)
    assert first == seal.build_manifest(root, anchor)
    assert "sealed_at" not in first
    manifest = first | {"sealed_at": SEALED_AT}
    validate_manifest(manifest)
    assert {i["path"] for i in manifest["source_identities"]} == set(SOURCES)
    identities = manifest["source_identities"] + [
        manifest["training"]["archive"],
        manifest["training"]["mfl_labels"],
        manifest["starts"]["identity"],
        manifest["recipes"]["ktc"]["mapping_data"],
    ]
    for identity in identities:
        raw = git(root, "show", anchor + ":" + identity["path"])
        assert identity["sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["training"]["ktc_member"] == "training-data.json"
    assert manifest["training"]["seasons"] == [2021, 2022, 2023, 2024, 2025]
    assert manifest["gate"] == {
        "minimum_overall_delta": -0.01,
        "minimum_position_delta": -0.05,
        "minimum_position_count": 30,
    }
    assert (
        manifest["input_registration_path"]
        == PREFIX + "data/ktc_replacement_2026_inputs.json"
    )
    assert not any(
        key in canonical_bytes(manifest)
        for key in (b'"target_csv"', b'"target_stats"', b'"raw_end"')
    )
    registry = seal.build_registry(manifest, SEALED_AT)
    validate_registry(registry)
    assert registry["entries"] == [
        {
            "season": 2026,
            "manifest_sha256": hashlib.sha256(canonical_bytes(manifest)).hexdigest(),
            "status": "sealed",
            "sealing_commit": anchor,
            "sealed_at": SEALED_AT,
        }
    ]
    assert registry == seal.build_registry(manifest, SEALED_AT)
    assert_absent(root)
    twin = root.parent / (root.name + "-twin")
    shutil.copytree(root, twin)
    seal.write_seal(twin, anchor, SEALED_AT)
    seal.write_seal(root, anchor, SEALED_AT)
    assert (root / MANIFEST).read_bytes() == canonical_bytes(manifest)
    assert (root / REGISTRY).read_bytes() == canonical_bytes(registry)
    for relative in (MANIFEST, REGISTRY):
        assert (twin / relative).read_bytes() == (root / relative).read_bytes()
    assert sorted(
        git(root, "ls-files", "--others", "--exclude-standard").decode().splitlines()
    ) == sorted([MANIFEST, REGISTRY])


@pytest.mark.parametrize(
    "mutation",
    [
        "dirty",
        "staged",
        "missing",
        "nonhead",
        "tree",
        "abbrev",
        "symbolic",
        "assume_unchanged",
        "symlink",
        "bad_start",
        "bad_provenance",
        "bad_archive",
        "bad_mapping",
    ],
)
def test_invalid_anchor_or_identity_writes_neither(committed, mutation):
    root, anchor = committed
    source = root / SOURCES[0]
    if mutation in {"dirty", "staged", "assume_unchanged"}:
        if mutation == "assume_unchanged":
            git(root, "update-index", "--assume-unchanged", SOURCES[0])
        source.write_text("changed")
        if mutation == "staged":
            git(root, "add", SOURCES[0])
    elif mutation == "missing":
        git(root, "rm", SOURCES[0])
        git(root, "commit", "-qm", "Remove required identity")
        anchor = git(root, "rev-parse", "HEAD").decode().strip()
    elif mutation == "nonhead":
        git(root, "commit", "--allow-empty", "-qm", "New HEAD")
    elif mutation in {"tree", "abbrev", "symbolic"}:
        anchor = {
            "tree": git(root, "rev-parse", "HEAD^{tree}").decode().strip(),
            "abbrev": anchor[:8],
            "symbolic": "HEAD",
        }[mutation]
    else:
        if mutation == "symlink":
            source.unlink()
            source.symlink_to("model.py")
        if mutation == "bad_start":
            (root / PREFIX / "data/start_values_2026.csv").write_bytes(b"not frozen")
        if mutation == "bad_provenance":
            (
                root / PREFIX / "data/backchannel_labels_2019_2025_mfl.provenance.json"
            ).write_bytes(b"{}")
        if mutation == "bad_mapping":
            (root / PREFIX / "data/ktc_replacement_mapping_source.json").write_bytes(
                b"{}"
            )
        if mutation == "bad_archive":
            with zipfile.ZipFile(root / "backend/data/training-data.zip", "w") as z:
                z.writestr("other.json", "{}")
        git(root, "add", ".")
        git(root, "commit", "-qm", "Invalid identity")
        anchor = git(root, "rev-parse", "HEAD").decode().strip()
    with pytest.raises(ValueError):
        module().write_seal(root, anchor, SEALED_AT)
    assert_absent(root)


@pytest.mark.parametrize("relative", [MANIFEST, REGISTRY])
@pytest.mark.parametrize("kind", ["file", "dangling_symlink", "directory"])
def test_existing_output_never_replaced_or_creates_partner(committed, relative, kind):
    root, anchor = committed
    path = root / relative
    if kind == "file":
        path.write_bytes(b"existing")
    elif kind == "directory":
        path.mkdir()
    else:
        path.symlink_to("missing")
    before = path.lstat()
    with pytest.raises(ValueError):
        module().write_seal(root, anchor, SEALED_AT)
    assert path.lstat().st_ino == before.st_ino
    assert not os.path.lexists(root / (REGISTRY if relative == MANIFEST else MANIFEST))


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-10-25",
        "2026-10-25T12:00:00",
        "2026-10-25T12:00:00Z",
        "2026-10-25T12:00:00-04:00",
    ],
)
def test_timestamp_must_be_explicit_canonical_utc(committed, timestamp):
    root, anchor = committed
    with pytest.raises(ValueError):
        module().write_seal(root, anchor, timestamp)
    assert_absent(root)


def test_registry_rejects_timestamp_mismatch(committed):
    root, anchor = committed
    manifest = module().build_manifest(root, anchor) | {"sealed_at": SEALED_AT}
    with pytest.raises(ValueError):
        module().build_registry(manifest, "2026-10-25T13:00:00+00:00")


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_hidden_dirty_tracked_file_is_not_a_clean_anchor(committed, flag):
    root, anchor = committed
    git(root, "update-index", flag, "README.md")
    (root / "README.md").write_text("Dirty but hidden from git status\n")
    with pytest.raises(ValueError):
        module().write_seal(root, anchor, SEALED_AT)
    assert_absent(root)


def test_failed_second_publication_preserves_collision(committed, monkeypatch):
    root, anchor = committed
    seal = module()
    link = seal.os.link

    def race(source, destination, **kwargs):
        if destination == root / REGISTRY:
            destination.write_bytes(b"concurrent output")
        return link(source, destination, **kwargs)

    monkeypatch.setattr(seal.os, "link", race)
    with pytest.raises(ValueError):
        seal.write_seal(root, anchor, SEALED_AT)
    assert not (root / MANIFEST).exists()
    assert (root / REGISTRY).read_bytes() == b"concurrent output"
    assert not list((root / REGISTRY).parent.glob(".replacement-seal-*"))


@pytest.mark.parametrize("failure_at", [1, 2, 3])
def test_fsync_failure_leaves_no_outputs_or_temporaries(
    committed, monkeypatch, failure_at
):
    root, anchor = committed
    seal = module()
    fsync = seal.os.fsync
    calls = 0

    def fail(descriptor):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError("injected fsync failure")
        return fsync(descriptor)

    monkeypatch.setattr(seal.os, "fsync", fail)
    with pytest.raises(OSError, match="injected"):
        seal.write_seal(root, anchor, SEALED_AT)
    assert_absent(root)
    assert not list((root / REGISTRY).parent.glob(".replacement-seal-*"))


def test_cli_requires_exact_flags_and_has_no_overrides():
    for args in (
        [],
        ["--root", "/tmp"],
        ["--source-anchor", "HEAD", "--sealed-at", SEALED_AT],
        ["--source-a", "a" * 40, "--sealed-at", SEALED_AT],
    ):
        result = subprocess.run(
            [sys.executable, "-m", "predictor_v5.replacement_seal", *args],
            cwd=ROOT / "backend",
            capture_output=True,
            check=False,
        )
        assert result.returncode == 2


def test_protected_baseline_bytes_and_git_identity():
    confirmation = json.loads(
        (ROOT / PREFIX / "data/model_2026_confirmation_manifest.json").read_bytes()
    )
    protected = list(confirmation["source_identities"]) + [
        PREFIX + "data/model_2026_confirmation_manifest.json",
        PREFIX + "data/model_2026_holdout_registry.json",
    ]
    for relative in protected:
        raw = git(ROOT, "show", BASELINE + ":" + relative)
        assert raw == (ROOT / relative).read_bytes()
        assert git(ROOT, "rev-parse", BASELINE + ":" + relative) == git(
            ROOT, "rev-parse", "HEAD:" + relative
        )
    assert not git(
        ROOT,
        "diff",
        BASELINE,
        "--",
        PREFIX + "season.py",
        "backend/app",
        "frontend",
        "Procfile",
        "runtime.txt",
        ".github",
        "bin",
        "requirements.txt",
        "package.json",
    )
    assert (
        git(
            ROOT,
            "log",
            "--diff-filter=A",
            "--format=%H",
            "--",
            PREFIX + "data/start_values_2026.csv",
        )
        .decode()
        .strip()
        .startswith("c415177")
    )


@pytest.mark.parametrize("mutation", ["working_pair", "later_head"])
def test_seal_handoff_rejects_uncommitted_pair_or_wrong_head(
    committed, monkeypatch, mutation
):
    root, anchor = committed
    seal = module()
    seal.write_seal(root, anchor, SEALED_AT)
    git(root, "add", MANIFEST, REGISTRY)
    git(root, "commit", "-qm", "Synthetic artifact-only seal")
    seal_commit = git(root, "rev-parse", "HEAD").decode().strip()
    monkeypatch.setattr(sys.modules[__name__], "ROOT", root)
    # A real committed pair must pass before testing either invalid handoff.
    test_source_anchor_and_artifact_only_seal()
    if mutation == "working_pair":
        manifest = json.loads((root / MANIFEST).read_bytes())
        manifest["sealed_at"] = "2026-10-25T13:00:00+00:00"
        registry = seal.build_registry(manifest, manifest["sealed_at"])
        (root / MANIFEST).write_bytes(canonical_bytes(manifest))
        (root / REGISTRY).write_bytes(canonical_bytes(registry))
        assert git(root, "rev-parse", "HEAD").decode().strip() == seal_commit
        assert sorted(git(root, "diff", "--name-only").decode().splitlines()) == sorted(
            [MANIFEST, REGISTRY]
        )
    else:
        git(root, "commit", "--allow-empty", "-qm", "Not the intended seal commit")
        assert git(root, "rev-parse", "HEAD").decode().strip() != seal_commit
    with pytest.raises(AssertionError):
        test_source_anchor_and_artifact_only_seal()


def test_source_anchor_and_artifact_only_seal():
    present = [os.path.lexists(ROOT / p) for p in (MANIFEST, REGISTRY)]
    if not any(present):
        pytest.skip("Real seal intentionally absent before independent source review")
    assert all(present), "partial seal"
    commit = (
        git(ROOT, "log", "--diff-filter=A", "--format=%H", "--", MANIFEST)
        .decode()
        .strip()
    )
    # Phase B handoff must verify the just-created artifact commit, not find
    # an older seal somewhere in the current branch's history.
    assert commit == git(ROOT, "rev-parse", "HEAD").decode().strip()
    committed_manifest = git(ROOT, "show", commit + ":" + MANIFEST)
    committed_registry = git(ROOT, "show", commit + ":" + REGISTRY)
    assert committed_manifest == (ROOT / MANIFEST).read_bytes()
    assert committed_registry == (ROOT / REGISTRY).read_bytes()
    manifest = json.loads(committed_manifest)
    registry = json.loads(committed_registry)
    validate_manifest(manifest)
    validate_registry(registry)
    assert canonical_bytes(manifest) == committed_manifest
    assert canonical_bytes(registry) == committed_registry
    assert registry == module().build_registry(manifest, manifest["sealed_at"])
    assert registry["entries"][0]["manifest_sha256"] == hashlib.sha256(
        committed_manifest
    ).hexdigest()
    anchor = manifest["sealing_commit"]
    assert git(ROOT, "rev-list", "--parents", "-n", "1", commit).decode().split() == [
        commit,
        anchor,
    ]
    assert sorted(
        git(ROOT, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
        .decode()
        .splitlines()
    ) == sorted(["A\t" + MANIFEST, "A\t" + REGISTRY])
    for identity in manifest["source_identities"] + [
        manifest["training"]["archive"],
        manifest["training"]["mfl_labels"],
        manifest["starts"]["identity"],
        manifest["recipes"]["ktc"]["mapping_data"],
    ]:
        raw = git(ROOT, "show", anchor + ":" + identity["path"])
        assert hashlib.sha256(raw).hexdigest() == identity["sha256"]
        assert raw == (ROOT / identity["path"]).read_bytes()
    assert {i["path"] for i in manifest["source_identities"]} == set(SOURCES)
