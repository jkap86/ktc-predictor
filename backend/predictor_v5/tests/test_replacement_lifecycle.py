"""Behavioral tests for replacement preflight and one-shot claiming."""

import hashlib
import json
import multiprocessing
import os
import shutil
import stat
from pathlib import Path

import pandas as pd
import pytest

from predictor_v5 import replacement_lifecycle as lifecycle
from predictor_v5.replacement_contract import canonical_bytes
from predictor_v5.tests.test_replacement_cohort import (
    TARGET_COLUMNS,
    mfl_rows,
    players,
    season,
    write_archive,
)
from predictor_v5.tests.test_replacement_contract import (
    attestation_fixture,
    cross_fixture,
    registration_fixture,
    registry_fixture,
)

ROOT = Path(__file__).resolve().parents[3]


def _context(tmp_path):
    parent = tmp_path / "backend/predictor_v5/artifacts/replacements"
    parent.mkdir(parents=True)
    frame = pd.DataFrame({"player_id": ["1"], "season": [2021]})
    starts = pd.DataFrame({"player_id": ["1"], "season": [2026]})
    objects = {
        "manifest": {
            "protocol": "ktc-replacement-2026",
            "output_template": "backend/predictor_v5/artifacts/replacements/2026-<manifest-sha256>.json",
            "claim_template": "backend/predictor_v5/artifacts/replacements/2026-<manifest-sha256>.claim.json",
        },
        "registry": {"kind": "replacement registry"},
        "registration": {"kind": "replacement registration"},
        "joint_review": {"kind": "joint review"},
        "confirmation_manifest": {"kind": "confirmation manifest"},
        "confirmation_registry": {"kind": "confirmation registry"},
        "confirmation_registration": {"kind": "confirmation registration"},
    }
    digests = {
        name + "_sha256": hashlib.sha256(canonical_bytes(value)).hexdigest()
        for name, value in objects.items()
    }
    manifest_sha = digests["manifest_sha256"]
    values = {
        **objects,
        **digests,
        "training_frames": {"ktc": frame, "mfl": frame.copy()},
        "starts": starts,
        "output_path": parent / ("2026-" + manifest_sha + ".json"),
        "claim_path": parent / ("2026-" + manifest_sha + ".claim.json"),
    }
    root_fd_path = tmp_path.resolve()
    root_path, root_fd = lifecycle._absolute_root(root_fd_path)
    parent_fd = lifecycle._open_directory(
        root_fd, "backend/predictor_v5/artifacts/replacements"
    )
    return lifecycle._PreflightContext(
        values,
        token=lifecycle._CAPABILITY,
        root_fd=root_fd,
        parent_fd=parent_fd,
        root_path=root_path,
        output_name=values["output_path"].name,
        claim_name=values["claim_path"].name,
    )


def _race_claim(context, gate, queue):
    gate.wait()
    try:
        path = lifecycle.claim(context)
    except Exception as exc:  # noqa: BLE001 - subprocess result boundary
        queue.put(("error", type(exc).__name__))
    else:
        queue.put(("ok", str(path)))


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def _copy(root, relative):
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, destination)


def _identity(root, relative):
    return {
        "path": relative,
        "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
    }


def _real_preflight_fixture(tmp_path, training_case=None):
    for directory in (
        "backend/predictor_v5/artifacts/replacements",
        "backend/predictor_v5/artifacts/confirmations",
    ):
        (tmp_path / directory).mkdir(parents=True)
    confirmation_manifest = json.loads(
        (ROOT / lifecycle.CONFIRMATION_MANIFEST_PATH).read_text()
    )
    protected = {
        lifecycle.CONFIRMATION_MANIFEST_PATH,
        lifecycle.LABEL_MANIFEST_PATH,
        confirmation_manifest["baseline"]["path"],
        confirmation_manifest["candidate"]["path"],
        confirmation_manifest["training_data"]["path"],
        confirmation_manifest["target_recipe"]["sql"]["path"],
        confirmation_manifest["target_recipe"]["build"]["path"],
        *confirmation_manifest["source_identities"],
        "backend/predictor_v5/data/start_values_2026.csv",
    }
    for label in ("baseline", "candidate"):
        path = Path(confirmation_manifest[label]["path"])
        sidecar = (path.parent / (path.stem + ".provenance.json")).as_posix()
        if (ROOT / sidecar).exists():
            protected.add(sidecar)
    for relative in protected:
        _copy(tmp_path, relative)
    shutil.copy2(ROOT / ".git", tmp_path / ".git")

    manifest, _, _, _, _, _ = cross_fixture()
    manifest["registry_path"] = "backend/predictor_v5/data/replacement-registry.json"
    manifest["input_registration_path"] = "backend/predictor_v5/data/replacement-inputs.json"
    recipe_paths = {
        "ktc_sql": "fixture/ktc.sql",
        "extractor": "fixture/extractor.py",
        "mapping_code": "fixture/mapping.py",
        "mapping_data": "fixture/mapping.json",
        "mfl_sql": "fixture/mfl.sql",
        "build": "fixture/build.py",
    }
    for name, relative in recipe_paths.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + "\n")
    identities = {name: _identity(tmp_path, path) for name, path in recipe_paths.items()}
    manifest["recipes"]["ktc"].update(
        sql=identities["ktc_sql"], extractor=identities["extractor"],
        mapping_code=identities["mapping_code"], mapping_data=identities["mapping_data"],
    )
    manifest["recipes"]["mfl"].update(
        sql=identities["mfl_sql"], build=identities["build"]
    )
    manifest["source_identities"] = sorted(
        [identities[key] for key in ("ktc_sql", "extractor", "mapping_code", "mfl_sql", "build")],
        key=lambda value: value["path"],
    )
    archive = tmp_path / "fixture/training.zip"
    labels = tmp_path / "fixture/labels.csv"
    rows = players()
    if training_case == "missing":
        rows = [row for row in rows if row["position"] != "TE"]
    elif training_case == "leakage":
        rows[0]["seasons"].append(season(2026))
    write_archive(archive, rows)
    pd.DataFrame(mfl_rows(rows), columns=TARGET_COLUMNS).to_csv(labels, index=False)
    manifest["training"].update(
        archive=_identity(tmp_path, "fixture/training.zip"),
        mfl_labels=_identity(tmp_path, "fixture/labels.csv"),
        ktc_member="training-data.json",
    )
    manifest_path = tmp_path / lifecycle.REPLACEMENT_MANIFEST_PATH
    _write_json(manifest_path, manifest)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    joint = attestation_fixture()
    joint.update(
        replacement_manifest_sha256=manifest_sha,
        confirmation_manifest_sha256=lifecycle.CONFIRMATION_MANIFEST_SHA256,
        evidence_reference="backend/predictor_v5/data/joint-review.json",
        reviewed_at="2026-10-25T00:00:00+00:00",
    )
    joint_path = tmp_path / joint["evidence_reference"]
    _write_json(joint_path, joint)
    joint_identity = _identity(tmp_path, joint["evidence_reference"])

    registry = registry_fixture()
    registry["entries"][0].update(
        manifest_sha256=manifest_sha,
        sealing_commit=manifest["sealing_commit"],
        sealed_at=manifest["sealed_at"],
        reviewer=joint["reviewer"], evidence_commit=joint["evidence_commit"],
        evidence_reference=joint["evidence_reference"], reviewed_at=joint["reviewed_at"],
        joint_review=joint_identity,
    )
    registration = registration_fixture()
    registration["manifest_sha256"] = manifest_sha
    registration["holdout_attestation"].update(
        registry_key="2026:" + manifest_sha, reviewer=joint["reviewer"],
        evidence_commit=joint["evidence_commit"],
        evidence_reference=joint["evidence_reference"], reviewed_at=joint["reviewed_at"],
        joint_review=joint_identity,
    )
    for source, keys in (
        ("ktc", ("sql", "extractor", "mapping_code", "mapping_data")),
        ("mfl", ("sql", "build")),
    ):
        for key in keys:
            registration[source][key] = manifest["recipes"][source][key]
    outcomes = []
    for source, keys in (
        ("ktc", ("raw_end", "target_csv", "target_stats", "target_sidecar")),
        ("mfl", ("target_csv", "target_stats", "target_sidecar")),
    ):
        for key in keys:
            relative = f"outcomes/replacement-{source}-{key}"
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"registered outcome; never read")
            registration[source][key] = {"path": relative, "sha256": "a" * 64}
            outcomes.append(path)
    _write_json(tmp_path / manifest["registry_path"], registry)
    _write_json(tmp_path / manifest["input_registration_path"], registration)

    confirmation_review = {
        "reviewer": joint["reviewer"], "evidence_commit": joint["evidence_commit"],
        "evidence_reference": joint["evidence_reference"],
        "reviewed_at": joint["reviewed_at"], "transition_source": "sealed",
    }
    confirmation_registry = {
        "schema_version": 2,
        "attestation": "review and Git evidence; not proof of non-observation",
        "entries": [{
            "season": 2026, "manifest_sha256": lifecycle.CONFIRMATION_MANIFEST_SHA256,
            "status": "untouched", "sealing_commit": confirmation_manifest["sealing_commit"],
            "sealed_at": confirmation_manifest["sealed_at"], **confirmation_review,
        }],
    }
    confirmation_registration = {
        "schema_version": 2, "season": 2026,
        "manifest_sha256": lifecycle.CONFIRMATION_MANIFEST_SHA256,
        "target_csv": None, "target_sidecar": None, "target_stats": None,
        "provenance": {"model_version": "baseline-hl28-v1", "verified": True, "identity": "fixture"},
        "target_contract": {
            "raw_market_columns": [
                "player_id", "season", "start_value", "end_value", "start_evidence",
                "end_evidence", "start_trades_window", "end_trades_window",
                "end_value_next_board", "end_evidence_next_board",
            ],
            "joined_model_columns": ["player_id", "season", "position", "age", "points", "start_value", "end_value"],
            "label_filter": {"positive_values": True, "min_trades": 5},
        },
        "recipe": {
            "sql_path": confirmation_manifest["target_recipe"]["sql"]["path"],
            "sql_sha256": confirmation_manifest["target_recipe"]["sql"]["sha256"],
            "build_path": confirmation_manifest["target_recipe"]["build"]["path"],
            "build_sha256": confirmation_manifest["target_recipe"]["build"]["sha256"],
        },
        "holdout_attestation": {
            "registry_key": "2026:" + lifecycle.CONFIRMATION_MANIFEST_SHA256,
            "status": "untouched", **confirmation_review,
        },
    }
    for key in ("target_csv", "target_sidecar", "target_stats"):
        relative = f"outcomes/confirmation-{key}"
        path = tmp_path / relative
        path.write_bytes(b"registered confirmation outcome; never read")
        confirmation_registration[key] = {"path": relative, "sha256": "b" * 64}
        outcomes.append(path)
    _write_json(tmp_path / confirmation_manifest["registry_path"], confirmation_registry)
    _write_json(tmp_path / confirmation_manifest["input_registration_path"], confirmation_registration)
    return {
        "root": tmp_path, "manifest": manifest, "manifest_sha": manifest_sha,
        "registry": registry, "registration": registration, "joint": joint,
        "confirmation_manifest": confirmation_manifest,
        "confirmation_registry": confirmation_registry,
        "confirmation_registration": confirmation_registration,
        "outcomes": outcomes,
    }


def test_preflight_real_validators_never_read_outcome_inodes(tmp_path, monkeypatch):
    """Break caught: real canonical/training/starts validation reads outcomes."""
    fixture = _real_preflight_fixture(tmp_path)
    outcome_inodes = {(path.stat().st_dev, path.stat().st_ino) for path in fixture["outcomes"]}
    original_read = lifecycle.os.read

    def guarded_read(descriptor, amount):
        info = os.fstat(descriptor)
        assert (info.st_dev, info.st_ino) not in outcome_inodes
        return original_read(descriptor, amount)

    monkeypatch.setattr(lifecycle.os, "read", guarded_read)
    context = lifecycle.preflight(tmp_path)
    assert type(context) is lifecycle._PreflightContext
    assert set(context["training_frames"]) == {"ktc", "mfl"}
    assert not context["claim_path"].exists()


@pytest.mark.parametrize(
    "alias",
    [
        "fixture/ktc.sql",
        "fixture/training.zip",
        "fixture/labels.csv",
        "backend/predictor_v5/data/start_values_2026.csv",
        "fixture/mapping.json",
        lifecycle.REPLACEMENT_MANIFEST_PATH,
        "backend/predictor_v5/data/replacement-registry.json",
        "backend/predictor_v5/data/replacement-inputs.json",
        "backend/predictor_v5/data/joint-review.json",
        "backend/predictor_v5/data/backchannel_labels_2019_2025.csv",
        "backend/data/training-data.zip",
        "backend/predictor_v5/confirmation.py",
        "backend/predictor_v5/sql/backchannel_labels.sql",
        lifecycle.CONFIRMATION_MANIFEST_PATH,
        "backend/predictor_v5/data/model_2026_holdout_registry.json",
        "backend/predictor_v5/data/model_2026_confirmation_inputs.json",
    ],
)
def test_preflight_rejects_physical_outcome_aliases(tmp_path, alias):
    """Break caught: distinct registered paths hardlink to later readable bytes."""
    fixture = _real_preflight_fixture(tmp_path)
    outcome = fixture["outcomes"][0]
    outcome.unlink()
    os.link(tmp_path / alias, outcome)
    with pytest.raises(ValueError, match="alias"):
        lifecycle.preflight(tmp_path)


def _rewrite(path, value):
    _write_json(path, value)


@pytest.mark.parametrize(
    "failure",
    [
        "joint",
        "registration",
        "existing_claim",
        "existing_report",
        "training_missing",
        "training_leakage",
        "confirmation_output",
        "confirmation_reused",
    ],
)
def test_preclaim_refusals_stop_pipeline_without_outcome_read(
    tmp_path, monkeypatch, failure
):
    """Break caught: a named preclaim refusal reaches outcomes, fit, or output."""
    training_case = failure.removeprefix("training_") if failure.startswith("training_") else None
    fixture = _real_preflight_fixture(tmp_path, training_case=training_case)
    if failure == "joint":
        fixture["joint"]["evidence_untouched"] = False
        _rewrite(tmp_path / fixture["joint"]["evidence_reference"], fixture["joint"])
    elif failure == "registration":
        fixture["registration"]["holdout_attestation"]["status"] = "reused"
        _rewrite(
            tmp_path / fixture["manifest"]["input_registration_path"],
            fixture["registration"],
        )
    elif failure in {"existing_claim", "existing_report"}:
        suffix = ".claim.json" if failure == "existing_claim" else ".json"
        path = (
            tmp_path
            / "backend/predictor_v5/artifacts/replacements"
            / f"2026-{fixture['manifest_sha']}{suffix}"
        )
        path.write_text("existing")
    elif failure == "confirmation_output":
        path = tmp_path / fixture["confirmation_manifest"]["output_template"].replace(
            "<manifest-sha256>", lifecycle.CONFIRMATION_MANIFEST_SHA256
        )
        path.write_text("exposed")
    elif failure == "confirmation_reused":
        fixture["confirmation_registry"]["entries"][0]["status"] = "reused"
        _rewrite(
            tmp_path / fixture["confirmation_manifest"]["registry_path"],
            fixture["confirmation_registry"],
        )

    outcome_inodes = {(path.stat().st_dev, path.stat().st_ino) for path in fixture["outcomes"]}
    original_read = lifecycle.os.read
    calls = []

    def guarded_read(descriptor, amount):
        info = os.fstat(descriptor)
        assert (info.st_dev, info.st_ino) not in outcome_inodes
        return original_read(descriptor, amount)

    def callback(name):
        def record(*_args):
            calls.append(name)

        return record

    monkeypatch.setattr(lifecycle.os, "read", guarded_read)
    monkeypatch.setattr(lifecycle, "claim", callback("claim"))
    with pytest.raises((ValueError, OSError)):
        context = lifecycle.preflight(tmp_path)
        lifecycle.run_claimed_stages(
            context,
            validate_outcomes=callback("outcomes"),
            fit=callback("fit"),
            write_report=callback("report"),
        )
    assert calls == []
    claims = list(tmp_path.rglob("*.claim.json"))
    assert len(claims) == (1 if failure == "existing_claim" else 0)
    reports = [
        path
        for path in (
            tmp_path / "backend/predictor_v5/artifacts/replacements"
        ).glob("*.json")
        if not path.name.endswith(".claim.json")
    ]
    assert len(reports) == (1 if failure == "existing_report" else 0)


def test_preflight_nofollow_open_rejects_check_open_swap(tmp_path, monkeypatch):
    """Break caught: source becomes an outcome symlink between metadata and open."""
    fixture = _real_preflight_fixture(tmp_path)
    source = tmp_path / "fixture/ktc.sql"
    outcome = fixture["outcomes"][0]
    outcome_inode = (outcome.stat().st_dev, outcome.stat().st_ino)
    original_open = lifecycle.os.open
    original_read = lifecycle.os.read
    swapped = False

    def swapping_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "ktc.sql" and not swapped:
            swapped = True
            source.unlink()
            source.symlink_to(outcome)
        return original_open(path, flags, *args, **kwargs)

    def guarded_read(descriptor, amount):
        info = os.fstat(descriptor)
        assert (info.st_dev, info.st_ino) != outcome_inode
        return original_read(descriptor, amount)

    monkeypatch.setattr(lifecycle.os, "open", swapping_open)
    monkeypatch.setattr(lifecycle.os, "read", guarded_read)
    with pytest.raises(OSError):
        lifecycle.preflight(tmp_path)


def test_concurrent_claim_is_unique_and_exact(tmp_path, monkeypatch):
    """Break caught: racing evaluators both consume the same untouched holdout."""
    context = _context(tmp_path)
    multiprocessing_context = multiprocessing.get_context("fork")
    gate = multiprocessing_context.Event()
    queue = multiprocessing_context.Queue()
    processes = [
        multiprocessing_context.Process(
            target=_race_claim, args=(context, gate, queue)
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    gate.set()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    results = [queue.get(timeout=2) for _ in processes]
    assert [result[0] for result in results].count("ok") == 1
    assert [result[0] for result in results].count("error") == 1

    payload = json.loads(context["claim_path"].read_text())
    assert set(payload) == {
        "claimed_at",
        "confirmation_manifest_sha256",
        "confirmation_registration_sha256",
        "confirmation_registry_sha256",
        "joint_review_sha256",
        "manifest_sha256",
        "protocol",
        "registration_sha256",
        "registry_sha256",
        "schema_version",
        "season",
    }
    assert payload["manifest_sha256"] == context["manifest_sha256"]
    assert payload["registration_sha256"] == context["registration_sha256"]
    assert payload["joint_review_sha256"] == context["joint_review_sha256"]
    assert payload["confirmation_manifest_sha256"] == context[
        "confirmation_manifest_sha256"
    ]
    assert payload["confirmation_registration_sha256"] == context[
        "confirmation_registration_sha256"
    ]
    assert context["claim_path"].read_bytes() == canonical_bytes(payload)


def test_claim_fsyncs_file_then_directory(tmp_path, monkeypatch):
    """Break caught: claim returns before both durability barriers complete."""
    context = _context(tmp_path)
    events = []
    original_fsync = os.fsync
    original_open = os.open

    def recording_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == context._claim_name:
            events.append(("open-file", descriptor))
        return descriptor

    def recording_fsync(descriptor):
        events.append(("fsync", descriptor))
        return original_fsync(descriptor)

    monkeypatch.setattr(lifecycle.os, "open", recording_open)
    monkeypatch.setattr(lifecycle.os, "fsync", recording_fsync)
    lifecycle.claim(context)
    assert [event[0] for event in events] == [
        "open-file",
        "fsync",
        "fsync",
    ]
    assert events[0][1] == events[1][1]
    assert events[2][1] == context._parent_fd


def test_claim_retries_interrupted_and_partial_writes(tmp_path, monkeypatch):
    """Break caught: a short or interrupted write publishes a truncated claim."""
    context = _context(tmp_path)
    original_write = os.write
    calls = 0

    def interrupted_partial_write(descriptor, raw):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError
        return original_write(descriptor, raw[:7])

    monkeypatch.setattr(lifecycle.os, "write", interrupted_partial_write)
    lifecycle.claim(context)
    payload = json.loads(context["claim_path"].read_text())
    assert context["claim_path"].read_bytes() == canonical_bytes(payload)
    assert calls > 2


def test_claim_timestamp_is_canonical_and_injectable(tmp_path, monkeypatch):
    """Break caught: claims contain ambiguous or timezone-naive timestamps."""
    context = _context(tmp_path)
    monkeypatch.setattr(
        lifecycle, "_claim_timestamp", lambda: "2026-12-31T23:59:58+00:00"
    )
    lifecycle.claim(context)
    assert json.loads(context["claim_path"].read_text())["claimed_at"] == (
        "2026-12-31T23:59:58+00:00"
    )

    second = _context(tmp_path / "other")
    monkeypatch.setattr(lifecycle, "_claim_timestamp", lambda: "2026-12-31T23:59:58")
    with pytest.raises(ValueError, match="timestamp"):
        lifecycle.claim(second)
    assert not second["claim_path"].exists()


@pytest.mark.parametrize("stage", ["validation", "fit", "write"])
def test_crash_after_claim_consumes_without_report(tmp_path, stage):
    """Break caught: a post-claim failure permits an outcome-informed retry."""
    context = _context(tmp_path)
    lifecycle.claim(context)
    with pytest.raises(RuntimeError, match=stage):
        raise RuntimeError(stage)
    assert context["claim_path"].is_file()
    assert not context["output_path"].exists()
    with pytest.raises(ValueError, match="claim already exists"):
        lifecycle.claim(context)


def test_claim_fsync_failure_keeps_consumed_claim(tmp_path, monkeypatch):
    """Break caught: durability failure deletes the evidence and enables retry."""
    context = _context(tmp_path)
    monkeypatch.setattr(lifecycle.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("sync")))
    with pytest.raises(OSError, match="sync"):
        lifecycle.claim(context)
    assert context["claim_path"].exists()
    with pytest.raises(ValueError, match="claim already exists"):
        lifecycle.claim(context)


def test_existing_report_always_refuses_claim(tmp_path):
    """Break caught: a completed protocol can be consumed a second time."""
    context = _context(tmp_path)
    context["output_path"].write_bytes(b"immutable report")
    with pytest.raises(ValueError, match="report already exists"):
        lifecycle.claim(context)
    assert not context["claim_path"].exists()


def test_claim_rejects_hash_consistent_plain_context(tmp_path):
    """Break caught: callers mint a claim without obtaining preflight capability."""
    context = dict(_context(tmp_path))
    with pytest.raises(ValueError, match="capability"):
        lifecycle.claim(context)
    assert not context["claim_path"].exists()


def test_claim_mode_is_exact_under_restrictive_umask(tmp_path):
    """Break caught: umask leaves the durable claim owner-unreadable."""
    context = _context(tmp_path)
    previous = os.umask(0o777)
    try:
        lifecycle.claim(context)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(context["claim_path"].stat().st_mode) == 0o600


def test_claim_first_stage_runner_orders_real_callbacks(tmp_path):
    """Break caught: Task 8 validation or fitting runs before durable claim."""
    context = _context(tmp_path)
    events = []

    def validate(bound):
        assert bound["claim_path"].exists()
        events.append("validate")
        return "outcomes"

    def fit(bound, outcomes):
        assert bound["claim_path"].exists()
        assert outcomes == "outcomes"
        events.append("fit")
        return "fit"

    def report(bound, fitted):
        assert bound["claim_path"].exists()
        assert fitted == "fit"
        events.append("report")
        return "done"

    assert lifecycle.run_claimed_stages(
        context, validate_outcomes=validate, fit=fit, write_report=report
    ) == "done"
    assert events == ["validate", "fit", "report"]


@pytest.mark.parametrize("mutation", ["hash", "cross_reference", "training", "starts", "rebase"])
def test_claim_rejects_mutated_preflight_capability(tmp_path, mutation):
    """Break caught: a valid capability is edited after preflight and still claims."""
    context = _context(tmp_path)
    if mutation == "hash":
        context["manifest_sha256"] = "0" * 64
    elif mutation == "cross_reference":
        context["registration"]["kind"] = "changed"
    elif mutation == "training":
        context["training_frames"]["ktc"].loc[0, "season"] = 2026
    elif mutation == "starts":
        context["starts"].loc[0, "season"] = 2025
    else:
        other = tmp_path / "other/backend/predictor_v5/artifacts/replacements"
        other.mkdir(parents=True)
        context["claim_path"] = other / context["claim_path"].name
        context["output_path"] = other / context["output_path"].name
    with pytest.raises(ValueError, match="changed"):
        lifecycle.claim(context)


def test_claim_rejects_replaced_parent_but_uses_pinned_descriptor(tmp_path):
    """Break caught: claim follows a replacement output parent by pathname."""
    context = _context(tmp_path)
    parent = context["claim_path"].parent
    moved = parent.with_name("replacements-moved")
    parent.rename(moved)
    parent.mkdir()
    with pytest.raises(ValueError, match="parent path changed"):
        lifecycle.claim(context)
    assert not (parent / context._claim_name).exists()
    assert not (moved / context._claim_name).exists()


def test_claim_creation_and_directory_fsync_stay_on_pinned_parent(
    tmp_path, monkeypatch
):
    """Break caught: a pathname swap redirects create or directory durability."""
    context = _context(tmp_path)
    parent = context["claim_path"].parent
    moved = parent.with_name("replacements-moved")
    original_fsync = lifecycle.os.fsync
    claim_fd = None
    parent_fsyncs = []

    def swapping_fsync(descriptor):
        nonlocal claim_fd
        info = os.fstat(descriptor)
        if stat.S_ISREG(info.st_mode):
            claim_fd = descriptor
            result = original_fsync(descriptor)
            parent.rename(moved)
            parent.mkdir()
            return result
        parent_fsyncs.append(descriptor)
        return original_fsync(descriptor)

    monkeypatch.setattr(lifecycle.os, "fsync", swapping_fsync)
    lifecycle.claim(context)
    assert claim_fd is not None
    assert parent_fsyncs == [context._parent_fd]
    assert (moved / context._claim_name).exists()
    assert not (parent / context._claim_name).exists()


@pytest.mark.parametrize(
    "failure", ["partial_write", "fchmod", "file_fsync", "close", "directory_fsync"]
)
def test_postcreate_io_failures_retain_claim(tmp_path, monkeypatch, failure):
    """Break caught: an IO failure after exclusive create re-enables retry."""
    context = _context(tmp_path)
    original_write = lifecycle.os.write
    original_fchmod = lifecycle.os.fchmod
    original_fsync = lifecycle.os.fsync
    original_close = lifecycle.os.close
    claim_fd = None
    original_open = lifecycle.os.open

    def recording_open(path, flags, *args, **kwargs):
        nonlocal claim_fd
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == context._claim_name:
            claim_fd = descriptor
        return descriptor

    monkeypatch.setattr(lifecycle.os, "open", recording_open)
    if failure == "partial_write":
        calls = 0

        def fail_write(descriptor, raw):
            nonlocal calls
            calls += 1
            if calls == 1:
                return original_write(descriptor, raw[:7])
            raise OSError("partial write failure")

        monkeypatch.setattr(lifecycle.os, "write", fail_write)
    elif failure == "fchmod":
        monkeypatch.setattr(
            lifecycle.os, "fchmod", lambda _fd, _mode: (_ for _ in ()).throw(OSError("fchmod"))
        )
    elif failure in {"file_fsync", "directory_fsync"}:

        def fail_fsync(descriptor):
            if (failure == "file_fsync" and descriptor == claim_fd) or (
                failure == "directory_fsync" and descriptor == context._parent_fd
            ):
                raise OSError(failure)
            return original_fsync(descriptor)

        monkeypatch.setattr(lifecycle.os, "fsync", fail_fsync)
    else:

        def fail_close(descriptor):
            if descriptor == claim_fd:
                original_close(descriptor)
                raise OSError("close")
            return original_close(descriptor)

        monkeypatch.setattr(lifecycle.os, "close", fail_close)

    with pytest.raises(OSError):
        lifecycle.claim(context)
    assert context["claim_path"].exists()
    monkeypatch.setattr(lifecycle.os, "write", original_write)
    monkeypatch.setattr(lifecycle.os, "fchmod", original_fchmod)
    monkeypatch.setattr(lifecycle.os, "fsync", original_fsync)
    monkeypatch.setattr(lifecycle.os, "close", original_close)
    with pytest.raises(ValueError, match="claim already exists"):
        lifecycle.claim(context)


@pytest.mark.parametrize("stage", ["validation", "fit", "report"])
def test_claimed_stage_failures_retain_claim_and_no_report(tmp_path, stage):
    """Break caught: a real callback failure rolls back the durable claim."""
    context = _context(tmp_path)

    def validate(_bound):
        if stage == "validation":
            raise RuntimeError(stage)
        return "outcomes"

    def fit(_bound, outcomes):
        assert outcomes == "outcomes"
        if stage == "fit":
            raise RuntimeError(stage)
        return "fitted"

    def report(_bound, fitted):
        assert fitted == "fitted"
        raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match=stage):
        lifecycle.run_claimed_stages(
            context, validate_outcomes=validate, fit=fit, write_report=report
        )
    assert context["claim_path"].exists()
    assert not context["output_path"].exists()
    with pytest.raises(ValueError, match="claim already exists"):
        lifecycle.claim(context)
