"""Behavioral tests for replacement preflight and one-shot claiming."""

import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import pandas as pd
import pytest

from predictor_v5 import replacement_lifecycle as lifecycle
from predictor_v5.replacement_contract import canonical_bytes


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
    return {
        **objects,
        **digests,
        "training_frames": {"ktc": frame, "mfl": frame.copy()},
        "starts": starts,
        "output_path": parent / ("2026-" + manifest_sha + ".json"),
        "claim_path": parent / ("2026-" + manifest_sha + ".claim.json"),
    }


def _race_claim(context, gate, queue):
    gate.wait()
    try:
        path = lifecycle.claim(context)
    except Exception as exc:  # noqa: BLE001 - subprocess result boundary
        queue.put(("error", type(exc).__name__))
    else:
        queue.put(("ok", str(path)))


def _preflight_fixture(tmp_path, monkeypatch):
    data = tmp_path / "backend/predictor_v5/data"
    outputs = tmp_path / "backend/predictor_v5/artifacts/replacements"
    data.mkdir(parents=True)
    outputs.mkdir(parents=True)
    paths = {
        "manifest": data / "ktc_replacement_2026_manifest.json",
        "registry": data / "replacement_registry.json",
        "registration": data / "replacement_inputs.json",
        "joint": data / "joint.json",
        "source": data / "source.py",
        "archive": data / "training.zip",
        "labels": data / "labels.csv",
        "starts": data / "start_values_2026.csv",
        "mapping": data / "mapping.json",
    }
    for name, path in paths.items():
        path.write_bytes((name + "\n").encode())
    outcome_paths = []
    for name in (
        "raw.csv",
        "ktc.csv",
        "ktc-stats.zip",
        "ktc-sidecar.json",
        "mfl.csv",
        "mfl-stats.zip",
        "mfl-sidecar.json",
    ):
        path = data / name
        path.write_bytes(b"OUTCOME MUST NOT BE READ")
        outcome_paths.append(path)

    def identity(name):
        path = paths[name]
        return {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    manifest_hash = "a" * 64
    joint_hash = "b" * 64
    manifest = {
        "registry_path": paths["registry"].relative_to(tmp_path).as_posix(),
        "input_registration_path": paths["registration"].relative_to(
            tmp_path
        ).as_posix(),
        "output_template": "backend/predictor_v5/artifacts/replacements/2026-<manifest-sha256>.json",
        "claim_template": "backend/predictor_v5/artifacts/replacements/2026-<manifest-sha256>.claim.json",
        "source_identities": [identity("source")],
        "training": {
            "archive": identity("archive"),
            "mfl_labels": identity("labels"),
        },
        "starts": {"identity": identity("starts")},
        "recipes": {"ktc": {"mapping_data": identity("mapping")}},
    }
    review_identity = {
        "path": paths["joint"].relative_to(tmp_path).as_posix(),
        "sha256": joint_hash,
    }
    registration = {
        "holdout_attestation": {"joint_review": review_identity},
        "ktc": dict(
            zip(
                ("raw_end", "target_csv", "target_stats", "target_sidecar"),
                [
                    {
                        "path": path.relative_to(tmp_path).as_posix(),
                        "sha256": "0" * 64,
                    }
                    for path in outcome_paths[:4]
                ],
            )
        ),
        "mfl": dict(
            zip(
                ("target_csv", "target_stats", "target_sidecar"),
                [
                    {
                        "path": path.relative_to(tmp_path).as_posix(),
                        "sha256": "0" * 64,
                    }
                    for path in outcome_paths[4:]
                ],
            )
        ),
    }
    joint = {"confirmation_manifest_sha256": "c" * 64}
    registry = {"entries": []}

    def load(path, _fields):
        name = Path(path).name
        if name == paths["manifest"].name:
            return manifest, manifest_hash
        if name == paths["registry"].name:
            return registry, "d" * 64
        if name == paths["registration"].name:
            return registration, "e" * 64
        if name == paths["joint"].name:
            return joint, joint_hash
        raise AssertionError(name)

    monkeypatch.setattr(lifecycle, "load_canonical", load)
    monkeypatch.setattr(lifecycle, "validate_references", lambda *a, **k: {})
    monkeypatch.setattr(
        lifecycle,
        "_load_confirmation",
        lambda *a: {
            "manifest": {},
            "manifest_sha256": "c" * 64,
            "registry": {},
            "registry_sha256": "f" * 64,
            "registration": {},
            "registration_sha256": "9" * 64,
        },
    )
    frames = {
        "ktc": pd.DataFrame({"player_id": ["1"]}),
        "mfl": pd.DataFrame({"player_id": ["1"]}),
    }
    starts = pd.DataFrame({"player_id": ["1"], "season": [2026]})
    monkeypatch.setattr(lifecycle, "prepare_training", lambda *a: frames)
    monkeypatch.setattr(lifecycle, "load_starts", lambda *a: starts)
    return manifest, paths, set(outcome_paths), frames, starts


def test_preflight_never_opens_outcomes(tmp_path, monkeypatch):
    """Break caught: a refusal or successful preflight consumes outcome bytes."""
    manifest, paths, outcomes, frames, starts = _preflight_fixture(
        tmp_path, monkeypatch
    )
    opened_outcomes = []
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if Path(path) in outcomes:
            opened_outcomes.append(Path(path))
            raise AssertionError("outcome bytes opened before claim")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    context = lifecycle.preflight(tmp_path)
    assert opened_outcomes == []
    assert context["training_frames"] is frames
    assert context["starts"] is starts
    assert not context["claim_path"].exists()
    assert not context["output_path"].exists()

    paths["source"].write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        lifecycle.preflight(tmp_path)
    assert opened_outcomes == []
    assert not context["claim_path"].exists()

    paths["source"].write_bytes(b"source\n")
    original_identity = manifest["source_identities"][0]
    outcome = next(iter(outcomes))
    manifest["source_identities"][0] = {
        "path": outcome.relative_to(tmp_path).as_posix(),
        "sha256": hashlib.sha256(b"OUTCOME MUST NOT BE READ").hexdigest(),
    }
    with pytest.raises(ValueError, match="aliases"):
        lifecycle.preflight(tmp_path)
    assert opened_outcomes == []
    manifest["source_identities"][0] = original_identity

    report = tmp_path / manifest["output_template"].replace(
        "<manifest-sha256>", "a" * 64
    )
    report.write_bytes(b"existing")
    with pytest.raises(ValueError, match="report already exists"):
        lifecycle.preflight(tmp_path)
    assert opened_outcomes == []


def test_preflight_training_failure_is_retryable(tmp_path, monkeypatch):
    """Break caught: missing training coverage writes a claim or reads outcomes."""
    _, _, outcomes, _, _ = _preflight_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        lifecycle,
        "prepare_training",
        lambda *a: (_ for _ in ()).throw(ValueError("training coverage missing")),
    )
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if Path(path) in outcomes:
            raise AssertionError("outcome bytes opened before claim")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="coverage"):
        lifecycle.preflight(tmp_path)
    assert list(tmp_path.rglob("*.claim.json")) == []


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

    def recording_open(path, flags, *args):
        descriptor = original_open(path, flags, *args)
        if Path(path) == context["claim_path"]:
            events.append(("open-file", descriptor))
        elif Path(path) == context["claim_path"].parent:
            events.append(("open-directory", descriptor))
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
        "open-directory",
        "fsync",
    ]
    assert events[0][1] == events[1][1]
    assert events[2][1] == events[3][1]


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
