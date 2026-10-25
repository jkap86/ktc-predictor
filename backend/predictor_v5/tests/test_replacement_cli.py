import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from predictor_v5.replacement_contract import canonical_bytes


def test_atomic_canonical_report(tmp_path, monkeypatch):
    from predictor_v5 import replacement_report

    (tmp_path / "reports").mkdir()
    destination = tmp_path / "reports" / "result.json"
    payload = {"z": [3, 2, 1], "a": {"value": "deterministic"}}
    wrapper = replacement_report.write_report(payload, destination)

    assert set(wrapper) == {"schema_version", "payload", "payload_sha256"}
    assert wrapper == {
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical_bytes(payload)).hexdigest(),
    }
    assert destination.read_bytes() == canonical_bytes(wrapper)
    with pytest.raises(FileExistsError):
        replacement_report.write_report({"other": True}, destination)
    assert destination.read_bytes() == canonical_bytes(wrapper)

    failed = tmp_path / "reports" / "failed.json"
    real_fsync = os.fsync
    calls = 0

    def fail_directory_sync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected directory fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(replacement_report.os, "fsync", fail_directory_sync)
    with pytest.raises(OSError, match="directory fsync"):
        replacement_report.write_report(payload, failed)
    assert not failed.exists()
    assert not list(failed.parent.glob(f".{failed.name}.tmp.*"))


def test_report_requires_existing_parent_directory(tmp_path):
    from predictor_v5.replacement_report import write_report

    destination = tmp_path / "missing" / "result.json"
    with pytest.raises(FileNotFoundError):
        write_report({"decision": "retain_ktc"}, destination)
    assert not destination.parent.exists()


def test_report_temp_collision_is_not_removed(tmp_path, monkeypatch):
    from predictor_v5 import replacement_report

    token = "collision"
    destination = tmp_path / "result.json"
    collision = tmp_path / f".{destination.name}.tmp.{token}"
    collision.write_bytes(b"unowned")
    monkeypatch.setattr(replacement_report.secrets, "token_hex", lambda _size: token)

    with pytest.raises(FileExistsError):
        replacement_report.write_report({"decision": "retain_ktc"}, destination)
    assert collision.read_bytes() == b"unowned"
    assert not destination.exists()


def test_report_cleanup_preserves_replacement_final_inode(tmp_path, monkeypatch):
    from predictor_v5 import replacement_report

    destination = tmp_path / "result.json"
    replacement = b"concurrent replacement"
    real_fsync = os.fsync
    calls = 0

    def replace_before_directory_sync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            destination.unlink()
            destination.write_bytes(replacement)
            raise OSError("injected directory fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(replacement_report.os, "fsync", replace_before_directory_sync)
    with pytest.raises(OSError, match="directory fsync"):
        replacement_report.write_report({"decision": "retain_ktc"}, destination)
    assert destination.read_bytes() == replacement


def _context(tmp_path):
    frame = pd.DataFrame(
        [{"player_id": "p", "season": 2025, "position": "QB",
          "age": 24.0, "games": 17, "points": 300.0,
          "start_value": 100.0, "end_value": 110.0}]
    )
    return {
        "manifest": {
            "schema_version": 1,
            "protocol": "ktc-replacement-2026",
            "sealing_commit": "a" * 40,
            "sealed_at": "2026-10-25T00:00:00+00:00",
            "source_identities": [{"path": "code.py", "sha256": "1" * 64}],
            "training": {"archive": {"path": "train.zip", "sha256": "2" * 64},
                         "ktc_member": "training-data.json",
                         "mfl_labels": {"path": "labels.csv", "sha256": "3" * 64},
                         "seasons": [2021, 2022, 2023, 2024, 2025],
                         "positions": ["QB", "RB", "TE", "WR"]},
            "starts": {"identity": {"path": "starts.csv", "sha256": "4" * 64},
                       "git_blob": "b" * 40, "introduced_commit": "c415177",
                       "cutoff": "2026-09-09",
                       "provenance_limit": "raw start provenance not independently verified"},
            "model": {"ktc_scale_max": 9999, "mfl_scale_max": 10000,
                      "features": ["log_start", "age", "points", "position_code"]},
            "cutoffs": {"start": "2026-09-09", "end": "2027-01-10"},
            "filters": {"ktc": {"minimum_exclusive": 0},
                        "mfl": {"minimum_exclusive": 0}},
            "recipes": {"ktc": {"mapping_data": {"path": "mapping.json", "sha256": "5" * 64}},
                        "mfl": {"build": {"path": "build.py", "sha256": "6" * 64}}},
            "diagnostics": {"bootstrap": {"seed": 20261024, "resamples": 10000,
                                             "percentiles": [2.5, 97.5]}},
        },
        "manifest_sha256": "7" * 64,
        "registry": {"entries": []}, "registry_sha256": "8" * 64,
        "registration": {"ktc": {}, "mfl": {}}, "registration_sha256": "9" * 64,
        "joint_review": {"reviewer": "reviewer"}, "joint_review_sha256": "a" * 64,
        "confirmation_manifest": {"protocol": "confirm-2026"},
        "confirmation_manifest_sha256": "b" * 64,
        "confirmation_registry": {"entries": []},
        "confirmation_registry_sha256": "c" * 64,
        "confirmation_registration": {},
        "confirmation_registration_sha256": "d" * 64,
        "training_frames": {"ktc": frame.copy(), "mfl": frame.copy()},
        "starts": pd.DataFrame(),
        "output_path": tmp_path / "report.json",
        "claim_path": tmp_path / "report.claim.json",
    }


def test_cli_exit_and_order(tmp_path):
    from predictor_v5 import ktc_replacement_2026 as cli

    events = []
    context = _context(tmp_path)

    def preflight(_root):
        events.append("preflight")
        return context

    def claimed(bound, *, validate_outcomes, fit, write_report):
        assert bound is context
        events.append("claim")
        context["claim_path"].write_text("claimed")
        outcomes = validate_outcomes(bound)
        fitted = fit(bound, outcomes)
        events.append("report")
        raw = write_report(fitted)
        context["output_path"].write_bytes(raw)
        return context["output_path"]

    cohort = pd.DataFrame(columns=[
        "player_id", "season", "position", "age", "games", "points",
        "ktc_start", "ktc_actual", "ktc_pred", "mfl_start", "mfl_actual",
        "mfl_pred",
    ])

    def validate(_bound, _deps, _root):
        events.extend(["verify_ktc", "verify_mfl", "verify_confirmation"])
        return {"ktc": object(), "mfl": object(), "actual_dates": {},
                "coverage": {"ktc": {}, "mfl": {}}}

    def fit(_bound, _outcomes, _deps):
        events.append("fit")
        gate = {"decision": "retain_ktc", "reasons": ["invalid_overall_count"],
                "primary": "ktc", "fallback": None, "promotion_authorized": False}
        return {"fits": {}, "cohort_frame": cohort, "coverage": {},
                "metrics": {}, "diagnostics": {}, "gate": gate}

    wrapper = cli.run(tmp_path, {
        "preflight": preflight,
        "run_claimed_stages": claimed,
        "validate_outcomes": validate,
        "fit_evaluation": fit,
    })
    assert events == ["preflight", "claim", "verify_ktc", "verify_mfl",
                      "verify_confirmation", "fit", "report"]
    assert wrapper["payload"]["decision"]["recommendation"] == "retain_ktc"
    assert wrapper["payload"]["diagnostics"]["bootstrap_statistic"] == (
        "paired raw MFL-minus-KTC MAE difference"
    )
    assert json.loads(context["output_path"].read_bytes()) == wrapper


def test_empty_targets_reach_real_gate_report_and_exit_0(tmp_path, monkeypatch):
    from predictor_v5 import ktc_replacement_2026 as cli
    from predictor_v5.replacement_cohort import START_COLUMNS
    from predictor_v5.replacement_ktc import TARGET_COLUMNS as KTC_COLUMNS
    from predictor_v5.replacement_mfl import TARGET_COLUMNS as MFL_COLUMNS

    context = _context(tmp_path)
    context["starts"] = pd.DataFrame(columns=START_COLUMNS)
    empty_ktc = pd.DataFrame(columns=KTC_COLUMNS)
    empty_mfl = pd.DataFrame(columns=MFL_COLUMNS)
    outcomes = {
        "ktc": empty_ktc,
        "mfl": empty_mfl,
        "actual_dates": {},
        "coverage": {"ktc": {}, "mfl": {}},
    }

    class Model:
        def fit(self, _training):
            return self

        def predict(self, _target):
            pytest.fail("empty target reached model.predict")

    def model_factory(**_kwargs):
        return Model()

    def claimed(bound, *, validate_outcomes, fit, write_report):
        bound["claim_path"].write_text("claimed")
        payload = fit(bound, validate_outcomes(bound))
        bound["output_path"].write_bytes(write_report(payload))
        return bound["output_path"]

    actual_run = cli.run
    monkeypatch.setattr(
        cli,
        "run",
        lambda root: actual_run(
            root,
            {
                "preflight": lambda _root: context,
                "run_claimed_stages": claimed,
                "validate_outcomes": lambda *_args: outcomes,
                "model_factory": model_factory,
            },
        ),
    )

    assert cli.main() == 0
    wrapper = json.loads(context["output_path"].read_bytes())
    assert wrapper["payload"]["cohort"]["rows"] == []
    assert wrapper["payload"]["gate"]["decision"] == "retain_ktc"
    assert wrapper["payload"]["decision"] == {
        "recommendation": "retain_ktc",
        "promotion_authorized": False,
    }


@pytest.mark.parametrize("phase", ["preflight", "validation"])
def test_protocol_failures_exit_2_and_do_not_reach_later_stages(
    phase, tmp_path, monkeypatch
):
    from predictor_v5 import ktc_replacement_2026 as cli

    context = _context(tmp_path)
    called = []
    actual_run = cli.run

    def preflight(_root):
        called.append("preflight")
        if phase == "preflight":
            raise ValueError("invalid identity")
        return context

    def claimed(bound, *, validate_outcomes, fit, write_report):
        called.append("claim")
        bound["claim_path"].write_text("claimed")
        validate_outcomes(bound)
        pytest.fail("validation refusal reached fit")

    def validate(*_args):
        called.append("validation")
        raise ValueError("invalid outcome")

    monkeypatch.setattr(cli, "run", lambda root: actual_run(root, {
        "preflight": preflight, "run_claimed_stages": claimed,
        "validate_outcomes": validate,
    }))
    assert cli.main() == 2
    if phase == "preflight":
        assert called == ["preflight"]
        assert not context["claim_path"].exists()
    else:
        assert called == ["preflight", "claim", "validation"]
        assert context["claim_path"].exists()
        assert not context["output_path"].exists()


def test_unexpected_fit_failure_exits_1_and_retains_claim(tmp_path, monkeypatch):
    from predictor_v5 import ktc_replacement_2026 as cli

    context = _context(tmp_path)
    actual_run = cli.run

    def claimed(bound, *, validate_outcomes, fit, write_report):
        bound["claim_path"].write_text("claimed")
        outcomes = validate_outcomes(bound)
        fit(bound, outcomes)

    monkeypatch.setattr(cli, "run", lambda root: actual_run(root, {
        "preflight": lambda _root: context,
        "run_claimed_stages": claimed,
        "validate_outcomes": lambda *_args: {},
        "fit_evaluation": lambda *_args: (_ for _ in ()).throw(RuntimeError("fit crashed")),
    }))
    assert cli.main() == 1
    assert context["claim_path"].exists()
    assert not context["output_path"].exists()


def test_standalone_cli_refuses_arguments():
    result = subprocess.run(
        [sys.executable, "-m", "predictor_v5.ktc_replacement_2026", "--anything"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
