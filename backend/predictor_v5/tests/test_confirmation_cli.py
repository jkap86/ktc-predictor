import hashlib
import json
from pathlib import Path

import pytest

from predictor_v5 import __main__ as cli
from predictor_v5 import confirmation

ROOT = Path(__file__).resolve().parents[3]


def test_confirmation_keeps_existing_cli_imports_and_rejects_promotion_authorization(tmp_path):
    # Break caught: confirmation rewires existing CLI dependencies or emits an authorizing report.
    source = (ROOT / "backend/predictor_v5/__main__.py").read_text(encoding="utf-8")
    for statement in (
        "from predictor_v5.data import DEFAULT_ZIP, build_dataset, load_season_stats",
        "from predictor_v5.evaluate import evaluate, holdout",
        "from predictor_v5.labels import get_label_source",
        "from predictor_v5.model import WhatIfModel",
    ):
        assert statement in source

    with pytest.raises(ValueError, match="promotion_authorized"):
        confirmation.write_report({"decision": {"promotion_authorized": True}}, tmp_path / "confirmation.json")


def test_write_report_hashes_only_the_enriched_payload_and_writes_canonical_bytes(tmp_path, monkeypatch):
    # Break caught: the digest becomes a self-hash or the report depends on JSON formatting.
    monkeypatch.setattr(confirmation, "_git_revision", lambda _root: None)
    final = tmp_path / "confirmation.json"

    wrapper = confirmation.write_report({"decision": {"promotion_authorized": False}, "name": "café"}, final)

    assert set(wrapper) == {"schema_version", "payload", "payload_sha256"}
    assert wrapper["schema_version"] == 1
    assert wrapper["payload"]["environment"]["git_revision"] is None
    assert set(wrapper["payload"]["environment"]) == {
        "python", "numpy", "pandas", "scikit_learn", "git_revision"
    }
    payload_bytes = confirmation.canonical_json_bytes(wrapper["payload"])
    assert wrapper["payload_sha256"] == hashlib.sha256(payload_bytes).hexdigest()
    assert wrapper["payload_sha256"] != hashlib.sha256(final.read_bytes()).hexdigest()
    assert final.read_bytes() == (
        json.dumps(wrapper, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def test_report_schema_rejects_unknown_fields():
    # Break caught: an unreviewed wrapper field is accepted into the protocol.
    payload = {"decision": {"promotion_authorized": False}, "environment": {}}
    wrapper = {
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": hashlib.sha256(confirmation.canonical_json_bytes(payload)).hexdigest(),
        "unknown": True,
    }
    with pytest.raises(ValueError, match="unknown"):
        confirmation.validate_report(wrapper)


def test_write_report_never_replaces_existing_output_or_leaves_temps(tmp_path):
    # Break caught: a second confirmation overwrites the first or strands partial bytes.
    final = tmp_path / "confirmation.json"
    final.write_bytes(b"original")

    with pytest.raises(FileExistsError):
        confirmation.write_report({"decision": {"promotion_authorized": False}}, final)

    assert final.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [final]


def test_write_failure_cleans_up_temporary_file(tmp_path, monkeypatch):
    # Break caught: a failed atomic publish leaves a misleading partial artifact.
    final = tmp_path / "confirmation.json"
    monkeypatch.setattr(confirmation.os, "link", lambda *_args: (_ for _ in ()).throw(OSError("publish failed")))

    with pytest.raises(OSError, match="publish failed"):
        confirmation.write_report({"decision": {"promotion_authorized": False}}, final)

    assert not final.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("option", [
    ["--labels", "ktc"], ["--labels=ktc"], ["--min-trades", "5"], ["--min-trades=5"],
    ["--end-column", "end_value"], ["--end-column=end_value"],
])
@pytest.mark.parametrize("before", [True, False])
def test_confirm_rejects_every_global_configuration_spelling_in_either_position(
    tmp_path, monkeypatch, capsys, option, before
):
    # Break caught: confirmation inherits a selectable global label setting.
    monkeypatch.setattr(confirmation, "REPOSITORY_ROOT", tmp_path)
    argv = option + ["confirm-2026"] if before else ["confirm-2026"] + option

    assert cli.main(argv) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "confirm-2026 accepts no options\n"
    assert list(tmp_path.rglob("*.json")) == []


def test_confirm_success_writes_once_and_prints_only_payload_fingerprint(tmp_path, monkeypatch, capsys):
    # Break caught: CLI prints unstable detail, skips the report, or authorizes promotion.
    output = tmp_path / "out.json"
    payload = {"decision": {"promotion_authorized": False}, "result": "ok"}
    monkeypatch.setattr(confirmation, "run_confirmation", lambda _config: payload)
    monkeypatch.setattr(confirmation, "report_path", lambda _config: output)

    assert cli.main(["confirm-2026"]) == 0

    wrapper = json.loads(output.read_text())
    captured = capsys.readouterr()
    assert captured.out == f"payload_sha256={wrapper['payload_sha256']}\n"
    assert captured.err == ""
    assert wrapper["payload"]["decision"]["promotion_authorized"] is False


def test_confirm_success_artifact_never_authorizes_promotion(tmp_path, monkeypatch, capsys):
    # Break caught: a successful CLI report authorizes a deployment.
    output = tmp_path / "out.json"
    monkeypatch.setattr(
        confirmation,
        "run_confirmation",
        lambda _config: {"decision": {"promotion_authorized": False}, "result": "ok"},
    )
    monkeypatch.setattr(confirmation, "report_path", lambda _config: output)

    assert cli.main(["confirm-2026"]) == 0

    wrapper = json.loads(output.read_text(encoding="utf-8"))
    assert wrapper["payload"]["decision"]["promotion_authorized"] is False
    assert capsys.readouterr().err == ""


def test_confirm_preflight_refusal_has_exact_streams_and_no_report(tmp_path, monkeypatch, capsys):
    # Break caught: expected sealed-input refusal is classified as a runtime failure or writes output.
    output = tmp_path / "out.json"
    monkeypatch.setattr(confirmation, "run_confirmation", lambda _config: (_ for _ in ()).throw(confirmation.PreflightError("input registration is not a regular file")))
    monkeypatch.setattr(confirmation, "report_path", lambda _config: output)

    assert cli.main(["confirm-2026"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "input registration is not a regular file\n"
    assert not output.exists()


def test_confirm_unexpected_or_write_failure_exits_one(tmp_path, monkeypatch, capsys):
    # Break caught: unexpected execution failures are mislabeled as user/preflight refusals.
    output = tmp_path / "out.json"
    monkeypatch.setattr(confirmation, "run_confirmation", lambda _config: {"decision": {"promotion_authorized": False}})
    monkeypatch.setattr(confirmation, "report_path", lambda _config: output)
    monkeypatch.setattr(confirmation, "write_report", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))

    assert cli.main(["confirm-2026"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "disk full\n"
    assert not output.exists()
