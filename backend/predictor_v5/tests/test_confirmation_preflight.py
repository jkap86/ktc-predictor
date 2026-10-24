"""Fail-closed registration checks run before a target can be loaded."""

import copy
import json

import pandas as pd
import pytest

import predictor_v5.confirmation as confirmation


def _write(path, value):
    path.write_bytes(confirmation.canonical_json_bytes(value))


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(confirmation, "validate_manifest", lambda value, root: None)
    target = tmp_path / "target.csv"
    raw = {"player_id": "p1", "season": 2026, "start_value": 100, "end_value": 110, "position": "QB", "age": 25, "games": 17, "points": 300, "start_evidence": 5, "end_evidence": 5, "start_trades_window": 5, "end_trades_window": 5, "end_value_next_board": None, "end_evidence_next_board": None}
    pd.DataFrame([raw], columns=confirmation._LABEL_COLUMNS).to_csv(target, index=False)
    sidecar = tmp_path / "target.provenance.json"; _write(sidecar, {})
    train = tmp_path / "training.zip"; train.write_bytes(b"training")
    for name in ("baseline.csv", "candidate.csv"):
        pd.DataFrame({"season": [2021, 2022, 2023, 2024, 2025]}).to_csv(tmp_path / name, index=False)
    sql = tmp_path / "recipe.sql"; sql.write_text("select 1")
    build = tmp_path / "build.py"; build.write_text("# build")
    sha = confirmation.file_sha256
    manifest = {"training_data": {"path": "training.zip", "sha256": sha(train)}, "baseline": {"path": "baseline.csv", "sha256": sha(tmp_path / "baseline.csv")}, "candidate": {"path": "candidate.csv", "sha256": sha(tmp_path / "candidate.csv")}, "target_recipe": {"sql": {"sha256": sha(sql)}, "build": {"sha256": sha(build)}}, "train_seasons": [2021, 2022, 2023, 2024, 2025], "positions": ["QB"]}
    manifest_path = tmp_path / "manifest.json"; _write(manifest_path, manifest); msha = sha(manifest_path)
    att = {"registry_key": "2026:" + msha, "status": "untouched", "reviewer": "r", "evidence_commit": "c", "evidence_reference": "e", "reviewed_at": "now", "transition_source": "review"}
    registry = {"schema_version": 1, "attestation": "review and Git evidence; not proof of non-observation", "entries": [{"season": 2026, "manifest_sha256": msha, "status": "untouched", "sealing_commit": "x", "sealed_at": "x", **{k:v for k,v in att.items() if k not in {"registry_key", "status"}}}]}
    registry_path = tmp_path / "registry.json"; _write(registry_path, registry)
    registration = {"schema_version": 1, "season": 2026, "manifest_sha256": msha, "target_csv": {"path": "target.csv", "sha256": sha(target)}, "target_sidecar": {"path": "target.provenance.json", "sha256": sha(sidecar)}, "provenance": {"model_version": "baseline-hl28-v1", "verified": True, "identity": "export"}, "target_contract": {"raw_market_columns": list(confirmation._RAW_MARKET_COLUMNS), "joined_model_columns": ["player_id", "season", "position", "age", "points", "start_value", "end_value"], "label_filter": {"positive_values": True, "min_trades": 5}}, "recipe": {"sql_path": "recipe.sql", "sql_sha256": sha(sql), "build_path": "build.py", "build_sha256": sha(build)}, "holdout_attestation": att}
    registration_path = tmp_path / "registration.json"; _write(registration_path, registration)
    calls = []
    config = confirmation.ConfirmationConfig(repo_root=tmp_path, manifest_path=manifest_path, registry_path=registry_path, input_registration_path=registration_path, target_loader=lambda *_: calls.append(True) or pd.DataFrame([raw]), output_dir=tmp_path / "out")
    return locals()


def _refuse(env):
    with pytest.raises(ValueError): confirmation.preflight(env["config"])
    assert not env["calls"]


@pytest.mark.parametrize("change", ["missing", "noncanonical", "manifest", "sidecar", "hash", "sealed", "reused"])
def test_pre_target_refusals_never_call_loader(sealed, change):
    if change == "missing": sealed["registration_path"].unlink()
    elif change == "noncanonical": sealed["registration_path"].write_text("{ }\n")
    elif change == "manifest": sealed["registration"]["manifest_sha256"] = "0" * 64; _write(sealed["registration_path"], sealed["registration"])
    elif change == "sidecar": sealed["sidecar"].unlink()
    elif change == "hash": sealed["registration"]["target_csv"]["sha256"] = "0" * 64; _write(sealed["registration_path"], sealed["registration"])
    else: sealed["registry"]["entries"][0]["status"] = change; _write(sealed["registry_path"], sealed["registry"])
    _refuse(sealed)


def test_duplicate_keys_and_unknown_wrapper_are_rejected(sealed):
    sealed["registration_path"].write_bytes(b'{"schema_version":1,"schema_version":1}\n'); _refuse(sealed)
    with pytest.raises(ValueError): confirmation.validate_input_registration({"unknown": True})


@pytest.mark.parametrize("field", ["reviewer", "evidence_commit", "evidence_reference", "reviewed_at", "transition_source", "status"])
def test_attestation_fields_are_required_before_loading(sealed, field):
    del sealed["registration"]["holdout_attestation"][field]; _write(sealed["registration_path"], sealed["registration"]); _refuse(sealed)


@pytest.mark.parametrize("change", ["season", "provenance", "recipe"])
def test_identity_drift_is_rejected_before_loading(sealed, change):
    r = sealed["registration"]
    if change == "season": r["season"] = 2025
    elif change == "provenance": r["provenance"]["verified"] = False
    else: r["recipe"]["sql_sha256"] = "0" * 64
    _write(sealed["registration_path"], r); _refuse(sealed)


def test_raw_duplicates_are_rejected_before_loader(sealed):
    raw = pd.read_csv(sealed["target"]); pd.concat([raw, raw]).to_csv(sealed["target"], index=False)
    sealed["registration"]["target_csv"]["sha256"] = confirmation.file_sha256(sealed["target"]); _write(sealed["registration_path"], sealed["registration"]); _refuse(sealed)


@pytest.mark.parametrize("frame", [pd.DataFrame([{"player_id":"x","season":2026,"position":"QB","age":1,"points":1,"start_value":1,"end_value":1,"start_trades_window":5,"end_trades_window":5}] * 2), pd.DataFrame([{"player_id":"x","season":2025,"position":"QB","age":1,"points":1,"start_value":1,"end_value":1,"start_trades_window":5,"end_trades_window":5}])])
def test_joined_duplicates_and_training_leakage_are_rejected(sealed, frame):
    sealed["config"].target_loader = lambda *_: frame
    with pytest.raises(ValueError): confirmation.preflight(sealed["config"])


def test_incomplete_training_coverage_is_rejected(sealed):
    pd.DataFrame({"season": [2021]}).to_csv(sealed["tmp_path"] / "baseline.csv", index=False)
    sealed["manifest"]["baseline"]["sha256"] = confirmation.file_sha256(sealed["tmp_path"] / "baseline.csv")
    _write(sealed["manifest_path"], sealed["manifest"])
    manifest_sha = confirmation.file_sha256(sealed["manifest_path"])
    sealed["registry"]["entries"][0]["manifest_sha256"] = manifest_sha
    _write(sealed["registry_path"], sealed["registry"])
    sealed["registration"]["manifest_sha256"] = manifest_sha
    sealed["registration"]["holdout_attestation"]["registry_key"] = "2026:" + manifest_sha
    _write(sealed["registration_path"], sealed["registration"])
    _refuse(sealed)


def test_exit_classification_reserves_one_for_unexpected_runtime_errors():
    assert confirmation.confirmation_exit_code(confirmation.PreflightError("refused")) == 2
    assert confirmation.confirmation_exit_code(RuntimeError("write failed")) == 1
