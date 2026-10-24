"""Real artifact preflight; every refusal asserts its stage and loader ordering."""
import json
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from predictor_v5 import __main__ as cli
from predictor_v5 import confirmation as c

ROOT = Path(__file__).resolve().parents[3]

_CONFIRMATION_SOURCE_PATHS = (
    "backend/predictor_v5/model.py",
    "backend/predictor_v5/sql/backchannel_labels.sql",
    "backend/predictor_v5/label_export.py",
    "backend/predictor_v5/data.py",
    "backend/predictor_v5/labels.py",
    "backend/predictor_v5/evaluate.py",
    "backend/predictor_v5/confirmation.py",
    "backend/predictor_v5/__main__.py",
)


def write(path, value):
    path.write_bytes(c.canonical_json_bytes(value))


def stats(path, seasons):
    players = [{"player_id": p, "position": p, "seasons": [
        {"year": s, "age": 25, "fantasy_points": 100, "games_played": 17, "years_exp": 2}
        for s in seasons]} for p in ("QB", "RB", "TE", "WR")]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("stats.json", json.dumps({"players": players}))


def labels(seasons):
    return pd.DataFrame([{
        "player_id": p, "season": s, "start_value": 100, "end_value": 110,
        "position": p, "age": 25, "games": 17, "points": 100,
        "start_evidence": 5, "end_evidence": 5, "start_trades_window": 5,
        "end_trades_window": 5, "end_value_next_board": None, "end_evidence_next_board": None,
    } for s in seasons for p in ("QB", "RB", "TE", "WR")], columns=c._LABEL_COLUMNS)


def rebind(e):
    for name in ("baseline", "candidate"):
        path = e["root"] / e["manifest"][name]["path"]
        e["manifest"][name]["sha256"] = c.file_sha256(path)
        write(c.label_export.provenance_path(path), {
            "label_file": {"sha256": c.file_sha256(path)},
            "model_version": "baseline-hl28-v1", "model_version_source": "export"})
    e["manifest"]["training_data"]["sha256"] = c.file_sha256(e["root"] / "training.zip")
    write(e["root"] / "manifest.json", e["manifest"])
    sha = c.file_sha256(e["root"] / "manifest.json")
    e["registry"]["entries"][0]["manifest_sha256"] = sha
    write(e["root"] / "registry.json", e["registry"])
    e["registration"]["manifest_sha256"] = sha
    e["registration"]["holdout_attestation"]["registry_key"] = "2026:" + sha
    for name in ("target_csv", "target_stats"):
        entry = e["registration"][name]
        entry["sha256"] = c.file_sha256(e["root"] / entry["path"])
        e["sidecar"][name] = dict(entry)
    write(e["root"] / "registered-sidecar.json", e["sidecar"])
    e["registration"]["target_sidecar"]["sha256"] = c.file_sha256(e["root"] / "registered-sidecar.json")
    save_registration(e)


def save_registration(e):
    write(e["root"] / "registration.json", e["registration"])


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    manifest = json.loads((ROOT / c.MANIFEST_PATH).read_text())
    manifest["source_identities"] = {
        source: c.file_sha256(ROOT / source) for source in _CONFIRMATION_SOURCE_PATHS
    }
    for source in _CONFIRMATION_SOURCE_PATHS:
        destination = tmp_path / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, destination)
    for recipe_name in ("sql", "build"):
        recipe = manifest["target_recipe"][recipe_name]
        recipe["sha256"] = c.file_sha256(tmp_path / recipe["path"])
    for name in ("baseline", "candidate"):
        labels(range(2021, 2026)).to_csv(tmp_path / f"{name}.csv", index=False)
        manifest[name]["path"] = f"{name}.csv"
    manifest["training_data"]["path"] = "training.zip"
    stats(tmp_path / "training.zip", range(2021, 2026))
    stats(tmp_path / "target-stats.zip", [2026])
    labels([2026]).to_csv(tmp_path / "target.csv", index=False)
    att = {"registry_key": "2026:" + "0" * 64, "status": "untouched", "reviewer": "reviewer",
           "evidence_commit": "review-commit", "evidence_reference": "review/2026",
           "reviewed_at": "2027-02-01T12:00:00+00:00", "transition_source": "sealed"}
    registry = {"schema_version": 2, "attestation": "review and Git evidence; not proof of non-observation",
                "entries": [{"season": 2026, "manifest_sha256": "0" * 64,
                             "sealing_commit": manifest["sealing_commit"], "sealed_at": manifest["sealed_at"],
                             **{k: v for k, v in att.items() if k != "registry_key"}}]}
    registration = {
        "schema_version": 2, "season": 2026, "manifest_sha256": "0" * 64,
        **{name: {"path": path, "sha256": "0" * 64} for name, path in (
            ("target_csv", "target.csv"), ("target_stats", "target-stats.zip"),
            ("target_sidecar", "registered-sidecar.json"))},
        "provenance": {"model_version": "baseline-hl28-v1", "verified": True, "identity": "sidecar"},
        "target_contract": {"raw_market_columns": list(c._RAW_MARKET_COLUMNS),
                            "joined_model_columns": ["player_id", "season", "position", "age", "points", "start_value", "end_value"],
                            "label_filter": {"positive_values": True, "min_trades": 5}},
        "recipe": {"sql_path": manifest["target_recipe"]["sql"]["path"],
                   "sql_sha256": manifest["target_recipe"]["sql"]["sha256"],
                   "build_path": manifest["target_recipe"]["build"]["path"],
                   "build_sha256": manifest["target_recipe"]["build"]["sha256"]},
        "holdout_attestation": att,
    }
    sidecar = {"schema_version": 1, "target_csv": {}, "target_stats": {},
               "model_version": "baseline-hl28-v1", "verified": True,
               "problems": [], "source": "export", "identity": "sidecar"}
    calls = []
    target_reads = []
    read_csv = pd.read_csv

    def observed_read(path, *args, **kwargs):
        if Path(path) == tmp_path / "target.csv":
            target_reads.append(path)
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(c.pd, "read_csv", observed_read)

    def loader(csv, archive):
        calls.append((csv, archive))
        return c._default_target_loader(csv, archive)

    config = c.ConfirmationConfig(repo_root=tmp_path, manifest_path=tmp_path / "manifest.json",
                                  registry_path=tmp_path / "registry.json",
                                  input_registration_path=tmp_path / "registration.json",
                                  target_loader=loader, output_dir=tmp_path / "out")
    result = {"root": tmp_path, "manifest": manifest, "registry": registry, "registration": registration,
              "sidecar": sidecar, "config": config, "calls": calls, "target_reads": target_reads}
    rebind(result)
    return result


def refuse(e, match, *, reads_target=False):
    e["target_reads"].clear()
    with pytest.raises(c.PreflightError, match=match):
        c.preflight(e["config"])
    assert e["calls"] == []
    assert bool(e["target_reads"]) is reads_target
    assert not e["config"].output_dir.exists()


def test_real_preflight_calls_default_loader_once_with_registered_stats(sealed):
    context = c.preflight(sealed["config"])
    assert sealed["calls"] == [(sealed["root"] / "target.csv", sealed["root"] / "target-stats.zip")]
    assert context.target_stats == sealed["root"] / "target-stats.zip"
    assert len(context.target_frame) == 4
    assert set(context.target_frame.season) == {2026}
    assert set(c._read_registered_stats_archive(context.training_archive, "training archive").season) == set(range(2021, 2026))


@pytest.mark.parametrize("source", [
    "backend/predictor_v5/data.py",
    "backend/predictor_v5/labels.py",
    "backend/predictor_v5/evaluate.py",
    "backend/predictor_v5/confirmation.py",
    "backend/predictor_v5/__main__.py",
])
def test_source_hash_drift_refuses_before_target_loading(sealed, source):
    # Break caught: a source that affects confirmation can change without invalidating the seal.
    source_copy = sealed["root"] / source
    registered_hash = sealed["manifest"]["source_identities"][source]
    source_copy.write_bytes(source_copy.read_bytes() + b"\n# test-only drift\n")

    assert c.file_sha256(source_copy) != registered_hash
    with pytest.raises(ValueError, match=re.escape(source)):
        c.validate_manifest(sealed["manifest"], sealed["root"])
    refuse(sealed, re.escape(source))


@pytest.mark.parametrize("change,match", [
    ("missing", "input registration.*regular"), ("noncanonical", "canonical"),
    ("manifest", "attestation key mismatch"), ("sidecar", "target sidecar.*regular"),
    ("hash", "target CSV hash mismatch"), ("sealed", "must be untouched"), ("reused", "must be untouched"),
])
def test_pre_target_refusals_never_call_loader(sealed, change, match):
    e = sealed
    if change == "missing":
        (e["root"] / "registration.json").unlink()
    elif change == "noncanonical":
        (e["root"] / "registration.json").write_text("{ }\n")
    elif change == "manifest":
        e["registration"]["manifest_sha256"] = "0" * 64
        save_registration(e)
    elif change == "sidecar":
        (e["root"] / "registered-sidecar.json").unlink()
    elif change == "hash":
        e["registration"]["target_csv"]["sha256"] = "0" * 64
        save_registration(e)
    else:
        entry = e["registry"]["entries"][0]
        entry["status"] = change
        if change == "sealed":
            for key in ("reviewer", "evidence_commit", "evidence_reference", "reviewed_at", "transition_source"):
                del entry[key]
        write(e["root"] / "registry.json", e["registry"])
    refuse(e, match)


@pytest.mark.parametrize("field", ["reviewer", "evidence_commit", "evidence_reference", "reviewed_at", "transition_source", "status"])
def test_attestation_fields_are_required_before_loading(sealed, field):
    del sealed["registration"]["holdout_attestation"][field]
    save_registration(sealed)
    refuse(sealed, "holdout_attestation missing fields")


@pytest.mark.parametrize("change,match", [("season", "identity drift"), ("provenance", "target provenance drift"), ("recipe", "recipe sql hash mismatch")])
def test_identity_drift_is_rejected_before_loading(sealed, change, match):
    registration = sealed["registration"]
    if change == "season":
        registration["season"] = 2025
    elif change == "provenance":
        registration["provenance"]["verified"] = False
    else:
        registration["recipe"]["sql_sha256"] = "0" * 64
    save_registration(sealed)
    refuse(sealed, match)


@pytest.mark.parametrize("change,match", [
    ("missing", "sidecar missing fields"), ("malformed", "invalid JSON"),
    ("stale", "sidecar target_csv mismatch"), ("wrong-version", "sidecar model_version"),
    ("unverified", "sidecar verified"), ("path", "sidecar target_csv mismatch"),
    ("identity", "sidecar identity"), ("problems", "sidecar problems"), ("source", "sidecar source"),
])
def test_exact_registered_sidecar_is_required(sealed, change, match):
    e = sealed
    write(c.label_export.provenance_path(e["root"] / "target.csv"), {
        "label_file": {"sha256": c.file_sha256(e["root"] / "target.csv")},
        "model_version": "baseline-hl28-v1", "model_version_source": "export"})
    data = e["sidecar"]
    if change == "missing":
        data = {}
    elif change == "stale":
        data["target_csv"]["sha256"] = "0" * 64
    elif change == "wrong-version":
        data["model_version"] = "v2"
    elif change == "unverified":
        data["verified"] = False
    elif change == "path":
        data["target_csv"]["path"] = "different.csv"
    elif change == "identity":
        e["registration"]["provenance"]["identity"] = "manifest"
    elif change == "problems":
        data["problems"] = ["unverified"]
    elif change == "source":
        data["source"] = "assertion"
    path = e["root"] / "registered-sidecar.json"
    write(path, data)
    if change == "malformed":
        path.write_bytes(b"not-json")
    e["registration"]["target_sidecar"]["sha256"] = c.file_sha256(path)
    save_registration(e)
    refuse(e, match)


@pytest.mark.parametrize("name", ["baseline", "candidate"])
@pytest.mark.parametrize("change,match", [
    ("duplicate", "raw duplicate.*count=2.*sample="),
    ("leakage", "raw future leakage.*count=1.*sample="),
    ("season", "trainable coverage.*count=4.*sample="),
    ("position", "trainable coverage.*count=1.*sample="),
    ("joined", "joined duplicate.*count=2.*sample="),
])
def test_training_integrity_fails_before_target_or_fit(sealed, name, change, match):
    e = sealed
    path = e["root"] / f"{name}.csv"
    data = pd.read_csv(path)
    if change == "duplicate":
        data = pd.concat([data, data.iloc[[0]].assign(start_trades_window=0)])
    elif change == "leakage":
        data = pd.concat([data, data.iloc[[0]].assign(season=2027, start_trades_window=0)])
    elif change == "season":
        data.loc[data.season == 2023, "start_trades_window"] = 0
    elif change == "position":
        data.loc[(data.season == 2023) & (data.position == "WR"), "start_value"] = 0
    else:
        archive_path = e["root"] / "training.zip"
        with zipfile.ZipFile(archive_path) as archive:
            payload = json.loads(archive.read("stats.json"))
        payload["players"][0]["seasons"].append(payload["players"][0]["seasons"][0])
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("stats.json", json.dumps(payload))
    data.to_csv(path, index=False)
    rebind(e)
    refuse(e, ("baseline" if change == "joined" else name) + ".*" + match)


@pytest.mark.parametrize("purpose,filename", [
    ("training archive", "training.zip"),
    ("target stats", "target-stats.zip"),
])
@pytest.mark.parametrize("corruption,cause", [
    ("not-zip", "invalid ZIP archive"),
    ("missing-json", "missing JSON member"),
    ("malformed-json", "malformed JSON"),
    ("invalid-structure", "invalid stats structure"),
    ("invalid-row", "invalid stats row"),
])
def test_corrupt_registered_stats_archive_is_preflight_refusal_and_cli_exit_two(
    sealed, monkeypatch, capsys, purpose, filename, corruption, cause
):
    archive_path = sealed["root"] / filename
    if corruption == "not-zip":
        archive_path.write_bytes(b"not a zip archive")
    else:
        payload = {
            "missing-json": ("readme.txt", "nothing to parse"),
            "malformed-json": ("stats.json", "{not-json"),
            "invalid-structure": ("stats.json", json.dumps({"players": {}})),
            "invalid-row": ("stats.json", json.dumps({"players": [{
                "player_id": "QB", "position": "QB", "seasons": [{"age": 25}],
            }]})),
        }[corruption]
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(*payload)
    rebind(sealed)
    fits = []
    monkeypatch.setattr(c.WhatIfModel, "fit", lambda self, frame: fits.append(frame))

    match = rf"{purpose}: {cause}"
    with pytest.raises(c.PreflightError, match=match):
        c.preflight(sealed["config"])
    monkeypatch.setattr(c, "ConfirmationConfig", lambda: sealed["config"])
    assert cli.main(["confirm-2026"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert re.search(match, captured.err)
    assert not sealed["config"].output_dir.exists()
    assert fits == []


def test_unexpected_registered_archive_reader_failure_remains_cli_exit_one(
    sealed, monkeypatch, capsys
):
    fits = []
    monkeypatch.setattr(c.WhatIfModel, "fit", lambda self, frame: fits.append(frame))
    monkeypatch.setattr(
        c.zipfile, "ZipFile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("reader bug")),
    )
    monkeypatch.setattr(c, "ConfirmationConfig", lambda: sealed["config"])

    with pytest.raises(RuntimeError, match="reader bug"):
        c.preflight(sealed["config"])
    assert cli.main(["confirm-2026"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "reader bug\n"
    assert not sealed["config"].output_dir.exists()
    assert fits == []


@pytest.mark.parametrize("field,value,match", [
    ("sealing_commit", None, "sealing_commit"), ("sealing_commit", {}, "sealing_commit"),
    ("sealing_commit", " ", "sealing_commit"),
    ("sealing_commit", "other", "sealing_commit mismatch"),
    ("sealed_at", "now", "sealed_at"), ("sealed_at", "2025-01-01T00:00:00+00:00", "sealed_at mismatch"),
    ("reviewed_at", "now", "reviewed_at"), ("reviewed_at", None, "reviewed_at"),
    ("reviewed_at", "2025-01-01T00:00:00+00:00", "predates sealing"),
    ("transition_source", "review", "transition_source"), ("transition_source", {}, "transition_source"),
    ("status", None, "invalid registry lifecycle"), ("status", {}, "invalid registry lifecycle"),
])
def test_registry_lifecycle_metadata_is_bound_before_target_read(sealed, field, value, match):
    sealed["registry"]["entries"][0][field] = value
    write(sealed["root"] / "registry.json", sealed["registry"])
    refuse(sealed, match)


@pytest.mark.parametrize("field,value", [("transition_source", "review"), ("reviewed_at", "now"), ("reviewed_at", {})])
def test_matching_but_invalid_attestation_does_not_authorize_loading(sealed, field, value):
    sealed["registry"]["entries"][0][field] = value
    sealed["registration"]["holdout_attestation"][field] = value
    write(sealed["root"] / "registry.json", sealed["registry"])
    save_registration(sealed)
    refuse(sealed, field)


@pytest.mark.parametrize("field", ["reviewer", "evidence_commit", "evidence_reference", "reviewed_at", "transition_source"])
def test_registration_review_fields_are_bound_to_registry(sealed, field):
    sealed["registration"]["holdout_attestation"][field] = (
        "2027-02-02T12:00:00+00:00" if field == "reviewed_at" else "different"
    )
    save_registration(sealed)
    refuse(sealed, field)


@pytest.mark.parametrize("change,match", [("hash", "target stats hash mismatch"), ("path", "repository-relative"), ("missing", "missing fields.*target_stats")])
def test_target_stats_registration_is_mandatory(sealed, change, match):
    entry = sealed["registration"]["target_stats"]
    if change == "hash":
        entry["sha256"] = "0" * 64
    elif change == "path":
        entry["path"] = "../stats.zip"
    else:
        del sealed["registration"]["target_stats"]
    save_registration(sealed)
    refuse(sealed, match)


@pytest.mark.parametrize("column,value", [("position", "WR"), ("age", 99), ("games", 0), ("points", 999)])
def test_default_loader_rejects_derived_csv_stat_mismatch(sealed, column, value):
    path = sealed["root"] / "target.csv"
    data = pd.read_csv(path)
    data.loc[0, column] = value
    data.to_csv(path, index=False)
    rebind(sealed)
    with pytest.raises(c.PreflightError, match=f"target stats mismatch.*{column}.*count=1.*sample="):
        c.preflight(sealed["config"])
    assert len(sealed["calls"]) == 1


def test_raw_target_duplicates_refuse_before_loading(sealed):
    path = sealed["root"] / "target.csv"
    data = pd.read_csv(path)
    pd.concat([data, data.iloc[[0]].assign(start_trades_window=0)]).to_csv(path, index=False)
    rebind(sealed)
    refuse(sealed, "duplicate player-season in market export: 1 keys repeated across 2 rows", reads_target=True)


def test_joined_target_duplicates_report_source_count_sample(sealed):
    data = labels([2026])
    sealed["config"].target_loader = lambda *_: pd.concat([data, data.iloc[[0]]])
    with pytest.raises(c.PreflightError, match="joined target duplicate.*count=2.*sample="):
        c.preflight(sealed["config"])


def test_joined_target_wrong_season_refuses(sealed):
    sealed["config"].target_loader = lambda *_: labels([2025])
    with pytest.raises(c.PreflightError, match="target contains training-season leakage"):
        c.preflight(sealed["config"])


@pytest.mark.parametrize("season", [2026, 2027])
def test_raw_training_leakage_is_rejected_even_with_zero_trades(sealed, season):
    path = sealed["root"] / "baseline.csv"
    data = pd.read_csv(path)
    pd.concat([data, data.iloc[[0]].assign(season=season, start_trades_window=0)]).to_csv(path, index=False)
    rebind(sealed)
    refuse(sealed, "baseline raw future leakage.*count=1.*sample=")


@pytest.mark.parametrize("change,match", [("duplicate-json", "duplicate JSON object key"), ("unknown", "unknown fields"), ("raw-columns", "column contract drift")])
def test_strict_registration_and_raw_column_refusals(sealed, change, match):
    if change == "duplicate-json":
        (sealed["root"] / "registration.json").write_bytes(b'{"season":2026,"season":2026}\n')
    elif change == "unknown":
        sealed["registration"]["extra"] = True
        save_registration(sealed)
    else:
        path = sealed["root"] / "target.csv"
        pd.read_csv(path).drop(columns="start_evidence").to_csv(path, index=False)
        rebind(sealed)
    refuse(sealed, match, reads_target=change == "raw-columns")


@pytest.mark.parametrize("empty", [True, False])
def test_real_cli_insufficient_target_writes_deterministic_atomic_retain(sealed, monkeypatch, capsys, empty):
    if empty:
        path = sealed["root"] / "target.csv"
        data = pd.read_csv(path)
        data["start_trades_window"] = 0
        data.to_csv(path, index=False)
        rebind(sealed)
    monkeypatch.setattr(c, "ConfirmationConfig", lambda: sealed["config"])
    assert cli.main(["confirm-2026"]) == 0
    output = next(sealed["config"].output_dir.glob("*.json"))
    first = output.read_bytes()
    wrapper = json.loads(first)
    assert wrapper["payload"]["decision"]["status"] == "retain"
    assert wrapper["payload"]["decision"]["promotion_authorized"] is False
    assert wrapper["payload"]["cohort"]["n"] == (0 if empty else 4)
    assert capsys.readouterr().out == f"payload_sha256={wrapper['payload_sha256']}\n"
    assert cli.main(["confirm-2026"]) == 2
    assert output.read_bytes() == first
    output.unlink()
    assert cli.main(["confirm-2026"]) == 0
    assert output.read_bytes() == first
    assert list(output.parent.iterdir()) == [output]


def test_exit_classification_reserves_one_for_unexpected_runtime_errors():
    assert c.confirmation_exit_code(c.PreflightError("refused")) == 2
    assert c.confirmation_exit_code(RuntimeError("write failed")) == 1
