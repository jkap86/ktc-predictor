import hashlib
import json
import re
from pathlib import Path

import pytest

from predictor_v5 import confirmation
from predictor_v5.confirmation import (
    MANIFEST_PATH,
    REGISTRY_PATH,
    canonical_json_bytes,
    load_canonical_json,
    sha256_bytes,
    validate_manifest,
    validate_registry,
    validate_repo_relative_path,
)

ROOT = Path(__file__).resolve().parents[3]


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def test_canonical_bytes_and_hash_are_strict_utf8_json():
    # Break caught: serialization no longer gives one deterministic byte representation.
    assert canonical_json_bytes({"z": "é", "a": [1, True]}) == b'{"a":[1,true],"z":"\xc3\xa9"}\n'
    assert sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


@pytest.mark.parametrize("value", ["/tmp/x", "../x", "a/../../x", "a/./b", "", "."])
def test_repo_relative_paths_reject_absolute_and_traversal(value):
    # Break caught: artifacts can escape the repository root.
    with pytest.raises(ValueError):
        validate_repo_relative_path(value)


def test_committed_manifest_is_canonical_and_binds_the_frozen_protocol():
    # Break caught: a selectable parameter drifts from the sealed protocol.
    manifest, manifest_hash = load_canonical_json(ROOT / MANIFEST_PATH, validate_manifest)
    assert manifest_hash == hashlib.sha256((ROOT / MANIFEST_PATH).read_bytes()).hexdigest()
    assert manifest["protocol"] == "model-2026-confirmation"
    assert manifest["train_seasons"] == [2021, 2022, 2023, 2024, 2025]
    assert manifest["positions"] == ["QB", "RB", "TE", "WR"]
    assert manifest["label_filter"]["min_trades"] == 5
    assert manifest["market_model_version"] == "baseline-hl28-v1"
    assert manifest["gate"] == {"minimum_gain": 0.02, "minimum_position_count": 30, "maximum_position_drop": 0.05}
    assert manifest["bootstrap"] == {"seed": 20261024, "resamples": 10000, "percentiles": [2.5, 97.5]}
    assert manifest["registry_path"] == "backend/predictor_v5/data/model_2026_holdout_registry.json"
    assert manifest["input_registration_path"] == "backend/predictor_v5/data/model_2026_confirmation_inputs.json"
    assert manifest["output_template"] == "backend/predictor_v5/artifacts/confirmations/2026-<manifest-sha256>.json"
    assert set(confirmation._SOURCE_PATHS) == {
        "backend/predictor_v5/model.py",
        "backend/predictor_v5/sql/backchannel_labels.sql",
        "backend/predictor_v5/label_export.py",
        "backend/predictor_v5/data.py",
        "backend/predictor_v5/labels.py",
        "backend/predictor_v5/evaluate.py",
        "backend/predictor_v5/confirmation.py",
        "backend/predictor_v5/__main__.py",
    }


def test_frozen_sql_recipe_has_only_the_announced_2026_cutoff_and_no_target_data():
    # Break caught: the future export recipe omits, duplicates, or misdates 2026,
    # changes historical cutoffs/default model, or commits holdout target data.
    sql = (ROOT / "backend/predictor_v5/sql/backchannel_labels.sql").read_text(encoding="utf-8")
    values = re.search(r"WITH cutoffs\(season, start_date, end_date\) AS \(\s*VALUES(?P<rows>.*?)\n\)", sql, re.DOTALL)
    assert values is not None
    rows = re.findall(r"\((\d+), DATE '([^']+)', DATE '([^']+)'\)", values.group("rows"))
    assert rows == [
        ("2019", "2019-09-05", "2019-12-29"),
        ("2020", "2020-09-10", "2021-01-03"),
        ("2021", "2021-09-09", "2022-01-09"),
        ("2022", "2022-09-08", "2023-01-08"),
        ("2023", "2023-09-07", "2024-01-07"),
        ("2024", "2024-09-05", "2025-01-05"),
        ("2025", "2025-09-04", "2026-01-04"),
        ("2026", "2026-09-09", "2027-01-10"),
    ]
    assert "\\set model_version baseline-hl28-v1" in sql
    assert not (ROOT / "backend/predictor_v5/data/model_2026_confirmation_inputs.json").exists()


def test_manifest_rejects_unknown_or_missing_fields():
    # Break caught: an unreviewed field is silently accepted or a required one omitted.
    manifest, _ = load_canonical_json(ROOT / MANIFEST_PATH, validate_manifest)
    with pytest.raises(ValueError, match="unknown"):
        validate_manifest({**manifest, "unreviewed": True})
    missing = dict(manifest)
    missing.pop("gate")
    with pytest.raises(ValueError, match="missing"):
        validate_manifest(missing)


def test_loader_rejects_duplicate_keys_and_noncanonical_bytes(tmp_path):
    # Break caught: two parsers can interpret a reviewed artifact differently.
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"schema_version":1,"schema_version":1}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_canonical_json(duplicate, lambda value: value)
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text('{ "a": 1 }\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        load_canonical_json(noncanonical, lambda value: value)


def test_initial_registry_has_one_sealed_entry_for_the_manifest():
    # Break caught: a holdout can be consumed or ambiguously registered before review.
    manifest, manifest_hash = load_canonical_json(ROOT / MANIFEST_PATH, validate_manifest)
    registry, _ = load_canonical_json(ROOT / REGISTRY_PATH, validate_registry)
    entries = [entry for entry in registry["entries"] if entry["season"] == 2026]
    assert entries == [
        {
            "season": 2026,
            "manifest_sha256": manifest_hash,
            "status": "sealed",
            "sealing_commit": manifest["sealing_commit"],
            "sealed_at": manifest["sealed_at"],
        }
    ]


def test_registry_rejects_duplicate_lifecycle_keys():
    # Break caught: two lifecycle records could authorize conflicting outcomes.
    manifest, manifest_hash = load_canonical_json(ROOT / MANIFEST_PATH, validate_manifest)
    entry = {"season": 2026, "manifest_sha256": manifest_hash, "status": "sealed", "sealing_commit": manifest["sealing_commit"], "sealed_at": manifest["sealed_at"]}
    with pytest.raises(ValueError, match="duplicate"):
        validate_registry({"schema_version": 2, "attestation": "review and Git evidence; not proof of non-observation", "entries": [entry, entry]})


def test_manifest_rejects_recipe_hash_that_does_not_match_the_live_recipe():
    # Break caught: a reviewable recipe can point at a live file but bless stale bytes.
    manifest, _ = load_canonical_json(ROOT / MANIFEST_PATH, validate_manifest)
    changed = json.loads(json.dumps(manifest))
    changed["target_recipe"]["sql"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match=r"backchannel_labels\.sql.*expected.*actual"):
        validate_manifest(changed)


def test_manifest_rejects_provenance_helper_that_breaks_verified_v1_contract(monkeypatch):
    # Break caught: the named helper remains callable but no longer proves v1 provenance.
    manifest, _ = load_canonical_json(ROOT / MANIFEST_PATH, validate_manifest)
    monkeypatch.setattr(
        confirmation.label_export,
        "label_identity",
        lambda path, manifest_path=None: {"sha256": "0" * 64, "model_version": None, "verified": False, "source": None, "problems": ["broken"]},
    )
    with pytest.raises(ValueError, match=r"baseline.*expected.*actual"):
        validate_manifest(manifest)
