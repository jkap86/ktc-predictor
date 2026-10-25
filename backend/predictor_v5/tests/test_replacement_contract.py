import copy
import hashlib
import json

import pytest

from predictor_v5.replacement_contract import canonical_bytes, load_canonical


def test_canonical_json_and_duplicate_keys(tmp_path):
    path = tmp_path / "object.json"
    path.write_bytes(b'{"a":1,"a":2}\n')
    with pytest.raises(ValueError):
        load_canonical(path, {"a"})
    path.write_bytes(b'{"a": 1}\n')
    with pytest.raises(ValueError):
        load_canonical(path, {"a"})
    assert canonical_bytes({"z": "é", "a": 1}) == b'{"a":1,"z":"\xc3\xa9"}\n'
    path.write_bytes(b'{"a":1}\n')
    assert load_canonical(path, {"a"})[0] == {"a": 1}
    with pytest.raises(ValueError):
        load_canonical(path, {"b"})


@pytest.mark.parametrize("path", ["/absolute", "a//b", "a/../b", "a/./b", "a\\b", ""])
def test_identity_rejects_unsafe_paths(path):
    from predictor_v5.replacement_contract import validate_identity

    with pytest.raises(ValueError):
        validate_identity({"path": path, "sha256": "a" * 64})


def test_identity_rejects_uppercase_digest_and_extra_fields():
    from predictor_v5.replacement_contract import validate_identity

    validate_identity({"path": "safe/file", "sha256": "a" * 64})
    for value in (
        {"path": "safe/file", "sha256": "A" * 64},
        {"path": "safe/file", "sha256": "a" * 64, "extra": 1},
    ):
        with pytest.raises(ValueError):
            validate_identity(value)


def manifest_fixture():
    i = {"path": "source/file", "sha256": "a" * 64}
    return {
        "schema_version": 1,
        "protocol": "ktc-replacement-2026",
        "sealing_commit": "b" * 40,
        "sealed_at": "2026-01-01T00:00:00+00:00",
        "registry_path": "registry.json",
        "input_registration_path": "registration.json",
        "output_template": "backend/predictor_v5/artifacts/replacements/2026-<manifest-sha256>.json",
        "claim_template": "backend/predictor_v5/artifacts/replacements/2026-<manifest-sha256>.claim.json",
        "source_identities": [i],
        "training": {
            "archive": i,
            "ktc_member": "data/file",
            "mfl_labels": i,
            "seasons": [2021, 2022, 2023, 2024, 2025],
            "positions": ["QB", "RB", "TE", "WR"],
        },
        "starts": {
            "identity": {
                "path": "backend/predictor_v5/data/start_values_2026.csv",
                "sha256": "6366b85300c4f9532f0866c602c6d262baaccf6c3a84b9252a6d2e8d46c541bb",
            },
            "git_blob": "ead24b508102ca2a1071d07f3f665e9ea920aa96",
            "introduced_commit": "c415177",
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
                "sql": i,
                "extractor": i,
                "mapping_code": i,
                "mapping_data": i,
                "raw_columns": ["format", "ktc_id", "sleeper_id", "date", "sf_value"],
                "target_columns": [
                    "player_id",
                    "season",
                    "start_value",
                    "end_value",
                    "position",
                    "age",
                    "games",
                    "points",
                ],
                "stats_format": "training-players-zip",
            },
            "mfl": {
                "sql": i,
                "build": i,
                "target_columns": [
                    "player_id",
                    "season",
                    "start_value",
                    "end_value",
                    "position",
                    "age",
                    "games",
                    "points",
                    "start_evidence",
                    "end_evidence",
                    "start_trades_window",
                    "end_trades_window",
                    "end_value_next_board",
                    "end_evidence_next_board",
                ],
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


def test_manifest_enforces_nested_schema_and_fixed_values():
    from predictor_v5.replacement_contract import validate_manifest

    manifest = manifest_fixture()
    validate_manifest(manifest)
    for mutate in (
        lambda x: x.pop("gate"),
        lambda x: x["model"].update(extra=1),
        lambda x: x["training"].update(seasons=[2026]),
        lambda x: x["model"].update(points_grid_step=True),
    ):
        invalid = copy.deepcopy(manifest)
        mutate(invalid)
        with pytest.raises(ValueError):
            validate_manifest(invalid)


def attestation_fixture():
    return {
        "schema_version": 1,
        "season": 2026,
        "replacement_manifest_sha256": "a" * 64,
        "confirmation_manifest_sha256": "b" * 64,
        "reviewer": "reviewer",
        "evidence_commit": "c" * 40,
        "evidence_reference": "review.json",
        "reviewed_at": "2026-01-02T00:00:00+00:00",
        "recipes_committed_before_outcomes": True,
        "evidence_untouched": True,
    }


def test_joint_attestation_requires_exact_fields_and_truthful_review():
    from predictor_v5.replacement_contract import validate_joint_attestation

    value = attestation_fixture()
    validate_joint_attestation(value)
    for key, replacement in (
        ("reviewer", " "),
        ("reviewed_at", "2026-01-02"),
        ("evidence_untouched", False),
        ("extra", 1),
    ):
        bad = value | {key: replacement}
        with pytest.raises(ValueError):
            validate_joint_attestation(bad)
    del value["season"]
    with pytest.raises(ValueError):
        validate_joint_attestation(value)


def registry_fixture(status="untouched"):
    entry = {
        "season": 2026,
        "manifest_sha256": "a" * 64,
        "status": status,
        "sealing_commit": "b" * 40,
        "sealed_at": "2026-01-01T00:00:00+00:00",
    }
    if status != "sealed":
        entry.update(
            {
                "reviewer": "reviewer",
                "evidence_commit": "c" * 40,
                "evidence_reference": "review.json",
                "reviewed_at": "2026-01-02T00:00:00+00:00",
                "transition_source": "untouched" if status == "used" else "sealed",
                "joint_review": {"path": "review.json", "sha256": "d" * 64},
            }
        )
    if status == "used":
        entry.update(
            {"claim": {"path": "claim.json", "sha256": "e" * 64}, "report": None}
        )
    return {
        "schema_version": 1,
        "attestation": "review and Git evidence; not proof of non-observation",
        "entries": [entry],
    }


def test_registry_states_and_unique_keys():
    from predictor_v5.replacement_contract import validate_registry

    for status in ("sealed", "untouched", "reused", "used"):
        validate_registry(registry_fixture(status))
    value = registry_fixture()
    value["entries"].append(copy.deepcopy(value["entries"][0]))
    with pytest.raises(ValueError):
        validate_registry(value)
    for key, replacement in (
        ("transition_source", "used"),
        ("reviewed_at", "2025-01-01T00:00:00+00:00"),
        ("extra", 1),
    ):
        value = registry_fixture()
        value["entries"][0][key] = replacement
        with pytest.raises(ValueError):
            validate_registry(value)


def registration_fixture():
    identity = {"path": "source/file", "sha256": "a" * 64}
    review = copy.deepcopy(registry_fixture()["entries"][0])
    for key in ("season", "manifest_sha256", "sealing_commit", "sealed_at"):
        del review[key]
    review["registry_key"] = "2026:" + "a" * 64
    return {
        "schema_version": 1,
        "season": 2026,
        "manifest_sha256": "a" * 64,
        "ktc": {
            key: identity
            for key in (
                "raw_end",
                "target_csv",
                "target_stats",
                "target_sidecar",
                "sql",
                "extractor",
                "mapping_code",
                "mapping_data",
            )
        },
        "mfl": {
            key: identity
            for key in ("target_csv", "target_stats", "target_sidecar", "sql", "build")
        },
        "holdout_attestation": review,
    }


def test_registration_exact_fields_and_registry_key():
    from predictor_v5.replacement_contract import validate_registration

    value = registration_fixture()
    validate_registration(value)
    for change in (
        lambda x: x["ktc"].pop("target_sidecar"),
        lambda x: x["mfl"].update(extra=1),
        lambda x: x["holdout_attestation"].update(registry_key="wrong"),
    ):
        bad = copy.deepcopy(value)
        change(bad)
        with pytest.raises(ValueError):
            validate_registration(bad)


def test_cross_references_reject_changes_and_symlink_escape(tmp_path):
    from predictor_v5.replacement_contract import validate_references

    def digest(value):
        return hashlib.sha256(
            (
                json.dumps(
                    value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
                + "\n"
            ).encode()
        ).hexdigest()

    manifest = manifest_fixture()
    manifest_hash = digest(manifest)
    attestation = attestation_fixture()
    attestation["replacement_manifest_sha256"] = manifest_hash
    identity = {"path": "review.json", "sha256": digest(attestation)}
    registry = registry_fixture()
    entry = registry["entries"][0]
    entry.update(manifest_sha256=manifest_hash, joint_review=identity)
    registration = registration_fixture()
    registration["manifest_sha256"] = manifest_hash
    registration["holdout_attestation"].update(
        registry_key="2026:" + manifest_hash, joint_review=identity
    )

    def check(m=manifest, r=registry, reg=registration, a=attestation, i=identity):
        return validate_references(
            m,
            r,
            reg,
            a,
            manifest_sha256=manifest_hash,
            attestation_identity=i,
            root=tmp_path,
        )

    assert check() == entry
    for key, replacement in (
        ("status", "reused"),
        ("reviewer", "other"),
        ("evidence_commit", "f" * 40),
        ("reviewed_at", "2026-01-03T00:00:00+00:00"),
    ):
        changed = copy.deepcopy(registry)
        changed["entries"][0][key] = replacement
        with pytest.raises(ValueError):
            check(r=changed)
    changed = copy.deepcopy(registration)
    changed["ktc"]["sql"] = {"path": "other", "sha256": "a" * 64}
    with pytest.raises(ValueError):
        check(reg=changed)
    (tmp_path / "source").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        check()
