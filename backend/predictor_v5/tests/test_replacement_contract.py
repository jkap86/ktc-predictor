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


def cross_fixture():
    manifest = manifest_fixture()
    identities = {
        "ktc_sql": {"path": "recipes/ktc.sql", "sha256": "1" * 64},
        "extractor": {"path": "recipes/extractor.py", "sha256": "2" * 64},
        "mapping_code": {"path": "recipes/mapping.py", "sha256": "3" * 64},
        "mapping_data": {"path": "mappings/mapping.csv", "sha256": "4" * 64},
        "mfl_sql": {"path": "recipes/mfl.sql", "sha256": "5" * 64},
        "build": {"path": "recipes/build.py", "sha256": "6" * 64},
    }
    for key in ("extractor", "mapping_code", "mapping_data"):
        manifest["recipes"]["ktc"][key] = copy.deepcopy(identities[key])
    manifest["recipes"]["ktc"]["sql"] = copy.deepcopy(identities["ktc_sql"])
    manifest["recipes"]["mfl"]["sql"] = copy.deepcopy(identities["mfl_sql"])
    manifest["recipes"]["mfl"]["build"] = copy.deepcopy(identities["build"])
    manifest["source_identities"] = sorted(
        [
            copy.deepcopy(identities[k])
            for k in ("ktc_sql", "extractor", "mapping_code", "mfl_sql", "build")
        ],
        key=lambda i: i["path"],
    )
    manifest["training"]["archive"] = {
        "path": "training/archive.zip",
        "sha256": "7" * 64,
    }
    manifest["training"]["mfl_labels"] = {
        "path": "training/labels.csv",
        "sha256": "8" * 64,
    }
    digest = lambda v: hashlib.sha256(
        (
            json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n"
        ).encode()
    ).hexdigest()
    manifest_hash = digest(manifest)
    attestation = attestation_fixture()
    attestation["replacement_manifest_sha256"] = manifest_hash
    identity = {"path": "review.json", "sha256": digest(attestation)}
    registry = registry_fixture()
    registry["entries"][0].update(
        manifest_sha256=manifest_hash, joint_review=copy.deepcopy(identity)
    )
    registration = registration_fixture()
    registration["manifest_sha256"] = manifest_hash
    registration["holdout_attestation"].update(
        registry_key="2026:" + manifest_hash, joint_review=copy.deepcopy(identity)
    )
    for name, keys in (
        ("ktc", ("sql", "extractor", "mapping_code", "mapping_data")),
        ("mfl", ("sql", "build")),
    ):
        for key in keys:
            registration[name][key] = copy.deepcopy(manifest["recipes"][name][key])
    for name, keys in (
        ("ktc", ("raw_end", "target_csv", "target_stats", "target_sidecar")),
        ("mfl", ("target_csv", "target_stats", "target_sidecar")),
    ):
        for n, key in enumerate(keys):
            registration[name][key] = {
                "path": f"outcomes/{name}-{key}",
                "sha256": str(n + 1) * 64,
            }
    return manifest, registry, registration, attestation, identity, manifest_hash


def call_references(values, root):
    from predictor_v5.replacement_contract import validate_references

    manifest, registry, registration, attestation, identity, digest = values
    return validate_references(
        manifest,
        registry,
        registration,
        attestation,
        manifest_sha256=digest,
        attestation_identity=identity,
        root=root,
    )


@pytest.mark.parametrize("path", ["bad\x00name", "a/\x00b"])
def test_nul_paths_rejected_without_filesystem_resolution(path):
    from predictor_v5.replacement_contract import validate_identity, validate_manifest

    with pytest.raises(ValueError):
        validate_identity({"path": path, "sha256": "a" * 64})
    manifest = manifest_fixture()
    manifest["training"]["ktc_member"] = path
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_archive_member_nul_rejected_by_manifest():
    from predictor_v5.replacement_contract import validate_manifest

    manifest = manifest_fixture()
    manifest["training"]["ktc_member"] = "archive/\x00member"
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_symlink_loop_resolution_is_value_error(tmp_path):
    values = cross_fixture()
    (tmp_path / "recipes").symlink_to("recipes")
    with pytest.raises(ValueError):
        call_references(values, tmp_path)


def test_archive_member_not_resolved_in_checkout(tmp_path):
    values = cross_fixture()
    (tmp_path / "data").symlink_to(tmp_path.parent, target_is_directory=True)
    assert call_references(values, tmp_path) == values[1]["entries"][0]


def object_locations(value, prefix=()):
    if isinstance(value, dict):
        yield prefix
        for key, item in value.items():
            yield from object_locations(item, (*prefix, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from object_locations(item, (*prefix, index))


def at(value, path):
    for key in path:
        value = value[key]
    return value


@pytest.mark.parametrize(
    "kind",
    [
        "manifest",
        "attestation",
        "registration",
        "sealed",
        "untouched",
        "reused",
        "used",
    ],
)
def test_each_nested_object_rejects_missing_and_extra_keys(kind):
    from predictor_v5.replacement_contract import (
        validate_joint_attestation,
        validate_manifest,
        validate_registration,
        validate_registry,
    )

    fixtures = {
        "manifest": (cross_fixture()[0], validate_manifest),
        "attestation": (attestation_fixture(), validate_joint_attestation),
        "registration": (cross_fixture()[2], validate_registration),
        **{
            state: (registry_fixture(state), validate_registry)
            for state in ("sealed", "untouched", "reused", "used")
        },
    }
    value, validate = fixtures[kind]
    validate(value)
    for location in object_locations(value):
        original = at(value, location)
        for key in original:
            changed = copy.deepcopy(value)
            del at(changed, location)[key]
            with pytest.raises(ValueError):
                validate(changed)
        changed = copy.deepcopy(value)
        at(changed, location)["unexpected"] = "x"
        with pytest.raises(ValueError):
            validate(changed)


@pytest.mark.parametrize("status", ["sealed", "used", "reused"])
def test_cross_references_reject_ineligible_valid_states(tmp_path, status):
    values = list(cross_fixture())
    entry = registry_fixture(status)["entries"][0]
    entry["manifest_sha256"] = values[5]
    if status != "sealed":
        entry["joint_review"] = copy.deepcopy(values[4])
    values[1]["entries"] = [entry]
    with pytest.raises(ValueError):
        call_references(values, tmp_path)


@pytest.mark.parametrize(
    "component,path,replacement",
    [
        (5, (), "f" * 64),
        (1, ("entries", 0, "manifest_sha256"), "f" * 64),
        (2, ("manifest_sha256",), "f" * 64),
        (3, ("replacement_manifest_sha256",), "f" * 64),
        (4, ("sha256",), "f" * 64),
        (4, ("path",), "elsewhere.json"),
        (1, ("entries", 0, "joint_review", "sha256"), "f" * 64),
        (1, ("entries", 0, "joint_review", "path"), "elsewhere.json"),
        (2, ("holdout_attestation", "joint_review", "sha256"), "f" * 64),
        (2, ("holdout_attestation", "joint_review", "path"), "elsewhere.json"),
        (1, ("entries", 0, "evidence_reference"), "elsewhere.json"),
        (2, ("holdout_attestation", "evidence_reference"), "elsewhere.json"),
        (2, ("holdout_attestation", "reviewer"), "different reviewer"),
        (2, ("holdout_attestation", "evidence_commit"), "f" * 40),
        (2, ("holdout_attestation", "reviewed_at"), "2026-01-03T00:00:00+00:00"),
        (1, ("entries", 0, "transition_source"), "untouched"),
        (2, ("holdout_attestation", "transition_source"), "untouched"),
    ],
)
def test_cross_reference_mutations_are_rejected(tmp_path, component, path, replacement):
    values = list(cross_fixture())
    if path:
        at(values[component], path[:-1])[path[-1]] = replacement
    else:
        values[component] = replacement
    with pytest.raises(ValueError):
        call_references(values, tmp_path)


@pytest.mark.parametrize(
    "name,key",
    [
        ("ktc", "sql"),
        ("ktc", "extractor"),
        ("ktc", "mapping_code"),
        ("ktc", "mapping_data"),
        ("mfl", "sql"),
        ("mfl", "build"),
    ],
)
@pytest.mark.parametrize(
    "field,replacement", [("path", "other/file"), ("sha256", "f" * 64)]
)
def test_each_recipe_identity_binds_path_and_hash(
    tmp_path, name, key, field, replacement
):
    values = cross_fixture()
    values[2][name][key][field] = replacement
    with pytest.raises(ValueError):
        call_references(values, tmp_path)


def test_cross_reference_never_opens_registered_outcomes(tmp_path, monkeypatch):
    import builtins
    from pathlib import Path

    values = cross_fixture()

    def forbid(*args, **kwargs):
        raise AssertionError("validation must not open files")

    monkeypatch.setattr(Path, "open", forbid)
    monkeypatch.setattr(Path, "read_bytes", forbid)
    monkeypatch.setattr(builtins, "open", forbid)
    assert call_references(values, tmp_path) == values[1]["entries"][0]


@pytest.mark.parametrize(
    "mutation",
    [
        "attestation_manifest",
        "attestation_evidence",
        "joint_path",
        "registration_manifest",
    ],
)
def test_independent_reference_mismatches_after_rebinding(tmp_path, mutation):
    values = list(cross_fixture())
    _, registry, registration, attestation, identity, _ = values
    if mutation == "registration_manifest":
        registration["manifest_sha256"] = "f" * 64
        registration["holdout_attestation"]["registry_key"] = "2026:" + "f" * 64
    elif mutation == "joint_path":
        identity["path"] = "elsewhere.json"
        registry["entries"][0]["joint_review"]["path"] = "elsewhere.json"
        registration["holdout_attestation"]["joint_review"]["path"] = "elsewhere.json"
    else:
        if mutation == "attestation_manifest":
            attestation["replacement_manifest_sha256"] = "f" * 64
        else:
            attestation["evidence_reference"] = "elsewhere.json"
        digest = hashlib.sha256(
            (
                json.dumps(
                    attestation,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            ).encode()
        ).hexdigest()
        identity["sha256"] = digest
        registry["entries"][0]["joint_review"]["sha256"] = digest
        registration["holdout_attestation"]["joint_review"]["sha256"] = digest
    with pytest.raises(ValueError):
        call_references(values, tmp_path)
