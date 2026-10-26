import math
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from predictor_v5.movement_evaluation.contract import (
    BASE_COMMIT,
    CALIBRATION_EDGES,
    FOLDS,
    INPUT_IDENTITIES,
    POSITIONS,
    SCALE_MAX,
    TARGET_ID,
    ContractRefusal,
    canonical_bytes,
    contract_payload,
    key_sha256,
    sorted_keys,
    validate_ktc_mode,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTECTED_BASE = "c4270804379c26bcb340f32a07e507153b4c7342"
PROTECTED_PATHS = (
    "backend/data/training-data.zip",
    "backend/predictor_v5/__init__.py",
    "backend/predictor_v5/__main__.py",
    "backend/predictor_v5/confirmation.py",
    "backend/predictor_v5/data.py",
    "backend/predictor_v5/data/backchannel_labels_2019_2025.csv",
    "backend/predictor_v5/data/backchannel_labels_2019_2025_mfl.csv",
    "backend/predictor_v5/data/backchannel_labels_2019_2025_mfl.provenance.json",
    "backend/predictor_v5/data/ktc_replacement_2026_holdout_registry.json",
    "backend/predictor_v5/data/ktc_replacement_2026_manifest.json",
    "backend/predictor_v5/data/ktc_replacement_mapping_snapshot.csv",
    "backend/predictor_v5/data/label_manifest.json",
    "backend/predictor_v5/data/model_2026_confirmation_manifest.json",
    "backend/predictor_v5/data/model_2026_holdout_registry.json",
    "backend/predictor_v5/data/start_values_2026.csv",
    "backend/predictor_v5/evaluate.py",
    "backend/predictor_v5/ktc_replacement_2026.py",
    "backend/predictor_v5/label_export.py",
    "backend/predictor_v5/labels.py",
    "backend/predictor_v5/model.py",
    "backend/predictor_v5/replacement_cohort.py",
    "backend/predictor_v5/replacement_contract.py",
    "backend/predictor_v5/replacement_ktc.py",
    "backend/predictor_v5/replacement_lifecycle.py",
    "backend/predictor_v5/replacement_metrics.py",
    "backend/predictor_v5/replacement_mfl.py",
    "backend/predictor_v5/replacement_report.py",
    "backend/predictor_v5/replacement_seal.py",
    "backend/predictor_v5/replacement_starts.py",
    "backend/predictor_v5/sql/backchannel_labels.sql",
    "backend/predictor_v5/sql/ktc_replacement_end.sql",
)


def test_fixed_folds_and_contract_identity_are_pinned():
    assert [(f.test_season, f.train_seasons) for f in FOLDS] == [
        (2021, (2020,)),
        (2022, (2020, 2021)),
        (2023, (2020, 2021, 2022)),
        (2024, (2020, 2021, 2022, 2023)),
        (2025, (2020, 2021, 2022, 2023, 2024)),
    ]
    assert FOLDS[0].short_history is True
    assert all(not fold.short_history for fold in FOLDS[1:])
    payload = contract_payload("off")
    assert payload["target"]["id"] == TARGET_ID
    assert payload["bootstrap"] == {
        "seed": 20261025,
        "resamples": 2000,
        "percentiles": [2.5, 97.5],
    }
    assert payload["base_commit"] == BASE_COMMIT


def test_target_dates_generations_model_and_metric_constants_are_exact():
    payload = contract_payload("off")
    assert payload["target"]["columns"] == {"start": "start_value", "end": "end_value"}
    assert payload["target"]["dates"] == {
        2020: ["2020-09-10", "2021-01-03"],
        2021: ["2021-09-09", "2022-01-09"],
        2022: ["2022-09-08", "2023-01-08"],
        2023: ["2023-09-07", "2024-01-07"],
        2024: ["2024-09-05", "2025-01-05"],
        2025: ["2025-09-04", "2026-01-04"],
    }
    assert payload["target"]["generations"] == {
        2020: {"start": {"id": 9381, "revision": 3}, "end": {"id": 9496, "revision": 3}},
        2021: {"start": {"id": 9719, "revision": 3}, "end": {"id": 9841, "revision": 3}},
        2022: {"start": {"id": 10026, "revision": 3}, "end": {"id": 10148, "revision": 3}},
        2023: {"start": {"id": 10350, "revision": 3}, "end": {"id": 10472, "revision": 3}},
        2024: {"start": {"id": 10665, "revision": 3}, "end": {"id": 10787, "revision": 3}},
        2025: {"start": {"id": 10983, "revision": 3}, "end": {"id": 11105, "revision": 3}},
    }
    assert POSITIONS == ("QB", "RB", "TE", "WR")
    assert SCALE_MAX == 10000.0
    assert payload["metrics"]["flat_band"] == math.log(1.10)
    assert payload["metrics"]["large_move_band"] == math.log(1.50)
    assert CALIBRATION_EDGES == (
        -math.log(2), -math.log(1.25), -math.log(1.10), 0.0, math.log(1.10), math.log(1.25), math.log(2)
    )
    assert payload["models"]["production"]["features"] == ("log_start", "age", "points", "position_code")
    assert payload["models"]["production"]["hgb_params"] == {
        "loss": "absolute_error", "max_iter": 300, "learning_rate": 0.05,
        "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 1.0,
        "random_state": 0,
    }
    assert payload["models"]["production"]["monotonic"] == (0, 0, 1, 0)
    assert payload["models"]["production"]["points_grid_step"] == 1.0
    assert payload["models"]["production"]["early_stopping"] == "auto"
    assert payload["models"]["production"]["early_stopping_note"] == "inactive for these small folds"


def test_pinned_input_and_source_hashes_are_present():
    assert INPUT_IDENTITIES == {
        "labels": {
            "path": "backend/predictor_v5/data/backchannel_labels_2019_2025_mfl.csv",
            "sha256": "3e57be0b6e6ccb8f626a8b4ee1e726eaad75f635174d147514c3aa8ccf3c8e89",
        },
        "provenance": {
            "path": "backend/predictor_v5/data/backchannel_labels_2019_2025_mfl.provenance.json",
            "sha256": "5258f1247b982b371e96125aa122d58f8afae3738b3e24d4fa39cacc05ee709a",
        },
        "archive": {
            "path": "backend/data/training-data.zip",
            "sha256": "b67e7b03e600ae9ad83b5b179b416f273b7e169c8d47eff7345ef257cb034a86",
            "member": "training-data.json",
            "member_sha256": "6dfa1e3eaed486d5a37880217c74fb3b139f19504aedb58b39a38013466a5388",
        },
        "model": {"path": "backend/predictor_v5/model.py", "sha256": "01674fa50df69a8e27f221f46eafa2e55b459df9e74c2c2357d6e03e733cade1"},
    }


def test_contract_payload_is_canonical_json_serializable():
    assert canonical_bytes(contract_payload("off"))


@pytest.mark.parametrize("value", [True, False, "2026", 2026, "on", "", None])
def test_validate_ktc_mode_rejects_everything_except_modes(value):
    with pytest.raises(ContractRefusal):
        validate_ktc_mode(value)


def test_validate_ktc_mode_accepts_only_off_and_archive():
    assert validate_ktc_mode("off") == "off"
    assert validate_ktc_mode("archive") == "archive"


def test_key_sorting_and_hashing_are_canonical_and_refuse_duplicates_or_2026():
    frame = pd.DataFrame({"season": [2021, 2020], "player_id": ["b", "a"]})
    assert sorted_keys(frame) == ((2020, "a"), (2021, "b"))
    first = key_sha256([(2021, "b"), (2020, "a")])
    second = key_sha256([(2020, "a"), (2021, "b")])
    assert first == second
    with pytest.raises(ContractRefusal):
        key_sha256([(2020, "a"), (2020, "a")])
    with pytest.raises(ContractRefusal):
        key_sha256([(2026, "a")])


def test_all_protected_paths_match_base_bytes_and_branch_diff_excludes_them() -> None:
    protected = set(PROTECTED_PATHS)
    for relative in PROTECTED_PATHS:
        expected = subprocess.run(
            ["git", "show", f"{PROTECTED_BASE}:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert (REPO_ROOT / relative).read_bytes() == expected, relative

    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{PROTECTED_BASE}...HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert protected.isdisjoint(changed)
