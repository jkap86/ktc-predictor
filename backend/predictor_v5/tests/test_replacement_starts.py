from pathlib import Path

import pandas as pd
import pytest

from predictor_v5.replacement_starts import check_target_starts, load_starts

ROOT = Path(__file__).resolve().parents[3]
STARTS = ROOT / "backend/predictor_v5/data/start_values_2026.csv"


def test_frozen_identity_and_dates():
    starts = load_starts(STARTS, ROOT)
    assert (
        starts.attrs["sha256"]
        == "6366b85300c4f9532f0866c602c6d262baaccf6c3a84b9252a6d2e8d46c541bb"
    )
    assert starts.attrs["git_blob"] == "ead24b508102ca2a1071d07f3f665e9ea920aa96"
    assert starts.attrs["introduction"] == "c415177"
    assert (
        starts.attrs["provenance_limit"]
        == "raw start provenance not independently verified"
    )
    assert starts["bc_model_version"].dropna().eq("baseline-hl28-v1").all()
    assert (
        pd.to_datetime(starts["bc_date"].dropna()).le(pd.Timestamp("2026-09-09")).all()
    )
    ktc_dates = pd.to_datetime(starts["ktc_date"].dropna())
    assert ktc_dates.between("2026-08-26", "2026-09-09").all()


def test_start_missing_is_coverage_but_conflict_refuses():
    starts = pd.DataFrame(
        [
            {
                "season": 2026,
                "player_id": "1",
                "ktc_value": 100,
                "bc_value": 80,
                "bc_trades": 5,
            }
        ]
    )
    target = pd.DataFrame(
        [
            {"season": 2026, "player_id": "1", "start_value": 100},
            {"season": 2026, "player_id": "2", "start_value": 10},
        ]
    )
    assert check_target_starts(starts, target, "ktc")["missing"] == 1
    with pytest.raises(ValueError):
        check_target_starts(starts, target.assign(start_value=[99, 10]), "ktc")
    mfl = target.assign(start_value=[80, 10], start_trades_window=[5, 1])
    assert check_target_starts(starts, mfl, "mfl")["missing"] == 1
    with pytest.raises(ValueError):
        check_target_starts(starts, mfl.assign(start_trades_window=[6, 1]), "mfl")
    with pytest.raises(ValueError):
        check_target_starts(starts, mfl.assign(start_value=[79, 10]), "mfl")


def test_missing_coverage_reports_sorted_identities():
    starts = pd.DataFrame([{"season": 2026, "player_id": "1", "ktc_value": 100}])
    target = pd.DataFrame(
        [
            {"season": 2026, "player_id": "3", "start_value": 3},
            {"season": 2026, "player_id": "1", "start_value": 100},
            {"season": 2026, "player_id": "2", "start_value": 2},
        ]
    )
    coverage = check_target_starts(starts, target, "ktc")
    assert coverage["source"] == "ktc"
    assert coverage["missing_keys"] == [(2026, "2"), (2026, "3")]
    assert coverage["missing_count"] == 2
    assert coverage["matched_count"] == 1
