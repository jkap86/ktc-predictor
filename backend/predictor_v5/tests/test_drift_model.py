"""Tests for the drift-split what-if model. Run: cd backend && python -m pytest predictor_v5/tests"""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from predictor_v5 import drift_model as dm
from predictor_v5.tests.test_whatif_v2 import _synthetic_v2

# A market that slides a known amount each season, on top of v2's per-game pricing.
DRIFTS = {2020: -0.12, 2021: -0.08, 2022: -0.10, 2023: -0.11, 2024: -0.09}


def _drifting(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _synthetic_v2(seed=seed)
    ppg = df["points"] / df["games"]
    relative = 0.06 * (ppg - 12) + rng.normal(0, 0.1, len(df))
    df["end_value"] = np.minimum(df["start_value"] * np.exp(df["season"].map(DRIFTS) + relative), 9999)
    return df


def _expected_drifts(df: pd.DataFrame) -> dict[int, float]:
    """What each season's median log change should be: its drift plus the relative term at its median points per game."""
    ppg = df["points"] / df["games"]
    return {int(s): DRIFTS[int(s)] + 0.06 * (ppg[df["season"] == s].median() - 12) for s in sorted(df["season"].unique())}


@pytest.fixture(scope="module")
def frame():
    return _drifting()


@pytest.fixture(scope="module")
def model(frame):
    return dm.DriftSplitModel(label_source="test", scale_max=9999).fit(frame)


def test_season_drifts_are_each_seasons_median_log_change(frame):
    drifts = dm.season_drifts(frame)
    expected = _expected_drifts(frame)
    assert set(drifts) == set(DRIFTS)
    for season in DRIFTS:
        assert abs(drifts[season] - expected[season]) < 0.05, season


def test_drift_estimate_is_the_mean_or_the_last_training_season():
    drifts = {2020: -0.2, 2021: -0.1, 2022: 0.0}
    assert dm.drift_estimate(drifts, "mean") == pytest.approx(-0.1)
    assert dm.drift_estimate(drifts, "last") == 0.0
    with pytest.raises(ValueError):
        dm.drift_estimate(drifts, "median")
    with pytest.raises(ValueError):
        dm.drift_estimate({}, "mean")


def test_fit_recovers_the_drift_and_the_relative_move_separately(model, frame):
    assert model.drift == pytest.approx(np.mean(list(_expected_drifts(frame).values())), abs=0.03)
    parts = model.predict_parts(frame)
    assert np.allclose(parts["drift"], model.drift)
    # The relative part is centred: a league-median player moves with the league, not against it.
    assert abs(np.median(parts["relative"])) < 0.05
    expected = frame["start_value"].to_numpy(float) * np.exp(parts["drift"] + parts["relative"])
    assert np.allclose(model.predict(frame), np.clip(expected, 0, 9999))


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE"])
def test_prediction_never_falls_as_points_rise(model, position):
    rows = pd.DataFrame(
        {"start_value": 3000, "age": 25, "position": position, "points": np.arange(0, 452, 0.7), "games": 17.0, "prior_points": 150}
    )
    assert all(b >= a for a, b in pairwise(model.predict(rows)))


def test_band_brackets_the_median_and_covers_most_outcomes(model, frame):
    band = model.predict_band(frame)
    low, high = band[min(band)], band[max(band)]
    mid = model.predict(frame)
    assert np.all(low <= mid) and np.all(mid <= high)
    inside = np.mean((frame["end_value"] >= low) & (frame["end_value"] <= high))
    assert 0.7 <= inside <= 0.95
    assert model.band_season == 2024


def test_evaluate_reports_both_models_the_drifts_and_the_oracle(frame):
    result = dm.evaluate_drift(frame, "test", 9999, min_train_seasons=2)
    seasons = [row["season"] for row in result["by_season"]]
    assert seasons == [2022, 2023, 2024]
    expected = _expected_drifts(frame)
    for row in result["by_season"]:
        assert abs(row["drift_actual"] - expected[row["season"]]) < 0.05
        assert row["skill_drift_split"] > 0 and row["skill_v2"] > 0
        # Knowing the season's real drift can only help, up to rounding.
        assert row["skill_oracle_drift"] >= row["skill_drift_split"] - 0.01
    assert result["overall"]["skill_drift_split"] > 0.3
    assert 0.7 <= result["band"]["overall"]["inside"] <= 0.95
    preds = result["predictions"]
    assert {"pred_v2", "pred_value", "pred_oracle", "band_low", "band_high"} <= set(preds.columns)


def test_last_season_drift_differs_from_the_mean_when_seasons_differ(frame):
    mean = dm.DriftSplitModel(label_source="test", scale_max=9999, drift_how="mean").fit(frame)
    last = dm.DriftSplitModel(label_source="test", scale_max=9999, drift_how="last").fit(frame)
    assert last.drift == pytest.approx(mean.season_drift[2024])
    assert last.drift != mean.drift


def test_drift_split_leaves_the_sealed_modules_alone():
    import predictor_v5.model as sealed

    assert sealed.FEATURES == ["log_start", "age", "points", "position_code"]
    assert not hasattr(sealed, "DriftSplitModel")
