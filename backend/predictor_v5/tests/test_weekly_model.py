"""Tests for the weekly-sequence what-if model. Run: cd backend && python -m pytest predictor_v5/tests"""

import numpy as np
import pandas as pd
import pytest

from predictor_v5 import weekly_model as wm
from predictor_v5.tests.test_whatif_v2 import _synthetic_v2


def _weekly(seed: int = 0) -> pd.DataFrame:
    """v2's synthetic seasons with the points spread over random played weeks; late finishers gain extra."""
    rng = np.random.default_rng(seed)
    df = _synthetic_v2(seed=seed).reset_index(drop=True)
    weekly = np.zeros((len(df), wm.WEEKS))
    for i, (points, games) in enumerate(zip(df["points"], df["games"])):
        played = np.sort(rng.choice(wm.WEEKS, size=int(games), replace=False))
        shares = rng.dirichlet(np.ones(len(played)))
        weekly[i, played] = points * shares
    for w, c in enumerate(wm.WEEK_COLUMNS):
        df[c] = weekly[:, w]
    df[wm.QUARTER_COLUMNS] = wm.quarter_sums(weekly)
    shape = wm.shape_features(weekly, df["games"].to_numpy(float))
    for k, v in shape.items():
        df[k] = v
    ppg = df["points"] / df["games"]
    df["end_value"] = np.minimum(
        df["start_value"] * np.exp(0.05 * (ppg - 12) + 0.02 * df["late_form"] + rng.normal(0, 0.1, len(df))), 9999
    )
    return df


@pytest.fixture(scope="module")
def frame():
    return _weekly()


@pytest.fixture(scope="module")
def model(frame):
    return wm.WeeklyModel(label_source="test", scale_max=9999).fit(frame)


def test_weekly_matrix_pads_to_eighteen_weeks_and_ignores_stray_weeks():
    seasons = [
        {"weekly_stats": [{"week": 1, "fantasy_points": 10.5}, {"week": 18, "fantasy_points": 3}, {"week": 19, "fantasy_points": 99}]},
        {"weekly_stats": None},
    ]
    m = wm.weekly_matrix(seasons)
    assert m.shape == (2, 18)
    assert m[0, 0] == 10.5 and m[0, 17] == 3 and m[0].sum() == 13.5
    assert m[1].sum() == 0


def test_shape_features_read_the_weeks_with_points():
    weekly = np.zeros((2, wm.WEEKS))
    weekly[0, :4] = [10, 10, 10, 10]
    weekly[0, 4:8] = [30, 30, 30, 30]  # finished strong
    weekly[1, :8] = 20  # flat, every week a boom
    shape = wm.shape_features(weekly, np.array([8.0, 8.0]))
    assert shape["late_form"][0] == pytest.approx(30 - 20)
    assert shape["trend"][0] > 0 and abs(shape["trend"][1]) < 1e-9
    assert list(shape["boom_weeks"]) == [4.0, 8.0]
    assert shape["late_form"][1] == pytest.approx(0.0)


def test_feature_names_and_monotone_constraints_line_up():
    assert len(wm.feature_names()) == len(wm.monotonic()) == 7 + 4 and sum(wm.monotonic()) == 2 + 4
    assert len(wm.feature_names(sequence="weeks")) == 7 + 18 and sum(wm.monotonic(sequence="weeks")) == 2 + 18
    assert len(wm.feature_names(sequence="shape")) == 7 + 3 and sum(wm.monotonic(sequence="shape")) == 2
    assert len(wm.feature_names(4, "quarters+shape")) == len(wm.monotonic(4, "quarters+shape")) == 7 + 7 + 4
    with pytest.raises(ValueError):
        wm.feature_names(sequence="months")


def test_quarter_sums_cover_the_season_once():
    weekly = np.arange(1.0, wm.WEEKS + 1).reshape(1, -1)
    q = wm.quarter_sums(weekly)
    assert q.shape == (1, 4) and q.sum() == weekly.sum()
    assert list(q[0]) == [10.0, 26.0, 55.0, 80.0]


@pytest.mark.parametrize("week", [1, 9, 17])
def test_more_points_in_any_week_never_lowers_the_prediction(model, frame, week):
    rows = frame.head(40).copy()
    before = model.predict(rows)
    bumped = rows.copy()
    bumped[f"wk_{week}"] = bumped[f"wk_{week}"] + 15
    bumped["points"] = bumped["points"] + 15
    bumped[wm.QUARTER_COLUMNS] = wm.quarter_sums(bumped[wm.WEEK_COLUMNS].to_numpy(float))
    # absolute_error refits leaves to medians after the split, which can break the
    # constraint by a hair (v2 closes that gap with a points grid; here it is ~0.05%).
    assert np.all(model.predict(bumped) >= before * (1 - 0.002))


def test_fit_predicts_from_start_and_the_band_brackets_it(model, frame):
    pred = model.predict(frame)
    assert np.allclose(pred, np.clip(frame["start_value"] * np.exp(model.log_ratio(frame)), 0, 9999))
    band = model.predict_band(frame)
    low, high = band[min(band)], band[max(band)]
    assert np.all(low <= pred) and np.all(pred <= high)
    inside = np.mean((frame["end_value"] >= low) & (frame["end_value"] <= high))
    assert 0.7 <= inside <= 0.95
    assert model.band_season == 2024


def test_evaluate_reports_both_models_per_season(frame):
    result = wm.evaluate_weekly(frame, "test", 9999, min_train_seasons=2)
    assert [r["season"] for r in result["by_season"]] == [2022, 2023, 2024]
    for r in result["by_season"]:
        assert r["skill_v2"] > 0 and r["skill_weekly"] > 0
    assert result["overall"]["skill_weekly"] > 0.3
    assert result["model"] == "weekly[quarters]"
    raw = wm.evaluate_weekly(frame, "test", 9999, min_train_seasons=2, sequence="weeks")
    assert raw["model"] == "weekly[weeks]" and len(raw["features"]) == 7 + 18
    assert {"pred_v2", "pred_value", "band_low", "band_high"} <= set(result["predictions"].columns)


def test_value_path_joins_log_ratios_and_fills_missing_with_zero(tmp_path, frame):
    rows = frame.head(3)[["player_id", "season", "start_value"]]
    path = tmp_path / "values.csv"
    pd.DataFrame(
        {
            "player_id": [rows.player_id.iloc[0]] * 2,
            "season": [rows.season.iloc[0]] * 2,
            "week": [1, 2],
            "value": [rows.start_value.iloc[0] * 1.1, rows.start_value.iloc[0] * 0.9],
        }
    ).to_csv(path, index=False)
    out = wm.attach_value_path(frame.head(3), path, 2)
    assert out["val_wk_1"].iloc[0] == pytest.approx(np.log(1.1))
    assert out["val_wk_2"].iloc[0] == pytest.approx(np.log(0.9))
    assert out["val_wk_1"].iloc[1] == 0.0 and out["val_wk_2"].iloc[2] == 0.0


def test_weekly_leaves_the_sealed_modules_alone():
    import predictor_v5.model as sealed

    assert sealed.FEATURES == ["log_start", "age", "points", "position_code"]
    assert not hasattr(sealed, "WeeklyModel")
