import numpy as np
import pandas as pd

from predictor_v5.short_window import FEATURES, build_windows, fit, predict, reversal


def _pts(weeks=18):
    rows = []
    for pid, ppg in (("a", 20.0), ("b", 5.0)):
        for w in range(1, weeks + 1):
            rows.append({"player_id": pid, "name": pid, "position": "WR", "season": 2023, "age": 25.0,
                         "season_points": ppg * weeks, "season_games": weeks, "week": w,
                         "points": ppg, "played": 1, "prior_points": 100.0})
    return pd.DataFrame(rows)


def _values():
    rows = []
    for pid, start in (("a", 1000.0), ("b", 1000.0)):
        for w in range(1, 20):
            rows.append({"player_id": pid, "season": 2023, "week": w, "value": start * (1.02 ** w if pid == "a" else 0.98 ** w),
                         "trades_window": 10 if w != 5 else 2, "trades_recent": 3, "uncertainty": 1.0})
    return pd.DataFrame(rows)


def test_windows_use_points_inside_the_window_and_values_at_both_ends():
    df = build_windows(_pts(), _values(), k=3, min_trades=5, last_value_week=None)
    a = df[(df["player_id"] == "a") & (df["week"] == 6)].iloc[0]
    assert a["win_points"] == 60.0 and a["win_games"] == 3 and a["win_ppg"] == 20.0
    assert a["std_games"] == 5 and a["recent_ppg"] == 20.0
    assert np.isclose(a["end_value"] / a["start_value"], 1.02 ** 3)
    assert np.isclose(a["momentum"], np.log(1.02 ** 3))
    # A window touching the 2-trade week (start 5, or ending on 5 from start 2) is dropped.
    assert not ((df["week"] == 5) | (df["week"] == 2)).any()
    # Last window covers weeks 16-18 and ends on the Tuesday after week 18.
    assert df["week"].max() == 16


def test_ktc_style_cap_ends_windows_before_week_18():
    df = build_windows(_pts(), _values(), k=3, min_trades=0, last_value_week=18)
    assert df["week"].max() == 15


def test_window_points_are_monotone_in_the_model():
    df = build_windows(_pts(), _values(), k=3, min_trades=0, last_value_week=None)
    df = pd.concat([df.assign(season=s) for s in (2021, 2022)])
    est = fit(df, FEATURES["full"])
    lo, hi = df.head(5).copy(), df.head(5).copy()
    lo["win_points"], lo["win_ppg"] = 0.0, 0.0
    hi["win_points"], hi["win_ppg"] = 90.0, 30.0
    assert (predict(est, hi, FEATURES["full"]) >= predict(est, lo, FEATURES["full"])).all()


def test_reversal_reads_negative_covariance_as_noise():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(200):
        truth = np.cumsum(rng.normal(0, 0.01, 19))
        obs = truth + rng.normal(0, 0.1, 19)  # measurement noise dominates
        for w in range(1, 17, 3):
            rows.append({"player_id": str(p), "season": 2023, "week": w, "k": 3, "log_change": obs[w + 2] - obs[w - 1]})
    r = reversal(pd.DataFrame(rows))
    assert r["corr_next"] < -0.3 and r["noise_share_lower_bound"] > 0.6
