import numpy as np
import pandas as pd
import pytest

from predictor_v5.short_window import FEATURES
from predictor_v5.window_tool import WindowModel, WindowTool


class _Linear:
    """log change = 0.002 * window points - 0.05: rising in points, deterministic."""

    def predict(self, X):
        col = FEATURES["full"].index("win_points")
        return 0.002 * X[:, col] - 0.05


def _model(source, games):
    return WindowModel(source=source, games=games, estimator=_Linear(), train_seasons=[2024, 2025],
                       band_offsets={0.1: -0.2, 0.9: 0.2}, band_season=2025)


def _inputs(tmp_path, bc_trades=80, played_through=11):
    rows = []
    for w in range(1, 20):
        rows.append({
            "player_id": "1", "name": "Test Player", "position": "WR", "team": "CIN", "birth_date": "2000-03-01",
            "week": w, "tuesday": f"2026-W{w}", "ktc_value": 8000 + 10 * w if w <= 12 else None,
            "ktc_date": f"2026-K{w}" if w <= 12 else None,
            "bc_value": 2500.0 if w <= 5 else None, "bc_trades": bc_trades if w <= 5 else None,
            "points": 20.0 if w <= played_through else None, "played": 1.0 if w <= played_through else None,
        })
    path = tmp_path / "inputs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    tool = WindowTool(season=2026, inputs_path=path)
    tool.__dict__["prior_points"] = {"1": 300.0}
    for s in ("ktc", "backchannel"):
        for g in (3, 6):
            tool._models[(s, g)] = _model(s, g)
    return tool


def test_default_window_is_the_latest_finished_one_with_actual_points(tmp_path):
    tool = _inputs(tmp_path)
    r = tool.predict("1", games=3)
    assert (r["start_week"], r["last_game_week"], r["finished"]) == (9, 11, True)
    assert r["actual_points"] == 60.0 and r["window_games"] == 3
    k = r["ktc"]
    assert k["start_value"] == 8090 and k["actual_value"] == 8120 and k["actual_change"] == 30
    assert k["predictions"][0]["value"] == round(8090 * np.exp(0.002 * 60 - 0.05))
    # Pace: 20 a game so far, shrunk toward last season's 300/17 with 3 games' weight.
    pace = (20 * 8 + 3 * 300 / 17) / 11
    assert k["pace"]["ppg"] == round(pace, 2)


def test_backchannel_needs_a_value_and_50_trades(tmp_path):
    r = _inputs(tmp_path).predict("1", games=3)
    assert r["backchannel"]["predictions"] is None and "No Backchannel value" in r["backchannel"]["note"]
    thin = _inputs(tmp_path, bc_trades=20).predict("1", games=3, start_week=2)
    assert thin["backchannel"]["predictions"] is None and "20 trades" in thin["backchannel"]["note"]
    deep = _inputs(tmp_path, bc_trades=80).predict("1", games=3, start_week=2)
    assert deep["backchannel"]["predictions"][0]["value"] > 0


def test_unfinished_window_needs_points_and_sweeps_monotone(tmp_path):
    tool = _inputs(tmp_path)
    with pytest.raises(ValueError, match="what-if"):
        tool.predict("1", games=6, start_week=10)
    r = tool.predict("1", games=6, start_week=10, points=[30, 90, 150])
    values = [p["value"] for p in r["ktc"]["predictions"]]
    assert values == sorted(values) and r["ktc"]["actual_value"] is None
    with pytest.raises(ValueError, match="haven't been played"):
        tool.predict("1", games=3, start_week=14, points=[50])


def test_late_season_window_carries_the_note(tmp_path):
    tool = _inputs(tmp_path, played_through=15)
    r = tool.predict("1", games=3, start_week=14, points=[50])
    assert r["late_season"] and "Late-season" in r["note"]
    assert tool.predict("1", games=3, start_week=11)["note"] is None


def test_rejects_other_window_lengths(tmp_path):
    with pytest.raises(ValueError, match="games must be one of"):
        _inputs(tmp_path).predict("1", games=4)
