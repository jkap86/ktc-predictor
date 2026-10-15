"""Tests for the season tool. Run: cd backend && python -m pytest predictor_v5/tests"""

from datetime import date

import pandas as pd
import pytest

from predictor_v5.model import WhatIfModel
from predictor_v5.season import SeasonTool, age_on
from predictor_v5.tests.test_predictor_v5 import _synthetic


@pytest.fixture(scope="module")
def tool(tmp_path_factory):
    path = tmp_path_factory.mktemp("sv") / "start_values_2026.csv"
    pd.DataFrame(
        {
            "player_id": ["1", "2", "3", "4"],
            "name": ["Alpha Wideout", "Beta Back", "Gamma Passer", "Delta Kicker"],
            "position": ["WR", "RB", "QB", "K"],
            "team": ["CIN", None, "BUF", "DAL"],
            "birth_date": ["2000-09-10", "2001-01-01", None, "1990-01-01"],
            "bc_value": [6000, 3000, 5000, 100],
            "bc_trades": [40, 2, 12, 1],
            "bc_date": ["2026-09-09"] * 4,
            "bc_model_version": ["v1"] * 4,
            "ktc_value": [7000, None, 6000, 500],
            "ktc_date": ["2026-09-08", None, "2026-09-08", "2026-09-08"],
        }
    ).to_csv(path, index=False)
    tool = SeasonTool(season=2026, start_values_path=path)
    model = WhatIfModel(label_source="test", scale_max=9999).fit(_synthetic())
    tool.models = {"backchannel": model, "ktc": model}
    return tool


def test_age_is_whole_years_on_the_cutoff():
    assert age_on("2000-09-10", date(2026, 9, 9)) == 25.0
    assert age_on("2000-09-09", date(2026, 9, 9)) == 26.0
    assert age_on(None, date(2026, 9, 9)) is None


def test_players_skip_non_fantasy_positions(tool):
    assert sorted(tool.players["position"]) == ["QB", "RB", "WR"]


def test_find_by_id_exact_name_or_substring(tool):
    assert [p["name"] for p in tool.find("2")] == ["Beta Back"]
    assert [p["name"] for p in tool.find("alpha wideout")] == ["Alpha Wideout"]
    assert {p["name"] for p in tool.find("a")} == {"Alpha Wideout", "Beta Back", "Gamma Passer"}


def test_predict_uses_each_sources_own_start_value(tool):
    out = tool.predict("1", [100, 300])
    assert out["age"] == 25.0 and out["cutoff"] == "2026-09-09"
    for name, start in (("backchannel", 6000), ("ktc", 7000)):
        preds = out[name]["predictions"]
        assert [p["points"] for p in preds] == [100, 300]
        assert preds[0]["value"] <= preds[1]["value"]
        assert all(p["change"] == p["value"] - start for p in preds)
    assert out["backchannel"]["note"] is None


def test_predict_flags_thin_backchannel_and_missing_ktc(tool):
    out = tool.predict("2", [200])
    assert out["backchannel"]["predictions"] and "2 trades" in out["backchannel"]["note"]
    assert out["ktc"]["predictions"] is None and "No pre-season KTC value" in out["ktc"]["note"]


def test_predict_needs_an_age_unless_given(tool):
    assert tool.predict("3", [250])["backchannel"]["predictions"] is None
    assert tool.predict("3", [250], age=27)["backchannel"]["predictions"]


def test_unknown_player_raises(tool):
    with pytest.raises(KeyError):
        tool.predict("999", [100])
