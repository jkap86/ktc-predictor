"""Tests for the v2 what-if model. Run: cd backend && python -m pytest predictor_v5/tests"""

import io
import json
import zipfile
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from predictor_v5 import whatif_v2 as v2
from predictor_v5.tests.test_predictor_v5 import _synthetic


def _synthetic_v2(n_per_season: int = 120, seasons=range(2020, 2025), seed: int = 0) -> pd.DataFrame:
    """v1's synthetic data plus games and prior points, with value driven by points per game."""
    rng = np.random.default_rng(seed)
    df = _synthetic(n_per_season, seasons, seed)
    df["games"] = rng.integers(4, 18, len(df)).astype(float)
    df["prior_points"] = rng.uniform(0, 300, len(df))
    ppg = df["points"] / df["games"]
    df["end_value"] = df["start_value"] * np.exp(0.06 * (ppg - 12) + rng.normal(0, 0.1, len(df)))
    return df


@pytest.fixture(scope="module")
def model():
    return v2.WhatIfModelV2(label_source="test", scale_max=9999).fit(_synthetic_v2())


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE"])
def test_prediction_never_falls_as_points_rise_at_fixed_games(model, position):
    points = np.arange(0, 452, 0.7)
    for games in (8.0, 17.0):
        for start in (1000, 8000):
            rows = pd.DataFrame(
                {"start_value": start, "age": 25, "position": position, "points": points, "games": games, "prior_points": 150}
            )
            values = model.predict(rows)
            assert all(b >= a for a, b in pairwise(values)), (start, games)


def test_fewer_games_for_the_same_points_is_worth_more(model):
    # The synthetic market prices points per game, so 200 points in 10 games beats 200 in 17.
    few = model.predict_one(start_value=4000, age=25, position="WR", points=200, games=10)
    many = model.predict_one(start_value=4000, age=25, position="WR", points=200, games=17)
    assert few > many


def test_predict_one_defaults_to_a_full_season(model):
    explicit = model.predict_one(start_value=4000, age=25, position="RB", points=220, games=v2.FULL_SEASON_GAMES)
    assert model.predict_one(start_value=4000, age=25, position="RB", points=220) == explicit
    with pytest.raises(ValueError):
        model.predict_one(start_value=4000, age=25, position="RB", points=220, games=-1)


def test_points_per_game_is_zero_without_games():
    assert list(v2.points_per_game([0, 170, 34], [0, 17, 2])) == [0.0, 10.0, 17.0]


def test_folds_beat_no_change_and_train_only_on_earlier_seasons(monkeypatch):
    seen = []
    real_fit = v2.WhatIfModelV2.fit

    def spy(self, df):
        seen.append(set(df["season"]))
        return real_fit(self, df)

    monkeypatch.setattr(v2.WhatIfModelV2, "fit", spy)
    result = v2.evaluate_v2(_synthetic_v2(), "test", 9999)
    assert result["beats_no_change"] and result["overall"]["skill"] > 0
    assert result["features"] == v2.FEATURES_V2
    test_seasons = sorted(result["by_season"])
    assert test_seasons == [2021, 2022, 2023, 2024]
    for train_seasons, test_season in zip(seen, test_seasons):
        assert max(train_seasons) < test_season


def _archive(tmp_path, players):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("training-data.json", json.dumps({"players": players}))
    path = tmp_path / "training-data.zip"
    path.write_bytes(buf.getvalue())
    return path


def test_prior_points_prefer_the_archives_own_previous_season(tmp_path):
    def season(year, points, games=17, years_exp=5, prior_year_fp=0.0):
        return {
            "year": year,
            "age": 20 + years_exp,
            "years_exp": years_exp,
            "fantasy_points": points,
            "games_played": games,
            "prior_year_fp": prior_year_fp,
        }

    archive = _archive(
        tmp_path,
        [
            {
                "player_id": 1,
                "name": "Vet",
                "position": "WR",
                # 2020's prior comes from the field (no 2019 row); 2021's from the 2020 row, not the field.
                "seasons": [season(2020, 200.0, prior_year_fp=180.0), season(2021, 250.0, prior_year_fp=999.0)],
            },
            {
                "player_id": 2,
                "name": "Rookie",
                "position": "RB",
                "seasons": [season(2020, 0.0, games=0, years_exp=0), season(2021, 120.0, games=15, years_exp=1)],
            },
            {"player_id": 3, "name": "Kicker", "position": "K", "seasons": [season(2021, 140.0)]},
        ],
    )
    stats = v2.load_season_stats_v2(archive)
    rows = {(r.player_id, r.season): r for r in stats.itertuples()}
    assert set(rows) == {("1", 2020), ("1", 2021), ("2", 2021)}  # devy row and kicker dropped
    assert rows[("1", 2020)].prior_points == 180.0
    assert rows[("1", 2021)].prior_points == 200.0
    assert rows[("2", 2021)].prior_points == 0.0

    assert v2.prior_points_for(2022, v2.read_season_stats_v2(archive)) == {"1": 250.0, "2": 120.0}
    assert v2.prior_points_for(2021, v2.read_season_stats_v2(archive)) == {"1": 200.0, "2": 0.0}


def test_v2_leaves_the_sealed_v1_model_alone():
    from predictor_v5 import model as v1

    assert v1.FEATURES == ["log_start", "age", "points", "position_code"]
    assert v1.MONOTONIC == [0, 0, 1, 0]
    assert len(v2.FEATURES_V2) == len(v2.MONOTONIC_V2) == 7
