"""Tests for the v5 what-if predictor. Run: cd backend && python -m pytest predictor_v5/tests"""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from predictor_v5 import evaluate as ev
from predictor_v5.data import build_dataset
from predictor_v5.labels import CsvLabels, get_label_source
from predictor_v5.model import WhatIfModel


def _synthetic(n_per_season: int = 120, seasons=range(2020, 2025), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for i in range(n_per_season):
            pos = ["QB", "RB", "WR", "TE"][i % 4]
            start = float(rng.uniform(500, 8000))
            points = float(rng.uniform(20, 350))
            end = start * np.exp(0.004 * (points - 180) + rng.normal(0, 0.1))
            rows.append(
                {
                    "player_id": f"p{i}",
                    "season": season,
                    "position": pos,
                    "age": float(rng.integers(21, 33)),
                    "points": points,
                    "start_value": start,
                    "end_value": end,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def model():
    return WhatIfModel(label_source="test", scale_max=9999).fit(_synthetic())


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE"])
def test_prediction_never_falls_as_points_rise(model, position):
    points = np.arange(0, 452, 0.7)
    for start in (1000, 4000, 8000):
        for age in (22, 25, 29):
            rows = pd.DataFrame({"start_value": start, "age": age, "position": position, "points": points})
            values = model.predict(rows)
            assert all(b >= a for a, b in pairwise(values)), (start, age)


def test_folds_only_train_on_earlier_seasons(monkeypatch):
    seen = []
    real_fit = WhatIfModel.fit

    def spy(self, df):
        seen.append(set(df["season"]))
        return real_fit(self, df)

    monkeypatch.setattr(WhatIfModel, "fit", spy)
    df = _synthetic()
    result = ev.evaluate(df, "test", 9999)
    test_seasons = sorted(result["by_season"])
    assert test_seasons == [2021, 2022, 2023, 2024]
    for train_seasons, test_season in zip(seen, test_seasons):
        assert max(train_seasons) < test_season


def test_beats_no_change_when_points_drive_value():
    result = ev.evaluate(_synthetic(), "test", 9999)
    assert result["beats_no_change"]
    assert result["overall"]["skill"] > 0


def test_predictions_clipped_to_scale(model):
    assert model.predict_one(start_value=9999, age=24, position="QB", points=500) <= 9999


def test_csv_label_source_swaps_in(tmp_path):
    df = _synthetic(n_per_season=8, seasons=[2024])
    stats = df[["player_id", "season", "position", "age", "points"]]
    path = tmp_path / "labels.csv"
    labels = df[["player_id", "season", "start_value", "end_value"]].copy()
    labels.loc[0, "start_value"] = 0  # untraded rows are dropped
    labels.to_csv(path, index=False)

    source = get_label_source(f"backchannel:{path}")
    assert isinstance(source, CsvLabels) and source.name == "backchannel"
    joined = build_dataset(stats, source)
    assert len(joined) == len(df) - 1


def test_csv_label_source_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"player_id": ["1"], "season": [2024], "value": [100]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        CsvLabels(path).load()
