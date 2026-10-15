"""The what-if model.

Target is log(end_value / start_value). A single gradient-boosted model
covers all positions and the predicted value never falls as season points
rise. Position is an ordinal code rather than a categorical feature:
sklearn 1.4 (the version this repo pins) mishandles monotonic constraints
alongside categorical features, which cost about 70 MAE on the folds.

The tree's monotonic constraint alone doesn't guarantee that here: with
``absolute_error`` loss sklearn refits leaves to medians after the split,
which can break the constraint by a few points of value. ``predict`` takes
a running max over a fixed points grid to close that gap. Absolute error beat
squared error on the season folds (MAE 434 vs 450-497).
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from predictor_v5.data import POSITIONS

FEATURES = ["log_start", "age", "points", "position_code"]
# +1: monotonic increasing in points; 0: unconstrained.
MONOTONIC = [0, 0, 1, 0]

HGB_PARAMS = {
    "loss": "absolute_error",
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": 0,
}
POINTS_GRID_STEP = 1.0


def make_features(df: pd.DataFrame) -> np.ndarray:
    unknown = set(df["position"]) - set(POSITIONS)
    if unknown:
        raise ValueError(f"Unknown positions: {sorted(unknown)}")
    return np.column_stack(
        [
            np.log(df["start_value"].astype(float)),
            df["age"].astype(float),
            df["points"].astype(float),
            df["position"].map(POSITIONS.index).astype(float),
        ]
    )


@dataclass
class WhatIfModel:
    label_source: str
    scale_max: float
    train_seasons: list[int] = field(default_factory=list)
    estimator: HistGradientBoostingRegressor | None = None

    def fit(self, df: pd.DataFrame) -> "WhatIfModel":
        self.estimator = HistGradientBoostingRegressor(
            monotonic_cst=MONOTONIC,
            **HGB_PARAMS,
        )
        y = np.log(df["end_value"] / df["start_value"])
        self.estimator.fit(make_features(df), y)
        self.train_seasons = sorted(int(s) for s in df["season"].unique())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Model is not fitted")
        log_ratio = self._monotone_log_ratio(make_features(df))
        pred = df["start_value"].to_numpy(float) * np.exp(log_ratio)
        return np.clip(pred, 0.0, self.scale_max)

    def _monotone_log_ratio(self, X: np.ndarray) -> np.ndarray:
        """Best prediction at any grid point at or below each row's points.

        The grid is fixed (0, 1, 2, ... points), so the result is a running
        max and can only rise with points. Points are snapped down to the
        grid, which moves predictions by at most one point's worth. Rows that
        differ only in points share one pass over the grid.
        """
        col = FEATURES.index("points")
        idx = np.floor(np.maximum(X[:, col], 0.0) / POINTS_GRID_STEP).astype(int)
        grid = np.arange(idx.max() + 1) * POINTS_GRID_STEP
        others = np.delete(X, col, axis=1)
        base, inverse = np.unique(others, axis=0, return_inverse=True)
        Xg = np.insert(np.repeat(base, len(grid), axis=0), col, np.tile(grid, len(base)), axis=1)
        on_grid = self.estimator.predict(Xg).reshape(len(base), len(grid))
        running_max = np.maximum.accumulate(on_grid, axis=1)
        return running_max[inverse.ravel(), idx]

    def predict_one(self, *, start_value: float, age: float, position: str, points: float) -> float:
        if start_value <= 0:
            raise ValueError("start_value must be > 0")
        row = pd.DataFrame([{"start_value": start_value, "age": age, "position": position.upper(), "points": points}])
        return float(self.predict(row)[0])

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | str) -> "WhatIfModel":
        return joblib.load(path)
