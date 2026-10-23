"""Chronological season folds against a "value doesn't change" baseline.

For each test season Y the model trains only on seasons < Y. MAE is on the
value scale. ``skill`` is 1 - model_mae / baseline_mae, so > 0 means the
model beats no-change.
"""

import numpy as np
import pandas as pd

from predictor_v5.model import WhatIfModel


def _mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def season_folds(df: pd.DataFrame, min_train_seasons: int = 1) -> list[int]:
    seasons = sorted(df["season"].unique())
    return [int(s) for s in seasons[min_train_seasons:]]


def summarize_exact(g: pd.DataFrame) -> dict:
    """Full-precision MAEs and skill; ``skill`` is None when no-change MAE is 0, everything is None with no rows."""
    if len(g) == 0:
        return {"n": 0, "model_mae": None, "no_change_mae": None, "skill": None}
    model_mae = _mae(g["end_value"], g["pred_value"])
    base_mae = _mae(g["end_value"], g["start_value"])
    return {
        "n": len(g),
        "model_mae": model_mae,
        "no_change_mae": base_mae,
        "skill": 1 - model_mae / base_mae if base_mae else None,
    }


def round_summary(summary: dict) -> dict:
    """The display rounding used in every report: MAE to 0.1, skill to 0.001."""
    return {
        "n": summary["n"],
        "model_mae": None if summary["model_mae"] is None else round(summary["model_mae"], 1),
        "no_change_mae": None if summary["no_change_mae"] is None else round(summary["no_change_mae"], 1),
        "skill": None if summary["skill"] is None else round(summary["skill"], 3),
    }


def summarize(g: pd.DataFrame) -> dict:
    return round_summary(summarize_exact(g))


def holdout(df: pd.DataFrame, label_source: str, scale_max: float, train_seasons: list[int], test_season: int) -> dict:
    """Train on ``train_seasons`` only, test on ``test_season``."""
    if max(train_seasons) >= test_season:
        raise ValueError("train seasons must all precede the test season")
    train = df[df["season"].isin(train_seasons)]
    test = df[df["season"] == test_season].copy()
    model = WhatIfModel(label_source=label_source, scale_max=scale_max).fit(train)
    test["pred_value"] = model.predict(test)
    return {
        "label_source": label_source,
        "train_seasons": sorted(train_seasons),
        "n_train": len(train),
        "test_season": test_season,
        "overall": summarize(test),
        "by_position": {p: summarize(g) for p, g in test.groupby("position")},
        "predictions": test,
    }


def evaluate(df: pd.DataFrame, label_source: str, scale_max: float, min_train_seasons: int = 1) -> dict:
    preds = []
    for test_season in season_folds(df, min_train_seasons):
        train = df[df["season"] < test_season]
        test = df[df["season"] == test_season].copy()
        model = WhatIfModel(label_source=label_source, scale_max=scale_max).fit(train)
        test["pred_value"] = model.predict(test)
        preds.append(test)
    out = pd.concat(preds, ignore_index=True)

    return {
        "label_source": label_source,
        "folds": "train on seasons < Y, test on Y",
        "overall": summarize(out),
        "by_season": {int(s): summarize(g) for s, g in out.groupby("season")},
        "by_position": {p: summarize(g) for p, g in out.groupby("position")},
        "beats_no_change": bool(_mae(out["end_value"], out["pred_value"]) < _mae(out["end_value"], out["start_value"])),
        "predictions": out,
    }
