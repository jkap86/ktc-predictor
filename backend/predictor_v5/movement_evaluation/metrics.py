"""Deterministic, finite-only metrics for historical market movement."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .contract import (
    CALIBRATION_EDGES,
    FLAT_BAND,
    LARGE_MOVE_BAND,
    POSITIONS,
    ContractRefusal,
    sorted_keys,
)

_SEASONS = tuple(range(2020, 2026))
_LIQUIDITY_BANDS = ("5-9", "10-24", "25+")
_TIERS = ("low", "mid", "high")
_MODEL_BASE = ("S", "E", "r", "d")


def _null(reason: str | None = None, value: float | None = None) -> dict[str, object]:
    if value is None:
        return {"value": None, "reason": reason or "undefined"}
    value = float(value)
    if not math.isfinite(value):
        return {"value": None, "reason": reason or "nonfinite"}
    return {"value": value, "reason": None}


def _array(values: ArrayLike, *, name: str = "values") -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractRefusal(f"{name} must be numeric") from exc
    if result.ndim != 1:
        raise ContractRefusal(f"{name} must be one-dimensional")
    if not np.isfinite(result).all():
        raise ContractRefusal(f"{name} must be finite")
    return result


def direction(values: ArrayLike) -> np.ndarray:
    """Classify movement as down/flat/up using the fixed inclusive flat band."""
    values_array = _array(values)
    result = np.full(values_array.shape, "flat", dtype="U5")
    result[values_array < -FLAT_BAND] = "down"
    result[values_array > FLAT_BAND] = "up"
    return result


def training_majority_class(actual_r: ArrayLike) -> str:
    classes = direction(actual_r)
    counts = {name: int(np.count_nonzero(classes == name)) for name in ("flat", "down", "up")}
    best = max(counts.values(), default=0)
    for name in ("flat", "down", "up"):
        if counts[name] == best:
            return name
    return "flat"


def _validate_frame(frame: pd.DataFrame, model: str | None = None) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ContractRefusal("metrics require a pandas DataFrame")
    required = set(_MODEL_BASE)
    if model is not None:
        required.update({f"{model}_E_hat", f"{model}_r_hat", f"{model}_d_hat"})
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractRefusal(f"prediction frame is missing columns: {', '.join(missing)}")
    if {"season", "player_id"}.issubset(frame.columns):
        keys = sorted_keys(frame)
        actual_keys = tuple(frame[["season", "player_id"]].itertuples(index=False, name=None))
        if actual_keys != keys:
            raise ContractRefusal("prediction frame keys must be sorted")
    for column in _MODEL_BASE:
        values = _array(frame[column].to_numpy(), name=column)
        if column in ("S", "E") and (values <= 0).any():
            raise ContractRefusal(f"{column} must be strictly positive")
    if model is not None:
        endpoint = _array(frame[f"{model}_E_hat"].to_numpy(), name=f"{model}_E_hat")
        if (endpoint <= 0).any():
            raise ContractRefusal(f"{model}_E_hat must be strictly positive")
        _array(frame[f"{model}_r_hat"].to_numpy(), name=f"{model}_r_hat")
        _array(frame[f"{model}_d_hat"].to_numpy(), name=f"{model}_d_hat")


def _metric_summary(errors: np.ndarray, *, empty_reason: str = "empty_cohort") -> dict[str, object]:
    if not len(errors):
        return {"n": 0, "value": None, "reason": empty_reason}
    return {"n": len(errors), "value": float(np.mean(errors)), "reason": None}


def _nullable_mean(values: np.ndarray, reason: str = "empty_cohort") -> dict[str, object]:
    return _null(reason if len(values) == 0 else None, None if len(values) == 0 else float(np.mean(values)))


def _direction_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    actual_class = direction(actual)
    pred_class = direction(predicted)
    labels = ("down", "flat", "up")
    matrix = {a: {p: int(np.count_nonzero((actual_class == a) & (pred_class == p))) for p in labels} for a in labels}
    recalls: dict[str, object] = {}
    supports: dict[str, int] = {}
    for label in labels:
        support = int(np.count_nonzero(actual_class == label))
        supports[label] = support
        recalls[label] = _null("absent_class" if support == 0 else None, None if support == 0 else matrix[label][label] / support)
    supported = [float(recalls[label]["value"]) for label in labels if recalls[label]["value"] is not None]
    movers = actual_class != "flat"
    return {
        "confusion_matrix": matrix,
        "counts": matrix,
        "direction_accuracy": _null("empty_cohort" if len(actual) == 0 else None, None if len(actual) == 0 else float(np.mean(actual_class == pred_class))),
        "class_recalls": recalls,
        "recalls": recalls,
        "balanced_accuracy": _null("no_supported_classes" if not supported else None, None if not supported else float(np.mean(supported))),
        "supported_class_count": len(supported),
        "support": supports,
        "mover_count": int(movers.sum()),
        "mover_accuracy": _null("no_movers" if not movers.any() else None, None if not movers.any() else float(np.mean(pred_class[movers] == actual_class[movers]))),
    }


def score_direction_baselines(frame: pd.DataFrame, training_majority: str) -> dict[str, object]:
    _validate_frame(frame)
    if training_majority not in {"down", "flat", "up"}:
        raise ContractRefusal("training majority class is invalid")
    actual = _array(frame["r"], name="r")
    no_change = np.zeros(len(actual), dtype=float)
    majority = np.full(len(actual), {"down": -2 * FLAT_BAND, "flat": 0.0, "up": 2 * FLAT_BAND}[training_majority])
    return {
        "no_change": _direction_metrics(actual, no_change),
        "majority": _direction_metrics(actual, majority),
        "training_majority": training_majority,
    }


def _ols(actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    if len(actual) == 0:
        return {"intercept": _null("empty_cohort"), "slope": _null("empty_cohort")}
    if len(predicted) < 2 or np.all(predicted == predicted[0]):
        return {"intercept": _null(None, float(np.mean(actual))), "slope": _null("constant_prediction")}
    variance = float(np.sum((predicted - predicted.mean()) ** 2))
    if variance == 0:
        return {"intercept": _null(None, float(np.mean(actual))), "slope": _null("constant_prediction")}
    slope = float(np.sum((predicted - predicted.mean()) * (actual - actual.mean())) / variance)
    return {"intercept": _null(None, float(actual.mean() - slope * predicted.mean())), "slope": _null(None, slope)}


def _calibration(actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    labels = ("lt_-log2", "-log2_to_-log1.25", "-log1.25_to_-log1.10", "-log1.10_to_0", "0_to_log1.10", "log1.10_to_log1.25", "log1.25_to_log2", "gte_log2")
    edges = (-math.inf, *CALIBRATION_EDGES, math.inf)
    bins: list[dict[str, object]] = []
    for i, label in enumerate(labels):
        mask = (predicted >= edges[i]) & (predicted < edges[i + 1])
        rows = actual[mask]
        pred_rows = predicted[mask]
        bins.append({
            "label": label,
            "lower": None if i == 0 else edges[i],
            "upper": None if i == len(labels) - 1 else edges[i + 1],
            "lower_unbounded": i == 0,
            "upper_unbounded": i == len(labels) - 1,
            "lower_inclusive": True,
            "upper_exclusive": True,
            "n": len(rows),
            "mean_predicted": _null("empty_bin" if len(rows) == 0 else None, None if len(rows) == 0 else float(pred_rows.mean())),
            "mean_actual": _null("empty_bin" if len(rows) == 0 else None, None if len(rows) == 0 else float(rows.mean())),
            "mean_difference": _null("empty_bin" if len(rows) == 0 else None, None if len(rows) == 0 else float(pred_rows.mean() - rows.mean())),
        })
    return {"edges": list(CALIBRATION_EDGES), "bins": bins, "ols": _ols(actual, predicted)}


def _spearman(actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    if len(actual) < 2:
        return _null("fewer_than_two")
    if np.all(actual == actual[0]):
        return _null("constant_actual")
    if np.all(predicted == predicted[0]):
        return _null("constant_prediction")
    actual_rank = pd.Series(actual).rank(method="average").to_numpy()
    predicted_rank = pd.Series(predicted).rank(method="average").to_numpy()
    return _null(None, float(np.corrcoef(actual_rank, predicted_rank)[0, 1]))


def _rank_metrics(frame: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    cells: dict[str, object] = {}
    if {"season", "position"}.issubset(frame.columns):
        for season in _SEASONS:
            for position in POSITIONS:
                mask = (frame["season"].to_numpy() == season) & (frame["position"].to_numpy() == position)
                cells[f"{season}:{position}"] = {"season": season, "position": position, "n": int(mask.sum()), "spearman": _spearman(actual[mask], predicted[mask])}
    equal_defined = {key: value["spearman"]["value"] for key, value in cells.items() if value["spearman"]["value"] is not None}
    equal = _null("no_defined_cells" if not equal_defined else None, None if not equal_defined else float(np.mean(list(equal_defined.values()))))
    equal = {**equal, "n_cells": len(equal_defined), "included_cells": sorted(equal_defined)}
    if len(predicted) == 0 or np.all(predicted == predicted[0]):
        top_bottom: dict[str, object] = _null("constant_prediction")
    else:
        work = frame[["season", "player_id"]].copy() if {"season", "player_id"}.issubset(frame.columns) else pd.DataFrame({"season": [0] * len(frame), "player_id": [str(i) for i in range(len(frame))]})
        work["r"] = actual
        work["pred"] = predicted
        cell_top: list[dict[str, object]] = []
        group_columns = ["season", "position"] if "position" in frame.columns else ["season"]
        if "position" in frame.columns:
            work["position"] = frame["position"].to_numpy()
        for cell, rows in work.groupby(group_columns, sort=True):
            n = len(rows)
            k = max(1, math.ceil(0.10 * n))
            pred_order = rows.sort_values(["pred", "player_id"], ascending=[False, True], kind="mergesort")
            actual_order = rows.sort_values(["r", "player_id"], ascending=[False, True], kind="mergesort")
            pred_bottom = rows.sort_values(["pred", "player_id"], ascending=[True, True], kind="mergesort")
            actual_bottom = rows.sort_values(["r", "player_id"], ascending=[True, True], kind="mergesort")
            top_ids = set(pred_order.head(k)["player_id"])
            bottom_ids = set(pred_bottom.head(k)["player_id"])
            cutoff_top = float(pred_order.iloc[min(k, n) - 1]["pred"])
            cutoff_bottom = float(pred_bottom.iloc[min(k, n) - 1]["pred"])
            cell_name = ":".join(str(part) for part in cell) if isinstance(cell, tuple) else str(cell)
            cell_top.append({
                "cell": cell_name, "n": n, "k": k,
                **({"season": int(cell[0]), "position": str(cell[1])} if isinstance(cell, tuple) else {"season": int(cell)}),
                "top_overlap": len(top_ids & set(actual_order.head(k)["player_id"])) / k,
                "bottom_overlap": len(bottom_ids & set(actual_bottom.head(k)["player_id"])) / k,
                "top_mean_actual_movement": float(pred_order.head(k)["r"].mean()),
                "bottom_mean_actual_movement": float(pred_bottom.head(k)["r"].mean()),
                "top_boundary_tie_count": int((rows["pred"] == cutoff_top).sum()),
                "bottom_boundary_tie_count": int((rows["pred"] == cutoff_bottom).sum()),
            })
        top_bottom = {
            "cells": cell_top,
            "n_cells": len(cell_top),
            "value": None,
            "reason": None,
            "k_rule": "max(1, ceil(0.10*n))",
        }
    return {"spearman": _spearman(actual, predicted), "season_position": cells, "equal_cell_spearman": equal, "top_bottom": top_bottom}


def _large_move(actual: np.ndarray, predicted: np.ndarray, frame: pd.DataFrame) -> dict[str, object]:
    mask = np.abs(actual) >= LARGE_MOVE_BAND
    selected_actual = actual[mask]
    selected_pred = predicted[mask]
    actual_pos = actual >= LARGE_MOVE_BAND
    actual_neg = actual <= -LARGE_MOVE_BAND
    pred_pos = predicted >= LARGE_MOVE_BAND
    pred_neg = predicted <= -LARGE_MOVE_BAND

    def _recall(tp: int, n: int) -> dict[str, object]:
        return _null("no_actual_positives" if n == 0 else None, None if n == 0 else tp / n)

    def _precision(tp: int, n: int) -> dict[str, object]:
        return _null("no_predicted_positives" if n == 0 else None, None if n == 0 else tp / n)

    errors = np.abs(predicted - actual)
    order = np.lexsort((frame["player_id"].astype(str).to_numpy() if "player_id" in frame else np.arange(len(frame)), frame["season"].to_numpy() if "season" in frame else np.zeros(len(frame)), -errors))
    largest = []
    for index in order[:20]:
        largest.append({"season": int(frame.iloc[index]["season"]) if "season" in frame else None, "player_id": str(frame.iloc[index]["player_id"]) if "player_id" in frame else str(index), "absolute_log_error": float(errors[index]), "signed_log_error": float(predicted[index] - actual[index])})
    return {
        "threshold": LARGE_MOVE_BAND,
        "n": int(mask.sum()),
        "coverage": _null("empty_cohort" if len(actual) == 0 else None, None if len(actual) == 0 else float(mask.mean())),
        "change_error": _metric_summary(np.abs(selected_pred - selected_actual)),
        "bias": _nullable_mean(selected_pred - selected_actual),
        "positive_recall": _recall(int(np.sum(pred_pos & actual_pos)), int(np.sum(actual_pos))),
        "positive_precision": _precision(int(np.sum(pred_pos & actual_pos)), int(np.sum(pred_pos))),
        "negative_recall": _recall(int(np.sum(pred_neg & actual_neg)), int(np.sum(actual_neg))),
        "negative_precision": _precision(int(np.sum(pred_neg & actual_neg)), int(np.sum(pred_neg))),
        "largest_errors": largest,
    }


def score_model(frame: pd.DataFrame, model: str) -> dict[str, object]:
    _validate_frame(frame, model)
    actual_r = _array(frame["r"], name="r")
    actual_d = _array(frame["d"], name="d")
    pred_r = _array(frame[f"{model}_r_hat"], name=f"{model}_r_hat")
    pred_d = _array(frame[f"{model}_d_hat"], name=f"{model}_d_hat")
    log_errors = np.abs(pred_r - actual_r)
    value_errors = np.abs(pred_d - actual_d)
    abs_actual_r = np.abs(actual_r)
    abs_actual_d = np.abs(actual_d)
    reason = "empty_cohort" if len(frame) == 0 else None
    log_baseline = float(np.mean(abs_actual_r)) if len(frame) else 0.0
    value_baseline = float(np.mean(abs_actual_d)) if len(frame) else 0.0
    summary: dict[str, object] = {
        "model": model,
        "n": len(frame),
        "n_players": int(frame["player_id"].nunique()) if "player_id" in frame else len(frame),
        "mae_log": _null(reason, None if reason else float(np.mean(log_errors))),
        "rmse_log": _null(reason, None if reason else float(np.sqrt(np.mean(log_errors**2)))),
        "median_abs_log_error": _null(reason, None if reason else float(np.median(log_errors))),
        "p90_abs_log_error": _null(reason, None if reason else float(np.quantile(log_errors, 0.9, method="linear"))),
        "signed_bias_log": _null(reason, None if reason else float(np.mean(pred_r - actual_r))),
        "mean_actual_movement": _null(reason, None if reason else float(np.mean(actual_r))),
        "mean_predicted_movement": _null(reason, None if reason else float(np.mean(pred_r))),
        "mae_value": _null(reason, None if reason else float(np.mean(value_errors))),
        "rmse_value": _null(reason, None if reason else float(np.sqrt(np.mean(value_errors**2)))),
        "median_abs_value_error": _null(reason, None if reason else float(np.median(value_errors))),
        "p90_abs_value_error": _null(reason, None if reason else float(np.quantile(value_errors, 0.9, method="linear"))),
        "signed_bias_value": _null(reason, None if reason else float(np.mean(pred_d - actual_d))),
        "mean_actual_value_change": _null(reason, None if reason else float(np.mean(actual_d))),
        "mean_predicted_value_change": _null(reason, None if reason else float(np.mean(pred_d))),
        "value_change_mae": _null(reason, None if reason else float(np.mean(value_errors))),
        "skill_log": _null("zero_baseline_error" if log_baseline == 0 else reason, None if reason or log_baseline == 0 else 1 - float(np.mean(log_errors)) / log_baseline),
        "skill_value": _null("zero_baseline_error" if value_baseline == 0 else reason, None if reason or value_baseline == 0 else 1 - float(np.mean(value_errors)) / value_baseline),
        "direction": _direction_metrics(actual_r, pred_r),
        "calibration": _calibration(actual_r, pred_r),
        "rank": _rank_metrics(frame, actual_r, pred_r),
        "tail": {"large_move": _large_move(actual_r, pred_r, frame), "largest_errors": _large_move(actual_r, pred_r, frame)["largest_errors"], "top_bottom": _rank_metrics(frame, actual_r, pred_r)["top_bottom"]},
        "descriptive": {
            "age": _distribution(frame.get("age", pd.Series(dtype=float))),
            "games": _distribution(frame.get("games", pd.Series(dtype=float))),
            "points": _distribution(frame.get("points", pd.Series(dtype=float))),
            "start_liquidity": _distribution(_liquidity(frame, "start")),
            "endpoint_liquidity": _distribution(_liquidity(frame, "end")),
        },
    }
    return summary


def _distribution(values: Any) -> dict[str, object]:
    try:
        arr = _array(pd.Series(values).dropna().to_numpy())
    except ContractRefusal:
        return {"n": 0, "mean": _null("empty_cohort"), "median": _null("empty_cohort"), "min": _null("empty_cohort"), "max": _null("empty_cohort")}
    if len(arr) == 0:
        return {"n": 0, "mean": _null("empty_cohort"), "median": _null("empty_cohort"), "min": _null("empty_cohort"), "max": _null("empty_cohort")}
    return {"n": len(arr), "mean": _null(None, arr.mean()), "median": _null(None, np.median(arr)), "min": _null(None, arr.min()), "max": _null(None, arr.max())}


def _liquidity(frame: pd.DataFrame, endpoint: str) -> pd.Series:
    names = (f"{endpoint}_trades_window", f"{endpoint}_trades")
    for name in names:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def _liquidity_band(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.select([numeric.between(5, 9), numeric.between(10, 24), numeric >= 25], _LIQUIDITY_BANDS, default="missing"), index=values.index)


def _slice_score(frame: pd.DataFrame, model: str, mask: pd.Series, *, retrospective: bool = False) -> dict[str, object]:
    result = score_model(frame.loc[mask].reset_index(drop=True), model)
    result["low_support"] = result["n"] < 30
    if retrospective:
        result["retrospective"] = True
    return result


def score_slices(frame: pd.DataFrame, model_names: tuple[str, ...]) -> dict[str, object]:
    _validate_frame(frame)
    result: dict[str, object] = {"overall": {model: score_model(frame, model) for model in model_names}}
    for model in model_names:
        model_result: dict[str, object] = {"overall": result["overall"][model]}
        model_result["season"] = {str(season): _slice_score(frame, model, frame["season"] == season) for season in _SEASONS}
        model_result["position"] = {position: _slice_score(frame, model, frame["position"] == position) for position in POSITIONS}
        model_result["season_position"] = {
            f"{season}:{position}": _slice_score(frame, model, (frame["season"] == season) & (frame["position"] == position))
            for season in _SEASONS for position in POSITIONS
        }
        tiers = frame["value_tier"] if "value_tier" in frame else pd.Series("missing", index=frame.index)
        model_result["value_tier"] = {tier: _slice_score(frame, model, tiers == tier) for tier in _TIERS}
        model_result["position_tier"] = {f"{position}:{tier}": _slice_score(frame, model, (frame["position"] == position) & (tiers == tier)) for position in POSITIONS for tier in _TIERS}
        start_band = _liquidity_band(_liquidity(frame, "start"))
        end_band = _liquidity_band(_liquidity(frame, "end"))
        model_result["start_liquidity"] = {band: _slice_score(frame, model, start_band == band) for band in _LIQUIDITY_BANDS}
        model_result["endpoint_liquidity"] = {band: _slice_score(frame, model, end_band == band, retrospective=True) for band in _LIQUIDITY_BANDS}
        result[model] = model_result
    for dimension in ("season", "position", "season_position", "value_tier", "position_tier", "start_liquidity", "endpoint_liquidity"):
        result[dimension] = {model: result[model][dimension] for model in model_names}
    return result


def score_contrast(frame: pd.DataFrame, left: str, right: str) -> dict[str, object]:
    left_score = score_model(frame, left)
    right_score = score_model(frame, right)

    def difference(key: str) -> dict[str, object]:
        a = left_score[key]["value"]
        b = right_score[key]["value"]
        return _null("undefined_component" if a is None or b is None else None, None if a is None or b is None else float(a - b))

    out: dict[str, object] = {
        "left": left_score,
        "right": right_score,
        "left_model": left,
        "right_model": right,
        "sign_convention": "left - right; positive_mae_log_improvement is right MAE minus left MAE, positive when left improves",
        "mae_log_difference": difference("mae_log"),
        "positive_mae_log_improvement": _null("undefined_component" if left_score["mae_log"]["value"] is None or right_score["mae_log"]["value"] is None else None, None if left_score["mae_log"]["value"] is None or right_score["mae_log"]["value"] is None else float(right_score["mae_log"]["value"] - left_score["mae_log"]["value"])),
        "mae_value_difference": difference("mae_value"),
        "direction_accuracy_difference": _null("undefined_component" if left_score["direction"]["direction_accuracy"]["value"] is None or right_score["direction"]["direction_accuracy"]["value"] is None else None, None if left_score["direction"]["direction_accuracy"]["value"] is None or right_score["direction"]["direction_accuracy"]["value"] is None else float(left_score["direction"]["direction_accuracy"]["value"] - right_score["direction"]["direction_accuracy"]["value"])),
    }
    out["skill_log_difference"] = difference("skill_log")
    return out


def target_diagnostics(frame: pd.DataFrame) -> dict[str, object]:
    _validate_frame(frame)
    seasons: dict[str, object] = {}
    for season in _SEASONS:
        subset = frame.loc[frame["season"] == season]
        r = _array(subset["r"], name="r")
        seasons[str(season)] = {
            "n": len(subset),
            "start_median": _null("empty_cohort" if len(subset) == 0 else None, None if len(subset) == 0 else float(np.median(subset["S"]))),
            "end_median": _null("empty_cohort" if len(subset) == 0 else None, None if len(subset) == 0 else float(np.median(subset["E"]))),
            "median_realized_log_movement": _null("empty_cohort" if len(subset) == 0 else None, None if len(subset) == 0 else float(np.median(r))),
            "start_liquidity": _distribution(_liquidity(subset, "start")),
            "end_liquidity": _distribution(_liquidity(subset, "end")),
        }
    return {"interpretation": "scale_drift_observation_not_normalization_or_correction", "season": seasons}


def macro_by_season(fold_scores: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total = len(fold_scores)
    output: dict[str, object] = {"n_folds": total}
    keys = ("mae_log", "rmse_log", "median_abs_log_error", "p90_abs_log_error", "mae_value", "rmse_value", "skill_log", "skill_value")
    for key in keys:
        values = []
        for score in fold_scores:
            cell = score.get(key)
            if isinstance(cell, Mapping) and cell.get("value") is not None:
                values.append(float(cell["value"]))
        output[key] = {"value": None if not values else float(np.mean(values)), "reason": "no_defined_folds" if not values else None, "n_defined": len(values), "n_total": total}
    return output

