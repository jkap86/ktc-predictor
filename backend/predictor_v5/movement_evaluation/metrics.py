"""Deterministic, finite-only metrics for historical market movement."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .contract import (
    CALIBRATION_EDGES,
    FLAT_BAND,
    FOLDS,
    LARGE_MOVE_BAND,
    POSITIONS,
    SCALE_MAX,
    ContractRefusal,
    sorted_keys,
)

_SEASONS = tuple(range(2020, 2026))
_LIQUIDITY_BANDS = ("5-9", "10-24", "25+")
_TIERS = ("low", "mid", "high")
_MODEL_BASE = ("S", "E", "r", "d")
_PRIMARY_MODELS = ("production", "no_production", "no_change")
_CONTRASTS = (
    ("production_vs_no_production", "production", "no_production"),
    ("production_vs_no_change", "production", "no_change"),
)
_BOOTSTRAP_METRICS = (
    "mae_log_difference",
    "rmse_log_difference",
    "mae_value_difference",
    "rmse_value_difference",
    "skill_log_gain",
    "skill_value_gain",
    "direction_accuracy_difference",
    "positive_mae_log_improvement",
)


class _FoldLike(Protocol):
    spec: object
    predictions: pd.DataFrame
    training_majority: str


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
    starts = frame["S"].to_numpy(dtype=float)
    ends = frame["E"].to_numpy(dtype=float)
    if not np.allclose(frame["r"].to_numpy(dtype=float), np.log(ends / starts), rtol=1e-10, atol=1e-12):
        raise ContractRefusal("r is inconsistent with log(E / S)")
    if not np.allclose(frame["d"].to_numpy(dtype=float), ends - starts, rtol=1e-10, atol=1e-12):
        raise ContractRefusal("d is inconsistent with E - S")
    if model is not None:
        endpoint = _array(frame[f"{model}_E_hat"].to_numpy(), name=f"{model}_E_hat")
        if (endpoint <= 0).any():
            raise ContractRefusal(f"{model}_E_hat must be strictly positive")
        predicted_r = _array(frame[f"{model}_r_hat"].to_numpy(), name=f"{model}_r_hat")
        predicted_d = _array(frame[f"{model}_d_hat"].to_numpy(), name=f"{model}_d_hat")
        if not np.allclose(predicted_r, np.log(endpoint / starts), rtol=1e-10, atol=1e-12):
            raise ContractRefusal(f"{model}_r_hat is inconsistent with log({model}_E_hat / S)")
        if not np.allclose(predicted_d, endpoint - starts, rtol=1e-10, atol=1e-12):
            raise ContractRefusal(f"{model}_d_hat is inconsistent with {model}_E_hat - S")


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
        "retrospective": True,
        "diagnostic_only": True,
        "n": int(mask.sum()),
        "coverage": _null("empty_cohort" if len(actual) == 0 else None, None if len(actual) == 0 else float(mask.mean())),
        "direction": _direction_metrics(selected_actual, selected_pred),
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
    endpoint_errors = np.abs(frame[f"{model}_E_hat"].to_numpy(dtype=float) - frame["E"].to_numpy(dtype=float))
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
        "endpoint_mae": _null(reason, None if reason else float(np.mean(endpoint_errors))),
        "endpoint_rmse": _null(reason, None if reason else float(np.sqrt(np.mean(endpoint_errors**2)))),
        "endpoint_median_abs_error": _null(reason, None if reason else float(np.median(endpoint_errors))),
        "endpoint_p90_abs_error": _null(reason, None if reason else float(np.quantile(endpoint_errors, 0.9, method="linear"))),
        "skill_log": _null("zero_baseline_error" if log_baseline == 0 else reason, None if reason or log_baseline == 0 else 1 - float(np.mean(log_errors)) / log_baseline),
        "skill_value": _null("zero_baseline_error" if value_baseline == 0 else reason, None if reason or value_baseline == 0 else 1 - float(np.mean(value_errors)) / value_baseline),
        "direction": _direction_metrics(actual_r, pred_r),
        "calibration": _calibration(actual_r, pred_r),
        "rank": _rank_metrics(frame, actual_r, pred_r),
        "tail": {"large_move": _large_move(actual_r, pred_r, frame), "largest_errors": _large_move(actual_r, pred_r, frame)["largest_errors"], "top_bottom": _rank_metrics(frame, actual_r, pred_r)["top_bottom"]},
        "low_support": len(frame) < 30,
        "status": "low_support" if len(frame) < 30 else "supported",
        "clipping": {"at_cap_count": int((frame[f"{model}_E_hat"].to_numpy(dtype=float) == SCALE_MAX).sum())},
        "exclusions": {"count": 0, "reason": "matched_prediction_cohort"},
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
    raw = pd.Series(values)
    if raw.isna().any():
        raise ContractRefusal("descriptive distribution contains a missing/nonfinite value")
    if len(raw) == 0:
        return {"n": 0, "mean": _null("empty_cohort"), "median": _null("empty_cohort"), "min": _null("empty_cohort"), "max": _null("empty_cohort")}
    arr = _array(raw.to_numpy())
    return {"n": len(arr), "mean": _null(None, arr.mean()), "median": _null(None, np.median(arr)), "min": _null(None, arr.min()), "max": _null(None, arr.max())}


def _liquidity(frame: pd.DataFrame, endpoint: str) -> pd.Series:
    names = (f"{endpoint}_trades_window", f"{endpoint}_trades")
    for name in names:
        if name in frame:
            return frame[name]
    return pd.Series(dtype=float)


def _validate_slice_inputs(frame: pd.DataFrame) -> None:
    if "value_tier" not in frame:
        raise ContractRefusal("slice frame is missing value_tier")
    tiers = frame["value_tier"]
    if tiers.isna().any() or (~tiers.isin(_TIERS)).any():
        raise ContractRefusal("value_tier must be low, mid, or high")
    for endpoint in ("start", "end"):
        names = (f"{endpoint}_trades_window", f"{endpoint}_trades")
        name = next((candidate for candidate in names if candidate in frame), None)
        if name is None:
            raise ContractRefusal(f"slice frame is missing {endpoint} liquidity")
        raw = frame[name]
        numeric = pd.to_numeric(raw, errors="coerce")
        malformed = raw.notna() & numeric.isna()
        if malformed.any() or numeric.isna().any() or (~np.isfinite(numeric.to_numpy(dtype=float))).any() or (numeric < 5).any():
            raise ContractRefusal(f"{endpoint} liquidity must be finite and at least 5")


def _liquidity_band(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.select([numeric.between(5, 9), numeric.between(10, 24), numeric >= 25], _LIQUIDITY_BANDS, default="missing"), index=values.index)


def _slice_score(frame: pd.DataFrame, model: str, mask: pd.Series, *, retrospective: bool = False) -> dict[str, object]:
    result = score_model(frame.loc[mask].reset_index(drop=True), model)
    result["low_support"] = result["n"] < 30
    result["status"] = "low_support" if result["n"] < 30 else "supported"
    result["exclusions"] = {
        "count": int((~mask).sum()),
        "reason": "outside_requested_slice" if (~mask).any() else "none",
    }
    if retrospective:
        result["retrospective"] = True
    return result


def score_slices(frame: pd.DataFrame, model_names: tuple[str, ...]) -> dict[str, object]:
    _validate_frame(frame)
    _validate_slice_inputs(frame)
    result: dict[str, object] = {"overall": {model: _slice_score(frame, model, pd.Series(True, index=frame.index)) for model in model_names}}
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

    def gain(key: str) -> dict[str, object]:
        return difference(key)

    left_cells = left_score["rank"]["season_position"]
    right_cells = right_score["rank"]["season_position"]
    common_cells = sorted(
        key for key in left_cells
        if key in right_cells and left_cells[key]["spearman"]["value"] is not None and right_cells[key]["spearman"]["value"] is not None
    )
    rank_differences = [left_cells[key]["spearman"]["value"] - right_cells[key]["spearman"]["value"] for key in common_cells]
    rank_difference: dict[str, object] = _null("no_common_defined_cells" if not rank_differences else None, None if not rank_differences else float(np.mean(rank_differences)))
    rank_difference["included_cells"] = common_cells
    rank_difference["n_cells"] = len(common_cells)
    per_position: dict[str, dict[str, object]] = {}
    for position in POSITIONS:
        position_cells = [key for key in common_cells if key.endswith(f":{position}")]
        position_values = [rank_differences[common_cells.index(key)] for key in position_cells]
        position_metric = _null("no_common_defined_cells" if not position_values else None, None if not position_values else float(np.mean(position_values)))
        per_position[position] = {
            **position_metric,
            "n_cells": len(position_cells),
            "included_cells": position_cells,
        }
    rank_difference["per_position"] = per_position

    out: dict[str, object] = {
        "left": left_score,
        "right": right_score,
        "left_model": left,
        "right_model": right,
        "sign_convention": "left - right; positive_mae_log_improvement is right MAE minus left MAE, positive when left improves",
        "mae_log_difference": difference("mae_log"),
        "rmse_log_difference": difference("rmse_log"),
        "positive_mae_log_improvement": _null("undefined_component" if left_score["mae_log"]["value"] is None or right_score["mae_log"]["value"] is None else None, None if left_score["mae_log"]["value"] is None or right_score["mae_log"]["value"] is None else float(right_score["mae_log"]["value"] - left_score["mae_log"]["value"])),
        "mae_value_difference": difference("mae_value"),
        "rmse_value_difference": difference("rmse_value"),
        "skill_log_gain": gain("skill_log"),
        "skill_value_gain": gain("skill_value"),
        "direction_accuracy_difference": _null("undefined_component" if left_score["direction"]["direction_accuracy"]["value"] is None or right_score["direction"]["direction_accuracy"]["value"] is None else None, None if left_score["direction"]["direction_accuracy"]["value"] is None or right_score["direction"]["direction_accuracy"]["value"] is None else float(left_score["direction"]["direction_accuracy"]["value"] - right_score["direction"]["direction_accuracy"]["value"])),
        "movement_spearman_difference": rank_difference,
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
            if cell is None:
                continue
            if not isinstance(cell, Mapping):
                raise ContractRefusal(f"fold metric {key} must be a nullable metric mapping")
            value = cell.get("value")
            if value is None:
                continue
            if isinstance(value, bool):
                raise ContractRefusal(f"fold metric {key} is not numeric")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ContractRefusal(f"fold metric {key} is not numeric") from exc
            if not math.isfinite(value):
                raise ContractRefusal(f"fold metric {key} must be finite")
            values.append(value)
        output[key] = {
            "value": None if not values else float(np.mean(values)),
            "reason": "no_defined_folds" if not values else None,
            "n_defined": len(values),
            "n_undefined": total - len(values),
            "n_total": total,
        }
    return output


def _ordered_uncertainty_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ContractRefusal("uncertainty requires a pandas DataFrame")
    required = {"season", "player_id", "position"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractRefusal(f"uncertainty frame is missing columns: {', '.join(missing)}")
    ordered = frame.sort_values(["season", "player_id"], kind="mergesort").reset_index(drop=True).copy(deep=True)
    sorted_keys(ordered)
    if ordered["position"].isna().any() or (~ordered["position"].isin(POSITIONS)).any():
        raise ContractRefusal("uncertainty frame contains an unknown position")
    for model in _PRIMARY_MODELS:
        _validate_frame(ordered, model)
    return ordered


def _cluster_draw_indices(
    frame: pd.DataFrame,
    cluster: Literal["player", "season"],
    rng: Any,
) -> np.ndarray:
    """Draw sorted clusters with replacement and retain every row in each draw."""
    if cluster not in ("player", "season"):
        raise ContractRefusal("cluster must be exactly 'player' or 'season'")
    column = "player_id" if cluster == "player" else "season"
    labels = sorted(frame[column].unique().tolist())
    if not labels:
        return np.asarray([], dtype=np.int64)
    groups = [np.flatnonzero(frame[column].to_numpy() == label) for label in labels]
    selected = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[int(index)] for index in selected]).astype(np.int64, copy=False)


def _finite(value: float | np.floating[Any] | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mean(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return _finite(np.mean(values))


def _rmse(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        return _finite(np.sqrt(np.mean(np.square(values))))


def _skill(errors: np.ndarray, actual: np.ndarray) -> float | None:
    error = _mean(errors)
    baseline = _mean(np.abs(actual))
    if error is None or baseline is None or baseline == 0:
        return None
    return _finite(1.0 - error / baseline)


def _direction_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(actual) == 0:
        return None
    return _finite(np.mean(direction(actual) == direction(predicted)))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    boundaries = np.concatenate(
        (
            np.asarray([0], dtype=int),
            np.flatnonzero(ordered[1:] != ordered[:-1]) + 1,
            np.asarray([len(values)], dtype=int),
        )
    )
    ranks = np.empty(len(values), dtype=float)
    for start, stop in pairwise(boundaries):
        ranks[order[start:stop]] = (float(start) + float(stop - 1)) / 2.0 + 1.0
    return ranks


def _spearman_value(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(actual) < 2 or np.all(actual == actual[0]) or np.all(predicted == predicted[0]):
        return None
    actual_rank = _average_ranks(actual)
    predicted_rank = _average_ranks(predicted)
    actual_centered = actual_rank - actual_rank.mean()
    predicted_centered = predicted_rank - predicted_rank.mean()
    denominator = math.sqrt(
        float(np.sum(actual_centered**2)) * float(np.sum(predicted_centered**2))
    )
    if denominator == 0:
        return None
    return _finite(float(np.sum(actual_centered * predicted_centered)) / denominator)


def _rank_contrast(
    frame: pd.DataFrame,
    left: str,
    right: str,
) -> tuple[float | None, dict[str, float | None]]:
    if right == "no_change":
        return None, {position: None for position in POSITIONS}
    actual = frame["r"].to_numpy(dtype=float)
    left_predicted = frame[f"{left}_r_hat"].to_numpy(dtype=float)
    right_predicted = frame[f"{right}_r_hat"].to_numpy(dtype=float)
    cell_values: dict[str, list[float]] = {position: [] for position in POSITIONS}
    for season in sorted(frame["season"].unique().tolist()):
        for position in POSITIONS:
            mask = (frame["season"].to_numpy() == season) & (
                frame["position"].to_numpy() == position
            )
            left_value = _spearman_value(actual[mask], left_predicted[mask])
            right_value = _spearman_value(actual[mask], right_predicted[mask])
            if left_value is not None and right_value is not None:
                cell_values[position].append(left_value - right_value)
    per_position = {
        position: _mean(np.asarray(values, dtype=float)) for position, values in cell_values.items()
    }
    defined = [value for values in cell_values.values() for value in values]
    return _mean(np.asarray(defined, dtype=float)), per_position


def _paired_statistics(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    actual_r = frame["r"].to_numpy(dtype=float)
    actual_d = frame["d"].to_numpy(dtype=float)
    scores: dict[str, dict[str, float | None]] = {}
    for model in _PRIMARY_MODELS:
        predicted_r = frame[f"{model}_r_hat"].to_numpy(dtype=float)
        predicted_d = frame[f"{model}_d_hat"].to_numpy(dtype=float)
        log_errors = np.abs(predicted_r - actual_r)
        value_errors = np.abs(predicted_d - actual_d)
        scores[model] = {
            "mae_log": _mean(log_errors),
            "rmse_log": _rmse(log_errors),
            "mae_value": _mean(value_errors),
            "rmse_value": _rmse(value_errors),
            "skill_log": _skill(log_errors, actual_r),
            "skill_value": _skill(value_errors, actual_d),
            "direction_accuracy": _direction_accuracy(actual_r, predicted_r),
        }

    def difference(left: str, right: str, metric: str) -> float | None:
        left_value = scores[left][metric]
        right_value = scores[right][metric]
        if left_value is None or right_value is None:
            return None
        return _finite(left_value - right_value)

    output: dict[str, dict[str, object]] = {}
    for name, left, right in _CONTRASTS:
        rank, per_position = _rank_contrast(frame, left, right)
        mae_difference = difference(left, right, "mae_log")
        output[name] = {
            "mae_log_difference": mae_difference,
            "rmse_log_difference": difference(left, right, "rmse_log"),
            "mae_value_difference": difference(left, right, "mae_value"),
            "rmse_value_difference": difference(left, right, "rmse_value"),
            "skill_log_gain": difference(left, right, "skill_log"),
            "skill_value_gain": difference(left, right, "skill_value"),
            "direction_accuracy_difference": difference(left, right, "direction_accuracy"),
            "positive_mae_log_improvement": None
            if mae_difference is None
            else -mae_difference,
            "movement_spearman_difference": {
                "value": rank,
                "per_position": per_position,
            },
        }
    return output


def _bootstrap_interval(
    estimate: float | None,
    draws: Sequence[float | None],
    requested: int,
) -> dict[str, object]:
    valid = np.asarray(
        [float(value) for value in draws if value is not None and math.isfinite(float(value))],
        dtype=float,
    )
    minimum = math.ceil(0.9 * requested)
    enough = len(valid) >= minimum
    if enough:
        bounds = np.quantile(valid, [0.025, 0.975], method="linear")
        lower, upper = float(bounds[0]), float(bounds[1])
    else:
        lower = upper = None
    return {
        "estimate": _finite(estimate),
        "lower": lower,
        "upper": upper,
        "percentiles": [2.5, 97.5],
        "requested_draws": requested,
        "effective_draws": len(valid),
        "undefined_draws": requested - len(valid),
        "minimum_valid_draws": minimum,
        "reason": None if enough else "insufficient_valid_resamples",
    }


def bootstrap_uncertainty(
    frame: pd.DataFrame,
    *,
    seed: int = 20261025,
    resamples: int = 2000,
    cluster: Literal["player", "season"],
) -> dict[str, object]:
    """Recompute paired contrasts on deterministic retained-row cluster draws."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ContractRefusal("seed must be a nonnegative integer")
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, (int, np.integer))
        or int(resamples) <= 0
    ):
        raise ContractRefusal("resamples must be a positive integer")
    if cluster not in ("player", "season"):
        raise ContractRefusal("cluster must be exactly 'player' or 'season'")
    seed = int(seed)
    resamples = int(resamples)
    ordered = _ordered_uncertainty_frame(frame)
    cluster_column = "player_id" if cluster == "player" else "season"
    labels = sorted(ordered[cluster_column].unique().tolist())
    point = _paired_statistics(ordered)
    accumulators: dict[str, dict[str, object]] = {}
    for contrast_name, _, _ in _CONTRASTS:
        accumulators[contrast_name] = {
            **{metric: [] for metric in _BOOTSTRAP_METRICS},
            "movement_spearman_difference": [],
            "per_position": {position: [] for position in POSITIONS},
        }

    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        indices = _cluster_draw_indices(ordered, cluster, rng)
        drawn = ordered.iloc[indices].reset_index(drop=True)
        statistics = _paired_statistics(drawn)
        for contrast_name, _, _ in _CONTRASTS:
            contrast = statistics[contrast_name]
            accumulator = accumulators[contrast_name]
            for metric in _BOOTSTRAP_METRICS:
                accumulator[metric].append(contrast[metric])
            rank = contrast["movement_spearman_difference"]
            accumulator["movement_spearman_difference"].append(rank["value"])
            for position in POSITIONS:
                accumulator["per_position"][position].append(rank["per_position"][position])

    contrasts: dict[str, object] = {}
    for contrast_name, _, _ in _CONTRASTS:
        point_contrast = point[contrast_name]
        accumulator = accumulators[contrast_name]
        contrast_intervals: dict[str, object] = {
            metric: _bootstrap_interval(point_contrast[metric], accumulator[metric], resamples)
            for metric in _BOOTSTRAP_METRICS
        }
        point_rank = point_contrast["movement_spearman_difference"]
        rank_interval = _bootstrap_interval(
            point_rank["value"], accumulator["movement_spearman_difference"], resamples
        )
        rank_interval["per_position"] = {
            position: _bootstrap_interval(
                point_rank["per_position"][position],
                accumulator["per_position"][position],
                resamples,
            )
            for position in POSITIONS
        }
        contrast_intervals["movement_spearman_difference"] = rank_interval
        contrasts[contrast_name] = contrast_intervals

    result: dict[str, object] = {
        "cluster": cluster,
        "sampling_unit": "distinct_player_id" if cluster == "player" else "entire_test_season",
        "seed": seed,
        "requested_resamples": resamples,
        "percentiles": [2.5, 97.5],
        "quantile_method": "linear",
        "minimum_valid_fraction": 0.9,
        "n_clusters": len(labels),
        "cluster_labels": labels,
        "paired_draws": True,
        "refit_models": False,
        "contrasts": contrasts,
        "caveat": (
            "Player-cluster intervals condition on observed seasons and do not capture common market shocks."
            if cluster == "player"
            else (
                f"Exploratory only: {'five' if len(labels) == 5 else len(labels)} observed test-season "
                "blocks do not provide a reliable "
                "new-season guarantee."
            )
        ),
        "joint_uncertainty_limit": (
            "Neither one-way bootstrap captures simultaneous player/team/season/league uncertainty; "
            "team and league identifiers are unavailable."
        ),
        "multiplicity": "family-wide multiplicity is uncontrolled",
    }
    if cluster == "season":
        result["exploratory"] = True
    return result


def _primary_scores(frame: pd.DataFrame) -> tuple[dict[str, object], dict[str, object]]:
    models = {model: score_model(frame, model) for model in _PRIMARY_MODELS}
    contrasts = {
        name: score_contrast(frame, left, right) for name, left, right in _CONTRASTS
    }
    return models, contrasts


def leave_one_season_out(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return pooled aggregate estimates after omitting each observed test season."""
    ordered = _ordered_uncertainty_frame(frame)
    seasons = sorted(int(season) for season in ordered["season"].unique().tolist())
    output: list[dict[str, object]] = []
    for omitted in seasons:
        retained = ordered.loc[ordered["season"] != omitted].reset_index(drop=True)
        models, contrasts = _primary_scores(retained)
        output.append(
            {
                "omitted_season": omitted,
                "included_seasons": [season for season in seasons if season != omitted],
                "n": len(retained),
                "models": models,
                "contrasts": contrasts,
            }
        )
    return output


def summarize_primary(
    folds: Sequence[_FoldLike],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Aggregate fixed held-out folds without importing or refitting model classes."""
    fold_list = list(folds)
    expected_seasons = [fold.test_season for fold in FOLDS]
    actual_seasons = [int(fold.spec.test_season) for fold in fold_list]
    if actual_seasons != expected_seasons:
        raise ContractRefusal("primary summaries require the five fixed folds in test-season order")

    summaries: list[dict[str, object]] = []
    pooled_frames: list[pd.DataFrame] = []
    fold_model_scores: dict[str, list[Mapping[str, object]]] = {
        model: [] for model in _PRIMARY_MODELS
    }
    for fold in fold_list:
        season = int(fold.spec.test_season)
        predictions = _ordered_uncertainty_frame(fold.predictions)
        if len(predictions) == 0 or set(predictions["season"].astype(int)) != {season}:
            raise ContractRefusal(f"fold {season} predictions must contain only its nonempty test season")
        models, contrasts = _primary_scores(predictions)
        for model in _PRIMARY_MODELS:
            fold_model_scores[model].append(models[model])
        summary: dict[str, object] = {
            "test_season": season,
            "train_seasons": list(fold.spec.train_seasons),
            "short_history": bool(fold.spec.short_history),
            "n": len(predictions),
            "models": models,
            "contrasts": contrasts,
            "direction_baselines": score_direction_baselines(
                predictions, str(fold.training_majority)
            ),
            "slices": score_slices(predictions, _PRIMARY_MODELS),
            "uncertainty": bootstrap_uncertainty(predictions, cluster="player"),
        }
        for field in ("tier_edges", "model_parameters", "clipping"):
            if hasattr(fold, field):
                summary[field] = getattr(fold, field)
        summaries.append(summary)
        pooled_frames.append(predictions)

    pooled_frame = (
        pd.concat(pooled_frames, ignore_index=True)
        .sort_values(["season", "player_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    pooled_models, pooled_contrasts = _primary_scores(pooled_frame)
    uncertainty = {
        "player_cluster": bootstrap_uncertainty(pooled_frame, cluster="player"),
        "by_position": {
            position: bootstrap_uncertainty(
                pooled_frame.loc[pooled_frame["position"] == position].reset_index(drop=True),
                cluster="player",
            )
            for position in POSITIONS
        },
        "season_block_exploratory": bootstrap_uncertainty(pooled_frame, cluster="season"),
    }
    pooled: dict[str, object] = {
        "n": len(pooled_frame),
        "models": pooled_models,
        "contrasts": pooled_contrasts,
        "macro_by_season": {
            model: macro_by_season(fold_model_scores[model]) for model in _PRIMARY_MODELS
        },
        "uncertainty": uncertainty,
        "leave_one_season_out": leave_one_season_out(pooled_frame),
        "slices": score_slices(pooled_frame, _PRIMARY_MODELS),
        "target_diagnostics": target_diagnostics(pooled_frame),
        "uncertainty_caveat": (
            "Neither one-way bootstrap captures simultaneous player/team/season/league uncertainty; "
            "team and league identifiers are unavailable."
        ),
        "multiplicity": "family-wide multiplicity is uncontrolled",
    }
    return summaries, pooled

