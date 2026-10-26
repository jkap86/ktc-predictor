"""Fixed expanding-fold adapters for the protected what-if model."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from predictor_v5.model import WhatIfModel

from .contract import (
    FEATURES,
    FOLDS,
    HGB_PARAMS,
    MONOTONIC,
    POINTS_GRID_STEP,
    POSITIONS,
    SCALE_MAX,
    TARGET_ID,
    ContractRefusal,
    FoldSpec,
    KtcMode,
    sorted_keys,
    validate_ktc_mode,
)
from .data import EvaluationData
from .metrics import score_model, training_majority_class

_MODEL_COLUMNS = (
    "player_id",
    "season",
    "position",
    "age",
    "points",
    "start_value",
    "end_value",
)
_OUTPUT_COLUMNS = (
    "season",
    "player_id",
    "position",
    "age",
    "games",
    "points",
    "start_value",
    "end_value",
    "start_evidence",
    "end_evidence",
    "start_trades_window",
    "end_trades_window",
)
_KTC_TARGET_ID = "archive-ktc-diagnostic-v1"
_KTC_SCALE_MAX = 9999.0
_KTC_MODEL_NAMES = (
    "ktc_ratio_transplant",
    "backchannel_production_refit",
    "backchannel_no_production_refit",
    "backchannel_no_change",
)


@dataclass(frozen=True)
class FoldResult:
    spec: FoldSpec
    train_keys: tuple[tuple[int, str], ...]
    test_keys: tuple[tuple[int, str], ...]
    predictions: pd.DataFrame
    tier_edges: dict[str, tuple[float, float] | None]
    training_majority: str
    model_parameters: dict[str, object]
    clipping: dict[str, int]
    _models: tuple[object, object] | None = field(default=None, repr=False, compare=False)


def _refuse(message: str, cause: BaseException | None = None) -> None:
    if cause is None:
        raise ContractRefusal(message)
    raise ContractRefusal(message) from cause


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        values = frame[column].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        _refuse(f"{column} must be finite", exc)
    if values.ndim != 1 or not np.isfinite(values).all():
        _refuse(f"{column} must be finite")
    return values


def _log_ratio(end: np.ndarray, start: np.ndarray, *, name: str) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with np.errstate(over="raise", under="ignore", invalid="raise", divide="raise"):
                ratio = np.divide(end, start)
                movement = np.log(ratio)
    except (FloatingPointError, Warning) as exc:
        _refuse(f"{name} requires a finite representable ratio", exc)
    if not np.isfinite(ratio).all() or (ratio <= 0).any() or not np.isfinite(movement).all():
        _refuse(f"{name} requires a finite representable ratio")
    return movement


def _validate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(rows, pd.DataFrame):
        _refuse("primary rows must be a pandas DataFrame")
    missing = sorted(set(_MODEL_COLUMNS) - set(rows.columns))
    if missing:
        _refuse(f"primary rows are missing columns: {', '.join(missing)}")
    ordered = rows.sort_values(["season", "player_id"], kind="mergesort").reset_index(drop=True).copy(deep=True)
    keys = sorted_keys(ordered)
    if len(keys) != len(ordered):
        _refuse("duplicate season/player_id key")
    if not ordered["position"].isin(POSITIONS).all():
        _refuse("primary rows contain an unknown position")
    for column in ("age", "points", "start_value", "end_value"):
        _numeric(ordered, column)
    starts = _numeric(ordered, "start_value")
    ends = _numeric(ordered, "end_value")
    if (starts <= 0).any() or (ends <= 0).any():
        _refuse("primary start and end values must be strictly positive")
    _log_ratio(ends, starts, name="primary movement")
    for spec in FOLDS:
        train = ordered["season"].isin(spec.train_seasons)
        test = ordered["season"] == spec.test_season
        if not test.any():
            _refuse(f"test season {spec.test_season} must be nonempty")
        for season in spec.train_seasons:
            if not ((ordered["season"] == season) & train).any():
                _refuse(f"required training season {season} must be nonempty")
    return ordered


def _model_frame(frame: pd.DataFrame, *, zero_points: bool) -> pd.DataFrame:
    model_frame = frame.loc[:, _MODEL_COLUMNS].copy(deep=True)
    if zero_points:
        model_frame.loc[:, "points"] = 0.0
    return model_frame


def _fit(model: object, frame: pd.DataFrame) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings(
            "ignore",
            message="Could not find the number of physical cores.*",
            category=UserWarning,
            module="joblib.externals.loky.backend.context",
        )
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            fitted = model.fit(frame.copy(deep=True))
    if fitted is None:
        _refuse("model fit returned no fitted estimator")
    return fitted


def _predict(model: object, frame: pd.DataFrame, expected_keys: tuple[tuple[int, str], ...]) -> np.ndarray:
    if tuple(frame[["season", "player_id"]].itertuples(index=False, name=None)) != expected_keys:
        _refuse("prediction key order does not match the requested cohort")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings(
            "ignore",
            message="Could not find the number of physical cores.*",
            category=UserWarning,
            module="joblib.externals.loky.backend.context",
        )
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            raw = model.predict(frame.copy(deep=True))
    try:
        predicted = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        _refuse("learned predictions must be numeric", exc)
    if predicted.ndim != 1 or len(predicted) != len(frame):
        _refuse("learned prediction length must match the held-out cohort")
    if not np.isfinite(predicted).all() or (predicted <= 0).any():
        _refuse("learned predictions must be strictly positive and finite")
    return predicted


def _tier_edges(train: pd.DataFrame) -> dict[str, tuple[float, float] | None]:
    result: dict[str, tuple[float, float] | None] = {}
    for position in POSITIONS:
        starts = train.loc[train["position"] == position, "start_value"].to_numpy(dtype=float)
        if len(starts) == 0:
            result[position] = None
        else:
            quantiles = np.quantile(starts, [1 / 3, 2 / 3], method="linear")
            result[position] = (float(quantiles[0]), float(quantiles[1]))
    return result


def _apply_tiers(test: pd.DataFrame, edges: dict[str, tuple[float, float] | None]) -> pd.Series:
    tiers: list[str] = []
    for position, start in test[["position", "start_value"]].itertuples(index=False, name=None):
        position_edges = edges[str(position)]
        if position_edges is None:
            _refuse(f"test position {position} has no eligible training starts")
        q1, q2 = position_edges
        if float(start) <= q1:
            tiers.append("low")
        elif float(start) <= q2:
            tiers.append("mid")
        else:
            tiers.append("high")
    return pd.Series(tiers, index=test.index, dtype="object")


def _parameters(model: object) -> dict[str, object]:
    result: dict[str, object] = {
        "label_source": TARGET_ID,
        "scale_max": SCALE_MAX,
        "features": list(FEATURES),
        "hgb_params": dict(HGB_PARAMS),
        "monotonic_cst": list(MONOTONIC),
        "points_grid_step": POINTS_GRID_STEP,
        "early_stopping": "auto",
        "do_early_stopping": None,
    }
    estimator = getattr(model, "estimator", None)
    if estimator is not None:
        parameters = estimator.get_params(deep=False)
        for name, expected in HGB_PARAMS.items():
            if parameters.get(name) != expected:
                _refuse(f"effective estimator parameter {name} differs from the contract")
        if list(parameters.get("monotonic_cst")) != list(MONOTONIC):
            _refuse("effective monotonic constraints differ from the contract")
        if parameters.get("early_stopping") != "auto":
            _refuse("effective early stopping setting differs from the contract")
        do_early_stopping = getattr(estimator, "do_early_stopping_", None)
        if do_early_stopping is not False:
            _refuse("early stopping must be inactive for historical folds")
        result["do_early_stopping"] = False
    return result


def _prediction_frame(
    test: pd.DataFrame,
    production: np.ndarray,
    no_production: np.ndarray,
    tiers: pd.Series,
) -> pd.DataFrame:
    present = [column for column in _OUTPUT_COLUMNS if column in test.columns]
    result = test.loc[:, present].copy(deep=True)
    result["S"] = test["start_value"].to_numpy(dtype=float)
    result["E"] = test["end_value"].to_numpy(dtype=float)
    result["r"] = _log_ratio(
        result["E"].to_numpy(dtype=float),
        result["S"].to_numpy(dtype=float),
        name="actual movement",
    )
    result["d"] = result["E"] - result["S"]
    result["value_tier"] = tiers.to_numpy()
    result["production_E_hat"] = production
    result["no_production_E_hat"] = no_production
    result["no_change_E_hat"] = result["S"]
    for name in ("production", "no_production", "no_change"):
        result[f"{name}_r_hat"] = _log_ratio(
            result[f"{name}_E_hat"].to_numpy(dtype=float),
            result["S"].to_numpy(dtype=float),
            name=f"{name} movement",
        )
        result[f"{name}_d_hat"] = result[f"{name}_E_hat"] - result["S"]
    return result.reset_index(drop=True)


def run_primary_folds(
    rows: pd.DataFrame,
    *,
    model_factory: Callable[..., WhatIfModel] = WhatIfModel,
) -> tuple[FoldResult, ...]:
    """Fit the fixed production and constant-points models on matched rows."""
    ordered = _validate_rows(rows)
    results: list[FoldResult] = []
    for spec in FOLDS:
        train = ordered.loc[ordered["season"].isin(spec.train_seasons)].reset_index(drop=True)
        test = ordered.loc[ordered["season"] == spec.test_season].reset_index(drop=True)
        train_keys = sorted_keys(train)
        test_keys = sorted_keys(test)
        production_fit = _model_frame(train, zero_points=False)
        no_production_fit = _model_frame(train, zero_points=True)
        production_predict = _model_frame(test, zero_points=False)
        no_production_predict = _model_frame(test, zero_points=True)

        production_model = _fit(
            model_factory(label_source=TARGET_ID, scale_max=SCALE_MAX), production_fit
        )
        no_production_model = _fit(
            model_factory(label_source=TARGET_ID, scale_max=SCALE_MAX), no_production_fit
        )
        production = _predict(production_model, production_predict, test_keys)
        no_production = _predict(no_production_model, no_production_predict, test_keys)
        edges = _tier_edges(train)
        tiers = _apply_tiers(test, edges)
        predictions = _prediction_frame(test, production, no_production, tiers)
        training_r = _log_ratio(
            train["end_value"].to_numpy(dtype=float),
            train["start_value"].to_numpy(dtype=float),
            name="training movement",
        )
        parameters = {
            "production": _parameters(production_model),
            "no_production": _parameters(no_production_model),
        }
        clipping = {
            "production_at_cap_count": int(np.count_nonzero(production == SCALE_MAX)),
            "no_production_at_cap_count": int(np.count_nonzero(no_production == SCALE_MAX)),
            "negative_points_train_count": int((train["points"] < 0).sum()),
            "negative_points_test_count": int((test["points"] < 0).sum()),
        }
        results.append(
            FoldResult(
                spec=spec,
                train_keys=train_keys,
                test_keys=test_keys,
                predictions=predictions,
                tier_edges=edges,
                training_majority=training_majority_class(training_r),
                model_parameters=parameters,
                clipping=clipping,
                _models=(production_model, no_production_model),
            )
        )
    return tuple(results)


def _validate_diagnostic_rows(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        _refuse("diagnostic rows must be a pandas DataFrame")
    missing = sorted(set(_MODEL_COLUMNS) - set(frame.columns))
    if missing:
        _refuse(f"diagnostic rows are missing columns: {', '.join(missing)}")
    ordered = frame.sort_values(["season", "player_id"], kind="mergesort").reset_index(
        drop=True
    ).copy(deep=True)
    sorted_keys(ordered)
    if not ordered["position"].isin(POSITIONS).all():
        _refuse("diagnostic rows contain an unknown position")
    for column in ("age", "points", "start_value", "end_value"):
        _numeric(ordered, column)
    starts = _numeric(ordered, "start_value")
    ends = _numeric(ordered, "end_value")
    if (starts <= 0).any():
        _refuse("diagnostic starts must be strictly positive")
    if kind == "thin_endpoint":
        if (ends <= 0).any():
            _refuse("thin diagnostic endpoints must be strictly positive")
        _log_ratio(ends, starts, name="thin diagnostic actual movement")
    elif kind == "observed_zero_end":
        if (ends != 0).any():
            _refuse("observed-zero diagnostic endpoints must equal zero")
    else:
        _refuse("unknown diagnostic cohort")
    return ordered


def _diagnostic_frame(frame: pd.DataFrame, fold: FoldResult, kind: str) -> list[dict[str, object]]:
    if fold._models is None:
        _refuse("fold does not retain fitted models for start-observable diagnostics")
    selected = frame.loc[frame["season"] == fold.spec.test_season].copy(deep=True)
    if selected.empty:
        return []
    selected.sort_values(["season", "player_id"], kind="mergesort", inplace=True)
    selected.reset_index(drop=True, inplace=True)
    actual_r = None
    if kind == "thin_endpoint":
        actual_r = _log_ratio(
            selected["end_value"].to_numpy(dtype=float),
            selected["start_value"].to_numpy(dtype=float),
            name="thin diagnostic actual movement",
        )
    keys = sorted_keys(selected)
    production_model, no_production_model = fold._models
    production = _predict(
        production_model, _model_frame(selected, zero_points=False), keys
    )
    no_production = _predict(
        no_production_model, _model_frame(selected, zero_points=True), keys
    )
    production_r = None
    no_production_r = None
    if kind == "thin_endpoint":
        starts = selected["start_value"].to_numpy(dtype=float)
        production_r = _log_ratio(
            production, starts, name="thin diagnostic production movement"
        )
        no_production_r = _log_ratio(
            no_production, starts, name="thin diagnostic no-production movement"
        )
    records: list[dict[str, object]] = []
    for index, row in selected.iterrows():
        start = float(row["start_value"])
        end = float(row["end_value"])
        record: dict[str, object] = {
            "season": int(row["season"]),
            "player_id": str(row["player_id"]),
            "position": str(row["position"]),
            "points": float(row["points"]),
            "S": start,
            "E": end,
            "production_E_hat": float(production[index]),
            "no_production_E_hat": float(no_production[index]),
            "no_change_E_hat": start,
            "production_absolute_error": abs(float(production[index]) - end),
            "no_production_absolute_error": abs(float(no_production[index]) - end),
            "no_change_absolute_error": abs(start - end),
        }
        if kind == "thin_endpoint":
            assert actual_r is not None
            assert production_r is not None
            assert no_production_r is not None
            record.update(
                {
                    "r": float(actual_r[index]),
                    "production_r_hat": float(production_r[index]),
                    "no_production_r_hat": float(no_production_r[index]),
                    "no_change_r_hat": 0.0,
                }
            )
        records.append(record)
    return records


def attach_start_observable_diagnostics(
    data: EvaluationData,
    folds: Sequence[FoldResult],
    *,
    model_factory: Callable[..., WhatIfModel] = WhatIfModel,
) -> dict[str, object]:
    """Predict diagnostic rows with the already-fitted fold estimators."""
    del model_factory  # Retained for the planned interface; diagnostics must never refit.
    if not isinstance(data, EvaluationData):
        _refuse("expected EvaluationData")
    if tuple(fold.spec for fold in folds) != FOLDS:
        _refuse("diagnostics require the complete fixed fold sequence")
    available_seasons = {fold.spec.test_season for fold in folds}
    result: dict[str, object] = {}
    validated = tuple(
        (name, _validate_diagnostic_rows(frame, name))
        for name, frame in (
            ("observed_zero_end", data.observed_zero_end),
            ("thin_endpoint", data.thin_endpoint),
        )
    )
    for name, frame in validated:
        predictions: list[dict[str, object]] = []
        for fold in folds:
            predictions.extend(_diagnostic_frame(frame, fold, name))
        unavailable = frame.loc[~frame["season"].isin(available_seasons)]
        result[name] = {
            "n": len(predictions),
            "unavailable_season_count": len(unavailable),
            "predictions": predictions,
            "primary_metrics_included": False,
        }
    return result


def _off_ktc_payload() -> dict[str, object]:
    return {
        "status": "off",
        "provenance_limits": {
            "provenance_limited": True,
            "reason": "diagnostic_off",
        },
        "cohorts": [],
        "fits": [],
        "scores": {"value": None, "reason": "diagnostic_off"},
        "predictions": [],
    }


def _ktc_invalid_reason(row: pd.Series) -> str | None:
    for column in ("start_ktc", "end_ktc"):
        value = row[column]
        if pd.isna(value):
            return f"missing_{column}"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return f"nonnumeric_{column}"
        if not np.isfinite(numeric):
            return f"nonfinite_{column}"
        if numeric <= 0:
            return f"nonpositive_{column}"
        if numeric > _KTC_SCALE_MAX:
            return f"above_max_{column}"
    return None


def _ktc_rows(data: EvaluationData) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(data, EvaluationData):
        _refuse("expected EvaluationData")
    if not isinstance(data.primary_scored, pd.DataFrame):
        _refuse("primary scored rows must be a pandas DataFrame")
    missing = sorted({"start_ktc", "end_ktc"} - set(data.primary_scored.columns))
    if missing:
        _refuse(f"archive KTC diagnostic rows are missing columns: {', '.join(missing)}")
    ordered = _validate_rows(data.primary_scored)
    reasons = ordered.apply(_ktc_invalid_reason, axis=1)
    return ordered, reasons


def _key_records(frame: pd.DataFrame) -> list[list[object]]:
    return [
        [int(season), str(player_id)]
        for season, player_id in frame[["season", "player_id"]].itertuples(
            index=False, name=None
        )
    ]


def _ktc_attrition(frame: pd.DataFrame, reasons: pd.Series) -> dict[str, object]:
    selected_reasons = reasons.loc[frame.index]
    invalid = selected_reasons.notna()
    invalid_frame = frame.loc[invalid]
    counts = selected_reasons.loc[invalid].value_counts(sort=False).to_dict()
    return {
        "count": int(invalid.sum()),
        "keys": _key_records(invalid_frame),
        "reasons": {name: int(counts[name]) for name in sorted(counts)},
    }


def _ktc_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    model_frame = _model_frame(frame, zero_points=False)
    model_frame["start_value"] = frame["start_ktc"].to_numpy(dtype=float)
    model_frame["end_value"] = frame["end_ktc"].to_numpy(dtype=float)
    return model_frame


def _ktc_score_frame(
    test: pd.DataFrame,
    ktc_prediction: np.ndarray,
    backchannel_production: np.ndarray,
    backchannel_no_production: np.ndarray,
) -> pd.DataFrame:
    starts = test["start_value"].to_numpy(dtype=float)
    ends = test["end_value"].to_numpy(dtype=float)
    ktc_starts = test["start_ktc"].to_numpy(dtype=float)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            ratio = np.divide(ktc_prediction, ktc_starts)
            transplant = np.multiply(ratio, starts)
    except FloatingPointError as exc:
        _refuse("KTC ratio transplant must be finite and representable", exc)
    if (
        not np.isfinite(ratio).all()
        or (ratio <= 0).any()
        or not np.isfinite(transplant).all()
        or (transplant <= 0).any()
    ):
        _refuse("KTC ratio transplant must be finite and representable")

    present = [column for column in _OUTPUT_COLUMNS if column in test.columns]
    frame = test.loc[:, present].copy(deep=True)
    frame["S"] = starts
    frame["E"] = ends
    frame["r"] = _log_ratio(ends, starts, name="KTC diagnostic actual movement")
    frame["d"] = ends - starts
    predictions = {
        "ktc_ratio_transplant": transplant,
        "backchannel_production_refit": backchannel_production,
        "backchannel_no_production_refit": backchannel_no_production,
        "backchannel_no_change": starts,
    }
    for name, endpoint in predictions.items():
        frame[f"{name}_E_hat"] = endpoint
        frame[f"{name}_r_hat"] = _log_ratio(
            endpoint, starts, name=f"{name} diagnostic movement"
        )
        frame[f"{name}_d_hat"] = endpoint - starts
    frame["ktc_S"] = ktc_starts
    frame["ktc_E"] = test["end_ktc"].to_numpy(dtype=float)
    frame["ktc_E_hat"] = ktc_prediction
    frame["ktc_ratio"] = ratio
    return frame.reset_index(drop=True)


def _ktc_prediction_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    columns = (
        "season",
        "player_id",
        "position",
        "S",
        "E",
        "ktc_S",
        "ktc_E",
        "ktc_E_hat",
        "ktc_ratio",
        "ktc_ratio_transplant_E_hat",
        "backchannel_production_refit_E_hat",
        "backchannel_no_production_refit_E_hat",
        "backchannel_no_change_E_hat",
    )
    records: list[dict[str, object]] = []
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        record = dict(zip(columns, row, strict=True))
        record["season"] = int(record["season"])
        record["player_id"] = str(record["player_id"])
        record["position"] = str(record["position"])
        for column in columns[3:]:
            record[column] = float(record[column])
        records.append(record)
    return records


def _require_source_cap(predicted: np.ndarray, cap: float, name: str) -> None:
    if (predicted > cap).any():
        _refuse(f"{name} learned predictions exceed source cap")


def _validate_primary_fold_keys(
    ordered: pd.DataFrame, primary_folds: Sequence[FoldResult] | None
) -> None:
    if primary_folds is None:
        return
    if tuple(fold.spec for fold in primary_folds) != FOLDS:
        _refuse("KTC diagnostic requires the complete fixed primary fold sequence")
    for fold in primary_folds:
        train = ordered.loc[ordered["season"].isin(fold.spec.train_seasons)]
        test = ordered.loc[ordered["season"] == fold.spec.test_season]
        if fold.train_keys != sorted_keys(train):
            _refuse("primary fold train keys do not match primary scored rows")
        if fold.test_keys != sorted_keys(test):
            _refuse("primary fold test keys do not match primary scored rows")


def run_ktc_diagnostic(
    data: EvaluationData,
    *,
    mode: KtcMode = "off",
    primary_folds: Sequence[FoldResult] | None = None,
    model_factory: Callable[..., WhatIfModel] = WhatIfModel,
) -> dict[str, object]:
    """Run the synthetic-tested optional KTC diagnostic without affecting primary folds."""
    validated_mode = validate_ktc_mode(mode)
    if validated_mode == "off":
        if not isinstance(data, EvaluationData):
            _refuse("expected EvaluationData")
        return _off_ktc_payload()

    ordered, reasons = _ktc_rows(data)
    _validate_primary_fold_keys(ordered, primary_folds)
    valid = reasons.isna()
    cohorts: list[dict[str, object]] = []
    fits: list[dict[str, object]] = []
    scores: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []

    for spec in FOLDS:
        train_primary = ordered.loc[ordered["season"].isin(spec.train_seasons)]
        test_primary = ordered.loc[ordered["season"] == spec.test_season]
        train = train_primary.loc[valid.loc[train_primary.index]].reset_index(drop=True)
        test = test_primary.loc[valid.loc[test_primary.index]].reset_index(drop=True)
        missing_train_seasons = [
            season for season in spec.train_seasons if not (train["season"] == season).any()
        ]
        unavailable_reason = None
        if missing_train_seasons:
            unavailable_reason = "empty_required_train_season"
        elif test.empty:
            unavailable_reason = "empty_test_intersection"
        cohort: dict[str, object] = {
            "test_season": spec.test_season,
            "train_seasons": list(spec.train_seasons),
            "status": "unavailable" if unavailable_reason else "available",
            "unavailable_reason": unavailable_reason,
            "missing_train_seasons": missing_train_seasons,
            "primary_train_count": len(train_primary),
            "primary_test_count": len(test_primary),
            "train_intersection_count": len(train),
            "test_intersection_count": len(test),
            "train_intersection_keys": _key_records(train),
            "test_intersection_keys": _key_records(test),
            "train_attrition": _ktc_attrition(train_primary, reasons),
            "test_attrition": _ktc_attrition(test_primary, reasons),
        }
        cohorts.append(cohort)
        if unavailable_reason:
            continue

        train_keys = sorted_keys(train)
        test_keys = sorted_keys(test)
        ktc_model = _fit(
            model_factory(label_source=_KTC_TARGET_ID, scale_max=_KTC_SCALE_MAX),
            _ktc_model_frame(train),
        )
        backchannel_production_model = _fit(
            model_factory(label_source=TARGET_ID, scale_max=SCALE_MAX),
            _model_frame(train, zero_points=False),
        )
        backchannel_no_production_model = _fit(
            model_factory(label_source=TARGET_ID, scale_max=SCALE_MAX),
            _model_frame(train, zero_points=True),
        )
        ktc_prediction = _predict(ktc_model, _ktc_model_frame(test), test_keys)
        _require_source_cap(ktc_prediction, _KTC_SCALE_MAX, "KTC")
        backchannel_production = _predict(
            backchannel_production_model,
            _model_frame(test, zero_points=False),
            test_keys,
        )
        backchannel_no_production = _predict(
            backchannel_no_production_model,
            _model_frame(test, zero_points=True),
            test_keys,
        )
        _require_source_cap(
            backchannel_production, SCALE_MAX, "Backchannel production"
        )
        _require_source_cap(
            backchannel_no_production,
            SCALE_MAX,
            "Backchannel no-production",
        )
        for name, target, scale_max, points_mode in (
            ("ktc_production", _KTC_TARGET_ID, _KTC_SCALE_MAX, "observed"),
            ("backchannel_production_refit", TARGET_ID, SCALE_MAX, "observed"),
            ("backchannel_no_production_refit", TARGET_ID, SCALE_MAX, "constant_zero"),
        ):
            fits.append(
                {
                    "test_season": spec.test_season,
                    "name": name,
                    "target": target,
                    "scale_max": scale_max,
                    "points_mode": points_mode,
                    "train_keys": [list(key) for key in train_keys],
                }
            )

        score_frame = _ktc_score_frame(
            test,
            ktc_prediction,
            backchannel_production,
            backchannel_no_production,
        )
        scores.append(
            {
                "test_season": spec.test_season,
                "test_keys": [list(key) for key in test_keys],
                "target": "backchannel_end_value",
                "start": "backchannel_start_value",
                "baseline": "backchannel_no_change",
                "models": {
                    name: score_model(score_frame, name) for name in _KTC_MODEL_NAMES
                },
                "cross_target_subtractions": [],
            }
        )
        predictions.extend(_ktc_prediction_records(score_frame))

    return {
        "status": "diagnostic_only",
        "provenance_limits": {
            "provenance_limited": True,
            "raw_observation_chain": "unverified_archive_fields",
            "weekly_ktc_used": False,
            "ratio_transplant": "uncalibrated_cross_market_comparability",
        },
        "cohorts": cohorts,
        "fits": fits,
        "scores": scores,
        "predictions": predictions,
    }
