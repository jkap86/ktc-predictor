"""Tests for fixed matched movement-evaluation folds."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from predictor_v5.movement_evaluation.contract import (
    FOLDS,
    HGB_PARAMS,
    MONOTONIC,
    POSITIONS,
    SCALE_MAX,
    TARGET_ID,
    ContractRefusal,
)
from predictor_v5.movement_evaluation.data import EvaluationData
from predictor_v5.movement_evaluation.models import (
    attach_start_observable_diagnostics,
    run_primary_folds,
)

MODEL_COLUMNS = [
    "player_id",
    "season",
    "position",
    "age",
    "points",
    "start_value",
    "end_value",
]


def _rows(*, n_per_season: int = 6) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in range(2020, 2026):
        for index in range(n_per_season):
            position = POSITIONS[index % len(POSITIONS)]
            start = float(600 + 75 * index + 13 * (season - 2020))
            points = float(-3 if index == 0 else 20 + 12 * index + 2 * (season - 2020))
            end = start * float(np.exp((points - 45) / 500))
            rows.append(
                {
                    "season": season,
                    "player_id": f"p{index:02d}",
                    "position": position,
                    "age": float(22 + index + season - 2020),
                    "games": 10 + index,
                    "years_exp": index,
                    "points": points,
                    "start_value": start,
                    "end_value": end,
                    "start_trades_window": 5 + index,
                    "end_trades_window": 7 + index,
                    "unused_outcome": end * 2,
                }
            )
    return pd.DataFrame(rows).sort_values(["season", "player_id"]).reset_index(drop=True)


class SpyModel:
    instances: ClassVar[list[SpyModel]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.fit_frame: pd.DataFrame | None = None
        self.predict_frames: list[pd.DataFrame] = []
        self.offset = 0.0
        type(self).instances.append(self)

    def fit(self, frame: pd.DataFrame) -> SpyModel:
        self.fit_frame = frame.copy(deep=True)
        self.offset = float(np.mean(np.log(frame["end_value"] / frame["start_value"])))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.predict_frames.append(frame.copy(deep=True))
        return np.minimum(
            frame["start_value"].to_numpy(dtype=float)
            * np.exp(self.offset + frame["points"].to_numpy(dtype=float) / 10000),
            float(self.kwargs["scale_max"]),
        )


@pytest.fixture(autouse=True)
def _clear_spies() -> None:
    SpyModel.instances.clear()


def _spy_factory(**kwargs: object) -> SpyModel:
    return SpyModel(**kwargs)


def _fit_and_primary_predict_frames() -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    return [
        (model.fit_frame, model.predict_frames[0])
        for model in SpyModel.instances
        if model.fit_frame is not None and model.predict_frames
    ]


def test_fixed_folds_fit_exact_common_sorted_cohorts() -> None:
    rows = _rows()
    folds = run_primary_folds(rows, model_factory=_spy_factory)

    assert len(folds) == 5
    assert len(SpyModel.instances) == 10
    assert tuple(fold.spec for fold in folds) == FOLDS
    for fold_index, (fold, spec) in enumerate(zip(folds, FOLDS, strict=True)):
        production = SpyModel.instances[2 * fold_index]
        no_production = SpyModel.instances[2 * fold_index + 1]
        assert production.kwargs == no_production.kwargs == {
            "label_source": TARGET_ID,
            "scale_max": SCALE_MAX,
        }
        assert list(production.fit_frame.columns) == MODEL_COLUMNS
        assert list(no_production.fit_frame.columns) == MODEL_COLUMNS
        assert list(production.predict_frames[0].columns) == MODEL_COLUMNS
        assert list(no_production.predict_frames[0].columns) == MODEL_COLUMNS
        assert sorted(production.fit_frame["season"].unique().tolist()) == list(spec.train_seasons)
        assert all((production.fit_frame["season"] == season).any() for season in spec.train_seasons)
        assert production.predict_frames[0]["season"].unique().tolist() == [spec.test_season]
        pd.testing.assert_frame_equal(
            production.fit_frame.drop(columns="points"),
            no_production.fit_frame.drop(columns="points"),
        )
        pd.testing.assert_frame_equal(
            production.predict_frames[0].drop(columns="points"),
            no_production.predict_frames[0].drop(columns="points"),
        )
        assert fold.train_keys == tuple(
            production.fit_frame[["season", "player_id"]].itertuples(index=False, name=None)
        )
        assert fold.test_keys == tuple(
            production.predict_frames[0][["season", "player_id"]].itertuples(index=False, name=None)
        )
        assert fold.test_keys == tuple(
            fold.predictions[["season", "player_id"]].itertuples(index=False, name=None)
        )
        assert fold.predictions["no_change_E_hat"].tolist() == fold.predictions["S"].tolist()


@pytest.mark.parametrize("missing_season", range(2020, 2026))
def test_missing_required_train_or_test_season_refuses(missing_season: int) -> None:
    rows = _rows()
    with pytest.raises(ContractRefusal, match="nonempty"):
        run_primary_folds(rows.loc[rows["season"] != missing_season], model_factory=_spy_factory)
    assert not SpyModel.instances


def test_constant_points_ablation_changes_only_points_and_ignores_held_out_points() -> None:
    rows = _rows()
    original = rows.copy(deep=True)
    first = run_primary_folds(rows, model_factory=_spy_factory)
    calls = _fit_and_primary_predict_frames()
    for fold_index, spec in enumerate(FOLDS):
        production_fit, production_predict = calls[2 * fold_index]
        no_production_fit, no_production_predict = calls[2 * fold_index + 1]
        train = original.loc[original["season"].isin(spec.train_seasons)]
        test = original.loc[original["season"] == spec.test_season]
        assert production_fit["points"].tolist() == train["points"].tolist()
        assert production_predict["points"].tolist() == test["points"].tolist()
        assert set(no_production_fit["points"]) == {0.0}
        assert set(no_production_predict["points"]) == {0.0}
        pd.testing.assert_frame_equal(
            production_fit.drop(columns="points"), no_production_fit.drop(columns="points")
        )
        pd.testing.assert_frame_equal(
            production_predict.drop(columns="points"),
            no_production_predict.drop(columns="points"),
        )
    assert rows.equals(original)

    SpyModel.instances.clear()
    changed = rows.copy()
    changed.loc[changed["season"] == 2024, "points"] += 1000
    second = run_primary_folds(changed, model_factory=_spy_factory)
    first_2024 = first[3].predictions
    second_2024 = second[3].predictions
    assert first_2024["no_production_E_hat"].tolist() == second_2024["no_production_E_hat"].tolist()
    assert first_2024["no_change_E_hat"].tolist() == second_2024["no_change_E_hat"].tolist()


@pytest.mark.parametrize("bad_prediction", [np.nan, np.inf, 0.0, -1.0])
def test_invalid_learned_predictions_refuse_without_row_salvage(bad_prediction: float) -> None:
    class InvalidModel(SpyModel):
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            result = super().predict(frame)
            result[0] = bad_prediction
            return result

    with pytest.raises(ContractRefusal, match="strictly positive and finite"):
        run_primary_folds(_rows(), model_factory=lambda **kwargs: InvalidModel(**kwargs))


def test_wrong_prediction_length_refuses() -> None:
    class ShortModel(SpyModel):
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return super().predict(frame)[:-1]

    with pytest.raises(ContractRefusal, match="length"):
        run_primary_folds(_rows(), model_factory=lambda **kwargs: ShortModel(**kwargs))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "duplicate"),
        (lambda frame: frame.assign(age=lambda value: value["age"].mask(value.index == 0, np.nan)), "finite"),
        (
            lambda frame: frame.assign(
                start_value=lambda value: value["start_value"].mask(value.index == 0, 0.0)
            ),
            "strictly positive",
        ),
    ],
)
def test_duplicate_and_nonfinite_primary_rows_refuse_before_fit(
    mutator: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    with pytest.raises(ContractRefusal, match=message):
        run_primary_folds(mutator(_rows()), model_factory=_spy_factory)
    assert not SpyModel.instances


def test_fold_boundaries_and_test_outcomes_do_not_leak() -> None:
    rows = _rows()
    baseline = run_primary_folds(rows, model_factory=_spy_factory)
    baseline_calls = deepcopy(_fit_and_primary_predict_frames())
    SpyModel.instances.clear()
    changed = rows.copy()
    changed.loc[(changed["season"] == 2023) & (changed["player_id"] == "p00"), "end_value"] *= 4
    mutated = run_primary_folds(changed, model_factory=_spy_factory)
    changed_calls = _fit_and_primary_predict_frames()

    for fold_index in range(3):
        pd.testing.assert_frame_equal(baseline_calls[2 * fold_index][0], changed_calls[2 * fold_index][0])
        pd.testing.assert_frame_equal(baseline_calls[2 * fold_index + 1][0], changed_calls[2 * fold_index + 1][0])
    prediction_columns = [
        column
        for column in baseline[2].predictions
        if column.endswith(("_E_hat", "_r_hat", "_d_hat"))
    ]
    pd.testing.assert_frame_equal(
        baseline[2].predictions[prediction_columns],
        mutated[2].predictions[prediction_columns],
    )
    assert baseline[2].tier_edges == mutated[2].tier_edges
    assert baseline[2].training_majority == mutated[2].training_majority


def test_future_fold_cannot_change_earlier_fold() -> None:
    baseline = run_primary_folds(_rows(), model_factory=_spy_factory)
    SpyModel.instances.clear()
    changed = _rows()
    changed.loc[changed["season"] == 2025, ["points", "start_value", "end_value"]] *= 3
    mutated = run_primary_folds(changed, model_factory=_spy_factory)
    for before, after in zip(baseline[:4], mutated[:4], strict=True):
        pd.testing.assert_frame_equal(before.predictions, after.predictions)
        assert before.train_keys == after.train_keys
        assert before.test_keys == after.test_keys
        assert before.tier_edges == after.tier_edges
        assert before.training_majority == after.training_majority


def test_test_liquidity_changes_mask_not_training() -> None:
    rows = _rows()
    baseline = run_primary_folds(rows, model_factory=_spy_factory)
    baseline_calls = deepcopy(_fit_and_primary_predict_frames())
    SpyModel.instances.clear()
    changed = rows.loc[
        ~((rows["season"] == 2023) & (rows["player_id"] == "p00"))
    ].reset_index(drop=True)
    changed.loc[changed["season"] == 2023, "end_trades_window"] += 100
    mutated = run_primary_folds(changed, model_factory=_spy_factory)
    changed_calls = _fit_and_primary_predict_frames()
    for fold_index in range(3):
        pd.testing.assert_frame_equal(baseline_calls[2 * fold_index][0], changed_calls[2 * fold_index][0])
        pd.testing.assert_frame_equal(baseline_calls[2 * fold_index + 1][0], changed_calls[2 * fold_index + 1][0])
    assert len(mutated[2].predictions) == len(baseline[2].predictions) - 1


def test_tiers_use_training_starts_with_linear_thirds_and_preserve_collapsed_tiers() -> None:
    rows = _rows()
    rows.loc[rows["position"] == "QB", "start_value"] = rows.loc[
        rows["position"] == "QB", "season"
    ].map({2020: 100.0, 2021: 200.0, 2022: 300.0, 2023: 400.0, 2024: 500.0, 2025: 9999.0})
    rows.loc[rows["position"] == "TE", "start_value"] = 777.0
    folds = run_primary_folds(rows, model_factory=_spy_factory)
    fold = folds[-1]
    qb_train = rows.loc[(rows["season"] <= 2024) & (rows["position"] == "QB"), "start_value"]
    assert fold.tier_edges["QB"] == tuple(np.quantile(qb_train, [1 / 3, 2 / 3], method="linear"))
    assert fold.tier_edges["TE"] == (777.0, 777.0)
    assert fold.predictions.loc[fold.predictions["position"] == "QB", "value_tier"].tolist() == ["high", "high"]
    assert fold.predictions.loc[fold.predictions["position"] == "TE", "value_tier"].tolist() == ["low"]

    no_wr = rows.loc[~((rows["season"] < 2025) & (rows["position"] == "WR"))]
    with pytest.raises(ContractRefusal, match="no eligible training starts"):
        run_primary_folds(no_wr, model_factory=_spy_factory)
    without_wr = rows.loc[rows["position"] != "WR"]
    no_wr_folds = run_primary_folds(without_wr, model_factory=_spy_factory)
    assert all(result.tier_edges["WR"] is None for result in no_wr_folds)


def test_single_training_direction_class_uses_that_majority() -> None:
    rows = _rows()
    rows["end_value"] = rows["start_value"]
    folds = run_primary_folds(rows, model_factory=_spy_factory)
    assert [fold.training_majority for fold in folds] == ["flat"] * 5


def test_real_models_keep_contract_parameters_and_monotone_grid() -> None:
    rows = _rows(n_per_season=24)
    folds = run_primary_folds(rows)
    for fold in folds:
        for name in ("production", "no_production"):
            parameters = fold.model_parameters[name]
            assert parameters["hgb_params"] == dict(HGB_PARAMS)
            assert parameters["monotonic_cst"] == list(MONOTONIC)
            assert parameters["early_stopping"] == "auto"
            assert parameters["do_early_stopping"] is False
    train = rows.loc[rows["season"] == 2020].copy()
    model = __import__("predictor_v5.model", fromlist=["WhatIfModel"]).WhatIfModel(
        label_source=TARGET_ID, scale_max=SCALE_MAX
    ).fit(train)
    probe = pd.DataFrame(
        {
            "start_value": [1000.0] * 5,
            "age": [25.0] * 5,
            "position": ["QB"] * 5,
            "points": [-20.0, 0.0, 0.9, 1.0, 2.0],
        }
    )
    predicted = model.predict(probe)
    assert predicted[0] == predicted[1] == predicted[2]
    assert np.all(np.diff(predicted) >= 0)
    assert sum(fold.clipping["negative_points_train_count"] for fold in folds) > 0


def test_cap_saturation_is_counted_without_claiming_strict_clipping() -> None:
    class CapModel(SpyModel):
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            super().predict(frame)
            return np.full(len(frame), SCALE_MAX)

    fold = run_primary_folds(_rows(), model_factory=lambda **kwargs: CapModel(**kwargs))[0]
    assert fold.clipping["production_at_cap_count"] == len(fold.predictions)
    assert fold.clipping["no_production_at_cap_count"] == len(fold.predictions)
    assert all("strict" not in key for key in fold.clipping)


def test_diagnostics_reuse_fitted_models_without_refitting_or_scoring_missing_endpoints() -> None:
    primary = _rows()
    folds = run_primary_folds(primary, model_factory=_spy_factory)
    fits_before = len(SpyModel.instances)
    zero = primary.loc[primary["season"].isin([2020, 2022]), [
        "season", "player_id", "position", "age", "games", "years_exp", "points",
        "start_value", "end_value", "start_trades_window", "end_trades_window",
    ]].groupby("season", sort=True).head(1).copy()
    zero["player_id"] += "-zero"
    zero["end_value"] = 0.0
    thin = primary.loc[primary["season"].isin([2021, 2023]), zero.columns].groupby(
        "season", sort=True
    ).head(1).copy()
    thin["player_id"] += "-thin"
    thin["end_trades_window"] = 2
    data = EvaluationData(
        joined=pd.concat([primary, zero, thin], ignore_index=True),
        primary_scored=primary,
        observed_zero_end=zero,
        thin_endpoint=thin,
        coverage={},
    )

    def forbidden_factory(**_: object) -> object:
        raise AssertionError("diagnostics must not refit")

    diagnostics = attach_start_observable_diagnostics(
        data, folds, model_factory=forbidden_factory
    )
    assert len(SpyModel.instances) == fits_before
    assert diagnostics["observed_zero_end"]["n"] == 1
    assert diagnostics["observed_zero_end"]["unavailable_season_count"] == 1
    assert diagnostics["thin_endpoint"]["n"] == 2
    assert "r" not in diagnostics["observed_zero_end"]["predictions"][0]
    assert diagnostics["observed_zero_end"]["predictions"][0]["E"] == 0.0
    assert diagnostics["thin_endpoint"]["predictions"][0]["E"] > 0.0

