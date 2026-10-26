"""Synthetic-only tests for the optional archive-KTC diagnostic."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from predictor_v5.movement_evaluation.contract import (
    FOLDS,
    SCALE_MAX,
    TARGET_ID,
    ContractRefusal,
)
from predictor_v5.movement_evaluation.data import EvaluationData
from predictor_v5.movement_evaluation.metrics import summarize_primary
from predictor_v5.movement_evaluation.models import (
    run_ktc_diagnostic,
    run_primary_folds,
)


class RecordingModel:
    instances: ClassVar[list[RecordingModel]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.fit_frame: pd.DataFrame | None = None
        self.predict_frames: list[pd.DataFrame] = []
        self.offset = 0.0
        type(self).instances.append(self)

    def fit(self, frame: pd.DataFrame) -> RecordingModel:
        self.fit_frame = frame.copy(deep=True)
        self.offset = float(np.mean(np.log(frame["end_value"] / frame["start_value"])))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.predict_frames.append(frame.copy(deep=True))
        if self.kwargs["scale_max"] == 9999.0:
            return np.full(len(frame), 9999.0)
        ratio = np.exp(self.offset + frame["points"].to_numpy(dtype=float) / 10000)
        return np.minimum(frame["start_value"].to_numpy(dtype=float) * ratio, SCALE_MAX)


class GuardedFrame(pd.DataFrame):
    """A DataFrame that fails if optional KTC fields are selected."""

    @property
    def _constructor(self) -> type[pd.DataFrame]:
        return pd.DataFrame

    def __getitem__(self, key: object) -> object:
        names = {key} if isinstance(key, str) else set(key) if isinstance(key, list) else set()
        forbidden = {"start_ktc", "end_ktc", "weekly_ktc"}
        if names & forbidden:
            raise AssertionError("off mode accessed optional KTC fields")
        return super().__getitem__(key)


@pytest.fixture(autouse=True)
def _clear_recording_models() -> None:
    RecordingModel.instances.clear()


def _factory(**kwargs: object) -> RecordingModel:
    return RecordingModel(**kwargs)


def _rows(*, n_per_season: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in range(2020, 2026):
        for index in range(n_per_season):
            start = float(2000 + 50 * index + 10 * (season - 2020))
            points = float(80 + 10 * index + season - 2020)
            rows.append(
                {
                    "season": season,
                    "player_id": f"p{index}",
                    "position": "QB",
                    "age": float(24 + index),
                    "games": 12 + index,
                    "years_exp": index,
                    "points": points,
                    "start_value": start,
                    "end_value": start * np.exp((points - 90) / 500),
                    "start_trades_window": 8 + index,
                    "end_trades_window": 9 + index,
                    "start_ktc": float(1000 + 25 * index),
                    "end_ktc": float(1100 + 25 * index),
                }
            )
    return pd.DataFrame(rows).sort_values(["season", "player_id"]).reset_index(drop=True)


def _data(rows: pd.DataFrame) -> EvaluationData:
    empty = rows.iloc[0:0].copy()
    return EvaluationData(
        joined=rows.copy(deep=True),
        primary_scored=rows.copy(deep=True),
        observed_zero_end=empty.copy(deep=True),
        thin_endpoint=empty.copy(deep=True),
        coverage={},
    )


def _fold_bytes(folds: object) -> bytes:
    payload = []
    for fold in folds:
        payload.append(
            {
                "test_season": fold.spec.test_season,
                "train_keys": fold.train_keys,
                "test_keys": fold.test_keys,
                "predictions": fold.predictions.to_dict(orient="records"),
                "tier_edges": fold.tier_edges,
                "training_majority": fold.training_majority,
                "model_parameters": fold.model_parameters,
                "clipping": fold.clipping,
            }
        )
    return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _diagnostic_bytes(value: dict[str, object]) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def test_off_mode_is_exact_and_never_accesses_ktc_or_changes_primary_bytes() -> None:
    rows = _rows()
    absent = rows.drop(columns=["start_ktc", "end_ktc"])
    absent_folds = run_primary_folds(absent, model_factory=_factory)
    absent_bytes = _fold_bytes(absent_folds)
    absent_summary_bytes = _diagnostic_bytes(
        dict(zip(("folds", "pooled"), summarize_primary(absent_folds), strict=True))
    )

    RecordingModel.instances.clear()
    guarded = GuardedFrame(rows.assign(weekly_ktc="must-not-be-read"))
    ignored_folds = run_primary_folds(guarded, model_factory=_factory)
    assert _fold_bytes(ignored_folds) == absent_bytes
    assert _diagnostic_bytes(
        dict(zip(("folds", "pooled"), summarize_primary(ignored_folds), strict=True))
    ) == absent_summary_bytes
    fit_count = len(RecordingModel.instances)

    empty = guarded.iloc[0:0]
    data = EvaluationData(
        joined=guarded,
        primary_scored=guarded,
        observed_zero_end=empty,
        thin_endpoint=empty,
        coverage={},
    )

    def forbidden_factory(**_: object) -> object:
        raise AssertionError("off mode must not fit")

    assert run_ktc_diagnostic(data, model_factory=forbidden_factory) == {
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
    assert len(RecordingModel.instances) == fit_count
    assert run_ktc_diagnostic(_data(absent), model_factory=forbidden_factory)["status"] == "off"


def test_archive_refits_exact_ordered_intersections_and_transplants_source_ratio() -> None:
    rows = _rows()
    rows.loc[(rows["season"] == 2025) & (rows["player_id"] == "p3"), "end_ktc"] = np.nan
    primary = run_primary_folds(rows, model_factory=_factory)
    primary_bytes = _fold_bytes(primary)
    assert len(RecordingModel.instances) == 10

    result = run_ktc_diagnostic(
        _data(rows),
        mode="archive",
        primary_folds=primary,
        model_factory=_factory,
    )

    assert set(result) == {
        "status",
        "provenance_limits",
        "cohorts",
        "fits",
        "scores",
        "predictions",
    }
    assert result["status"] == "diagnostic_only"
    assert result["provenance_limits"] == {
        "provenance_limited": True,
        "raw_observation_chain": "unverified_archive_fields",
        "weekly_ktc_used": False,
        "ratio_transplant": "uncalibrated_cross_market_comparability",
    }
    assert len(RecordingModel.instances) == 25
    diagnostic_models = RecordingModel.instances[10:]
    assert len(diagnostic_models) == 3 * len(FOLDS)
    assert len(result["fits"]) == 3 * len(FOLDS)

    for fold_index, spec in enumerate(FOLDS):
        cohort = result["cohorts"][fold_index]
        assert cohort["test_season"] == spec.test_season
        assert cohort["status"] == "available"
        train_keys = [
            [int(season), str(player_id)]
            for season, player_id in rows.loc[
                rows["season"].isin(spec.train_seasons), ["season", "player_id"]
            ].itertuples(index=False, name=None)
        ]
        test = rows.loc[
            (rows["season"] == spec.test_season)
            & rows["start_ktc"].notna()
            & rows["end_ktc"].notna()
        ]
        test_keys = [
            [int(season), str(player_id)]
            for season, player_id in test[["season", "player_id"]].itertuples(
                index=False, name=None
            )
        ]
        assert cohort["train_intersection_keys"] == train_keys
        assert cohort["test_intersection_keys"] == test_keys
        assert cohort["test_attrition"]["count"] == (1 if spec.test_season == 2025 else 0)

        models = diagnostic_models[3 * fold_index : 3 * fold_index + 3]
        assert [model.kwargs for model in models] == [
            {"label_source": "archive-ktc-diagnostic-v1", "scale_max": 9999.0},
            {"label_source": TARGET_ID, "scale_max": SCALE_MAX},
            {"label_source": TARGET_ID, "scale_max": SCALE_MAX},
        ]
        model_fit_keys = [
            list(model.fit_frame[["season", "player_id"]].itertuples(index=False, name=None))
            for model in models
        ]
        assert model_fit_keys == [
            [tuple(key) for key in train_keys],
            [tuple(key) for key in train_keys],
            [tuple(key) for key in train_keys],
        ]
        assert set(models[2].fit_frame["points"]) == {0.0}
        pd.testing.assert_frame_equal(
            models[1].fit_frame.drop(columns="points"),
            models[2].fit_frame.drop(columns="points"),
        )
        predict_keys = [
            list(model.predict_frames[0][["season", "player_id"]].itertuples(index=False, name=None))
            for model in models
        ]
        assert predict_keys == [[tuple(key) for key in test_keys]] * 3

    last_prediction = result["predictions"][-1]
    assert last_prediction["ktc_ratio"] == pytest.approx(
        last_prediction["ktc_E_hat"] / last_prediction["ktc_S"]
    )
    assert last_prediction["ktc_ratio_transplant_E_hat"] == pytest.approx(
        last_prediction["ktc_ratio"] * last_prediction["S"]
    )
    assert last_prediction["ktc_ratio_transplant_E_hat"] > SCALE_MAX
    assert all(
        score["target"] == "backchannel_end_value"
        and score["start"] == "backchannel_start_value"
        and score["baseline"] == "backchannel_no_change"
        and score["cross_target_subtractions"] == []
        and set(score["models"]) == {
            "ktc_ratio_transplant",
            "backchannel_production_refit",
            "backchannel_no_production_refit",
            "backchannel_no_change",
        }
        and all(
            model_score["n"] == len(score["test_keys"])
            for model_score in score["models"].values()
        )
        for score in result["scores"]
    )
    assert _fold_bytes(primary) == primary_bytes


def test_archive_reports_invalid_label_attrition_and_unavailable_folds_without_primary_drift() -> None:
    rows = _rows(n_per_season=5)
    rows.loc[(rows["season"] == 2023) & (rows["player_id"] == "p0"), "start_ktc"] = np.nan
    rows.loc[(rows["season"] == 2023) & (rows["player_id"] == "p1"), "end_ktc"] = np.nan
    rows.loc[(rows["season"] == 2023) & (rows["player_id"] == "p2"), "start_ktc"] = 0.0
    rows.loc[(rows["season"] == 2023) & (rows["player_id"] == "p3"), "end_ktc"] = np.inf
    rows.loc[(rows["season"] == 2023) & (rows["player_id"] == "p4"), "end_ktc"] = 10000.0
    primary = run_primary_folds(rows, model_factory=_factory)
    before = _fold_bytes(primary)

    result = run_ktc_diagnostic(_data(rows), mode="archive", model_factory=_factory)

    assert len(RecordingModel.instances) == 16
    assert [cohort["status"] for cohort in result["cohorts"]] == [
        "available",
        "available",
        "unavailable",
        "unavailable",
        "unavailable",
    ]
    fold_2023 = result["cohorts"][2]
    assert fold_2023["unavailable_reason"] == "empty_test_intersection"
    assert fold_2023["test_attrition"] == {
        "count": 5,
        "keys": [
            [2023, "p0"],
            [2023, "p1"],
            [2023, "p2"],
            [2023, "p3"],
            [2023, "p4"],
        ],
        "reasons": {
            "above_max_end_ktc": 1,
            "missing_start_ktc": 1,
            "missing_end_ktc": 1,
            "nonpositive_start_ktc": 1,
            "nonfinite_end_ktc": 1,
        },
    }
    for cohort in result["cohorts"][3:]:
        assert cohort["unavailable_reason"] == "empty_required_train_season"
        assert cohort["missing_train_seasons"] == [2023]
    assert len(result["fits"]) == 6
    assert [score["test_season"] for score in result["scores"]] == [2021, 2022]
    assert {prediction["season"] for prediction in result["predictions"]} == {2021, 2022}
    assert _fold_bytes(primary) == before


def test_archive_refuses_duplicate_or_mismatched_keys_before_any_fit() -> None:
    rows = _rows()
    primary = run_primary_folds(rows, model_factory=_factory)
    RecordingModel.instances.clear()
    duplicate = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(ContractRefusal, match="duplicate"):
        run_ktc_diagnostic(_data(duplicate), mode="archive", model_factory=_factory)
    assert not RecordingModel.instances

    missing_column = rows.drop(columns="end_ktc")
    with pytest.raises(ContractRefusal, match="missing columns: end_ktc"):
        run_ktc_diagnostic(_data(missing_column), mode="archive", model_factory=_factory)
    assert not RecordingModel.instances

    bad_fold = replace(primary[0], test_keys=primary[0].test_keys[:-1])
    with pytest.raises(ContractRefusal, match="primary fold test keys"):
        run_ktc_diagnostic(
            _data(rows),
            mode="archive",
            primary_folds=(bad_fold, *primary[1:]),
            model_factory=_factory,
        )
    assert not RecordingModel.instances


def test_archive_output_is_deterministic_for_shuffled_input() -> None:
    rows = _rows()
    first = run_ktc_diagnostic(_data(rows), mode="archive", model_factory=_factory)
    RecordingModel.instances.clear()
    shuffled = rows.sample(frac=1, random_state=9).reset_index(drop=True)
    second = run_ktc_diagnostic(_data(shuffled), mode="archive", model_factory=_factory)
    assert _diagnostic_bytes(first) == _diagnostic_bytes(second)


def test_archive_refuses_learned_predictions_above_their_source_cap() -> None:
    class OverCapModel(RecordingModel):
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            self.predict_frames.append(frame.copy(deep=True))
            return np.full(len(frame), float(self.kwargs["scale_max"]) + 1.0)

    with pytest.raises(ContractRefusal, match="KTC learned predictions exceed source cap"):
        run_ktc_diagnostic(
            _data(_rows()),
            mode="archive",
            model_factory=lambda **kwargs: OverCapModel(**kwargs),
        )
