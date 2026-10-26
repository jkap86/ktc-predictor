import json
import math

import numpy as np
import pandas as pd
import pytest

from predictor_v5.movement_evaluation.contract import FLAT_BAND, ContractRefusal
from predictor_v5.movement_evaluation.metrics import (
    direction,
    macro_by_season,
    score_contrast,
    score_direction_baselines,
    score_model,
    score_slices,
    target_diagnostics,
    training_majority_class,
)


def _frame(rows=None):
    if rows is None:
        rows = [
            (2020, "a", "QB", 24, 10, 100, 1000, 2000),
            (2020, "b", "RB", 25, 12, 80, 1000, 500),
            (2021, "c", "WR", 26, 8, 0, 1000, 1000),
        ]
    frame = pd.DataFrame(rows, columns=["season", "player_id", "position", "age", "games", "points", "S", "E"])
    frame["r"] = np.log(frame["E"] / frame["S"])
    frame["d"] = frame["E"] - frame["S"]
    frame["start_trades_window"] = [5] * len(frame)
    frame["end_trades_window"] = [10] * len(frame)
    frame["start_evidence"] = [2] * len(frame)
    frame["end_evidence"] = [3] * len(frame)
    frame["production_E_hat"] = ([1800, 700, 900] * ((len(frame) + 2) // 3))[: len(frame)]
    frame["no_production_E_hat"] = ([1600, 600, 1000] * ((len(frame) + 2) // 3))[: len(frame)]
    for model in ("production", "no_production"):
        frame[f"{model}_r_hat"] = np.log(frame[f"{model}_E_hat"] / frame["S"])
        frame[f"{model}_d_hat"] = frame[f"{model}_E_hat"] - frame["S"]
    frame["no_change_E_hat"] = frame["S"]
    frame["no_change_r_hat"] = 0.0
    frame["no_change_d_hat"] = 0.0
    frame["production_value_tier"] = (["low", "mid", "high"] * ((len(frame) + 2) // 3))[: len(frame)]
    return frame


def test_direction_uses_exact_symmetric_flat_band_boundaries():
    values = np.array([-FLAT_BAND, -FLAT_BAND - 1e-12, 0.0, FLAT_BAND, FLAT_BAND + 1e-12])
    assert direction(values).tolist() == ["flat", "down", "flat", "flat", "up"]


def test_hand_computed_error_and_skill_metrics_use_pooled_denominators():
    frame = _frame()
    result = score_model(frame, "production")
    log_error = np.abs(frame["production_r_hat"] - frame["r"])
    value_error = np.abs(frame["production_d_hat"] - frame["d"])
    assert result["n"] == 3
    assert result["mae_log"]["value"] == pytest.approx(log_error.mean())
    assert result["rmse_log"]["value"] == pytest.approx(np.sqrt(np.mean(log_error**2)))
    assert result["median_abs_log_error"]["value"] == pytest.approx(np.median(log_error))
    assert result["p90_abs_log_error"]["value"] == pytest.approx(np.quantile(log_error, 0.9, method="linear"))
    assert result["signed_bias_log"]["value"] == pytest.approx((frame["production_r_hat"] - frame["r"]).mean())
    assert result["mean_actual_movement"]["value"] == pytest.approx(frame["r"].mean())
    assert result["mean_predicted_movement"]["value"] == pytest.approx(frame["production_r_hat"].mean())
    assert result["mae_value"]["value"] == pytest.approx(value_error.mean())
    assert result["value_change_mae"]["value"] == result["mae_value"]["value"]
    assert result["endpoint_mae"]["value"] == result["mae_value"]["value"]
    assert result["endpoint_rmse"]["value"] == result["rmse_value"]["value"]
    assert result["skill_log"]["value"] == pytest.approx(1 - log_error.mean() / np.mean(np.abs(frame["r"])))
    assert result["skill_value"]["value"] == pytest.approx(1 - value_error.mean() / np.mean(np.abs(frame["d"])))


def test_contrast_reports_all_paired_error_skill_direction_and_rank_families():
    result = score_contrast(_frame(), "production", "no_production")
    for key in ("mae_log_difference", "rmse_log_difference", "mae_value_difference", "rmse_value_difference", "skill_log_gain", "skill_value_gain", "direction_accuracy_difference"):
        assert key in result
        assert set(result[key]) == {"value", "reason"}
    assert "movement_spearman_difference" in result
    assert "included_cells" in result["movement_spearman_difference"]
    assert "per_position" in result["movement_spearman_difference"]


def test_zero_denominators_empty_and_nonfinite_predictions_are_explicit():
    frame = _frame([(2020, "a", "QB", 24, 10, 1, 1000, 1000)])
    result = score_model(frame, "no_change")
    assert result["skill_log"] == {"value": None, "reason": "zero_baseline_error"}
    assert result["skill_value"] == {"value": None, "reason": "zero_baseline_error"}
    empty = score_model(frame.iloc[0:0].copy(), "production")
    assert empty["mae_log"] == {"value": None, "reason": "empty_cohort"}
    bad = frame.copy()
    bad["production_E_hat"] = bad["production_E_hat"].astype(float)
    bad.loc[0, "production_E_hat"] = np.inf
    with pytest.raises(ContractRefusal):
        score_model(bad, "production")


def test_metric_input_requires_sorted_unique_keys_and_finite_actuals():
    frame = _frame()
    with pytest.raises(ContractRefusal):
        score_model(frame.iloc[::-1].reset_index(drop=True), "production")
    bad = frame.copy()
    bad["r"] = bad["r"].astype(float)
    bad.loc[0, "r"] = np.nan
    with pytest.raises(ContractRefusal):
        score_model(bad, "production")
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ContractRefusal):
        score_model(duplicate, "production")


def test_metric_input_rejects_inconsistent_endpoint_and_movement_equivalents():
    bad = _frame([(2020, "a", "QB", 24, 10, 1, 1000, 2000)])
    bad.loc[0, "r"] = 0.0
    with pytest.raises(ContractRefusal):
        score_model(bad, "production")
    bad = _frame([(2020, "a", "QB", 24, 10, 1, 1000, 2000)])
    bad.loc[0, "production_d_hat"] += 1.0
    with pytest.raises(ContractRefusal):
        score_model(bad, "production")
    for column in ("production_r_hat", "production_d_hat"):
        bad = _frame([(2020, "a", "QB", 24, 10, 1, 1000, 2000)])
        bad[column] = np.inf
        with pytest.raises(ContractRefusal):
            score_model(bad, "production")


def test_score_contrast_signs_and_common_endpoint_error_are_explicit():
    result = score_contrast(_frame(), "production", "no_production")
    assert result["mae_log_difference"]["value"] == pytest.approx(
        result["left"]["mae_log"]["value"] - result["right"]["mae_log"]["value"]
    )
    assert result["positive_mae_log_improvement"]["value"] == pytest.approx(
        result["right"]["mae_log"]["value"] - result["left"]["mae_log"]["value"]
    )
    assert "left - right" in result["sign_convention"]


def test_primary_contrast_reports_relative_mae_reduction_with_comparator_denominator():
    frame = _frame(
        [
            (2020, "a", "QB", 24, 10, 1, 1.0, math.exp(1.0)),
            (2020, "b", "QB", 24, 10, 1, 1.0, math.exp(-1.0)),
        ]
    )
    for model, predicted_r in (
        ("production", np.array([0.75, -0.75])),
        ("no_production", np.array([0.5, -0.5])),
    ):
        frame[f"{model}_r_hat"] = predicted_r
        frame[f"{model}_E_hat"] = np.exp(predicted_r)
        frame[f"{model}_d_hat"] = frame[f"{model}_E_hat"] - frame["S"]

    contrast = score_contrast(frame, "production", "no_production")

    assert contrast["relative_mae_log_reduction"] == {
        "value": pytest.approx(0.5),
        "reason": None,
    }
    assert contrast["skill_log_gain"]["value"] == pytest.approx(0.25)
    assert contrast["relative_mae_log_reduction"]["value"] != pytest.approx(
        contrast["skill_log_gain"]["value"]
    )


def test_relative_mae_reduction_is_undefined_for_zero_comparator_mae():
    frame = _frame()
    frame["no_production_E_hat"] = frame["E"]
    frame["no_production_r_hat"] = frame["r"]
    frame["no_production_d_hat"] = frame["d"]

    contrast = score_contrast(frame, "production", "no_production")

    assert contrast["relative_mae_log_reduction"] == {
        "value": None,
        "reason": "zero_comparator_mae",
    }


def test_direction_baselines_majority_ties_and_mover_accuracy():
    actual = np.array([-2 * FLAT_BAND, 0, 2 * FLAT_BAND])
    assert training_majority_class(np.array([-1, 1, 0])) == "flat"
    assert training_majority_class(np.array([-1, -2 * FLAT_BAND, 2 * FLAT_BAND])) == "down"
    frame = _frame()
    frame["E"] = frame["S"] * np.exp(actual)
    frame["r"] = actual
    frame["d"] = frame["E"] - frame["S"]
    result = score_direction_baselines(frame, "flat")
    assert result["no_change"]["direction_accuracy"]["value"] == pytest.approx(1 / 3)
    assert result["no_change"]["mover_accuracy"]["value"] == 0.0
    assert result["majority"]["direction_accuracy"]["value"] == pytest.approx(1 / 3)


def test_direction_confusion_and_supported_balanced_accuracy_have_absent_nulls():
    frame = _frame()
    values = np.array([-2 * FLAT_BAND, 0.0, 2 * FLAT_BAND])
    frame["E"] = frame["S"] * np.exp(values)
    frame["r"] = values
    frame["d"] = frame["E"] - frame["S"]
    frame["production_E_hat"] = frame["S"] * np.exp(values)
    frame["production_r_hat"] = values
    frame["production_d_hat"] = frame["production_E_hat"] - frame["S"]
    score = score_model(frame, "production")["direction"]
    assert score["confusion_matrix"] == {"down": {"down": 1, "flat": 0, "up": 0}, "flat": {"down": 0, "flat": 1, "up": 0}, "up": {"down": 0, "flat": 0, "up": 1}}
    assert score["balanced_accuracy"]["value"] == 1.0
    absent_frame = frame.copy()
    absent_frame["E"] = absent_frame["S"]
    absent_frame["r"] = 0.0
    absent_frame["d"] = 0.0
    absent = score_direction_baselines(absent_frame, "flat")["no_change"]
    assert absent["class_recalls"]["down"] == {"value": None, "reason": "absent_class"}


def test_calibration_bins_rank_and_tail_metrics_include_empty_cells_and_ties():
    frame = _frame()
    result = score_model(frame, "production")
    assert len(result["calibration"]["bins"]) == 8
    assert result["calibration"]["ols"]["slope"]["value"] is not None
    assert result["rank"]["spearman"]["value"] is not None
    assert "season_position" in result["rank"]
    assert result["tail"]["largest_errors"] == sorted(
        result["tail"]["largest_errors"], key=lambda row: (-row["absolute_log_error"], row["season"], row["player_id"])
    )
    assert result["tail"]["large_move"]["threshold"] == pytest.approx(math.log(1.5))
    assert result["tail"]["large_move"]["retrospective"] is True
    assert "direction" in result["tail"]["large_move"]


def test_constant_prediction_rank_and_ols_are_nullable():
    frame = _frame()
    frame["production_E_hat"] = frame["S"]
    frame["production_r_hat"] = 0.0
    frame["production_d_hat"] = 0.0
    result = score_model(frame, "production")
    assert result["rank"]["spearman"] == {"value": None, "reason": "constant_prediction"}
    assert result["calibration"]["ols"]["slope"] == {"value": None, "reason": "constant_prediction"}
    assert result["tail"]["top_bottom"] == {"value": None, "reason": "constant_prediction"}


def test_rank_uses_average_ties_and_top_k_uses_player_id_boundary_order():
    frame = _frame([
        (2020, "b", "QB", 24, 10, 100, 1000, 1200),
        (2020, "a", "QB", 24, 10, 100, 1000, 1100),
        (2020, "c", "QB", 24, 10, 100, 1000, 900),
    ])
    frame["production_E_hat"] = [1100.0, 1100.0, 900.0]
    frame["production_r_hat"] = np.log(frame["production_E_hat"] / frame["S"])
    frame["production_d_hat"] = frame["production_E_hat"] - frame["S"]
    frame = frame.sort_values(["season", "player_id"]).reset_index(drop=True)
    rank = score_model(frame, "production")["rank"]
    assert rank["spearman"]["value"] == pytest.approx(np.corrcoef(pd.Series(frame["r"]).rank(method="average"), pd.Series(frame["production_r_hat"]).rank(method="average"))[0, 1])
    cell = rank["top_bottom"]["cells"][0]
    assert cell["k"] == 1 and cell["top_boundary_tie_count"] == 2
    assert cell["top_mean_actual_movement"] == pytest.approx(np.log(1100 / 1000))


def test_rank_abstains_only_in_constant_season_position_cells():
    frame = _frame(
        [
            (2020, "qb-a", "QB", 24, 10, 100, 1000, 1300),
            (2020, "qb-b", "QB", 24, 10, 100, 1000, 700),
            (2020, "rb-a", "RB", 24, 10, 100, 1000, 1400),
            (2020, "rb-b", "RB", 24, 10, 100, 1000, 600),
        ]
    )
    frame["production_E_hat"] = [1200.0, 1200.0, 1350.0, 650.0]
    frame["production_r_hat"] = np.log(frame["production_E_hat"] / frame["S"])
    frame["production_d_hat"] = frame["production_E_hat"] - frame["S"]

    rank = score_model(frame, "production")["rank"]
    cells = {cell["cell"]: cell for cell in rank["top_bottom"]["cells"]}

    constant = cells["2020:QB"]
    assert constant["n"] == 2
    assert constant["k"] == 1
    assert constant["selection_reason"] == "constant_prediction"
    assert constant["top_overlap"] is None
    assert constant["bottom_overlap"] is None
    assert constant["top_mean_actual_movement"] is None
    assert constant["bottom_mean_actual_movement"] is None
    assert rank["season_position"]["2020:QB"]["spearman"] == {
        "value": None,
        "reason": "constant_prediction",
    }

    scored = cells["2020:RB"]
    assert scored["selection_reason"] is None
    assert scored["top_overlap"] == 1.0
    assert scored["bottom_overlap"] == 1.0


def test_large_move_precision_recall_and_largest_error_limit_are_deterministic():
    rows = [(2020, f"p{i:02d}", "QB", 24, 10, 100, 1000, 2000 if i == 0 else 500 if i == 1 else 1000) for i in range(25)]
    frame = _frame(rows)
    frame["production_E_hat"] = 1000.0
    frame["production_r_hat"] = 0.0
    frame["production_d_hat"] = 0.0
    tail = score_model(frame, "production")["tail"]
    assert tail["large_move"]["positive_recall"]["value"] == 0.0
    assert tail["large_move"]["positive_precision"] == {"value": None, "reason": "no_predicted_positives"}
    assert len(tail["largest_errors"]) == 20
    assert tail["largest_errors"] == sorted(tail["largest_errors"], key=lambda x: (-x["absolute_log_error"], x["season"], x["player_id"]))


def test_slices_emit_requested_empty_cells_and_support_descriptions():
    frame = _frame()
    frame["value_tier"] = ["low", "mid", "high"]
    frame["production_value_tier"] = frame["value_tier"]
    result = score_slices(frame, ("production", "no_change"))
    assert "overall" in result
    assert "season" in result and "2022" in result["season"]["production"]
    assert "position" in result and "TE" in result["position"]["production"]
    assert "season_position" in result
    assert "value_tier" in result
    assert "start_liquidity" in result["production"]
    assert result["production"]["position"]["TE"]["low_support"] is True
    assert result["production"]["endpoint_liquidity"]["25+"]["retrospective"] is True
    assert result["production"]["overall"]["low_support"] is True
    assert result["production"]["start_liquidity"]["5-9"]["n"] == 3


def test_slices_refuse_missing_or_malformed_tiers_and_liquidity():
    with pytest.raises(ContractRefusal):
        score_slices(_frame(), ("production",))
    bad_tier = _frame()
    bad_tier["value_tier"] = ["low", "bogus", "high"]
    with pytest.raises(ContractRefusal):
        score_slices(bad_tier, ("production",))
    bad_liquidity = _frame()
    bad_liquidity["value_tier"] = ["low", "mid", "high"]
    bad_liquidity["start_trades_window"] = [5, "bad", 25]
    with pytest.raises(ContractRefusal):
        score_slices(bad_liquidity, ("production",))


def test_liquidity_requires_primary_domain_and_keeps_exact_band_boundaries():
    below = _frame()
    below["value_tier"] = ["low", "mid", "high"]
    below["start_trades_window"] = [4, 5, 9]
    with pytest.raises(ContractRefusal):
        score_slices(below, ("production",))
    below = _frame()
    below["value_tier"] = ["low", "mid", "high"]
    below["end_trades_window"] = [4, 5, 9]
    with pytest.raises(ContractRefusal):
        score_slices(below, ("production",))
    rows = [(2020, f"p{i}", "QB", 24, 10, 100, 1000, 1100 + i) for i in range(5)]
    boundary = _frame(rows)
    boundary["value_tier"] = ["low", "mid", "high", "low", "mid"]
    boundary["start_trades_window"] = [5, 9, 10, 24, 25]
    boundary["end_trades_window"] = [5, 9, 10, 24, 25]
    sliced = score_slices(boundary, ("production",))["production"]["start_liquidity"]
    assert [sliced[key]["n"] for key in ("5-9", "10-24", "25+")] == [2, 2, 1]


def test_nonfinite_descriptive_values_and_macro_values_refuse_and_outputs_serialize():
    bad = _frame()
    bad["value_tier"] = ["low", "mid", "high"]
    bad["age"] = [24.0, np.inf, 26.0]
    with pytest.raises(ContractRefusal):
        score_model(bad, "production")
    bad = _frame()
    bad["age"] = [24.0, np.nan, 26.0]
    with pytest.raises(ContractRefusal):
        score_model(bad, "production")
    bad = _frame()
    bad["start_trades_window"] = [5, "malformed", 25]
    with pytest.raises(ContractRefusal):
        score_model(bad, "production")
    with pytest.raises(ContractRefusal):
        macro_by_season([{"mae_log": {"value": np.inf, "reason": None}}])
    result = score_model(_frame(), "production")
    json.dumps(result, allow_nan=False)


def test_cap_count_requires_exact_cap_and_accepts_uncapped_ratio_endpoint():
    capped = _frame([(2020, "a", "QB", 24, 10, 1, 1000, 2000)])
    capped["production_E_hat"] = float(10000)
    capped["production_r_hat"] = np.log(capped["production_E_hat"] / capped["S"])
    capped["production_d_hat"] = capped["production_E_hat"] - capped["S"]
    assert score_model(capped, "production")["clipping"]["at_cap_count"] == 1
    below = capped.copy()
    below["production_E_hat"] = 9999.0
    below["production_r_hat"] = np.log(below["production_E_hat"] / below["S"])
    below["production_d_hat"] = below["production_E_hat"] - below["S"]
    assert score_model(below, "production")["clipping"]["at_cap_count"] == 0
    above = capped.copy()
    above["production_E_hat"] = 10001.0
    above["production_r_hat"] = np.log(above["production_E_hat"] / above["S"])
    above["production_d_hat"] = above["production_E_hat"] - above["S"]
    above_score = score_model(above, "production")
    assert above_score["clipping"]["at_cap_count"] == 0


def test_per_position_rank_contrast_has_membership_and_numeric_aggregates():
    rows = []
    for position in ("QB", "RB", "TE", "WR"):
        for suffix, end in (("a", 1200), ("b", 900)):
            rows.append((2020, f"{position}-{suffix}", position, 24, 10, 100, 1000, end))
    frame = _frame(rows)
    frame["no_production_E_hat"] = frame["E"]
    frame["no_production_r_hat"] = frame["r"]
    frame["no_production_d_hat"] = frame["d"]
    contrast = score_contrast(frame, "production", "no_production")
    rank = contrast["movement_spearman_difference"]
    assert rank["n_cells"] == 4
    assert rank["included_cells"] == [f"2020:{position}" for position in ("QB", "RB", "TE", "WR")]
    for position in ("QB", "RB", "TE", "WR"):
        assert rank["per_position"][position]["n_cells"] == 1
        assert rank["per_position"][position]["included_cells"] == [f"2020:{position}"]
        expected = (
            contrast["left"]["rank"]["season_position"][f"2020:{position}"]["spearman"]["value"]
            - contrast["right"]["rank"]["season_position"][f"2020:{position}"]["spearman"]["value"]
        )
        assert rank["per_position"][position]["value"] == pytest.approx(expected)


def test_target_diagnostics_are_labeled_scale_drift_observations():
    result = target_diagnostics(_frame())
    assert result["interpretation"] == "scale_drift_observation_not_normalization_or_correction"
    assert result["season"]["2020"]["start_median"]["value"] == 1000
    assert result["season"]["2020"]["end_median"]["value"] == 1250


def test_macro_by_season_does_not_average_undefined_values():
    result = macro_by_season([
        {"test_season": 2021, "skill_log": {"value": 0.2, "reason": None}},
        {"test_season": 2022, "skill_log": {"value": None, "reason": "zero_baseline_error"}},
    ])
    assert result["skill_log"]["value"] == pytest.approx(0.2)
    assert result["skill_log"]["n_defined"] == 1
    assert result["skill_log"]["n_total"] == 2
