#!/usr/bin/env python3
"""
Comprehensive Prediction vs Actuals Analysis for KTC Model.

This script analyzes model predictions to find patterns, systematic errors,
and improvement opportunities across multiple dimensions:

1. Error Distribution Analysis - normality, outliers, heteroscedasticity
2. Segmented Error Analysis - by position, age, KTC tier, year
3. Biggest Misses Deep Dive - breakouts and busts
4. Feature Residual Analysis - nonlinear patterns
5. Breakout/Bust Detection - why we miss major value changes
6. Temporal Analysis - accuracy over time
7. Position-Specific Insights - per-position deep dives
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import Optional
from collections import defaultdict
from scipy import stats

from app.services.data_loader import (
    get_data_loader,
    calculate_derived_features,
    get_age_bracket,
    get_position_age_factor,
    get_position_ktc_ceiling,
    get_draft_capital_score,
)
from app.services.model_service import get_model_service


def generate_all_predictions(model_type: str = "xgb_calibrated_breakout") -> pd.DataFrame:
    """Generate predictions for all training pairs with full metadata.

    Args:
        model_type: Which model to use:
            - "xgb_base": Base XGBoost predictor (no calibration/boost)
            - "xgb_calibrated_breakout": XGBoost with calibration + breakout (NOTE: has _apply_performance_floor bug)
            - "calibrated": Ridge with tier calibration
            - "base": Base Ridge model
    """
    model_service = get_model_service()
    model_service.initialize()

    # Select predictor based on model type
    if model_type == "xgb_calibrated_breakout":
        model_service.initialize_xgb_calibrated_breakout()
        predictor = model_service.xgb_calibrated_breakout_predictor
        print("Using XGBCalibratedBreakoutPredictor (primary model)")
    elif model_type == "xgb_base":
        # Use base XGBoost model - train it on the fly
        from app.models.predictor import XGBKTCPredictor
        dl = get_data_loader()
        X, y = dl.get_feature_matrix()
        predictor = XGBKTCPredictor()
        predictor.train(X, y, use_log_transform=True)
        print("Using base XGBKTCPredictor")
    elif model_type == "calibrated":
        model_service.initialize_calibrated()
        predictor = model_service.calibrated_predictor
        print("Using CalibratedPredictor for tier-based adjustments")
    else:
        predictor = model_service.predictor
        print("Using base KTCPredictor (Ridge)")

    data_loader = get_data_loader()
    df = data_loader.get_training_dataframe()
    df = df.sort_values(["player_id", "year"])

    results = []

    for player_id, player_df in df.groupby("player_id"):
        player_df = player_df.sort_values("year")
        seasons = player_df.to_dict("records")

        for i in range(len(seasons) - 1):
            current = seasons[i]
            next_s = seasons[i + 1]

            # Skip non-consecutive years
            if next_s["year"] != current["year"] + 1:
                continue
            # Skip invalid KTC
            if current["end_ktc"] <= 0 or next_s["end_ktc"] <= 0:
                continue

            position = current["position"]
            current_ktc = current["end_ktc"]
            next_age = current["age"] + 1
            games_played = current["games_played"]
            age_bracket = get_age_bracket(next_age, position)

            # Calculate derived features
            derived = calculate_derived_features(current)

            # Calculate breakout detection features
            fp = current["fantasy_points"]
            fp_per_game = fp / max(games_played, 1)
            last_4_vs_season = current.get("last_4_vs_season", 0)
            fp_second_half_ratio = derived.get("fp_second_half_ratio", 1)
            momentum_score = last_4_vs_season * fp_second_half_ratio
            ktc_ceiling = get_position_ktc_ceiling(position)
            ktc_upside_ratio = (ktc_ceiling - current_ktc) / ktc_ceiling if ktc_ceiling > 0 else 0

            # Draft capital
            draft_round = current.get("draft_round")
            draft_pick = current.get("draft_pick")
            draft_capital_score = get_draft_capital_score(draft_round, draft_pick)

            # Build features dict
            features = model_service._build_features_dict(
                current, derived, position, current_ktc, next_age, games_played, 0.0
            )

            # Model predicts ratio (next_ktc / current_ktc), convert to absolute
            predicted_ratio = predictor.predict(features)
            predicted = predicted_ratio * current_ktc
            actual = next_s["end_ktc"]
            error = predicted - actual

            # KTC tier
            if current_ktc < 2000:
                ktc_tier = "low"
            elif current_ktc < 5000:
                ktc_tier = "mid"
            elif current_ktc < 7500:
                ktc_tier = "high"
            else:
                ktc_tier = "elite"

            results.append({
                # Identifiers
                "player_id": player_id,
                "name": current["name"],
                "position": position,
                "source_year": current["year"],
                "target_year": next_s["year"],
                # Core values
                "current_ktc": current_ktc,
                "actual_ktc": actual,
                "predicted_ktc": predicted,
                "error": error,
                "abs_error": abs(error),
                "pct_error": (error / actual * 100) if actual > 0 else 0,
                # Segments
                "age": next_age,
                "age_bracket": age_bracket,
                "ktc_tier": ktc_tier,
                # Key features
                "games_played": games_played,
                "fantasy_points": fp,
                "fp_per_game": fp_per_game,
                "momentum_score": momentum_score,
                "ktc_upside_ratio": ktc_upside_ratio,
                "ktc_season_trend": derived.get("ktc_season_trend", 0),
                "draft_capital_score": draft_capital_score,
                "pff_overall_grade": current.get("pff_overall_grade", 0),
                "fp_second_half_ratio": fp_second_half_ratio,
                # Actual changes
                "actual_change": actual - current_ktc,
                "actual_change_pct": ((actual - current_ktc) / current_ktc * 100) if current_ktc > 0 else 0,
                "predicted_change": predicted - current_ktc,
            })

    return pd.DataFrame(results)


def analyze_error_distribution(df: pd.DataFrame) -> dict:
    """Part 1: Analyze error distribution shape, outliers, normality."""
    errors = df["error"].values
    abs_errors = df["abs_error"].values

    # Basic stats
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    skewness = stats.skew(errors)
    kurtosis = stats.kurtosis(errors)

    # Outlier analysis
    outlier_threshold = 2 * std_error
    outliers_high = df[df["error"] > outlier_threshold]
    outliers_low = df[df["error"] < -outlier_threshold]

    # Over vs under prediction
    over_predict = df[df["error"] > 0]
    under_predict = df[df["error"] < 0]

    # Normality test (Shapiro-Wilk on sample for speed)
    sample = errors[:5000] if len(errors) > 5000 else errors
    _, normality_p = stats.shapiro(sample)

    # MAE and MAPE
    mae = np.mean(abs_errors)
    mape = np.mean(np.abs(df["pct_error"]))

    # Percentiles
    percentiles = np.percentile(abs_errors, [25, 50, 75, 90, 95, 99])

    return {
        "summary_stats": {
            "n_samples": len(df),
            "mean_error": round(mean_error, 1),
            "median_error": round(median_error, 1),
            "std_error": round(std_error, 1),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurtosis, 3),
            "mae": round(mae, 1),
            "mape": round(mape, 2),
        },
        "outliers": {
            "threshold": round(outlier_threshold, 1),
            "n_high_outliers": len(outliers_high),
            "n_low_outliers": len(outliers_low),
            "outlier_pct": round((len(outliers_high) + len(outliers_low)) / len(df) * 100, 2),
        },
        "direction": {
            "over_predict_count": len(over_predict),
            "under_predict_count": len(under_predict),
            "over_predict_pct": round(len(over_predict) / len(df) * 100, 2),
            "over_predict_avg": round(over_predict["error"].mean(), 1) if len(over_predict) > 0 else 0,
            "under_predict_avg": round(under_predict["error"].mean(), 1) if len(under_predict) > 0 else 0,
        },
        "normality": {
            "shapiro_p_value": round(normality_p, 4),
            "is_normal": normality_p > 0.05,
        },
        "percentiles": {
            "p25": round(percentiles[0], 1),
            "p50": round(percentiles[1], 1),
            "p75": round(percentiles[2], 1),
            "p90": round(percentiles[3], 1),
            "p95": round(percentiles[4], 1),
            "p99": round(percentiles[5], 1),
        },
    }


def analyze_segments(df: pd.DataFrame) -> dict:
    """Part 2: Segmented error analysis by position, age, KTC tier, year."""

    def calc_segment_metrics(subset: pd.DataFrame) -> dict:
        if len(subset) == 0:
            return None

        errors = subset["error"].values
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        n = len(subset)

        # 95% CI for bias
        se = std_error / np.sqrt(n) if n > 1 else 0
        ci_low = mean_error - 1.96 * se
        ci_high = mean_error + 1.96 * se

        # Significant bias if CI doesn't cross 0
        significant_bias = ci_low > 0 or ci_high < 0

        return {
            "n_samples": n,
            "mae": round(np.mean(np.abs(errors)), 1),
            "bias": round(mean_error, 1),
            "std": round(std_error, 1),
            "ci_95": (round(ci_low, 1), round(ci_high, 1)),
            "significant_bias": significant_bias,
            "bias_direction": "over" if mean_error > 0 else "under",
        }

    segments = {}

    # By position
    segments["by_position"] = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        subset = df[df["position"] == pos]
        metrics = calc_segment_metrics(subset)
        if metrics:
            segments["by_position"][pos] = metrics

    # By age bracket
    segments["by_age_bracket"] = {}
    for bracket in ["young", "prime", "declining"]:
        subset = df[df["age_bracket"] == bracket]
        metrics = calc_segment_metrics(subset)
        if metrics:
            segments["by_age_bracket"][bracket] = metrics

    # By KTC tier
    segments["by_ktc_tier"] = {}
    for tier in ["low", "mid", "high", "elite"]:
        subset = df[df["ktc_tier"] == tier]
        metrics = calc_segment_metrics(subset)
        if metrics:
            segments["by_ktc_tier"][tier] = metrics

    # By year
    segments["by_year"] = {}
    for year in sorted(df["target_year"].unique()):
        subset = df[df["target_year"] == year]
        metrics = calc_segment_metrics(subset)
        if metrics:
            segments["by_year"][int(year)] = metrics

    # Cross: Position × Age
    segments["position_x_age"] = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        for bracket in ["young", "prime", "declining"]:
            subset = df[(df["position"] == pos) & (df["age_bracket"] == bracket)]
            if len(subset) >= 10:  # Require minimum sample size
                metrics = calc_segment_metrics(subset)
                if metrics:
                    segments["position_x_age"][f"{pos}_{bracket}"] = metrics

    # Cross: Position × KTC tier
    segments["position_x_ktc"] = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        for tier in ["low", "mid", "high", "elite"]:
            subset = df[(df["position"] == pos) & (df["ktc_tier"] == tier)]
            if len(subset) >= 10:
                metrics = calc_segment_metrics(subset)
                if metrics:
                    segments["position_x_ktc"][f"{pos}_{tier}"] = metrics

    # Find significant biases
    significant_biases = []
    for segment_name, segment_data in segments.items():
        if isinstance(segment_data, dict):
            for key, metrics in segment_data.items():
                if metrics and metrics.get("significant_bias"):
                    significant_biases.append({
                        "segment": segment_name,
                        "key": key,
                        "bias": metrics["bias"],
                        "direction": metrics["bias_direction"],
                        "n_samples": metrics["n_samples"],
                    })

    segments["significant_biases"] = sorted(significant_biases, key=lambda x: abs(x["bias"]), reverse=True)

    return segments


def analyze_biggest_misses(df: pd.DataFrame, n: int = 25) -> dict:
    """Part 3: Deep dive into biggest prediction errors."""

    # Sort by error for over/under predictions
    df_sorted_over = df.nlargest(n, "error")
    df_sorted_under = df.nsmallest(n, "error")

    # Breakouts missed: actual >> predicted by 1500+
    breakouts = df[(df["actual_change"] - df["predicted_change"]) > 1500]
    breakouts = breakouts.nlargest(min(n, len(breakouts)), "actual_change")

    # Busts missed: actual << predicted by 1500+
    busts = df[(df["predicted_change"] - df["actual_change"]) > 1500]
    busts = busts.nlargest(min(n, len(busts)), "error")

    def analyze_group(subset: pd.DataFrame, group_name: str) -> dict:
        if len(subset) == 0:
            return {"count": 0}

        return {
            "count": len(subset),
            "position_dist": subset["position"].value_counts().to_dict(),
            "age_bracket_dist": subset["age_bracket"].value_counts().to_dict(),
            "ktc_tier_dist": subset["ktc_tier"].value_counts().to_dict(),
            "avg_games_played": round(subset["games_played"].mean(), 1),
            "avg_fp_per_game": round(subset["fp_per_game"].mean(), 1),
            "avg_momentum_score": round(subset["momentum_score"].mean(), 3),
            "avg_draft_capital": round(subset["draft_capital_score"].mean(), 1),
            "top_examples": subset.head(10)[[
                "name", "position", "source_year", "age",
                "current_ktc", "predicted_ktc", "actual_ktc", "error"
            ]].to_dict("records"),
        }

    return {
        "over_predictions": analyze_group(df_sorted_over, "over"),
        "under_predictions": analyze_group(df_sorted_under, "under"),
        "missed_breakouts": analyze_group(breakouts, "breakouts"),
        "missed_busts": analyze_group(busts, "busts"),
    }


def analyze_feature_residuals(df: pd.DataFrame) -> dict:
    """Part 4: Analyze error correlation with key features."""

    features_to_analyze = [
        "current_ktc", "age", "ktc_upside_ratio", "momentum_score",
        "fp_per_game", "games_played", "draft_capital_score",
        "pff_overall_grade", "fp_second_half_ratio", "ktc_season_trend"
    ]

    residual_analysis = {}

    for feature in features_to_analyze:
        if feature not in df.columns:
            continue

        # Handle missing values
        valid_mask = df[feature].notna() & (df[feature] != 0)
        subset = df[valid_mask]

        if len(subset) < 50:
            continue

        # Skip if feature is constant (no variance)
        if subset[feature].std() < 1e-10:
            continue

        # Correlation with error
        corr, p_value = stats.pearsonr(subset[feature], subset["error"])

        # Spearman for nonlinear relationships
        spearman_corr, spearman_p = stats.spearmanr(subset[feature], subset["error"])

        # Bin analysis (quintiles)
        try:
            subset = subset.copy()
            subset["quintile"] = pd.qcut(subset[feature], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
            quintile_errors = subset.groupby("quintile")["error"].agg(["mean", "std", "count"])
            quintile_dict = {
                str(idx): {
                    "mean_error": round(row["mean"], 1),
                    "std_error": round(row["std"], 1),
                    "count": int(row["count"]),
                }
                for idx, row in quintile_errors.iterrows()
            }
        except Exception:
            quintile_dict = {}

        residual_analysis[feature] = {
            "pearson_corr": round(corr, 4),
            "pearson_p": round(p_value, 4),
            "spearman_corr": round(spearman_corr, 4),
            "spearman_p": round(spearman_p, 4),
            "significant": p_value < 0.05,
            "quintile_errors": quintile_dict,
            "interpretation": interpret_correlation(feature, corr, p_value),
        }

    # Sort by absolute correlation
    sorted_features = sorted(
        residual_analysis.items(),
        key=lambda x: abs(x[1]["pearson_corr"]),
        reverse=True
    )

    return {
        "by_feature": dict(sorted_features),
        "significant_correlations": [
            f for f, data in sorted_features if data["significant"]
        ],
    }


def interpret_correlation(feature: str, corr: float, p_value: float) -> str:
    """Generate interpretation of error-feature correlation."""
    if p_value >= 0.05:
        return "No significant relationship"

    direction = "over-predicting" if corr > 0 else "under-predicting"
    strength = "strongly" if abs(corr) > 0.3 else "moderately" if abs(corr) > 0.15 else "slightly"

    interpretations = {
        "current_ktc": f"Model is {strength} {direction} high-value players" if corr > 0 else f"Model is {strength} {direction} low-value players",
        "age": f"Model is {strength} {direction} older players" if corr > 0 else f"Model is {strength} {direction} younger players",
        "ktc_upside_ratio": f"Model is {strength} {direction} players with upside room" if corr > 0 else f"Model may be missing breakout potential",
        "momentum_score": f"Model is {strength} {direction} players with late-season surge" if corr > 0 else f"Model may be under-weighting momentum",
        "fp_per_game": f"Model is {strength} {direction} efficient players" if corr > 0 else f"Model may be under-weighting per-game production",
        "games_played": f"Model is {strength} {direction} players with more games" if corr > 0 else f"Model may be penalizing missed games too much",
        "draft_capital_score": f"Model is {strength} {direction} high-pedigree players" if corr > 0 else f"Model may be over-weighting draft capital",
        "pff_overall_grade": f"Model is {strength} {direction} high-grade players" if corr > 0 else f"Model may be missing quality signal from PFF",
    }

    return interpretations.get(feature, f"Model is {strength} {direction} for high {feature}")


def analyze_breakout_detection(df: pd.DataFrame) -> dict:
    """Part 5: Analyze why we miss breakouts and busts."""

    # Define breakouts and busts
    df = df.copy()
    df["is_breakout"] = (df["actual_change_pct"] > 50).astype(int)
    df["is_bust"] = (df["actual_change_pct"] < -30).astype(int)

    # Did we catch them?
    df["caught_breakout"] = (df["is_breakout"] == 1) & (df["predicted_change"] > df["current_ktc"] * 0.3)
    df["caught_bust"] = (df["is_bust"] == 1) & (df["predicted_change"] < -df["current_ktc"] * 0.15)

    breakouts = df[df["is_breakout"] == 1]
    busts = df[df["is_bust"] == 1]

    # Detection rates
    breakout_detection_rate = breakouts["caught_breakout"].mean() * 100 if len(breakouts) > 0 else 0
    bust_detection_rate = busts["caught_bust"].mean() * 100 if len(busts) > 0 else 0

    # Compare features: missed vs caught breakouts
    missed_breakouts = breakouts[~breakouts["caught_breakout"]]
    caught_breakouts = breakouts[breakouts["caught_breakout"]]

    feature_comparison = {}
    compare_features = ["age", "games_played", "fp_per_game", "momentum_score",
                       "ktc_upside_ratio", "draft_capital_score", "current_ktc"]

    for feature in compare_features:
        if feature not in df.columns:
            continue

        missed_mean = missed_breakouts[feature].mean() if len(missed_breakouts) > 0 else 0
        caught_mean = caught_breakouts[feature].mean() if len(caught_breakouts) > 0 else 0

        # T-test if enough samples
        if len(missed_breakouts) >= 10 and len(caught_breakouts) >= 10:
            _, p_value = stats.ttest_ind(
                missed_breakouts[feature].dropna(),
                caught_breakouts[feature].dropna()
            )
            significant = p_value < 0.05
        else:
            p_value = None
            significant = False

        feature_comparison[feature] = {
            "missed_mean": round(missed_mean, 2),
            "caught_mean": round(caught_mean, 2),
            "difference": round(caught_mean - missed_mean, 2),
            "p_value": round(p_value, 4) if p_value else None,
            "significant": significant,
        }

    return {
        "breakouts": {
            "total": len(breakouts),
            "caught": int(breakouts["caught_breakout"].sum()),
            "missed": len(breakouts) - int(breakouts["caught_breakout"].sum()),
            "detection_rate": round(breakout_detection_rate, 1),
        },
        "busts": {
            "total": len(busts),
            "caught": int(busts["caught_bust"].sum()),
            "missed": len(busts) - int(busts["caught_bust"].sum()),
            "detection_rate": round(bust_detection_rate, 1),
        },
        "feature_differences": feature_comparison,
        "missed_breakout_examples": missed_breakouts.head(10)[[
            "name", "position", "source_year", "age", "current_ktc",
            "actual_change_pct", "predicted_change", "momentum_score"
        ]].to_dict("records") if len(missed_breakouts) > 0 else [],
    }


def analyze_directional_accuracy(df: pd.DataFrame) -> dict:
    """Part 6: Analyze directional accuracy (did we predict up/down correctly?)."""
    df = df.copy()

    # Actual direction: did KTC go up, down, or stay flat?
    df["actual_direction"] = "flat"
    df.loc[df["actual_change"] > 100, "actual_direction"] = "up"
    df.loc[df["actual_change"] < -100, "actual_direction"] = "down"

    # Predicted direction
    df["pred_direction"] = "flat"
    df.loc[df["predicted_change"] > 100, "pred_direction"] = "up"
    df.loc[df["predicted_change"] < -100, "pred_direction"] = "down"

    # Overall directional accuracy
    df["direction_correct"] = df["actual_direction"] == df["pred_direction"]
    overall_accuracy = df["direction_correct"].mean() * 100

    # Up/down specific accuracy
    actual_up = df[df["actual_direction"] == "up"]
    actual_down = df[df["actual_direction"] == "down"]

    up_accuracy = actual_up["direction_correct"].mean() * 100 if len(actual_up) > 0 else 0
    down_accuracy = actual_down["direction_correct"].mean() * 100 if len(actual_down) > 0 else 0

    # By segment
    segment_accuracy = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        subset = df[df["position"] == pos]
        if len(subset) >= 20:
            segment_accuracy[pos] = round(subset["direction_correct"].mean() * 100, 1)

    for tier in ["low", "mid", "high", "elite"]:
        subset = df[df["ktc_tier"] == tier]
        if len(subset) >= 20:
            segment_accuracy[f"ktc_{tier}"] = round(subset["direction_correct"].mean() * 100, 1)

    for bracket in ["young", "prime", "declining"]:
        subset = df[df["age_bracket"] == bracket]
        if len(subset) >= 20:
            segment_accuracy[f"age_{bracket}"] = round(subset["direction_correct"].mean() * 100, 1)

    return {
        "overall_accuracy": round(overall_accuracy, 1),
        "up_accuracy": round(up_accuracy, 1),
        "down_accuracy": round(down_accuracy, 1),
        "n_actual_up": len(actual_up),
        "n_actual_down": len(actual_down),
        "n_actual_flat": len(df[df["actual_direction"] == "flat"]),
        "by_segment": segment_accuracy,
    }


def analyze_temporal(df: pd.DataFrame) -> dict:
    """Part 7: Analyze model accuracy over time."""

    yearly_metrics = []

    for year in sorted(df["target_year"].unique()):
        subset = df[df["target_year"] == year]

        errors = subset["error"].values
        mae = np.mean(np.abs(errors))
        bias = np.mean(errors)
        std = np.std(errors)

        # Error percentiles
        percentiles = np.percentile(np.abs(errors), [25, 50, 75, 90])

        yearly_metrics.append({
            "year": int(year),
            "n_samples": len(subset),
            "mae": round(mae, 1),
            "bias": round(bias, 1),
            "std": round(std, 1),
            "p25": round(percentiles[0], 1),
            "p50": round(percentiles[1], 1),
            "p75": round(percentiles[2], 1),
            "p90": round(percentiles[3], 1),
        })

    # Trend analysis
    years = [m["year"] for m in yearly_metrics]
    maes = [m["mae"] for m in yearly_metrics]

    if len(years) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, maes)
        trend = {
            "slope": round(slope, 2),
            "r_squared": round(r_value ** 2, 3),
            "p_value": round(p_value, 4),
            "direction": "degrading" if slope > 0 else "improving",
            "significant": p_value < 0.05,
        }
    else:
        trend = {"note": "Not enough years for trend analysis"}

    return {
        "by_year": yearly_metrics,
        "trend": trend,
    }


def analyze_position_specific(df: pd.DataFrame) -> dict:
    """Part 7: Position-specific deep dives."""

    position_analysis = {}

    for pos in ["QB", "RB", "WR", "TE"]:
        subset = df[df["position"] == pos].copy()
        if len(subset) < 20:
            continue

        # Basic metrics
        errors = subset["error"].values
        mae = np.mean(np.abs(errors))
        bias = np.mean(errors)

        # Age curve analysis
        age_errors = []
        for age in range(21, 36):
            age_subset = subset[subset["age"] == age]
            if len(age_subset) >= 5:
                age_errors.append({
                    "age": age,
                    "n_samples": len(age_subset),
                    "mae": round(np.mean(np.abs(age_subset["error"])), 1),
                    "bias": round(np.mean(age_subset["error"]), 1),
                })

        # Feature correlations with error (position-specific)
        feature_corrs = {}
        for feature in ["fp_per_game", "games_played", "momentum_score", "current_ktc"]:
            if feature in subset.columns:
                valid = subset[subset[feature].notna()]
                if len(valid) >= 30:
                    corr, p = stats.pearsonr(valid[feature], valid["error"])
                    feature_corrs[feature] = {
                        "correlation": round(corr, 3),
                        "p_value": round(p, 4),
                        "significant": p < 0.05,
                    }

        # Top 5 biggest misses
        top_misses = subset.nlargest(5, "abs_error")[[
            "name", "source_year", "age", "current_ktc",
            "predicted_ktc", "actual_ktc", "error"
        ]].to_dict("records")

        # Special analysis for RB decline
        if pos == "RB":
            aging_rbs = subset[subset["age"] >= 27]
            if len(aging_rbs) >= 10:
                aging_bias = np.mean(aging_rbs["error"])
                aging_analysis = {
                    "n_samples": len(aging_rbs),
                    "bias": round(aging_bias, 1),
                    "over_predicting": aging_bias > 0,
                    "interpretation": "Model is over-predicting aging RB value" if aging_bias > 100 else "Model handles RB aging reasonably well",
                }
            else:
                aging_analysis = None
        else:
            aging_analysis = None

        position_analysis[pos] = {
            "n_samples": len(subset),
            "mae": round(mae, 1),
            "bias": round(bias, 1),
            "age_curve": age_errors,
            "feature_correlations": feature_corrs,
            "top_5_misses": top_misses,
            "aging_analysis": aging_analysis,
        }

    return position_analysis


def generate_improvement_opportunities(analysis: dict) -> list[dict]:
    """Synthesize findings into actionable improvement opportunities."""

    opportunities = []

    # 1. Check for significant segment biases
    significant_biases = analysis["segments"].get("significant_biases", [])
    for bias in significant_biases[:5]:  # Top 5
        opportunities.append({
            "type": "segment_calibration",
            "priority": "high" if abs(bias["bias"]) > 300 else "medium",
            "finding": f"{bias['segment']}/{bias['key']}: {bias['direction']}-predicting by {abs(bias['bias'])} KTC",
            "suggestion": f"Add calibration term for {bias['key']} players",
            "impact_estimate": f"Could reduce MAE by ~{abs(bias['bias']) * bias['n_samples'] / analysis['error_distribution']['summary_stats']['n_samples']:.0f}",
        })

    # 2. Check breakout detection
    breakout_data = analysis["breakout_detection"]["breakouts"]
    if breakout_data["detection_rate"] < 30:
        opportunities.append({
            "type": "feature_engineering",
            "priority": "high",
            "finding": f"Only catching {breakout_data['detection_rate']:.0f}% of breakouts",
            "suggestion": "Add more breakout-predictive features (opportunity metrics, efficiency trends)",
            "impact_estimate": "Could catch additional value surges",
        })

    # 3. Check feature residuals
    for feature, data in analysis["feature_residuals"]["by_feature"].items():
        if data["significant"] and abs(data["pearson_corr"]) > 0.15:
            opportunities.append({
                "type": "feature_transformation",
                "priority": "medium",
                "finding": f"Error correlates with {feature} (r={data['pearson_corr']:.2f})",
                "suggestion": data["interpretation"],
                "impact_estimate": "Nonlinear transformation or interaction terms may help",
            })

    # 4. Check temporal degradation
    temporal = analysis["temporal"]
    if temporal["trend"].get("significant") and temporal["trend"].get("direction") == "degrading":
        opportunities.append({
            "type": "retraining",
            "priority": "high",
            "finding": f"MAE degrading over time (slope={temporal['trend']['slope']:.1f}/year)",
            "suggestion": "Consider more recent data weighting or periodic retraining",
            "impact_estimate": "Newer predictions may be less accurate",
        })

    # 5. Position-specific issues
    for pos, data in analysis["position_specific"].items():
        if data.get("aging_analysis") and data["aging_analysis"].get("over_predicting"):
            if abs(data["aging_analysis"]["bias"]) > 200:
                opportunities.append({
                    "type": "age_curve_adjustment",
                    "priority": "medium",
                    "finding": f"{pos} aging players: {data['aging_analysis']['interpretation']}",
                    "suggestion": f"Strengthen age depreciation factor for {pos}",
                    "impact_estimate": f"Could reduce {pos} MAE for aging players",
                })

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    opportunities.sort(key=lambda x: priority_order.get(x["priority"], 99))

    return opportunities


def print_report(analysis: dict):
    """Print formatted analysis report."""

    print("\n" + "=" * 80)
    print("KTC MODEL PREDICTION ANALYSIS REPORT")
    print("=" * 80)

    # 1. Error Distribution
    print("\n" + "-" * 40)
    print("1. ERROR DISTRIBUTION ANALYSIS")
    print("-" * 40)

    stats_data = analysis["error_distribution"]["summary_stats"]
    print(f"\nSamples: {stats_data['n_samples']}")
    print(f"MAE: {stats_data['mae']}")
    print(f"MAPE: {stats_data['mape']}%")
    print(f"Mean Error (Bias): {stats_data['mean_error']}")
    print(f"Median Error: {stats_data['median_error']}")
    print(f"Std Dev: {stats_data['std_error']}")
    print(f"Skewness: {stats_data['skewness']} (positive = right-skewed)")
    print(f"Kurtosis: {stats_data['kurtosis']} (>0 = heavy tails)")

    direction = analysis["error_distribution"]["direction"]
    print(f"\nOver-predictions: {direction['over_predict_pct']}% (avg: +{direction['over_predict_avg']})")
    print(f"Under-predictions: {100 - direction['over_predict_pct']:.1f}% (avg: {direction['under_predict_avg']})")

    outliers = analysis["error_distribution"]["outliers"]
    print(f"\nOutliers (>{outliers['threshold']:.0f}): {outliers['outlier_pct']}% of predictions")

    normality = analysis["error_distribution"]["normality"]
    print(f"Normality: {'Yes' if normality['is_normal'] else 'No'} (p={normality['shapiro_p_value']})")

    pct = analysis["error_distribution"]["percentiles"]
    print(f"\nAbsolute Error Percentiles:")
    print(f"  P50: {pct['p50']} | P75: {pct['p75']} | P90: {pct['p90']} | P95: {pct['p95']}")

    # 2. Segmented Analysis
    print("\n" + "-" * 40)
    print("2. SEGMENTED ERROR ANALYSIS")
    print("-" * 40)

    print("\nBy Position:")
    for pos, data in analysis["segments"]["by_position"].items():
        sig = "*" if data["significant_bias"] else ""
        print(f"  {pos}: MAE={data['mae']}, Bias={data['bias']:+.0f}{sig}, n={data['n_samples']}")

    print("\nBy Age Bracket:")
    for bracket, data in analysis["segments"]["by_age_bracket"].items():
        sig = "*" if data["significant_bias"] else ""
        print(f"  {bracket}: MAE={data['mae']}, Bias={data['bias']:+.0f}{sig}, n={data['n_samples']}")

    print("\nBy KTC Tier:")
    for tier, data in analysis["segments"]["by_ktc_tier"].items():
        sig = "*" if data["significant_bias"] else ""
        print(f"  {tier}: MAE={data['mae']}, Bias={data['bias']:+.0f}{sig}, n={data['n_samples']}")

    print("\nBy Year:")
    for year, data in analysis["segments"]["by_year"].items():
        sig = "*" if data["significant_bias"] else ""
        print(f"  {year}: MAE={data['mae']}, Bias={data['bias']:+.0f}{sig}, n={data['n_samples']}")

    # Significant biases
    sig_biases = analysis["segments"]["significant_biases"]
    if sig_biases:
        print("\n* Significant Biases Detected:")
        for bias in sig_biases[:5]:
            print(f"  - {bias['segment']}/{bias['key']}: {bias['direction']}-predicting by {abs(bias['bias'])}")

    # 3. Biggest Misses
    print("\n" + "-" * 40)
    print("3. BIGGEST PREDICTION MISSES")
    print("-" * 40)

    misses = analysis["biggest_misses"]

    print(f"\nTop Over-Predictions ({misses['over_predictions']['count']} analyzed):")
    for ex in misses["over_predictions"]["top_examples"][:5]:
        print(f"  {ex['name']} ({ex['position']}, {ex['source_year']}): "
              f"Predicted {ex['predicted_ktc']:.0f}, Actual {ex['actual_ktc']:.0f}, Error: {ex['error']:+.0f}")

    print(f"\nTop Under-Predictions ({misses['under_predictions']['count']} analyzed):")
    for ex in misses["under_predictions"]["top_examples"][:5]:
        print(f"  {ex['name']} ({ex['position']}, {ex['source_year']}): "
              f"Predicted {ex['predicted_ktc']:.0f}, Actual {ex['actual_ktc']:.0f}, Error: {ex['error']:+.0f}")

    print(f"\nMissed Breakouts ({misses['missed_breakouts']['count']} total):")
    if misses["missed_breakouts"]["count"] > 0:
        print(f"  Position dist: {misses['missed_breakouts']['position_dist']}")
        for ex in misses["missed_breakouts"]["top_examples"][:3]:
            print(f"  - {ex['name']}: predicted {ex['predicted_ktc']:.0f}, actual {ex['actual_ktc']:.0f}")

    # 4. Feature Residuals
    print("\n" + "-" * 40)
    print("4. FEATURE RESIDUAL ANALYSIS")
    print("-" * 40)

    print("\nError correlations with features:")
    for feature, data in list(analysis["feature_residuals"]["by_feature"].items())[:8]:
        sig = "*" if data["significant"] else ""
        print(f"  {feature}: r={data['pearson_corr']:+.3f}{sig}")
        if data["interpretation"] != "No significant relationship":
            print(f"    -> {data['interpretation']}")

    # 5. Breakout Detection
    print("\n" + "-" * 40)
    print("5. BREAKOUT/BUST DETECTION")
    print("-" * 40)

    bd = analysis["breakout_detection"]
    print(f"\nBreakouts (>50% value increase):")
    print(f"  Total: {bd['breakouts']['total']}, Caught: {bd['breakouts']['caught']}, "
          f"Rate: {bd['breakouts']['detection_rate']}%")

    print(f"\nBusts (>30% value decrease):")
    print(f"  Total: {bd['busts']['total']}, Caught: {bd['busts']['caught']}, "
          f"Rate: {bd['busts']['detection_rate']}%")

    print("\nFeature differences (caught vs missed breakouts):")
    for feature, data in bd["feature_differences"].items():
        if data["significant"]:
            print(f"  {feature}: caught={data['caught_mean']}, missed={data['missed_mean']} *")

    # 6. Directional Accuracy
    print("\n" + "-" * 40)
    print("6. DIRECTIONAL ACCURACY")
    print("-" * 40)

    da = analysis["directional_accuracy"]
    print(f"\nOverall Directional Accuracy: {da['overall_accuracy']}%")
    print(f"  Up predictions (n={da['n_actual_up']}): {da['up_accuracy']}% correct")
    print(f"  Down predictions (n={da['n_actual_down']}): {da['down_accuracy']}% correct")
    print(f"  Flat (n={da['n_actual_flat']})")

    print("\nBy Segment:")
    for segment, accuracy in da["by_segment"].items():
        print(f"  {segment}: {accuracy}%")

    # 7. Temporal Analysis
    print("\n" + "-" * 40)
    print("7. TEMPORAL ANALYSIS")
    print("-" * 40)

    print("\nMAE by Target Year:")
    for year_data in analysis["temporal"]["by_year"]:
        print(f"  {year_data['year']}: MAE={year_data['mae']}, Bias={year_data['bias']:+.0f}, n={year_data['n_samples']}")

    trend = analysis["temporal"]["trend"]
    if "slope" in trend:
        print(f"\nTrend: {trend['direction']} ({trend['slope']:+.1f} MAE/year)")
        print(f"  Significant: {'Yes' if trend['significant'] else 'No'}")

    # 8. Position-Specific
    print("\n" + "-" * 40)
    print("8. POSITION-SPECIFIC INSIGHTS")
    print("-" * 40)

    for pos, data in analysis["position_specific"].items():
        print(f"\n{pos} (n={data['n_samples']}):")
        print(f"  MAE: {data['mae']}, Bias: {data['bias']:+.0f}")

        if data["aging_analysis"]:
            aging = data["aging_analysis"]
            print(f"  Aging Analysis (age>=27): {aging['interpretation']}")

        if data["feature_correlations"]:
            sig_corrs = [(f, d) for f, d in data["feature_correlations"].items() if d["significant"]]
            if sig_corrs:
                corr_strs = [f"{f}={d['correlation']:.2f}" for f, d in sig_corrs]
                print(f"  Significant error correlations: {', '.join(corr_strs)}")

    # 8. Improvement Opportunities
    print("\n" + "=" * 40)
    print("IMPROVEMENT OPPORTUNITIES")
    print("=" * 40)

    for i, opp in enumerate(analysis["improvement_opportunities"], 1):
        print(f"\n{i}. [{opp['priority'].upper()}] {opp['type']}")
        print(f"   Finding: {opp['finding']}")
        print(f"   Suggestion: {opp['suggestion']}")
        print(f"   Impact: {opp['impact_estimate']}")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)


def main(model_type: str = "xgb_base"):
    """Run comprehensive prediction analysis."""
    print("Generating predictions for all training pairs...")
    df = generate_all_predictions(model_type=model_type)
    print(f"Generated {len(df)} predictions")

    print("\nRunning analysis modules...")

    # Run all analyses
    analysis = {
        "error_distribution": analyze_error_distribution(df),
        "segments": analyze_segments(df),
        "biggest_misses": analyze_biggest_misses(df),
        "feature_residuals": analyze_feature_residuals(df),
        "breakout_detection": analyze_breakout_detection(df),
        "directional_accuracy": analyze_directional_accuracy(df),
        "temporal": analyze_temporal(df),
        "position_specific": analyze_position_specific(df),
    }

    # Generate improvement opportunities
    analysis["improvement_opportunities"] = generate_improvement_opportunities(analysis)

    # Print report
    print_report(analysis)

    return analysis


if __name__ == "__main__":
    main()
