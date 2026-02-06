"""
Tune calibration factors and breakout boost for ratio-based model.

Current targets:
- Elite tier bias: <±500 (currently -910)
- Breakout detection: >50% (currently 35.5%)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from app.services.data_loader import get_data_loader, calculate_derived_features, get_age_bracket, get_position_age_factor, get_position_ktc_ceiling
from app.models.predictor import KTCPredictor


def build_features_dict(season, derived, position, current_ktc, next_age, games_played, prior_predicted_ktc):
    """Build features dict (copy from model_service)."""
    import math

    age_bracket = get_age_bracket(next_age, position)
    age_depreciation_factor = get_position_age_factor(next_age, position)

    prior_pred_ratio = (
        prior_predicted_ktc / current_ktc
        if current_ktc > 0 and prior_predicted_ktc > 0
        else 0.0
    )

    fp = season.get("fantasy_points", 0)
    fp_per_game = fp / max(games_played, 1)
    opportunity_gap = ((17 - games_played) / 17) * fp_per_game if games_played > 0 else 0
    last_4_vs_season = season.get("last_4_vs_season", 0)
    fp_second_half_ratio = derived.get("fp_second_half_ratio", 1)
    momentum_score = last_4_vs_season * fp_second_half_ratio
    ktc_ceiling = get_position_ktc_ceiling(position)
    ktc_upside_ratio = (ktc_ceiling - current_ktc) / ktc_ceiling if ktc_ceiling > 0 else 0
    undervalued_young = int(next_age <= 25 and current_ktc < 3000 and fp_per_game > 8)
    young_x_momentum = (1 if next_age <= 25 else 0) * momentum_score
    young_x_upside = (1 if next_age <= 25 else 0) * ktc_upside_ratio
    low_games_x_efficiency = (1 if games_played < 10 else 0) * fp_per_game

    def safe_num(val, default=0):
        if val is None:
            return default
        try:
            if math.isnan(val):
                return default
        except (TypeError, ValueError):
            pass
        return val

    pff_overall = safe_num(season.get("pff_overall_grade"), 0)
    pff_receiving = safe_num(season.get("pff_receiving_grade"), 0)
    pff_run = safe_num(season.get("pff_run_grade"), 0)
    pff_pass = safe_num(season.get("pff_pass_grade"), 0)
    pff_prior = safe_num(season.get("pff_prior_year_grade"), 0)

    return {
        "current_ktc": current_ktc,
        "age": next_age,
        "years_exp": season.get("years_exp", 0) + 1,
        "fantasy_points": season.get("fantasy_points", 0),
        "games_played": games_played,
        "games_missed": 17 - games_played,
        "fp_vs_position_avg": 1.0,
        **derived,
        f"pos_{position}": 1,
        f"age_{age_bracket}": 1,
        "boom_rate": season.get("boom_rate", 0),
        "bust_rate": season.get("bust_rate", 0),
        "last_4_vs_season": season.get("last_4_vs_season", 0),
        "weekly_fp_cv": season.get("weekly_fp_cv", 0),
        "prior_predicted_ktc_ratio": prior_pred_ratio,
        "yards_per_carry": season.get("rushing_yards", 0) / max(season.get("carries", 1), 1),
        "yards_per_target": season.get("receiving_yards", 0) / max(season.get("targets", 1), 1),
        "target_share": season.get("target_share", 0),
        "rush_share": season.get("rush_share", 0),
        "age_depreciation_factor": age_depreciation_factor,
        "age_x_is_rb": next_age * (1 if position == "RB" else 0),
        "age_x_is_qb": next_age * (1 if position == "QB" else 0),
        "games_x_consistency": games_played * derived.get("fp_consistency", 0),
        "ktc_x_volatility": current_ktc * derived.get("ktc_in_season_volatility", 0) / 10000,
        "ktc_x_trend": current_ktc * derived.get("ktc_season_trend", 0) / 10000,
        "fp_per_game": fp_per_game,
        "opportunity_gap": opportunity_gap,
        "momentum_score": momentum_score,
        "ktc_upside_ratio": ktc_upside_ratio,
        "undervalued_young": undervalued_young,
        "young_x_momentum": young_x_momentum,
        "young_x_upside": young_x_upside,
        "low_games_x_efficiency": low_games_x_efficiency,
        "pff_overall_grade": pff_overall,
        "pff_receiving_grade": pff_receiving,
        "pff_run_grade": pff_run,
        "pff_pass_grade": pff_pass,
        "pff_prior_year_grade": pff_prior,
        "pff_grade_per_fp": pff_overall / max(season.get("fantasy_points", 1), 1),
        "pff_elite": 1 if pff_overall >= 80 else 0,
        "pff_grade_x_age": pff_overall * (30 - next_age) / 10 if next_age < 30 else 0,
        "pff_grade_x_fp_per_game": pff_overall * fp_per_game / 100,
        "pff_tier_70": 1 if pff_overall >= 70 else 0,
        "pff_tier_80": 1 if pff_overall >= 80 else 0,
        "pff_tier_90": 1 if pff_overall >= 90 else 0,
        "pff_position_grade": (
            pff_receiving if position in ("WR", "TE")
            else pff_run if position == "RB"
            else pff_pass if position == "QB"
            else pff_overall
        ),
        "pff_improvement": (pff_overall - pff_prior) if pff_prior > 0 else 0,
        "efficiency_surge": derived.get("efficiency_surge", 0),
        "low_ktc_x_high_efficiency": (
            (1 if current_ktc < 3000 else 0) * (1 if fp_per_game > 12 else 0) * fp_per_game
        ),
        "snap_trend_positive": 1 if derived.get("snap_pct_trend", 0) > 0.1 else 0,
        "opportunity_explosion": (
            ((17 - games_played) / 17) * fp_per_game * ktc_upside_ratio
        ) if (17 - games_played) > 4 else 0,
        "breakout_signal": int(
            next_age <= 25 and
            current_ktc < 3500 and
            (derived.get("efficiency_surge", 0) > 0.15 or momentum_score > 1.0)
        ),
        "low_ktc_young_momentum": (
            (1 if current_ktc < 3000 else 0) *
            (1 if next_age <= 25 else 0) *
            max(momentum_score, 0)
        ),
    }


def get_all_predictions(predictor):
    """Generate predictions for all year-over-year pairs."""
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

            if next_s["year"] != current["year"] + 1:
                continue
            if current["end_ktc"] <= 0 or next_s["end_ktc"] <= 0:
                continue

            position = current["position"]
            current_ktc = current["end_ktc"]
            actual_ktc = next_s["end_ktc"]
            actual_ratio = actual_ktc / current_ktc
            next_age = current["age"] + 1
            games_played = current["games_played"]

            derived = calculate_derived_features(current)
            features = build_features_dict(
                current, derived, position, current_ktc, next_age, games_played, 0.0
            )

            predicted_ratio = predictor.predict(features)
            predicted_ktc = predicted_ratio * current_ktc

            # Determine tier
            if current_ktc > 5000:
                tier = "elite"
            elif current_ktc > 3000:
                tier = "high"
            elif current_ktc > 2000:
                tier = "mid"
            else:
                tier = "low"

            # Determine if this is a breakout case
            is_riser = (actual_ratio > 1.15)  # >15% increase
            is_young = next_age <= 25

            results.append({
                "player_id": player_id,
                "name": current.get("name", ""),
                "position": position,
                "year": current["year"],
                "age": next_age,
                "current_ktc": current_ktc,
                "actual_ktc": actual_ktc,
                "predicted_ktc": predicted_ktc,
                "actual_ratio": actual_ratio,
                "predicted_ratio": predicted_ratio,
                "error": predicted_ktc - actual_ktc,
                "tier": tier,
                "is_riser": is_riser,
                "is_young": is_young,
                "features": features,
            })

    return results


def analyze_by_tier(results):
    """Analyze errors by KTC tier."""
    print("\n=== ERROR ANALYSIS BY TIER ===")

    for tier in ["elite", "high", "mid", "low"]:
        tier_results = [r for r in results if r["tier"] == tier]
        if not tier_results:
            continue

        errors = [r["error"] for r in tier_results]
        actuals = [r["actual_ktc"] for r in tier_results]
        preds = [r["predicted_ktc"] for r in tier_results]

        mae = mean_absolute_error(actuals, preds)
        bias = np.mean(errors)

        print(f"\n{tier.upper()} tier (n={len(tier_results)}):")
        print(f"  MAE: {mae:.1f}")
        print(f"  Bias: {bias:+.1f} (positive = over-predicting)")


def analyze_breakout_detection(results):
    """Analyze breakout detection rate."""
    print("\n=== BREAKOUT DETECTION ANALYSIS ===")

    # Young risers (age ≤ 25, actual increase > 15%)
    young_risers = [r for r in results if r["is_young"] and r["is_riser"]]

    if not young_risers:
        print("No young risers found in data")
        return

    # Count how many we correctly predicted to rise
    correctly_predicted = sum(1 for r in young_risers if r["predicted_ratio"] > 1.10)  # Predicted >10% increase
    detection_rate = correctly_predicted / len(young_risers) * 100

    print(f"\nYoung Risers (age <= 25, actual > +15%):")
    print(f"  Total: {len(young_risers)}")
    print(f"  Correctly predicted rise: {correctly_predicted}")
    print(f"  Detection rate: {detection_rate:.1f}%")

    # Show examples of missed breakouts
    missed = [r for r in young_risers if r["predicted_ratio"] <= 1.10]
    if missed:
        print(f"\nMissed breakouts (top 5 by actual ratio):")
        missed.sort(key=lambda x: x["actual_ratio"], reverse=True)
        for r in missed[:5]:
            print(f"  {r['position']} age {r['age']}: predicted {r['predicted_ratio']:.2f}x, actual {r['actual_ratio']:.2f}x (KTC {r['current_ktc']:.0f} -> {r['actual_ktc']:.0f})")


def tune_calibration_factors(results, predictor):
    """Find optimal calibration factors per tier."""
    print("\n=== TUNING CALIBRATION FACTORS ===")

    optimal_factors = {}

    for tier in ["elite", "high", "mid", "low"]:
        tier_results = [r for r in results if r["tier"] == tier]
        if not tier_results:
            continue

        # Find factor that minimizes bias
        best_factor = 1.0
        best_bias = float("inf")

        for factor in np.arange(0.90, 1.30, 0.01):
            adjusted_errors = []
            for r in tier_results:
                adjusted_pred = r["predicted_ratio"] * factor * r["current_ktc"]
                adjusted_errors.append(adjusted_pred - r["actual_ktc"])

            bias = abs(np.mean(adjusted_errors))
            if bias < best_bias:
                best_bias = bias
                best_factor = factor

        optimal_factors[tier] = best_factor

        # Calculate MAE with optimal factor
        adjusted_preds = [r["predicted_ratio"] * best_factor * r["current_ktc"] for r in tier_results]
        actuals = [r["actual_ktc"] for r in tier_results]
        mae = mean_absolute_error(actuals, adjusted_preds)
        adjusted_bias = np.mean([p - a for p, a in zip(adjusted_preds, actuals)])

        print(f"\n{tier.upper()} tier:")
        print(f"  Optimal factor: {best_factor:.2f}")
        print(f"  Adjusted MAE: {mae:.1f}")
        print(f"  Adjusted bias: {adjusted_bias:+.1f}")

    return optimal_factors


def tune_breakout_boost(results):
    """Find optimal breakout boost parameters."""
    print("\n=== TUNING BREAKOUT BOOST ===")

    young_risers = [r for r in results if r["is_young"] and r["is_riser"]]

    if not young_risers:
        return {}

    # Grid search for boost factor
    best_boost = 1.0
    best_momentum_thresh = 0.5
    best_detection_rate = 0

    for boost in np.arange(1.05, 1.30, 0.02):
        for momentum_thresh in np.arange(0.3, 1.2, 0.1):
            correctly_predicted = 0

            for r in young_risers:
                pred_ratio = r["predicted_ratio"]
                momentum = r["features"].get("momentum_score", 0)
                upside = r["features"].get("ktc_upside_ratio", 0)

                # Apply boost if conditions met
                if momentum > momentum_thresh and upside > 0.4:
                    pred_ratio *= boost

                if pred_ratio > 1.10:
                    correctly_predicted += 1

            detection_rate = correctly_predicted / len(young_risers) * 100

            if detection_rate > best_detection_rate:
                best_detection_rate = detection_rate
                best_boost = boost
                best_momentum_thresh = momentum_thresh

    print(f"\nOptimal breakout parameters:")
    print(f"  Boost factor: {best_boost:.2f}")
    print(f"  Momentum threshold: {best_momentum_thresh:.2f}")
    print(f"  Detection rate: {best_detection_rate:.1f}%")

    return {
        "boost_factor": best_boost,
        "momentum_threshold": best_momentum_thresh,
        "detection_rate": best_detection_rate,
    }


def evaluate_combined(results, calibration_factors, breakout_params):
    """Evaluate combined calibration + breakout boost."""
    print("\n=== COMBINED EVALUATION ===")

    adjusted_results = []

    for r in results:
        pred_ratio = r["predicted_ratio"]

        # Apply breakout boost for young players
        if r["is_young"]:
            momentum = r["features"].get("momentum_score", 0)
            upside = r["features"].get("ktc_upside_ratio", 0)

            if momentum > breakout_params.get("momentum_threshold", 0.5) and upside > 0.4:
                pred_ratio *= breakout_params.get("boost_factor", 1.15)

        # Apply tier calibration
        cal_factor = calibration_factors.get(r["tier"], 1.0)
        adjusted_pred = pred_ratio * cal_factor * r["current_ktc"]

        adjusted_results.append({
            **r,
            "adjusted_pred": adjusted_pred,
            "adjusted_error": adjusted_pred - r["actual_ktc"],
        })

    # Overall metrics
    actuals = [r["actual_ktc"] for r in adjusted_results]
    preds = [r["adjusted_pred"] for r in adjusted_results]
    overall_mae = mean_absolute_error(actuals, preds)
    print(f"\nOverall MAE: {overall_mae:.1f}")

    # By tier
    for tier in ["elite", "high", "mid", "low"]:
        tier_results = [r for r in adjusted_results if r["tier"] == tier]
        if not tier_results:
            continue

        errors = [r["adjusted_error"] for r in tier_results]
        tier_mae = mean_absolute_error(
            [r["actual_ktc"] for r in tier_results],
            [r["adjusted_pred"] for r in tier_results]
        )
        bias = np.mean(errors)

        print(f"\n{tier.upper()} tier (n={len(tier_results)}):")
        print(f"  MAE: {tier_mae:.1f}")
        print(f"  Bias: {bias:+.1f}")

    # Breakout detection
    young_risers = [r for r in adjusted_results if r["is_young"] and r["is_riser"]]
    if young_risers:
        correctly_predicted = sum(
            1 for r in young_risers
            if (r["adjusted_pred"] / r["current_ktc"]) > 1.10
        )
        detection_rate = correctly_predicted / len(young_risers) * 100
        print(f"\nBreakout detection rate: {detection_rate:.1f}%")

    return adjusted_results


def main():
    print("Loading model and generating predictions...")

    # Load current model
    from app.config import MODEL_PATH
    predictor = KTCPredictor()
    predictor.load(MODEL_PATH)

    print(f"Model loaded. Log transform: {getattr(predictor, '_use_log_transform', False)}")

    # Generate predictions
    results = get_all_predictions(predictor)
    print(f"Generated {len(results)} predictions")

    # Analyze current performance
    analyze_by_tier(results)
    analyze_breakout_detection(results)

    # Tune parameters
    calibration_factors = tune_calibration_factors(results, predictor)
    breakout_params = tune_breakout_boost(results)

    # Evaluate combined
    evaluate_combined(results, calibration_factors, breakout_params)

    # Print final recommendations
    print("\n" + "="*60)
    print("RECOMMENDED UPDATES TO predictor.py")
    print("="*60)

    print("\nCalibratedPredictor.calibration_factors = {")
    for tier, factor in calibration_factors.items():
        print(f'    "{tier}": {factor:.2f},')
    print("}")

    if breakout_params:
        print(f"\nBreakoutAwarePredictor:")
        print(f"    young_boost_factor = {breakout_params.get('boost_factor', 1.15):.2f}")
        print(f"    momentum_threshold = {breakout_params.get('momentum_threshold', 0.5):.2f}")


if __name__ == "__main__":
    main()
