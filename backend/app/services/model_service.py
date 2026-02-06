from typing import Optional
from pathlib import Path
import math

import pandas as pd

from app.config import MODEL_PATH, WEEKLY_MODEL_PATH, XGB_CALIBRATED_BREAKOUT_MODEL_PATH
from app.models.predictor import (
    KTCPredictor,
    WeeklyKTCPredictor,
    PositionEnsemblePredictor,
    XGBKTCPredictor,
    XGBOOST_AVAILABLE,
    LGBMKTCPredictor,
    LIGHTGBM_AVAILABLE,
    CatBoostKTCPredictor,
    CATBOOST_AVAILABLE,
    BreakoutAwarePredictor,
    CalibratedPredictor,
    CalibratedBreakoutPredictor,
    VotingEnsemblePredictor,
    XGBCalibratedBreakoutPredictor,
    HybridEnsemblePredictor,
    LinearRegressionPredictor,
)
from app.services.data_loader import (
    get_data_loader,
    calculate_derived_features,
    calculate_offseason_features,
    get_age_bracket,
    get_position_age_factor,
    get_position_ktc_ceiling,
    get_draft_capital_score,
)


class ModelService:
    """Service for managing the KTC prediction model."""

    def __init__(self):
        self.predictor = KTCPredictor()
        self.weekly_predictor = WeeklyKTCPredictor()
        self.ensemble_predictor: Optional[PositionEnsemblePredictor] = None
        self.xgb_predictor: Optional[XGBKTCPredictor] = None
        self.lgbm_predictor: Optional[LGBMKTCPredictor] = None
        self.catboost_predictor: Optional[CatBoostKTCPredictor] = None
        self.breakout_predictor: Optional[BreakoutAwarePredictor] = None
        self.calibrated_predictor: Optional[CalibratedPredictor] = None
        self.calibrated_breakout_predictor: Optional[CalibratedBreakoutPredictor] = None
        self.voting_ensemble_predictor: Optional[VotingEnsemblePredictor] = None
        self.xgb_calibrated_breakout_predictor: Optional[XGBCalibratedBreakoutPredictor] = None
        self.hybrid_ensemble_predictor: Optional[HybridEnsemblePredictor] = None
        self.linear_predictor: Optional[LinearRegressionPredictor] = None
        self._is_initialized = False
        self._weekly_initialized = False
        self._ensemble_initialized = False
        self._xgb_initialized = False
        self._lgbm_initialized = False
        self._catboost_initialized = False
        self._breakout_initialized = False
        self._calibrated_initialized = False
        self._calibrated_breakout_initialized = False
        self._voting_ensemble_initialized = False
        self._xgb_calibrated_breakout_initialized = False
        self._hybrid_ensemble_initialized = False
        self._linear_initialized = False

    def initialize(self, force_retrain: bool = False) -> dict:
        """Initialize the model - load from disk or train new."""
        if self._is_initialized and not force_retrain:
            return self.predictor.metrics

        # Try to load existing model
        if MODEL_PATH.exists() and not force_retrain:
            try:
                self.predictor.load(MODEL_PATH)
                self._is_initialized = True
                return self.predictor.metrics
            except Exception as e:
                print(f"Failed to load model: {e}. Training new model...")

        # Train new model
        return self.train_model()

    def train_model(
        self,
        use_temporal: bool = True,
        use_log_transform: bool = True,
        train_cutoff: int = 2023,
    ) -> dict:
        """Train a new model from the training data.

        Args:
            use_temporal: If True (default), use temporal validation for more realistic
                         evaluation by testing on future years. If False, use random split.
            use_log_transform: If True (default), apply log1p transform to target for better
                              handling of wide KTC range (500-10,000+). Expected 5-10% MAE
                              improvement on high-value players.
            train_cutoff: For temporal validation, train on years <= cutoff, test on years > cutoff.
        """
        data_loader = get_data_loader()
        X, y, metadata = data_loader.get_feature_matrix(include_metadata=True)

        if use_temporal:
            years = metadata["year"]
            metrics = self.predictor.train_temporal(
                X, y, years,
                train_cutoff=train_cutoff,
                use_log_transform=use_log_transform,
            )
        else:
            metrics = self.predictor.train(
                X, y,
                use_log_transform=use_log_transform,
                use_cv=True,
            )

        # Save the trained model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.predictor.save(MODEL_PATH)

        self._is_initialized = True
        return metrics

    def train_model_optimized(
        self,
        n_trials: int = 50,
        use_log_transform: bool = True,
    ) -> dict:
        """Train model with Optuna hyperparameter optimization.

        Uses cross-validation to find optimal hyperparameters, then trains
        the final model on all data. This typically yields 3-7% MAE improvement
        over hand-tuned parameters.

        Args:
            n_trials: Number of Optuna optimization trials (default 50)
            use_log_transform: Apply log transform to target (default True)

        Returns:
            Dictionary with best_params, metrics, and optimization details
        """
        data_loader = get_data_loader()
        X, y = data_loader.get_feature_matrix()

        print(f"Starting Optuna optimization with {n_trials} trials...")
        result = self.predictor.optimize_hyperparameters(
            X, y,
            n_trials=n_trials,
            use_log_transform=use_log_transform,
        )

        if "error" in result:
            return result

        # Save the optimized model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.predictor.save(MODEL_PATH)

        self._is_initialized = True

        print(f"Optimization complete!")
        print(f"Best CV MAE: {result['best_cv_mae']:.1f}")
        print(f"Test MAE: {result['metrics']['test_mae']:.1f}")
        print(f"Best params: {result['best_params']}")

        return result

    def predict_for_player(self, player_id: str, use_ensemble: bool = False) -> Optional[dict]:
        """Get prediction for a specific player.

        Args:
            player_id: The player ID
            use_ensemble: If True, use position-specific ensemble model.
                         Default is False to use XGBCalibratedBreakoutPredictor.
        """
        # Use ensemble only if explicitly requested
        if use_ensemble:
            if not self._ensemble_initialized:
                self.initialize_ensemble()
            if self._ensemble_initialized:
                return self.predict_with_ensemble(player_id)

        # Use XGBCalibratedBreakoutPredictor as default (tuned calibration + bust detection)
        if not self._xgb_calibrated_breakout_initialized:
            self.initialize_xgb_calibrated_breakout()

        data_loader = get_data_loader()
        player = data_loader.get_player_by_id(player_id)

        if not player:
            return None

        # Get seasons sorted by year
        seasons = player.get("seasons", [])
        if not seasons:
            return None

        seasons_sorted = sorted(seasons, key=lambda s: s["year"])
        latest = seasons_sorted[-1]
        current_ktc = latest.get("end_ktc") or latest.get("start_ktc", 0)
        position = player["position"]

        # Calculate prior_predicted_ktc if we have a previous season
        prior_predicted_ktc = 0.0
        if len(seasons_sorted) >= 2:
            prev_season = seasons_sorted[-2]
            if prev_season["year"] == latest["year"] - 1:
                # Use prev_season to predict what latest season's end_ktc would be
                prev_derived = calculate_derived_features(prev_season)
                prev_ktc = prev_season.get("end_ktc") or prev_season.get("start_ktc", 0)
                prev_age = prev_season.get("age", 25) + 1
                prev_games = prev_season.get("games_played", 0)
                prev_features = self._build_features_dict(
                    prev_season, prev_derived, position, prev_ktc, prev_age, prev_games, 0.0
                )
                # Model predicts ratio, convert to absolute
                prior_predicted_ratio = self.xgb_calibrated_breakout_predictor.predict(prev_features)
                prior_predicted_ktc = prior_predicted_ratio * prev_ktc

        # Calculate derived features from the latest season's weekly data
        derived = calculate_derived_features(latest)

        # Calculate new features
        next_age = latest.get("age", 25) + 1
        games_played = latest.get("games_played", 0)

        # Build features with all Phase 1, 2, 3 additions
        features = self._build_features_dict(
            latest, derived, position, current_ktc, next_age, games_played, prior_predicted_ktc
        )

        # Model predicts ratio (next_ktc / current_ktc), convert to absolute
        predicted_ratio = self.xgb_calibrated_breakout_predictor.predict(features)
        predicted_ktc = predicted_ratio * current_ktc
        ktc_change = predicted_ktc - current_ktc
        ktc_change_pct = (predicted_ratio - 1) * 100

        # Get tier and check if breakout/bust was applied
        tier = self.xgb_calibrated_breakout_predictor._get_tier(current_ktc)
        breakout_applied = self.xgb_calibrated_breakout_predictor._should_apply_breakout_boost(features, tier)
        bust_applied, bust_factor = self.xgb_calibrated_breakout_predictor._should_apply_bust_penalty(
            features, tier, position
        )

        return {
            "player_id": player_id,
            "name": player["name"],
            "position": position,
            "current_ktc": current_ktc,
            "predicted_ktc": round(predicted_ktc, 2),
            "ktc_change": round(ktc_change, 2),
            "ktc_change_pct": round(ktc_change_pct, 2),
            "tier": tier,
            "breakout_boost_applied": breakout_applied,
            "bust_penalty_applied": bust_applied,
            "model": "xgb_calibrated_breakout",
        }

    def _build_features_dict(
        self,
        season: dict,
        derived: dict,
        position: str,
        current_ktc: float,
        next_age: int,
        games_played: int,
        prior_predicted_ktc: float,
    ) -> dict:
        """Build the full features dictionary for prediction.

        Includes all features from training data (88 total).
        """
        age_bracket = get_age_bracket(next_age, position)
        age_depreciation_factor = get_position_age_factor(next_age, position)

        # Phase 1.3: Prior prediction as ratio for better normalization
        prior_pred_ratio = (
            prior_predicted_ktc / current_ktc
            if current_ktc > 0 and prior_predicted_ktc > 0
            else 0.0
        )

        # === BREAKOUT DETECTION FEATURES ===
        fp = season.get("fantasy_points", 0)
        fp_per_game = fp / max(games_played, 1)

        # Opportunity gap: limited games but decent per-game production
        opportunity_gap = ((17 - games_played) / 17) * fp_per_game if games_played > 0 else 0

        # Momentum score: late-season surge indicators
        last_4_vs_season = season.get("last_4_vs_season", 0)
        fp_second_half_ratio = derived.get("fp_second_half_ratio", 1)
        momentum_score = last_4_vs_season * fp_second_half_ratio

        # KTC upside ratio: room to grow to position ceiling
        ktc_ceiling = get_position_ktc_ceiling(position)
        ktc_upside_ratio = (ktc_ceiling - current_ktc) / ktc_ceiling if ktc_ceiling > 0 else 0

        # Undervalued young flag
        undervalued_young = int(next_age <= 25 and current_ktc < 3000 and fp_per_game > 8)

        # Interaction features for breakout detection
        young_x_momentum = (1 if next_age <= 25 else 0) * momentum_score
        young_x_upside = (1 if next_age <= 25 else 0) * ktc_upside_ratio
        low_games_x_efficiency = (1 if games_played < 10 else 0) * fp_per_game

        # Helper to safely get numeric values (handles NaN from pandas)
        def safe_num(val, default=0):
            if val is None:
                return default
            try:
                if math.isnan(val):
                    return default
            except (TypeError, ValueError):
                pass
            return val

        # Pre-extract PFF grades with NaN handling
        pff_overall = safe_num(season.get("pff_overall_grade"), 0)
        pff_receiving = safe_num(season.get("pff_receiving_grade"), 0)
        pff_run = safe_num(season.get("pff_run_grade"), 0)
        pff_pass = safe_num(season.get("pff_pass_grade"), 0)
        pff_prior = safe_num(season.get("pff_prior_year_grade"), 0)

        # === DRAFT CAPITAL FEATURES ===
        draft_round = season.get("draft_round")
        draft_pick = season.get("draft_pick")
        draft_capital_score = get_draft_capital_score(draft_round, draft_pick)

        # Parse draft round for flags
        draft_round_valid = None
        if draft_round is not None:
            try:
                draft_round_valid = int(draft_round)
            except (ValueError, TypeError):
                pass

        is_first_rounder = 1 if draft_round_valid == 1 else 0
        is_day_one_pick = 1 if draft_round_valid in (1, 2) else 0
        is_day_two_pick = 1 if draft_round_valid == 3 else 0
        is_udfa = 1 if draft_round_valid is None else 0

        # === POSITIONAL DOMINANCE ===
        positional_competition = season.get("positional_competition", 0)
        positional_dominance = 1.0 / (1 + positional_competition / 5000)
        is_positional_alpha = 1 if current_ktc > positional_competition else 0

        # === OFFSEASON FEATURES ===
        offseason_features = calculate_offseason_features(
            season,
            season.get("prior_end_ktc", 0)
        )

        # Build features dict (reduced from 101 to ~65 by removing noisy/redundant features)
        # Note: Some features from 'derived' dict are no longer used but included for compatibility
        return {
            "current_ktc": current_ktc,
            "age": next_age,
            "years_exp": season.get("years_exp", 0) + 1,
            "fantasy_points": season.get("fantasy_points", 0),
            "games_played": games_played,
            # REMOVED: games_missed (redundant with games_played)
            "fp_vs_position_avg": season.get("fp_vs_position_avg", 1.0),
            # Derived features from weekly data
            "fp_std_dev": derived.get("fp_std_dev", 0),
            # REMOVED: fp_consistency (identical to games_played - r=1.00)
            # REMOVED: fp_max_week (highly correlated with fp_std_dev - r=0.96)
            "fp_second_half_ratio": derived.get("fp_second_half_ratio", 1),
            "ktc_in_season_volatility": derived.get("ktc_in_season_volatility", 0),
            "ktc_season_trend": derived.get("ktc_season_trend", 0),
            # REMOVED: ktc_max_swing (r=0.986 with ktc_season_trend)
            # REMOVED: snap_pct_avg, snap_pct_trend (constant: all 0)
            # Position one-hot encoding (removed pos_RB - zero importance)
            "pos_QB": 1 if position == "QB" else 0,
            # REMOVED: pos_RB (zero importance)
            "pos_WR": 1 if position == "WR" else 0,
            "pos_TE": 1 if position == "TE" else 0,
            # Age bracket one-hot encoding (removed age_prime - zero importance)
            "age_young": 1 if age_bracket == "young" else 0,
            # REMOVED: age_prime (zero importance)
            "age_declining": 1 if age_bracket == "declining" else 0,
            # Phase 1: New features
            "boom_rate": season.get("boom_rate", 0),
            "bust_rate": season.get("bust_rate", 0),
            "last_4_vs_season": season.get("last_4_vs_season", 0),
            "weekly_fp_cv": season.get("weekly_fp_cv", 0),
            # REMOVED: prior_predicted_ktc_ratio (constant: all 0)
            # Phase 2: Efficiency ratios
            "yards_per_carry": season.get("rushing_yards", 0) / max(season.get("carries", 1), 1),
            "yards_per_target": season.get("receiving_yards", 0) / max(season.get("targets", 1), 1),
            # REMOVED: target_share, rush_share (constant: all 0)
            # Phase 3.2: Age depreciation factor
            "age_depreciation_factor": age_depreciation_factor,
            # Feature interactions
            # REMOVED: age_x_is_rb, age_x_is_qb (highly correlated with pos_*)
            # REMOVED: games_x_consistency (r=0.97 with fp_consistency)
            "ktc_x_volatility": current_ktc * derived.get("ktc_in_season_volatility", 0) / 10000,
            "ktc_x_trend": current_ktc * derived.get("ktc_season_trend", 0) / 10000,
            # === BREAKOUT DETECTION FEATURES ===
            "fp_per_game": fp_per_game,
            "opportunity_gap": opportunity_gap,
            "momentum_score": momentum_score,
            "ktc_upside_ratio": ktc_upside_ratio,
            "undervalued_young": undervalued_young,
            "young_x_momentum": young_x_momentum,
            "young_x_upside": young_x_upside,
            "low_games_x_efficiency": low_games_x_efficiency,
            # === DRAFT CAPITAL FEATURES ===
            "draft_capital_score": draft_capital_score,
            # REMOVED: is_first_rounder, is_day_one_pick, first_round_x_young (zero importance)
            "is_day_two_pick": is_day_two_pick,
            "is_udfa": is_udfa,
            "draft_capital_x_age": draft_capital_score * (30 - next_age) / 10 if next_age < 30 else 0,
            # === ENHANCED PFF FEATURES ===
            "pff_overall_grade": pff_overall,
            "pff_receiving_grade": pff_receiving,
            "pff_run_grade": pff_run,
            "pff_pass_grade": pff_pass,
            "pff_prior_year_grade": pff_prior,
            # REMOVED: pff_grade_per_fp (low importance)
            # REMOVED: pff_elite, pff_tier_80, pff_tier_90 (zero importance or redundant)
            "pff_grade_x_age": pff_overall * (30 - next_age) / 10 if next_age < 30 else 0,
            "pff_grade_x_fp_per_game": pff_overall * fp_per_game / 100,
            "pff_tier_70": 1 if pff_overall >= 70 else 0,
            # REMOVED: pff_position_grade (r=0.985 with pff_overall_grade)
            "pff_improvement": (pff_overall - pff_prior) if pff_prior > 0 else 0,
            # === POSITIONAL DOMINANCE ===
            "positional_dominance": positional_dominance,
            "is_positional_alpha": is_positional_alpha,
            # === RED ZONE / OPPORTUNITY FEATURES ===
            "red_zone_targets": season.get("red_zone_targets", 0),
            "red_zone_touches": season.get("red_zone_touches", 0),
            "momentum_ratio": momentum_score,  # Alias for compatibility
            "snap_share": derived.get("snap_pct_avg", 0),
            "first_down_rate": season.get("first_down_rate", 0),
            # === ENHANCED BREAKOUT FEATURES ===
            "efficiency_surge": derived.get("efficiency_surge", 0),
            # REMOVED: low_ktc_x_high_efficiency, snap_trend_positive (zero importance/constant)
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
            # === YOY IMPROVEMENT FEATURES ===
            "fp_yoy_ratio_capped": self._calculate_fp_yoy_ratio(season, games_played),
            "fp_per_game_yoy_ratio_capped": self._calculate_fp_per_game_yoy_ratio(season, fp_per_game),
            "performance_breakout": self._calculate_performance_breakout(season, games_played),
            "young_performance_breakout": self._calculate_young_performance_breakout(season, games_played, next_age),
            # === OFFSEASON KTC FEATURES ===
            "offseason_ktc_retention": offseason_features["offseason_ktc_retention"],
            "offseason_trend": offseason_features["offseason_trend"],
            "offseason_volatility": offseason_features["offseason_volatility"],
            "draft_impact": offseason_features["draft_impact"],
            "training_camp_surge": offseason_features["training_camp_surge"],
            "free_agency_impact": offseason_features["free_agency_impact"],
            "offseason_max_drawdown": offseason_features["offseason_max_drawdown"],
            "offseason_recovery": offseason_features["offseason_recovery"],
            # === PHASE 1 NEW FEATURES: KTC MOMENTUM ===
            "ktc_last_4_weeks_trend": derived.get("ktc_last_4_weeks_trend", 0),
            # REMOVED: ktc_acceleration (redundant with ktc_season_trend - r=-0.94)
            "ktc_relative_to_season_high": derived.get("ktc_relative_to_season_high", 1.0),
            # === PHASE 1 NEW FEATURES: POSITIONAL SCARCITY ===
            "position_scarcity_index": season.get("position_scarcity_index", 1.0),
            "elite_at_position_count": season.get("elite_at_position_count", 5),
            # === PHASE 1 NEW FEATURES: AVAILABILITY TREND ===
            "availability_trend": season.get("availability_trend", 0),
            # REMOVED: consecutive_healthy_seasons (zero importance)
            # === PHASE 1 NEW FEATURES: CONTRACT YEAR ===
            "is_contract_year": season.get("is_contract_year", 0),
            "years_remaining": season.get("years_remaining", 3),
            # REMOVED: contract_year_x_age (r=0.993 with is_contract_year)
            # === NEW INTERACTION FEATURES (from linear regression insights) ===
            "contract_year_x_performance": season.get("is_contract_year", 0) * fp_per_game,
            "volatility_x_young": derived.get("ktc_in_season_volatility", 0) * (1 if next_age < 25 else 0),
            # === PHASE 1 NEW FEATURES: TEAM OFFENSIVE QUALITY ===
            # REMOVED: team_pass_rate, team_plays_per_game (constant values)
            "team_points_per_game": season.get("team_points_per_game", 22),
            # === TEAM CONTEXT FEATURES ===
            "positional_competition": season.get("positional_competition") or 0,
            "team_total_ktc": season.get("team_total_ktc") or 0,
            "qb_ktc": season.get("qb_ktc") or 0,
        }

    def _calculate_fp_yoy_ratio(self, season: dict, games_played: int) -> float:
        """Calculate YoY fantasy points ratio (capped at 3.0)."""
        prior_year_fp = season.get("prior_year_fp") or 0
        prior_year_games = season.get("prior_year_games") or 0
        fp = season.get("fantasy_points") or 0

        if prior_year_fp > 0 and prior_year_games >= 4:
            fp_yoy_ratio = fp / prior_year_fp
            return min(fp_yoy_ratio, 3.0)
        return 1.0

    def _calculate_fp_per_game_yoy_ratio(self, season: dict, fp_per_game: float) -> float:
        """Calculate YoY fantasy points per game ratio (capped at 3.0)."""
        prior_year_fp = season.get("prior_year_fp") or 0
        prior_year_games = season.get("prior_year_games") or 0

        if prior_year_games >= 4:
            prior_fp_per_game = prior_year_fp / max(prior_year_games, 1)
            if prior_fp_per_game > 0:
                fp_per_game_yoy_ratio = fp_per_game / prior_fp_per_game
                return min(fp_per_game_yoy_ratio, 3.0)
        return 1.0

    def _calculate_performance_breakout(self, season: dict, games_played: int) -> int:
        """Calculate performance breakout flag (40%+ YoY improvement)."""
        prior_year_fp = season.get("prior_year_fp") or 0
        prior_year_games = season.get("prior_year_games") or 0
        fp = season.get("fantasy_points") or 0

        if prior_year_fp > 0 and prior_year_games >= 4:
            fp_yoy_ratio = fp / prior_year_fp
            fp_yoy_ratio_capped = min(fp_yoy_ratio, 3.0)
            if fp_yoy_ratio_capped >= 1.40 and games_played >= 10:
                return 1
        return 0

    def _calculate_young_performance_breakout(
        self, season: dict, games_played: int, next_age: int
    ) -> int:
        """Calculate young performance breakout flag (age ≤25 + 40%+ YoY improvement)."""
        if next_age > 25:
            return 0
        return self._calculate_performance_breakout(season, games_played)

    def get_metrics(self) -> dict:
        """Get model performance metrics."""
        return self.predictor.metrics

    def get_metrics_by_year(self) -> dict:
        """Calculate R² and MAE (absolute KTC) for each prediction target year.

        Uses the XGBoost model for predictions and returns metrics in absolute
        KTC values (not ratios) for easier interpretation.
        """
        # Initialize XGBoost model
        if not self._xgb_initialized:
            self.initialize_xgboost()

        if not self.xgb_predictor:
            return {"error": "XGBoost model not available"}

        from sklearn.metrics import mean_absolute_error, r2_score
        import numpy as np

        data_loader = get_data_loader()
        df = data_loader.get_training_dataframe()
        df = df.sort_values(["player_id", "year"])

        # Build year-over-year pairs with target year info
        yearly_data = {}  # year -> [(actual, predicted, current_ktc), ...]
        all_data = []  # For overall metrics

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
                next_age = current["age"] + 1
                games_played = current["games_played"]

                # Build features using the standard method
                derived = calculate_derived_features(current)
                features = self._build_features_dict(
                    current, derived, position, current_ktc, next_age, games_played, 0.0
                )

                # Model predicts ratio, convert to absolute
                predicted_ratio = self.xgb_predictor.predict(features)
                predicted_ktc = predicted_ratio * current_ktc
                actual_ktc = next_s["end_ktc"]
                target_year = next_s["year"]

                if target_year not in yearly_data:
                    yearly_data[target_year] = []
                yearly_data[target_year].append((actual_ktc, predicted_ktc, current_ktc))
                all_data.append((actual_ktc, predicted_ktc, current_ktc))

        # Calculate metrics for each year (absolute KTC values)
        by_year = []
        for year in sorted(yearly_data.keys()):
            pairs = yearly_data[year]
            actuals = [p[0] for p in pairs]
            predictions = [p[1] for p in pairs]

            r2 = r2_score(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)

            by_year.append({
                "year": year,
                "r2": round(r2, 3),
                "mae": round(mae, 0),  # Absolute KTC points
                "n_samples": len(pairs),
            })

        # Calculate overall metrics (absolute KTC values)
        all_actuals = [p[0] for p in all_data]
        all_predictions = [p[1] for p in all_data]

        overall_r2 = r2_score(all_actuals, all_predictions)
        overall_mae = mean_absolute_error(all_actuals, all_predictions)
        overall_mape = np.mean(np.abs((np.array(all_actuals) - np.array(all_predictions)) / np.array(all_actuals))) * 100

        return {
            "overall": {
                "test_r2": round(overall_r2, 4),
                "test_mae": round(overall_mae, 0),  # Absolute KTC points
                "test_mape": round(overall_mape, 1),
                "n_samples": len(all_data),
                "model": "xgboost",
            },
            "by_year": by_year,
        }

    def predict_historical_seasons(self, player_id: str) -> list[dict]:
        """Generate ML predictions for all historical seasons of a player.

        For each season N, uses season N-1's data to predict season N's end_ktc.
        This allows comparison between what the model would have predicted
        and what actually happened.

        Args:
            player_id: The player ID

        Returns:
            List of {year, predicted_ktc} pairs. The first season won't have a
            prediction since there's no prior data.
        """
        # Initialize XGBCalibratedBreakoutPredictor if not already done
        if not self._xgb_calibrated_breakout_initialized:
            self.initialize_xgb_calibrated_breakout()

        if not self.xgb_calibrated_breakout_predictor:
            return []

        data_loader = get_data_loader()
        player = data_loader.get_player_by_id(player_id)

        if not player:
            return []

        seasons = player.get("seasons", [])
        if len(seasons) < 2:
            return []  # Need at least 2 seasons for predictions

        # Sort by year
        seasons_sorted = sorted(seasons, key=lambda s: s["year"])
        position = player["position"]

        predictions = []

        # For each season after the first, predict using the previous season's data
        for i in range(1, len(seasons_sorted)):
            prev_season = seasons_sorted[i - 1]
            current_season = seasons_sorted[i]

            # Skip if seasons aren't consecutive
            if current_season["year"] != prev_season["year"] + 1:
                continue

            prev_ktc = prev_season.get("end_ktc") or prev_season.get("start_ktc", 0)
            if prev_ktc <= 0:
                continue

            # Build features from the previous season
            prev_derived = calculate_derived_features(prev_season)
            next_age = prev_season.get("age", 25) + 1
            games_played = prev_season.get("games_played", 0)

            # Calculate prior_predicted_ktc (prediction from 2 seasons ago)
            prior_predicted_ktc = 0.0
            if i >= 2:
                prev_prev_season = seasons_sorted[i - 2]
                if prev_prev_season["year"] == prev_season["year"] - 1:
                    prev_prev_ktc = prev_prev_season.get("end_ktc") or prev_prev_season.get("start_ktc", 0)
                    if prev_prev_ktc > 0:
                        prev_prev_derived = calculate_derived_features(prev_prev_season)
                        prev_prev_age = prev_prev_season.get("age", 25) + 1
                        prev_prev_games = prev_prev_season.get("games_played", 0)
                        prev_prev_features = self._build_features_dict(
                            prev_prev_season, prev_prev_derived, position,
                            prev_prev_ktc, prev_prev_age, prev_prev_games, 0.0
                        )
                        prior_predicted_ratio = self.xgb_calibrated_breakout_predictor.predict(prev_prev_features)
                        prior_predicted_ktc = prior_predicted_ratio * prev_prev_ktc

            # Build features for prediction
            features = self._build_features_dict(
                prev_season, prev_derived, position,
                prev_ktc, next_age, games_played, prior_predicted_ktc
            )

            # Predict: model returns ratio, multiply by current KTC
            predicted_ratio = self.xgb_calibrated_breakout_predictor.predict(features)
            predicted_ktc = predicted_ratio * prev_ktc

            predictions.append({
                "year": current_season["year"],
                "predicted_ktc": round(predicted_ktc, 2),
            })

        return predictions

    def get_feature_importance(self) -> dict:
        """Get feature importance from the model."""
        return self.predictor.get_feature_importance()

    def analyze_feature_importance(self, importance_threshold: float = 0.01) -> dict:
        """Analyze feature importance and identify low-importance features for pruning.

        With 50+ features and only ~1,400 samples, there's risk of overfitting.
        This method identifies features that contribute <threshold cumulative importance.

        Args:
            importance_threshold: Features below this cumulative importance threshold
                                 will be marked for pruning (default 1%)

        Returns:
            Dictionary with:
            - importance_ranking: Sorted dict of feature -> importance
            - cumulative_importance: Dict of feature -> cumulative importance
            - features_to_keep: List of features above threshold
            - features_to_prune: List of features below threshold
            - summary: Summary statistics
        """
        if not self._is_initialized:
            self.initialize()

        importance = self.predictor.get_feature_importance()

        if not importance:
            return {"error": "Model not trained or no feature importance available"}

        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # Calculate cumulative importance
        total_importance = sum(imp for _, imp in sorted_importance)
        cumulative = {}
        running_sum = 0
        features_to_keep = []
        features_to_prune = []

        for feature, imp in sorted_importance:
            running_sum += imp
            cumulative[feature] = running_sum / total_importance if total_importance > 0 else 0

            if cumulative[feature] <= (1 - importance_threshold):
                features_to_keep.append(feature)
            else:
                features_to_prune.append(feature)

        # Top 10 features
        top_10 = sorted_importance[:10]

        return {
            "importance_ranking": dict(sorted_importance),
            "cumulative_importance": cumulative,
            "features_to_keep": features_to_keep,
            "features_to_prune": features_to_prune,
            "top_10_features": [{"feature": f, "importance": round(imp, 4)} for f, imp in top_10],
            "summary": {
                "total_features": len(importance),
                "features_to_keep": len(features_to_keep),
                "features_to_prune": len(features_to_prune),
                "importance_threshold": importance_threshold,
                "top_feature": sorted_importance[0][0] if sorted_importance else None,
                "top_feature_importance": round(sorted_importance[0][1], 4) if sorted_importance else 0,
            }
        }

    def get_error_analysis(self) -> dict:
        """Analyze prediction errors by segment (position, age bracket, KTC level)."""
        if not self._is_initialized:
            self.initialize()

        from sklearn.metrics import mean_absolute_error
        import numpy as np

        data_loader = get_data_loader()
        df = data_loader.get_training_dataframe()
        df = df.sort_values(["player_id", "year"])

        # Collect predictions with segment info
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
                next_age = current["age"] + 1
                games_played = current["games_played"]
                age_bracket = get_age_bracket(next_age, position)

                # Determine KTC level
                if current_ktc < 2000:
                    ktc_level = "low"
                elif current_ktc < 5000:
                    ktc_level = "mid"
                else:
                    ktc_level = "high"

                # Build features
                derived = calculate_derived_features(current)
                features = {
                    "current_ktc": current_ktc,
                    "age": next_age,
                    "years_exp": current["years_exp"] + 1,
                    "fantasy_points": current["fantasy_points"],
                    "games_played": games_played,
                    "games_missed": 17 - games_played,
                    "fp_vs_position_avg": 1.0,
                    **derived,
                    f"pos_{position}": 1,
                    f"age_{age_bracket}": 1,
                    "prior_predicted_ktc": 0,
                    "age_x_is_rb": next_age * (1 if position == "RB" else 0),
                    "age_x_is_qb": next_age * (1 if position == "QB" else 0),
                    "games_x_consistency": games_played * derived.get("fp_consistency", 0),
                    "ktc_x_volatility": current_ktc * derived.get("ktc_in_season_volatility", 0) / 10000,
                    "ktc_x_trend": current_ktc * derived.get("ktc_season_trend", 0) / 10000,
                }

                # Model predicts ratio, convert to absolute
                predicted_ratio = self.predictor.predict(features)
                predicted = predicted_ratio * current_ktc
                actual = next_s["end_ktc"]
                error = predicted - actual

                results.append({
                    "position": position,
                    "age_bracket": age_bracket,
                    "ktc_level": ktc_level,
                    "actual": actual,
                    "predicted": predicted,
                    "error": error,
                })

        # Aggregate by segments
        def calc_segment_metrics(subset):
            if not subset:
                return None
            errors = [r["error"] for r in subset]
            actuals = [r["actual"] for r in subset]
            predictions = [r["predicted"] for r in subset]
            return {
                "mae": round(mean_absolute_error(actuals, predictions), 1),
                "bias": round(float(np.mean(errors)), 1),  # Positive = over-predicting
                "n_samples": len(subset),
            }

        analysis = {
            "by_position": {},
            "by_age_bracket": {},
            "by_ktc_level": {},
        }

        # By position
        for pos in ["QB", "RB", "WR", "TE"]:
            subset = [r for r in results if r["position"] == pos]
            metrics = calc_segment_metrics(subset)
            if metrics:
                analysis["by_position"][pos] = metrics

        # By age bracket
        for bracket in ["young", "prime", "declining"]:
            subset = [r for r in results if r["age_bracket"] == bracket]
            metrics = calc_segment_metrics(subset)
            if metrics:
                analysis["by_age_bracket"][bracket] = metrics

        # By KTC level
        for level in ["low", "mid", "high"]:
            subset = [r for r in results if r["ktc_level"] == level]
            metrics = calc_segment_metrics(subset)
            if metrics:
                analysis["by_ktc_level"][level] = metrics

        # Overall
        analysis["overall"] = calc_segment_metrics(results)

        return analysis

    def train_model_with_prior_predictions(self) -> dict:
        """Two-stage training: baseline → generate priors → retrain."""
        data_loader = get_data_loader()

        # Stage 1: Train baseline model (without prior_predicted_ktc)
        print("Stage 1: Training baseline model...")
        X_baseline, y_baseline = data_loader.get_feature_matrix()
        baseline_metrics = self.predictor.train(X_baseline, y_baseline)
        print(
            f"Baseline - R²: {baseline_metrics['test_r2']:.3f}, "
            f"MAE: {baseline_metrics['test_mae']:.1f}"
        )

        # Stage 2: Generate prior predictions using baseline model
        print("Stage 2: Generating prior predictions...")
        prior_predictions = self._generate_prior_predictions(data_loader)
        print(f"Generated {len(prior_predictions)} prior predictions")

        # Stage 3: Retrain with prior predictions as feature
        print("Stage 3: Retraining with prior_predicted_ktc...")
        X_enhanced, y_enhanced = data_loader.get_feature_matrix(prior_predictions)
        enhanced_metrics = self.predictor.train(X_enhanced, y_enhanced)
        print(
            f"Enhanced - R²: {enhanced_metrics['test_r2']:.3f}, "
            f"MAE: {enhanced_metrics['test_mae']:.1f}"
        )

        # Save enhanced model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.predictor.save(MODEL_PATH)

        self._is_initialized = True

        improvement_pct = (
            (baseline_metrics["test_mae"] - enhanced_metrics["test_mae"])
            / baseline_metrics["test_mae"]
            * 100
        )
        print(f"MAE Improvement: {improvement_pct:.2f}%")

        return {
            "baseline": baseline_metrics,
            "enhanced": enhanced_metrics,
            "improvement_pct": round(improvement_pct, 2),
        }

    def _generate_prior_predictions(self, data_loader) -> dict:
        """Generate predictions for all season N using season N-1 data."""
        prior_predictions = {}

        df = data_loader.get_training_dataframe()

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

                # Use current season to predict next season
                derived = calculate_derived_features(current)
                position = current["position"]
                current_ktc = current["end_ktc"]
                next_age = current["age"] + 1
                games_played = current["games_played"]
                age_bracket = get_age_bracket(next_age, position)

                features = {
                    "current_ktc": current_ktc,
                    "age": next_age,
                    "years_exp": current["years_exp"] + 1,
                    "fantasy_points": current["fantasy_points"],
                    "games_played": games_played,
                    "games_missed": 17 - games_played,
                    "fp_vs_position_avg": 1.0,
                    **derived,
                    f"pos_{position}": 1,
                    f"age_{age_bracket}": 1,
                    "prior_predicted_ktc": 0,  # Not available for baseline
                    "age_x_is_rb": next_age * (1 if position == "RB" else 0),
                    "age_x_is_qb": next_age * (1 if position == "QB" else 0),
                    "games_x_consistency": games_played * derived.get("fp_consistency", 0),
                    "ktc_x_volatility": current_ktc * derived.get("ktc_in_season_volatility", 0) / 10000,
                    "ktc_x_trend": current_ktc * derived.get("ktc_season_trend", 0) / 10000,
                }

                # Model predicts ratio, convert to absolute
                predicted_ratio = self.predictor.predict(features)
                predicted_ktc = predicted_ratio * current_ktc
                # Store as: what we predicted for next_season's end_ktc
                prior_predictions[(player_id, next_s["year"])] = predicted_ktc

        return prior_predictions

    # ========== Position Ensemble Methods ==========

    def train_position_ensemble(self) -> dict:
        """Train position-specific ensemble models.

        Trains separate GradientBoosting models for each position (QB, RB, WR, TE)
        with position-specific hyperparameters for better accuracy.
        """
        data_loader = get_data_loader()
        X, y = data_loader.get_feature_matrix()

        # Build position series aligned with training pairs
        df = data_loader.get_training_dataframe()
        df = df.sort_values(["player_id", "year"])

        positions = []
        for player_id, player_df in df.groupby("player_id"):
            player_df = player_df.sort_values("year")
            seasons = player_df.to_dict("records")
            for i in range(len(seasons) - 1):
                if seasons[i + 1]["year"] == seasons[i]["year"] + 1:
                    if seasons[i]["end_ktc"] > 0 and seasons[i + 1]["end_ktc"] > 0:
                        positions.append(seasons[i]["position"])

        positions_series = pd.Series(positions)

        # Create and train ensemble
        self.ensemble_predictor = PositionEnsemblePredictor()
        metrics = self.ensemble_predictor.train(X, y, positions_series)

        # Save model
        ENSEMBLE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.ensemble_predictor.save(ENSEMBLE_MODEL_PATH)

        self._ensemble_initialized = True
        print("Position ensemble trained:")
        for pos, pos_metrics in metrics.items():
            if pos != "combined":
                print(f"  {pos}: MAE={pos_metrics['test_mae']:.1f}, R²={pos_metrics['test_r2']:.3f}")
        if "combined" in metrics:
            print(f"  Combined: MAE={metrics['combined']['test_mae']:.1f}, R²={metrics['combined']['test_r2']:.3f}")

        return metrics

    def initialize_ensemble(self, force_retrain: bool = False) -> dict:
        """Initialize the ensemble model - load from disk or train new."""
        if self._ensemble_initialized and not force_retrain:
            return self.ensemble_predictor.metrics if self.ensemble_predictor else {}

        # Try to load existing model
        if ENSEMBLE_MODEL_PATH.exists() and not force_retrain:
            try:
                self.ensemble_predictor = PositionEnsemblePredictor()
                self.ensemble_predictor.load(ENSEMBLE_MODEL_PATH)
                self._ensemble_initialized = True
                return self.ensemble_predictor.metrics
            except Exception as e:
                print(f"Failed to load ensemble model: {e}. Training new model...")

        # Train new model
        return self.train_position_ensemble()

    def get_ensemble_metrics(self) -> dict:
        """Get ensemble model performance metrics."""
        if self.ensemble_predictor:
            return self.ensemble_predictor.metrics
        return {}

    def predict_with_ensemble(self, player_id: str) -> Optional[dict]:
        """Get prediction using position-specific ensemble model."""
        if not self._ensemble_initialized:
            self.initialize_ensemble()

        if not self.ensemble_predictor:
            return None

        data_loader = get_data_loader()
        player = data_loader.get_player_by_id(player_id)

        if not player:
            return None

        seasons = player.get("seasons", [])
        if not seasons:
            return None

        seasons_sorted = sorted(seasons, key=lambda s: s["year"])
        latest = seasons_sorted[-1]
        current_ktc = latest.get("end_ktc") or latest.get("start_ktc", 0)
        position = player["position"]

        # Calculate derived features
        derived = calculate_derived_features(latest)
        next_age = latest.get("age", 25) + 1
        games_played = latest.get("games_played", 0)
        age_bracket = get_age_bracket(next_age, position)

        features = {
            "current_ktc": current_ktc,
            "age": next_age,
            "years_exp": latest.get("years_exp", 0) + 1,
            "fantasy_points": latest.get("fantasy_points", 0),
            "games_played": games_played,
            "games_missed": 17 - games_played,
            "fp_vs_position_avg": 1.0,
            **derived,
            f"pos_{position}": 1,
            f"age_{age_bracket}": 1,
            "prior_predicted_ktc": 0,
            "age_x_is_rb": next_age * (1 if position == "RB" else 0),
            "age_x_is_qb": next_age * (1 if position == "QB" else 0),
            "games_x_consistency": games_played * derived.get("fp_consistency", 0),
            "ktc_x_volatility": current_ktc * derived.get("ktc_in_season_volatility", 0) / 10000,
            "ktc_x_trend": current_ktc * derived.get("ktc_season_trend", 0) / 10000,
        }

        # Model predicts ratio, convert to absolute
        predicted_ratio = self.ensemble_predictor.predict(features, position)
        predicted_ktc = predicted_ratio * current_ktc
        ktc_change = predicted_ktc - current_ktc
        ktc_change_pct = (predicted_ratio - 1) * 100

        return {
            "player_id": player_id,
            "name": player["name"],
            "position": position,
            "current_ktc": current_ktc,
            "predicted_ktc": round(predicted_ktc, 2),
            "ktc_change": round(ktc_change, 2),
            "ktc_change_pct": round(ktc_change_pct, 2),
        }

    # ========== XGBoost Model Methods ==========

    def train_xgboost(self, use_optuna: bool = False, n_trials: int = 50) -> dict:
        """Train XGBoost model using NFL-only data (excludes college prospects).

        College/prospect data (games_played=0) is excluded because it adds noise
        and reduces model accuracy by ~7% MAE.

        Args:
            use_optuna: If True, use Optuna to optimize hyperparameters
                        (achieves ~30% MAE reduction but slower)
            n_trials: Number of Optuna trials if use_optuna is True
        """
        if not XGBOOST_AVAILABLE:
            return {"error": "xgboost not installed. Run: pip install xgboost>=2.0.0"}

        data_loader = get_data_loader()
        X, y = data_loader.get_feature_matrix_nfl_only()
        print(f"Training XGBoost with NFL-only data: {len(X)} samples (college prospects excluded)")

        self.xgb_predictor = XGBKTCPredictor()

        if use_optuna:
            print(f"Optimizing XGBoost with {n_trials} Optuna trials...")
            opt_result = self.xgb_predictor.optimize_hyperparameters(
                X, y, n_trials=n_trials, use_log_transform=True
            )
            # Check for error (e.g., optuna not installed)
            if "error" in opt_result:
                return opt_result
            metrics = opt_result["metrics"]
            print(f"Best params: {opt_result['best_params']}")
        else:
            metrics = self.xgb_predictor.train(X, y, use_log_transform=True, use_cv=True)

        # Save model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.xgb_predictor.save(MODEL_PATH)

        self._xgb_initialized = True
        print(f"XGBoost model trained - MAE: {metrics['test_mae']:.4f}, R²: {metrics['test_r2']:.3f}")

        return metrics

    def train_xgboost_nfl_only(self) -> dict:
        """Train XGBoost model excluding college/prospect data.

        Filters out training pairs where the source season has 0 games played,
        which represents college prospects or injured players.

        This tests whether removing noisy college data improves model accuracy.
        NOTE: This model is NOT saved - it's for research comparison only.

        Returns:
            Dictionary with training metrics and comparison to full model.
        """
        if not XGBOOST_AVAILABLE:
            return {"error": "xgboost not installed. Run: pip install xgboost>=2.0.0"}

        data_loader = get_data_loader()

        # Get NFL-only data
        X_nfl, y_nfl = data_loader.get_feature_matrix_nfl_only()
        n_nfl = len(X_nfl)

        # Get full data for comparison
        X_full, y_full = data_loader.get_feature_matrix()
        n_full = len(X_full)
        n_excluded = n_full - n_nfl

        print(f"Training NFL-only model: {n_nfl} samples ({n_excluded} college samples excluded)")

        # Train NFL-only model
        nfl_predictor = XGBKTCPredictor()
        nfl_metrics = nfl_predictor.train(X_nfl, y_nfl, use_log_transform=True, use_cv=True)

        return {
            "nfl_only_metrics": nfl_metrics,
            "n_samples_nfl": n_nfl,
            "n_samples_full": n_full,
            "n_excluded": n_excluded,
            "pct_excluded": round(100 * n_excluded / n_full, 1),
        }

    def train_xgboost_college_weighted(self, college_weight: float = 0.25) -> dict:
        """Train XGBoost model with lower weight for college/prospect data.

        Applies sample weights during training:
        - NFL seasons (games > 0): weight = 1.0
        - College/prospect seasons (games = 0): weight = college_weight

        This allows the model to see college data but not rely on it as heavily.
        NOTE: This model is NOT saved - it's for research comparison only.

        Args:
            college_weight: Weight for college samples (default 0.25 = 4x less important)

        Returns:
            Dictionary with training metrics.
        """
        if not XGBOOST_AVAILABLE:
            return {"error": "xgboost not installed. Run: pip install xgboost>=2.0.0"}

        data_loader = get_data_loader()
        X, y, sample_weights = data_loader.get_feature_matrix_with_college_weights(
            college_weight=college_weight
        )

        n_college = (sample_weights < 1.0).sum()
        n_nfl = (sample_weights == 1.0).sum()

        print(f"Training with college weighting: {n_nfl} NFL + {n_college} college (weight={college_weight})")

        # Train with sample weights
        weighted_predictor = XGBKTCPredictor()
        metrics = weighted_predictor.train(
            X, y, use_log_transform=True, use_cv=True, sample_weight=sample_weights.values
        )

        return {
            "weighted_metrics": metrics,
            "college_weight": college_weight,
            "n_nfl_samples": int(n_nfl),
            "n_college_samples": int(n_college),
        }

    def compare_college_data_impact(self, college_weight: float = 0.25) -> dict:
        """Compare three training approaches for handling college data.

        Trains and compares:
        1. Full model (includes all data)
        2. NFL-only model (excludes college data)
        3. Weighted model (college data at lower weight)

        Returns:
            Comparison of all three approaches with recommendations.
        """
        if not XGBOOST_AVAILABLE:
            return {"error": "xgboost not installed. Run: pip install xgboost>=2.0.0"}

        data_loader = get_data_loader()
        results = {}

        # 1. Train full model
        print("=== Training FULL model (all data) ===")
        X_full, y_full = data_loader.get_feature_matrix()
        full_predictor = XGBKTCPredictor()
        full_metrics = full_predictor.train(X_full, y_full, use_log_transform=True, use_cv=True)
        results["full"] = {
            "test_mae": full_metrics["test_mae"],
            "test_r2": full_metrics["test_r2"],
            "cv_mae": full_metrics.get("cv_mae"),
            "n_samples": len(X_full),
        }

        # 2. Train NFL-only model
        print("\n=== Training NFL-ONLY model (excluding college) ===")
        X_nfl, y_nfl = data_loader.get_feature_matrix_nfl_only()
        nfl_predictor = XGBKTCPredictor()
        nfl_metrics = nfl_predictor.train(X_nfl, y_nfl, use_log_transform=True, use_cv=True)
        results["nfl_only"] = {
            "test_mae": nfl_metrics["test_mae"],
            "test_r2": nfl_metrics["test_r2"],
            "cv_mae": nfl_metrics.get("cv_mae"),
            "n_samples": len(X_nfl),
        }

        # 3. Train weighted model
        print(f"\n=== Training WEIGHTED model (college at {college_weight}x) ===")
        X_w, y_w, weights = data_loader.get_feature_matrix_with_college_weights(
            college_weight=college_weight
        )
        weighted_predictor = XGBKTCPredictor()
        weighted_metrics = weighted_predictor.train(
            X_w, y_w, use_log_transform=True, use_cv=True, sample_weight=weights.values
        )
        results["weighted"] = {
            "test_mae": weighted_metrics["test_mae"],
            "test_r2": weighted_metrics["test_r2"],
            "cv_mae": weighted_metrics.get("cv_mae"),
            "n_samples": len(X_w),
            "college_weight": college_weight,
        }

        # Determine best approach
        mae_scores = {
            "full": results["full"]["test_mae"],
            "nfl_only": results["nfl_only"]["test_mae"],
            "weighted": results["weighted"]["test_mae"],
        }
        best_approach = min(mae_scores, key=mae_scores.get)

        # Calculate improvements
        baseline_mae = results["full"]["test_mae"]
        results["comparison"] = {
            "best_approach": best_approach,
            "nfl_only_vs_full_pct": round(
                100 * (baseline_mae - results["nfl_only"]["test_mae"]) / baseline_mae, 2
            ),
            "weighted_vs_full_pct": round(
                100 * (baseline_mae - results["weighted"]["test_mae"]) / baseline_mae, 2
            ),
        }

        return results

    def train_xgboost_weighted(self) -> dict:
        """Train XGBoost model with sample weighting for under-represented segments.

        This approach applies higher weights during training for segments that
        XGBoost systematically under-predicts:
        - Low KTC players (<2K): 1.5x weight
        - Young players (<25): 1.3x weight
        - Both low KTC AND young: 2.0x weight

        This teaches the model to prioritize accuracy on these samples during
        training, rather than applying post-hoc corrections.

        Returns:
            Dictionary with training metrics
        """
        if not XGBOOST_AVAILABLE:
            return {"error": "xgboost not installed. Run: pip install xgboost>=2.0.0"}

        data_loader = get_data_loader()
        X, y = data_loader.get_feature_matrix()

        self.xgb_predictor = XGBKTCPredictor()
        metrics = self.xgb_predictor.train(
            X, y,
            use_log_transform=True,
            use_cv=True,
            use_sample_weights=True,
        )

        # Save model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.xgb_predictor.save(MODEL_PATH)

        self._xgb_initialized = True
        print(f"XGBoost (weighted) model trained - MAE: {metrics['test_mae']:.4f}, R²: {metrics['test_r2']:.3f}")

        return metrics

    def initialize_xgboost(self, force_retrain: bool = False) -> dict:
        """Initialize XGBoost model - load from disk or train new."""
        if not XGBOOST_AVAILABLE:
            return {"error": "xgboost not installed"}

        if self._xgb_initialized and not force_retrain:
            return self.xgb_predictor.metrics if self.xgb_predictor else {}

        # Try to load existing model
        if MODEL_PATH.exists() and not force_retrain:
            try:
                self.xgb_predictor = XGBKTCPredictor()
                self.xgb_predictor.load(MODEL_PATH)
                self._xgb_initialized = True
                return self.xgb_predictor.metrics
            except Exception as e:
                print(f"Failed to load XGBoost model: {e}. Training new model...")

        return self.train_xgboost()

    def initialize_xgboost_weighted(self, force_retrain: bool = False) -> dict:
        """Initialize XGBoost model with sample weights - load from disk or train new.

        This is the default model for predictions. It loads the existing XGBoost model
        and checks if it was trained with sample weights. If not, retrains with weights.
        """
        if not XGBOOST_AVAILABLE:
            return {"error": "xgboost not installed"}

        if self._xgb_initialized and not force_retrain:
            return self.xgb_predictor.metrics if self.xgb_predictor else {}

        # Try to load existing model
        if MODEL_PATH.exists() and not force_retrain:
            try:
                self.xgb_predictor = XGBKTCPredictor()
                self.xgb_predictor.load(MODEL_PATH)

                # Check if the model was trained with sample weights
                if self.xgb_predictor.metrics.get("sample_weights"):
                    self._xgb_initialized = True
                    return self.xgb_predictor.metrics

                # Model exists but wasn't trained with sample weights - retrain
                print("Existing XGBoost model not trained with sample weights. Retraining...")
            except Exception as e:
                print(f"Failed to load XGBoost model: {e}. Training new model...")

        return self.train_xgboost_weighted()

    def get_xgb_metrics(self) -> dict:
        """Get XGBoost model performance metrics."""
        if self.xgb_predictor:
            return self.xgb_predictor.metrics
        return {}

    def get_xgb_feature_importance(self) -> dict:
        """Get feature importance from XGBoost model."""
        if self.xgb_predictor:
            return self.xgb_predictor.get_feature_importance()
        return {}

    def compare_models(self) -> dict:
        """Compare all available model types including voting ensemble."""
        if not self._is_initialized:
            self.initialize()
        if not self._xgb_initialized:
            self.initialize_xgboost()
        if not self._lgbm_initialized:
            self.initialize_lightgbm()
        if not self._catboost_initialized:
            self.initialize_catboost()
        if not self._voting_ensemble_initialized:
            self.initialize_voting_ensemble()

        gb_metrics = self.get_metrics()
        xgb_metrics = self.get_xgb_metrics()
        lgbm_metrics = self.get_lgbm_metrics()
        catboost_metrics = self.get_catboost_metrics()
        voting_ensemble_metrics = self.get_voting_ensemble_metrics()

        comparison = {
            "gradient_boosting": gb_metrics,
            "xgboost": xgb_metrics,
            "lightgbm": lgbm_metrics,
            "catboost": catboost_metrics,
            "voting_ensemble": voting_ensemble_metrics,
        }

        # Find the best model by MAE
        models_with_mae = []
        if gb_metrics:
            models_with_mae.append(("gradient_boosting", gb_metrics.get("test_mae", float("inf"))))
        if xgb_metrics:
            models_with_mae.append(("xgboost", xgb_metrics.get("test_mae", float("inf"))))
        if lgbm_metrics:
            models_with_mae.append(("lightgbm", lgbm_metrics.get("test_mae", float("inf"))))
        if catboost_metrics:
            models_with_mae.append(("catboost", catboost_metrics.get("test_mae", float("inf"))))
        if voting_ensemble_metrics:
            models_with_mae.append(("voting_ensemble", voting_ensemble_metrics.get("test_mae", float("inf"))))

        if models_with_mae:
            best_model = min(models_with_mae, key=lambda x: x[1])
            comparison["winner"] = best_model[0]
            comparison["best_mae"] = best_model[1]

        return comparison

    def benchmark_improvements(self) -> dict:
        """Benchmark model improvements: baseline vs log-transform vs optimized.

        Trains three versions of the model and compares their metrics:
        1. Baseline: Original model without improvements
        2. Log-transform: Model with log-transform on target
        3. Optimized: Model with Optuna hyperparameter optimization

        Returns comprehensive comparison with improvement percentages.
        """
        from app.models.predictor import KTCPredictor, OPTUNA_AVAILABLE

        data_loader = get_data_loader()
        X, y, metadata = data_loader.get_feature_matrix(include_metadata=True)
        years = metadata["year"]

        results = {}

        # 1. Baseline (no log transform, no optimization)
        print("Training baseline model...")
        baseline_predictor = KTCPredictor()
        baseline_metrics = baseline_predictor.train_temporal(
            X, y, years,
            train_cutoff=2023,
            use_log_transform=False,
        )
        results["baseline"] = {
            "test_mae": baseline_metrics["test_mae"],
            "test_mape": baseline_metrics["test_mape"],
            "test_r2": baseline_metrics["test_r2"],
        }
        print(f"  Baseline MAE: {baseline_metrics['test_mae']:.1f}, MAPE: {baseline_metrics['test_mape']:.2f}%")

        # 2. With log transform
        print("Training with log-transform...")
        log_predictor = KTCPredictor()
        log_metrics = log_predictor.train_temporal(
            X, y, years,
            train_cutoff=2023,
            use_log_transform=True,
        )
        results["log_transform"] = {
            "test_mae": log_metrics["test_mae"],
            "test_mape": log_metrics["test_mape"],
            "test_r2": log_metrics["test_r2"],
        }
        log_improvement = (baseline_metrics["test_mae"] - log_metrics["test_mae"]) / baseline_metrics["test_mae"] * 100
        print(f"  Log-transform MAE: {log_metrics['test_mae']:.1f} ({log_improvement:+.1f}%)")

        # 3. With Optuna optimization (if available)
        if OPTUNA_AVAILABLE:
            print("Training with Optuna optimization (this may take a few minutes)...")
            opt_predictor = KTCPredictor()
            opt_result = opt_predictor.optimize_hyperparameters(
                X, y,
                n_trials=30,  # Reduced trials for benchmark
                use_log_transform=True,
            )
            opt_metrics = opt_result["metrics"]
            results["optimized"] = {
                "test_mae": opt_metrics["test_mae"],
                "test_mape": opt_metrics["test_mape"],
                "test_r2": opt_metrics["test_r2"],
                "best_params": opt_result["best_params"],
            }
            opt_improvement = (baseline_metrics["test_mae"] - opt_metrics["test_mae"]) / baseline_metrics["test_mae"] * 100
            print(f"  Optimized MAE: {opt_metrics['test_mae']:.1f} ({opt_improvement:+.1f}%)")
        else:
            results["optimized"] = {"error": "Optuna not installed"}

        # Summary
        results["summary"] = {
            "baseline_mae": results["baseline"]["test_mae"],
            "log_transform_improvement_pct": round(log_improvement, 2),
        }

        if "test_mae" in results.get("optimized", {}):
            results["summary"]["optimized_improvement_pct"] = round(opt_improvement, 2)
            results["summary"]["best_model"] = "optimized" if opt_metrics["test_mae"] < log_metrics["test_mae"] else "log_transform"
        else:
            results["summary"]["best_model"] = "log_transform"

        return results

    # ========== LightGBM Model Methods (Phase 3.1) ==========

    def train_lightgbm(self) -> dict:
        """Train LightGBM model as alternative to GradientBoosting."""
        if not LIGHTGBM_AVAILABLE:
            return {"error": "lightgbm not installed. Run: pip install lightgbm>=4.0.0"}

        data_loader = get_data_loader()
        X, y = data_loader.get_feature_matrix()

        self.lgbm_predictor = LGBMKTCPredictor()
        metrics = self.lgbm_predictor.train(X, y)

        # Save model
        LIGHTGBM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.lgbm_predictor.save(LIGHTGBM_MODEL_PATH)

        self._lgbm_initialized = True
        print(f"LightGBM model trained - MAE: {metrics['test_mae']:.1f}, R²: {metrics['test_r2']:.3f}")

        return metrics

    def initialize_lightgbm(self, force_retrain: bool = False) -> dict:
        """Initialize LightGBM model - load from disk or train new."""
        if not LIGHTGBM_AVAILABLE:
            return {"error": "lightgbm not installed"}

        if self._lgbm_initialized and not force_retrain:
            return self.lgbm_predictor.metrics if self.lgbm_predictor else {}

        # Try to load existing model
        if LIGHTGBM_MODEL_PATH.exists() and not force_retrain:
            try:
                self.lgbm_predictor = LGBMKTCPredictor()
                self.lgbm_predictor.load(LIGHTGBM_MODEL_PATH)
                self._lgbm_initialized = True
                return self.lgbm_predictor.metrics
            except Exception as e:
                print(f"Failed to load LightGBM model: {e}. Training new model...")

        return self.train_lightgbm()

    def get_lgbm_metrics(self) -> dict:
        """Get LightGBM model performance metrics."""
        if self.lgbm_predictor:
            return self.lgbm_predictor.metrics
        return {}

    def get_lgbm_feature_importance(self) -> dict:
        """Get feature importance from LightGBM model."""
        if self.lgbm_predictor:
            return self.lgbm_predictor.get_feature_importance()
        return {}

    # ========== CatBoost Model Methods ==========

    def train_catboost(self) -> dict:
        """Train CatBoost model as alternative to GradientBoosting.

        CatBoost offers native categorical feature handling and ordered
        boosting which may improve predictions for position/age features.
        """
        if not CATBOOST_AVAILABLE:
            return {"error": "catboost not installed. Run: pip install catboost>=1.2.0"}

        data_loader = get_data_loader()
        X, y = data_loader.get_feature_matrix()

        self.catboost_predictor = CatBoostKTCPredictor()
        metrics = self.catboost_predictor.train(X, y)

        # Save model
        CATBOOST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.catboost_predictor.save(CATBOOST_MODEL_PATH)

        self._catboost_initialized = True
        print(f"CatBoost model trained - MAE: {metrics['test_mae']:.1f}, R²: {metrics['test_r2']:.3f}")

        return metrics

    def initialize_catboost(self, force_retrain: bool = False) -> dict:
        """Initialize CatBoost model - load from disk or train new."""
        if not CATBOOST_AVAILABLE:
            return {"error": "catboost not installed"}

        if self._catboost_initialized and not force_retrain:
            return self.catboost_predictor.metrics if self.catboost_predictor else {}

        # Try to load existing model
        if CATBOOST_MODEL_PATH.exists() and not force_retrain:
            try:
                self.catboost_predictor = CatBoostKTCPredictor()
                self.catboost_predictor.load(CATBOOST_MODEL_PATH)
                self._catboost_initialized = True
                return self.catboost_predictor.metrics
            except Exception as e:
                print(f"Failed to load CatBoost model: {e}. Training new model...")

        return self.train_catboost()

    def get_catboost_metrics(self) -> dict:
        """Get CatBoost model performance metrics."""
        if self.catboost_predictor:
            return self.catboost_predictor.metrics
        return {}

    def get_catboost_feature_importance(self) -> dict:
        """Get feature importance from CatBoost model."""
        if self.catboost_predictor:
            return self.catboost_predictor.get_feature_importance()
        return {}

    # ========== Breakout-Aware Model Methods ==========

    def train_breakout_model(self) -> dict:
        """Train breakout-aware model for better young player predictions."""
        data_loader = get_data_loader()
        X, y = data_loader.get_feature_matrix()

        self.breakout_predictor = BreakoutAwarePredictor()
        metrics = self.breakout_predictor.train(X, y)

        self._breakout_initialized = True
        print(f"Breakout-aware model trained - MAE: {metrics['test_mae']:.1f}, R²: {metrics['test_r2']:.3f}")

        return metrics

    def initialize_breakout(self, force_retrain: bool = False) -> dict:
        """Initialize breakout-aware model."""
        if self._breakout_initialized and not force_retrain:
            return self.breakout_predictor.metrics if self.breakout_predictor else {}

        return self.train_breakout_model()

    def predict_with_breakout(self, player_id: str) -> Optional[dict]:
        """Get prediction using breakout-aware model."""
        if not self._breakout_initialized:
            self.initialize_breakout()

        data_loader = get_data_loader()
        player = data_loader.get_player_by_id(player_id)

        if not player:
            return None

        seasons = player.get("seasons", [])
        if not seasons:
            return None

        seasons_sorted = sorted(seasons, key=lambda x: x["year"])
        latest = seasons_sorted[-1]
        position = player["position"]
        current_ktc = latest.get("end_ktc") or latest.get("start_ktc", 0)

        # Calculate derived features
        derived = calculate_derived_features(latest)
        next_age = latest.get("age", 25) + 1
        games_played = latest.get("games_played", 0)

        # Build features
        features = self._build_features_dict(
            latest, derived, position, current_ktc, next_age, games_played, 0.0
        )

        # Model predicts ratio, convert to absolute
        predicted_ratio = self.breakout_predictor.predict(features)
        predicted_ktc = predicted_ratio * current_ktc
        ktc_change = predicted_ktc - current_ktc
        ktc_change_pct = (predicted_ratio - 1) * 100

        # Determine if breakout boost was applied
        breakout_boost_applied = (
            next_age <= 25 and
            features.get("momentum_score", 0) > 0.8 and
            features.get("ktc_upside_ratio", 0) > 0.4
        )

        return {
            "player_id": player_id,
            "name": player["name"],
            "position": position,
            "current_ktc": current_ktc,
            "predicted_ktc": round(predicted_ktc, 2),
            "ktc_change": round(ktc_change, 2),
            "ktc_change_pct": round(ktc_change_pct, 2),
            "breakout_boost_applied": breakout_boost_applied,
            "momentum_score": round(features.get("momentum_score", 0), 3),
            "ktc_upside_ratio": round(features.get("ktc_upside_ratio", 0), 3),
        }

    def get_breakout_metrics(self) -> dict:
        """Get breakout-aware model metrics."""
        if self.breakout_predictor:
            return self.breakout_predictor.metrics
        return {}

    # ========== Calibrated Model Methods ==========

    def train_calibrated_model(self) -> dict:
        """Train calibrated model for better elite tier predictions.

        Applies tier-based post-prediction calibration to address
        systematic under-prediction of elite tier players (~2,096 KTC bias).
        """
        data_loader = get_data_loader()
        X, y = data_loader.get_feature_matrix()

        self.calibrated_predictor = CalibratedPredictor()
        metrics = self.calibrated_predictor.train(X, y)

        # Save model
        CALIBRATED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.calibrated_predictor.save(CALIBRATED_MODEL_PATH)

        self._calibrated_initialized = True
        print(f"Calibrated model trained - MAE: {metrics['test_mae']:.1f}, R²: {metrics['test_r2']:.3f}")

        return metrics

    def initialize_calibrated(self, force_retrain: bool = False) -> dict:
        """Initialize calibrated model - load from disk or train new."""
        if self._calibrated_initialized and not force_retrain:
            return self.calibrated_predictor.metrics if self.calibrated_predictor else {}

        # Try to load existing model
        if CALIBRATED_MODEL_PATH.exists() and not force_retrain:
            try:
                self.calibrated_predictor = CalibratedPredictor()
                self.calibrated_predictor.load(CALIBRATED_MODEL_PATH)
                self._calibrated_initialized = True
                return self.calibrated_predictor.metrics
            except Exception as e:
                print(f"Failed to load calibrated model: {e}. Training new model...")

        return self.train_calibrated_model()

    def predict_with_calibration(self, player_id: str) -> Optional[dict]:
        """Get prediction using calibrated model."""
        if not self._calibrated_initialized:
            self.initialize_calibrated()

        data_loader = get_data_loader()
        player = data_loader.get_player_by_id(player_id)

        if not player:
            return None

        seasons = player.get("seasons", [])
        if not seasons:
            return None

        seasons_sorted = sorted(seasons, key=lambda x: x["year"])
        latest = seasons_sorted[-1]
        position = player["position"]
        current_ktc = latest.get("end_ktc") or latest.get("start_ktc", 0)

        # Calculate derived features
        derived = calculate_derived_features(latest)
        next_age = latest.get("age", 25) + 1
        games_played = latest.get("games_played", 0)

        # Build features
        features = self._build_features_dict(
            latest, derived, position, current_ktc, next_age, games_played, 0.0
        )

        # Model predicts ratio, convert to absolute
        predicted_ratio = self.calibrated_predictor.predict(features)
        predicted_ktc = predicted_ratio * current_ktc
        ktc_change = predicted_ktc - current_ktc
        ktc_change_pct = (predicted_ratio - 1) * 100

        # Determine which tier calibration was applied
        tier = self.calibrated_predictor._get_tier(current_ktc)
        calibration_factor = self.calibrated_predictor.calibration_factors[tier]

        return {
            "player_id": player_id,
            "name": player["name"],
            "position": position,
            "current_ktc": current_ktc,
            "predicted_ktc": round(predicted_ktc, 2),
            "ktc_change": round(ktc_change, 2),
            "ktc_change_pct": round(ktc_change_pct, 2),
            "calibration_tier": tier,
            "calibration_factor": calibration_factor,
        }

    def get_calibrated_metrics(self) -> dict:
        """Get calibrated model metrics."""
        if self.calibrated_predictor:
            return self.calibrated_predictor.metrics
        return {}

    # ========== Calibrated Breakout Model Methods ==========

    def train_calibrated_breakout_model(self) -> dict:
        """Train combined calibration + breakout model.

        This model achieves:
        - Elite tier bias: <±500 (target met)
        - Breakout detection: >50% (target exceeded at 67%+)

        It applies:
        - Tier-based calibration for all predictions
        - Breakout boost for mid/low tier young players with signals
        - Reduced calibration when breakout boost is applied to avoid over-correction
        """
        data_loader = get_data_loader()
        X, y = data_loader.get_feature_matrix()

        self.calibrated_breakout_predictor = CalibratedBreakoutPredictor()
        metrics = self.calibrated_breakout_predictor.train(X, y, use_log_transform=True)

        # Save model
        CALIBRATED_BREAKOUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.calibrated_breakout_predictor.save(CALIBRATED_BREAKOUT_MODEL_PATH)

        self._calibrated_breakout_initialized = True
        print(f"Calibrated+Breakout model trained - MAE: {metrics['test_mae']:.1f}, R²: {metrics['test_r2']:.3f}")

        return metrics

    def initialize_calibrated_breakout(self, force_retrain: bool = False) -> dict:
        """Initialize calibrated+breakout model - load from disk or train new."""
        if self._calibrated_breakout_initialized and not force_retrain:
            return self.calibrated_breakout_predictor.metrics if self.calibrated_breakout_predictor else {}

        # Try to load existing model
        if CALIBRATED_BREAKOUT_MODEL_PATH.exists() and not force_retrain:
            try:
                self.calibrated_breakout_predictor = CalibratedBreakoutPredictor()
                self.calibrated_breakout_predictor.load(CALIBRATED_BREAKOUT_MODEL_PATH)
                self._calibrated_breakout_initialized = True
                return self.calibrated_breakout_predictor.metrics
            except Exception as e:
                print(f"Failed to load calibrated+breakout model: {e}. Training new model...")

        return self.train_calibrated_breakout_model()

    def predict_with_calibrated_breakout(self, player_id: str) -> Optional[dict]:
        """Get prediction using calibrated+breakout model."""
        if not self._calibrated_breakout_initialized:
            self.initialize_calibrated_breakout()

        data_loader = get_data_loader()
        player = data_loader.get_player_by_id(player_id)

        if not player:
            return None

        seasons = player.get("seasons", [])
        if not seasons:
            return None

        seasons_sorted = sorted(seasons, key=lambda x: x["year"])
        latest = seasons_sorted[-1]
        position = player["position"]
        current_ktc = latest.get("end_ktc") or latest.get("start_ktc", 0)

        # Calculate derived features
        derived = calculate_derived_features(latest)
        next_age = latest.get("age", 25) + 1
        games_played = latest.get("games_played", 0)

        # Build features
        features = self._build_features_dict(
            latest, derived, position, current_ktc, next_age, games_played, 0.0
        )

        # Model predicts ratio, convert to absolute
        predicted_ratio = self.calibrated_breakout_predictor.predict(features)
        predicted_ktc = predicted_ratio * current_ktc
        ktc_change = predicted_ktc - current_ktc
        ktc_change_pct = (predicted_ratio - 1) * 100

        # Determine which adjustments were applied
        tier = self.calibrated_breakout_predictor._get_tier(current_ktc)
        breakout_applied = self.calibrated_breakout_predictor._should_apply_breakout_boost(features, tier)

        return {
            "player_id": player_id,
            "name": player["name"],
            "position": position,
            "current_ktc": current_ktc,
            "predicted_ktc": round(predicted_ktc, 2),
            "ktc_change": round(ktc_change, 2),
            "ktc_change_pct": round(ktc_change_pct, 2),
            "tier": tier,
            "breakout_boost_applied": breakout_applied,
        }

    def get_calibrated_breakout_metrics(self) -> dict:
        """Get calibrated+breakout model metrics."""
        if self.calibrated_breakout_predictor:
            return self.calibrated_breakout_predictor.metrics
        return {}

    # ========== Voting Ensemble Model Methods ==========

    def train_voting_ensemble(self, optimize_weights: bool = False) -> dict:
        """Train voting ensemble combining GB, XGBoost, LightGBM, and CatBoost.

        This ensemble averages predictions from multiple model types to reduce
        prediction variance. Expected improvement: 10-15% MAE reduction.

        Args:
            optimize_weights: If True, optimize weights via CV (slower but better)

        Returns:
            Dictionary with ensemble metrics and individual model metrics
        """
        data_loader = get_data_loader()
        X, y = data_loader.get_feature_matrix()

        self.voting_ensemble_predictor = VotingEnsemblePredictor()
        metrics = self.voting_ensemble_predictor.train(
            X, y,
            use_log_transform=True,
            optimize_weights=optimize_weights,
        )

        # Save model
        VOTING_ENSEMBLE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.voting_ensemble_predictor.save(VOTING_ENSEMBLE_MODEL_PATH)

        self._voting_ensemble_initialized = True
        print(f"Voting ensemble trained - MAE: {metrics['test_mae']:.4f}, R²: {metrics['test_r2']:.4f}")

        return metrics

    def initialize_voting_ensemble(self, force_retrain: bool = False) -> dict:
        """Initialize voting ensemble - load from disk or train new."""
        if self._voting_ensemble_initialized and not force_retrain:
            return self.voting_ensemble_predictor.metrics if self.voting_ensemble_predictor else {}

        # Try to load existing model
        if VOTING_ENSEMBLE_MODEL_PATH.exists() and not force_retrain:
            try:
                self.voting_ensemble_predictor = VotingEnsemblePredictor()
                self.voting_ensemble_predictor.load(VOTING_ENSEMBLE_MODEL_PATH)
                self._voting_ensemble_initialized = True
                return self.voting_ensemble_predictor.metrics
            except Exception as e:
                print(f"Failed to load voting ensemble: {e}. Training new ensemble...")

        return self.train_voting_ensemble()

    def predict_with_voting_ensemble(self, player_id: str) -> Optional[dict]:
        """Get prediction using voting ensemble."""
        if not self._voting_ensemble_initialized:
            self.initialize_voting_ensemble()

        if not self.voting_ensemble_predictor:
            return None

        data_loader = get_data_loader()
        player = data_loader.get_player_by_id(player_id)

        if not player:
            return None

        seasons = player.get("seasons", [])
        if not seasons:
            return None

        seasons_sorted = sorted(seasons, key=lambda s: s["year"])
        latest = seasons_sorted[-1]
        position = player["position"]
        current_ktc = latest.get("end_ktc") or latest.get("start_ktc", 0)

        # Calculate derived features
        derived = calculate_derived_features(latest)
        next_age = latest.get("age", 25) + 1
        games_played = latest.get("games_played", 0)

        # Build features
        features = self._build_features_dict(
            latest, derived, position, current_ktc, next_age, games_played, 0.0
        )

        # Model predicts ratio, convert to absolute
        predicted_ratio = self.voting_ensemble_predictor.predict(features)
        predicted_ktc = predicted_ratio * current_ktc
        ktc_change = predicted_ktc - current_ktc
        ktc_change_pct = (predicted_ratio - 1) * 100

        return {
            "player_id": player_id,
            "name": player["name"],
            "position": position,
            "current_ktc": current_ktc,
            "predicted_ktc": round(predicted_ktc, 2),
            "ktc_change": round(ktc_change, 2),
            "ktc_change_pct": round(ktc_change_pct, 2),
            "ensemble_models": self.voting_ensemble_predictor.model_names,
            "ensemble_weights": dict(zip(
                self.voting_ensemble_predictor.model_names,
                self.voting_ensemble_predictor.weights
            )),
        }

    def get_voting_ensemble_metrics(self) -> dict:
        """Get voting ensemble metrics."""
        if self.voting_ensemble_predictor:
            return self.voting_ensemble_predictor.metrics
        return {}

    def get_voting_ensemble_feature_importance(self) -> dict:
        """Get feature importance from voting ensemble."""
        if self.voting_ensemble_predictor:
            return self.voting_ensemble_predictor.get_feature_importance()
        return {}

    # ========== XGBoost Calibrated Breakout Model Methods ==========

    def train_xgb_calibrated_breakout(self, use_optuna: bool = False, n_trials: int = 50) -> dict:
        """Train XGBoost with calibration + breakout detection (NFL-only data).

        Combines the accuracy of XGBoost with tier-based calibration and
        breakout detection for young players. Uses NFL-only data (excludes
        college prospects) for ~7% better accuracy.

        Args:
            use_optuna: If True, optimize XGBoost hyperparameters first
            n_trials: Number of Optuna trials if use_optuna is True

        Returns:
            Dictionary with training metrics
        """
        if not XGBOOST_AVAILABLE:
            return {"error": "xgboost not installed. Run: pip install xgboost>=2.0.0"}

        data_loader = get_data_loader()
        X, y = data_loader.get_feature_matrix_nfl_only()
        print(f"Training XGBoost Calibrated+Breakout with NFL-only data: {len(X)} samples")

        self.xgb_calibrated_breakout_predictor = XGBCalibratedBreakoutPredictor()

        if use_optuna:
            print(f"Optimizing XGBoost with {n_trials} Optuna trials...")
            opt_result = self.xgb_calibrated_breakout_predictor.optimize_hyperparameters(
                X, y, n_trials=n_trials, use_log_transform=True
            )
            if "error" in opt_result:
                return opt_result
            metrics = opt_result["metrics"]
            print(f"Best params: {opt_result['best_params']}")
        else:
            metrics = self.xgb_calibrated_breakout_predictor.train(
                X, y, use_log_transform=True, use_cv=True
            )

        # Save model
        XGB_CALIBRATED_BREAKOUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.xgb_calibrated_breakout_predictor.save(XGB_CALIBRATED_BREAKOUT_MODEL_PATH)

        self._xgb_calibrated_breakout_initialized = True
        print(f"XGBoost Calibrated+Breakout model trained - MAE: {metrics['test_mae']:.4f}, R²: {metrics['test_r2']:.4f}")

        return metrics

    def initialize_xgb_calibrated_breakout(self, force_retrain: bool = False) -> dict:
        """Initialize XGBoost calibrated+breakout model - load from disk or train new."""
        if not XGBOOST_AVAILABLE:
            return {"error": "xgboost not installed"}

        if self._xgb_calibrated_breakout_initialized and not force_retrain:
            return self.xgb_calibrated_breakout_predictor.metrics if self.xgb_calibrated_breakout_predictor else {}

        # Try to load existing model
        if XGB_CALIBRATED_BREAKOUT_MODEL_PATH.exists() and not force_retrain:
            try:
                self.xgb_calibrated_breakout_predictor = XGBCalibratedBreakoutPredictor()
                self.xgb_calibrated_breakout_predictor.load(XGB_CALIBRATED_BREAKOUT_MODEL_PATH)
                self._xgb_calibrated_breakout_initialized = True
                return self.xgb_calibrated_breakout_predictor.metrics
            except Exception as e:
                print(f"Failed to load XGBoost calibrated+breakout model: {e}. Training new model...")

        return self.train_xgb_calibrated_breakout()

    def predict_with_xgb_calibrated_breakout(self, player_id: str) -> Optional[dict]:
        """Get prediction using XGBoost calibrated+breakout model."""
        if not self._xgb_calibrated_breakout_initialized:
            self.initialize_xgb_calibrated_breakout()

        if not self.xgb_calibrated_breakout_predictor:
            return None

        data_loader = get_data_loader()
        player = data_loader.get_player_by_id(player_id)

        if not player:
            return None

        seasons = player.get("seasons", [])
        if not seasons:
            return None

        seasons_sorted = sorted(seasons, key=lambda s: s["year"])
        latest = seasons_sorted[-1]
        position = player["position"]
        current_ktc = latest.get("end_ktc") or latest.get("start_ktc", 0)

        # Calculate derived features
        derived = calculate_derived_features(latest)
        next_age = latest.get("age", 25) + 1
        games_played = latest.get("games_played", 0)

        # Build features
        features = self._build_features_dict(
            latest, derived, position, current_ktc, next_age, games_played, 0.0
        )

        # Model predicts ratio, convert to absolute
        predicted_ratio = self.xgb_calibrated_breakout_predictor.predict(features)
        predicted_ktc = predicted_ratio * current_ktc
        ktc_change = predicted_ktc - current_ktc
        ktc_change_pct = (predicted_ratio - 1) * 100

        # Determine which adjustments were applied
        tier = self.xgb_calibrated_breakout_predictor._get_tier(current_ktc)
        breakout_applied = self.xgb_calibrated_breakout_predictor._should_apply_breakout_boost(features, tier)

        return {
            "player_id": player_id,
            "name": player["name"],
            "position": position,
            "current_ktc": current_ktc,
            "predicted_ktc": round(predicted_ktc, 2),
            "ktc_change": round(ktc_change, 2),
            "ktc_change_pct": round(ktc_change_pct, 2),
            "tier": tier,
            "breakout_boost_applied": breakout_applied,
            "model": "xgb_calibrated_breakout",
        }

    def get_xgb_calibrated_breakout_metrics(self) -> dict:
        """Get XGBoost calibrated+breakout model metrics."""
        if self.xgb_calibrated_breakout_predictor:
            return self.xgb_calibrated_breakout_predictor.metrics
        return {}

    def get_xgb_calibrated_breakout_feature_importance(self) -> dict:
        """Get feature importance from XGBoost calibrated+breakout model."""
        if self.xgb_calibrated_breakout_predictor:
            return self.xgb_calibrated_breakout_predictor.get_feature_importance()
        return {}

    # ========== Hybrid Ensemble Model Methods ==========

    def train_hybrid_ensemble(self) -> dict:
        """Train hybrid ensemble combining ratio-based and absolute-value models.

        This ensemble addresses the systematic under-prediction for low-KTC players
        by adaptively weighting predictions from:
        - Ratio model (XGBoost): Works well for high-KTC players
        - Absolute model (XGBoost): Avoids compression bias for low-KTC players

        Adaptive weighting by KTC tier:
        | Tier | Current KTC | Absolute Weight | Ratio Weight |
        |------|-------------|-----------------|--------------|
        | Low  | < 2,000     | 70%             | 30%          |
        | Mid  | 2K - 5K     | 50%             | 50%          |
        | High | > 5,000     | 30%             | 70%          |

        Expected improvement:
        - Low-KTC bias: -59% → ~-15% (70% improvement)
        - Overall MAE: ~5% reduction through variance reduction

        Returns:
            Dictionary with training metrics including tier-specific performance
        """
        if not XGBOOST_AVAILABLE:
            return {"error": "xgboost not installed. Run: pip install xgboost>=2.0.0"}

        data_loader = get_data_loader()
        X, y_ratio, y_absolute = data_loader.get_feature_matrix_with_absolute()

        print("Training hybrid ensemble (ratio + absolute models)...")
        self.hybrid_ensemble_predictor = HybridEnsemblePredictor()
        metrics = self.hybrid_ensemble_predictor.train(X, y_ratio, y_absolute)

        # Save model
        HYBRID_ENSEMBLE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.hybrid_ensemble_predictor.save(HYBRID_ENSEMBLE_MODEL_PATH)

        self._hybrid_ensemble_initialized = True
        print(f"Hybrid ensemble trained - MAE: {metrics['test_mae']:.1f}, R²: {metrics['test_r2']:.3f}")

        return metrics

    def initialize_hybrid_ensemble(self, force_retrain: bool = False) -> dict:
        """Initialize hybrid ensemble - load from disk or train new."""
        if not XGBOOST_AVAILABLE:
            return {"error": "xgboost not installed"}

        if self._hybrid_ensemble_initialized and not force_retrain:
            return self.hybrid_ensemble_predictor.metrics if self.hybrid_ensemble_predictor else {}

        # Try to load existing model
        if HYBRID_ENSEMBLE_MODEL_PATH.exists() and not force_retrain:
            try:
                self.hybrid_ensemble_predictor = HybridEnsemblePredictor()
                self.hybrid_ensemble_predictor.load(HYBRID_ENSEMBLE_MODEL_PATH)
                self._hybrid_ensemble_initialized = True
                return self.hybrid_ensemble_predictor.metrics
            except Exception as e:
                print(f"Failed to load hybrid ensemble: {e}. Training new model...")

        return self.train_hybrid_ensemble()

    def predict_with_hybrid_ensemble(self, player_id: str) -> Optional[dict]:
        """Get prediction using hybrid ensemble with tier-adaptive weighting.

        Returns prediction along with:
        - tier: The player's KTC tier (low/mid/high)
        - absolute_weight, ratio_weight: The weights used for blending
        - ratio_prediction, absolute_prediction: Individual model contributions
        """
        if not self._hybrid_ensemble_initialized:
            self.initialize_hybrid_ensemble()

        if not self.hybrid_ensemble_predictor:
            return None

        data_loader = get_data_loader()
        player = data_loader.get_player_by_id(player_id)

        if not player:
            return None

        seasons = player.get("seasons", [])
        if not seasons:
            return None

        seasons_sorted = sorted(seasons, key=lambda s: s["year"])
        latest = seasons_sorted[-1]
        position = player["position"]
        current_ktc = latest.get("end_ktc") or latest.get("start_ktc", 0)

        # Calculate derived features
        derived = calculate_derived_features(latest)
        next_age = latest.get("age", 25) + 1
        games_played = latest.get("games_played", 0)

        # Build features
        features = self._build_features_dict(
            latest, derived, position, current_ktc, next_age, games_played, 0.0
        )

        # Get detailed prediction
        pred_details = self.hybrid_ensemble_predictor.predict_with_details(features)

        predicted_ktc = pred_details["predicted_ktc"]
        ktc_change = predicted_ktc - current_ktc
        ktc_change_pct = ((predicted_ktc / current_ktc) - 1) * 100 if current_ktc > 0 else 0

        return {
            "player_id": player_id,
            "name": player["name"],
            "position": position,
            "current_ktc": current_ktc,
            "predicted_ktc": round(predicted_ktc, 2),
            "ktc_change": round(ktc_change, 2),
            "ktc_change_pct": round(ktc_change_pct, 2),
            "tier": pred_details["tier"],
            "absolute_weight": pred_details["absolute_weight"],
            "ratio_weight": pred_details["ratio_weight"],
            "ratio_prediction": round(pred_details["ratio_prediction"], 2),
            "absolute_prediction": round(pred_details["absolute_prediction"], 2),
            "model": "hybrid_ensemble",
        }

    def get_hybrid_ensemble_metrics(self) -> dict:
        """Get hybrid ensemble metrics including tier-specific performance."""
        if self.hybrid_ensemble_predictor:
            return self.hybrid_ensemble_predictor.metrics
        return {}

    def get_hybrid_ensemble_feature_importance(self) -> dict:
        """Get feature importance from hybrid ensemble (averaged across both models)."""
        if self.hybrid_ensemble_predictor:
            return self.hybrid_ensemble_predictor.get_feature_importance()
        return {}

    # ========== Weekly Model Methods ==========

    def initialize_weekly(self, force_retrain: bool = False) -> dict:
        """Initialize the weekly model - load from disk or train new."""
        if self._weekly_initialized and not force_retrain:
            return self.weekly_predictor.metrics

        # Try to load existing model
        if WEEKLY_MODEL_PATH.exists() and not force_retrain:
            try:
                self.weekly_predictor.load(WEEKLY_MODEL_PATH)
                self._weekly_initialized = True
                return self.weekly_predictor.metrics
            except Exception as e:
                print(f"Failed to load weekly model: {e}. Training new model...")

        # Train new model
        return self.train_weekly_model()

    def train_weekly_model(self) -> dict:
        """Train a new weekly KTC change model."""
        data_loader = get_data_loader()
        X, y = data_loader.get_weekly_feature_matrix()

        print(f"Training weekly model on {len(X)} samples...")
        metrics = self.weekly_predictor.train(X, y)

        # Save the trained model
        WEEKLY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.weekly_predictor.save(WEEKLY_MODEL_PATH)

        self._weekly_initialized = True
        print(f"Weekly model trained - MAE: {metrics['test_mae']:.1f}, R²: {metrics['test_r2']:.3f}")
        return metrics

    def get_weekly_metrics(self) -> dict:
        """Get weekly model performance metrics."""
        return self.weekly_predictor.metrics

    def get_weekly_feature_importance(self) -> dict:
        """Get feature importance from the weekly model."""
        return self.weekly_predictor.get_feature_importance()

    def simulate_trajectory(
        self, player_id: str, games: int, expected_ppg: float
    ) -> Optional[dict]:
        """Simulate KTC trajectory for a player over N games at given PPG."""
        if not self._weekly_initialized:
            self.initialize_weekly()

        data_loader = get_data_loader()
        player = data_loader.get_player_by_id(player_id)

        if not player:
            return None

        seasons = player.get("seasons", [])
        if not seasons:
            return None

        # Get latest season data
        latest = max(seasons, key=lambda s: s["year"])
        current_ktc = latest.get("end_ktc") or latest.get("start_ktc", 0)
        age = latest.get("age", 25)
        years_exp = latest.get("years_exp", 0)
        position = player["position"]

        if current_ktc <= 0:
            return None

        # Initialize trajectory
        trajectory = [{"week": 0, "ktc": current_ktc, "fp": 0, "change": 0}]

        running_ktc = current_ktc
        ytd_fp = 0.0
        ytd_games = 0

        # Simulate each week/game
        for week in range(1, games + 1):
            features = {
                "current_ktc": running_ktc,
                "weekly_fantasy_points": expected_ppg,
                "games_played_this_week": 1,
                "week_of_season": week,
                "ytd_fantasy_points": ytd_fp,
                "ytd_games": ytd_games,
                "age": age,
                "years_exp": years_exp,
                f"pos_{position}": 1,
            }

            predicted_change = self.weekly_predictor.predict(features)
            running_ktc = max(0, running_ktc + predicted_change)
            ytd_fp += expected_ppg
            ytd_games += 1

            trajectory.append({
                "week": week,
                "ktc": round(running_ktc, 2),
                "fp": expected_ppg,
                "change": round(predicted_change, 2),
            })

        total_change = running_ktc - current_ktc
        change_pct = (total_change / current_ktc * 100) if current_ktc > 0 else 0

        return {
            "player_id": player_id,
            "name": player["name"],
            "position": position,
            "starting_ktc": current_ktc,
            "final_ktc": round(running_ktc, 2),
            "total_change": round(total_change, 2),
            "total_change_pct": round(change_pct, 2),
            "games": games,
            "ppg": expected_ppg,
            "trajectory": trajectory,
        }

    def simulate_curve(self, player_id: str, games: int) -> Optional[list[dict]]:
        """Generate PPG-to-KTC curve for a player over N games.

        Returns array of {ppg, predicted_ktc} for PPG values 0 to 40.
        """
        if not self._weekly_initialized:
            self.initialize_weekly()

        data_loader = get_data_loader()
        player = data_loader.get_player_by_id(player_id)

        if not player:
            return None

        seasons = player.get("seasons", [])
        if not seasons:
            return None

        latest = max(seasons, key=lambda s: s["year"])
        current_ktc = latest.get("end_ktc") or latest.get("start_ktc", 0)

        if current_ktc <= 0:
            return None

        # Calculate current PPG for reference
        games_played = latest.get("games_played", 0)
        fantasy_points = latest.get("fantasy_points", 0)
        current_ppg = fantasy_points / games_played if games_played > 0 else 0

        results = []

        # Generate curve for PPG 0 to 40 (step 2)
        for ppg in range(0, 42, 2):
            simulation = self.simulate_trajectory(player_id, games, float(ppg))
            if simulation:
                results.append({
                    "ppg": ppg,
                    "predicted_ktc": simulation["final_ktc"],
                })

        return {
            "player_id": player_id,
            "name": player["name"],
            "position": player["position"],
            "starting_ktc": current_ktc,
            "current_ppg": round(current_ppg, 2),
            "games": games,
            "curve": results,
        }

    # ========== Linear Regression Baseline Methods ==========

    def train_linear_baseline(self) -> dict:
        """Train linear regression baseline model for comparison with XGBoost.

        Uses Ridge regression with StandardScaler to handle the 100+ features
        with different scales. Uses the same train/test split and log transform
        as XGBoost for fair comparison.

        Returns:
            Dictionary with training metrics
        """
        from app.config import MODELS_DIR

        data_loader = get_data_loader()
        X, y = data_loader.get_feature_matrix()

        self.linear_predictor = LinearRegressionPredictor()
        metrics = self.linear_predictor.train(
            X, y,
            use_log_transform=True,  # Same as XGBoost
            use_cv=True,
        )

        # Save model
        linear_model_path = MODELS_DIR / "ktc_linear_baseline.joblib"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.linear_predictor.save(linear_model_path)

        self._linear_initialized = True
        print(f"Linear baseline model trained - MAE: {metrics['test_mae']:.4f}, R²: {metrics['test_r2']:.3f}")

        return metrics

    def initialize_linear_baseline(self, force_retrain: bool = False) -> dict:
        """Initialize linear baseline model - load from disk or train new."""
        from app.config import MODELS_DIR

        if self._linear_initialized and not force_retrain:
            return self.linear_predictor.metrics if self.linear_predictor else {}

        linear_model_path = MODELS_DIR / "ktc_linear_baseline.joblib"

        # Try to load existing model
        if linear_model_path.exists() and not force_retrain:
            try:
                self.linear_predictor = LinearRegressionPredictor()
                self.linear_predictor.load(linear_model_path)
                self._linear_initialized = True
                return self.linear_predictor.metrics
            except Exception as e:
                print(f"Failed to load linear model: {e}. Training new model...")

        return self.train_linear_baseline()

    def get_linear_metrics(self) -> dict:
        """Get linear baseline model performance metrics."""
        if not self._linear_initialized:
            self.initialize_linear_baseline()
        return self.linear_predictor.metrics if self.linear_predictor else {}

    def get_linear_feature_importance(self) -> dict:
        """Get feature importance (coefficients) from linear model."""
        if not self._linear_initialized:
            self.initialize_linear_baseline()
        if not self.linear_predictor:
            return {}
        return self.linear_predictor.get_feature_importance()

    def get_linear_coefficients(self) -> dict:
        """Get raw coefficients from linear model for interpretability."""
        if not self._linear_initialized:
            self.initialize_linear_baseline()
        if not self.linear_predictor:
            return {}
        return self.linear_predictor.get_coefficients()

    def compare_with_linear_baseline(self) -> dict:
        """Compare XGBoost vs linear regression baseline metrics.

        Returns comprehensive comparison including:
        - Individual model metrics
        - Improvement percentages
        - Feature counts
        - Model complexity comparison

        This helps determine if XGBoost's complexity is justified.
        """
        # Ensure both models are trained
        if not self._xgb_initialized:
            self.initialize_xgboost()
        if not self._linear_initialized:
            self.initialize_linear_baseline()

        xgb_metrics = self.xgb_predictor.metrics if self.xgb_predictor else {}
        linear_metrics = self.linear_predictor.metrics if self.linear_predictor else {}

        if not xgb_metrics or not linear_metrics:
            return {"error": "Both models must be trained for comparison"}

        # Calculate improvement percentages (negative means XGBoost is better)
        xgb_mae = xgb_metrics.get("test_mae", 0)
        linear_mae = linear_metrics.get("test_mae", 0)
        xgb_r2 = xgb_metrics.get("test_r2", 0)
        linear_r2 = linear_metrics.get("test_r2", 0)
        xgb_mape = xgb_metrics.get("test_mape", 0)
        linear_mape = linear_metrics.get("test_mape", 0)

        mae_reduction_pct = ((linear_mae - xgb_mae) / linear_mae * 100) if linear_mae > 0 else 0
        r2_improvement = xgb_r2 - linear_r2
        mape_reduction_pct = ((linear_mape - xgb_mape) / linear_mape * 100) if linear_mape > 0 else 0

        return {
            "linear_regression": {
                "test_mae": round(linear_mae, 4),
                "test_r2": round(linear_r2, 4),
                "test_mape": round(linear_mape, 2),
                "train_mae": round(linear_metrics.get("train_mae", 0), 4),
                "train_r2": round(linear_metrics.get("train_r2", 0), 4),
                "n_features": linear_metrics.get("n_features", len(self.linear_predictor.feature_columns)),
                "model_type": "ridge_regression",
            },
            "xgboost": {
                "test_mae": round(xgb_mae, 4),
                "test_r2": round(xgb_r2, 4),
                "test_mape": round(xgb_mape, 2),
                "train_mae": round(xgb_metrics.get("train_mae", 0), 4),
                "train_r2": round(xgb_metrics.get("train_r2", 0), 4),
                "n_features": len(self.xgb_predictor.feature_columns),
                "model_type": "xgboost_regressor",
            },
            "improvement": {
                "mae_reduction_pct": round(mae_reduction_pct, 2),
                "r2_improvement": round(r2_improvement, 4),
                "mape_reduction_pct": round(mape_reduction_pct, 2),
            },
            "interpretation": self._interpret_comparison(mae_reduction_pct, r2_improvement),
        }

    def _interpret_comparison(self, mae_reduction_pct: float, r2_improvement: float) -> str:
        """Generate human-readable interpretation of model comparison."""
        if mae_reduction_pct > 20:
            return (
                f"XGBoost provides significant value ({mae_reduction_pct:.1f}% MAE reduction). "
                "The non-linear patterns and feature interactions captured by tree-based "
                "learning justify the added complexity."
            )
        elif mae_reduction_pct > 10:
            return (
                f"XGBoost provides moderate improvement ({mae_reduction_pct:.1f}% MAE reduction). "
                "Worth keeping for production use, but the linear model could be a fallback."
            )
        elif mae_reduction_pct > 5:
            return (
                f"XGBoost provides marginal improvement ({mae_reduction_pct:.1f}% MAE reduction). "
                "Consider whether the added complexity is worth the small gain."
            )
        else:
            return (
                f"Linear regression performs nearly as well ({mae_reduction_pct:.1f}% MAE difference). "
                "Consider using the simpler model for better interpretability and faster inference."
            )

    # ========== Research Model: XGBoost Without Offseason Features ==========

    def train_xgboost_no_offseason(self) -> dict:
        """Train XGBoost excluding offseason_ktc_retention to test in-season predictors.

        The offseason_ktc_retention feature has 0.93 correlation with the target,
        which means it dominates predictions. This method trains a model without it
        to understand what the model can learn from in-season performance alone.

        This is useful for:
        1. Understanding how much the model relies on offseason vs in-season data
        2. Making mid-season predictions where offseason data isn't relevant
        3. Testing if XGBoost captures meaningful performance patterns

        Returns:
            Dictionary with training metrics and comparison to full model
        """
        if not XGBOOST_AVAILABLE:
            return {"error": "xgboost not installed. Run: pip install xgboost>=2.0.0"}

        data_loader = get_data_loader()
        X, y = data_loader.get_feature_matrix()

        # Remove offseason features
        offseason_features = [
            "offseason_ktc_retention",
            "offseason_trend",
            "offseason_volatility",
            "draft_impact",
            "training_camp_surge",
            "free_agency_impact",
            "offseason_max_drawdown",
            "offseason_recovery",
        ]

        # Filter to only features that exist in X
        features_to_remove = [f for f in offseason_features if f in X.columns]
        X_no_offseason = X.drop(columns=features_to_remove)

        print(f"Training XGBoost without {len(features_to_remove)} offseason features...")
        print(f"Features removed: {features_to_remove}")
        print(f"Feature count: {len(X.columns)} -> {len(X_no_offseason.columns)}")

        # Create a new predictor for this experiment (don't overwrite main model)
        no_offseason_predictor = XGBKTCPredictor()
        metrics = no_offseason_predictor.train(X_no_offseason, y, use_log_transform=True, use_cv=True)

        # Add context
        metrics["features_removed"] = features_to_remove
        metrics["feature_count"] = len(X_no_offseason.columns)
        metrics["experiment"] = "xgboost_no_offseason"

        print(f"XGBoost (no offseason) - MAE: {metrics['test_mae']:.4f}, R²: {metrics['test_r2']:.3f}")

        return metrics

    def compare_with_and_without_offseason(self) -> dict:
        """Compare XGBoost with/without the dominant offseason feature.

        This helps understand how much predictive power comes from:
        1. Offseason market sentiment (offseason_ktc_retention)
        2. In-season performance metrics

        Returns:
            Comprehensive comparison showing:
            - Full model metrics
            - No-offseason model metrics
            - Improvement breakdown
            - Interpretation
        """
        # Train full model if not already done
        if not self._xgb_initialized:
            self.initialize_xgboost()

        full_metrics = self.xgb_predictor.metrics if self.xgb_predictor else {}

        # Train no-offseason model
        no_offseason_metrics = self.train_xgboost_no_offseason()

        if "error" in no_offseason_metrics or not full_metrics:
            return {"error": "Could not complete comparison"}

        # Calculate differences
        full_mae = full_metrics.get("test_mae", 0)
        no_off_mae = no_offseason_metrics.get("test_mae", 0)
        full_r2 = full_metrics.get("test_r2", 0)
        no_off_r2 = no_offseason_metrics.get("test_r2", 0)

        mae_increase_pct = ((no_off_mae - full_mae) / full_mae * 100) if full_mae > 0 else 0
        r2_decrease = full_r2 - no_off_r2

        return {
            "full_model": {
                "test_mae": round(full_mae, 4),
                "test_r2": round(full_r2, 4),
                "n_features": len(self.xgb_predictor.feature_columns) if self.xgb_predictor else 0,
                "includes_offseason": True,
            },
            "no_offseason_model": {
                "test_mae": round(no_off_mae, 4),
                "test_r2": round(no_off_r2, 4),
                "n_features": no_offseason_metrics.get("feature_count", 0),
                "includes_offseason": False,
                "features_removed": no_offseason_metrics.get("features_removed", []),
            },
            "impact": {
                "mae_increase_pct": round(mae_increase_pct, 2),
                "r2_decrease": round(r2_decrease, 4),
                "offseason_contribution": f"{mae_increase_pct:.1f}% of predictive power comes from offseason features",
            },
            "interpretation": self._interpret_offseason_impact(mae_increase_pct, r2_decrease),
        }

    def _interpret_offseason_impact(self, mae_increase_pct: float, r2_decrease: float) -> str:
        """Generate interpretation of offseason feature impact."""
        if mae_increase_pct > 50:
            return (
                f"Offseason features are CRITICAL ({mae_increase_pct:.1f}% MAE increase without them). "
                "The model heavily relies on offseason market sentiment. In-season performance alone "
                "has limited predictive power for dynasty value."
            )
        elif mae_increase_pct > 25:
            return (
                f"Offseason features are important ({mae_increase_pct:.1f}% MAE increase without them). "
                "The model captures meaningful in-season patterns, but offseason sentiment provides "
                "significant additional value."
            )
        elif mae_increase_pct > 10:
            return (
                f"Offseason features provide moderate value ({mae_increase_pct:.1f}% MAE increase without them). "
                "In-season performance metrics are reasonably predictive on their own."
            )
        else:
            return (
                f"Offseason features provide minimal additional value ({mae_increase_pct:.1f}% MAE increase). "
                "In-season performance metrics are nearly as predictive. Consider simplifying the model."
            )


# Singleton instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
