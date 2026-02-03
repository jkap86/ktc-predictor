from typing import Optional
from pathlib import Path

import pandas as pd

from app.config import MODEL_PATH, WEEKLY_MODEL_PATH, ENSEMBLE_MODEL_PATH, XGBOOST_MODEL_PATH
from app.models.predictor import (
    KTCPredictor,
    WeeklyKTCPredictor,
    PositionEnsemblePredictor,
    XGBKTCPredictor,
    XGBOOST_AVAILABLE,
)
from app.services.data_loader import (
    get_data_loader,
    calculate_derived_features,
    get_age_bracket,
)


class ModelService:
    """Service for managing the KTC prediction model."""

    def __init__(self):
        self.predictor = KTCPredictor()
        self.weekly_predictor = WeeklyKTCPredictor()
        self.ensemble_predictor: Optional[PositionEnsemblePredictor] = None
        self.xgb_predictor: Optional[XGBKTCPredictor] = None
        self._is_initialized = False
        self._weekly_initialized = False
        self._ensemble_initialized = False
        self._xgb_initialized = False

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

    def train_model(self) -> dict:
        """Train a new model from the training data."""
        data_loader = get_data_loader()
        X, y = data_loader.get_feature_matrix()

        metrics = self.predictor.train(X, y)

        # Save the trained model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.predictor.save(MODEL_PATH)

        self._is_initialized = True
        return metrics

    def predict_for_player(self, player_id: str) -> Optional[dict]:
        """Get prediction for a specific player."""
        if not self._is_initialized:
            self.initialize()

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
                prev_features = {
                    "current_ktc": prev_ktc,
                    "age": prev_age,
                    "years_exp": prev_season.get("years_exp", 0) + 1,
                    "fantasy_points": prev_season.get("fantasy_points", 0),
                    "games_played": prev_games,
                    "games_missed": 17 - prev_games,
                    "fp_vs_position_avg": 1.0,  # Approximate
                    **prev_derived,
                    f"pos_{position}": 1,
                    f"age_{get_age_bracket(prev_age, position)}": 1,
                    "prior_predicted_ktc": 0,
                    # Feature interactions
                    "age_x_is_rb": prev_age * (1 if position == "RB" else 0),
                    "age_x_is_qb": prev_age * (1 if position == "QB" else 0),
                    "games_x_consistency": prev_games * prev_derived.get("fp_consistency", 0),
                    "ktc_x_volatility": prev_ktc * prev_derived.get("ktc_in_season_volatility", 0) / 10000,
                    "ktc_x_trend": prev_ktc * prev_derived.get("ktc_season_trend", 0) / 10000,
                }
                prior_predicted_ktc = self.predictor.predict(prev_features)

        # Calculate derived features from the latest season's weekly data
        derived = calculate_derived_features(latest)

        # Calculate new features
        next_age = latest.get("age", 25) + 1
        games_played = latest.get("games_played", 0)
        age_bracket = get_age_bracket(next_age, position)

        # Prepare features for prediction (must match training feature names)
        features = {
            "current_ktc": current_ktc,
            "age": next_age,
            "years_exp": latest.get("years_exp", 0) + 1,
            "fantasy_points": latest.get("fantasy_points", 0),
            "games_played": games_played,
            "games_missed": 17 - games_played,
            "fp_vs_position_avg": 1.0,  # Approximate for single player prediction
            # Derived features from weekly data
            **derived,
            # Position one-hot encoding
            f"pos_{position}": 1,
            # Age bracket one-hot encoding
            f"age_{age_bracket}": 1,
            # Prior prediction feature
            "prior_predicted_ktc": prior_predicted_ktc,
            # Feature interactions
            "age_x_is_rb": next_age * (1 if position == "RB" else 0),
            "age_x_is_qb": next_age * (1 if position == "QB" else 0),
            "games_x_consistency": games_played * derived.get("fp_consistency", 0),
            "ktc_x_volatility": current_ktc * derived.get("ktc_in_season_volatility", 0) / 10000,
            "ktc_x_trend": current_ktc * derived.get("ktc_season_trend", 0) / 10000,
        }

        predicted_ktc = self.predictor.predict(features)
        ktc_change = predicted_ktc - current_ktc
        ktc_change_pct = (ktc_change / current_ktc * 100) if current_ktc > 0 else 0

        return {
            "player_id": player_id,
            "name": player["name"],
            "position": position,
            "current_ktc": current_ktc,
            "predicted_ktc": round(predicted_ktc, 2),
            "ktc_change": round(ktc_change, 2),
            "ktc_change_pct": round(ktc_change_pct, 2),
        }

    def get_metrics(self) -> dict:
        """Get model performance metrics."""
        return self.predictor.metrics

    def get_metrics_by_year(self) -> dict:
        """Calculate R² and MAE for each prediction target year."""
        if not self._is_initialized:
            self.initialize()

        from sklearn.metrics import mean_absolute_error, r2_score

        data_loader = get_data_loader()
        df = data_loader.get_training_dataframe()
        df = df.sort_values(["player_id", "year"])

        # Build year-over-year pairs with target year info
        yearly_data = {}  # year -> [(actual, predicted), ...]

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

                # Calculate prior_predicted_ktc for this pair
                position = current["position"]
                prior_predicted_ktc = 0.0
                if i >= 1:
                    prev = seasons[i - 1]
                    if prev["year"] == current["year"] - 1 and prev["end_ktc"] > 0:
                        prev_derived = calculate_derived_features(prev)
                        prev_ktc = prev["end_ktc"]
                        prev_age = prev["age"] + 1
                        prev_games = prev["games_played"]
                        prev_features = {
                            "current_ktc": prev_ktc,
                            "age": prev_age,
                            "years_exp": prev["years_exp"] + 1,
                            "fantasy_points": prev["fantasy_points"],
                            "games_played": prev_games,
                            "games_missed": 17 - prev_games,
                            "fp_vs_position_avg": 1.0,
                            **prev_derived,
                            f"pos_{position}": 1,
                            f"age_{get_age_bracket(prev_age, position)}": 1,
                            "prior_predicted_ktc": 0,
                            "age_x_is_rb": prev_age * (1 if position == "RB" else 0),
                            "age_x_is_qb": prev_age * (1 if position == "QB" else 0),
                            "games_x_consistency": prev_games * prev_derived.get("fp_consistency", 0),
                            "ktc_x_volatility": prev_ktc * prev_derived.get("ktc_in_season_volatility", 0) / 10000,
                            "ktc_x_trend": prev_ktc * prev_derived.get("ktc_season_trend", 0) / 10000,
                        }
                        prior_predicted_ktc = self.predictor.predict(prev_features)

                # Build features from current season
                derived = calculate_derived_features(current)
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
                    "prior_predicted_ktc": prior_predicted_ktc,
                    "age_x_is_rb": next_age * (1 if position == "RB" else 0),
                    "age_x_is_qb": next_age * (1 if position == "QB" else 0),
                    "games_x_consistency": games_played * derived.get("fp_consistency", 0),
                    "ktc_x_volatility": current_ktc * derived.get("ktc_in_season_volatility", 0) / 10000,
                    "ktc_x_trend": current_ktc * derived.get("ktc_season_trend", 0) / 10000,
                }

                predicted = self.predictor.predict(features)
                actual = next_s["end_ktc"]
                target_year = next_s["year"]

                if target_year not in yearly_data:
                    yearly_data[target_year] = []
                yearly_data[target_year].append((actual, predicted))

        # Calculate metrics for each year
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
                "mae": round(mae, 1),
                "n_samples": len(pairs),
            })

        return {
            "overall": self.get_metrics(),
            "by_year": by_year,
        }

    def get_feature_importance(self) -> dict:
        """Get feature importance from the model."""
        return self.predictor.get_feature_importance()

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

                predicted = self.predictor.predict(features)
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

                predicted_ktc = self.predictor.predict(features)
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

        predicted_ktc = self.ensemble_predictor.predict(features, position)
        ktc_change = predicted_ktc - current_ktc
        ktc_change_pct = (ktc_change / current_ktc * 100) if current_ktc > 0 else 0

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

    def train_xgboost(self) -> dict:
        """Train XGBoost model as alternative to GradientBoosting.

        XGBoost offers better regularization (L1/L2) which may help
        reduce overfitting on high-value players.
        """
        if not XGBOOST_AVAILABLE:
            return {"error": "xgboost not installed. Run: pip install xgboost>=2.0.0"}

        data_loader = get_data_loader()
        X, y = data_loader.get_feature_matrix()

        self.xgb_predictor = XGBKTCPredictor()
        metrics = self.xgb_predictor.train(X, y)

        # Save model
        XGBOOST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.xgb_predictor.save(XGBOOST_MODEL_PATH)

        self._xgb_initialized = True
        print(f"XGBoost model trained - MAE: {metrics['test_mae']:.1f}, R²: {metrics['test_r2']:.3f}")

        return metrics

    def initialize_xgboost(self, force_retrain: bool = False) -> dict:
        """Initialize XGBoost model - load from disk or train new."""
        if not XGBOOST_AVAILABLE:
            return {"error": "xgboost not installed"}

        if self._xgb_initialized and not force_retrain:
            return self.xgb_predictor.metrics if self.xgb_predictor else {}

        # Try to load existing model
        if XGBOOST_MODEL_PATH.exists() and not force_retrain:
            try:
                self.xgb_predictor = XGBKTCPredictor()
                self.xgb_predictor.load(XGBOOST_MODEL_PATH)
                self._xgb_initialized = True
                return self.xgb_predictor.metrics
            except Exception as e:
                print(f"Failed to load XGBoost model: {e}. Training new model...")

        return self.train_xgboost()

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
        """Compare GradientBoosting vs XGBoost performance."""
        if not self._is_initialized:
            self.initialize()
        if not self._xgb_initialized:
            self.initialize_xgboost()

        gb_metrics = self.get_metrics()
        xgb_metrics = self.get_xgb_metrics()

        comparison = {
            "gradient_boosting": gb_metrics,
            "xgboost": xgb_metrics,
        }

        if gb_metrics and xgb_metrics:
            gb_mae = gb_metrics.get("test_mae", 0)
            xgb_mae = xgb_metrics.get("test_mae", 0)
            if gb_mae > 0:
                improvement = (gb_mae - xgb_mae) / gb_mae * 100
                comparison["xgb_improvement_pct"] = round(improvement, 2)
                comparison["winner"] = "xgboost" if improvement > 0 else "gradient_boosting"

        return comparison

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


# Singleton instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
