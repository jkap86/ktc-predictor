import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score

# Optuna availability check
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error.

    MAPE = mean(|y_true - y_pred| / y_true) * 100

    Better interpretability than MAE since errors are relative to actual value.
    A 500-point error matters more for a 1000-value player than 7000-value player.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Avoid division by zero
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


class KTCPredictor:
    """Gradient Boosting model for predicting KTC values."""

    def __init__(self):
        self.model: Optional[GradientBoostingRegressor] = None
        self.feature_columns: list[str] = []
        self.metrics: dict = {}

    def _create_model(self) -> GradientBoostingRegressor:
        """Create a new model instance with Optuna-optimized hyperparameters.

        These parameters were found via 50-trial Optuna optimization,
        achieving 25.7% MAE reduction vs previous defaults.
        """
        return GradientBoostingRegressor(
            n_estimators=400,           # Optuna: more trees for better learning
            max_depth=3,                # Optuna: shallower trees reduce overfitting
            learning_rate=0.019,        # Optuna: slower learning with more trees
            min_samples_split=10,       # Optuna: allow finer splits
            min_samples_leaf=14,        # Optuna: balanced leaf size
            subsample=0.816,            # Optuna: stochastic gradient boosting
            max_features=0.747,         # Optuna: feature bagging
            loss="huber",               # Robust to outliers
            random_state=42,
        )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        use_log_transform: bool = False,
        use_cv: bool = True,
    ) -> dict:
        """Train the model on the provided data.

        Args:
            X: Feature matrix
            y: Target variable (KTC values)
            test_size: Fraction for test set (default 0.2)
            use_log_transform: If True, apply log1p transform to target for better
                              handling of the wide KTC range (500-10,000+)
            use_cv: If True, run 5-fold cross-validation for more stable metrics
        """
        self.feature_columns = list(X.columns)
        self._use_log_transform = use_log_transform

        # Create KTC level bins for stratification to ensure high-value players
        # are proportionally represented in both train and test sets
        ktc_bins = pd.cut(
            X["current_ktc"],
            bins=[0, 2000, 5000, float("inf")],
            labels=["low", "mid", "high"]
        )

        # Split data with stratification by KTC level
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=ktc_bins
        )

        # Apply log transform if requested
        if use_log_transform:
            y_train_fit = np.log1p(y_train)
            y_test_fit = np.log1p(y_test)
        else:
            y_train_fit = y_train
            y_test_fit = y_test

        # Initialize and train model (Phase 2.3: Tuned hyperparameters)
        self.model = self._create_model()
        self.model.fit(X_train, y_train_fit)

        # Evaluate (predictions need inverse transform if log was used)
        train_pred_raw = self.model.predict(X_train)
        test_pred_raw = self.model.predict(X_test)

        if use_log_transform:
            train_pred = np.expm1(train_pred_raw)
            test_pred = np.expm1(test_pred_raw)
        else:
            train_pred = train_pred_raw
            test_pred = test_pred_raw

        # Ensure non-negative predictions
        train_pred = np.maximum(train_pred, 0)
        test_pred = np.maximum(test_pred, 0)

        self.metrics = {
            "train_mae": float(mean_absolute_error(y_train, train_pred)),
            "test_mae": float(mean_absolute_error(y_test, test_pred)),
            "train_r2": float(r2_score(y_train, train_pred)),
            "test_r2": float(r2_score(y_test, test_pred)),
            "train_mape": mape(y_train, train_pred),
            "test_mape": mape(y_test, test_pred),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "log_transform": use_log_transform,
        }

        # K-Fold Cross-Validation for more stable metrics
        if use_cv:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            y_for_cv = np.log1p(y) if use_log_transform else y

            cv_scores = cross_val_score(
                self._create_model(), X, y_for_cv, cv=kf, scoring="neg_mean_absolute_error"
            )

            # If log transform, the CV MAE is in log space - approximate back
            if use_log_transform:
                # Approximate: MAE in log space * median(y) gives rough original scale
                median_y = float(np.median(y))
                self.metrics["cv_mae"] = float(-cv_scores.mean() * median_y / np.log1p(median_y))
            else:
                self.metrics["cv_mae"] = float(-cv_scores.mean())
            self.metrics["cv_mae_std"] = float(cv_scores.std())

        return self.metrics

    def train_temporal(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        years: pd.Series,
        train_cutoff: int = 2023,
        use_log_transform: bool = False,
    ) -> dict:
        """Train with temporal split: older years for training, recent for testing.

        This provides a more realistic validation by testing on future data,
        which better simulates real-world prediction scenarios.

        Args:
            X: Feature matrix
            y: Target variable (next season KTC)
            years: Series containing the season year for each sample
            train_cutoff: Train on years <= cutoff, test on years > cutoff
            use_log_transform: If True, apply log1p transform to target

        Returns:
            Dictionary with train/test metrics
        """
        self.feature_columns = list(X.columns)
        self._use_log_transform = use_log_transform

        # Temporal split: train on older data, test on recent
        train_mask = years <= train_cutoff
        test_mask = years > train_cutoff

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(X_train) < 50 or len(X_test) < 20:
            raise ValueError(
                f"Insufficient data for temporal split. "
                f"Train: {len(X_train)}, Test: {len(X_test)}"
            )

        # Apply log transform if requested
        if use_log_transform:
            y_train_fit = np.log1p(y_train)
        else:
            y_train_fit = y_train

        # Train model with tuned hyperparameters
        self.model = self._create_model()
        self.model.fit(X_train, y_train_fit)

        # Evaluate (predictions need inverse transform if log was used)
        train_pred_raw = self.model.predict(X_train)
        test_pred_raw = self.model.predict(X_test)

        if use_log_transform:
            train_pred = np.expm1(train_pred_raw)
            test_pred = np.expm1(test_pred_raw)
        else:
            train_pred = train_pred_raw
            test_pred = test_pred_raw

        # Ensure non-negative predictions
        train_pred = np.maximum(train_pred, 0)
        test_pred = np.maximum(test_pred, 0)

        self.metrics = {
            "train_mae": float(mean_absolute_error(y_train, train_pred)),
            "test_mae": float(mean_absolute_error(y_test, test_pred)),
            "train_r2": float(r2_score(y_train, train_pred)),
            "test_r2": float(r2_score(y_test, test_pred)),
            "train_mape": mape(y_train, train_pred),
            "test_mape": mape(y_test, test_pred),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_years": f"<= {train_cutoff}",
            "test_years": f"> {train_cutoff}",
            "validation_type": "temporal",
            "log_transform": use_log_transform,
        }

        return self.metrics

    def predict(self, features: dict) -> float:
        """Predict KTC value for a single player."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create feature vector with correct column order
        X = pd.DataFrame([features])

        # Ensure all expected columns exist
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns]

        prediction = self.model.predict(X)[0]

        # Inverse log transform if model was trained with it
        if getattr(self, "_use_log_transform", False):
            prediction = np.expm1(prediction)

        return max(0, prediction)  # KTC can't be negative

    def predict_batch(self, features_list: list[dict]) -> list[float]:
        """Predict KTC values for multiple players."""
        return [self.predict(f) for f in features_list]

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}

        importance = dict(
            zip(self.feature_columns, self.model.feature_importances_)
        )
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Path) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")

        data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "use_log_transform": getattr(self, "_use_log_transform", False),
        }
        joblib.dump(data, path)

    def load(self, path: Path) -> None:
        """Load model from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.metrics = data.get("metrics", {})
        self._use_log_transform = data.get("use_log_transform", False)

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self.model is not None

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        use_log_transform: bool = True,
        n_cv_folds: int = 5,
    ) -> dict:
        """Optimize hyperparameters using Optuna with cross-validation.

        Args:
            X: Feature matrix
            y: Target variable
            n_trials: Number of Optuna trials (default 50)
            use_log_transform: Apply log transform to target (default True)
            n_cv_folds: Number of CV folds (default 5)

        Returns:
            Dictionary with best_params, best_score, study details
        """
        if not OPTUNA_AVAILABLE:
            return {"error": "optuna not installed. Run: pip install optuna>=3.0.0"}

        self.feature_columns = list(X.columns)
        self._use_log_transform = use_log_transform

        # Prepare target
        if use_log_transform:
            y_fit = np.log1p(y)
        else:
            y_fit = y

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "min_samples_split": trial.suggest_int("min_samples_split", 10, 50),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            }

            model = GradientBoostingRegressor(
                **params,
                loss="huber",
                random_state=42,
            )

            kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y_fit, cv=kf, scoring="neg_mean_absolute_error")

            return -cv_scores.mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Train final model with best params
        best_params = study.best_params
        self.model = GradientBoostingRegressor(
            **best_params,
            loss="huber",
            random_state=42,
        )
        self.model.fit(X, y_fit)

        # Evaluate with train/test split for metrics
        ktc_bins = pd.cut(
            X["current_ktc"],
            bins=[0, 2000, 5000, float("inf")],
            labels=["low", "mid", "high"]
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=ktc_bins
        )

        if use_log_transform:
            train_pred = np.expm1(self.model.predict(X_train))
            test_pred = np.expm1(self.model.predict(X_test))
        else:
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)

        train_pred = np.maximum(train_pred, 0)
        test_pred = np.maximum(test_pred, 0)

        self.metrics = {
            "train_mae": float(mean_absolute_error(y_train, train_pred)),
            "test_mae": float(mean_absolute_error(y_test, test_pred)),
            "train_r2": float(r2_score(y_train, train_pred)),
            "test_r2": float(r2_score(y_test, test_pred)),
            "train_mape": mape(y_train, train_pred),
            "test_mape": mape(y_test, test_pred),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "log_transform": use_log_transform,
            "optimized": True,
            "best_cv_mae": study.best_value,
            "n_trials": n_trials,
        }

        return {
            "best_params": best_params,
            "best_cv_mae": study.best_value,
            "metrics": self.metrics,
            "n_trials": n_trials,
        }


class PositionEnsemblePredictor:
    """Trains separate models per position for better accuracy."""

    def __init__(self):
        self.models: dict[str, GradientBoostingRegressor] = {}
        self.feature_columns: dict[str, list[str]] = {}
        self.metrics: dict = {}
        # Phase 2.3: Tuned hyperparameters per position
        self.position_configs = {
            "QB": {"max_depth": 5, "n_estimators": 300, "max_features": 0.8},
            "RB": {"max_depth": 4, "n_estimators": 350, "max_features": 0.7},  # More regularization for RB
            "WR": {"max_depth": 5, "n_estimators": 300, "max_features": 0.8},
            "TE": {"max_depth": 4, "n_estimators": 300, "max_features": 0.8},
        }

    def train(
        self, X: pd.DataFrame, y: pd.Series, positions: pd.Series, test_size: float = 0.2
    ) -> dict:
        """Train separate model for each position."""
        all_metrics = {}
        combined_train_mae = 0
        combined_test_mae = 0
        combined_train_r2 = 0
        combined_test_r2 = 0
        combined_train_mape = 0
        combined_test_mape = 0
        total_train = 0
        total_test = 0

        for pos in ["QB", "RB", "WR", "TE"]:
            mask = positions == pos
            X_pos, y_pos = X[mask], y[mask]

            if len(X_pos) < 50:  # Skip if too few samples
                continue

            config = self.position_configs[pos]
            model = GradientBoostingRegressor(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                learning_rate=0.03,       # Slower learning
                min_samples_split=30,     # More conservative
                min_samples_leaf=15,      # More conservative
                subsample=0.7,            # More stochastic
                max_features=config.get("max_features", 0.8),  # Feature bagging
                loss="huber",
                random_state=42,
            )

            # Stratify by KTC level within each position
            ktc_bins = pd.cut(
                X_pos["current_ktc"],
                bins=[0, 2000, 5000, float("inf")],
                labels=["low", "mid", "high"]
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X_pos, y_pos, test_size=test_size, random_state=42, stratify=ktc_bins
            )
            model.fit(X_train, y_train)

            self.models[pos] = model
            self.feature_columns[pos] = list(X.columns)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_mape_val = mape(y_train, train_pred)
            test_mape_val = mape(y_test, test_pred)

            pos_metrics = {
                "train_mae": float(mean_absolute_error(y_train, train_pred)),
                "test_mae": float(mean_absolute_error(y_test, test_pred)),
                "train_r2": float(r2_score(y_train, train_pred)),
                "test_r2": float(r2_score(y_test, test_pred)),
                "train_mape": train_mape_val,
                "test_mape": test_mape_val,
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
            all_metrics[pos] = pos_metrics

            # Weighted contribution to combined metrics
            combined_train_mae += pos_metrics["train_mae"] * len(X_train)
            combined_test_mae += pos_metrics["test_mae"] * len(X_test)
            combined_train_r2 += pos_metrics["train_r2"] * len(X_train)
            combined_test_r2 += pos_metrics["test_r2"] * len(X_test)
            combined_train_mape += train_mape_val * len(X_train)
            combined_test_mape += test_mape_val * len(X_test)
            total_train += len(X_train)
            total_test += len(X_test)

        # Calculate weighted average metrics
        if total_train > 0:
            all_metrics["combined"] = {
                "train_mae": combined_train_mae / total_train,
                "test_mae": combined_test_mae / total_test,
                "train_r2": combined_train_r2 / total_train,
                "test_r2": combined_test_r2 / total_test,
                "train_mape": combined_train_mape / total_train,
                "test_mape": combined_test_mape / total_test,
                "n_train": total_train,
                "n_test": total_test,
            }

        self.metrics = all_metrics
        return all_metrics

    def predict(self, features: dict, position: str) -> float:
        """Predict using position-specific model."""
        if position not in self.models:
            raise ValueError(f"No model for position: {position}")

        model = self.models[position]
        X = pd.DataFrame([features])

        for col in self.feature_columns[position]:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns[position]]
        prediction = model.predict(X)[0]
        return max(0, prediction)

    def get_feature_importance(self, position: Optional[str] = None) -> dict[str, float]:
        """Get feature importance scores for a position or averaged across all."""
        if position and position in self.models:
            importance = dict(
                zip(
                    self.feature_columns[position],
                    self.models[position].feature_importances_,
                )
            )
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        # Average across all positions
        if not self.models:
            return {}

        avg_importance: dict[str, float] = {}
        for pos, model in self.models.items():
            for feat, imp in zip(self.feature_columns[pos], model.feature_importances_):
                avg_importance[feat] = avg_importance.get(feat, 0) + imp

        # Divide by number of models
        n_models = len(self.models)
        for feat in avg_importance:
            avg_importance[feat] /= n_models

        return dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Path) -> None:
        """Save all models to disk."""
        if not self.models:
            raise ValueError("No models to save.")

        data = {
            "models": self.models,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "position_configs": self.position_configs,
        }
        joblib.dump(data, path)

    def load(self, path: Path) -> None:
        """Load models from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        self.models = data["models"]
        self.feature_columns = data["feature_columns"]
        self.metrics = data.get("metrics", {})
        self.position_configs = data.get("position_configs", self.position_configs)

    @property
    def is_trained(self) -> bool:
        """Check if models are trained."""
        return len(self.models) > 0


class WeeklyKTCPredictor:
    """Gradient Boosting model for predicting weekly KTC changes."""

    def __init__(self):
        self.model: Optional[GradientBoostingRegressor] = None
        self.feature_columns: list[str] = []
        self.metrics: dict = {}

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict:
        """Train the model on weekly KTC change data."""
        self.feature_columns = list(X.columns)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Use more estimators and deeper trees for the larger dataset
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        self.metrics = {
            "train_mae": mean_absolute_error(y_train, train_pred),
            "test_mae": mean_absolute_error(y_test, test_pred),
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        return self.metrics

    def predict(self, features: dict) -> float:
        """Predict weekly KTC change for given features."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create feature vector with correct column order
        X = pd.DataFrame([features])

        # Ensure all expected columns exist
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns]

        return self.model.predict(X)[0]

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}

        importance = dict(
            zip(self.feature_columns, self.model.feature_importances_)
        )
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Path) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")

        data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
        }
        joblib.dump(data, path)

    def load(self, path: Path) -> None:
        """Load model from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.metrics = data.get("metrics", {})

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self.model is not None


# XGBoost availability check
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBKTCPredictor:
    """XGBoost model for predicting KTC values.

    Uses XGBoost with L1/L2 regularization for potentially better
    generalization than GradientBoostingRegressor.

    Enhanced features (matching GB capabilities):
    - Log transform support for handling wide KTC range
    - Optuna hyperparameter optimization
    - Cross-validation support
    - Improved default hyperparameters
    """

    def __init__(self):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is not installed. Install with: pip install xgboost"
            )
        self.model = None
        self.feature_columns: list[str] = []
        self.metrics: dict = {}
        self._use_log_transform = False

    def _create_model(self) -> "XGBRegressor":
        """Create a new XGBoost model with Optuna-optimized hyperparameters.

        These parameters were found via 50-trial Optuna optimization,
        achieving ~30% MAE reduction vs original defaults.
        """
        return XGBRegressor(
            n_estimators=180,           # Optuna: fewer but more effective trees
            max_depth=8,                # Optuna: deeper trees capture complex patterns
            learning_rate=0.017,        # Optuna: slow learning for precision
            reg_alpha=0.014,            # Optuna: light L1 regularization
            reg_lambda=0.33,            # Optuna: moderate L2 regularization
            subsample=0.84,             # Optuna: balanced row subsampling
            colsample_bytree=0.74,      # Optuna: feature subsampling
            min_child_weight=15,        # Optuna: moderate leaf size requirement
            gamma=0.13,                 # Optuna: split threshold
            random_state=42,
            n_jobs=-1,
        )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        use_log_transform: bool = True,
        use_cv: bool = True,
        use_sample_weights: bool = False,
        sample_weight: Optional[np.ndarray] = None,
    ) -> dict:
        """Train the XGBoost model on the provided data.

        Args:
            X: Feature matrix
            y: Target variable (KTC values)
            test_size: Fraction for test set (default 0.2)
            use_log_transform: If True, apply log1p transform to target for better
                              handling of the wide KTC range (500-10,000+)
            use_cv: If True, run 5-fold cross-validation for more stable metrics
            use_sample_weights: If True, apply higher weights to under-represented
                               segments (low KTC + young players) during training
            sample_weight: Optional external sample weights array. If provided,
                          overrides use_sample_weights internal calculation.
        """
        self.feature_columns = list(X.columns)
        self._use_log_transform = use_log_transform
        self._use_sample_weights = use_sample_weights

        # Create KTC level bins for stratification
        ktc_bins = pd.cut(
            X["current_ktc"],
            bins=[0, 2000, 5000, float("inf")],
            labels=["low", "mid", "high"]
        )

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=ktc_bins
        )

        # Apply log transform if requested
        if use_log_transform:
            y_train_fit = np.log1p(y_train)
            y_test_fit = np.log1p(y_test)
        else:
            y_train_fit = y_train
            y_test_fit = y_test

        # Calculate sample weights
        # Use external weights if provided, otherwise calculate internal weights if requested
        train_sample_weight = None
        if sample_weight is not None:
            # External weights provided - need to split them same as data
            _, _, train_sample_weight, _ = train_test_split(
                X, sample_weight, test_size=test_size, random_state=42, stratify=ktc_bins
            )
        elif use_sample_weights:
            train_sample_weight = self._calculate_sample_weights(X_train)

        # Initialize and train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train_fit, sample_weight=train_sample_weight)

        # Evaluate (predictions need inverse transform if log was used)
        train_pred_raw = self.model.predict(X_train)
        test_pred_raw = self.model.predict(X_test)

        if use_log_transform:
            train_pred = np.expm1(train_pred_raw)
            test_pred = np.expm1(test_pred_raw)
        else:
            train_pred = train_pred_raw
            test_pred = test_pred_raw

        # Ensure non-negative predictions
        train_pred = np.maximum(train_pred, 0)
        test_pred = np.maximum(test_pred, 0)

        self.metrics = {
            "train_mae": float(mean_absolute_error(y_train, train_pred)),
            "test_mae": float(mean_absolute_error(y_test, test_pred)),
            "train_r2": float(r2_score(y_train, train_pred)),
            "test_r2": float(r2_score(y_test, test_pred)),
            "train_mape": mape(y_train, train_pred),
            "test_mape": mape(y_test, test_pred),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "log_transform": use_log_transform,
            "sample_weights": use_sample_weights,
        }

        # K-Fold Cross-Validation for more stable metrics
        if use_cv:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            y_for_cv = np.log1p(y) if use_log_transform else y

            cv_scores = cross_val_score(
                self._create_model(), X, y_for_cv, cv=kf, scoring="neg_mean_absolute_error"
            )

            # If log transform, the CV MAE is in log space - approximate back
            if use_log_transform:
                median_y = float(np.median(y))
                self.metrics["cv_mae"] = float(-cv_scores.mean() * median_y / np.log1p(median_y))
            else:
                self.metrics["cv_mae"] = float(-cv_scores.mean())
            self.metrics["cv_mae_std"] = float(cv_scores.std())

        return self.metrics

    def _calculate_sample_weights(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate sample weights to address systematic under-prediction.

        Applies higher weights to segments that XGBoost systematically under-predicts:
        - Low KTC players (<2000): bias of -0.74
        - Young players (<25): bias of -0.49

        The overlap (young + low KTC) gets the highest weight.

        Returns:
            numpy array of sample weights aligned with X
        """
        weights = np.ones(len(X))

        # Identify under-represented segments
        low_ktc = X["current_ktc"] < 2000
        young = X["age"] < 25

        # Apply weights:
        # - Low KTC alone: 1.5x weight
        # - Young alone: 1.3x weight
        # - Both low KTC AND young: 2.0x weight (highest priority)
        weights = np.where(low_ktc & young, 2.0, weights)
        weights = np.where(low_ktc & ~young, 1.5, weights)
        weights = np.where(~low_ktc & young, 1.3, weights)

        return weights

    def predict(self, features: dict) -> float:
        """Predict KTC value for a single player."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create feature vector with correct column order
        X = pd.DataFrame([features])

        # Ensure all expected columns exist
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns]

        prediction = self.model.predict(X)[0]

        # Inverse log transform if model was trained with it
        if self._use_log_transform:
            prediction = np.expm1(prediction)

        return max(0, prediction)  # KTC can't be negative

    def predict_batch(self, features_list: list[dict]) -> list[float]:
        """Predict KTC values for multiple players."""
        return [self.predict(f) for f in features_list]

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}

        importance = dict(
            zip(self.feature_columns, self.model.feature_importances_)
        )
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        use_log_transform: bool = True,
        n_cv_folds: int = 5,
    ) -> dict:
        """Optimize hyperparameters using Optuna with cross-validation.

        Args:
            X: Feature matrix
            y: Target variable
            n_trials: Number of Optuna trials (default 50)
            use_log_transform: Apply log transform to target (default True)
            n_cv_folds: Number of CV folds (default 5)

        Returns:
            Dictionary with best_params, best_score, study details
        """
        if not OPTUNA_AVAILABLE:
            return {"error": "optuna not installed. Run: pip install optuna>=3.0.0"}

        self.feature_columns = list(X.columns)
        self._use_log_transform = use_log_transform

        # Prepare target
        if use_log_transform:
            y_fit = np.log1p(y)
        else:
            y_fit = y

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            }

            model = XGBRegressor(
                **params,
                random_state=42,
                n_jobs=-1,
            )

            kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y_fit, cv=kf, scoring="neg_mean_absolute_error")

            return -cv_scores.mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Train final model with best params
        best_params = study.best_params
        self.model = XGBRegressor(
            **best_params,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y_fit)

        # Evaluate with train/test split for metrics
        ktc_bins = pd.cut(
            X["current_ktc"],
            bins=[0, 2000, 5000, float("inf")],
            labels=["low", "mid", "high"]
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=ktc_bins
        )

        if use_log_transform:
            train_pred = np.expm1(self.model.predict(X_train))
            test_pred = np.expm1(self.model.predict(X_test))
        else:
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)

        train_pred = np.maximum(train_pred, 0)
        test_pred = np.maximum(test_pred, 0)

        self.metrics = {
            "train_mae": float(mean_absolute_error(y_train, train_pred)),
            "test_mae": float(mean_absolute_error(y_test, test_pred)),
            "train_r2": float(r2_score(y_train, train_pred)),
            "test_r2": float(r2_score(y_test, test_pred)),
            "train_mape": mape(y_train, train_pred),
            "test_mape": mape(y_test, test_pred),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "log_transform": use_log_transform,
            "optimized": True,
            "best_cv_mae": study.best_value,
            "n_trials": n_trials,
        }

        return {
            "best_params": best_params,
            "best_cv_mae": study.best_value,
            "metrics": self.metrics,
            "n_trials": n_trials,
        }

    def save(self, path: Path) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")

        data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "use_log_transform": self._use_log_transform,
            "use_sample_weights": getattr(self, "_use_sample_weights", False),
        }
        joblib.dump(data, path)

    def load(self, path: Path) -> None:
        """Load model from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.metrics = data.get("metrics", {})
        self._use_log_transform = data.get("use_log_transform", False)
        self._use_sample_weights = data.get("use_sample_weights", False)

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self.model is not None


# LightGBM availability check (Phase 3.1)
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class LGBMKTCPredictor:
    """LightGBM model for predicting KTC values.

    LightGBM offers several advantages for this dataset:
    - Better performance on smaller datasets (~1,400 samples)
    - Native categorical feature support
    - Built-in regularization
    - Faster training for hyperparameter tuning
    """

    def __init__(self):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "lightgbm is not installed. Install with: pip install lightgbm>=4.0.0"
            )
        self.model = None
        self.feature_columns: list[str] = []
        self.metrics: dict = {}

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict:
        """Train the LightGBM model on the provided data."""
        self.feature_columns = list(X.columns)

        # Create KTC level bins for stratification
        ktc_bins = pd.cut(
            X["current_ktc"],
            bins=[0, 2000, 5000, float("inf")],
            labels=["low", "mid", "high"]
        )

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=ktc_bins
        )

        # Initialize LightGBM with regularization
        self.model = LGBMRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_alpha=0.1,   # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            random_state=42,
            n_jobs=-1,
            verbose=-1,      # Suppress output
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        self.metrics = {
            "train_mae": mean_absolute_error(y_train, train_pred),
            "test_mae": mean_absolute_error(y_test, test_pred),
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        return self.metrics

    def predict(self, features: dict) -> float:
        """Predict KTC value for a single player."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create feature vector with correct column order
        X = pd.DataFrame([features])

        # Ensure all expected columns exist
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns]

        prediction = self.model.predict(X)[0]
        return max(0, prediction)  # KTC can't be negative

    def predict_batch(self, features_list: list[dict]) -> list[float]:
        """Predict KTC values for multiple players."""
        return [self.predict(f) for f in features_list]

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}

        importance = dict(
            zip(self.feature_columns, self.model.feature_importances_)
        )
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Path) -> None:
        """Save LightGBM model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")

        data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
        }
        joblib.dump(data, path)

    def load(self, path: Path) -> None:
        """Load LightGBM model from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.metrics = data.get("metrics", {})

    @property
    def is_trained(self) -> bool:
        """Check if LightGBM model is trained."""
        return self.model is not None


# CatBoost availability check
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class CatBoostKTCPredictor:
    """CatBoost model for predicting KTC values.

    CatBoost offers several advantages for this dataset:
    - Native categorical feature handling (position, age brackets)
    - Built-in missing value support
    - Ordered boosting to reduce prediction shift
    - Often outperforms XGBoost/LightGBM on tabular data with categoricals
    """

    def __init__(self):
        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "catboost is not installed. Install with: pip install catboost>=1.2.0"
            )
        self.model = None
        self.feature_columns: list[str] = []
        self.cat_features: list[str] = []
        self.metrics: dict = {}

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict:
        """Train the CatBoost model on the provided data."""
        self.feature_columns = list(X.columns)

        # Identify categorical features (position, age bracket columns)
        self.cat_features = [col for col in X.columns
                            if col.startswith('pos_') or col.startswith('age_')]

        # Create KTC level bins for stratification
        ktc_bins = pd.cut(
            X["current_ktc"],
            bins=[0, 2000, 5000, float("inf")],
            labels=["low", "mid", "high"]
        )

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=ktc_bins
        )

        # Initialize CatBoost with regularization
        self.model = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=3.0,  # L2 regularization
            random_seed=42,
            verbose=False,
            loss_function='RMSE',
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        self.metrics = {
            "train_mae": mean_absolute_error(y_train, train_pred),
            "test_mae": mean_absolute_error(y_test, test_pred),
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        return self.metrics

    def predict(self, features: dict) -> float:
        """Predict KTC value for a single player."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create feature vector with correct column order
        X = pd.DataFrame([features])

        # Ensure all expected columns exist
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns]

        prediction = self.model.predict(X)[0]
        return max(0, prediction)  # KTC can't be negative

    def predict_batch(self, features_list: list[dict]) -> list[float]:
        """Predict KTC values for multiple players."""
        return [self.predict(f) for f in features_list]

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}

        importance = dict(
            zip(self.feature_columns, self.model.get_feature_importance())
        )
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Path) -> None:
        """Save CatBoost model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")

        data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "cat_features": self.cat_features,
            "metrics": self.metrics,
        }
        joblib.dump(data, path)

    def load(self, path: Path) -> None:
        """Load CatBoost model from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.cat_features = data.get("cat_features", [])
        self.metrics = data.get("metrics", {})

    @property
    def is_trained(self) -> bool:
        """Check if CatBoost model is trained."""
        return self.model is not None


class BreakoutAwarePredictor(KTCPredictor):
    """Predictor with adjustments to better capture breakout potential.

    This class extends KTCPredictor with post-hoc adjustments that
    boost predictions for young players showing breakout signals.
    The model systematically underestimates young risers, so we apply
    a targeted boost when momentum and upside signals are present.

    Updated parameters (tuned on ratio-based model):
    - Boost factor: 1.29 (improves detection from 23% to 67%)
    - Momentum threshold: 0.30 (more sensitive to late-season surges)
    """

    def __init__(self):
        super().__init__()
        # Boost factor for young players with high momentum and upside
        # Tuned via grid search to balance riser detection vs decliner accuracy
        self.young_boost_factor = 1.29  # 29% boost
        # Thresholds for triggering the boost
        self.momentum_threshold = 0.30  # momentum_score > 0.30
        self.upside_threshold = 0.40    # ktc_upside_ratio > 0.40
        self.max_age_for_boost = 25     # Only apply to players 25 or younger

    def predict(self, features: dict) -> float:
        """Predict KTC value with breakout adjustment.

        Applies a boost for young players showing breakout signals:
        - Age <= 25
        - Momentum score > 0.30 (late-season surge)
        - Upside ratio > 0.40 (room to grow)
        """
        base_pred = super().predict(features)

        # Check if this player qualifies for breakout boost
        age = features.get("age", 30)
        if age <= self.max_age_for_boost:
            momentum = features.get("momentum_score", 0)
            upside = features.get("ktc_upside_ratio", 0)
            undervalued = features.get("undervalued_young", 0)

            # Apply boost if momentum and upside signals are strong
            if momentum > self.momentum_threshold and upside > self.upside_threshold:
                base_pred *= self.young_boost_factor
            # Additional boost for undervalued young players
            elif undervalued == 1 and upside > 0.4:
                base_pred *= (1 + (self.young_boost_factor - 1) / 2)  # Half boost

        return max(0, base_pred)

    def save(self, path: Path) -> None:
        """Save model with breakout parameters."""
        if self.model is None:
            raise ValueError("No model to save.")

        data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "young_boost_factor": self.young_boost_factor,
            "momentum_threshold": self.momentum_threshold,
            "upside_threshold": self.upside_threshold,
            "max_age_for_boost": self.max_age_for_boost,
        }
        joblib.dump(data, path)

    def load(self, path: Path) -> None:
        """Load model with breakout parameters."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.metrics = data.get("metrics", {})
        self.young_boost_factor = data.get("young_boost_factor", 1.12)
        self.momentum_threshold = data.get("momentum_threshold", 0.8)
        self.upside_threshold = data.get("upside_threshold", 0.4)
        self.max_age_for_boost = data.get("max_age_for_boost", 25)


class CalibratedPredictor(KTCPredictor):
    """Post-hoc calibration for KTC tiers.

    Addresses systematic under-prediction across all tiers by applying
    tier-based multipliers after base prediction.

    Updated calibration factors (tuned on ratio-based model):
    - Elite: 1.12 (reduces bias from -669 to ~-24)
    - High: 1.12
    - Mid: 1.15
    - Low: 1.19
    """

    calibration_factors = {
        "elite": 1.12,   # >5000 KTC
        "high": 1.12,    # 3000-5000 KTC
        "mid": 1.15,     # 2000-3000 KTC
        "low": 1.19,     # <2000 KTC
    }

    def _get_tier(self, current_ktc: float) -> str:
        """Determine KTC tier for calibration."""
        if current_ktc > 5000:
            return "elite"
        elif current_ktc > 3000:
            return "high"
        elif current_ktc > 2000:
            return "mid"
        else:
            return "low"

    def predict(self, features: dict) -> float:
        """Predict KTC value with tier-based calibration."""
        base_pred = super().predict(features)
        tier = self._get_tier(features.get("current_ktc", 0))
        calibrated = base_pred * self.calibration_factors[tier]
        return max(0, calibrated)

    def save(self, path: Path) -> None:
        """Save model with calibration factors."""
        if self.model is None:
            raise ValueError("No model to save.")

        data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "use_log_transform": getattr(self, "_use_log_transform", False),
            "calibration_factors": self.calibration_factors,
        }
        joblib.dump(data, path)

    def load(self, path: Path) -> None:
        """Load model with calibration factors."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.metrics = data.get("metrics", {})
        self._use_log_transform = data.get("use_log_transform", False)
        self.calibration_factors = data.get("calibration_factors", self.calibration_factors)


class CalibratedBreakoutPredictor(KTCPredictor):
    """Combined predictor with tier calibration and breakout detection.

    Intelligently applies both calibration and breakout boost without
    over-correcting:
    - Elite/high tier: Only calibration (already high value, less breakout potential)
    - Mid/low tier young players with signals: Breakout boost + reduced calibration
    - Mid/low tier without breakout signals: Full calibration

    This achieves:
    - Elite tier bias: <±500 (target met)
    - Breakout detection: >50% (target exceeded)
    """

    # Calibration factors (tuned on ratio-based model)
    calibration_factors = {
        "elite": 1.12,   # >5000 KTC
        "high": 1.12,    # 3000-5000 KTC
        "mid": 1.15,     # 2000-3000 KTC
        "low": 1.19,     # <2000 KTC
    }

    # Breakout parameters (tuned for 67%+ detection rate)
    young_boost_factor = 1.29
    momentum_threshold = 0.30
    upside_threshold = 0.40
    max_age_for_boost = 25

    def _get_tier(self, current_ktc: float) -> str:
        """Determine KTC tier for calibration."""
        if current_ktc > 5000:
            return "elite"
        elif current_ktc > 3000:
            return "high"
        elif current_ktc > 2000:
            return "mid"
        else:
            return "low"

    def _should_apply_breakout_boost(self, features: dict, tier: str) -> bool:
        """Check if breakout boost should be applied."""
        # Don't apply breakout boost to elite/high tier (already high value)
        if tier in ("elite", "high"):
            return False

        age = features.get("age", 30)
        if age > self.max_age_for_boost:
            return False

        momentum = features.get("momentum_score", 0)
        upside = features.get("ktc_upside_ratio", 0)

        return momentum > self.momentum_threshold and upside > self.upside_threshold

    def predict(self, features: dict) -> float:
        """Predict KTC value with combined calibration and breakout adjustment."""
        base_pred = super().predict(features)

        current_ktc = features.get("current_ktc", 0)
        tier = self._get_tier(current_ktc)

        if self._should_apply_breakout_boost(features, tier):
            # Apply breakout boost with reduced calibration to avoid over-correction
            # Use sqrt of calibration factor when also applying breakout boost
            reduced_cal = 1 + (self.calibration_factors[tier] - 1) * 0.5
            base_pred *= self.young_boost_factor * reduced_cal
        else:
            # Apply full calibration
            base_pred *= self.calibration_factors[tier]

        return max(0, base_pred)

    def save(self, path: Path) -> None:
        """Save model with combined parameters."""
        if self.model is None:
            raise ValueError("No model to save.")

        data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "use_log_transform": getattr(self, "_use_log_transform", False),
            "calibration_factors": self.calibration_factors,
            "young_boost_factor": self.young_boost_factor,
            "momentum_threshold": self.momentum_threshold,
            "upside_threshold": self.upside_threshold,
            "max_age_for_boost": self.max_age_for_boost,
        }
        joblib.dump(data, path)

    def load(self, path: Path) -> None:
        """Load model with combined parameters."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.metrics = data.get("metrics", {})
        self._use_log_transform = data.get("use_log_transform", False)
        if "calibration_factors" in data:
            self.calibration_factors = data["calibration_factors"]
        if "young_boost_factor" in data:
            self.young_boost_factor = data["young_boost_factor"]
        if "momentum_threshold" in data:
            self.momentum_threshold = data["momentum_threshold"]
        if "upside_threshold" in data:
            self.upside_threshold = data["upside_threshold"]
        if "max_age_for_boost" in data:
            self.max_age_for_boost = data["max_age_for_boost"]


class XGBCalibratedBreakoutPredictor(XGBKTCPredictor):
    """XGBoost with tier calibration, breakout detection, and bust detection.

    Combines the accuracy of XGBoost with calibration adjustments based on
    error analysis. Addresses systematic biases through:
    1. Position-specific tier calibration (reduces over-prediction for elite/high)
    2. Breakout boost for young players with momentum signals
    3. Bust detection for aging/declining players

    Analysis findings (Feb 2026):
    - Elite tier was over-predicting by +1,372 KTC
    - High tier was over-predicting by +1,047 KTC
    - WR_elite and RB_high had highest biases
    - Bust detection was 0% (now addressed)
    """

    # Base calibration factors by KTC tier (tuned based on bias analysis)
    # Iteration 4: Further reduced mid-tier TE/QB, reduced low tier
    calibration_factors = {
        "elite": 0.92,   # Bias at +333 (acceptable)
        "high": 0.95,    # Bias at +317 (acceptable)
        "mid": 0.88,     # Was +417 bias → reduce from 0.90
        "low": 0.94,     # Was +350 bias → reduce from 0.98
    }

    # Position-specific calibration adjustments (multiplied with tier factor)
    # Iteration 8: Balance QB calibration (was -490 at 0.90, +556 at 1.02)
    position_calibration = {
        # Elite tier
        ("elite", "WR"): 0.88,
        ("elite", "QB"): 0.96,    # Split difference between 0.90 and 1.02
        ("elite", "RB"): 0.92,
        ("elite", "TE"): 0.92,
        # High tier
        ("high", "RB"): 0.91,
        ("high", "WR"): 0.89,
        ("high", "QB"): 0.95,
        ("high", "TE"): 0.93,
        # Mid tier
        ("mid", "TE"): 0.78,
        ("mid", "QB"): 0.85,
        ("mid", "WR"): 0.86,
        ("mid", "RB"): 0.90,
        # Low tier
        ("low", "TE"): 1.00,
        ("low", "WR"): 0.88,
        ("low", "RB"): 0.90,
        ("low", "QB"): 0.94,
    }

    # Age bracket adjustments - position specific
    # TE_young over-predicting by +373, RB_prime by +322
    age_calibration = {
        "young": 0.96,          # TE_young high bias
        "prime": 0.92,          # RB_prime still over-predicting
        "declining": 0.92,
    }

    # Breakout parameters (tuned for high detection rate)
    young_boost_factor = 1.25  # Reduced from 1.29 to avoid over-prediction
    momentum_threshold = 0.30
    upside_threshold = 0.40
    max_age_for_boost = 25

    # Bust detection parameters
    aging_rb_threshold = 27      # RBs 27+ are at risk
    aging_other_threshold = 30   # Other positions age slower
    bust_factor_aging_rb = 0.88  # 12% reduction for aging RBs
    bust_factor_elite_regression = 0.94  # Elite players regress to mean
    bust_factor_declining_efficiency = 0.92  # Declining FP/game

    def _get_tier(self, current_ktc: float) -> str:
        """Determine KTC tier for calibration."""
        if current_ktc > 5000:
            return "elite"
        elif current_ktc > 3000:
            return "high"
        elif current_ktc > 2000:
            return "mid"
        else:
            return "low"

    def _get_position(self, features: dict) -> str:
        """Extract position from feature flags."""
        if features.get("pos_QB", 0) == 1:
            return "QB"
        elif features.get("pos_WR", 0) == 1:
            return "WR"
        elif features.get("pos_TE", 0) == 1:
            return "TE"
        else:
            return "RB"

    def _get_age_bracket(self, age: int, position: str) -> str:
        """Determine age bracket based on position-specific thresholds."""
        if position == "RB":
            if age <= 24:
                return "young"
            elif age <= 26:
                return "prime"
            else:
                return "declining"
        else:  # QB, WR, TE
            if age <= 25:
                return "young"
            elif age <= 29:
                return "prime"
            else:
                return "declining"

    def _get_calibration_factor(self, tier: str, position: str, age: int) -> float:
        """Get combined calibration factor for tier, position, and age."""
        base_factor = self.calibration_factors.get(tier, 1.0)
        position_adj = self.position_calibration.get((tier, position), 1.0)
        age_bracket = self._get_age_bracket(age, position)
        age_adj = self.age_calibration.get(age_bracket, 1.0)
        return base_factor * position_adj * age_adj

    def _should_apply_breakout_boost(self, features: dict, tier: str) -> bool:
        """Check if breakout boost should be applied.

        Only applies to mid/low tier young players with strong momentum
        and upside signals. Elite/high tier players are already high value
        with less breakout potential.
        """
        # Don't apply breakout boost to elite/high tier
        if tier in ("elite", "high"):
            return False

        age = features.get("age", 30)
        if age > self.max_age_for_boost:
            return False

        momentum = features.get("momentum_score", 0)
        upside = features.get("ktc_upside_ratio", 0)

        # Original trigger: momentum + upside
        original_trigger = momentum > self.momentum_threshold and upside > self.upside_threshold

        # YoY performance breakout for young players with upside
        young_performance_breakout = features.get("young_performance_breakout", 0)
        yoy_trigger = young_performance_breakout == 1 and upside > 0.3

        return original_trigger or yoy_trigger

    def _should_apply_bust_penalty(self, features: dict, tier: str, position: str) -> tuple[bool, float]:
        """Check if bust penalty should be applied and return the factor.

        Returns (should_apply, bust_factor) tuple.

        Bust triggers:
        1. Aging RBs (27+) - historically steep decline
        2. Elite players with low momentum - regression to mean
        3. Declining efficiency (FP/game dropping YoY)
        """
        age = features.get("age", 25)
        momentum = features.get("momentum_score", 0)
        fp_yoy_ratio = features.get("fp_yoy_ratio_capped", 1.0)
        games_played = features.get("games_played", 10)

        bust_factors = []

        # Trigger 1: Aging RBs
        if position == "RB" and age >= self.aging_rb_threshold:
            # More aggressive penalty for older RBs
            age_penalty = self.bust_factor_aging_rb - (age - self.aging_rb_threshold) * 0.02
            bust_factors.append(max(0.80, age_penalty))

        # Trigger 2: Elite regression to mean (elite players with negative momentum)
        if tier == "elite" and momentum < 0:
            bust_factors.append(self.bust_factor_elite_regression)

        # Trigger 3: Declining efficiency (YoY FP decline + played enough games)
        if fp_yoy_ratio < 0.75 and games_played >= 8:
            # Significant decline in production
            bust_factors.append(self.bust_factor_declining_efficiency)

        # Trigger 4: Aging non-RB veterans with declining stats
        if position != "RB" and age >= self.aging_other_threshold and fp_yoy_ratio < 0.85:
            bust_factors.append(0.94)

        if bust_factors:
            # Apply the most aggressive bust factor
            return True, min(bust_factors)

        return False, 1.0

    # Performance floor thresholds by position (for severe KTC-performance disconnects)
    position_floors = {"QB": 4000, "RB": 3500, "WR": 3000, "TE": 2500}

    def predict(self, features: dict) -> float:
        """Predict KTC value with calibration, breakout, and bust adjustments.

        Pipeline:
        1. Get base XGBoost prediction
        2. Apply position-specific tier + age calibration
        3. Apply breakout boost OR bust penalty (mutually exclusive)
        4. Apply performance floor for severe KTC-performance disconnects
        """
        # Get base prediction from XGBoost parent class
        base_pred = super().predict(features)

        current_ktc = features.get("current_ktc", 0)
        age = features.get("age", 25)
        tier = self._get_tier(current_ktc)
        position = self._get_position(features)

        # Get position-specific calibration factor (includes age adjustment)
        calibration = self._get_calibration_factor(tier, position, age)

        # Check for breakout or bust conditions
        apply_breakout = self._should_apply_breakout_boost(features, tier)
        apply_bust, bust_factor = self._should_apply_bust_penalty(features, tier, position)

        if apply_breakout:
            # Apply breakout boost with reduced calibration (don't apply age penalty to breakouts)
            base_calibration = self.calibration_factors.get(tier, 1.0) * self.position_calibration.get((tier, position), 1.0)
            reduced_cal = 1 + (base_calibration - 1) * 0.5
            base_pred *= self.young_boost_factor * reduced_cal
        elif apply_bust:
            # Apply bust penalty with calibration
            base_pred *= calibration * bust_factor
        else:
            # Apply standard calibration
            base_pred *= calibration

        # Apply performance-based floor for severe KTC-performance disconnects
        base_pred = self._apply_performance_floor(base_pred, features, current_ktc)

        return max(0, base_pred)

    def _apply_performance_floor(
        self, prediction: float, features: dict, current_ktc: float
    ) -> float:
        """Apply performance-based floor for severe KTC-performance disconnects.

        For young players with 50%+ FP improvement but KTC < 2500, apply a weighted
        floor based on position-typical minimums. This prevents the model from
        anchoring too heavily on collapsed baselines.

        Note: prediction is a RATIO (e.g., 1.2 means 120% of current_ktc).
        We convert the absolute floor to a ratio before combining.
        """
        # Only apply to low KTC players
        if current_ktc >= 2500 or current_ktc <= 0:
            return prediction

        # Check for significant YoY improvement
        fp_yoy_ratio = features.get("fp_yoy_ratio_capped", 1.0)
        if fp_yoy_ratio < 1.50:
            return prediction

        # Get position for floor lookup
        position = None
        if features.get("pos_QB", 0) == 1:
            position = "QB"
        elif features.get("pos_WR", 0) == 1:
            position = "WR"
        elif features.get("pos_TE", 0) == 1:
            position = "TE"
        else:
            position = "RB"  # Default to RB if no position flag set

        floor_absolute = self.position_floors.get(position, 3000)

        # Convert floor to ratio (what multiplier of current_ktc equals the floor)
        floor_ratio = floor_absolute / current_ktc

        # Apply weighted floor: 70% floor_ratio + 30% original prediction
        # This prevents over-correction while establishing a reasonable minimum
        floored_prediction = 0.7 * floor_ratio + 0.3 * prediction

        # Only use floor if it would increase the prediction
        return max(prediction, floored_prediction)

    def save(self, path: Path) -> None:
        """Save model with calibration, breakout, and bust parameters."""
        if self.model is None:
            raise ValueError("No model to save.")

        # Convert position_calibration keys from tuples to strings for JSON compatibility
        position_cal_serializable = {
            f"{tier}_{pos}": factor
            for (tier, pos), factor in self.position_calibration.items()
        }

        data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "use_log_transform": self._use_log_transform,
            # Calibration
            "calibration_factors": self.calibration_factors,
            "position_calibration": position_cal_serializable,
            "age_calibration": self.age_calibration,
            # Breakout
            "young_boost_factor": self.young_boost_factor,
            "momentum_threshold": self.momentum_threshold,
            "upside_threshold": self.upside_threshold,
            "max_age_for_boost": self.max_age_for_boost,
            # Bust detection
            "aging_rb_threshold": self.aging_rb_threshold,
            "aging_other_threshold": self.aging_other_threshold,
            "bust_factor_aging_rb": self.bust_factor_aging_rb,
            "bust_factor_elite_regression": self.bust_factor_elite_regression,
            "bust_factor_declining_efficiency": self.bust_factor_declining_efficiency,
            # Floors
            "position_floors": self.position_floors,
        }
        joblib.dump(data, path)

    def load(self, path: Path) -> None:
        """Load model with calibration, breakout, and bust parameters."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.metrics = data.get("metrics", {})
        self._use_log_transform = data.get("use_log_transform", False)

        # Load calibration parameters (use defaults if not present)
        if "calibration_factors" in data:
            self.calibration_factors = data["calibration_factors"]
        if "position_calibration" in data:
            # Convert back from string keys to tuple keys
            self.position_calibration = {
                (key.split("_")[0], key.split("_")[1]): factor
                for key, factor in data["position_calibration"].items()
            }
        if "age_calibration" in data:
            self.age_calibration = data["age_calibration"]
        # Breakout parameters
        if "young_boost_factor" in data:
            self.young_boost_factor = data["young_boost_factor"]
        if "momentum_threshold" in data:
            self.momentum_threshold = data["momentum_threshold"]
        if "upside_threshold" in data:
            self.upside_threshold = data["upside_threshold"]
        if "max_age_for_boost" in data:
            self.max_age_for_boost = data["max_age_for_boost"]
        # Bust parameters
        if "aging_rb_threshold" in data:
            self.aging_rb_threshold = data["aging_rb_threshold"]
        if "aging_other_threshold" in data:
            self.aging_other_threshold = data["aging_other_threshold"]
        if "bust_factor_aging_rb" in data:
            self.bust_factor_aging_rb = data["bust_factor_aging_rb"]
        if "bust_factor_elite_regression" in data:
            self.bust_factor_elite_regression = data["bust_factor_elite_regression"]
        if "bust_factor_declining_efficiency" in data:
            self.bust_factor_declining_efficiency = data["bust_factor_declining_efficiency"]
        # Floors
        if "position_floors" in data:
            self.position_floors = data["position_floors"]


class AbsoluteXGBPredictor(XGBKTCPredictor):
    """XGBoost model for predicting absolute KTC values (not ratios).

    This model is trained on raw end_ktc values rather than ratios,
    which avoids the compression bias that affects low-KTC players
    when using ratio-based predictions.

    Key differences from base XGBKTCPredictor:
    - Target is absolute end_ktc (0-12000+ range)
    - Deeper trees (max_depth=10) to capture wider target range
    - More regularization to prevent overfitting to outliers
    - No log transform on target (absolute values already normalized by current_ktc feature)
    """

    def _create_model(self) -> "XGBRegressor":
        """Create XGBoost model optimized for absolute KTC prediction.

        Hyperparameters tuned for wider target range (0-12000+).
        """
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=250,           # More trees for complex absolute patterns
            max_depth=10,               # Deeper trees for wider target range
            learning_rate=0.015,        # Slower learning for stability
            reg_alpha=0.5,              # Strong L1 regularization
            reg_lambda=1.0,             # Strong L2 regularization
            subsample=0.8,              # Row subsampling
            colsample_bytree=0.7,       # Feature subsampling
            min_child_weight=20,        # Larger leaf size for smoother predictions
            gamma=0.2,                  # Higher split threshold
            random_state=42,
            n_jobs=-1,
        )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        use_cv: bool = True,
    ) -> dict:
        """Train the absolute XGBoost model.

        Note: Does NOT use log transform since absolute KTC values
        are already reasonably distributed when normalized by current_ktc feature.
        """
        # Call parent train with no log transform
        return super().train(
            X, y,
            test_size=test_size,
            use_log_transform=False,  # Absolute values don't need log transform
            use_cv=use_cv,
            use_sample_weights=False,
        )


class HybridEnsemblePredictor:
    """Hybrid ensemble combining ratio-based and absolute-value models.

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
    """

    # Tier thresholds and weights
    TIER_CONFIG = {
        "low": {"max_ktc": 2000, "absolute_weight": 0.70, "ratio_weight": 0.30},
        "mid": {"max_ktc": 5000, "absolute_weight": 0.50, "ratio_weight": 0.50},
        "high": {"max_ktc": float("inf"), "absolute_weight": 0.30, "ratio_weight": 0.70},
    }

    def __init__(self):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is not installed. Install with: pip install xgboost"
            )
        self.ratio_model: Optional[XGBKTCPredictor] = None
        self.absolute_model: Optional[AbsoluteXGBPredictor] = None
        self.feature_columns: list[str] = []
        self.metrics: dict = {}
        self._is_trained = False

    def _get_tier(self, current_ktc: float) -> str:
        """Determine KTC tier for weight lookup."""
        if current_ktc < self.TIER_CONFIG["low"]["max_ktc"]:
            return "low"
        elif current_ktc < self.TIER_CONFIG["mid"]["max_ktc"]:
            return "mid"
        else:
            return "high"

    def _get_weights(self, current_ktc: float) -> tuple[float, float]:
        """Get (absolute_weight, ratio_weight) for a given current_ktc."""
        tier = self._get_tier(current_ktc)
        config = self.TIER_CONFIG[tier]
        return config["absolute_weight"], config["ratio_weight"]

    def train(
        self,
        X: pd.DataFrame,
        y_ratio: pd.Series,
        y_absolute: pd.Series,
        test_size: float = 0.2,
    ) -> dict:
        """Train both ratio and absolute models.

        Args:
            X: Feature matrix
            y_ratio: Target as ratio (next_ktc / current_ktc)
            y_absolute: Target as absolute value (next_ktc)
            test_size: Fraction for test set

        Returns:
            Dictionary with combined and individual model metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score

        self.feature_columns = list(X.columns)

        # Create KTC level bins for stratification
        ktc_bins = pd.cut(
            X["current_ktc"],
            bins=[0, 2000, 5000, float("inf")],
            labels=["low", "mid", "high"]
        )

        # Split data with stratification
        X_train, X_test, y_ratio_train, y_ratio_test, y_abs_train, y_abs_test = train_test_split(
            X, y_ratio, y_absolute,
            test_size=test_size,
            random_state=42,
            stratify=ktc_bins
        )

        # Train ratio model
        print("Training ratio model (XGBoost)...")
        self.ratio_model = XGBKTCPredictor()
        ratio_metrics = self.ratio_model.train(
            X, y_ratio,
            test_size=test_size,
            use_log_transform=True,
            use_cv=True,
        )
        print(f"  Ratio model MAE: {ratio_metrics['test_mae']:.4f}")

        # Train absolute model
        print("Training absolute model (XGBoost)...")
        self.absolute_model = AbsoluteXGBPredictor()
        absolute_metrics = self.absolute_model.train(X, y_absolute, test_size=test_size)
        print(f"  Absolute model MAE: {absolute_metrics['test_mae']:.1f}")

        # Calculate hybrid ensemble metrics on test set
        print("Calculating hybrid ensemble metrics...")
        test_predictions = []
        test_actuals = []

        for idx in X_test.index:
            features = X_test.loc[idx].to_dict()
            current_ktc = features["current_ktc"]

            # Get hybrid prediction (returns absolute value)
            pred_absolute = self._predict_hybrid(features, current_ktc)
            test_predictions.append(pred_absolute)
            test_actuals.append(y_abs_test.loc[idx])

        test_predictions = np.array(test_predictions)
        test_actuals = np.array(test_actuals)

        # Calculate ensemble metrics
        ensemble_mae = float(mean_absolute_error(test_actuals, test_predictions))
        ensemble_r2 = float(r2_score(test_actuals, test_predictions))
        ensemble_mape = mape(test_actuals, test_predictions)

        # Calculate train metrics
        train_predictions = []
        train_actuals = []
        for idx in X_train.index:
            features = X_train.loc[idx].to_dict()
            current_ktc = features["current_ktc"]
            pred_absolute = self._predict_hybrid(features, current_ktc)
            train_predictions.append(pred_absolute)
            train_actuals.append(y_abs_train.loc[idx])

        train_predictions = np.array(train_predictions)
        train_actuals = np.array(train_actuals)

        train_mae = float(mean_absolute_error(train_actuals, train_predictions))
        train_r2 = float(r2_score(train_actuals, train_predictions))
        train_mape = mape(train_actuals, train_predictions)

        # Tier-specific metrics
        tier_metrics = {}
        for tier in ["low", "mid", "high"]:
            tier_mask = [self._get_tier(X_test.loc[idx]["current_ktc"]) == tier for idx in X_test.index]
            if sum(tier_mask) > 0:
                tier_preds = test_predictions[tier_mask]
                tier_actual = test_actuals[tier_mask]
                tier_metrics[tier] = {
                    "mae": float(mean_absolute_error(tier_actual, tier_preds)),
                    "bias": float(np.mean(tier_preds - tier_actual)),
                    "n_samples": sum(tier_mask),
                }

        self.metrics = {
            "train_mae": train_mae,
            "test_mae": ensemble_mae,
            "train_r2": train_r2,
            "test_r2": ensemble_r2,
            "train_mape": train_mape,
            "test_mape": ensemble_mape,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "ratio_model_metrics": ratio_metrics,
            "absolute_model_metrics": absolute_metrics,
            "tier_metrics": tier_metrics,
            "tier_config": self.TIER_CONFIG,
        }

        self._is_trained = True
        print(f"Hybrid ensemble trained - MAE: {ensemble_mae:.1f}, R²: {ensemble_r2:.3f}")

        return self.metrics

    def _predict_hybrid(self, features: dict, current_ktc: float) -> float:
        """Internal method for hybrid prediction returning absolute KTC."""
        # Get weights for this KTC tier
        abs_weight, ratio_weight = self._get_weights(current_ktc)

        # Get ratio prediction and convert to absolute
        ratio_pred = self.ratio_model.predict(features)
        ratio_absolute = ratio_pred * current_ktc

        # Get absolute prediction
        absolute_pred = self.absolute_model.predict(features)

        # Weighted blend
        hybrid_pred = abs_weight * absolute_pred + ratio_weight * ratio_absolute

        return max(0, hybrid_pred)

    def predict(self, features: dict) -> float:
        """Predict absolute KTC value using adaptive weighted blend.

        Args:
            features: Dictionary of feature values (must include 'current_ktc')

        Returns:
            Predicted absolute KTC value
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Fill missing feature columns with 0
        for col in self.feature_columns:
            if col not in features:
                features[col] = 0

        current_ktc = features.get("current_ktc", 0)
        return self._predict_hybrid(features, current_ktc)

    def predict_with_details(self, features: dict) -> dict:
        """Predict with detailed breakdown of model contributions.

        Returns prediction along with:
        - ratio_prediction: What ratio model predicted (as absolute)
        - absolute_prediction: What absolute model predicted
        - tier: The player's KTC tier
        - weights: The weights used for blending
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Fill missing feature columns with 0
        for col in self.feature_columns:
            if col not in features:
                features[col] = 0

        current_ktc = features.get("current_ktc", 0)
        tier = self._get_tier(current_ktc)
        abs_weight, ratio_weight = self._get_weights(current_ktc)

        # Get individual predictions
        ratio_pred = self.ratio_model.predict(features)
        ratio_absolute = ratio_pred * current_ktc
        absolute_pred = self.absolute_model.predict(features)

        # Weighted blend
        hybrid_pred = abs_weight * absolute_pred + ratio_weight * ratio_absolute

        return {
            "predicted_ktc": max(0, hybrid_pred),
            "ratio_prediction": ratio_absolute,
            "absolute_prediction": absolute_pred,
            "tier": tier,
            "absolute_weight": abs_weight,
            "ratio_weight": ratio_weight,
        }

    def predict_batch(self, features_list: list[dict]) -> list[float]:
        """Predict KTC values for multiple players."""
        return [self.predict(f) for f in features_list]

    def get_feature_importance(self) -> dict[str, float]:
        """Get averaged feature importance from both models."""
        if not self._is_trained:
            return {}

        # Average importance from both models
        ratio_importance = self.ratio_model.get_feature_importance()
        absolute_importance = self.absolute_model.get_feature_importance()

        combined = {}
        all_features = set(ratio_importance.keys()) | set(absolute_importance.keys())

        for feat in all_features:
            ratio_imp = ratio_importance.get(feat, 0)
            abs_imp = absolute_importance.get(feat, 0)
            combined[feat] = (ratio_imp + abs_imp) / 2

        return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Path) -> None:
        """Save hybrid ensemble to disk."""
        if not self._is_trained:
            raise ValueError("No models to save.")

        data = {
            "ratio_model": self.ratio_model,
            "absolute_model": self.absolute_model,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "tier_config": self.TIER_CONFIG,
        }
        joblib.dump(data, path)

    def load(self, path: Path) -> None:
        """Load hybrid ensemble from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        self.ratio_model = data["ratio_model"]
        self.absolute_model = data["absolute_model"]
        self.feature_columns = data["feature_columns"]
        self.metrics = data.get("metrics", {})
        if "tier_config" in data:
            self.TIER_CONFIG = data["tier_config"]
        self._is_trained = True

    @property
    def is_trained(self) -> bool:
        """Check if hybrid ensemble is trained."""
        return self._is_trained


class VotingEnsemblePredictor:
    """Ensemble that averages predictions from multiple model types.

    Combines GradientBoosting, XGBoost, LightGBM, and CatBoost to reduce
    prediction variance. Equal-weighted averaging typically yields 10-15%
    MAE reduction through variance reduction.

    Gracefully handles missing dependencies - uses only available models.
    """

    def __init__(self):
        self.models: list = []
        self.model_names: list[str] = []
        self.weights: list[float] = []
        self.feature_columns: list[str] = []
        self.metrics: dict = {}
        self._use_log_transform = False

    def _get_available_models(self) -> list[tuple[str, object]]:
        """Get list of available model classes based on installed packages."""
        available = [("gradient_boosting", KTCPredictor)]

        if XGBOOST_AVAILABLE:
            available.append(("xgboost", XGBKTCPredictor))
        if LIGHTGBM_AVAILABLE:
            available.append(("lightgbm", LGBMKTCPredictor))
        if CATBOOST_AVAILABLE:
            available.append(("catboost", CatBoostKTCPredictor))

        return available

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        use_log_transform: bool = True,
        optimize_weights: bool = False,
    ) -> dict:
        """Train all available models in the ensemble.

        Args:
            X: Feature matrix
            y: Target variable (KTC values or ratios)
            test_size: Fraction for test set
            use_log_transform: Apply log transform to target (for GB model)
            optimize_weights: If True, optimize weights via CV (slower)
        """
        self.feature_columns = list(X.columns)
        self._use_log_transform = use_log_transform

        # Create KTC level bins for stratification
        ktc_bins = pd.cut(
            X["current_ktc"],
            bins=[0, 2000, 5000, float("inf")],
            labels=["low", "mid", "high"]
        )

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=ktc_bins
        )

        # Train each available model
        available_models = self._get_available_models()
        self.models = []
        self.model_names = []
        model_metrics = {}

        print(f"Training {len(available_models)} models in ensemble...")

        for name, model_class in available_models:
            print(f"  Training {name}...")
            try:
                model = model_class()

                if name == "gradient_boosting" and use_log_transform:
                    # GB model supports log transform natively
                    model.train(X, y, test_size=test_size, use_log_transform=True)
                else:
                    model.train(X, y, test_size=test_size)

                self.models.append(model)
                self.model_names.append(name)
                model_metrics[name] = model.metrics

                print(f"    {name}: MAE={model.metrics.get('test_mae', 'N/A'):.4f}")
            except Exception as e:
                print(f"    Failed to train {name}: {e}")

        if not self.models:
            raise ValueError("No models could be trained for ensemble")

        # Set weights based on individual model performance (inverse MAE weighting)
        # Filter out models with very poor performance (MAE > 2x best MAE)
        maes = [model_metrics[name].get("test_mae", float("inf")) for name in self.model_names]
        best_mae = min(maes)

        # Calculate inverse-MAE weights for models within 2x of best
        raw_weights = []
        for i, mae in enumerate(maes):
            if mae <= best_mae * 2:  # Only include models within 2x of best
                raw_weights.append(1.0 / mae)
            else:
                raw_weights.append(0.0)
                print(f"    Excluding {self.model_names[i]} from ensemble (MAE {mae:.4f} > 2x best {best_mae:.4f})")

        # Normalize weights to sum to 1
        total_weight = sum(raw_weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in raw_weights]
        else:
            # Fallback to equal weights if all models are poor
            n_models = len(self.models)
            self.weights = [1.0 / n_models] * n_models

        print(f"  Model weights: {dict(zip(self.model_names, [f'{w:.3f}' for w in self.weights]))}")

        # Optionally optimize weights via cross-validation (more expensive)
        n_models = len(self.models)
        if optimize_weights and n_models > 1:
            self._optimize_weights(X_train, y_train)

        # Calculate ensemble metrics on test set
        test_preds = self._predict_batch_internal(X_test)

        # Count active models (weight > 0)
        n_active_models = sum(1 for w in self.weights if w > 0)

        self.metrics = {
            "train_mae": float(mean_absolute_error(y_train, self._predict_batch_internal(X_train))),
            "test_mae": float(mean_absolute_error(y_test, test_preds)),
            "train_r2": float(r2_score(y_train, self._predict_batch_internal(X_train))),
            "test_r2": float(r2_score(y_test, test_preds)),
            "train_mape": mape(y_train, self._predict_batch_internal(X_train)),
            "test_mape": mape(y_test, test_preds),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_models": n_models,
            "n_active_models": n_active_models,
            "model_names": self.model_names,
            "weights": self.weights,
            "individual_metrics": model_metrics,
            "log_transform": use_log_transform,
        }

        print(f"Ensemble trained: MAE={self.metrics['test_mae']:.4f}, R²={self.metrics['test_r2']:.4f}")

        return self.metrics

    def _optimize_weights(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> None:
        """Optimize ensemble weights using cross-validation.

        Uses grid search over weight combinations to find the
        combination that minimizes CV MAE.
        """
        from itertools import product

        print("  Optimizing ensemble weights...")

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        n_models = len(self.models)

        # Generate weight combinations (coarse grid)
        weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        best_weights = self.weights.copy()
        best_mae = float("inf")

        # Generate valid weight combinations (sum to 1.0)
        for combo in product(weight_options, repeat=n_models):
            if abs(sum(combo) - 1.0) > 0.01:
                continue
            if all(w == 0 for w in combo):
                continue

            # Evaluate this weight combination via CV
            cv_maes = []
            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Get predictions from each model
                preds = np.zeros(len(X_val))
                for model, weight in zip(self.models, combo):
                    if weight > 0:
                        model_preds = np.array([model.predict(row.to_dict()) for _, row in X_val.iterrows()])
                        preds += weight * model_preds

                cv_maes.append(mean_absolute_error(y_val, preds))

            avg_mae = np.mean(cv_maes)
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_weights = list(combo)

        self.weights = best_weights
        print(f"    Optimized weights: {dict(zip(self.model_names, self.weights))}")

    def _predict_batch_internal(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble predictions for a DataFrame (internal use)."""
        preds = np.zeros(len(X))

        for model, weight in zip(self.models, self.weights):
            if weight > 0:
                model_preds = np.array([model.predict(row.to_dict()) for _, row in X.iterrows()])
                preds += weight * model_preds

        return np.maximum(preds, 0)

    def predict(self, features: dict) -> float:
        """Predict using weighted average of all models."""
        if not self.models:
            raise ValueError("Ensemble not trained. Call train() first.")

        # Fill missing feature columns with 0
        for col in self.feature_columns:
            if col not in features:
                features[col] = 0

        prediction = 0.0
        for model, weight in zip(self.models, self.weights):
            if weight > 0:
                prediction += weight * model.predict(features)

        return max(0, prediction)

    def predict_batch(self, features_list: list[dict]) -> list[float]:
        """Predict for multiple players."""
        return [self.predict(f) for f in features_list]

    def get_feature_importance(self) -> dict[str, float]:
        """Get averaged feature importance across all models."""
        if not self.models:
            return {}

        avg_importance: dict[str, float] = {}
        total_weight = sum(self.weights)

        for model, weight in zip(self.models, self.weights):
            if weight > 0:
                model_importance = model.get_feature_importance()
                for feat, imp in model_importance.items():
                    if feat not in avg_importance:
                        avg_importance[feat] = 0
                    avg_importance[feat] += (weight / total_weight) * imp

        return dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Path) -> None:
        """Save ensemble to disk."""
        if not self.models:
            raise ValueError("No models to save.")

        data = {
            "models": self.models,
            "model_names": self.model_names,
            "weights": self.weights,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "use_log_transform": self._use_log_transform,
        }
        joblib.dump(data, path)

    def load(self, path: Path) -> None:
        """Load ensemble from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Ensemble file not found: {path}")

        data = joblib.load(path)
        self.models = data["models"]
        self.model_names = data["model_names"]
        self.weights = data["weights"]
        self.feature_columns = data["feature_columns"]
        self.metrics = data.get("metrics", {})
        self._use_log_transform = data.get("use_log_transform", False)

    @property
    def is_trained(self) -> bool:
        """Check if ensemble is trained."""
        return len(self.models) > 0


class LinearRegressionPredictor:
    """Linear regression baseline model for KTC prediction.

    This serves as a benchmark to measure the value XGBoost provides over
    a simple linear model. Uses Ridge regression (slight L2 regularization)
    to handle multicollinearity from the 100+ correlated features.

    Key design decisions:
    - StandardScaler: Critical since features have vastly different scales
      (KTC: 500-10000, age: 20-35, etc.)
    - Ridge regression: Handles multicollinearity better than plain OLS
    - Same train/test split as XGBoost for fair comparison
    - Same log transform on target for consistency
    """

    def __init__(self):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        self.model = Ridge(alpha=1.0)  # Light regularization
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []
        self.metrics: dict = {}
        self._use_log_transform = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        use_log_transform: bool = True,
        use_cv: bool = True,
    ) -> dict:
        """Train the linear regression model.

        Args:
            X: Feature matrix
            y: Target variable (KTC ratio values)
            test_size: Fraction for test set (default 0.2)
            use_log_transform: If True, apply log1p transform to target
            use_cv: If True, run 5-fold cross-validation
        """
        self.feature_columns = list(X.columns)
        self._use_log_transform = use_log_transform

        # Create KTC level bins for stratification (same as XGBoost)
        ktc_bins = pd.cut(
            X["current_ktc"],
            bins=[0, 2000, 5000, float("inf")],
            labels=["low", "mid", "high"]
        )

        # Split data with stratification (same as XGBoost)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=ktc_bins
        )

        # Scale features (critical for linear regression)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Apply log transform if requested
        if use_log_transform:
            y_train_fit = np.log1p(y_train)
            y_test_fit = np.log1p(y_test)
        else:
            y_train_fit = y_train
            y_test_fit = y_test

        # Train model
        self.model.fit(X_train_scaled, y_train_fit)

        # Evaluate (predictions need inverse transform if log was used)
        train_pred_raw = self.model.predict(X_train_scaled)
        test_pred_raw = self.model.predict(X_test_scaled)

        if use_log_transform:
            train_pred = np.expm1(train_pred_raw)
            test_pred = np.expm1(test_pred_raw)
        else:
            train_pred = train_pred_raw
            test_pred = test_pred_raw

        # Ensure non-negative predictions
        train_pred = np.maximum(train_pred, 0)
        test_pred = np.maximum(test_pred, 0)

        self.metrics = {
            "train_mae": float(mean_absolute_error(y_train, train_pred)),
            "test_mae": float(mean_absolute_error(y_test, test_pred)),
            "train_r2": float(r2_score(y_train, train_pred)),
            "test_r2": float(r2_score(y_test, test_pred)),
            "train_mape": mape(y_train, train_pred),
            "test_mape": mape(y_test, test_pred),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "log_transform": use_log_transform,
            "n_features": len(self.feature_columns),
            "model_type": "ridge_regression",
        }

        # K-Fold Cross-Validation for more stable metrics
        if use_cv:
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            # Create pipeline for proper scaling during CV
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=1.0))
            ])

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            y_for_cv = np.log1p(y) if use_log_transform else y

            cv_scores = cross_val_score(
                pipeline, X, y_for_cv, cv=kf, scoring="neg_mean_absolute_error"
            )

            # If log transform, the CV MAE is in log space - approximate back
            if use_log_transform:
                median_y = float(np.median(y))
                self.metrics["cv_mae"] = float(-cv_scores.mean() * median_y / np.log1p(median_y))
            else:
                self.metrics["cv_mae"] = float(-cv_scores.mean())
            self.metrics["cv_mae_std"] = float(cv_scores.std())

        return self.metrics

    def predict(self, features: dict) -> float:
        """Predict KTC ratio for a single player."""
        if not hasattr(self.model, 'coef_') or self.model.coef_ is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create feature vector with correct column order
        X = pd.DataFrame([features])

        # Ensure all expected columns exist
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns]

        # Scale features
        X_scaled = self.scaler.transform(X)

        prediction = self.model.predict(X_scaled)[0]

        # Inverse log transform if model was trained with it
        if self._use_log_transform:
            prediction = np.expm1(prediction)

        return max(0, prediction)

    def predict_batch(self, features_list: list[dict]) -> list[float]:
        """Predict KTC ratios for multiple players."""
        return [self.predict(f) for f in features_list]

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature coefficients as importance scores.

        For linear regression, the absolute value of standardized coefficients
        indicates feature importance (how much a 1-std change in feature
        affects the prediction).
        """
        if not hasattr(self.model, 'coef_') or self.model.coef_ is None:
            return {}

        # Use absolute coefficients as importance
        importance = dict(
            zip(self.feature_columns, np.abs(self.model.coef_))
        )
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def get_coefficients(self) -> dict[str, float]:
        """Get raw coefficients (including sign) for interpretability.

        Positive coefficient = feature increases predicted KTC ratio
        Negative coefficient = feature decreases predicted KTC ratio
        """
        if not hasattr(self.model, 'coef_') or self.model.coef_ is None:
            return {}

        coefficients = dict(
            zip(self.feature_columns, self.model.coef_)
        )
        return dict(sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True))

    def save(self, path: Path) -> None:
        """Save model to disk."""
        if not hasattr(self.model, 'coef_') or self.model.coef_ is None:
            raise ValueError("No model to save.")

        data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "use_log_transform": self._use_log_transform,
        }
        joblib.dump(data, path)

    def load(self, path: Path) -> None:
        """Load model from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_columns = data["feature_columns"]
        self.metrics = data.get("metrics", {})
        self._use_log_transform = data.get("use_log_transform", False)

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return hasattr(self.model, 'coef_') and self.model.coef_ is not None
