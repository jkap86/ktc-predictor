import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


class KTCPredictor:
    """Gradient Boosting model for predicting KTC values."""

    def __init__(self):
        self.model: Optional[GradientBoostingRegressor] = None
        self.feature_columns: list[str] = []
        self.metrics: dict = {}

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict:
        """Train the model on the provided data."""
        self.feature_columns = list(X.columns)

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

        # Initialize and train model
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            loss="huber",  # Robust to outliers
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


class PositionEnsemblePredictor:
    """Trains separate models per position for better accuracy."""

    def __init__(self):
        self.models: dict[str, GradientBoostingRegressor] = {}
        self.feature_columns: dict[str, list[str]] = {}
        self.metrics: dict = {}
        self.position_configs = {
            "QB": {"max_depth": 6, "n_estimators": 200},
            "RB": {"max_depth": 4, "n_estimators": 250},  # More regularization
            "WR": {"max_depth": 6, "n_estimators": 200},
            "TE": {"max_depth": 4, "n_estimators": 200},
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
                learning_rate=0.05,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
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

            pos_metrics = {
                "train_mae": mean_absolute_error(y_train, train_pred),
                "test_mae": mean_absolute_error(y_test, test_pred),
                "train_r2": r2_score(y_train, train_pred),
                "test_r2": r2_score(y_test, test_pred),
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
            all_metrics[pos] = pos_metrics

            # Weighted contribution to combined metrics
            combined_train_mae += pos_metrics["train_mae"] * len(X_train)
            combined_test_mae += pos_metrics["test_mae"] * len(X_test)
            combined_train_r2 += pos_metrics["train_r2"] * len(X_train)
            combined_test_r2 += pos_metrics["test_r2"] * len(X_test)
            total_train += len(X_train)
            total_test += len(X_test)

        # Calculate weighted average metrics
        if total_train > 0:
            all_metrics["combined"] = {
                "train_mae": combined_train_mae / total_train,
                "test_mae": combined_test_mae / total_test,
                "train_r2": combined_train_r2 / total_train,
                "test_r2": combined_test_r2 / total_test,
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
    """

    def __init__(self):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is not installed. Install with: pip install xgboost"
            )
        self.model = None
        self.feature_columns: list[str] = []
        self.metrics: dict = {}

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict:
        """Train the XGBoost model on the provided data."""
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

        # Initialize XGBoost with regularization
        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            reg_alpha=1.0,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,  # Use all cores
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
