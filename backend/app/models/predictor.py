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

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Initialize and train model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
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
