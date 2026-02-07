"""Monotone linear calibrator (no SciPy dependency)."""

import numpy as np


class MonotoneLinearCalibrator:
    """Linear calibrator with monotonicity constraint (slope >= min_slope).

    Uses closed-form OLS with slope clamping to avoid SciPy dependency.
    Unlike IsotonicRegression, this calibrator never collapses an interval
    to a constant value, preserving sensitivity across the input range.

    Parameters
    ----------
    min_slope : float, default=0.01
        Minimum allowed slope to ensure strict monotonicity.
        Set to 0 to allow flat calibration.
    """

    def __init__(self, min_slope: float = 0.01):
        self.slope: float | None = None
        self.intercept: float | None = None
        self.min_slope = min_slope

    def fit(self, X, y):
        """Fit monotone linear calibrator.

        Parameters
        ----------
        X : array-like
            Input predictions to calibrate.
        y : array-like
            Target values (actual log_ratio).

        Returns
        -------
        self
        """
        X = np.asarray(X).ravel()
        y = np.asarray(y).ravel()

        # Filter out NaN values
        mask = ~(np.isnan(X) | np.isnan(y))
        X, y = X[mask], y[mask]

        if len(X) == 0:
            self.slope = 1.0
            self.intercept = 0.0
            return self

        # Closed-form OLS: slope = cov(X,y) / var(X), intercept = mean(y) - slope*mean(X)
        x_mean, y_mean = X.mean(), y.mean()
        x_var = ((X - x_mean) ** 2).sum()

        if x_var < 1e-12:  # degenerate case (all X values identical)
            self.slope = 1.0
            self.intercept = y_mean
            return self

        slope = ((X - x_mean) * (y - y_mean)).sum() / x_var

        # Clamp slope to min_slope for monotonicity
        slope = max(slope, self.min_slope)

        # Recompute intercept given clamped slope
        intercept = y_mean - slope * x_mean

        self.slope = slope
        self.intercept = intercept
        return self

    def predict(self, X):
        """Apply calibration to input predictions.

        Parameters
        ----------
        X : array-like
            Input predictions to calibrate.

        Returns
        -------
        np.ndarray
            Calibrated predictions.
        """
        if self.slope is None or self.intercept is None:
            raise ValueError("Calibrator has not been fitted yet.")

        X = np.asarray(X).ravel()
        return self.slope * X + self.intercept

    def __repr__(self):
        if self.slope is None:
            return "MonotoneLinearCalibrator(not fitted)"
        return f"MonotoneLinearCalibrator(slope={self.slope:.4f}, intercept={self.intercept:.4f})"
