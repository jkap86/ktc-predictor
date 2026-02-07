"""Monotone linear calibrator to replace isotonic pos_cal."""

import numpy as np
from scipy.optimize import minimize


class MonotoneLinearCalibrator:
    """Linear calibrator with monotonicity constraint (slope >= 0).

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
        X = X[mask]
        y = y[mask]

        if len(X) == 0:
            self.slope = 1.0
            self.intercept = 0.0
            return self

        def loss(params):
            slope, intercept = params
            pred = slope * X + intercept
            return np.sum((pred - y) ** 2)

        result = minimize(
            loss,
            [1.0, 0.0],
            bounds=[(self.min_slope, None), (None, None)],
            method="L-BFGS-B",
        )
        self.slope, self.intercept = result.x
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
