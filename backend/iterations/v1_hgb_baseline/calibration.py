"""Monotone linear calibrator (no SciPy dependency)."""

import numpy as np


class MonotoneLinearCalibrator:
    """Linear calibrator with monotonicity constraint and identity shrinkage.

    Uses closed-form OLS with slope clamping to avoid SciPy dependency.
    Unlike IsotonicRegression, this calibrator never collapses an interval
    to a constant value, preserving sensitivity across the input range.

    Identity shrinkage regularizes the calibrator toward the identity function
    (slope=1, intercept=0), preventing overly aggressive corrections when
    training data is noisy or narrow.

    Parameters
    ----------
    min_slope : float, default=0.0
        Minimum allowed slope to ensure monotonicity.
        Set to 0 to allow near-identity calibration.
    identity_strength : float, default=10.0
        Shrinkage strength toward identity function.
        Higher values = stronger preference for slope=1, intercept=0.
        Set to 0 to disable shrinkage (pure OLS).
    """

    def __init__(self, min_slope: float = 0.0, identity_strength: float = 10.0):
        self.slope: float | None = None
        self.intercept: float | None = None
        self.min_slope = min_slope
        self.identity_strength = identity_strength

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

        raw_slope = ((X - x_mean) * (y - y_mean)).sum() / x_var

        # Apply identity shrinkage: regularize toward slope=1, intercept=0
        lam = self.identity_strength
        if lam > 0:
            # Shrink slope toward 1.0 (identity)
            slope = (raw_slope + lam * 1.0) / (1.0 + lam)

            # Compute raw intercept, then shrink toward 0.0
            raw_intercept = y_mean - slope * x_mean
            intercept = raw_intercept / (1.0 + lam)
        else:
            # No shrinkage: pure OLS
            slope = raw_slope
            intercept = y_mean - slope * x_mean

        # Clamp slope to min_slope for monotonicity
        slope = max(slope, self.min_slope)

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
        return (
            f"MonotoneLinearCalibrator(slope={self.slope:.4f}, "
            f"intercept={self.intercept:.4f}, "
            f"identity_strength={self.identity_strength})"
        )
