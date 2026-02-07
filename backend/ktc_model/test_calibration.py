"""Regression tests for calibration pipeline."""

import numpy as np
import pytest
from .calibration import MonotoneLinearCalibrator


def test_calibrator_preserves_variance():
    """Ensure calibrators don't collapse input range to near-constant output."""
    # Create a calibrator with identity shrinkage (new default behavior)
    cal = MonotoneLinearCalibrator(min_slope=0.0, identity_strength=10.0)

    # Simulate training data with reasonable variance
    X_train = np.linspace(-0.3, 0.3, 100)
    y_train = X_train * 0.9 + 0.05  # Slight compression is OK

    cal.fit(X_train, y_train)

    # Test that output has meaningful variance
    X_test = np.linspace(-0.3, 0.3, 200)
    y_test = cal.predict(X_test)

    unique_outputs = np.unique(np.round(y_test, 6))

    # Should have many distinct outputs, not collapsed to a step function
    assert len(unique_outputs) > 20, f"Calibrator collapsed to only {len(unique_outputs)} unique values"

    # Variance should be preserved (not reduced by more than 50%)
    input_var = np.var(X_test)
    output_var = np.var(y_test)
    assert output_var > 0.5 * input_var, f"Variance reduced from {input_var:.4f} to {output_var:.4f}"


def test_calibrator_slope_not_flat():
    """Ensure minimum slope constraint prevents flat calibration."""
    cal = MonotoneLinearCalibrator(min_slope=0.01, identity_strength=0.0)

    # Even with flat training data, slope should be at least min_slope
    X_train = np.linspace(-0.2, 0.2, 50)
    y_train = np.zeros_like(X_train)  # Completely flat target

    cal.fit(X_train, y_train)

    assert cal.slope >= 0.01, f"Slope {cal.slope} is below minimum"


def test_calibrator_linear_output():
    """Ensure calibrator produces smooth linear output, not step function."""
    cal = MonotoneLinearCalibrator(min_slope=0.0, identity_strength=10.0)

    X_train = np.linspace(-0.5, 0.5, 200)
    y_train = X_train * 0.8 + np.random.normal(0, 0.02, 200)  # Add noise

    cal.fit(X_train, y_train)

    # Test on dense grid
    X_test = np.linspace(-0.4, 0.4, 1000)
    y_test = cal.predict(X_test)

    # Check that consecutive outputs differ (no flat regions)
    diffs = np.diff(y_test)
    assert np.all(diffs > 0) or np.all(diffs >= 0), "Output should be monotonically non-decreasing"

    # Most diffs should be non-zero (not a step function)
    nonzero_diffs = np.sum(diffs > 1e-10)
    assert nonzero_diffs > 900, f"Too many flat regions: only {nonzero_diffs}/999 segments are non-flat"


def test_identity_shrinkage_pulls_toward_identity():
    """Ensure identity shrinkage pulls slope toward 1 and intercept toward 0."""
    # Without shrinkage: pure OLS
    cal_no_shrink = MonotoneLinearCalibrator(min_slope=0.0, identity_strength=0.0)
    # With strong shrinkage
    cal_shrink = MonotoneLinearCalibrator(min_slope=0.0, identity_strength=10.0)

    # Training data with slope=0.5 and intercept=0.1
    X_train = np.linspace(-0.3, 0.3, 100)
    y_train = X_train * 0.5 + 0.1

    cal_no_shrink.fit(X_train, y_train)
    cal_shrink.fit(X_train, y_train)

    # Without shrinkage: slope should be ~0.5
    assert 0.45 < cal_no_shrink.slope < 0.55, f"No-shrink slope {cal_no_shrink.slope} should be ~0.5"

    # With shrinkage: slope should be pulled toward 1.0
    # Formula: (0.5 + 10*1.0) / 11 ≈ 0.95
    assert cal_shrink.slope > 0.9, f"Shrunk slope {cal_shrink.slope} should be closer to 1.0"

    # Intercept should also be smaller with shrinkage
    assert abs(cal_shrink.intercept) < abs(cal_no_shrink.intercept), (
        f"Shrunk intercept {cal_shrink.intercept} should be smaller than {cal_no_shrink.intercept}"
    )


def test_identity_shrinkage_strength():
    """Test that stronger shrinkage pulls more aggressively toward identity."""
    X_train = np.linspace(-0.3, 0.3, 100)
    y_train = X_train * 0.3  # Very flat slope

    cal_weak = MonotoneLinearCalibrator(min_slope=0.0, identity_strength=5.0)
    cal_strong = MonotoneLinearCalibrator(min_slope=0.0, identity_strength=20.0)

    cal_weak.fit(X_train, y_train)
    cal_strong.fit(X_train, y_train)

    # Stronger shrinkage should result in slope closer to 1.0
    assert cal_strong.slope > cal_weak.slope, (
        f"Strong shrink slope {cal_strong.slope} should be larger than weak {cal_weak.slope}"
    )


def test_default_parameters():
    """Test default parameters are set correctly."""
    cal = MonotoneLinearCalibrator()
    assert cal.min_slope == 0.0, f"Default min_slope should be 0.0, got {cal.min_slope}"
    assert cal.identity_strength == 10.0, f"Default identity_strength should be 10.0, got {cal.identity_strength}"
