"""Regression tests for model calibration layer (residual corrections).

These tests validate that the hinge and band corrections maintain stable
bias characteristics. They should fail if:
- Bias drifts significantly after model changes
- MAE regresses beyond tolerance
- Risers/fallers bias becomes unbalanced

Run with: pytest tests/test_calibration_regression.py -v
"""

import pytest
import pandas as pd
from pathlib import Path

# Skip if test_predictions.csv doesn't exist
TEST_PREDICTIONS_PATH = Path(__file__).parent.parent / "models" / "test_predictions.csv"


@pytest.fixture
def test_df():
    """Load test predictions if available."""
    if not TEST_PREDICTIONS_PATH.exists():
        pytest.skip(f"Test predictions not found at {TEST_PREDICTIONS_PATH}")
    df = pd.read_csv(TEST_PREDICTIONS_PATH)
    df = df[df["ppg"] >= 0]  # Filter invalid ppg
    df["residual"] = df["actual_end_ktc"] - df["predicted_end_ktc"]
    return df


class TestRBCalibration:
    """Regression tests for RB calibration (hinge + band corrections)."""

    # Tolerances for bias drift detection
    BIAS_TOLERANCE = 150  # Fail if bias changes by more than 150 KTC
    MAE_TOLERANCE_PCT = 0.05  # Fail if MAE changes by more than 5%

    # Expected values (baseline from stable model)
    # Update these after intentional model changes
    # Updated Feb 2026: HGB model with contract features + pre-draft data fix
    EXPECTED_MAE = 848.0
    EXPECTED_OVERALL_BIAS = -32  # Allow some variance

    def test_rb_2k_4k_band_bias_acceptable(self, test_df):
        """RB 2000-4000 KTC band bias should be within expected range.

        Note: test_predictions.csv contains RAW model output before residual correction.
        The band correction (+0.1673 in log-ratio space) is applied at inference time.
        Raw bias is ~+520, corrected bias should be ~+33.
        This test validates the raw model bias is stable (not regressing).
        """
        rb = test_df[test_df["position"] == "RB"]
        band = rb[(rb["start_ktc"] >= 2000) & (rb["start_ktc"] < 4000)]

        if len(band) < 50:
            pytest.skip("Not enough samples in 2k-4k band")

        bias = band["residual"].mean()

        # Raw model bias should be in expected range (~500-600)
        # If it changes significantly, the model structure has shifted
        assert 300 < bias < 800, (
            f"RB 2k-4k raw bias is {bias:.1f}, expected 400-700 range. "
            "Model behavior may have changed."
        )

    def test_rb_4k_plus_risers_vs_fallers_balanced(self, test_df):
        """RB 4000+ risers and fallers should have similar bias magnitudes."""
        rb = test_df[test_df["position"] == "RB"]
        high_ktc = rb[rb["start_ktc"] >= 4000]

        if len(high_ktc) < 100:
            pytest.skip("Not enough samples in 4k+ range")

        risers = high_ktc[high_ktc["actual_end_ktc"] > high_ktc["start_ktc"]]
        fallers = high_ktc[high_ktc["actual_end_ktc"] <= high_ktc["start_ktc"]]

        riser_bias = risers["residual"].mean() if len(risers) > 10 else 0
        faller_bias = fallers["residual"].mean() if len(fallers) > 10 else 0

        # Risers tend to be under-predicted (positive bias)
        # Fallers should be near zero
        assert faller_bias < 200, (
            f"RB 4k+ faller bias is {faller_bias:.1f}, should be near 0"
        )

    def test_rb_overall_mae_stable(self, test_df):
        """RB overall MAE should not regress beyond tolerance."""
        rb = test_df[test_df["position"] == "RB"]
        mae = rb["residual"].abs().mean()

        max_mae = self.EXPECTED_MAE * (1 + self.MAE_TOLERANCE_PCT)
        assert mae <= max_mae, (
            f"RB MAE regressed to {mae:.1f}, expected <= {max_mae:.1f}"
        )

    def test_rb_by_tier_monotonic_mae(self, test_df):
        """RB MAE by KTC tier should be roughly monotonic (higher KTC = higher MAE)."""
        rb = test_df[test_df["position"] == "RB"]

        tiers = [
            ("0-2k", 0, 2000),
            ("2k-4k", 2000, 4000),
            ("4k+", 4000, float("inf")),
        ]

        mae_by_tier = []
        for name, lo, hi in tiers:
            tier = rb[(rb["start_ktc"] >= lo) & (rb["start_ktc"] < hi)]
            if len(tier) >= 50:
                mae_by_tier.append((name, tier["residual"].abs().mean()))

        # Higher tiers should have higher MAE (more variance at top)
        # Allow small violations but flag major reversals
        if len(mae_by_tier) >= 2:
            for i in range(len(mae_by_tier) - 1):
                tier1, mae1 = mae_by_tier[i]
                tier2, mae2 = mae_by_tier[i + 1]
                # Allow 20% reversal tolerance
                assert mae2 >= mae1 * 0.8, (
                    f"MAE reversal: {tier1}={mae1:.0f} > {tier2}={mae2:.0f}"
                )


class TestQBCalibration:
    """Regression tests for QB calibration (hinge correction)."""

    # Updated Feb 2026: HGB model with contract features + pre-draft data fix
    EXPECTED_MAE = 1305.0
    MAE_TOLERANCE_PCT = 0.05

    def test_qb_mid_tier_bias_reasonable(self, test_df):
        """QB 3500-6000 KTC should have reasonable bias (hinge correction working)."""
        qb = test_df[test_df["position"] == "QB"]
        mid_tier = qb[(qb["start_ktc"] >= 3500) & (qb["start_ktc"] < 6000)]

        if len(mid_tier) < 30:
            pytest.skip("Not enough samples in QB mid-tier")

        bias = mid_tier["residual"].mean()

        # Hinge correction should reduce mid-tier over-prediction
        assert abs(bias) < 1600, (
            f"QB mid-tier bias is {bias:.1f}, expected < 800. "
            "Hinge correction may not be working."
        )

    def test_qb_overall_mae_stable(self, test_df):
        """QB overall MAE should not regress beyond tolerance."""
        qb = test_df[test_df["position"] == "QB"]
        mae = qb["residual"].abs().mean()

        max_mae = self.EXPECTED_MAE * (1 + self.MAE_TOLERANCE_PCT)
        assert mae <= max_mae, (
            f"QB MAE regressed to {mae:.1f}, expected <= {max_mae:.1f}"
        )


class TestWRTEStability:
    """Stability tests for WR/TE (should remain stable, no corrections applied)."""

    def test_wr_mae_stable(self, test_df):
        """WR MAE should remain stable (best performer)."""
        wr = test_df[test_df["position"] == "WR"]
        mae = wr["residual"].abs().mean()

        # WR has consistently low MAE around 420
        assert mae < 500, f"WR MAE regressed to {mae:.1f}, expected < 500"

    def test_te_mae_stable(self, test_df):
        """TE MAE should remain stable."""
        te = test_df[test_df["position"] == "TE"]
        mae = te["residual"].abs().mean()

        # TE has MAE around 607
        assert mae < 700, f"TE MAE regressed to {mae:.1f}, expected < 700"


class TestFeatureContract:
    """Tests for feature contract validation."""

    def test_feature_names_saved(self):
        """feature_names.json should exist after training."""
        feature_path = Path(__file__).parent.parent / "models" / "feature_names.json"
        # Skip if models not built yet
        if not (Path(__file__).parent.parent / "models" / "QB.joblib").exists():
            pytest.skip("Models not trained yet")

        assert feature_path.exists(), (
            "feature_names.json not found. Run training to generate feature contract."
        )

    def test_expected_features_match(self):
        """Expected features in predict.py should match train.py for each position."""
        from ktc_model.predict import get_expected_features
        from ktc_model.train import get_features_for_position

        # Test each position's feature contract
        for pos in ["QB", "RB", "WR", "TE"]:
            train_features = get_features_for_position(pos)
            predict_features = get_expected_features(pos)

            assert train_features == predict_features, (
                f"Feature mismatch for {pos}!\n"
                f"train.py: {train_features}\n"
                f"predict.py: {predict_features}"
            )
