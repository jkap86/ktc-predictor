"""Regression tests to prevent pre-draft/college data contamination.

These tests verify that pre-draft seasons (years_exp < 0) never enter
the training pipeline, and that feature source seasons always have
actual NFL games played.

Run with: pytest tests/test_data_contamination.py -v
"""

import pytest
from app.services.data_loader import DataLoader
from app.services.ktc_utils import select_anchor_ktc, compute_prior_ktc_features


@pytest.fixture(scope="module")
def loader():
    return DataLoader()


class TestTrainingDataFilters:
    """Ensure pre-draft seasons are excluded from training data."""

    def test_no_predraft_in_training_dataframe(self, loader):
        """Training DataFrame must not contain years_exp < 0."""
        df = loader.get_training_dataframe()
        predraft = df[df["years_exp"] < 0]
        assert len(predraft) == 0, (
            f"Found {len(predraft)} pre-draft rows in training DataFrame"
        )

    def test_no_zero_games_feature_source(self, loader):
        """Feature matrix must not use seasons with 0 games as feature source."""
        X, y = loader.get_feature_matrix()
        zero_games = X[X["games_played"] == 0]
        assert len(zero_games) == 0, (
            f"Found {len(zero_games)} zero-game feature source rows"
        )

    def test_no_predraft_in_weekly_matrix(self, loader):
        """Weekly feature matrix must not contain years_exp < 0."""
        X, y = loader.get_weekly_feature_matrix()
        predraft = X[X["years_exp"] < 0]
        assert len(predraft) == 0, (
            f"Found {len(predraft)} pre-draft rows in weekly matrix"
        )


class TestKtcUtilsSkipPredraft:
    """Ensure KTC selection utilities skip pre-draft seasons."""

    def test_anchor_ktc_skips_predraft_only(self):
        """Player with only pre-draft season should return None."""
        seasons = [
            {"year": 2023, "years_exp": -1, "start_ktc": 3827, "end_ktc": 3500}
        ]
        assert select_anchor_ktc(seasons) is None

    def test_anchor_ktc_prefers_nfl_over_predraft(self):
        """Should select NFL season even if pre-draft season is newer."""
        seasons = [
            {"year": 2022, "years_exp": 1, "start_ktc": 2000, "end_ktc": 2500},
            {"year": 2023, "years_exp": -1, "start_ktc": 3827, "end_ktc": 3500},
        ]
        result = select_anchor_ktc(seasons)
        assert result is not None
        assert result[1] == 2022  # Should pick the NFL season

    def test_prior_ktc_skips_predraft(self):
        """Pre-draft seasons should not contribute to prior KTC features."""
        seasons = [
            {"year": 2022, "years_exp": -1, "end_ktc": 3827},
            {"year": 2023, "years_exp": 0, "end_ktc": 2500},
        ]
        prior_end_ktc, max_ktc_prior = compute_prior_ktc_features(seasons, 2023)
        # Pre-draft season should be skipped, so no prior data
        assert prior_end_ktc is None
        assert max_ktc_prior is None
