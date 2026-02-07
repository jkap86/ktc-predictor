"""Tests for age decline adjustment."""

import pytest
from ktc_model.age_adjustment import apply_age_decline_adjustment


class TestApplyAgeDeclineAdjustment:
    """Test cases for apply_age_decline_adjustment."""

    def test_rb_age_28_reduces_log_ratio(self):
        """RB age 28 should reduce by 0.0065 * (28-26) = 0.013."""
        result = apply_age_decline_adjustment(0.5, age=28, position="RB")
        expected = 0.5 - 0.0065 * 2
        assert abs(result - expected) < 1e-9

    def test_qb_age_33_reduces_log_ratio(self):
        """QB age 33 should reduce by 0.0045 * (33-29) = 0.018."""
        result = apply_age_decline_adjustment(0.5, age=33, position="QB")
        expected = 0.5 - 0.0045 * 4
        assert abs(result - expected) < 1e-9

    def test_wr_age_31_reduces_log_ratio(self):
        """WR age 31 should reduce by 0.0015 * (31-29) = 0.003."""
        result = apply_age_decline_adjustment(0.5, age=31, position="WR")
        expected = 0.5 - 0.0015 * 2
        assert abs(result - expected) < 1e-9

    def test_wr_age_28_unchanged(self):
        """WR age 28 (below threshold) should be unchanged."""
        result = apply_age_decline_adjustment(0.5, age=28, position="WR")
        assert result == 0.5

    def test_te_unchanged(self):
        """TE should have no adjustment regardless of age."""
        result = apply_age_decline_adjustment(0.5, age=35, position="TE")
        assert result == 0.5

    def test_none_age_unchanged(self):
        """None age should return original log_ratio."""
        result = apply_age_decline_adjustment(0.5, age=None, position="RB")
        assert result == 0.5

    def test_none_position_unchanged(self):
        """None position should return original log_ratio."""
        result = apply_age_decline_adjustment(0.5, age=30, position=None)
        assert result == 0.5

    def test_position_case_insensitive(self):
        """Position should be case insensitive."""
        result_lower = apply_age_decline_adjustment(0.5, age=28, position="rb")
        result_upper = apply_age_decline_adjustment(0.5, age=28, position="RB")
        assert result_lower == result_upper

    def test_deterministic(self):
        """Function should be deterministic (no external state)."""
        results = [apply_age_decline_adjustment(0.5, age=30, position="QB") for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_rb_boundary_age_26_unchanged(self):
        """RB exactly at age 26 should be unchanged (threshold is >26)."""
        result = apply_age_decline_adjustment(0.5, age=26, position="RB")
        assert result == 0.5

    def test_qb_boundary_age_29_unchanged(self):
        """QB exactly at age 29 should be unchanged (threshold is >29)."""
        result = apply_age_decline_adjustment(0.5, age=29, position="QB")
        assert result == 0.5
