"""Tests for mobility models."""

import pytest

from sim_city.models import apply_income_effect, epr_decision


class TestEPRDecision:
    """Tests for EPR decision function."""

    def test_first_visit_always_explores(self):
        """First visit should always be exploration."""
        # TODO: Implement when Agent class is complete
        pass

    def test_exploration_probability_decreases(self):
        """Exploration probability should decrease with more locations."""
        # TODO: Implement when Agent class is complete
        pass


class TestIncomeEffect:
    """Tests for income effect on mobility."""

    def test_higher_income_lower_alpha(self):
        """Higher income should result in lower alpha (more distant trips)."""
        base_alpha = 0.84
        low_income_alpha = apply_income_effect(base_alpha, income_quantile=1)
        high_income_alpha = apply_income_effect(base_alpha, income_quantile=5)
        assert high_income_alpha < low_income_alpha

    def test_middle_income_unchanged(self):
        """Middle income (quantile 3) should not change alpha."""
        base_alpha = 0.84
        result = apply_income_effect(base_alpha, income_quantile=3)
        assert result == base_alpha
