"""Tests for devig.py — odds conversion and devigorization methods."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from devig import (
    american_to_implied,
    decimal_to_implied,
    fractional_to_implied,
    devig_multiplicative,
    devig_shin,
    devig_power,
    devig_market,
)


# --- Odds conversion ---

class TestAmericanToImplied:
    def test_even_money(self):
        assert american_to_implied(100) == pytest.approx(0.5)

    def test_favorite(self):
        # -200 means risk 200 to win 100 → implied = 200/300 = 0.6667
        assert american_to_implied(-200) == pytest.approx(2 / 3)

    def test_heavy_favorite(self):
        assert american_to_implied(-500) == pytest.approx(5 / 6)

    def test_underdog(self):
        # +300 means risk 100 to win 300 → implied = 100/400 = 0.25
        assert american_to_implied(300) == pytest.approx(0.25)

    def test_longshot(self):
        # +10000 → 100/10100 ≈ 0.0099
        result = american_to_implied(10000)
        assert 0.009 < result < 0.01

    def test_implied_between_0_and_1(self):
        for odds in [-500, -200, -100, 100, 200, 500, 5000]:
            p = american_to_implied(odds)
            assert 0 < p < 1


class TestDecimalToImplied:
    def test_even_money(self):
        assert decimal_to_implied(2.0) == pytest.approx(0.5)

    def test_favorite(self):
        assert decimal_to_implied(1.5) == pytest.approx(2 / 3)

    def test_longshot(self):
        assert decimal_to_implied(10.0) == pytest.approx(0.1)


class TestFractionalToImplied:
    def test_even_money(self):
        assert fractional_to_implied(1, 1) == pytest.approx(0.5)

    def test_odds_on(self):
        # 1/2 → den/(num+den) = 2/3
        assert fractional_to_implied(1, 2) == pytest.approx(2 / 3)

    def test_longshot(self):
        # 10/1 → 1/11
        assert fractional_to_implied(10, 1) == pytest.approx(1 / 11)


# --- Devig methods ---

@pytest.fixture
def viggy_probs():
    """Implied probabilities with ~10% overround (sum ≈ 1.10)."""
    return np.array([0.40, 0.25, 0.20, 0.15, 0.10])


@pytest.fixture
def fair_field():
    """A fair 22-driver field: implied probs sum to ~1.15 (typical vig)."""
    rng = np.random.default_rng(42)
    raw = rng.dirichlet(np.ones(22))  # fair probs
    # Add ~15% vig
    return raw * 1.15


class TestDevigMultiplicative:
    def test_sums_to_one(self, viggy_probs):
        result = devig_multiplicative(viggy_probs)
        assert result.sum() == pytest.approx(1.0)

    def test_preserves_ranking(self, viggy_probs):
        result = devig_multiplicative(viggy_probs)
        # Largest implied → largest fair
        assert np.argmax(result) == np.argmax(viggy_probs)

    def test_all_positive(self, viggy_probs):
        result = devig_multiplicative(viggy_probs)
        assert (result > 0).all()

    def test_zero_sum_raises(self):
        with pytest.raises(ValueError):
            devig_multiplicative(np.array([0.0, 0.0]))


class TestDevigShin:
    def test_sums_to_one(self, viggy_probs):
        result = devig_shin(viggy_probs)
        assert result.sum() == pytest.approx(1.0, abs=1e-6)

    def test_preserves_ranking(self, viggy_probs):
        result = devig_shin(viggy_probs)
        order_implied = np.argsort(-viggy_probs)
        order_fair = np.argsort(-result)
        np.testing.assert_array_equal(order_implied, order_fair)

    def test_all_positive(self, viggy_probs):
        result = devig_shin(viggy_probs)
        assert (result > 0).all()

    def test_reduces_overround(self, viggy_probs):
        result = devig_shin(viggy_probs)
        # Fair probs should be smaller than implied (each individually)
        assert (result <= viggy_probs).all()

    def test_no_vig_passthrough(self):
        """If probs already sum to 1, Shin should return them unchanged."""
        fair = np.array([0.5, 0.3, 0.2])
        result = devig_shin(fair)
        np.testing.assert_allclose(result, fair, atol=1e-6)

    def test_favorite_longshot_bias(self):
        """Shin should remove more from longshots than favorites (FLB correction)."""
        implied = np.array([0.60, 0.30, 0.10, 0.05, 0.05])  # sum=1.10
        fair = devig_shin(implied)
        mult = devig_multiplicative(implied)
        # For the favorite, Shin's fair prob should be higher than multiplicative
        # (FLB means favorites are underpriced, longshots overpriced)
        assert fair[0] > mult[0]

    def test_large_field(self, fair_field):
        result = devig_shin(fair_field)
        assert result.sum() == pytest.approx(1.0, abs=1e-6)
        assert (result > 0).all()


class TestDevigPower:
    def test_sums_to_one(self, viggy_probs):
        result = devig_power(viggy_probs)
        assert result.sum() == pytest.approx(1.0, abs=1e-5)

    def test_preserves_ranking(self, viggy_probs):
        result = devig_power(viggy_probs)
        order_implied = np.argsort(-viggy_probs)
        order_fair = np.argsort(-result)
        np.testing.assert_array_equal(order_implied, order_fair)

    def test_all_positive(self, viggy_probs):
        result = devig_power(viggy_probs)
        assert (result > 0).all()


# --- Market-level devig ---

class TestDevigMarket:
    @pytest.fixture
    def win_odds(self):
        return {
            "Russell": -150,
            "Antonelli": 200,
            "Leclerc": 600,
            "Verstappen": 1200,
        }

    def test_returns_dict(self, win_odds):
        result = devig_market(win_odds, format="american", method="shin")
        assert isinstance(result, dict)
        assert set(result.keys()) == set(win_odds.keys())

    def test_sums_to_one(self, win_odds):
        result = devig_market(win_odds, format="american", method="shin")
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-5)

    def test_all_methods(self, win_odds):
        for method in ["shin", "multiplicative", "power"]:
            result = devig_market(win_odds, format="american", method=method)
            assert sum(result.values()) == pytest.approx(1.0, abs=1e-4)

    def test_decimal_format(self):
        odds = {"A": 2.0, "B": 3.0, "C": 6.0}  # sum implied = 1.0 + overlap
        # 1/2 + 1/3 + 1/6 = 1.0 exactly — no vig
        result = devig_market(odds, format="decimal", method="multiplicative")
        assert result["A"] == pytest.approx(0.5, abs=1e-5)

    def test_unknown_format_raises(self, win_odds):
        with pytest.raises(ValueError):
            devig_market(win_odds, format="unknown")

    def test_unknown_method_raises(self, win_odds):
        with pytest.raises(ValueError):
            devig_market(win_odds, format="american", method="unknown")
