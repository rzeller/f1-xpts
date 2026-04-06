"""Tests for backtest evaluation metrics."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from backtest.metrics import (
    BacktestResult,
    RaceMetrics,
    brier_score,
    compute_calibration,
    compute_race_metrics,
    dnf_brier_score,
    expected_points_error,
    fantasy_portfolio,
    log_likelihood,
    rank_correlation,
)


class TestLogLikelihood:
    def test_perfect_prediction(self):
        """If model predicts 100% for actual position, LL = 0."""
        pos_probs = np.zeros((3, 4))  # 3 drivers, 3 positions + DNF
        pos_probs[0, 0] = 1.0  # Driver 0 certainly wins
        pos_probs[1, 1] = 1.0  # Driver 1 certainly P2
        pos_probs[2, 2] = 1.0  # Driver 2 certainly P3

        actual_pos = {0: 1, 1: 2, 2: 3}
        actual_dnfs = {0: False, 1: False, 2: False}

        ll = log_likelihood(pos_probs, actual_pos, actual_dnfs)
        assert ll == pytest.approx(0.0, abs=1e-6)

    def test_uniform_prediction(self):
        """Uniform predictions should give negative LL."""
        n = 4
        pos_probs = np.full((n, n + 1), 1.0 / (n + 1))
        actual_pos = {i: i + 1 for i in range(n)}
        actual_dnfs = {i: False for i in range(n)}

        ll = log_likelihood(pos_probs, actual_pos, actual_dnfs)
        assert ll < 0  # Log of probabilities < 1 is negative
        expected = n * np.log(1.0 / (n + 1))
        assert ll == pytest.approx(expected, abs=1e-6)

    def test_dnf_prediction(self):
        """DNF should use the last column."""
        pos_probs = np.zeros((2, 3))  # 2 drivers, 2 pos + DNF
        pos_probs[0, 0] = 0.8
        pos_probs[0, 2] = 0.2  # DNF column
        pos_probs[1, 1] = 0.5
        pos_probs[1, 2] = 0.5

        actual_pos = {0: 1, 1: 2}
        actual_dnfs = {0: False, 1: True}  # Driver 1 DNFs

        ll = log_likelihood(pos_probs, actual_pos, actual_dnfs)
        expected = np.log(0.8) + np.log(0.5)  # P(pos=1 for D0) + P(DNF for D1)
        assert ll == pytest.approx(expected, abs=1e-6)

    def test_zero_prob_clipped(self):
        """Zero probabilities should not cause log(0)."""
        pos_probs = np.zeros((2, 3))
        pos_probs[0, 1] = 1.0  # Predicted P2, but actually P1
        pos_probs[1, 0] = 1.0

        actual_pos = {0: 1, 1: 2}
        actual_dnfs = {0: False, 1: False}

        ll = log_likelihood(pos_probs, actual_pos, actual_dnfs)
        assert np.isfinite(ll)
        assert ll < -20  # Very negative since prediction was zero


class TestBrierScore:
    def test_perfect_binary_prediction(self):
        """Perfect prediction should give Brier = 0."""
        pos_probs = np.zeros((3, 4))
        pos_probs[0, 0] = 1.0  # P1
        pos_probs[1, 1] = 1.0  # P2
        pos_probs[2, 2] = 1.0  # P3

        actual_pos = {0: 1, 1: 2, 2: 3}
        actual_dnfs = {0: False, 1: False, 2: False}

        # Win: driver 0 predicted 100% win, others 0%
        b = brier_score(pos_probs, actual_pos, actual_dnfs, cutoff=1)
        assert b == pytest.approx(0.0, abs=1e-6)

    def test_worst_prediction(self):
        """Completely wrong prediction should give Brier = 1."""
        pos_probs = np.zeros((2, 3))
        pos_probs[0, 1] = 1.0  # Predict P2
        pos_probs[1, 0] = 1.0  # Predict P1

        actual_pos = {0: 1, 1: 2}  # Opposite of prediction
        actual_dnfs = {0: False, 1: False}

        b = brier_score(pos_probs, actual_pos, actual_dnfs, cutoff=1)
        # D0: predicted 0% win, actual win → (0-1)^2 = 1
        # D1: predicted 100% win, actual P2 → (1-0)^2 = 1
        assert b == pytest.approx(1.0, abs=1e-6)

    def test_brier_range(self):
        """Brier score should be between 0 and 1."""
        rng = np.random.default_rng(42)
        n = 10
        pos_probs = rng.dirichlet(np.ones(n + 1), size=n)
        actual_pos = {i: i + 1 for i in range(n)}
        actual_dnfs = {i: False for i in range(n)}

        for cutoff in [1, 3, 6, 10]:
            b = brier_score(pos_probs, actual_pos, actual_dnfs, cutoff=cutoff)
            assert 0 <= b <= 1

    def test_dnf_counts_as_not_in_top(self):
        """DNF drivers should not count as being in top-N."""
        pos_probs = np.zeros((2, 3))
        pos_probs[0, 0] = 0.5  # 50% P1
        pos_probs[0, 2] = 0.5  # 50% DNF
        pos_probs[1, 1] = 1.0

        actual_pos = {0: 1, 1: 2}
        actual_dnfs = {0: True, 1: False}  # D0 DNFs

        b = brier_score(pos_probs, actual_pos, actual_dnfs, cutoff=1)
        # D0: predicted 50% win, actual DNF (not win) → (0.5-0)^2 = 0.25
        # D1: predicted 0% win, actual P2 (not win) → (0-0)^2 = 0
        assert b == pytest.approx(0.125, abs=1e-6)


class TestExpectedPointsError:
    def test_perfect_prediction(self):
        mae, rmse = expected_points_error({0: 25, 1: 18}, {0: 25, 1: 18})
        assert mae == pytest.approx(0.0)
        assert rmse == pytest.approx(0.0)

    def test_known_error(self):
        mae, rmse = expected_points_error({0: 20, 1: 10}, {0: 25, 1: 18})
        # Errors: -5, -8
        assert mae == pytest.approx(6.5)
        assert rmse == pytest.approx(np.sqrt((25 + 64) / 2))

    def test_handles_missing_drivers(self):
        mae, rmse = expected_points_error({0: 10}, {0: 10, 1: 5})
        assert mae == pytest.approx(0.0)


class TestRankCorrelation:
    def test_perfect_correlation(self):
        # Predicted EP matches actual finish order
        predicted_ep = {0: 25, 1: 18, 2: 15}  # Rank: 1, 2, 3
        actual_pos = {0: 1, 1: 2, 2: 3}
        spearman, kendall = rank_correlation(predicted_ep, actual_pos)
        assert spearman == pytest.approx(1.0, abs=0.01)
        assert kendall == pytest.approx(1.0, abs=0.01)

    def test_inverse_correlation(self):
        # Predicted EP is exactly backwards
        predicted_ep = {0: 5, 1: 10, 2: 25}  # Rank: 3, 2, 1
        actual_pos = {0: 1, 1: 2, 2: 3}       # But actual: 1, 2, 3
        spearman, kendall = rank_correlation(predicted_ep, actual_pos)
        assert spearman == pytest.approx(-1.0, abs=0.01)
        assert kendall == pytest.approx(-1.0, abs=0.01)


class TestFantasyPortfolio:
    def test_picks_top_five(self):
        predicted_ep = {i: 25 - i for i in range(10)}  # D0=25, D1=24, ...
        actual_points = {i: 25 - i for i in range(10)}  # Same order

        port, opt, _ = fantasy_portfolio(predicted_ep, actual_points, n_picks=5)
        # Top 5 by predicted: D0-D4, actual points: 25+24+23+22+21 = 115
        assert port == pytest.approx(115)
        assert opt == pytest.approx(115)

    def test_suboptimal_picks(self):
        predicted_ep = {0: 10, 1: 5, 2: 20}
        actual_points = {0: 0, 1: 25, 2: 1}

        port, opt, _ = fantasy_portfolio(predicted_ep, actual_points, n_picks=2)
        # Predicted top 2: D2 (20), D0 (10) → actual: 1 + 0 = 1
        # Optimal top 2: D1 (25), D2 (1) → 26
        assert port == pytest.approx(1)
        assert opt == pytest.approx(26)


class TestCalibration:
    def test_perfect_calibration(self):
        # 50% predictions that happen 50% of the time
        preds = [(0.5, 1), (0.5, 0)] * 50
        cal = compute_calibration(preds, n_bins=5)
        assert "bin_centers" in cal
        assert "actual_freq" in cal

    def test_empty_input(self):
        cal = compute_calibration([])
        assert cal == {}

    def test_bins_cover_range(self):
        preds = [(i / 10, 1 if i > 5 else 0) for i in range(11)]
        cal = compute_calibration(preds, n_bins=10)
        assert len(cal["bin_centers"]) == 10
        assert cal["bin_edges"][0] == 0.0
        assert cal["bin_edges"][-1] == 1.0


class TestComputeRaceMetrics:
    def test_returns_race_metrics(self):
        n = 5
        pos_probs = np.full((n, n + 1), 1.0 / (n + 1))
        predicted_ep = {i: 10.0 - i for i in range(n)}
        actual_pos = {i: i + 1 for i in range(n)}
        actual_dnfs = {i: False for i in range(n)}
        actual_pts = {0: 25, 1: 18, 2: 15, 3: 12, 4: 10}

        metrics = compute_race_metrics(
            race_id="test_01",
            season=2024,
            round_num=1,
            race_name="Test GP",
            pos_probs=pos_probs,
            predicted_ep=predicted_ep,
            actual_positions=actual_pos,
            actual_dnfs=actual_dnfs,
            actual_points=actual_pts,
        )

        assert isinstance(metrics, RaceMetrics)
        assert metrics.race_id == "test_01"
        assert metrics.n_drivers == n
        assert metrics.log_likelihood < 0
        assert 0 <= metrics.brier_win <= 1
        assert metrics.ep_mae >= 0


class TestBacktestResult:
    def test_aggregate(self):
        m1 = RaceMetrics(
            race_id="r1", season=2024, round_num=1, race_name="GP1", n_drivers=20,
            log_likelihood=-10, brier_win=0.05, brier_podium=0.1,
            brier_top6=0.15, brier_top10=0.2,
            ep_mae=5, ep_rmse=7, spearman_rho=0.8, kendall_tau=0.7,
            portfolio_points=50, optimal_points=100,
        )
        m2 = RaceMetrics(
            race_id="r2", season=2024, round_num=2, race_name="GP2", n_drivers=20,
            log_likelihood=-12, brier_win=0.06, brier_podium=0.12,
            brier_top6=0.18, brier_top10=0.22,
            ep_mae=6, ep_rmse=8, spearman_rho=0.7, kendall_tau=0.6,
            portfolio_points=60, optimal_points=110,
        )

        result = BacktestResult(params={"test": True}, race_metrics=[m1, m2])
        result.aggregate()

        assert result.n_races == 2
        assert result.mean_log_likelihood == pytest.approx(-11.0)
        assert result.mean_brier_win == pytest.approx(0.055)
        assert result.total_portfolio_points == pytest.approx(110)
        assert result.total_optimal_points == pytest.approx(210)
        assert result.fantasy_efficiency == pytest.approx(110 / 210)

    def test_to_dict(self):
        m = RaceMetrics(
            race_id="r1", season=2024, round_num=1, race_name="GP1", n_drivers=20,
            log_likelihood=-10, brier_win=0.05, brier_podium=0.1,
            brier_top6=0.15, brier_top10=0.2,
            ep_mae=5, ep_rmse=7, spearman_rho=0.8, kendall_tau=0.7,
            portfolio_points=50, optimal_points=100,
        )
        result = BacktestResult(params={"test": True}, race_metrics=[m])
        result.aggregate()

        d = result.to_dict()
        assert "params" in d
        assert "per_race" in d
        assert len(d["per_race"]) == 1
        assert d["n_races"] == 1
