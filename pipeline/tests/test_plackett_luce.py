"""Tests for plackett_luce.py — simulation, expected points, and model fitting."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from plackett_luce import (
    simulate_races,
    compute_expected_points,
    compute_variance,
    generate_full_output,
    find_top_lineups,
)
from config import RACE_POINTS, SPRINT_POINTS, DNF_PENALTY


# A synthetic 22-driver / 11-team grid. The pipeline no longer ships with a
# hardcoded grid (the live roster is fetched from Jolpica at runtime), so the
# tests construct a stand-in here.
N_DRIVERS = 22
N_TEAMS = 11


def _stub_drivers():
    """22 drivers paired into 11 teams, indexed in roster order."""
    drivers = []
    for i in range(N_DRIVERS):
        team_idx = i // 2
        drivers.append({
            "name": f"Driver {i}",
            "abbr": f"D{i:02d}",
            "team_id": f"team_{team_idx}",
            "team_name": f"Team {team_idx}",
            "team_idx": team_idx,
            "team_color": "#888888",
        })
    return drivers


@pytest.fixture
def stub_drivers():
    return _stub_drivers()


@pytest.fixture
def simple_lambdas():
    """3 drivers with clear strength ordering."""
    return np.log(np.array([3.0, 2.0, 1.0]))


@pytest.fixture
def simple_dnfs():
    """3 drivers with low DNF rates."""
    return np.array([0.05, 0.05, 0.05])


@pytest.fixture
def full_grid_lambdas():
    """22 drivers with realistic strength spread."""
    rng = np.random.default_rng(42)
    return np.sort(rng.normal(0, 1, N_DRIVERS))[::-1]  # strongest first


@pytest.fixture
def full_grid_dnfs():
    return np.full(N_DRIVERS, 0.10)


@pytest.fixture
def team_indices(stub_drivers):
    return np.array([d["team_idx"] for d in stub_drivers])


# --- simulate_races ---

class TestSimulateRaces:
    def test_output_shape(self, simple_lambdas, simple_dnfs):
        result = simulate_races(simple_lambdas, simple_dnfs, n_sims=1000)
        n = len(simple_lambdas)
        assert result.shape == (n, n + 1)  # positions + DNF column

    def test_rows_sum_to_one(self, simple_lambdas, simple_dnfs):
        result = simulate_races(simple_lambdas, simple_dnfs, n_sims=5000)
        for i in range(len(simple_lambdas)):
            assert result[i].sum() == pytest.approx(1.0, abs=0.01)

    def test_strongest_driver_wins_most(self, simple_lambdas, simple_dnfs):
        result = simulate_races(simple_lambdas, simple_dnfs, n_sims=10000)
        # Driver 0 (λ=3) should win more than driver 2 (λ=1)
        assert result[0, 0] > result[2, 0]

    def test_dnf_probability_respected(self):
        lambdas = np.zeros(3)
        high_dnf = np.array([0.5, 0.5, 0.5])
        result = simulate_races(lambdas, high_dnf, n_sims=10000)
        # Each driver should DNF ~50% of the time
        for i in range(3):
            assert result[i, -1] == pytest.approx(0.5, abs=0.05)

    def test_zero_dnf(self):
        lambdas = np.zeros(3)
        zero_dnf = np.array([0.0, 0.0, 0.0])
        result = simulate_races(lambdas, zero_dnf, n_sims=1000)
        # No DNFs
        for i in range(3):
            assert result[i, -1] == 0.0

    def test_deterministic_with_seed(self, simple_lambdas, simple_dnfs):
        r1 = simulate_races(simple_lambdas, simple_dnfs, n_sims=1000, seed=123)
        r2 = simulate_races(simple_lambdas, simple_dnfs, n_sims=1000, seed=123)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self, simple_lambdas, simple_dnfs):
        r1 = simulate_races(simple_lambdas, simple_dnfs, n_sims=1000, seed=1)
        r2 = simulate_races(simple_lambdas, simple_dnfs, n_sims=1000, seed=2)
        assert not np.array_equal(r1, r2)

    def test_equal_lambdas_uniform(self):
        """Equal-strength drivers should have ~equal win probabilities."""
        lambdas = np.zeros(4)  # all log(1) = 0
        dnfs = np.zeros(4)
        result = simulate_races(lambdas, dnfs, n_sims=20000)
        # Each driver should win ~25% of the time
        for i in range(4):
            assert result[i, 0] == pytest.approx(0.25, abs=0.03)

    def test_full_grid(self, full_grid_lambdas, full_grid_dnfs):
        result = simulate_races(full_grid_lambdas, full_grid_dnfs, n_sims=5000)
        assert result.shape == (N_DRIVERS, N_DRIVERS + 1)
        for i in range(N_DRIVERS):
            assert result[i].sum() == pytest.approx(1.0, abs=0.01)

    def test_with_correlation(self, full_grid_lambdas, full_grid_dnfs, team_indices):
        correlation = {"sigma_team": 0.5, "sigma_global": 0.3, "sigma_dnf": 0.2}
        result = simulate_races(
            full_grid_lambdas, full_grid_dnfs, n_sims=5000,
            team_indices=team_indices, correlation=correlation,
        )
        assert result.shape == (N_DRIVERS, N_DRIVERS + 1)
        for i in range(N_DRIVERS):
            assert result[i].sum() == pytest.approx(1.0, abs=0.01)


# --- compute_expected_points ---

class TestComputeExpectedPoints:
    def test_certain_winner(self):
        """If a driver wins 100% of the time, EP = P1 points."""
        dist = np.zeros(23)
        dist[0] = 1.0  # Always P1
        ep = compute_expected_points(dist, RACE_POINTS)
        assert ep == 25.0

    def test_certain_dnf(self):
        """If a driver always DNFs, EP = DNF penalty."""
        dist = np.zeros(23)
        dist[-1] = 1.0
        ep = compute_expected_points(dist, RACE_POINTS)
        assert ep == DNF_PENALTY

    def test_certain_p11(self):
        """P11 scores 0 points."""
        dist = np.zeros(23)
        dist[10] = 1.0  # P11
        ep = compute_expected_points(dist, RACE_POINTS)
        assert ep == 0.0

    def test_50_50_win_or_dnf(self):
        dist = np.zeros(23)
        dist[0] = 0.5
        dist[-1] = 0.5
        ep = compute_expected_points(dist, RACE_POINTS)
        assert ep == pytest.approx(0.5 * 25 + 0.5 * DNF_PENALTY)

    def test_sprint_scoring(self):
        dist = np.zeros(23)
        dist[0] = 1.0
        ep = compute_expected_points(dist, SPRINT_POINTS)
        assert ep == 8.0


# --- compute_variance ---

class TestComputeVariance:
    def test_certain_outcome_zero_variance(self):
        dist = np.zeros(23)
        dist[0] = 1.0
        var = compute_variance(dist, RACE_POINTS)
        assert var == pytest.approx(0.0)

    def test_positive_variance_for_mixed(self):
        dist = np.zeros(23)
        dist[0] = 0.5
        dist[-1] = 0.5
        var = compute_variance(dist, RACE_POINTS)
        assert var > 0

    def test_variance_formula(self):
        """Manual check: 50% P1 (25pts), 50% DNF (-10pts)."""
        dist = np.zeros(23)
        dist[0] = 0.5
        dist[-1] = 0.5
        ep = 0.5 * 25 + 0.5 * DNF_PENALTY
        expected_var = 0.5 * (25 - ep) ** 2 + 0.5 * (DNF_PENALTY - ep) ** 2
        var = compute_variance(dist, RACE_POINTS)
        assert var == pytest.approx(expected_var)


# --- generate_full_output ---

class TestGenerateFullOutput:
    def test_returns_22_drivers(self, full_grid_lambdas, full_grid_dnfs, team_indices, stub_drivers):
        output = generate_full_output(
            full_grid_lambdas, full_grid_dnfs, stub_drivers, n_sims=2000,
            team_indices=team_indices,
        )
        assert len(output) == N_DRIVERS

    def test_sorted_by_ep_total(self, full_grid_lambdas, full_grid_dnfs, team_indices, stub_drivers):
        output = generate_full_output(
            full_grid_lambdas, full_grid_dnfs, stub_drivers, n_sims=2000,
            team_indices=team_indices,
        )
        totals = [d["ep_total"] for d in output]
        assert totals == sorted(totals, reverse=True)

    def test_required_fields(self, full_grid_lambdas, full_grid_dnfs, team_indices, stub_drivers):
        output = generate_full_output(
            full_grid_lambdas, full_grid_dnfs, stub_drivers, n_sims=2000,
            team_indices=team_indices,
        )
        required = {
            "name", "abbr", "team_idx", "lambda", "p_dnf",
            "ep_race", "ep_sprint", "ep_total", "std_dev",
            "p_win", "p_podium", "p_top6", "p_top10",
            "p_no_points", "position_distribution",
        }
        for d in output:
            assert required.issubset(d.keys())

    def test_position_distribution_length(self, full_grid_lambdas, full_grid_dnfs, team_indices, stub_drivers):
        output = generate_full_output(
            full_grid_lambdas, full_grid_dnfs, stub_drivers, n_sims=2000,
            team_indices=team_indices,
        )
        for d in output:
            assert len(d["position_distribution"]) == N_DRIVERS + 1

    def test_probabilities_valid(self, full_grid_lambdas, full_grid_dnfs, team_indices, stub_drivers):
        output = generate_full_output(
            full_grid_lambdas, full_grid_dnfs, stub_drivers, n_sims=2000,
            team_indices=team_indices,
        )
        for d in output:
            assert 0 <= d["p_win"] <= 1
            assert 0 <= d["p_podium"] <= 1
            assert 0 <= d["p_top6"] <= 1
            assert 0 <= d["p_top10"] <= 1
            assert d["p_win"] <= d["p_podium"] <= d["p_top6"] <= d["p_top10"]

    def test_sprint_adds_points(self, full_grid_lambdas, full_grid_dnfs, team_indices, stub_drivers):
        output_no_sprint = generate_full_output(
            full_grid_lambdas, full_grid_dnfs, stub_drivers, is_sprint=False, n_sims=2000,
            team_indices=team_indices,
        )
        output_sprint = generate_full_output(
            full_grid_lambdas, full_grid_dnfs, stub_drivers, is_sprint=True, n_sims=2000,
            team_indices=team_indices,
        )
        # Sprint weekend should add ep_sprint > 0 for top drivers
        top_sprint = max(d["ep_sprint"] for d in output_sprint)
        top_no_sprint = max(d["ep_sprint"] for d in output_no_sprint)
        assert top_sprint > 0
        assert top_no_sprint == 0


# --- find_top_lineups ---

class TestFindTopLineups:
    @pytest.fixture
    def drivers_data(self, full_grid_lambdas, full_grid_dnfs, team_indices, stub_drivers):
        return generate_full_output(
            full_grid_lambdas, full_grid_dnfs, stub_drivers, n_sims=2000,
            team_indices=team_indices,
        )

    def test_returns_requested_count(self, drivers_data):
        lineups = find_top_lineups(drivers_data, top_n=5)
        assert len(lineups) == 5

    def test_ranked_in_order(self, drivers_data):
        lineups = find_top_lineups(drivers_data, top_n=10)
        for i, lineup in enumerate(lineups):
            assert lineup["rank"] == i + 1

    def test_grand_total_decreasing(self, drivers_data):
        lineups = find_top_lineups(drivers_data, top_n=10)
        totals = [l["ep_grand_total"] for l in lineups]
        assert totals == sorted(totals, reverse=True)

    def test_five_picks_per_lineup(self, drivers_data):
        lineups = find_top_lineups(drivers_data, n_picks=5, top_n=3)
        for lineup in lineups:
            assert len(lineup["picks"]) == 5

    def test_slots_1_through_5(self, drivers_data):
        lineups = find_top_lineups(drivers_data, n_picks=5, top_n=3)
        for lineup in lineups:
            slots = sorted(p["slot"] for p in lineup["picks"])
            assert slots == [1, 2, 3, 4, 5]

    def test_grand_total_equals_base_plus_bonus(self, drivers_data):
        lineups = find_top_lineups(drivers_data, top_n=3)
        for lineup in lineups:
            expected = lineup["ep_base_total"] + lineup["ep_bonus_total"]
            assert lineup["ep_grand_total"] == pytest.approx(expected, abs=0.02)
