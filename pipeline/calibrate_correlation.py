#!/usr/bin/env python3
"""
Calibrate race-day correlation parameters (sigma_team, sigma_global, sigma_dnf)
from historical F1 race results.

Methodology:
  1. Load historical results (from download_historical.py output)
  2. Compute three target statistics from real data:
     - Teammate residual correlation (constrains sigma_team)
     - Race-level variance ratio (constrains sigma_global)
     - DNF overdispersion ratio (constrains sigma_dnf)
  3. Search for sigma values that make the simulation match these statistics

Usage:
    # With downloaded data:
    python calibrate_correlation.py --data pipeline/historical_results.json

    # Dry run showing the methodology with known F1 statistics:
    python calibrate_correlation.py --use-defaults
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import N_DRIVERS, N_TEAMS


# ---------------------------------------------------------------------------
# Step 1: Compute target statistics from historical data
# ---------------------------------------------------------------------------

def load_historical(path: str) -> list:
    """Load races from the JSON produced by download_historical.py."""
    with open(path) as f:
        data = json.load(f)
    return data["races"]


def build_team_map(races: list) -> dict:
    """
    Build a mapping from (season, driver_code) -> team_name.
    Also returns the set of (season, team) -> [driver_codes] for teammate lookup.
    """
    driver_team = {}
    team_drivers = defaultdict(list)
    for race in races:
        season = race["season"]
        for r in race["results"]:
            key = (season, r["driver"])
            if key not in driver_team:
                driver_team[key] = r["team"]
            tkey = (season, r["team"])
            if r["driver"] not in team_drivers[tkey]:
                team_drivers[tkey].append(r["driver"])
    return driver_team, dict(team_drivers)


def compute_season_strengths(races: list) -> dict:
    """
    Compute average finishing position per driver per season.
    Only count races where the driver finished (not DNF).
    Returns {(season, driver): mean_position}.
    """
    positions = defaultdict(list)
    for race in races:
        for r in race["results"]:
            if not r["dnf"]:
                positions[(race["season"], r["driver"])].append(r["position"])

    return {
        key: np.mean(vals)
        for key, vals in positions.items()
        if len(vals) >= 5  # need enough races for a stable average
    }


def compute_teammate_correlation(races: list) -> float:
    """
    Compute the correlation between teammates' race-day residuals.

    For each race, compute each driver's residual = position - season_average.
    Then for each teammate pair in that race, record both residuals.
    Return the Pearson correlation across all (teammate_A_residual, teammate_B_residual) pairs.
    """
    strengths = compute_season_strengths(races)
    driver_team, team_drivers = build_team_map(races)

    pair_a = []
    pair_b = []

    for race in races:
        season = race["season"]
        # Build position/residual lookup for this race's finishers
        race_residuals = {}
        for r in race["results"]:
            if r["dnf"]:
                continue
            key = (season, r["driver"])
            if key in strengths:
                residual = r["position"] - strengths[key]
                race_residuals[r["driver"]] = residual

        # Find teammate pairs that both finished
        seen_teams = set()
        for r in race["results"]:
            team_key = (season, r["team"])
            if team_key in seen_teams:
                continue
            seen_teams.add(team_key)

            teammates = team_drivers.get(team_key, [])
            if len(teammates) < 2:
                continue

            # Get the two regular drivers (most appearances)
            present = [d for d in teammates if d in race_residuals]
            if len(present) >= 2:
                pair_a.append(race_residuals[present[0]])
                pair_b.append(race_residuals[present[1]])

    if len(pair_a) < 10:
        print(f"  WARNING: Only {len(pair_a)} teammate pairs found")
        return 0.35  # fallback

    corr = np.corrcoef(pair_a, pair_b)[0, 1]
    print(f"  Teammate residual correlation: {corr:.4f} ({len(pair_a)} pairs)")
    return corr


def compute_race_variance_ratio(races: list) -> float:
    """
    Compute how much race-level finishing variance varies across races.

    For each race, compute the std dev of finisher residuals.
    The coefficient of variation of these per-race std devs tells us
    how much "chaos" varies race to race.

    Returns the CV (std of race_stds / mean of race_stds).
    """
    strengths = compute_season_strengths(races)
    race_stds = []

    for race in races:
        season = race["season"]
        residuals = []
        for r in race["results"]:
            if r["dnf"]:
                continue
            key = (season, r["driver"])
            if key in strengths:
                residuals.append(r["position"] - strengths[key])

        if len(residuals) >= 10:
            race_stds.append(np.std(residuals))

    if len(race_stds) < 10:
        print(f"  WARNING: Only {len(race_stds)} races with enough data")
        return 0.30  # fallback

    cv = np.std(race_stds) / np.mean(race_stds)
    print(f"  Race variance CV: {cv:.4f} (mean std={np.mean(race_stds):.2f}, "
          f"std of stds={np.std(race_stds):.2f}, {len(race_stds)} races)")
    return cv


def compute_dnf_overdispersion(races: list) -> float:
    """
    Compute the overdispersion ratio of DNF counts per race.

    If DNFs were independent Bernoulli trials, Var(count) = n*p*(1-p).
    The overdispersion ratio = observed_var / expected_var tells us
    how correlated DNFs are.
    """
    dnf_counts = []
    n_drivers_per_race = []

    for race in races:
        n_total = len(race["results"])
        n_dnf = sum(1 for r in race["results"] if r["dnf"])
        dnf_counts.append(n_dnf)
        n_drivers_per_race.append(n_total)

    dnf_counts = np.array(dnf_counts, dtype=float)
    n_drivers = np.mean(n_drivers_per_race)

    observed_mean = np.mean(dnf_counts)
    observed_var = np.var(dnf_counts)

    # Under independence: Var = n * p * (1-p) where p = mean/n
    p_dnf = observed_mean / n_drivers
    expected_var = n_drivers * p_dnf * (1 - p_dnf)

    ratio = observed_var / max(expected_var, 0.01)
    print(f"  DNF stats: mean={observed_mean:.2f}/race, var={observed_var:.2f}, "
          f"expected_var={expected_var:.2f}")
    print(f"  DNF overdispersion ratio: {ratio:.4f} ({len(dnf_counts)} races)")
    return ratio


# ---------------------------------------------------------------------------
# Step 2: Simulation-based moment matching
# ---------------------------------------------------------------------------

def simulate_teammate_correlation(
    sigma_team: float,
    sigma_global: float,
    n_drivers: int = 20,
    n_teams: int = 10,
    n_sims: int = 5000,
    seed: int = 42,
) -> float:
    """
    Simulate PL races with the given sigma_team and measure teammate
    residual correlation (position minus expected position).

    Uses a realistic spread of driver strengths, with teammates having
    similar but not identical strengths (matching real F1 team structures).
    """
    rng = np.random.default_rng(seed)
    team_indices = np.repeat(np.arange(n_teams), n_drivers // n_teams)

    # Realistic strengths: teams spaced out, small intra-team gap
    team_strengths = np.linspace(1.5, -1.0, n_teams)
    log_lambdas = np.zeros(n_drivers)
    for t in range(n_teams):
        teammates = np.where(team_indices == t)[0]
        log_lambdas[teammates[0]] = team_strengths[t] + 0.15  # faster teammate
        log_lambdas[teammates[1]] = team_strengths[t] - 0.15  # slower teammate

    # Compute expected positions from a large no-team-noise baseline
    n_baseline = 20000
    baseline_positions = np.zeros((n_baseline, n_drivers))
    baseline_rng = np.random.default_rng(seed + 999)
    for sim in range(n_baseline):
        gumbel = baseline_rng.gumbel(size=n_drivers)
        utilities = log_lambdas + gumbel
        order = np.argsort(-utilities)
        positions = np.empty(n_drivers, dtype=int)
        positions[order] = np.arange(1, n_drivers + 1)
        baseline_positions[sim] = positions
    expected_pos = baseline_positions.mean(axis=0)

    # Simulate with team noise and measure residual correlation
    pair_a = []
    pair_b = []

    for sim in range(n_sims):
        # Team noise
        z_team = rng.standard_normal(n_teams)
        team_noise = sigma_team * z_team[team_indices]

        # Chaos scaling
        chaos_scale = 1.0
        if sigma_global > 0:
            z_global = rng.standard_normal()
            chaos_scale = np.exp(sigma_global * z_global)

        # Gumbel noise
        gumbel = rng.gumbel(size=n_drivers)

        # Utilities
        utilities = (log_lambdas + team_noise) + chaos_scale * gumbel

        # Positions (1-indexed)
        order = np.argsort(-utilities)
        positions = np.empty(n_drivers, dtype=int)
        positions[order] = np.arange(1, n_drivers + 1)

        # Record teammate residual pairs
        residuals = positions - expected_pos
        for t in range(n_teams):
            teammates = np.where(team_indices == t)[0]
            if len(teammates) == 2:
                pair_a.append(residuals[teammates[0]])
                pair_b.append(residuals[teammates[1]])

    return np.corrcoef(pair_a, pair_b)[0, 1]


def simulate_race_variance_cv(
    sigma_global: float,
    n_drivers: int = 20,
    n_sims: int = 5000,
    seed: int = 123,
) -> float:
    """
    Simulate PL races with the given sigma_global and measure the CV of
    per-race finishing position std devs.

    Uses a realistic spread of driver strengths (not equal), because the
    chaos effect works by changing the noise-to-signal ratio. With equal
    strengths, Gumbel scaling is invariant under ranking.
    """
    rng = np.random.default_rng(seed)
    # Realistic log-lambda spread: top drivers ~1.5, backmarkers ~-1.0
    log_lambdas = np.linspace(1.5, -1.0, n_drivers)

    # Compute "expected position" from a large baseline simulation (no chaos)
    n_baseline = 20000
    baseline_positions = np.zeros((n_baseline, n_drivers))
    for sim in range(n_baseline):
        gumbel = rng.gumbel(size=n_drivers)
        utilities = log_lambdas + gumbel
        order = np.argsort(-utilities)
        positions = np.empty(n_drivers, dtype=int)
        positions[order] = np.arange(1, n_drivers + 1)
        baseline_positions[sim] = positions
    expected_pos = baseline_positions.mean(axis=0)

    # Now simulate with chaos scaling and measure residual variance per race
    race_stds = []
    for sim in range(n_sims):
        chaos_scale = 1.0
        if sigma_global > 0:
            z_global = rng.standard_normal()
            chaos_scale = np.exp(sigma_global * z_global)

        gumbel = rng.gumbel(size=n_drivers)
        utilities = log_lambdas + chaos_scale * gumbel

        order = np.argsort(-utilities)
        positions = np.empty(n_drivers, dtype=int)
        positions[order] = np.arange(1, n_drivers + 1)

        residuals = positions - expected_pos
        race_stds.append(np.std(residuals))

    return np.std(race_stds) / np.mean(race_stds)


def simulate_dnf_overdispersion(
    sigma_dnf: float,
    p_dnf_base: float = 0.10,
    n_drivers: int = 20,
    n_sims: int = 10000,
    seed: int = 456,
) -> float:
    """
    Simulate DNF counts with correlated DNF probabilities and measure
    the overdispersion ratio.
    """
    rng = np.random.default_rng(seed)
    dnf_counts = []

    for sim in range(n_sims):
        # Log-normal multiplier on DNF probabilities
        if sigma_dnf > 0:
            z = rng.standard_normal()
            mult = np.exp(sigma_dnf * z)
            effective_p = min(p_dnf_base * mult, 0.5)
        else:
            effective_p = p_dnf_base

        # All drivers share the same effective DNF probability this race
        n_dnf = rng.binomial(n_drivers, effective_p)
        dnf_counts.append(n_dnf)

    dnf_counts = np.array(dnf_counts, dtype=float)
    observed_var = np.var(dnf_counts)
    observed_mean = np.mean(dnf_counts)
    expected_var = n_drivers * (observed_mean / n_drivers) * (1 - observed_mean / n_drivers)

    return observed_var / max(expected_var, 0.01)


# ---------------------------------------------------------------------------
# Step 3: Fit each sigma independently via 1D search
# ---------------------------------------------------------------------------

def fit_sigma_team(target_corr: float, sigma_global: float = 0.0) -> float:
    """Find sigma_team that produces the target teammate correlation."""
    def loss(sigma_team):
        sim_corr = simulate_teammate_correlation(
            sigma_team=sigma_team,
            sigma_global=sigma_global,
            n_sims=8000,
        )
        return (sim_corr - target_corr) ** 2

    result = minimize_scalar(loss, bounds=(0.0, 3.0), method="bounded",
                             options={"maxiter": 40, "xatol": 0.005})
    return result.x


def fit_sigma_global(target_cv: float) -> float:
    """Find sigma_global that produces the target race-variance CV."""
    def loss(sigma_global):
        sim_cv = simulate_race_variance_cv(sigma_global=sigma_global, n_sims=8000)
        return (sim_cv - target_cv) ** 2

    result = minimize_scalar(loss, bounds=(0.0, 3.0), method="bounded",
                             options={"maxiter": 40, "xatol": 0.005})
    return result.x


def fit_sigma_dnf(target_ratio: float, p_dnf_base: float = 0.10) -> float:
    """Find sigma_dnf that produces the target DNF overdispersion ratio."""
    def loss(sigma_dnf):
        sim_ratio = simulate_dnf_overdispersion(
            sigma_dnf=sigma_dnf, p_dnf_base=p_dnf_base, n_sims=20000,
        )
        return (sim_ratio - target_ratio) ** 2

    result = minimize_scalar(loss, bounds=(0.0, 3.0), method="bounded",
                             options={"maxiter": 40, "xatol": 0.005})
    return result.x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def calibrate_from_data(data_path: str) -> dict:
    """Full calibration pipeline from historical data file."""
    print(f"Loading historical data from {data_path}...")
    races = load_historical(data_path)
    print(f"  {len(races)} races loaded\n")

    # Compute target statistics
    print("Computing target statistics from historical data:")
    target_corr = compute_teammate_correlation(races)
    target_cv = compute_race_variance_ratio(races)
    target_overdispersion = compute_dnf_overdispersion(races)

    # Compute base DNF rate
    total_entries = sum(len(r["results"]) for r in races)
    total_dnfs = sum(sum(1 for d in r["results"] if d["dnf"]) for r in races)
    base_dnf_rate = total_dnfs / total_entries
    print(f"\n  Base DNF rate: {base_dnf_rate:.4f} ({total_dnfs}/{total_entries})")

    print(f"\nTarget statistics:")
    print(f"  Teammate correlation:    {target_corr:.4f}")
    print(f"  Race variance CV:        {target_cv:.4f}")
    print(f"  DNF overdispersion:      {target_overdispersion:.4f}")

    return fit_to_targets(target_corr, target_cv, target_overdispersion, base_dnf_rate)


def fit_to_targets(
    target_corr: float,
    target_cv: float,
    target_overdispersion: float,
    base_dnf_rate: float = 0.10,
) -> dict:
    """Fit sigma values to match target statistics."""
    # Fit sigma_global first (independent of sigma_team)
    print(f"\nFitting sigma_global to match race variance CV = {target_cv:.4f}...")
    sigma_global = fit_sigma_global(target_cv)
    verify_cv = simulate_race_variance_cv(sigma_global, n_sims=10000)
    print(f"  sigma_global = {sigma_global:.4f} (simulated CV = {verify_cv:.4f})")

    # Fit sigma_team (accounting for sigma_global's contribution)
    print(f"\nFitting sigma_team to match teammate correlation = {target_corr:.4f}...")
    sigma_team = fit_sigma_team(target_corr, sigma_global=sigma_global)
    verify_corr = simulate_teammate_correlation(sigma_team, sigma_global, n_sims=10000)
    print(f"  sigma_team = {sigma_team:.4f} (simulated correlation = {verify_corr:.4f})")

    # Fit sigma_dnf
    print(f"\nFitting sigma_dnf to match DNF overdispersion = {target_overdispersion:.4f}...")
    sigma_dnf = fit_sigma_dnf(target_overdispersion, base_dnf_rate)
    verify_od = simulate_dnf_overdispersion(sigma_dnf, base_dnf_rate, n_sims=20000)
    print(f"  sigma_dnf = {sigma_dnf:.4f} (simulated overdispersion = {verify_od:.4f})")

    result = {
        "sigma_team": round(sigma_team, 4),
        "sigma_global": round(sigma_global, 4),
        "sigma_dnf": round(sigma_dnf, 4),
    }

    print(f"\n{'='*60}")
    print(f"Calibrated correlation parameters:")
    print(f"  sigma_team   = {result['sigma_team']}")
    print(f"  sigma_global = {result['sigma_global']}")
    print(f"  sigma_dnf    = {result['sigma_dnf']}")
    print(f"{'='*60}")
    print(f"\nTo use these values, update CORRELATION_DEFAULTS in pipeline/config.py:")
    print(f'  CORRELATION_DEFAULTS = {json.dumps(result, indent=4)}')

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate correlation parameters from historical F1 data"
    )
    parser.add_argument(
        "--data", "-d",
        help="Path to historical_results.json (from download_historical.py)",
    )
    parser.add_argument(
        "--use-defaults", action="store_true",
        help="Use known F1 summary statistics instead of computing from data",
    )
    parser.add_argument(
        "--teammate-corr", type=float, default=None,
        help="Override: target teammate residual correlation",
    )
    parser.add_argument(
        "--race-cv", type=float, default=None,
        help="Override: target race-variance coefficient of variation",
    )
    parser.add_argument(
        "--dnf-overdispersion", type=float, default=None,
        help="Override: target DNF overdispersion ratio",
    )
    args = parser.parse_args()

    if args.data:
        result = calibrate_from_data(args.data)
    elif args.use_defaults or (args.teammate_corr or args.race_cv or args.dnf_overdispersion):
        # Use known statistics from F1 analysis or user overrides.
        # Default targets are rough estimates; run with --data for real values.
        target_corr = args.teammate_corr if args.teammate_corr is not None else 0.35
        target_cv = args.race_cv if args.race_cv is not None else 0.28
        target_od = args.dnf_overdispersion if args.dnf_overdispersion is not None else 1.8

        print("Using target statistics (override with --data for real calibration):")
        print(f"  Teammate correlation:    {target_corr}")
        print(f"  Race variance CV:        {target_cv}")
        print(f"  DNF overdispersion:      {target_od}")

        result = fit_to_targets(target_corr, target_cv, target_od)
    else:
        # Try to find historical data file
        default_path = os.path.join(os.path.dirname(__file__), "historical_results.json")
        if os.path.exists(default_path):
            result = calibrate_from_data(default_path)
        else:
            print("No historical data found.")
            print("Options:")
            print("  1. Download data first:")
            print("     python pipeline/download_historical.py")
            print("  2. Use estimated defaults:")
            print("     python pipeline/calibrate_correlation.py --use-defaults")
            print("  3. Provide custom targets:")
            print("     python pipeline/calibrate_correlation.py --teammate-corr 0.35 --race-cv 0.28 --dnf-overdispersion 1.8")
            sys.exit(1)


if __name__ == "__main__":
    main()
