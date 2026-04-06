"""
Core backtest runner: runs the full pipeline on historical races and evaluates.

For each race with available odds:
1. Load odds from historical_odds/{file}.json
2. Build dynamic driver config
3. Devig odds
4. Fit Plackett-Luce model
5. Run final simulation → position distributions
6. Compare against actual results → compute metrics
"""

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add pipeline dir to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest.config_historical import build_race_config, get_actual_results, RaceConfig
from backtest.metrics import (
    BacktestResult,
    RaceMetrics,
    compute_calibration,
    compute_race_metrics,
)
from config import RACE_POINTS, DNF_PENALTY, CORRELATION_DEFAULTS
from devig import devig_market, american_to_implied
from odds_fetcher import process_odds_to_fair_probs
from plackett_luce import (
    compute_expected_points,
    fit_plackett_luce,
    simulate_races,
)


# Default backtest parameters
DEFAULT_PARAMS = {
    "devig_method": "shin",
    "sigma_team": CORRELATION_DEFAULTS["sigma_team"],
    "sigma_global": CORRELATION_DEFAULTS["sigma_global"],
    "sigma_dnf": CORRELATION_DEFAULTS["sigma_dnf"],
    "team_reg": 0.02,
    "smoothness_reg": 0.005,
    "n_fit_sims": 10000,
    "n_final_sims": 50000,
    "optimizer_method": "Powell",
}


def _cache_key(race_id: str, params: dict) -> str:
    """
    Generate a cache key for a fitted model.

    Only includes parameters that affect fitting (not simulation-only params).
    """
    fit_params = {
        "devig_method": params.get("devig_method", "shin"),
        "team_reg": params.get("team_reg", 0.02),
        "smoothness_reg": params.get("smoothness_reg", 0.005),
        "n_fit_sims": params.get("n_fit_sims", 10000),
        "optimizer_method": params.get("optimizer_method", "Powell"),
    }
    key_str = f"{race_id}|{json.dumps(fit_params, sort_keys=True)}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def _load_cached_fit(cache_dir: str, cache_key: str) -> Optional[Tuple]:
    """Load a cached (log_lambdas, p_dnfs, fit_info) if available."""
    if not cache_dir:
        return None
    cache_path = Path(cache_dir) / f"{cache_key}.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _save_cached_fit(cache_dir: str, cache_key: str, data: Tuple):
    """Save fitted model to cache."""
    if not cache_dir:
        return
    cache_path = Path(cache_dir) / f"{cache_key}.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)


def load_odds_file(filepath: str) -> Optional[dict]:
    """Load a historical odds file. Returns None if no data."""
    with open(filepath) as f:
        data = json.load(f)
    if data.get("no_data", False):
        return None
    if not data.get("markets"):
        return None
    return data


def run_single_race(
    race_results: dict,
    odds_data: dict,
    params: dict,
    cache_dir: str = None,
    verbose: bool = True,
) -> Optional[RaceMetrics]:
    """
    Run the backtest pipeline for a single race.

    Parameters
    ----------
    race_results : dict from historical_results.json
    odds_data : dict from historical odds file
    params : pipeline parameters
    cache_dir : directory for caching fitted models
    verbose : print progress

    Returns
    -------
    RaceMetrics or None if the race couldn't be processed
    """
    season = race_results["season"]
    rnd = race_results["round"]
    name = race_results["name"]
    race_id = f"{season}_{rnd:02d}"

    # Build dynamic driver config from results
    config = build_race_config(race_results)
    n_drivers = config.n_drivers

    if verbose:
        print(f"  [{race_id}] {name}: {n_drivers} drivers, "
              f"{len(odds_data.get('markets', {}))} markets")

    # Process odds through devigging
    raw_odds = odds_data.get("markets", {})
    devig_method = params.get("devig_method", "shin")

    observed_probs = process_odds_to_fair_probs(
        raw_odds,
        devig_method=devig_method,
        driver_name_map=config.driver_name_map,
    )

    if not observed_probs:
        if verbose:
            print(f"    SKIP: no observed probabilities")
        return None

    # Ensure we have at least win probabilities
    if "win" not in observed_probs:
        if verbose:
            print(f"    SKIP: no win market")
        return None

    # Fill missing DNF probs with defaults
    if "dnf" not in observed_probs:
        observed_probs["dnf"] = {i: 0.10 for i in range(n_drivers)}

    team_indices = np.array(config.team_indices)

    # Check cache
    ck = _cache_key(race_id, params)
    cached = _load_cached_fit(cache_dir, ck)

    if cached is not None:
        log_lambdas, p_dnfs, fit_info = cached
        if verbose:
            print(f"    Using cached fit (loss={fit_info.get('loss', '?'):.6f})")
    else:
        # Fit Plackett-Luce model
        correlation = {
            "sigma_team": params.get("sigma_team", CORRELATION_DEFAULTS["sigma_team"]),
            "sigma_global": params.get("sigma_global", CORRELATION_DEFAULTS["sigma_global"]),
            "sigma_dnf": params.get("sigma_dnf", CORRELATION_DEFAULTS["sigma_dnf"]),
        }

        try:
            log_lambdas, p_dnfs, fit_info = fit_plackett_luce(
                observed_probs=observed_probs,
                team_indices=team_indices,
                n_sims=params.get("n_fit_sims", 10000),
                method=params.get("optimizer_method", "Powell"),
                team_reg=params.get("team_reg", 0.02),
                smoothness_reg=params.get("smoothness_reg", 0.005),
                correlation=correlation,
                drivers_list=config.drivers,
            )
        except Exception as e:
            if verbose:
                print(f"    FIT ERROR: {e}")
            return None

        # Cache the fit
        _save_cached_fit(cache_dir, ck, (log_lambdas, p_dnfs, fit_info))

        if verbose:
            print(f"    Fit: loss={fit_info.get('loss', 0):.6f}, "
                  f"converged={fit_info.get('success', False)}")

    # Run final simulation
    correlation = {
        "sigma_team": params.get("sigma_team", CORRELATION_DEFAULTS["sigma_team"]),
        "sigma_global": params.get("sigma_global", CORRELATION_DEFAULTS["sigma_global"]),
        "sigma_dnf": params.get("sigma_dnf", CORRELATION_DEFAULTS["sigma_dnf"]),
    }

    pos_probs = simulate_races(
        log_lambdas, p_dnfs,
        n_sims=params.get("n_final_sims", 50000),
        seed=12345,
        team_indices=team_indices,
        correlation=correlation,
    )

    # Compute predicted expected points per driver
    predicted_ep = {}
    for i in range(n_drivers):
        predicted_ep[i] = compute_expected_points(
            pos_probs[i], RACE_POINTS, DNF_PENALTY
        )

    # Get actual results
    actuals = get_actual_results(race_results, config)

    # Compute metrics
    metrics = compute_race_metrics(
        race_id=race_id,
        season=season,
        round_num=rnd,
        race_name=name,
        pos_probs=pos_probs,
        predicted_ep=predicted_ep,
        actual_positions=actuals["positions"],
        actual_dnfs=actuals["dnfs"],
        actual_points=actuals["points"],
        drivers_list=config.drivers,
        fit_loss=fit_info.get("loss", 0.0),
        fit_converged=fit_info.get("success", True),
    )

    if verbose:
        print(f"    LL={metrics.log_likelihood:.2f}, "
              f"Brier(win)={metrics.brier_win:.4f}, "
              f"MAE={metrics.ep_mae:.2f}, "
              f"Portfolio={metrics.portfolio_points:.0f}/{metrics.optimal_points:.0f}")

    return metrics


def run_backtest(
    historical_results_path: str = "pipeline/historical_results.json",
    odds_dir: str = "pipeline/historical_odds",
    params: dict = None,
    cache_dir: str = "pipeline/backtest_cache",
    seasons: Optional[List[int]] = None,
    verbose: bool = True,
) -> BacktestResult:
    """
    Run the full backtest across all historical races with odds data.

    Parameters
    ----------
    historical_results_path : path to historical_results.json
    odds_dir : directory containing per-race odds files
    params : pipeline parameters (uses DEFAULT_PARAMS if None)
    cache_dir : directory for caching fitted models (None to disable)
    seasons : filter to specific seasons (None for all)
    verbose : print progress

    Returns
    -------
    BacktestResult with per-race and aggregated metrics
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Load historical results
    with open(historical_results_path) as f:
        all_results = json.load(f)["races"]

    if seasons:
        all_results = [r for r in all_results if r["season"] in seasons]

    # Find matching odds files
    odds_path = Path(odds_dir)
    odds_files = {f.stem: f for f in odds_path.glob("*.json")}

    if verbose:
        print(f"Backtest configuration:")
        print(f"  Races: {len(all_results)}")
        print(f"  Odds files: {len(odds_files)}")
        print(f"  Params: {json.dumps(params, indent=4)}")
        print()

    race_metrics = []
    calibration_data = {
        "win": [], "podium": [], "top6": [], "top10": [],
    }

    start_time = time.time()

    for race in all_results:
        season = race["season"]
        rnd = race["round"]

        # Find matching odds file
        odds_data = None
        for stem, fpath in odds_files.items():
            if stem.startswith(f"{season}_{rnd:02d}"):
                odds_data = load_odds_file(str(fpath))
                break

        if odds_data is None:
            if verbose:
                print(f"  [{season}_{rnd:02d}] {race['name']}: no odds data, skipping")
            continue

        metrics = run_single_race(
            race, odds_data, params,
            cache_dir=cache_dir,
            verbose=verbose,
        )

        if metrics is None:
            continue

        race_metrics.append(metrics)

        # Collect calibration data (predictions for all drivers across races)
        # We need to re-derive these from the simulation, so we store them during
        # the single-race computation. For now, we'll compute calibration from
        # the Brier score components in the aggregate step.

    elapsed = time.time() - start_time

    result = BacktestResult(
        params=params,
        race_metrics=race_metrics,
    )
    result.aggregate()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Backtest complete: {len(race_metrics)} races in {elapsed:.1f}s")
        print(f"  Mean LL:       {result.mean_log_likelihood:.4f}")
        print(f"  Mean Brier(W): {result.mean_brier_win:.6f}")
        print(f"  Mean Brier(P): {result.mean_brier_podium:.6f}")
        print(f"  Mean EP MAE:   {result.mean_ep_mae:.4f}")
        print(f"  Mean Spearman: {result.mean_spearman:.4f}")
        print(f"  Fantasy:       {result.total_portfolio_points:.0f} / "
              f"{result.total_optimal_points:.0f} = "
              f"{result.fantasy_efficiency:.1%}")
        print(f"{'=' * 60}")

    return result
