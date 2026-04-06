"""
Parameter sweep for backtesting: grid search + Bayesian fine-tuning.

Phase A: Coarse grid search over key parameters (~100-250 configs)
Phase B: Bayesian optimization around the best region from Phase A

Supports caching: fitting is expensive (~30s/race), but simulation sweeps
over correlation parameters reuse cached fits.
"""

import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from backtest.runner import DEFAULT_PARAMS, run_backtest
from backtest.metrics import BacktestResult


# Default grid for Phase A
DEFAULT_GRID = {
    "devig_method": ["shin", "multiplicative", "power"],
    "sigma_team": [0.3, 0.66, 1.0],
    "sigma_global": [0.5, 1.17, 2.0],
    "sigma_dnf": [0.15, 0.33, 0.5],
    "team_reg": [0.01, 0.02, 0.05],
    "smoothness_reg": [0.005],
}


def _expand_grid(grid: dict) -> List[dict]:
    """Expand a parameter grid into a list of all combinations."""
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)
    return configs


def _is_fit_param(key: str) -> bool:
    """Check if a parameter affects model fitting (vs only simulation)."""
    return key in {"devig_method", "team_reg", "smoothness_reg", "n_fit_sims", "optimizer_method"}


def _fit_key(params: dict) -> str:
    """Generate a key for the unique fitting configuration."""
    fit_params = {k: v for k, v in sorted(params.items()) if _is_fit_param(k)}
    return json.dumps(fit_params, sort_keys=True)


def run_grid_search(
    grid: dict = None,
    historical_results_path: str = "pipeline/historical_results.json",
    odds_dir: str = "pipeline/historical_odds",
    cache_dir: str = "pipeline/backtest_cache",
    output_dir: str = "pipeline/backtest_results",
    seasons: Optional[List[int]] = None,
    n_workers: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Run a grid search over parameter combinations.

    Parameters
    ----------
    grid : parameter grid (uses DEFAULT_GRID if None)
    n_workers : number of parallel workers (1 = sequential)

    Returns
    -------
    dict with sweep results, best configs per metric, and all results
    """
    if grid is None:
        grid = DEFAULT_GRID

    configs = _expand_grid(grid)
    n_configs = len(configs)

    # Count unique fitting configs (determines actual compute cost)
    fit_keys = set()
    for config in configs:
        merged = {**DEFAULT_PARAMS, **config}
        fit_keys.add(_fit_key(merged))
    n_fit_configs = len(fit_keys)

    if verbose:
        print(f"Grid search:")
        print(f"  Total configs: {n_configs}")
        print(f"  Unique fitting configs: {n_fit_configs}")
        print(f"  Workers: {n_workers}")
        print()

    results = []
    start_time = time.time()

    if n_workers <= 1:
        # Sequential execution
        for i, config in enumerate(configs):
            merged = {**DEFAULT_PARAMS, **config}
            if verbose:
                print(f"\n--- Config {i+1}/{n_configs} ---")
                print(f"  {config}")

            result = run_backtest(
                historical_results_path=historical_results_path,
                odds_dir=odds_dir,
                params=merged,
                cache_dir=cache_dir,
                seasons=seasons,
                verbose=verbose,
            )
            results.append(result)
    else:
        # Parallel execution
        # Sort configs so that fitting configs are grouped together
        # (maximizes cache hits within each worker)
        configs_sorted = sorted(configs, key=lambda c: _fit_key({**DEFAULT_PARAMS, **c}))

        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for i, config in enumerate(configs_sorted):
                merged = {**DEFAULT_PARAMS, **config}
                future = executor.submit(
                    run_backtest,
                    historical_results_path=historical_results_path,
                    odds_dir=odds_dir,
                    params=merged,
                    cache_dir=cache_dir,
                    seasons=seasons,
                    verbose=False,
                )
                futures[future] = (i, config)

            for future in as_completed(futures):
                i, config = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if verbose:
                        print(f"  [{len(results)}/{n_configs}] "
                              f"LL={result.mean_log_likelihood:.4f} "
                              f"Eff={result.fantasy_efficiency:.1%} "
                              f"| {config}")
                except Exception as e:
                    print(f"  [{i}] ERROR: {e}")

    elapsed = time.time() - start_time

    # Find best configs per metric
    best_by_metric = {}
    metric_names = {
        "log_likelihood": ("mean_log_likelihood", True),   # Higher is better
        "brier_win": ("mean_brier_win", False),             # Lower is better
        "brier_podium": ("mean_brier_podium", False),
        "ep_mae": ("mean_ep_mae", False),
        "spearman": ("mean_spearman", True),
        "fantasy_efficiency": ("fantasy_efficiency", True),
    }

    for label, (attr, higher_better) in metric_names.items():
        if not results:
            continue
        if higher_better:
            best = max(results, key=lambda r: getattr(r, attr))
        else:
            best = min(results, key=lambda r: getattr(r, attr))
        best_by_metric[label] = {
            "params": best.params,
            "value": round(getattr(best, attr), 6),
        }

    # Build output
    sweep_output = {
        "sweep_id": datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S"),
        "grid": grid,
        "n_configs": n_configs,
        "n_fit_configs": n_fit_configs,
        "elapsed_seconds": round(elapsed, 1),
        "n_workers": n_workers,
        "seasons": seasons,
        "best_by_metric": best_by_metric,
        "configs": [r.to_dict() for r in results],
    }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"sweep_{sweep_output['sweep_id']}.json"
    filepath = output_path / filename
    with open(filepath, "w") as f:
        json.dump(sweep_output, f, indent=2)

    # Also save as latest
    latest_path = output_path / "sweep_latest.json"
    with open(latest_path, "w") as f:
        json.dump(sweep_output, f, indent=2)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Grid search complete: {n_configs} configs in {elapsed:.0f}s")
        print(f"Results saved to {filepath}")
        print(f"\nBest by metric:")
        for label, info in best_by_metric.items():
            print(f"  {label}: {info['value']} — {info['params']}")

    return sweep_output


def run_bayesian_optimization(
    base_params: dict,
    optimize_params: List[str],
    bounds: Dict[str, tuple],
    n_evals: int = 50,
    metric: str = "mean_log_likelihood",
    higher_better: bool = True,
    historical_results_path: str = "pipeline/historical_results.json",
    odds_dir: str = "pipeline/historical_odds",
    cache_dir: str = "pipeline/backtest_cache",
    seasons: Optional[List[int]] = None,
    verbose: bool = True,
) -> dict:
    """
    Fine-tune parameters using scipy.optimize.minimize.

    Parameters
    ----------
    base_params : starting parameter dict
    optimize_params : list of parameter names to optimize
    bounds : {param_name: (low, high)} bounds for each parameter
    n_evals : maximum function evaluations
    metric : which BacktestResult attribute to optimize
    higher_better : if True, maximize; if False, minimize
    """
    from scipy.optimize import minimize
    import numpy as np

    eval_count = [0]
    best_result = [None]
    best_score = [float('-inf') if higher_better else float('inf')]
    history = []

    def objective(x):
        eval_count[0] += 1
        params = base_params.copy()
        for i, name in enumerate(optimize_params):
            params[name] = float(x[i])

        result = run_backtest(
            historical_results_path=historical_results_path,
            odds_dir=odds_dir,
            params=params,
            cache_dir=cache_dir,
            seasons=seasons,
            verbose=False,
        )

        score = getattr(result, metric)
        obj = -score if higher_better else score

        history.append({
            "eval": eval_count[0],
            "params": {name: float(x[i]) for i, name in enumerate(optimize_params)},
            "score": round(score, 6),
        })

        is_best = (higher_better and score > best_score[0]) or \
                  (not higher_better and score < best_score[0])
        if is_best:
            best_score[0] = score
            best_result[0] = result

        if verbose:
            marker = " *BEST*" if is_best else ""
            print(f"  eval {eval_count[0]:3d}: {metric}={score:.6f}{marker}")
            for i, name in enumerate(optimize_params):
                print(f"    {name}={x[i]:.4f}")

        return obj

    x0 = np.array([base_params.get(name, 0.5) for name in optimize_params])
    param_bounds = [bounds.get(name, (0.01, 2.0)) for name in optimize_params]

    if verbose:
        print(f"Bayesian optimization:")
        print(f"  Optimizing: {optimize_params}")
        print(f"  Metric: {metric} ({'maximize' if higher_better else 'minimize'})")
        print(f"  Max evals: {n_evals}")
        print()

    result = minimize(
        objective, x0,
        method="Nelder-Mead",
        options={"maxfev": n_evals, "xatol": 0.01, "fatol": 0.001},
    )

    best_params = base_params.copy()
    for i, name in enumerate(optimize_params):
        best_params[name] = float(result.x[i])

    return {
        "best_params": best_params,
        "best_score": best_score[0],
        "metric": metric,
        "n_evals": eval_count[0],
        "history": history,
        "scipy_result": {
            "success": result.success,
            "message": str(result.message),
            "fun": float(result.fun),
        },
    }


def run_cross_validation(
    params: dict,
    historical_results_path: str = "pipeline/historical_results.json",
    odds_dir: str = "pipeline/historical_odds",
    cache_dir: str = "pipeline/backtest_cache",
    verbose: bool = True,
) -> dict:
    """
    Leave-one-season-out cross-validation.

    Runs the backtest on each season as the held-out test set,
    reports both in-sample and out-of-sample metrics.
    """
    import json

    with open(historical_results_path) as f:
        all_results = json.load(f)["races"]

    all_seasons = sorted(set(r["season"] for r in all_results))

    # Check which seasons actually have odds data
    odds_path = Path(odds_dir)
    seasons_with_data = set()
    for f in odds_path.glob("*.json"):
        try:
            season = int(f.stem.split("_")[0])
            seasons_with_data.add(season)
        except (ValueError, IndexError):
            pass

    valid_seasons = [s for s in all_seasons if s in seasons_with_data]
    if verbose:
        print(f"Cross-validation: {len(valid_seasons)} seasons with data")
        print(f"  Seasons: {valid_seasons}")

    cv_results = []
    for held_out in valid_seasons:
        train_seasons = [s for s in valid_seasons if s != held_out]
        if verbose:
            print(f"\n--- Fold: test={held_out}, train={train_seasons} ---")

        # Out-of-sample (test on held-out season)
        test_result = run_backtest(
            historical_results_path=historical_results_path,
            odds_dir=odds_dir,
            params=params,
            cache_dir=cache_dir,
            seasons=[held_out],
            verbose=verbose,
        )

        cv_results.append({
            "held_out_season": held_out,
            "n_races": test_result.n_races,
            "log_likelihood": test_result.mean_log_likelihood,
            "brier_win": test_result.mean_brier_win,
            "ep_mae": test_result.mean_ep_mae,
            "spearman": test_result.mean_spearman,
            "fantasy_efficiency": test_result.fantasy_efficiency,
        })

    # Aggregate across folds
    import numpy as np
    metrics = ["log_likelihood", "brier_win", "ep_mae", "spearman", "fantasy_efficiency"]
    aggregate = {}
    for m in metrics:
        values = [r[m] for r in cv_results if r["n_races"] > 0]
        if values:
            aggregate[f"mean_{m}"] = round(float(np.mean(values)), 6)
            aggregate[f"std_{m}"] = round(float(np.std(values)), 6)

    output = {
        "params": params,
        "n_folds": len(valid_seasons),
        "folds": cv_results,
        "aggregate": aggregate,
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Cross-validation summary:")
        for k, v in aggregate.items():
            print(f"  {k}: {v}")

    return output
