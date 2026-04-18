#!/usr/bin/env python3
"""
CLI entry point for the F1 backtesting framework.

Usage:
    # Fetch historical odds (requires API key)
    python -m pipeline.backtest.cli fetch --api-key KEY

    # Run single backtest with default params
    python -m pipeline.backtest.cli run

    # Run parameter sweep
    python -m pipeline.backtest.cli sweep --n-workers 4

    # Run cross-validation
    python -m pipeline.backtest.cli cv

    # Fine-tune parameters with Bayesian optimization
    python -m pipeline.backtest.cli optimize

    # Generate plots from sweep results
    python -m pipeline.backtest.cli plot --results pipeline/backtest_results/sweep_latest.json
"""

import argparse
import json
import os
import sys

# Add pipeline dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def cmd_fetch(args):
    """Fetch historical odds from The Odds API."""
    from backtest.fetch_historical_odds import run_fetch

    api_key = args.api_key or os.environ.get("ODDS_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: --api-key or ODDS_API_KEY env var required")
        sys.exit(1)

    run_fetch(
        api_key=api_key or "",
        output_dir=args.odds_dir,
        historical_results_path=args.results,
        seasons=args.seasons,
        max_priority=args.max_priority,
        regions=args.regions,
        dry_run=args.dry_run,
        budget_limit=args.budget,
    )


def cmd_run(args):
    """Run a single backtest with specified parameters."""
    from backtest.runner import run_backtest, DEFAULT_PARAMS

    params = DEFAULT_PARAMS.copy()
    if args.devig_method:
        params["devig_method"] = args.devig_method
    if args.sigma_team is not None:
        params["sigma_team"] = args.sigma_team
    if args.sigma_global is not None:
        params["sigma_global"] = args.sigma_global
    if args.sigma_dnf is not None:
        params["sigma_dnf"] = args.sigma_dnf
    if args.team_reg is not None:
        params["team_reg"] = args.team_reg
    if args.fit_sims is not None:
        params["n_fit_sims"] = args.fit_sims
    if args.final_sims is not None:
        params["n_final_sims"] = args.final_sims

    result = run_backtest(
        historical_results_path=args.results,
        odds_dir=args.odds_dir,
        params=params,
        cache_dir=args.cache_dir,
        seasons=args.seasons,
        time_point=args.time_point,
        verbose=True,
    )

    # Save result
    output = result.to_dict()
    output_path = os.path.join(args.output_dir, "backtest_single.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


def cmd_sweep(args):
    """Run parameter grid search."""
    from backtest.sweep import run_grid_search

    grid = None
    if args.grid_file:
        with open(args.grid_file) as f:
            grid = json.load(f)

    run_grid_search(
        grid=grid,
        historical_results_path=args.results,
        odds_dir=args.odds_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        seasons=args.seasons,
        n_workers=args.n_workers,
        verbose=True,
    )


def cmd_cv(args):
    """Run leave-one-season-out cross-validation."""
    from backtest.sweep import run_cross_validation
    from backtest.runner import DEFAULT_PARAMS

    params = DEFAULT_PARAMS.copy()
    if args.params_file:
        with open(args.params_file) as f:
            params.update(json.load(f))

    result = run_cross_validation(
        params=params,
        historical_results_path=args.results,
        odds_dir=args.odds_dir,
        cache_dir=args.cache_dir,
        verbose=True,
    )

    output_path = os.path.join(args.output_dir, "cv_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nCV results saved to {output_path}")


def cmd_optimize(args):
    """Run Bayesian optimization to fine-tune parameters."""
    from backtest.sweep import run_bayesian_optimization
    from backtest.runner import DEFAULT_PARAMS

    params = DEFAULT_PARAMS.copy()
    if args.base_params_file:
        with open(args.base_params_file) as f:
            params.update(json.load(f))

    optimize_params = args.optimize or ["sigma_team", "sigma_global", "sigma_dnf"]
    bounds = {
        "sigma_team": (0.1, 1.5),
        "sigma_global": (0.1, 3.0),
        "sigma_dnf": (0.05, 1.0),
        "team_reg": (0.001, 0.1),
        "smoothness_reg": (0.001, 0.05),
    }

    result = run_bayesian_optimization(
        base_params=params,
        optimize_params=optimize_params,
        bounds=bounds,
        n_evals=args.n_evals,
        metric=args.metric,
        higher_better=args.metric in ["mean_log_likelihood", "mean_spearman", "fantasy_efficiency"],
        historical_results_path=args.results,
        odds_dir=args.odds_dir,
        cache_dir=args.cache_dir,
        seasons=args.seasons,
        verbose=True,
    )

    output_path = os.path.join(args.output_dir, "optimization_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nOptimization results saved to {output_path}")
    print(f"Best params: {json.dumps(result['best_params'], indent=2)}")


def cmd_plot(args):
    """Generate plots from sweep results."""
    from backtest.plot_results import plot_all
    plot_all(args.results_file, output_dir=args.output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="F1 Expected Points Backtesting Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pipeline.backtest.cli fetch --api-key KEY --seasons 2022 2023 2024
  python -m pipeline.backtest.cli run --seasons 2024
  python -m pipeline.backtest.cli sweep --n-workers 4
  python -m pipeline.backtest.cli cv
  python -m pipeline.backtest.cli optimize --n-evals 50
  python -m pipeline.backtest.cli plot --results pipeline/backtest_results/sweep_latest.json
        """,
    )

    # Common arguments
    parser.add_argument("--results", default="pipeline/historical_results.json",
                        help="Path to historical results JSON")
    parser.add_argument("--odds-dir", default="pipeline/historical_odds",
                        help="Directory for historical odds files")
    parser.add_argument("--cache-dir", default="pipeline/backtest_cache",
                        help="Directory for model cache")
    parser.add_argument("--output-dir", default="pipeline/backtest_results",
                        help="Directory for output files")
    parser.add_argument("--seasons", nargs="+", type=int,
                        help="Filter to specific seasons")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # fetch command
    p_fetch = subparsers.add_parser("fetch", help="Fetch historical odds from The Odds API")
    p_fetch.add_argument("--api-key", "-k", help="The Odds API key")
    p_fetch.add_argument("--max-priority", "-p", type=int, default=6,
                         help="Max time point priority to fetch, 1-6 (default: 6)")
    p_fetch.add_argument("--regions", default="us,uk,eu,au",
                         help="Bookmaker regions (default: us,uk,eu,au)")
    p_fetch.add_argument("--budget", type=int, default=20000,
                         help="API credit budget limit (default: 20000)")
    p_fetch.add_argument("--dry-run", action="store_true")

    # run command
    p_run = subparsers.add_parser("run", help="Run single backtest")
    p_run.add_argument("--time-point",
                       help="Which time point to use (e.g. before_qualifying, after_fp2)")
    p_run.add_argument("--devig-method", choices=["shin", "multiplicative", "power"])
    p_run.add_argument("--sigma-team", type=float)
    p_run.add_argument("--sigma-global", type=float)
    p_run.add_argument("--sigma-dnf", type=float)
    p_run.add_argument("--team-reg", type=float)
    p_run.add_argument("--fit-sims", type=int)
    p_run.add_argument("--final-sims", type=int)

    # sweep command
    p_sweep = subparsers.add_parser("sweep", help="Run parameter grid search")
    p_sweep.add_argument("--n-workers", type=int, default=1,
                         help="Parallel workers (default: 1)")
    p_sweep.add_argument("--grid-file", help="JSON file with custom grid")

    # cv command
    p_cv = subparsers.add_parser("cv", help="Run cross-validation")
    p_cv.add_argument("--params-file", help="JSON file with params to validate")

    # optimize command
    p_opt = subparsers.add_parser("optimize", help="Bayesian optimization")
    p_opt.add_argument("--optimize", nargs="+",
                       help="Parameters to optimize (default: sigma_team sigma_global sigma_dnf)")
    p_opt.add_argument("--n-evals", type=int, default=50,
                       help="Max function evaluations (default: 50)")
    p_opt.add_argument("--metric", default="mean_log_likelihood",
                       help="Metric to optimize (default: mean_log_likelihood)")
    p_opt.add_argument("--base-params-file", help="JSON file with base params")

    # plot command
    p_plot = subparsers.add_parser("plot", help="Generate plots from results")
    p_plot.add_argument("--results-file",
                        default="pipeline/backtest_results/sweep_latest.json",
                        help="Path to sweep results JSON")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "fetch": cmd_fetch,
        "run": cmd_run,
        "sweep": cmd_sweep,
        "cv": cmd_cv,
        "optimize": cmd_optimize,
        "plot": cmd_plot,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
