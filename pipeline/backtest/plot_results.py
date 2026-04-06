"""
Visualization for backtest results using matplotlib.

Generates:
1. Calibration plot (predicted probability vs actual frequency)
2. Parameter sensitivity heatmaps
3. Fantasy portfolio cumulative points over season
4. Per-race metric time series
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def _ensure_matplotlib():
    """Import matplotlib, with a helpful error if missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)


def plot_calibration(results: dict, output_path: str = "pipeline/backtest_results/calibration.png"):
    """Plot calibration curve from sweep results."""
    plt = _ensure_matplotlib()

    # Use the best config's per-race data to build calibration
    configs = results.get("configs", [])
    if not configs:
        print("No configs to plot")
        return

    # Use the config with best log-likelihood
    best_config = max(configs, key=lambda c: c.get("mean_log_likelihood", -999))
    calibration = best_config.get("calibration", {})

    if not calibration:
        print("No calibration data available in results")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    if "bin_centers" in calibration and "actual_freq" in calibration:
        centers = calibration["bin_centers"]
        actual = calibration["actual_freq"]
        n_per_bin = calibration.get("n_per_bin", [1] * len(centers))

        # Filter bins with data
        mask = [n > 0 for n in n_per_bin]
        centers_f = [c for c, m in zip(centers, mask) if m]
        actual_f = [a for a, m in zip(actual, mask) if m]
        sizes = [n for n, m in zip(n_per_bin, mask) if m]

        ax.scatter(centers_f, actual_f, s=[max(20, s * 2) for s in sizes],
                   alpha=0.7, zorder=5, label="Observed")
        ax.plot(centers_f, actual_f, 'b-', alpha=0.5)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Perfect calibration")

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Actual Frequency", fontsize=12)
    ax.set_title("Model Calibration", fontsize=14)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved calibration plot to {output_path}")


def plot_parameter_heatmap(
    results: dict,
    param_x: str = "sigma_team",
    param_y: str = "sigma_global",
    metric: str = "mean_log_likelihood",
    output_path: str = "pipeline/backtest_results/heatmap.png",
):
    """Plot a 2D parameter sensitivity heatmap."""
    plt = _ensure_matplotlib()

    configs = results.get("configs", [])
    if not configs:
        print("No configs to plot")
        return

    # Extract unique values for each parameter
    x_vals = sorted(set(c["params"].get(param_x, 0) for c in configs))
    y_vals = sorted(set(c["params"].get(param_y, 0) for c in configs))

    if len(x_vals) < 2 or len(y_vals) < 2:
        print(f"Not enough variation in {param_x} or {param_y} for heatmap")
        return

    # Build matrix (average metric across other params for each x,y combo)
    matrix = np.full((len(y_vals), len(x_vals)), np.nan)
    counts = np.zeros_like(matrix)

    for config in configs:
        x_val = config["params"].get(param_x)
        y_val = config["params"].get(param_y)
        score = config.get(metric, 0)

        if x_val in x_vals and y_val in y_vals:
            xi = x_vals.index(x_val)
            yi = y_vals.index(y_val)
            if np.isnan(matrix[yi, xi]):
                matrix[yi, xi] = 0
            matrix[yi, xi] += score
            counts[yi, xi] += 1

    # Average
    mask = counts > 0
    matrix[mask] /= counts[mask]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"{v:.3g}" for v in x_vals])
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{v:.3g}" for v in y_vals])
    ax.set_xlabel(param_x, fontsize=12)
    ax.set_ylabel(param_y, fontsize=12)
    ax.set_title(f"{metric} by {param_x} × {param_y}", fontsize=14)

    plt.colorbar(im, ax=ax, label=metric)

    # Add text annotations
    for yi in range(len(y_vals)):
        for xi in range(len(x_vals)):
            if not np.isnan(matrix[yi, xi]):
                ax.text(xi, yi, f"{matrix[yi, xi]:.4f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if matrix[yi, xi] < np.nanmedian(matrix) else "black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap to {output_path}")


def plot_fantasy_cumulative(
    results: dict,
    output_path: str = "pipeline/backtest_results/fantasy_cumulative.png",
):
    """Plot cumulative fantasy portfolio points vs optimal over time."""
    plt = _ensure_matplotlib()

    configs = results.get("configs", [])
    if not configs:
        return

    # Use the best config by fantasy efficiency
    best = max(configs, key=lambda c: c.get("fantasy_efficiency", 0))
    per_race = best.get("per_race", [])
    if not per_race:
        print("No per-race data for cumulative plot")
        return

    # Sort by season + round
    per_race = sorted(per_race, key=lambda r: (r.get("season", 0), r.get("round", 0)))

    labels = [f"S{r['season']%100}R{r['round']}" for r in per_race]
    portfolio = np.cumsum([r["portfolio_points"] for r in per_race])
    optimal = np.cumsum([r["optimal_points"] for r in per_race])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(len(portfolio)), portfolio, 'b-o', markersize=4,
            label=f"Model picks ({portfolio[-1]:.0f} total)")
    ax.plot(range(len(optimal)), optimal, 'g-o', markersize=4,
            label=f"Hindsight optimal ({optimal[-1]:.0f} total)")
    ax.fill_between(range(len(portfolio)), portfolio, optimal, alpha=0.1, color='red')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Race", fontsize=12)
    ax.set_ylabel("Cumulative Fantasy Points", fontsize=12)
    ax.set_title("Fantasy Portfolio: Model vs Optimal", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    efficiency = portfolio[-1] / optimal[-1] if optimal[-1] > 0 else 0
    ax.text(0.02, 0.98, f"Efficiency: {efficiency:.1%}",
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved fantasy cumulative plot to {output_path}")


def plot_per_race_metrics(
    results: dict,
    output_path: str = "pipeline/backtest_results/per_race_metrics.png",
):
    """Plot per-race metric time series for the best configuration."""
    plt = _ensure_matplotlib()

    configs = results.get("configs", [])
    if not configs:
        return

    best = max(configs, key=lambda c: c.get("mean_log_likelihood", -999))
    per_race = best.get("per_race", [])
    if not per_race:
        return

    per_race = sorted(per_race, key=lambda r: (r.get("season", 0), r.get("round", 0)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    labels = [f"S{r['season']%100}R{r['round']}" for r in per_race]
    x = range(len(per_race))

    # Log-likelihood
    ax = axes[0, 0]
    vals = [r["log_likelihood"] for r in per_race]
    ax.bar(x, vals, color='steelblue', alpha=0.7)
    ax.axhline(np.mean(vals), color='red', linestyle='--', alpha=0.5, label=f"Mean: {np.mean(vals):.2f}")
    ax.set_title("Log-Likelihood per Race")
    ax.legend()

    # Brier (win)
    ax = axes[0, 1]
    vals = [r["brier_win"] for r in per_race]
    ax.bar(x, vals, color='coral', alpha=0.7)
    ax.axhline(np.mean(vals), color='red', linestyle='--', alpha=0.5, label=f"Mean: {np.mean(vals):.4f}")
    ax.set_title("Brier Score (Win) per Race")
    ax.legend()

    # EP MAE
    ax = axes[1, 0]
    vals = [r["ep_mae"] for r in per_race]
    ax.bar(x, vals, color='mediumpurple', alpha=0.7)
    ax.axhline(np.mean(vals), color='red', linestyle='--', alpha=0.5, label=f"Mean: {np.mean(vals):.2f}")
    ax.set_title("Expected Points MAE per Race")
    ax.legend()

    # Spearman
    ax = axes[1, 1]
    vals = [r["spearman"] for r in per_race]
    ax.bar(x, vals, color='seagreen', alpha=0.7)
    ax.axhline(np.mean(vals), color='red', linestyle='--', alpha=0.5, label=f"Mean: {np.mean(vals):.3f}")
    ax.set_title("Spearman Correlation per Race")
    ax.legend()

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Race Backtest Metrics (Best Config)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved per-race metrics plot to {output_path}")


def plot_all(results_path: str, output_dir: str = None):
    """Generate all plots from a sweep results file."""
    with open(results_path) as f:
        results = json.load(f)

    if output_dir is None:
        output_dir = str(Path(results_path).parent)

    plot_calibration(results, f"{output_dir}/calibration.png")
    plot_parameter_heatmap(results, "sigma_team", "sigma_global",
                           output_path=f"{output_dir}/heatmap_sigma.png")
    plot_parameter_heatmap(results, "team_reg", "sigma_global",
                           metric="fantasy_efficiency",
                           output_path=f"{output_dir}/heatmap_reg_sigma.png")
    plot_fantasy_cumulative(results, f"{output_dir}/fantasy_cumulative.png")
    plot_per_race_metrics(results, f"{output_dir}/per_race_metrics.png")
