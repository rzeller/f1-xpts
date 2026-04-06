"""
Evaluation metrics for backtesting the F1 expected points model.

All metric functions take model predictions (position distributions, expected
points) and actual race results, returning scalar scores.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RaceMetrics:
    """Metrics for a single backtested race."""
    race_id: str                     # e.g. "2024_01_australian-gp"
    season: int
    round_num: int
    race_name: str
    n_drivers: int

    # Core metrics
    log_likelihood: float            # Log-likelihood of actual finishing order
    brier_win: float                 # Brier score for win predictions
    brier_podium: float              # Brier score for podium predictions
    brier_top6: float                # Brier score for top-6 predictions
    brier_top10: float               # Brier score for top-10 predictions

    # Points prediction accuracy
    ep_mae: float                    # Mean absolute error: predicted vs actual points
    ep_rmse: float                   # Root mean squared error

    # Rank correlation
    spearman_rho: float              # Spearman rank correlation
    kendall_tau: float               # Kendall tau rank correlation

    # Fantasy portfolio
    portfolio_points: float          # Actual points from model's top-5 picks
    optimal_points: float            # Hindsight-optimal top-5 points
    portfolio_drivers: List[str] = field(default_factory=list)

    # DNF prediction
    dnf_brier: float = 0.0           # Brier score for DNF predictions

    # Fit info
    fit_loss: float = 0.0
    fit_converged: bool = True


@dataclass
class BacktestResult:
    """Aggregated results across all backtested races."""
    params: dict
    race_metrics: List[RaceMetrics]
    n_races: int = 0

    # Aggregated metrics (computed from race_metrics)
    mean_log_likelihood: float = 0.0
    mean_brier_win: float = 0.0
    mean_brier_podium: float = 0.0
    mean_brier_top6: float = 0.0
    mean_brier_top10: float = 0.0
    mean_ep_mae: float = 0.0
    mean_ep_rmse: float = 0.0
    mean_spearman: float = 0.0
    mean_kendall: float = 0.0

    # Fantasy
    total_portfolio_points: float = 0.0
    total_optimal_points: float = 0.0
    fantasy_efficiency: float = 0.0  # portfolio / optimal

    # Calibration data (filled by compute_calibration)
    calibration: dict = field(default_factory=dict)

    def aggregate(self):
        """Compute aggregate metrics from per-race metrics."""
        if not self.race_metrics:
            return
        n = len(self.race_metrics)
        self.n_races = n
        self.mean_log_likelihood = np.mean([m.log_likelihood for m in self.race_metrics])
        self.mean_brier_win = np.mean([m.brier_win for m in self.race_metrics])
        self.mean_brier_podium = np.mean([m.brier_podium for m in self.race_metrics])
        self.mean_brier_top6 = np.mean([m.brier_top6 for m in self.race_metrics])
        self.mean_brier_top10 = np.mean([m.brier_top10 for m in self.race_metrics])
        self.mean_ep_mae = np.mean([m.ep_mae for m in self.race_metrics])
        self.mean_ep_rmse = np.mean([m.ep_rmse for m in self.race_metrics])
        self.mean_spearman = np.mean([m.spearman_rho for m in self.race_metrics])
        self.mean_kendall = np.mean([m.kendall_tau for m in self.race_metrics])

        self.total_portfolio_points = sum(m.portfolio_points for m in self.race_metrics)
        self.total_optimal_points = sum(m.optimal_points for m in self.race_metrics)
        if self.total_optimal_points > 0:
            self.fantasy_efficiency = self.total_portfolio_points / self.total_optimal_points
        else:
            self.fantasy_efficiency = 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "params": self.params,
            "n_races": self.n_races,
            "mean_log_likelihood": round(self.mean_log_likelihood, 4),
            "mean_brier_win": round(self.mean_brier_win, 6),
            "mean_brier_podium": round(self.mean_brier_podium, 6),
            "mean_brier_top6": round(self.mean_brier_top6, 6),
            "mean_brier_top10": round(self.mean_brier_top10, 6),
            "mean_ep_mae": round(self.mean_ep_mae, 4),
            "mean_ep_rmse": round(self.mean_ep_rmse, 4),
            "mean_spearman": round(self.mean_spearman, 4),
            "mean_kendall": round(self.mean_kendall, 4),
            "total_portfolio_points": round(self.total_portfolio_points, 2),
            "total_optimal_points": round(self.total_optimal_points, 2),
            "fantasy_efficiency": round(self.fantasy_efficiency, 4),
            "calibration": self.calibration,
            "per_race": [
                {
                    "race_id": m.race_id,
                    "season": m.season,
                    "round": m.round_num,
                    "name": m.race_name,
                    "log_likelihood": round(m.log_likelihood, 4),
                    "brier_win": round(m.brier_win, 6),
                    "brier_podium": round(m.brier_podium, 6),
                    "ep_mae": round(m.ep_mae, 4),
                    "spearman": round(m.spearman_rho, 4),
                    "portfolio_points": round(m.portfolio_points, 2),
                    "optimal_points": round(m.optimal_points, 2),
                    "portfolio_drivers": m.portfolio_drivers,
                }
                for m in self.race_metrics
            ],
        }


def log_likelihood(
    pos_probs: np.ndarray,
    actual_positions: Dict[int, int],
    actual_dnfs: Dict[int, bool],
) -> float:
    """
    Compute log-likelihood of the actual result under the model.

    Parameters
    ----------
    pos_probs : (n_drivers, n_drivers+1) array from simulate_races
    actual_positions : {driver_idx: finishing_position (1-based)}
    actual_dnfs : {driver_idx: True if DNF}

    Returns
    -------
    Total log-likelihood (higher = better, always negative)
    """
    eps = 1e-10  # Avoid log(0)
    ll = 0.0
    for driver_idx in actual_positions:
        if actual_dnfs.get(driver_idx, False):
            # DNF: probability is in the last column
            p = pos_probs[driver_idx, -1]
        else:
            pos = actual_positions[driver_idx]
            if pos - 1 < pos_probs.shape[1] - 1:
                p = pos_probs[driver_idx, pos - 1]
            else:
                p = eps
        ll += np.log(max(p, eps))
    return float(ll)


def brier_score(
    pos_probs: np.ndarray,
    actual_positions: Dict[int, int],
    actual_dnfs: Dict[int, bool],
    cutoff: int,
) -> float:
    """
    Compute Brier score for a binary outcome (e.g., finish in top-N).

    Parameters
    ----------
    cutoff : position cutoff (1 for win, 3 for podium, 6 for top-6, 10 for top-10)

    Returns
    -------
    Mean Brier score (lower = better, range [0, 1])
    """
    scores = []
    for driver_idx in actual_positions:
        if actual_dnfs.get(driver_idx, False):
            actual = 0  # DNF never in top-N
        else:
            actual = 1 if actual_positions[driver_idx] <= cutoff else 0

        # Model's predicted probability of finishing in top-cutoff
        predicted = float(pos_probs[driver_idx, :cutoff].sum())
        scores.append((predicted - actual) ** 2)

    return float(np.mean(scores)) if scores else 0.0


def dnf_brier_score(
    pos_probs: np.ndarray,
    actual_dnfs: Dict[int, bool],
) -> float:
    """Brier score for DNF predictions."""
    scores = []
    for driver_idx, is_dnf in actual_dnfs.items():
        predicted = float(pos_probs[driver_idx, -1])
        actual = 1 if is_dnf else 0
        scores.append((predicted - actual) ** 2)
    return float(np.mean(scores)) if scores else 0.0


def expected_points_error(
    predicted_ep: Dict[int, float],
    actual_points: Dict[int, float],
) -> Tuple[float, float]:
    """
    Compute MAE and RMSE between predicted and actual fantasy points.

    Returns (mae, rmse)
    """
    errors = []
    for driver_idx in actual_points:
        if driver_idx in predicted_ep:
            errors.append(predicted_ep[driver_idx] - actual_points[driver_idx])

    if not errors:
        return 0.0, 0.0
    errors = np.array(errors)
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return mae, rmse


def rank_correlation(
    predicted_ep: Dict[int, float],
    actual_positions: Dict[int, int],
) -> Tuple[float, float]:
    """
    Compute Spearman and Kendall rank correlations between
    predicted expected points ranking and actual finishing order.

    Returns (spearman_rho, kendall_tau)
    """
    # Build aligned arrays
    common = sorted(set(predicted_ep.keys()) & set(actual_positions.keys()))
    if len(common) < 3:
        return 0.0, 0.0

    # Predicted ranking: higher EP = better = lower rank number
    pred_values = [predicted_ep[i] for i in common]
    actual_pos = [actual_positions[i] for i in common]

    # Convert predicted EP to ranks (descending: highest EP = rank 1)
    pred_order = np.argsort(-np.array(pred_values))
    pred_ranks = np.empty_like(pred_order)
    pred_ranks[pred_order] = np.arange(1, len(common) + 1)

    actual_ranks = np.array(actual_pos, dtype=float)

    # Spearman: correlation of ranks
    n = len(common)
    d = pred_ranks - actual_ranks
    spearman = 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1))

    # Kendall tau: count concordant/discordant pairs
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            pred_diff = pred_ranks[i] - pred_ranks[j]
            actual_diff = actual_ranks[i] - actual_ranks[j]
            if pred_diff * actual_diff > 0:
                concordant += 1
            elif pred_diff * actual_diff < 0:
                discordant += 1
    total_pairs = n * (n - 1) / 2
    kendall = (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0

    return float(spearman), float(kendall)


def fantasy_portfolio(
    predicted_ep: Dict[int, float],
    actual_points: Dict[int, float],
    n_picks: int = 5,
    drivers_list: list = None,
) -> Tuple[float, float, List[str]]:
    """
    Simulate the fantasy portfolio strategy: pick top-N drivers by predicted EP.

    Returns (portfolio_points, optimal_points, picked_driver_abbrs)
    """
    # Sort drivers by predicted EP (descending), pick top n_picks
    ranked = sorted(predicted_ep.items(), key=lambda x: -x[1])
    picked = [idx for idx, _ in ranked[:n_picks]]

    # Actual points for picked drivers
    portfolio_pts = sum(actual_points.get(idx, 0) for idx in picked)

    # Hindsight-optimal: pick the n_picks drivers who scored most
    optimal_ranked = sorted(actual_points.items(), key=lambda x: -x[1])
    optimal_pts = sum(pts for _, pts in optimal_ranked[:n_picks])

    # Driver abbreviations for reporting
    picked_abbrs = []
    if drivers_list:
        for idx in picked:
            if idx < len(drivers_list):
                picked_abbrs.append(drivers_list[idx].get("abbr", f"D{idx}"))
    else:
        picked_abbrs = [f"D{idx}" for idx in picked]

    return float(portfolio_pts), float(optimal_pts), picked_abbrs


def compute_calibration(
    all_predictions: List[Tuple[float, int]],
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration data from a list of (predicted_probability, actual_outcome) pairs.

    Parameters
    ----------
    all_predictions : list of (predicted_prob, actual_indicator) where
        actual_indicator is 0 or 1
    n_bins : number of bins

    Returns
    -------
    dict with keys: bin_edges, bin_centers, actual_freq, n_per_bin, predicted_mean
    """
    if not all_predictions:
        return {}

    preds = np.array([p for p, _ in all_predictions])
    actuals = np.array([a for _, a in all_predictions])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    actual_freq = []
    predicted_mean = []
    n_per_bin = []

    for i in range(n_bins):
        mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
        if i == n_bins - 1:  # Include upper edge in last bin
            mask = (preds >= bin_edges[i]) & (preds <= bin_edges[i + 1])
        count = mask.sum()
        n_per_bin.append(int(count))
        bin_centers.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))
        if count > 0:
            actual_freq.append(float(actuals[mask].mean()))
            predicted_mean.append(float(preds[mask].mean()))
        else:
            actual_freq.append(0.0)
            predicted_mean.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))

    return {
        "bin_edges": [round(e, 4) for e in bin_edges.tolist()],
        "bin_centers": [round(c, 4) for c in bin_centers],
        "actual_freq": [round(f, 4) for f in actual_freq],
        "predicted_mean": [round(m, 4) for m in predicted_mean],
        "n_per_bin": n_per_bin,
    }


def compute_race_metrics(
    race_id: str,
    season: int,
    round_num: int,
    race_name: str,
    pos_probs: np.ndarray,
    predicted_ep: Dict[int, float],
    actual_positions: Dict[int, int],
    actual_dnfs: Dict[int, bool],
    actual_points: Dict[int, float],
    drivers_list: list = None,
    fit_loss: float = 0.0,
    fit_converged: bool = True,
) -> RaceMetrics:
    """Compute all metrics for a single race."""
    ll = log_likelihood(pos_probs, actual_positions, actual_dnfs)
    b_win = brier_score(pos_probs, actual_positions, actual_dnfs, cutoff=1)
    b_pod = brier_score(pos_probs, actual_positions, actual_dnfs, cutoff=3)
    b_t6 = brier_score(pos_probs, actual_positions, actual_dnfs, cutoff=6)
    b_t10 = brier_score(pos_probs, actual_positions, actual_dnfs, cutoff=10)
    b_dnf = dnf_brier_score(pos_probs, actual_dnfs)
    mae, rmse = expected_points_error(predicted_ep, actual_points)
    spearman, kendall = rank_correlation(predicted_ep, actual_positions)
    port_pts, opt_pts, picked = fantasy_portfolio(
        predicted_ep, actual_points, drivers_list=drivers_list
    )

    return RaceMetrics(
        race_id=race_id,
        season=season,
        round_num=round_num,
        race_name=race_name,
        n_drivers=pos_probs.shape[0],
        log_likelihood=ll,
        brier_win=b_win,
        brier_podium=b_pod,
        brier_top6=b_t6,
        brier_top10=b_t10,
        ep_mae=mae,
        ep_rmse=rmse,
        spearman_rho=spearman,
        kendall_tau=kendall,
        portfolio_points=port_pts,
        optimal_points=opt_pts,
        portfolio_drivers=picked,
        dnf_brier=b_dnf,
        fit_loss=fit_loss,
        fit_converged=fit_converged,
    )
