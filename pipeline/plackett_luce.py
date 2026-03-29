"""
Plackett-Luce model: simulation, optimization, and expected points computation.

The PL model assigns each driver a strength parameter λ_i. The probability
of a finishing order is:

  P(σ) = ∏_{k=1}^{n} λ_{σ(k)} / Σ_{j=k}^{n} λ_{σ(j)}

We decompose log(λ) = μ_team + δ_driver, fit via optimization against
observed odds, then simulate to get full position distributions.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from config import RACE_POINTS, SPRINT_POINTS, DNF_PENALTY, N_DRIVERS, N_TEAMS, DRIVERS


def simulate_races(
    log_lambdas: np.ndarray,
    p_dnfs: np.ndarray,
    n_sims: int = 50000,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate races from the Plackett-Luce model with DNFs.

    Parameters
    ----------
    log_lambdas : (n_drivers,) array of log-strength parameters
    p_dnfs : (n_drivers,) array of DNF probabilities
    n_sims : number of races to simulate
    seed : random seed

    Returns
    -------
    position_probs : (n_drivers, n_drivers + 1) array where
        position_probs[i, k] = P(driver i finishes in position k+1)
        position_probs[i, -1] = P(driver i DNFs)
    """
    rng = np.random.default_rng(seed)
    n = len(log_lambdas)
    lambdas = np.exp(log_lambdas)
    position_counts = np.zeros((n, n + 1), dtype=np.float64)

    for _ in range(n_sims):
        # Determine DNFs
        dnf_rolls = rng.random(n)
        finisher_mask = dnf_rolls >= p_dnfs
        finisher_indices = np.where(finisher_mask)[0]
        dnf_indices = np.where(~finisher_mask)[0]

        # PL sampling for finishers
        remaining = list(finisher_indices)
        remaining_lambdas = lambdas[remaining].copy()

        for pos in range(len(remaining)):
            total = remaining_lambdas.sum()
            if total <= 0:
                break
            probs = remaining_lambdas / total
            chosen_local = rng.choice(len(remaining), p=probs)
            chosen_driver = remaining[chosen_local]
            position_counts[chosen_driver, pos] += 1

            # Remove from remaining
            remaining.pop(chosen_local)
            remaining_lambdas = np.delete(remaining_lambdas, chosen_local)

        # Record DNFs
        for d in dnf_indices:
            position_counts[d, -1] += 1

    return position_counts / n_sims


def compute_expected_points(
    pos_probs: np.ndarray,
    points_map: Dict[int, int],
    dnf_penalty: float = DNF_PENALTY,
) -> float:
    """Compute expected points for a single driver given position distribution."""
    n = pos_probs.shape[0] - 1  # Last entry is DNF
    ep = 0.0
    for k in range(n):
        pos = k + 1
        ep += pos_probs[k] * points_map.get(pos, 0)
    ep += pos_probs[-1] * dnf_penalty
    return ep


def compute_variance(
    pos_probs: np.ndarray,
    points_map: Dict[int, int],
    dnf_penalty: float = DNF_PENALTY,
) -> float:
    """Compute variance of points for a single driver."""
    n = pos_probs.shape[0] - 1
    ep = compute_expected_points(pos_probs, points_map, dnf_penalty)

    var = 0.0
    for k in range(n):
        pos = k + 1
        pts = points_map.get(pos, 0)
        var += pos_probs[k] * (pts - ep) ** 2
    var += pos_probs[-1] * (dnf_penalty - ep) ** 2
    return var


def fit_plackett_luce(
    observed_probs: Dict[str, Dict[str, float]],
    team_indices: np.ndarray,
    n_sims: int = 30000,
    method: str = "L-BFGS-B",
    team_reg: float = 0.1,
    smoothness_reg: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit Plackett-Luce model parameters to match observed market probabilities.

    Parameters
    ----------
    observed_probs : dict mapping market type to {driver_idx: probability}
        Supported keys: "win", "podium", "top6", "top10", "dnf"
    team_indices : (n_drivers,) array mapping driver to team index
    n_sims : simulations per objective evaluation (tradeoff speed vs accuracy)
    method : scipy optimizer method
    team_reg : regularization strength for teammate similarity
    smoothness_reg : regularization for parameter magnitudes (toward equal)

    Returns
    -------
    log_lambdas : (n_drivers,) fitted log-strength parameters
    p_dnfs : (n_drivers,) fitted DNF probabilities
    fit_info : dict with loss, residuals, etc.
    """
    n = len(team_indices)

    # Initial guess: roughly ordered by win probability if available
    if "win" in observed_probs:
        win_probs = observed_probs["win"]
        # Higher win prob → higher lambda
        init_log_lambdas = np.array([
            np.log(max(win_probs.get(i, 0.01), 0.001) * 100 + 1)
            for i in range(n)
        ])
    else:
        init_log_lambdas = np.zeros(n)

    if "dnf" in observed_probs:
        dnf_probs = observed_probs["dnf"]
        init_logit_dnfs = np.array([
            np.log(max(dnf_probs.get(i, 0.1), 0.01) / max(1 - dnf_probs.get(i, 0.1), 0.01))
            for i in range(n)
        ])
    else:
        init_logit_dnfs = np.full(n, -2.0)  # ~12% DNF

    x0 = np.concatenate([init_log_lambdas, init_logit_dnfs])

    eval_count = [0]

    def objective(x):
        eval_count[0] += 1
        log_lambdas = x[:n]
        logit_dnfs = x[n:]
        p_dnfs = 1.0 / (1.0 + np.exp(-logit_dnfs))

        # Use a different seed each eval for smoother optimization landscape
        seed = 42 + eval_count[0]
        pos_probs = simulate_races(log_lambdas, p_dnfs, n_sims=n_sims, seed=seed)

        loss = 0.0
        residuals = {}

        # Match observed cumulative probabilities
        market_cutoffs = {
            "win": 1,
            "podium": 3,
            "top6": 6,
            "top10": 10,
        }

        for market, cutoff in market_cutoffs.items():
            if market not in observed_probs:
                continue
            for i, obs_p in observed_probs[market].items():
                # Model probability: P(pos <= cutoff | this driver)
                # = sum of position probs for positions 1..cutoff
                # (excluding DNF — DNF is separate)
                model_p = pos_probs[i, :cutoff].sum()
                residual = model_p - obs_p
                loss += residual ** 2
                residuals[(market, i)] = residual

        # Match DNF probabilities
        if "dnf" in observed_probs:
            for i, obs_p in observed_probs["dnf"].items():
                model_p = pos_probs[i, -1]
                residual = model_p - obs_p
                loss += residual ** 2
                residuals[("dnf", i)] = residual

        # Regularization: teammates should have similar lambdas
        for t in range(N_TEAMS):
            teammates = [j for j in range(n) if team_indices[j] == t]
            if len(teammates) == 2:
                diff = log_lambdas[teammates[0]] - log_lambdas[teammates[1]]
                loss += team_reg * diff ** 2

        # Regularization: mild shrinkage toward mean (prevents extreme values)
        mean_ll = log_lambdas.mean()
        loss += smoothness_reg * np.sum((log_lambdas - mean_ll) ** 2)

        if eval_count[0] % 20 == 0:
            print(f"  Eval {eval_count[0]:4d}: loss={loss:.6f}")

        return loss

    print("Fitting Plackett-Luce model...")
    result = minimize(
        objective,
        x0,
        method=method,
        options={"maxiter": 200, "ftol": 1e-8},
    )
    print(f"  Converged: {result.success}, final loss: {result.fun:.6f}")

    log_lambdas = result.x[:n]
    logit_dnfs = result.x[n:]
    p_dnfs = 1.0 / (1.0 + np.exp(-logit_dnfs))

    # Normalize: set mean log_lambda to 0 (arbitrary scale)
    log_lambdas -= log_lambdas.mean()

    fit_info = {
        "loss": float(result.fun),
        "success": result.success,
        "n_evals": eval_count[0],
        "message": result.message if hasattr(result, "message") else "",
    }

    return log_lambdas, p_dnfs, fit_info


def generate_full_output(
    log_lambdas: np.ndarray,
    p_dnfs: np.ndarray,
    is_sprint: bool = False,
    n_sims: int = 50000,
) -> List[dict]:
    """
    Generate the complete output for all drivers.

    Returns a list of driver dicts with all computed statistics,
    ready to be serialized to JSON.
    """
    pos_probs = simulate_races(log_lambdas, p_dnfs, n_sims=n_sims, seed=12345)

    drivers_output = []
    for i in range(len(log_lambdas)):
        dist = pos_probs[i]
        ep_race = compute_expected_points(dist, RACE_POINTS)
        ep_sprint = compute_expected_points(dist, SPRINT_POINTS) if is_sprint else 0.0
        ep_total = ep_race + ep_sprint

        var_race = compute_variance(dist, RACE_POINTS)
        std_race = np.sqrt(var_race)

        # Key probabilities
        p_win = float(dist[0])
        p_podium = float(dist[:3].sum())
        p_top6 = float(dist[:6].sum())
        p_top10 = float(dist[:10].sum())
        p_points_zone = p_top10  # P(scoring race points)
        p_no_points = float(1.0 - p_top10 - dist[-1])
        p_dnf = float(dist[-1])

        driver_info = DRIVERS[i]

        drivers_output.append({
            "name": driver_info["name"],
            "abbr": driver_info["abbr"],
            "team_idx": driver_info["team_idx"],
            "lambda": float(log_lambdas[i]),
            "p_dnf": float(p_dnfs[i]),
            "ep_race": round(ep_race, 2),
            "ep_sprint": round(ep_sprint, 2),
            "ep_total": round(ep_total, 2),
            "std_dev": round(std_race, 2),
            "p_win": round(p_win, 4),
            "p_podium": round(p_podium, 4),
            "p_top6": round(p_top6, 4),
            "p_top10": round(p_top10, 4),
            "p_no_points": round(p_no_points, 4),
            # Full distribution (for charts)
            "position_distribution": [round(float(dist[k]), 5) for k in range(len(dist))],
        })

    # Sort by expected total points
    drivers_output.sort(key=lambda d: -d["ep_total"])

    return drivers_output


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)

    # Fake observed probs (as if from devigged odds)
    team_idx = np.array([d["team_idx"] for d in DRIVERS])
    fake_lambdas = np.array([
        4.8, 4.2,   # Mercedes
        3.1, 2.9,   # Ferrari
        2.4, 1.4,   # Red Bull
        2.2, 2.0,   # McLaren
        1.0, 0.65,  # AM
        1.5, 0.7,   # Alpine
        1.3, 1.2,   # Williams
        0.95, 0.85, # RB
        1.8, 0.8,   # Haas
        0.5, 0.45,  # Sauber
        0.35, 0.3,  # Cadillac
    ])
    fake_dnfs = np.full(N_DRIVERS, 0.10)

    print("Simulating with fake parameters...")
    pos_probs = simulate_races(np.log(fake_lambdas), fake_dnfs, n_sims=50000)

    print("\nExpected Race Points:")
    for i, d in enumerate(DRIVERS):
        ep = compute_expected_points(pos_probs[i], RACE_POINTS)
        p_win = pos_probs[i, 0]
        p_dnf = pos_probs[i, -1]
        print(f"  {d['name']:20s} E[pts]={ep:6.2f}  P(win)={p_win:.3f}  P(DNF)={p_dnf:.3f}")

    print("\nGenerating full output...")
    output = generate_full_output(np.log(fake_lambdas), fake_dnfs, is_sprint=False)
    for d in output[:5]:
        print(f"  {d['name']:20s} E[pts]={d['ep_race']:6.2f}  σ={d['std_dev']:.1f}")
