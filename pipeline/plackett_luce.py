"""
Plackett-Luce model: simulation, optimization, and expected points computation.

The PL model assigns each driver a strength parameter λ_i. The probability
of a finishing order is:

  P(σ) = ∏_{k=1}^{n} λ_{σ(k)} / Σ_{j=k}^{n} λ_{σ(j)}

We decompose log(λ) = μ_team + δ_driver, fit via optimization against
observed odds, then simulate to get full position distributions.
"""

import numpy as np
from itertools import combinations, permutations
from scipy.optimize import minimize, linear_sum_assignment
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

    Uses the Gumbel-max trick for vectorized PL sampling:
    ranking by (log_lambda + Gumbel noise) is equivalent to sequential
    PL draws, but runs entirely in numpy with no Python loops over sims.

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

    # Gumbel-max trick: sample Gumbel(0,1) noise, add to log-lambdas, argsort
    # This gives a Plackett-Luce draw in O(n log n) with full vectorization
    gumbel_noise = rng.gumbel(size=(n_sims, n))  # (n_sims, n_drivers)
    utilities = log_lambdas[np.newaxis, :] + gumbel_noise  # (n_sims, n_drivers)

    # DNF mask: True if driver DNFs in this sim
    dnf_mask = rng.random((n_sims, n)) < p_dnfs[np.newaxis, :]  # (n_sims, n_drivers)

    # For the PL ranking, exclude DNF drivers by setting utility to -inf
    utilities_for_sort = utilities.copy()
    utilities_for_sort[dnf_mask] = -np.inf

    # Argsort descending = finishing order (highest utility = P1)
    rankings = np.argsort(-utilities_for_sort, axis=1)  # (n_sims, n_drivers)

    # Convert rankings to positions: positions[s, i] = raw rank of driver i
    positions = np.argsort(rankings, axis=1)  # (n_sims, n_drivers)

    # Count finishers per sim to know how many real positions exist
    n_finishers = (~dnf_mask).sum(axis=1)  # (n_sims,)

    # Count position frequencies
    position_counts = np.zeros((n, n + 1), dtype=np.float64)
    for i in range(n):
        driver_positions = positions[:, i]  # (n_sims,) raw rank for driver i
        driver_dnfs = dnf_mask[:, i]       # (n_sims,) DNF mask for driver i

        # Count finishing positions (only for non-DNF sims)
        # DNF drivers sort last but their positions are meaningless —
        # only count positions for drivers who actually finished
        finish_positions = driver_positions[~driver_dnfs]
        if len(finish_positions) > 0:
            counts = np.bincount(finish_positions, minlength=n)
            position_counts[i, :n] = counts

        # Count DNFs
        position_counts[i, -1] = driver_dnfs.sum()

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
    n_sims: int = 10000,
    method: str = "Powell",
    team_reg: float = 0.02,
    smoothness_reg: float = 0.005,
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

    # Initial guess: in PL, P(i wins) ≈ λ_i / Σ λ_j, so log(P_win) is a
    # good starting point for log(λ). This preserves the full spread of the
    # odds rather than compressing it.
    if "win" in observed_probs:
        win_probs = observed_probs["win"]
        init_log_lambdas = np.array([
            np.log(max(win_probs.get(i, 0.0001), 0.0001))
            for i in range(n)
        ])
        init_log_lambdas -= init_log_lambdas.mean()  # center at 0
    else:
        init_log_lambdas = np.zeros(n)

    # Fix DNF probabilities directly from odds (don't optimize them).
    # This halves the parameter space and removes a major source of noise.
    if "dnf" in observed_probs:
        dnf_probs = observed_probs["dnf"]
        fixed_p_dnfs = np.array([dnf_probs.get(i, 0.10) for i in range(n)])
    else:
        fixed_p_dnfs = np.full(n, 0.10)

    # Only optimize the 22 lambda parameters (not 44 = lambda + DNF)
    x0 = init_log_lambdas.copy()

    n_params = len(x0)
    eval_count = [0]
    step_count = [0]
    best_loss = [float('inf')]
    loss_history = []  # (eval_num, loss, loss_data, loss_team, loss_shrink)
    step_losses = []   # loss at each optimizer step
    import time
    start_time = [time.time()]

    def objective(x):
        eval_count[0] += 1
        log_lambdas = x
        p_dnfs = fixed_p_dnfs

        # Use a different seed each eval for smoother optimization landscape
        seed = 42 + eval_count[0]
        pos_probs = simulate_races(log_lambdas, p_dnfs, n_sims=n_sims, seed=seed)

        loss = 0.0
        residuals = {}
        loss_data = 0.0
        loss_team = 0.0
        loss_shrink = 0.0

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
                model_p = pos_probs[i, :cutoff].sum()
                residual = model_p - obs_p
                loss_data += residual ** 2
                residuals[(market, i)] = residual

        # DNF probabilities are fixed from odds, not optimized.
        # (No DNF loss term needed.)

        # Regularization: teammates should have similar lambdas
        for t in range(N_TEAMS):
            teammates = [j for j in range(n) if team_indices[j] == t]
            if len(teammates) == 2:
                diff = log_lambdas[teammates[0]] - log_lambdas[teammates[1]]
                loss_team += team_reg * diff ** 2

        # Regularization: mild shrinkage toward mean (prevents extreme values)
        mean_ll = log_lambdas.mean()
        loss_shrink = smoothness_reg * np.sum((log_lambdas - mean_ll) ** 2)

        loss = loss_data + loss_team + loss_shrink

        if loss < best_loss[0]:
            best_loss[0] = loss

        # Record loss history (subsample to keep JSON manageable)
        if eval_count[0] <= 50 or eval_count[0] % 10 == 0:
            loss_history.append({
                "eval": eval_count[0],
                "loss": round(loss, 6),
                "data": round(loss_data, 6),
                "team": round(loss_team, 6),
                "shrink": round(loss_shrink, 6),
            })

        # Log every eval with timing
        elapsed = time.time() - start_time[0]
        evals_per_sec = eval_count[0] / max(elapsed, 0.01)
        print(
            f"  eval {eval_count[0]:5d} | "
            f"loss={loss:.6f} (data={loss_data:.6f} team={loss_team:.6f} shrink={loss_shrink:.6f}) | "
            f"best={best_loss[0]:.6f} | "
            f"{elapsed:.1f}s ({evals_per_sec:.1f} eval/s)",
            flush=True,
        )

        return loss

    print(f"Fitting Plackett-Luce model...")
    print(f"  Parameters: {n_params} (lambdas only; DNF probs fixed from odds)")
    print(f"  Sims per eval: {n_sims:,}")
    print(f"  Method: {method}")
    print(flush=True)

    def callback(xk):
        step_count[0] += 1
        elapsed = time.time() - start_time[0]
        step_losses.append({
            "step": step_count[0],
            "eval": eval_count[0],
            "best_loss": round(best_loss[0], 6),
            "elapsed": round(elapsed, 1),
        })
        print(
            f"  --- STEP {step_count[0]:3d} complete | "
            f"{eval_count[0]} total evals | "
            f"best loss={best_loss[0]:.6f} | "
            f"{elapsed:.1f}s elapsed ---",
            flush=True,
        )

    result = minimize(
        objective,
        x0,
        method=method,
        callback=callback,
        options={"maxiter": 200, "ftol": 1e-8},
    )
    elapsed = time.time() - start_time[0]
    print(f"  Converged: {result.success}, final loss: {result.fun:.6f}")
    print(f"  Total: {eval_count[0]} evals, {step_count[0]} steps, {elapsed:.1f}s")
    if hasattr(result, 'message'):
        print(f"  Message: {result.message}")

    log_lambdas = result.x
    p_dnfs = fixed_p_dnfs

    # Normalize: set mean log_lambda to 0 (arbitrary scale)
    log_lambdas -= log_lambdas.mean()

    # Compute final residuals with a large simulation for accuracy
    final_pos_probs = simulate_races(log_lambdas, p_dnfs, n_sims=50000, seed=99999)
    market_cutoffs = {"win": 1, "podium": 3, "top6": 6, "top10": 10}
    residuals = []
    for market, cutoff in market_cutoffs.items():
        if market not in observed_probs:
            continue
        for i, obs_p in observed_probs[market].items():
            model_p = float(final_pos_probs[i, :cutoff].sum())
            residuals.append({
                "market": market,
                "driver_idx": i,
                "driver": DRIVERS[i]["abbr"],
                "observed": round(obs_p, 4),
                "model": round(model_p, 4),
                "residual": round(model_p - obs_p, 4),
            })

    fit_info = {
        "loss": float(result.fun),
        "success": result.success,
        "n_evals": eval_count[0],
        "n_steps": step_count[0],
        "elapsed_seconds": round(time.time() - start_time[0], 1),
        "method": method,
        "n_sims_per_eval": n_sims,
        "n_params": n_params,
        "team_reg": team_reg,
        "smoothness_reg": smoothness_reg,
        "message": result.message if hasattr(result, "message") else "",
        "loss_history": loss_history,
        "step_losses": step_losses,
        "residuals": residuals,
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


def find_top_lineups(
    drivers_data: List[dict],
    n_picks: int = 5,
    exact_pos_bonus: float = 10.0,
    top_n: int = 10,
) -> List[dict]:
    """
    Find the top_n highest-scoring ordered lineups by expected points
    including the exact-position bonus, returned in descending order.

    For each selection of n_picks drivers from the grid, the optimal slot
    ordering maximises Σ P(driver finishes in their pick slot) * bonus.
    We enumerate all C(n_drivers, n_picks) selections and score them against
    all n_picks! permutations in a single vectorised numpy operation, then
    sort and return the top_n.

    At 22 drivers / 5 picks: C(22,5)=26,334 selections × 5!=120 permutations.
    The inner product is computed entirely in numpy with no Python inner loop.

    Score matrix identity
    ---------------------
    total[combo, perm] = Σ_i  ep_base[combo[i]] + pos_bonus[combo[i], perm[i]]
                       = ep_base_sum[combo]  +  Σ_i B[combo, i, perm[i]]

    where B[c,i,s] = pos_bonus[combo_c_driver_i, slot_s].

    Parameters
    ----------
    drivers_data : list of driver dicts from generate_full_output
    n_picks : number of picks (default 5)
    exact_pos_bonus : bonus for exact position match (default 10)
    top_n : number of lineups to return (default 10)

    Returns
    -------
    List of dicts (length top_n), each with:
        rank, picks, ep_base_total, ep_bonus_total, ep_grand_total
    """
    n = len(drivers_data)
    ep_totals = np.array([d["ep_total"] for d in drivers_data])

    # pos_bonus[d, s] = P(driver d finishes position s+1) * exact_pos_bonus
    pos_bonus = np.array([
        [d["position_distribution"][s] * exact_pos_bonus for s in range(n_picks)]
        for d in drivers_data
    ])  # (n_drivers, n_picks)

    # All C(n, n_picks) driver-index combinations: shape (n_combos, n_picks)
    all_combos = np.array(list(combinations(range(n), n_picks)))  # (26334, 5)

    # Bonus submatrix for every combo: B_all[c, i, s] = pos_bonus for the
    # i-th driver of combo c assigned to slot s.
    B_all = pos_bonus[all_combos, :]  # (n_combos, n_picks, n_picks)

    # All n_picks! slot permutations: all_perms[p, i] = slot assigned to
    # driver i under permutation p.
    all_perms = np.array(list(permutations(range(n_picks))))  # (120, 5)

    # bonus_scores[c, p] = Σ_i B_all[c, i, all_perms[p, i]]
    # Computed without a Python loop by summing n_picks (n_combos, n_perms) slices.
    bonus_scores = sum(
        B_all[:, i, all_perms[:, i]]   # (n_combos, n_perms)
        for i in range(n_picks)
    )  # (n_combos, n_perms)

    best_perm_idx = bonus_scores.argmax(axis=1)        # (n_combos,)
    best_bonus    = bonus_scores.max(axis=1)            # (n_combos,)
    ep_base       = ep_totals[all_combos].sum(axis=1)  # (n_combos,)
    totals        = ep_base + best_bonus               # (n_combos,)

    top_combo_indices = np.argsort(-totals)[:top_n]

    lineups = []
    for rank_0, combo_idx in enumerate(top_combo_indices):
        combo = all_combos[combo_idx]                   # (n_picks,) driver indices
        perm  = all_perms[best_perm_idx[combo_idx]]     # (n_picks,) slot assignments

        picks = []
        for local_i, slot_0idx in enumerate(perm):
            driver_idx = int(combo[local_i])
            d = drivers_data[driver_idx]
            picks.append({
                "slot": int(slot_0idx + 1),
                "name": d["name"],
                "abbr": d["abbr"],
                "team_idx": d["team_idx"],
                "ep_base": round(float(ep_totals[driver_idx]), 2),
                "slot_bonus_ev": round(float(pos_bonus[driver_idx, slot_0idx]), 3),
            })
        picks.sort(key=lambda p: p["slot"])

        lineups.append({
            "rank": rank_0 + 1,
            "picks": picks,
            "ep_base_total": round(float(ep_base[combo_idx]), 2),
            "ep_bonus_total": round(float(best_bonus[combo_idx]), 3),
            "ep_grand_total": round(float(totals[combo_idx]), 2),
            "exact_pos_bonus": exact_pos_bonus,
        })

    return lineups


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
