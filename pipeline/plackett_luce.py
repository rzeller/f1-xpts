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
from config import RACE_POINTS, SPRINT_POINTS, DNF_PENALTY, EXACT_BONUS, CORRELATION_DEFAULTS


def simulate_races(
    log_lambdas: np.ndarray,
    p_dnfs: np.ndarray,
    n_sims: int = 50000,
    seed: int = 42,
    team_indices: np.ndarray = None,
    correlation: dict = None,
    sigma_drv: np.ndarray = None,
    chaos_alpha_drv: np.ndarray = None,
) -> np.ndarray:
    """
    Simulate races from the Plackett-Luce model with DNFs.

    Uses the Gumbel-max trick for vectorized PL sampling:
    ranking by (log_lambda + Gumbel noise) is equivalent to sequential
    PL draws, but runs entirely in numpy with no Python loops over sims.

    `sigma_drv` (length n_drivers, default 1.0 each) is a per-driver scaling
    on the Gumbel noise. Larger σ_i widens that driver's outcome distribution
    around their λ-implied position; combined with chaos noise this gives
    each driver a tunable "win-or-fade" vs. "consistent" shape.

    Parameters
    ----------
    log_lambdas : (n_drivers,) array of log-strength parameters
    p_dnfs : (n_drivers,) array of DNF probabilities
    n_sims : number of races to simulate
    seed : random seed
    team_indices : (n_drivers,) array mapping driver index to team index
    correlation : dict with keys sigma_team, sigma_global, sigma_dnf
        If None, no correlation noise is applied (backward compatible).

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
    if sigma_drv is not None:
        gumbel_noise = gumbel_noise * sigma_drv[np.newaxis, :]
    elif correlation is not None and "sigma_drv_base" in correlation:
        # Default per-driver Gumbel scale, applied uniformly when no explicit
        # per-driver σ is given. Smaller values make wins more deterministic
        # (less "lost a Gumbel coin flip to NOR" P2-P3 outcomes for top
        # drivers) at the cost of needing larger chaos magnitude to keep
        # backmarker leakage bounded.
        sdb = correlation.get("sigma_drv_base", 1.0)
        if sdb != 1.0:
            gumbel_noise = gumbel_noise * sdb

    # --- Correlated noise layers ---
    # Use a separate RNG so that when correlation is disabled, the Gumbel
    # stream (and thus results) are identical to the non-correlation code path.
    if correlation is not None and team_indices is not None:
        corr_rng = np.random.default_rng(seed + 1_000_000)
        sigma_team = correlation.get("sigma_team", 0.0)
        sigma_global = correlation.get("sigma_global", 0.0)
        sigma_dnf = correlation.get("sigma_dnf", 0.0)
        chaos_model = correlation.get("chaos_model", "symmetric")
        n_teams = int(team_indices.max()) + 1

        # 1. Team race-day noise: one z per team per sim, shared by teammates
        team_noise = 0.0
        if sigma_team > 0:
            z_team = corr_rng.standard_normal((n_sims, n_teams))  # (n_sims, n_teams)
            team_noise = sigma_team * z_team[:, team_indices]       # (n_sims, n_drivers)

        # 2. Chaos noise. Three models supported:
        #    - "symmetric" (legacy, calibrated to historical race
        #      finishing-position variance): log-normal multiplier on each
        #      driver's Gumbel — symmetric across drivers, can boost or hurt.
        #    - "one_sided" (current default — see issue #36 discussion):
        #      per-driver, per-race exponential downside event,
        #      `utility -= Exp(σ_global)`. Drivers can fall back due to
        #      incidents/mechanical/strategy issues but can't gain pace they
        #      don't have. Decouples backmarker performance from favorites:
        #      Bottas's draw doesn't matter for whether Russell wins; only
        #      whether Russell himself has trouble.
        #    - "bimodal": continuous downside is replaced by a categorical
        #      mixture per driver per race —
        #          P(no chaos)   = 1 − p_mod − p_sev  (utility unchanged)
        #          P(moderate)   = p_mod              (utility -= Exp(σ_mod))
        #          P(severe)     = p_sev              (utility -= Exp(σ_sev))
        #      Lets us separate "small spin / undercut / traffic" events
        #      from rare "mechanical failure / crash / weather" events,
        #      since their physical magnitudes differ. Exponential one-sided
        #      is a special case (set p_mod=1, p_sev=0).
        if chaos_model == "bimodal":
            p_moderate = correlation.get("chaos_p_moderate", 0.30)
            p_severe = correlation.get("chaos_p_severe", 0.05)
            sigma_moderate = correlation.get("chaos_sigma_moderate", sigma_global)
            sigma_severe = correlation.get("chaos_sigma_severe", sigma_global * 6.0)
            u = corr_rng.random((n_sims, n))
            mod_mask = (u < p_moderate).astype(np.float64)
            sev_mask = ((u >= p_moderate) & (u < p_moderate + p_severe)).astype(np.float64)
            mod_amt = corr_rng.exponential(scale=max(sigma_moderate, 1e-9), size=(n_sims, n))
            sev_amt = corr_rng.exponential(scale=max(sigma_severe, 1e-9), size=(n_sims, n))
            chaos_noise = -mod_mask * mod_amt - sev_mask * sev_amt
            gumbel_term = gumbel_noise
        elif chaos_model == "lomax":
            # Heavy-tailed downside: -Lomax(α, σ) per driver per race.
            # Lomax = "exponential with gamma-distributed rate," so it
            # corresponds to "different races have different chaos
            # severities." Mode at 0, polynomial-decay tail (vs exponential).
            # As α → ∞, recovers Exp(σ). Lower α = fatter tail. Mean
            # = σ/(α-1) for α > 1.
            #
            # Why fatter tail helps: scaling Exp(σ) to push P(podium) down
            # also reduces P(win) about as fast (every driver gets more
            # typical-magnitude hits). With Lomax, the modal/typical hit
            # stays small (P(win) preserved) but rare big hits get bigger
            # (more drivers fall off podium occasionally). Decouples the
            # win/podium residuals.
            #
            # Per-driver α (chaos_alpha_drv) lets reliable drivers have
            # thinner tails (rarer catastrophic events) and crash-prone
            # drivers fatter tails. Sampled via inverse-CDF on Uniform(0,1)
            # so α can be a per-driver array without a Python loop.
            if sigma_global > 0:
                if chaos_alpha_drv is not None:
                    alpha = chaos_alpha_drv[np.newaxis, :]
                else:
                    alpha = correlation.get("chaos_alpha", 2.0)
                u = corr_rng.random((n_sims, n))
                # Lomax(α, σ) inverse CDF: σ · ((1-u)^(-1/α) - 1)
                # Clip α away from 1 to keep mean finite.
                if np.isscalar(alpha):
                    alpha = max(alpha, 1.001)
                else:
                    alpha = np.maximum(alpha, 1.001)
                magnitude = sigma_global * ((1.0 - u) ** (-1.0 / alpha) - 1.0)
                chaos_noise = -magnitude
            else:
                chaos_noise = 0.0
            gumbel_term = gumbel_noise
        elif chaos_model == "one_sided":
            if sigma_global > 0:
                # Independent per-driver downside event each race
                downside = corr_rng.exponential(scale=sigma_global, size=(n_sims, n))
                chaos_noise = -downside
            else:
                chaos_noise = 0.0
            # base Gumbel still applies
            gumbel_term = gumbel_noise
        else:  # symmetric (legacy)
            chaos_scale = 1.0
            if sigma_global > 0:
                z_global = corr_rng.standard_normal(n_sims)              # (n_sims,)
                chaos_scale = np.exp(sigma_global * z_global)[:, np.newaxis]
            gumbel_term = chaos_scale * gumbel_noise
            chaos_noise = 0.0

        # 3. Correlated DNFs: log-normal multiplier on DNF probabilities
        #    Some races have more incidents than others.
        if sigma_dnf > 0:
            z_dnf = corr_rng.standard_normal(n_sims)                 # (n_sims,)
            dnf_multiplier = np.exp(sigma_dnf * z_dnf)[:, np.newaxis]  # (n_sims, 1)
            effective_p_dnfs = np.clip(p_dnfs[np.newaxis, :] * dnf_multiplier, 0.0, 0.5)
        else:
            effective_p_dnfs = p_dnfs[np.newaxis, :]  # (1, n_drivers)

        utilities = (log_lambdas[np.newaxis, :] + team_noise) + gumbel_term + chaos_noise
    else:
        utilities = log_lambdas[np.newaxis, :] + gumbel_noise  # (n_sims, n_drivers)
        effective_p_dnfs = p_dnfs[np.newaxis, :]  # (1, n_drivers)

    # DNF mask: True if driver DNFs in this sim
    dnf_mask = rng.random((n_sims, n)) < effective_p_dnfs  # (n_sims, n_drivers)

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


# Gauss-Laguerre nodes/weights for ∫₀^∞ exp(-t) f(t) dt ≈ Σₖ wₖ f(tₖ).
# Cached at module load — picking K=80 keeps top-driver win-prob error
# comfortably below MC noise (verified on synthetic 22-driver fits).
from numpy.polynomial.laguerre import laggauss as _laggauss
_LAGUERRE_K = 80
_LAGUERRE_NODES, _LAGUERRE_WEIGHTS = _laggauss(_LAGUERRE_K)
_LAGUERRE_LOG_T = np.log(_LAGUERRE_NODES)  # (K,)

# Gauss-Hermite nodes/weights for ∫_{-∞}^{∞} exp(-z²/2) / √(2π) f(z) dz.
# Used for the chaos-noise integral over z_global ~ Normal(0,1). 12 nodes
# is plenty for a smooth log-normal-multiplier integrand.
from numpy.polynomial.hermite_e import hermegauss as _hermegauss
_HERMITE_K = 12
_HERMITE_NODES, _HERMITE_WEIGHTS = _hermegauss(_HERMITE_K)
_HERMITE_WEIGHTS = _HERMITE_WEIGHTS / np.sqrt(2 * np.pi)  # normalize to N(0,1) measure


def analytic_win_probs(
    log_lambdas: np.ndarray,
    p_dnfs: np.ndarray,
    sigma_drv: np.ndarray = None,
    team_indices: np.ndarray = None,
    correlation: dict = None,
    n_team_samples: int = 80,
    seed: int = 4242,
) -> np.ndarray:
    """Deterministic P(driver i wins) via quadrature, replacing MC for the
    win market only. Uses the heterogeneous-scale Gumbel-max identity:

        P(i wins | μ, β) = ∫₀^∞ exp(-t) ∏_{j≠i} [p_dnf_j + (1 − p_dnf_j)
                                                  · exp(-aᵢⱼ · t^{rᵢⱼ})] dt

    where μⱼ = log(λⱼ) + σ_team · z_t(j), βⱼ = c · σ_drv_j, c = exp(σ_global · z),
    aᵢⱼ = exp((μⱼ − μᵢ)/βⱼ), rᵢⱼ = βᵢ/βⱼ. Marginalized over the chaos noise
    z ~ N(0, 1) by Gauss-Hermite (12 nodes) and over the team noise z_t by
    Monte Carlo (deterministic at fixed seed). The final survival factor
    multiplies by (1 − p_dnf_i).

    Why this exists: the MC simulator can't reach P(win) values much below
    1/n_sims, which broke the "anchor λ to win" inner loop for backmarkers.
    The quadrature formula has no such floor — at λ_i → −∞, P(i wins) → 0
    smoothly — so the bilevel anchor stays well-conditioned.

    Returns: (n,) array of P(i wins) values summing to ≤ 1 (≈1 minus the
    probability that everyone DNFs; deterministic to ~1e-3 vs MC).
    """
    n = len(log_lambdas)
    if sigma_drv is None:
        sigma_drv = np.ones(n)
    if correlation is None:
        sigma_team = sigma_global = sigma_dnf = 0.0
    else:
        sigma_team = correlation.get("sigma_team", 0.0)
        sigma_global = correlation.get("sigma_global", 0.0)
        sigma_dnf = correlation.get("sigma_dnf", 0.0)

    n_teams = int(team_indices.max()) + 1 if team_indices is not None else 0

    # Pre-sample team noise z_t for MC averaging (deterministic at fixed seed).
    rng = np.random.default_rng(seed)
    if sigma_team > 0 and n_teams > 0 and team_indices is not None:
        z_team_samples = rng.standard_normal((n_team_samples, n_teams))  # (S, T)
    else:
        z_team_samples = np.zeros((1, max(n_teams, 1)))

    # Pre-sample DNF correlation z_d for the survival mixture (independent
    # of the inner Gumbel-max integral; we average it as a separate MC layer).
    survival = 1.0 - p_dnfs  # (n,)

    # Iterate over (chaos node, team-noise sample) combinations and accumulate.
    win_probs = np.zeros(n)
    total_weight = 0.0
    log_t = _LAGUERRE_LOG_T  # (K,)
    laguerre_w = _LAGUERRE_WEIGHTS  # (K,)

    for k_chaos, (z_g, w_chaos) in enumerate(zip(_HERMITE_NODES, _HERMITE_WEIGHTS)):
        c = np.exp(sigma_global * z_g) if sigma_global > 0 else 1.0
        beta = c * sigma_drv  # (n,) per-driver scale this race-day

        for s, z_t in enumerate(z_team_samples):
            # Per-driver location with team noise.
            if sigma_team > 0 and n_teams > 0:
                mu = log_lambdas + sigma_team * z_t[team_indices]  # (n,)
            else:
                mu = log_lambdas

            # Inner: P(i wins | this race) for each driver via Laguerre.
            # Vectorize over j ≠ i using a precomputed broadcast.
            # For each driver i: a[j] = exp((μⱼ − μᵢ)/βⱼ), r[j] = βᵢ/βⱼ
            # log_term[k, j] = log(a[j]) + r[j] · log(t_k)
            # mixture_factor[k, j] = (1 - p_dnf_j) · exp(-a[j] · t_k^{r[j]})
            #                        + p_dnf_j  (driver j DNFs → "wins" against everyone)
            # P(i wins | race) = Σ_k w_k · ∏_{j≠i} mixture_factor[k, j]
            for i in range(n):
                mask = np.arange(n) != i
                mu_j = mu[mask]
                beta_j = beta[mask]
                p_dnf_j = p_dnfs[mask]

                # log_a[j] = (μⱼ − μᵢ) / βⱼ
                log_a = (mu_j - mu[i]) / beta_j  # (n-1,)
                r = beta[i] / beta_j  # (n-1,)
                # log_terms[k, j] = log_a[j] + r[j] · log(t_k)
                log_terms = log_a[None, :] + r[None, :] * log_t[:, None]  # (K, n-1)
                # exp(-a · t^r) per (k, j); clip to avoid overflow at extreme λ
                neg_aterms = -np.exp(np.clip(log_terms, -50, 50))
                survive_factor = np.exp(neg_aterms)  # (K, n-1) ∈ [0, 1]
                # Mixture: j contributes survive_factor if it finishes, 1.0 if it DNFs
                # (DNF equivalent to ∞ utility from i's perspective… wait, no — DNF means
                # j is REMOVED from the race, so j doesn't beat i. Treat DNF j the same
                # as j having infinitely-low utility, i.e. mixture factor = 1.)
                mix = (1.0 - p_dnf_j[None, :]) * survive_factor + p_dnf_j[None, :]
                prod = np.prod(mix, axis=1)  # (K,)
                integral = np.sum(laguerre_w * prod)
                win_probs[i] += w_chaos * integral

            total_weight += w_chaos

    # Average over team noise (uniform weight); chaos already weighted via Hermite.
    win_probs = win_probs / len(z_team_samples)
    # Apply survival to driver i (i must finish to win at all).
    win_probs = win_probs * survival
    return win_probs


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


def anchor_lambda_to_win_market(
    observed_win: Dict[int, float],
    sigma_drv: np.ndarray,
    p_dnfs: np.ndarray,
    team_indices: np.ndarray = None,
    correlation: dict = None,
    init_log_lambdas: np.ndarray = None,
    n_team_samples: int = 80,
    max_iters: int = 30,
    tol: float = 1e-3,
) -> np.ndarray:
    """Solve for log_lambdas s.t. analytic_win_probs(log_lambdas, σ_drv) ≈
    observed_win. Uses fixed-point iteration in log-space:

        log_λ_i ← log_λ_i + log(obs_p_win_i / analytic_p_win_i)

    Convergence is fast (~5-10 iters) because the analytic mapping λ → P(win)
    is monotonic per-driver and approximately multiplicative across drivers
    in log-space. Unlike the MC-based anchor we tried earlier, this stays
    well-conditioned for backmarkers because analytic_win_probs has no MC
    floor — at λ_i → −∞, P(i wins) → 0 smoothly.
    """
    n = len(sigma_drv)
    if init_log_lambdas is not None:
        ll = init_log_lambdas.copy()
    else:
        ll = np.zeros(n)
        for i, p in observed_win.items():
            if p > 0:
                ll[i] = np.log(max(p, 1e-12))
        ll -= ll.mean()

    obs_arr = np.array([observed_win.get(i, 1e-12) for i in range(n)])
    obs_arr = np.maximum(obs_arr, 1e-12)
    log_obs = np.log(obs_arr)

    for _ in range(max_iters):
        analytic = analytic_win_probs(
            ll, p_dnfs, sigma_drv=sigma_drv,
            team_indices=team_indices, correlation=correlation,
            n_team_samples=n_team_samples,
        )
        analytic = np.maximum(analytic, 1e-12)
        update = log_obs - np.log(analytic)
        if np.max(np.abs(update)) < tol:
            break
        ll = ll + update
        ll -= ll.mean()
        ll = np.clip(ll, -15, 15)
    return ll


DEFAULT_MARKET_WEIGHTS = {
    # The win market is the cleanest signal in the input odds (every other
    # market is a placement aggregate). It's also where the regularization
    # ceiling bites hardest — the optimizer's natural posture pulls all
    # drivers toward the mean, which under-predicts favorites and over-
    # predicts backmarkers. A 16x weight on the win market plus the
    # relaxed `smoothness_reg` below brings top-of-field residuals to
    # ~3-4pp on both Miami and Japan inputs. Going higher (24x+) trades
    # placement-market fit for marginal win-market improvement (issue #36
    # follow-up sweep, /tmp/sweep6.py).
    "win": 16.0,
    "podium": 1.0,
    "top5": 1.0,
    "top6": 1.0,
    "top10": 1.0,
}


def fit_plackett_luce(
    observed_probs: Dict[str, Dict[str, float]],
    team_indices: np.ndarray,
    n_sims: int = 10000,
    method: str = "Powell",
    # The two regularizers below are heuristic: they exist to keep the
    # optimizer from chasing noise in tiny markets, but at the original
    # values (0.02 / 0.005) they also pinned favorite λ values too close
    # to the field mean, capping how concentrated the modeled win
    # distribution could be. Verified empirically (see /tmp/ceiling.py)
    # that the simulator's chaos noise alone does NOT cap top-of-field
    # P(win) — even at sigma_global=1.17 a sufficiently large λ ratio
    # reaches P(win)≈0.80. So the ceiling we were hitting was the
    # regularization, not the model.
    team_reg: float = 0.005,
    smoothness_reg: float = 0.0001,
    sigma_drv_reg: float = 0.005,
    chaos_alpha_drv_reg: float = 0.05,
    correlation: dict = None,
    market_weights: dict = None,
    # σ_drv per-driver volatility: experimental. Verified to give exact
    # win-prob match via the analytic Gumbel-max formula (see
    # analytic_win_probs), but the resulting joint 44-D Powell objective
    # is too expensive for production (~10x slower per fit). Disabled by
    # default; flip to True for offline experiments only.
    fit_sigma_drv: bool = False,
    # Per-driver chaos α — lets reliable drivers have thinner tails, crash-
    # prone drivers fatter ones. Adds n parameters to the optimizer.
    fit_chaos_alpha_drv: bool = False,
    n_team_samples: int = 30,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit Plackett-Luce model parameters to match observed market probabilities.

    Parameters
    ----------
    observed_probs : dict mapping market type to {driver_idx: probability}
        Supported keys: "win", "podium", "top5", "top6", "top10", "dnf"
    team_indices : (n_drivers,) array mapping driver to team index
    n_sims : simulations per objective evaluation (tradeoff speed vs accuracy)
    method : scipy optimizer method
    team_reg : regularization strength for teammate similarity
    smoothness_reg : regularization for parameter magnitudes (toward equal)
    market_weights : per-market multiplier on squared residuals (defaults to
        DEFAULT_MARKET_WEIGHTS, which up-weights the win market 4x).

    Returns
    -------
    log_lambdas : (n_drivers,) fitted log-strength parameters
    p_dnfs : (n_drivers,) fitted DNF probabilities
    fit_info : dict with loss, residuals, etc.
    """
    if market_weights is None:
        market_weights = DEFAULT_MARKET_WEIGHTS
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

    # Pack optimizer state. Layout (n drivers):
    #   [0:n]               log_lambdas
    #   [n:2n]      (opt)   log_sigma_drv (only when fit_sigma_drv)
    #   [next n]    (opt)   z = log(α_drv - 1) (only when fit_chaos_alpha_drv)
    # Both are independent opt-ins.
    base_alpha = (correlation or {}).get("chaos_alpha", 2.0) if correlation else 2.0
    z_alpha_init = np.log(max(base_alpha - 1.0, 0.001)) * np.ones(n)
    parts = [init_log_lambdas]
    if fit_sigma_drv:
        parts.append(np.zeros(n))
    if fit_chaos_alpha_drv:
        parts.append(z_alpha_init)
    x0 = np.concatenate(parts)

    n_params = len(x0)
    eval_count = [0]
    step_count = [0]
    best_loss = [float('inf')]
    loss_history = []  # (eval_num, loss, loss_data, loss_team, loss_shrink)
    step_losses = []   # loss at each optimizer step
    import time
    start_time = [time.time()]

    def _unpack(x):
        log_lambdas = x[:n]
        offset = n
        if fit_sigma_drv:
            sigma_drv = np.exp(x[offset:offset + n])
            offset += n
        else:
            sigma_drv = None
        if fit_chaos_alpha_drv:
            chaos_alpha_drv = 1.0 + np.exp(x[offset:offset + n])
            offset += n
        else:
            chaos_alpha_drv = None
        return log_lambdas, sigma_drv, chaos_alpha_drv

    def objective(x):
        eval_count[0] += 1
        log_lambdas, sigma_drv, chaos_alpha_drv = _unpack(x)
        p_dnfs = fixed_p_dnfs

        # Placement markets (podium / top5 / top6 / top10) need MC because
        # chaos noise materially changes the position spread, not just the
        # argmax. CRN seed makes this deterministic in (λ, σ).
        pos_probs = simulate_races(
            log_lambdas, p_dnfs, n_sims=n_sims, seed=42,
            team_indices=team_indices, correlation=correlation,
            sigma_drv=sigma_drv,
            chaos_alpha_drv=chaos_alpha_drv,
        )

        # Win residual: prefer the analytic formula (deterministic, no MC
        # noise floor) when σ_drv is being fit AND we're in the symmetric
        # chaos model the formula was derived for. Otherwise use MC.
        is_symmetric = (correlation is None
                        or correlation.get("chaos_model", "symmetric") == "symmetric")
        if fit_sigma_drv and is_symmetric:
            analytic_win = analytic_win_probs(
                log_lambdas, p_dnfs, sigma_drv=sigma_drv,
                team_indices=team_indices, correlation=correlation,
                n_team_samples=n_team_samples,
            )
        else:
            analytic_win = None

        loss = 0.0
        residuals = {}
        loss_data = 0.0
        loss_team = 0.0
        loss_shrink = 0.0
        loss_sigma = 0.0

        market_cutoffs = {
            "win": 1,
            "podium": 3,
            "top5": 5,
            "top6": 6,
            "top10": 10,
        }

        for market, cutoff in market_cutoffs.items():
            if market not in observed_probs:
                continue
            weight = market_weights.get(market, 1.0)
            for i, obs_p in observed_probs[market].items():
                if market == "win" and analytic_win is not None:
                    model_p = analytic_win[i]
                else:
                    model_p = pos_probs[i, :cutoff].sum()
                residual = model_p - obs_p
                loss_data += weight * residual ** 2
                residuals[(market, i)] = residual

        # DNF probabilities are fixed from odds, not optimized.

        # Regularization: teammates should have similar lambdas
        n_teams = int(team_indices.max()) + 1 if len(team_indices) else 0
        for t in range(n_teams):
            teammates = [j for j in range(n) if team_indices[j] == t]
            if len(teammates) == 2:
                diff = log_lambdas[teammates[0]] - log_lambdas[teammates[1]]
                loss_team += team_reg * diff ** 2

        # Regularization: mild shrinkage toward mean (prevents extreme values)
        mean_ll = log_lambdas.mean()
        loss_shrink = smoothness_reg * np.sum((log_lambdas - mean_ll) ** 2)

        # Regularization: σ_drv toward 1 (i.e. log σ toward 0). Anchors the
        # outer optimization — without this the optimizer can drive σ to
        # extreme values where the Hermite quadrature loses precision.
        offset_reg = n
        if fit_sigma_drv:
            loss_sigma = sigma_drv_reg * np.sum(x[offset_reg:offset_reg + n] ** 2)
            offset_reg += n

        # Regularization: chaos_alpha_drv toward the global default α (i.e.
        # log(α-1) toward log(default-1)). Without this the optimizer pushes
        # individual drivers' α to extreme values.
        loss_alpha = 0.0
        if fit_chaos_alpha_drv:
            z_default = np.log(max(base_alpha - 1.0, 0.001))
            loss_alpha = chaos_alpha_drv_reg * np.sum(
                (x[offset_reg:offset_reg + n] - z_default) ** 2
            )

        loss = loss_data + loss_team + loss_shrink + loss_sigma + loss_alpha

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
                "sigma": round(loss_sigma, 6),
                "alpha": round(loss_alpha, 6),
            })

        # Log every eval with timing
        elapsed = time.time() - start_time[0]
        evals_per_sec = eval_count[0] / max(elapsed, 0.01)
        print(
            f"  eval {eval_count[0]:5d} | "
            f"loss={loss:.6f} (data={loss_data:.6f} team={loss_team:.6f} "
            f"shrink={loss_shrink:.6f} sigma={loss_sigma:.6f}) | "
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
        # Powell tolerances. CRN keeps the surface deterministic so Powell
        # can chase real minima instead of bailing on noisy plateaus
        # (issue #36 had n_steps=8 / final_loss=0.6314). With one-sided
        # chaos the loss landscape is much better-conditioned, so a few
        # Powell sweeps reach near-optimum loss; tight ftol keeps Powell
        # iterating long after the marginal improvements are noise.
        options={"maxiter": 8, "ftol": 1e-4, "xtol": 1e-3},
    )
    elapsed = time.time() - start_time[0]
    print(f"  Converged: {result.success}, final loss: {result.fun:.6f}")
    print(f"  Total: {eval_count[0]} evals, {step_count[0]} steps, {elapsed:.1f}s")
    if hasattr(result, 'message'):
        print(f"  Message: {result.message}")

    log_lambdas, sigma_drv, chaos_alpha_drv = _unpack(result.x)
    p_dnfs = fixed_p_dnfs

    # Normalize: set mean log_lambda to 0 (arbitrary scale)
    log_lambdas = log_lambdas - log_lambdas.mean()

    # Compute final residuals with a large simulation for accuracy
    final_pos_probs = simulate_races(
        log_lambdas, p_dnfs, n_sims=50000, seed=99999,
        team_indices=team_indices, correlation=correlation,
        sigma_drv=sigma_drv,
        chaos_alpha_drv=chaos_alpha_drv,
    )
    market_cutoffs = {"win": 1, "podium": 3, "top5": 5, "top6": 6, "top10": 10}
    residuals = []
    for market, cutoff in market_cutoffs.items():
        if market not in observed_probs:
            continue
        for i, obs_p in observed_probs[market].items():
            model_p = float(final_pos_probs[i, :cutoff].sum())
            residuals.append({
                "market": market,
                "driver_idx": i,
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
        "sigma_drv_reg": sigma_drv_reg if fit_sigma_drv else None,
        "chaos_alpha_drv_reg": chaos_alpha_drv_reg if fit_chaos_alpha_drv else None,
        "fit_sigma_drv": fit_sigma_drv,
        "fit_chaos_alpha_drv": fit_chaos_alpha_drv,
        "market_weights": dict(market_weights),
        "message": result.message if hasattr(result, "message") else "",
        "loss_history": loss_history,
        "step_losses": step_losses,
        "residuals": residuals,
        "correlation": correlation,
        "sigma_drv": [float(s) for s in sigma_drv] if sigma_drv is not None else None,
        "chaos_alpha_drv": [float(a) for a in chaos_alpha_drv] if chaos_alpha_drv is not None else None,
    }

    return log_lambdas, p_dnfs, fit_info


def simulate_event_metrics(
    log_lambdas: np.ndarray,
    p_dnfs: np.ndarray,
    points_table: Dict[int, int],
    n_sims: int = 50000,
    seed: int = 12345,
    team_indices: np.ndarray = None,
    correlation: dict = None,
    sigma_drv: np.ndarray = None,
    chaos_alpha_drv: np.ndarray = None,
) -> dict:
    """
    Run one event's worth of simulations and return per-driver metrics.

    Returns a dict with parallel arrays indexed by driver:
      pos_probs      : (n_drivers, n_drivers + 1) full position distribution
      ep             : (n_drivers,) expected points under this event's table
      std_dev        : (n_drivers,) std dev of points
      p_win, p_podium, p_top6, p_top10, p_no_points : (n_drivers,) summaries

    The event-agnostic split exists so race and sprint can each have their
    own PL fit + simulation on sprint weekends.
    """
    pos_probs = simulate_races(
        log_lambdas, p_dnfs, n_sims=n_sims, seed=seed,
        team_indices=team_indices, correlation=correlation,
        sigma_drv=sigma_drv,
        chaos_alpha_drv=chaos_alpha_drv,
    )

    n = len(log_lambdas)
    ep = np.zeros(n)
    std_dev = np.zeros(n)
    for i in range(n):
        dist = pos_probs[i]
        ep[i] = compute_expected_points(dist, points_table)
        std_dev[i] = np.sqrt(compute_variance(dist, points_table))

    p_win = pos_probs[:, 0]
    p_podium = pos_probs[:, :3].sum(axis=1)
    p_top6 = pos_probs[:, :6].sum(axis=1)
    p_top10 = pos_probs[:, :10].sum(axis=1)
    p_no_points = 1.0 - p_top10 - pos_probs[:, -1]

    return {
        "pos_probs": pos_probs,
        "ep": ep,
        "std_dev": std_dev,
        "p_win": p_win,
        "p_podium": p_podium,
        "p_top6": p_top6,
        "p_top10": p_top10,
        "p_no_points": p_no_points,
    }


def assemble_driver_records(
    drivers: List[dict],
    race_metrics: dict,
    race_log_lambdas: np.ndarray,
    race_p_dnfs: np.ndarray,
    race_sigma_drv: np.ndarray = None,
    race_chaos_alpha_drv: np.ndarray = None,
    sprint_metrics: dict = None,
    sprint_log_lambdas: np.ndarray = None,
    sprint_p_dnfs: np.ndarray = None,
    sprint_sigma_drv: np.ndarray = None,
    sprint_chaos_alpha_drv: np.ndarray = None,
) -> List[dict]:
    """
    Combine race + (optional) sprint event metrics into per-driver output records.

    For sprint weekends, sprint_metrics + sprint_log_lambdas come from a
    separate PL fit on sprint odds; ep_sprint and the *_sprint stats reflect
    the sprint-specific model. For non-sprint weekends, pass sprint_metrics=None.

    The race-fit fields keep their unsuffixed names (lambda, p_dnf,
    position_distribution, p_win, ...) for backward compat with the
    Methodology page and any existing dashboard code. Sprint-fit fields
    get the _sprint suffix.
    """
    n = len(drivers)
    drivers_output = []

    for i in range(n):
        race_dist = race_metrics["pos_probs"][i]
        ep_race = float(race_metrics["ep"][i])

        if sprint_metrics is not None:
            ep_sprint = float(sprint_metrics["ep"][i])
        else:
            ep_sprint = 0.0
        ep_total = ep_race + ep_sprint

        rec = {
            "name": drivers[i]["name"],
            "abbr": drivers[i]["abbr"],
            "team_idx": drivers[i]["team_idx"],
            "lambda": float(race_log_lambdas[i]),
            "sigma_drv": float(race_sigma_drv[i]) if race_sigma_drv is not None else 1.0,
            "chaos_alpha": float(race_chaos_alpha_drv[i]) if race_chaos_alpha_drv is not None else None,
            "p_dnf": float(race_p_dnfs[i]),
            "ep_race": round(ep_race, 2),
            "ep_sprint": round(ep_sprint, 2),
            "ep_total": round(ep_total, 2),
            "std_dev": round(float(race_metrics["std_dev"][i]), 2),
            "p_win": round(float(race_metrics["p_win"][i]), 4),
            "p_podium": round(float(race_metrics["p_podium"][i]), 4),
            "p_top6": round(float(race_metrics["p_top6"][i]), 4),
            "p_top10": round(float(race_metrics["p_top10"][i]), 4),
            "p_no_points": round(float(race_metrics["p_no_points"][i]), 4),
            "position_distribution": [round(float(race_dist[k]), 5) for k in range(len(race_dist))],
        }

        if sprint_metrics is not None:
            sprint_dist = sprint_metrics["pos_probs"][i]
            rec.update({
                "lambda_sprint": float(sprint_log_lambdas[i]) if sprint_log_lambdas is not None else None,
                "sigma_drv_sprint": float(sprint_sigma_drv[i]) if sprint_sigma_drv is not None else 1.0,
                "chaos_alpha_sprint": float(sprint_chaos_alpha_drv[i]) if sprint_chaos_alpha_drv is not None else None,
                "p_dnf_sprint": float(sprint_p_dnfs[i]),
                "std_dev_sprint": round(float(sprint_metrics["std_dev"][i]), 2),
                "p_win_sprint": round(float(sprint_metrics["p_win"][i]), 4),
                "p_podium_sprint": round(float(sprint_metrics["p_podium"][i]), 4),
                "p_top6_sprint": round(float(sprint_metrics["p_top6"][i]), 4),
                "p_top10_sprint": round(float(sprint_metrics["p_top10"][i]), 4),
                "p_no_points_sprint": round(float(sprint_metrics["p_no_points"][i]), 4),
                "position_distribution_sprint": [
                    round(float(sprint_dist[k]), 5) for k in range(len(sprint_dist))
                ],
            })

        drivers_output.append(rec)

    drivers_output.sort(key=lambda d: -d["ep_total"])
    return drivers_output


def generate_full_output(
    log_lambdas: np.ndarray,
    p_dnfs: np.ndarray,
    drivers: List[dict],
    is_sprint: bool = False,
    n_sims: int = 50000,
    team_indices: np.ndarray = None,
    correlation: dict = None,
    sigma_drv: np.ndarray = None,
    chaos_alpha_drv: np.ndarray = None,
) -> List[dict]:
    """
    Backward-compatible single-fit driver output.

    Used when only a race PL fit exists (every non-sprint weekend, or when
    sprint odds aren't available). For sprint weekends with dual fits, call
    simulate_event_metrics + assemble_driver_records directly.
    """
    race_metrics = simulate_event_metrics(
        log_lambdas, p_dnfs, RACE_POINTS,
        n_sims=n_sims, team_indices=team_indices, correlation=correlation,
        sigma_drv=sigma_drv, chaos_alpha_drv=chaos_alpha_drv,
    )
    if is_sprint:
        # Single-fit sprint path: reuse the race PL distribution to score sprint
        # points (legacy behavior — only correct when no separate sprint odds).
        sprint_metrics = {
            "pos_probs": race_metrics["pos_probs"],
            "ep": np.array([
                compute_expected_points(race_metrics["pos_probs"][i], SPRINT_POINTS)
                for i in range(len(log_lambdas))
            ]),
            "std_dev": np.zeros(len(log_lambdas)),
            "p_win": race_metrics["p_win"],
            "p_podium": race_metrics["p_podium"],
            "p_top6": race_metrics["p_top6"],
            "p_top10": race_metrics["p_top10"],
            "p_no_points": race_metrics["p_no_points"],
        }
        # Compute sprint std_dev from the same distribution but sprint scoring
        for i in range(len(log_lambdas)):
            sprint_metrics["std_dev"][i] = np.sqrt(
                compute_variance(race_metrics["pos_probs"][i], SPRINT_POINTS)
            )
        return assemble_driver_records(
            drivers, race_metrics, log_lambdas, p_dnfs,
            race_sigma_drv=sigma_drv, race_chaos_alpha_drv=chaos_alpha_drv,
            sprint_metrics=sprint_metrics,
            sprint_log_lambdas=log_lambdas, sprint_p_dnfs=p_dnfs,
            sprint_sigma_drv=sigma_drv, sprint_chaos_alpha_drv=chaos_alpha_drv,
        )
    return assemble_driver_records(
        drivers, race_metrics, log_lambdas, p_dnfs,
        race_sigma_drv=sigma_drv, race_chaos_alpha_drv=chaos_alpha_drv,
    )


def find_top_lineups(
    drivers_data: List[dict],
    n_picks: int = 5,
    exact_pos_bonus: float = EXACT_BONUS,
    top_n: int = 10,
    score_key: str = "ep_total",
    dist_keys: Tuple[str, ...] = ("position_distribution",),
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

    Parameters
    ----------
    drivers_data : list of driver dicts from generate_full_output
    n_picks : number of picks (default 5)
    exact_pos_bonus : bonus for exact position match (default EXACT_BONUS)
    top_n : number of lineups to return (default 10)
    score_key : driver-dict key for the per-driver base ep used in lineup scoring
        (e.g. "ep_total" for combined, "ep_race", "ep_sprint")
    dist_keys : tuple of driver-dict keys whose position distributions are
        SUMMED to compute the slot bonus. The slot bonus represents
        P(driver finishes pos = slot) summed across applicable events. Use
        ("position_distribution",) for race-only or single-fit weekends, or
        ("position_distribution", "position_distribution_sprint") for the
        combined race+sprint lineup on dual-fit sprint weekends.

    Returns
    -------
    List of dicts (length top_n), each with:
        rank, picks, ep_base_total, ep_bonus_total, ep_grand_total
    """
    n = len(drivers_data)
    ep_totals = np.array([d[score_key] for d in drivers_data])

    # pos_bonus[d, s] = (Σ_event P(driver d finishes position s+1)) * exact_pos_bonus
    pos_bonus = np.zeros((n, n_picks))
    for dk in dist_keys:
        pos_bonus += np.array([
            [d[dk][s] for s in range(n_picks)]
            for d in drivers_data
        ])
    pos_bonus *= exact_pos_bonus

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
    # Quick smoke test against the live roster.
    from roster import fetch_current_roster

    np.random.seed(42)
    drivers = fetch_current_roster()
    team_idx = np.array([d["team_idx"] for d in drivers])
    n = len(drivers)
    fake_lambdas = np.linspace(5.0, 0.3, n)
    fake_dnfs = np.full(n, 0.10)

    print(f"Simulating {n} drivers with fake parameters (with correlation)...")
    pos_probs = simulate_races(
        np.log(fake_lambdas), fake_dnfs, n_sims=50000,
        team_indices=team_idx, correlation=CORRELATION_DEFAULTS,
    )

    print("\nExpected Race Points:")
    for i, d in enumerate(drivers):
        ep = compute_expected_points(pos_probs[i], RACE_POINTS)
        p_win = pos_probs[i, 0]
        p_dnf = pos_probs[i, -1]
        print(f"  {d['name']:25s} E[pts]={ep:6.2f}  P(win)={p_win:.3f}  P(DNF)={p_dnf:.3f}")

    print("\nGenerating full output...")
    output = generate_full_output(
        np.log(fake_lambdas), fake_dnfs, drivers, is_sprint=False,
        team_indices=team_idx, correlation=CORRELATION_DEFAULTS,
    )
    for d in output[:5]:
        print(f"  {d['name']:25s} E[pts]={d['ep_race']:6.2f}  σ={d['std_dev']:.1f}")
