"""
Devigorization methods for converting bookmaker odds to fair probabilities.

Supports:
- Multiplicative (simple normalization)
- Shin's method (accounts for favorite-longshot bias)
- Power method
"""

import numpy as np
from typing import List, Optional


def american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def decimal_to_implied(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1.0 / odds


def fractional_to_implied(num: float, den: float) -> float:
    """Convert fractional odds to implied probability."""
    return den / (num + den)


def devig_multiplicative(implied_probs: np.ndarray) -> np.ndarray:
    """
    Simple multiplicative devig: divide each probability by the sum.
    Fast but doesn't handle favorite-longshot bias.
    """
    total = implied_probs.sum()
    if total <= 0:
        raise ValueError("Implied probabilities sum to zero or negative")
    return implied_probs / total


def devig_shin(implied_probs: np.ndarray, tol: float = 1e-8, max_iter: int = 100) -> np.ndarray:
    """
    Shin's method for devigorization.

    Accounts for favorite-longshot bias by assuming the overround is caused
    by insider trading. Solves for the insider trading parameter z, then
    backs out fair probabilities.

    Reference: Shin, H. (1991). "Optimal Betting Odds Against Insider Traders."

    Parameters
    ----------
    implied_probs : array of implied probabilities (sum > 1 due to vig)
    tol : convergence tolerance
    max_iter : maximum iterations

    Returns
    -------
    fair_probs : array of fair probabilities summing to 1.0
    """
    n = len(implied_probs)
    overround = implied_probs.sum()

    if overround <= 1.0 + tol:
        # No vig to remove, just normalize
        return implied_probs / implied_probs.sum()

    # Solve for Shin's z parameter using bisection
    # z is the probability that the bettor is an insider
    # Fair prob p_i = (sqrt(z^2 + 4*(1-z)*q_i^2/S) - z) / (2*(1-z))
    # where q_i = implied prob, S = sum of implied probs

    def shin_fair_probs(z):
        """Compute fair probs for a given z."""
        fair = np.zeros(n)
        for i in range(n):
            discriminant = z**2 + 4 * (1 - z) * implied_probs[i]**2 / overround
            fair[i] = (np.sqrt(discriminant) - z) / (2 * (1 - z))
        return fair

    def shin_residual(z):
        """Fair probs should sum to 1."""
        return shin_fair_probs(z).sum() - 1.0

    # Bisection search for z in (0, 1)
    z_lo, z_hi = 0.0, 1.0

    for _ in range(max_iter):
        z_mid = (z_lo + z_hi) / 2.0
        residual = shin_residual(z_mid)

        if abs(residual) < tol:
            return shin_fair_probs(z_mid)

        if residual > 0:
            z_lo = z_mid
        else:
            z_hi = z_mid

    # Fallback: return best estimate
    return shin_fair_probs((z_lo + z_hi) / 2.0)


def devig_power(implied_probs: np.ndarray, tol: float = 1e-8, max_iter: int = 100) -> np.ndarray:
    """
    Power method devig: find exponent k such that sum(p_i^k) = 1.

    More accurate than multiplicative for moderate vig, but doesn't
    handle extreme favorite-longshot bias as well as Shin's.
    """
    # Bisection for k in (0, 1) — raising to power < 1 shrinks large values more
    k_lo, k_hi = 0.01, 1.0

    for _ in range(max_iter):
        k_mid = (k_lo + k_hi) / 2.0
        total = np.sum(implied_probs ** k_mid)

        if abs(total - 1.0) < tol:
            return implied_probs ** k_mid

        if total > 1.0:
            k_hi = k_mid  # Need to shrink more
        else:
            k_lo = k_mid

    k_final = (k_lo + k_hi) / 2.0
    result = implied_probs ** k_final
    return result / result.sum()  # Ensure sums to 1


def devig_market(
    odds: dict,
    format: str = "american",
    method: str = "shin",
) -> dict:
    """
    Devig a full market (e.g., all drivers' win odds).

    Parameters
    ----------
    odds : dict mapping driver name/id to odds value
    format : "american", "decimal", or "fractional"
    method : "shin", "multiplicative", or "power"

    Returns
    -------
    dict mapping driver name/id to fair probability
    """
    names = list(odds.keys())
    raw_odds = np.array([odds[name] for name in names], dtype=float)

    # Convert to implied probabilities
    if format == "american":
        implied = np.array([american_to_implied(o) for o in raw_odds])
    elif format == "decimal":
        implied = np.array([decimal_to_implied(o) for o in raw_odds])
    else:
        raise ValueError(f"Unknown format: {format}")

    # Devig
    if method == "shin":
        fair = devig_shin(implied)
    elif method == "multiplicative":
        fair = devig_multiplicative(implied)
    elif method == "power":
        fair = devig_power(implied)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {name: float(p) for name, p in zip(names, fair)}


if __name__ == "__main__":
    # Quick test with example odds
    test_odds = {
        "Russell": -150,
        "Antonelli": 200,
        "Leclerc": 600,
        "Hamilton": 800,
        "Verstappen": 1200,
        "Norris": 1400,
        "Piastri": 1800,
        "Bearman": 3000,
        "Gasly": 4000,
        "Others": 5000,
    }

    for method in ["multiplicative", "shin", "power"]:
        result = devig_market(test_odds, method=method)
        print(f"\n{method.upper()}:")
        for name, p in sorted(result.items(), key=lambda x: -x[1]):
            print(f"  {name:15s} {p:.4f} ({p*100:.1f}%)")
        print(f"  {'SUM':15s} {sum(result.values()):.6f}")
