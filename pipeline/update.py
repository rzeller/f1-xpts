#!/usr/bin/env python3
"""
F1 Expected Points Pipeline — Main Entry Point

Usage:
  # From manual odds file:
  python update.py --manual data/odds_input/japanese-gp-2026.json

  # From The Odds API:
  python update.py --api-key YOUR_KEY

  # Env var (for GitHub Actions):
  ODDS_API_KEY=xxx python update.py
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Add pipeline dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DRIVERS, TEAMS, N_DRIVERS, N_TEAMS,
    RACE_POINTS, SPRINT_POINTS, DNF_PENALTY,
    SPRINT_WEEKENDS,
)
from odds_fetcher import get_observed_probs
from plackett_luce import (
    fit_plackett_luce,
    generate_full_output,
    simulate_races,
    compute_expected_points,
)


def build_output_json(
    drivers_data: list,
    race_info: dict,
    fit_info: dict,
    log_lambdas: np.ndarray,
    p_dnfs: np.ndarray,
    observed_probs: dict = None,
    n_final_sims: int = 50000,
    devig_method: str = "shin",
) -> dict:
    """Assemble the final JSON that the frontend reads."""
    # Build market input summary
    market_inputs = []
    if observed_probs:
        for market, probs in observed_probs.items():
            market_inputs.append({
                "market": market,
                "n_drivers": len(probs),
                "drivers": [DRIVERS[i]["abbr"] for i in sorted(probs.keys())],
            })

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "race": race_info.get("race", "Unknown"),
            "date": race_info.get("date", ""),
            "is_sprint": race_info.get("is_sprint", False),
            "model": "plackett-luce",
            "n_simulations": n_final_sims,
            "devig_method": devig_method,
            "fit_loss": fit_info.get("loss", None),
            "fit_converged": fit_info.get("success", None),
        },
        "teams": [
            {"name": t["name"], "color": t["color"]}
            for t in TEAMS
        ],
        "scoring": {
            "race": RACE_POINTS,
            "sprint": SPRINT_POINTS,
            "dnf_penalty": DNF_PENALTY,
        },
        "drivers": drivers_data,
        "fit": {
            "method": fit_info.get("method", "unknown"),
            "converged": fit_info.get("success", None),
            "final_loss": fit_info.get("loss", None),
            "n_evals": fit_info.get("n_evals", None),
            "n_steps": fit_info.get("n_steps", None),
            "elapsed_seconds": fit_info.get("elapsed_seconds", None),
            "n_sims_per_eval": fit_info.get("n_sims_per_eval", None),
            "n_params": fit_info.get("n_params", None),
            "team_reg": fit_info.get("team_reg", None),
            "smoothness_reg": fit_info.get("smoothness_reg", None),
            "message": fit_info.get("message", ""),
            "loss_history": fit_info.get("loss_history", []),
            "step_losses": fit_info.get("step_losses", []),
            "residuals": fit_info.get("residuals", []),
            "market_inputs": market_inputs,
        },
    }


def run_pipeline(
    manual_file: str = None,
    api_key: str = None,
    output_dir: str = "data",
    n_fit_sims: int = 20000,
    n_final_sims: int = 50000,
    devig_method: str = "shin",
):
    """Run the full pipeline: fetch odds → fit model → simulate → output JSON."""

    print("=" * 60)
    print("F1 Expected Points Pipeline")
    print("=" * 60)

    # Step 1: Get observed probabilities
    print("\n[1/4] Loading odds data...")
    observed_probs, race_info = get_observed_probs(
        manual_file=manual_file,
        api_key=api_key,
        devig_method=devig_method,
    )

    if not observed_probs:
        print("ERROR: No odds data loaded. Exiting.")
        sys.exit(1)

    # Check if sprint weekend
    race_slug = race_info.get("race", "").lower().replace(" ", "-").replace("grand-prix", "gp")
    if race_slug in SPRINT_WEEKENDS:
        race_info["is_sprint"] = True
        print(f"  Sprint weekend detected: {race_info['race']}")

    # Step 2: Fit the model
    print("\n[2/4] Fitting Plackett-Luce model...")
    team_indices = np.array([d["team_idx"] for d in DRIVERS])

    # If we only have win odds and no other markets, we can still fit
    # (just fewer constraints, so regularization matters more)
    n_markets = len(observed_probs)
    n_constraints = sum(len(v) for v in observed_probs.values())
    print(f"  {n_markets} markets, {n_constraints} total constraints")

    # Fill in missing DNF probs with defaults
    if "dnf" not in observed_probs:
        print("  No DNF odds available, using defaults (10% base)")
        observed_probs["dnf"] = {i: 0.10 for i in range(N_DRIVERS)}

    log_lambdas, p_dnfs, fit_info = fit_plackett_luce(
        observed_probs=observed_probs,
        team_indices=team_indices,
        n_sims=n_fit_sims,
    )

    print(f"\n  Fit complete. Loss: {fit_info['loss']:.6f}")

    # Step 3: Generate full simulation output
    print("\n[3/4] Running final simulation (50K races)...")
    drivers_data = generate_full_output(
        log_lambdas,
        p_dnfs,
        is_sprint=race_info.get("is_sprint", False),
        n_sims=n_final_sims,
    )

    # Print summary
    print("\n  Expected Points Summary:")
    print(f"  {'Rank':>4} {'Driver':20s} {'E[Race]':>8} {'E[Sprint]':>9} {'E[Total]':>8} {'σ':>6} {'P(Win)':>7} {'P(DNF)':>7}")
    print("  " + "-" * 80)
    for rank, d in enumerate(drivers_data[:10], 1):
        print(f"  {rank:4d} {d['name']:20s} {d['ep_race']:8.2f} {d['ep_sprint']:9.2f} {d['ep_total']:8.2f} {d['std_dev']:6.1f} {d['p_win']:7.3f} {d['p_dnf']:7.3f}")

    # Step 4: Write output
    print("\n[4/4] Writing output files...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "races").mkdir(exist_ok=True)

    output = build_output_json(
        drivers_data, race_info, fit_info, log_lambdas, p_dnfs,
        observed_probs=observed_probs,
        n_final_sims=n_final_sims,
        devig_method=devig_method,
    )

    # Write latest.json (what the frontend reads)
    latest_path = output_dir / "latest.json"
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Wrote {latest_path}")

    # Write race-specific snapshot
    race_slug = race_info.get("race", "unknown").lower().replace(" ", "-").replace("grand-prix", "gp")
    race_path = output_dir / "races" / f"{race_slug}.json"
    with open(race_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Wrote {race_path}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    return output


def main():
    parser = argparse.ArgumentParser(description="F1 Expected Points Pipeline")
    parser.add_argument(
        "--manual", "-m",
        help="Path to manual odds JSON file",
    )
    parser.add_argument(
        "--api-key", "-k",
        help="The Odds API key (or set ODDS_API_KEY env var)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--fit-sims",
        type=int,
        default=10000,
        help="Simulations per optimizer evaluation (default: 10000)",
    )
    parser.add_argument(
        "--final-sims",
        type=int,
        default=50000,
        help="Final simulation count (default: 50000)",
    )
    parser.add_argument(
        "--devig-method",
        default="shin",
        choices=["shin", "multiplicative", "power"],
        help="Devigorization method (default: shin)",
    )

    args = parser.parse_args()

    # Resolve API key from args or env
    api_key = args.api_key or os.environ.get("ODDS_API_KEY")

    if not args.manual and not api_key:
        # Look for the most recent manual file
        odds_dir = Path("data/odds_input")
        if odds_dir.exists():
            manual_files = sorted(odds_dir.glob("*.json"), reverse=True)
            if manual_files:
                args.manual = str(manual_files[0])
                print(f"Auto-detected manual file: {args.manual}")
            else:
                parser.error("No --manual file or --api-key provided, and no files in data/odds_input/")
        else:
            parser.error("No --manual file or --api-key provided")

    run_pipeline(
        manual_file=args.manual,
        api_key=api_key,
        output_dir=args.output,
        n_fit_sims=args.fit_sims,
        n_final_sims=args.final_sims,
        devig_method=args.devig_method,
    )


if __name__ == "__main__":
    main()
