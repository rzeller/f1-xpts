#!/usr/bin/env python3
"""
F1 Expected Points Pipeline — Main Entry Point

Usage:
  # Scrape Oddschecker for the next race (default):
  python update.py --output public/data

  # From manual odds file (overrides scraped per-market):
  python update.py --manual public/data/odds_input/japanese-gp-2026.json

  # Manual file only (skip scraping):
  python update.py --manual <file> --no-scrape
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
    RACE_POINTS, SPRINT_POINTS, DNF_PENALTY, EXACT_BONUS,
    SPRINT_WEEKENDS, CORRELATION_DEFAULTS,
)
from roster import fetch_current_roster, teams_from_roster, build_name_map
from odds_fetcher import get_observed_probs
from plackett_luce import (
    fit_plackett_luce,
    generate_full_output,
    simulate_event_metrics,
    assemble_driver_records,
    find_top_lineups,
    simulate_races,
    compute_expected_points,
)


def _fit_section(fit_info: dict, observed_event: dict, roster: list) -> dict:
    """Build the JSON `fit` block for one event's PL fit."""
    market_inputs = []
    for market, probs in (observed_event or {}).items():
        market_inputs.append({
            "market": market,
            "n_drivers": len(probs),
            "drivers": [roster[i]["abbr"] for i in sorted(probs.keys())],
        })

    residuals = []
    for r in fit_info.get("residuals", []):
        idx = r.get("driver_idx")
        residuals.append({
            **r,
            "driver": roster[idx]["abbr"] if idx is not None and 0 <= idx < len(roster) else "?",
        })

    return {
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
        "sigma_drv_reg": fit_info.get("sigma_drv_reg", None),
        "chaos_alpha_drv_reg": fit_info.get("chaos_alpha_drv_reg", None),
        "fit_sigma_drv": fit_info.get("fit_sigma_drv", None),
        "fit_chaos_alpha_drv": fit_info.get("fit_chaos_alpha_drv", None),
        "market_weights": fit_info.get("market_weights", None),
        "message": fit_info.get("message", ""),
        "loss_history": fit_info.get("loss_history", []),
        "step_losses": fit_info.get("step_losses", []),
        "residuals": residuals,
        "market_inputs": market_inputs,
    }


def build_output_json(
    drivers_data: list,
    teams: list,
    roster: list,
    race_info: dict,
    race_fit_info: dict,
    sprint_fit_info: dict = None,
    top_lineups: list = None,
    top_lineups_race: list = None,
    top_lineups_sprint: list = None,
    observed_probs: dict = None,
    n_final_sims: int = 50000,
    devig_method: str = "shin",
    run_type: str = "",
    correlation: dict = None,
    raw_odds: dict = None,
) -> dict:
    """Assemble the final JSON that the frontend reads.

    `observed_probs` is the event-keyed dict {event: {market: {idx: prob}}}.
    `raw_odds` is also event-keyed, before devig.
    """
    has_sprint = sprint_fit_info is not None
    obs_race = (observed_probs or {}).get("race", {})
    obs_sprint = (observed_probs or {}).get("sprint", {})

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "race": race_info.get("race", "Unknown"),
            "date": race_info.get("date", ""),
            "is_sprint": race_info.get("is_sprint", False),
            "model": "plackett-luce",
            "n_simulations": n_final_sims,
            "devig_method": devig_method,
            "fit_loss": race_fit_info.get("loss", None),
            "fit_converged": race_fit_info.get("success", None),
            "fit_loss_sprint": (sprint_fit_info or {}).get("loss"),
            "fit_converged_sprint": (sprint_fit_info or {}).get("success"),
            "run_type": run_type,
            "correlation": correlation,
            # Pre-devig snapshot of the odds that fed this run, keyed by the
            # driver names as the scraper / manual file produced them. Lets us
            # tell scraper / devig / model failures apart in audits (issue #36).
            "raw_odds": raw_odds,
        },
        "teams": teams,
        "scoring": {
            "race": RACE_POINTS,
            "sprint": SPRINT_POINTS,
            "dnf_penalty": DNF_PENALTY,
            "exact_bonus": EXACT_BONUS,
        },
        "drivers": drivers_data,
        "top_lineups": top_lineups,
        "top_lineups_race": top_lineups_race,
        "top_lineups_sprint": top_lineups_sprint,
        "fit": _fit_section(race_fit_info, obs_race, roster),
        "fit_sprint": _fit_section(sprint_fit_info, obs_sprint, roster) if has_sprint else None,
    }


def _write_race_index(races_dir: Path):
    """Scan race snapshot files and write an index.json manifest."""
    races = []
    for race_file in sorted(races_dir.glob("*.json")):
        if race_file.name == "index.json":
            continue
        try:
            with open(race_file) as f:
                meta = json.load(f).get("meta", {})
            races.append({
                "slug": race_file.stem,
                "name": meta.get("race", race_file.stem),
                "date": meta.get("date", ""),
                "is_sprint": meta.get("is_sprint", False),
                "generated_at": meta.get("generated_at", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue

    # Sort by race date
    races.sort(key=lambda r: r["date"])

    # Latest = most recently generated
    latest_slug = ""
    if races:
        latest_slug = max(races, key=lambda r: r["generated_at"])["slug"]

    index = {"races": races, "latest": latest_slug}
    index_path = races_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Wrote {index_path}")


def run_pipeline(
    manual_file: str = None,
    scrape: bool = True,
    output_dir: str = "data",
    n_fit_sims: int = 20000,
    n_final_sims: int = 50000,
    devig_method: str = "shin",
    run_type: str = "",
    sigma_team: float = None,
    sigma_global: float = None,
    sigma_dnf: float = None,
    chaos_model: str = None,
    fit_chaos_alpha_drv: bool = False,
):
    """Run the full pipeline: fetch odds → fit model → simulate → output JSON."""

    # Build correlation parameters: start from defaults (so bimodal-specific
    # knobs flow through automatically), then apply CLI overrides.
    correlation = dict(CORRELATION_DEFAULTS)
    if sigma_team is not None:
        correlation["sigma_team"] = sigma_team
    if sigma_global is not None:
        correlation["sigma_global"] = sigma_global
    if sigma_dnf is not None:
        correlation["sigma_dnf"] = sigma_dnf
    if chaos_model is not None:
        correlation["chaos_model"] = chaos_model

    print("=" * 60)
    print("F1 Expected Points Pipeline")
    print("=" * 60)
    print(f"  Correlation: sigma_team={correlation['sigma_team']}, "
          f"sigma_global={correlation['sigma_global']}, "
          f"sigma_dnf={correlation['sigma_dnf']}, "
          f"chaos_model={correlation['chaos_model']}")

    # Step 0: Fetch the active driver roster + team mapping from Jolpica F1 API.
    # This is the source of truth for who's driving — Oddschecker is matched
    # against it.
    print("\n[0/4] Fetching current roster from Jolpica F1 API...")
    roster = fetch_current_roster()
    teams = teams_from_roster(roster)
    name_map = build_name_map(roster)
    n_drivers = len(roster)
    print(f"  {n_drivers} drivers, {len(teams)} teams")
    for d in roster:
        print(f"    {d['abbr']:4s} {d['name']:25s} → {d['team_name']}")

    # Step 1: Get observed probabilities (event-keyed: {event: {market: {idx: prob}}})
    print("\n[1/4] Loading odds data...")
    observed_probs, race_info, raw_odds = get_observed_probs(
        roster=roster,
        name_map=name_map,
        manual_file=manual_file,
        scrape=scrape,
        devig_method=devig_method,
    )

    if not observed_probs or "race" not in observed_probs:
        print("ERROR: No race odds loaded. Exiting.")
        sys.exit(1)

    # Check if sprint weekend (schedule + observed odds both signal)
    race_slug_for_sprint = race_info.get("race", "").lower().replace(" ", "-").replace("grand-prix", "gp")
    if race_slug_for_sprint in SPRINT_WEEKENDS:
        race_info["is_sprint"] = True
    has_sprint_odds = "sprint" in observed_probs and observed_probs["sprint"]
    if race_info.get("is_sprint") and has_sprint_odds:
        print(f"  Sprint weekend detected with sprint odds: {race_info['race']}")
    elif race_info.get("is_sprint"):
        print(f"  Sprint weekend detected (no sprint-specific odds — sprint EP "
              f"will reuse race PL fit)")

    # Step 2: Fit the model(s) — race always; sprint only if its own odds present.
    team_indices = np.array([d["team_idx"] for d in roster])

    def _fit_event(event_label, event_obs):
        n_markets = len(event_obs)
        n_constraints = sum(len(v) for v in event_obs.values())
        print(f"\n[{event_label}] {n_markets} markets, {n_constraints} constraints")
        if "dnf" not in event_obs:
            print(f"  [{event_label}] No DNF odds available, using defaults (10% base)")
            event_obs = {**event_obs, "dnf": {i: 0.10 for i in range(n_drivers)}}
        ll, pdf, info = fit_plackett_luce(
            observed_probs=event_obs,
            team_indices=team_indices,
            n_sims=n_fit_sims,
            correlation=correlation,
            fit_chaos_alpha_drv=fit_chaos_alpha_drv,
        )
        print(f"  [{event_label}] Fit complete. Loss: {info['loss']:.6f}")
        return ll, pdf, info, event_obs

    print("\n[2/4] Fitting Plackett-Luce model(s)...")
    race_ll, race_pdf, race_fit, observed_probs["race"] = _fit_event("race", observed_probs["race"])

    sprint_ll = sprint_pdf = sprint_fit = None
    if has_sprint_odds:
        sprint_ll, sprint_pdf, sprint_fit, observed_probs["sprint"] = _fit_event(
            "sprint", observed_probs["sprint"]
        )

    # Step 3: Run final simulation per event.
    print("\n[3/4] Running final simulation (50K races) per event...")

    race_sigma_drv = np.array(race_fit["sigma_drv"]) if race_fit.get("sigma_drv") else None
    race_chaos_alpha = (np.array(race_fit["chaos_alpha_drv"])
                        if race_fit.get("chaos_alpha_drv") else None)
    race_metrics = simulate_event_metrics(
        race_ll, race_pdf, RACE_POINTS,
        n_sims=n_final_sims, team_indices=team_indices, correlation=correlation,
        sigma_drv=race_sigma_drv, chaos_alpha_drv=race_chaos_alpha,
    )

    sprint_metrics = None
    sprint_sigma_drv = sprint_chaos_alpha = None
    if sprint_fit is not None:
        sprint_sigma_drv = np.array(sprint_fit["sigma_drv"]) if sprint_fit.get("sigma_drv") else None
        sprint_chaos_alpha = (np.array(sprint_fit["chaos_alpha_drv"])
                              if sprint_fit.get("chaos_alpha_drv") else None)
        sprint_metrics = simulate_event_metrics(
            sprint_ll, sprint_pdf, SPRINT_POINTS,
            n_sims=n_final_sims, team_indices=team_indices, correlation=correlation,
            sigma_drv=sprint_sigma_drv, chaos_alpha_drv=sprint_chaos_alpha,
        )
    elif race_info.get("is_sprint"):
        # Single-fit fallback: reuse race PL distribution scored with sprint table.
        sprint_metrics = simulate_event_metrics(
            race_ll, race_pdf, SPRINT_POINTS,
            n_sims=n_final_sims, team_indices=team_indices, correlation=correlation,
            sigma_drv=race_sigma_drv, chaos_alpha_drv=race_chaos_alpha,
            seed=12346,
        )

    drivers_data = assemble_driver_records(
        roster,
        race_metrics, race_ll, race_pdf,
        race_sigma_drv=race_sigma_drv, race_chaos_alpha_drv=race_chaos_alpha,
        sprint_metrics=sprint_metrics,
        sprint_log_lambdas=(sprint_ll if sprint_ll is not None else (race_ll if sprint_metrics is not None else None)),
        sprint_p_dnfs=(sprint_pdf if sprint_pdf is not None else (race_pdf if sprint_metrics is not None else None)),
        sprint_sigma_drv=sprint_sigma_drv if sprint_fit is not None else (race_sigma_drv if sprint_metrics is not None else None),
        sprint_chaos_alpha_drv=sprint_chaos_alpha if sprint_fit is not None else (race_chaos_alpha if sprint_metrics is not None else None),
    )

    # Print summary
    print("\n  Expected Points Summary:")
    print(f"  {'Rank':>4} {'Driver':20s} {'E[Race]':>8} {'E[Sprint]':>9} {'E[Total]':>8} {'σ':>6} {'P(Win)':>7} {'P(DNF)':>7}")
    print("  " + "-" * 80)
    for rank, d in enumerate(drivers_data[:10], 1):
        print(f"  {rank:4d} {d['name']:20s} {d['ep_race']:8.2f} {d['ep_sprint']:9.2f} {d['ep_total']:8.2f} {d['std_dev']:6.1f} {d['p_win']:7.3f} {d['p_dnf']:7.3f}")

    # Step 3b: Find top lineups (combined; race-only and sprint-only on sprint weekends)
    print("\n[3b/4] Finding top 10 lineups...")
    has_dual_distributions = sprint_metrics is not None and any(
        "position_distribution_sprint" in d for d in drivers_data
    )
    combined_dist_keys = (
        ("position_distribution", "position_distribution_sprint")
        if has_dual_distributions else ("position_distribution",)
    )
    top_lineups = find_top_lineups(
        drivers_data, top_n=10,
        score_key="ep_total", dist_keys=combined_dist_keys,
    )
    for lineup in top_lineups[:3]:
        abbrs = " ".join(f"{p['abbr']}→P{p['slot']}" for p in lineup["picks"])
        print(f"  Combined  #{lineup['rank']}: {abbrs}  →  {lineup['ep_grand_total']:.2f} E[pts]")

    top_lineups_race = None
    top_lineups_sprint = None
    if sprint_metrics is not None:
        top_lineups_race = find_top_lineups(
            drivers_data, top_n=10,
            score_key="ep_race", dist_keys=("position_distribution",),
        )
        for lineup in top_lineups_race[:3]:
            abbrs = " ".join(f"{p['abbr']}→P{p['slot']}" for p in lineup["picks"])
            print(f"  Race-only #{lineup['rank']}: {abbrs}  →  {lineup['ep_grand_total']:.2f} E[pts]")
        if has_dual_distributions:
            top_lineups_sprint = find_top_lineups(
                drivers_data, top_n=10,
                score_key="ep_sprint", dist_keys=("position_distribution_sprint",),
            )
            for lineup in top_lineups_sprint[:3]:
                abbrs = " ".join(f"{p['abbr']}→P{p['slot']}" for p in lineup["picks"])
                print(f"  Sprint    #{lineup['rank']}: {abbrs}  →  {lineup['ep_grand_total']:.2f} E[pts]")

    # Step 4: Write output
    print("\n[4/4] Writing output files...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "races").mkdir(exist_ok=True)

    output = build_output_json(
        drivers_data, teams, roster, race_info,
        race_fit_info=race_fit,
        sprint_fit_info=sprint_fit,
        top_lineups=top_lineups,
        top_lineups_race=top_lineups_race,
        top_lineups_sprint=top_lineups_sprint,
        observed_probs=observed_probs,
        n_final_sims=n_final_sims,
        devig_method=devig_method,
        run_type=run_type,
        correlation=correlation,
        raw_odds=raw_odds,
    )

    # Write latest.json (what the frontend reads)
    latest_path = output_dir / "latest.json"
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Wrote {latest_path}")

    # Write race-specific snapshot (latest for this race)
    race_slug = race_info.get("race", "unknown").lower().replace(" ", "-").replace("grand-prix", "gp")
    race_path = output_dir / "races" / f"{race_slug}.json"
    with open(race_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Wrote {race_path}")

    # Write timestamped snapshot for this run
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = output_dir / "races" / race_slug
    snapshot_dir.mkdir(exist_ok=True)
    snapshot_path = snapshot_dir / f"{ts}.json"
    with open(snapshot_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Wrote {snapshot_path}")

    # Write races/index.json manifest (scan all race snapshots)
    _write_race_index(output_dir / "races")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    return output


def main():
    parser = argparse.ArgumentParser(description="F1 Expected Points Pipeline")
    parser.add_argument(
        "--manual", "-m",
        help="Path to manual odds JSON file (overrides scraped odds per-market)",
    )
    parser.add_argument(
        "--no-scrape",
        action="store_true",
        help="Skip the Oddschecker scrape; use --manual file only",
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
    parser.add_argument(
        "--run-type",
        default="",
        help="Run type label (pre_weekend, pre_qualifying, manual)",
    )
    parser.add_argument(
        "--sigma-team", type=float, default=None,
        help=f"Team race-day noise (default: {CORRELATION_DEFAULTS['sigma_team']})",
    )
    parser.add_argument(
        "--sigma-global", type=float, default=None,
        help=f"Global chaos scaling (default: {CORRELATION_DEFAULTS['sigma_global']})",
    )
    parser.add_argument(
        "--sigma-dnf", type=float, default=None,
        help=f"DNF correlation (default: {CORRELATION_DEFAULTS['sigma_dnf']})",
    )
    parser.add_argument(
        "--chaos-model", type=str, default=None,
        choices=["lomax", "bimodal", "one_sided", "symmetric"],
        help=f"Chaos noise model (default: {CORRELATION_DEFAULTS.get('chaos_model', 'symmetric')})",
    )
    parser.add_argument(
        "--fit-chaos-alpha-drv", action="store_true",
        help="Fit per-driver chaos α (Lomax shape) — adds n params to optimizer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scrape odds and print results, but skip model fitting and file writes",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run scraper in headed mode (visible browser) for local debugging",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="On scrape failures, dump page HTML + screenshots to this directory",
    )

    args = parser.parse_args()

    if args.no_scrape and not args.manual:
        parser.error("--no-scrape requires --manual <file>")

    if args.dry_run:
        print("=" * 60)
        print("F1 Expected Points Pipeline — DRY RUN (scrape only)")
        print("=" * 60)
        try:
            print("\nFetching current roster from Jolpica F1 API...")
            roster = fetch_current_roster()
            name_map = build_name_map(roster)
            print(f"  {len(roster)} drivers, {len(teams_from_roster(roster))} teams")
            observed_probs, race_info, _raw_odds = get_observed_probs(
                roster=roster,
                name_map=name_map,
                manual_file=args.manual,
                scrape=not args.no_scrape,
                devig_method=args.devig_method,
                headed=args.debug,
                debug_dir=args.debug_dir,
            )
        except Exception as e:
            print(f"\nERROR: {e}")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("Dry-run summary")
        print("=" * 60)
        print(f"Race:    {race_info.get('race', '?')}")
        print(f"Date:    {race_info.get('date', '?')}")
        print(f"Sprint:  {race_info.get('is_sprint', False)}")
        print(f"Events:  {list(observed_probs.keys())}")

        for event, event_markets in observed_probs.items():
            print(f"\n=== EVENT: {event} ===")
            print(f"Markets: {list(event_markets.keys())}")
            for market, probs in event_markets.items():
                print(f"\n[{event}/{market}] {len(probs)} drivers — fair probabilities (devigged):")
                ranked = sorted(probs.items(), key=lambda x: -x[1])
                for idx, p in ranked:
                    d = roster[idx]
                    print(f"  {d['abbr']:5s} {d['name']:25s} p={p:.4f}")

        print("\nDry run complete — no files written.")
        return

    run_pipeline(
        manual_file=args.manual,
        scrape=not args.no_scrape,
        output_dir=args.output,
        n_fit_sims=args.fit_sims,
        n_final_sims=args.final_sims,
        devig_method=args.devig_method,
        run_type=args.run_type,
        sigma_team=args.sigma_team,
        sigma_global=args.sigma_global,
        sigma_dnf=args.sigma_dnf,
        chaos_model=args.chaos_model,
        fit_chaos_alpha_drv=args.fit_chaos_alpha_drv,
    )


if __name__ == "__main__":
    main()
