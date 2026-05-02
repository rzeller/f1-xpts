#!/usr/bin/env python3
"""
Backfill old race-output JSON files for the DNF penalty correction.

Context: the league's actual DNF penalty is -10, but the pipeline used -20
until this PR. Every existing race output therefore has wrong ep_race /
ep_sprint / ep_total / std_dev / lineup numbers. Position distributions are
correct (the simulator didn't change), so we can recompute everything
downstream from each file's existing position_distribution arrays without
re-simulating.

Scope: rewrites the per-race latest snapshots only —
  public/data/latest.json
  public/data/races/<slug>.json

Timestamped snapshots under public/data/races/<slug>/<ts>.json are left
untouched: they're historical "what we predicted at the time" artifacts and
shouldn't be silently rewritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Allow running from repo root or pipeline/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RACE_POINTS, SPRINT_POINTS, DNF_PENALTY, EXACT_BONUS
from plackett_luce import (
    compute_expected_points, compute_variance, find_top_lineups,
)
import numpy as np


def _coerce_points_table(raw: dict) -> Dict[int, int]:
    """JSON keys are strings — coerce back to int → int."""
    return {int(k): int(v) for k, v in (raw or {}).items()}


def _recompute_driver(d: dict, race_points: Dict[int, int],
                      sprint_points: Dict[int, int], is_sprint: bool,
                      dnf_penalty: int) -> dict:
    """Recompute ep_*/std_dev for one driver from its existing distribution."""
    dist = np.array(d["position_distribution"], dtype=np.float64)
    ep_race = compute_expected_points(dist, race_points, dnf_penalty=dnf_penalty)
    ep_sprint = (compute_expected_points(dist, sprint_points, dnf_penalty=dnf_penalty)
                 if is_sprint else 0.0)
    var_race = compute_variance(dist, race_points, dnf_penalty=dnf_penalty)

    # If the file has a sprint-specific distribution (post-dual-fit files),
    # use it for sprint EP. This branch is currently a no-op for old files
    # but keeps the script idempotent on newer files.
    if "position_distribution_sprint" in d and is_sprint:
        sprint_dist = np.array(d["position_distribution_sprint"], dtype=np.float64)
        ep_sprint = compute_expected_points(sprint_dist, sprint_points, dnf_penalty=dnf_penalty)
        var_sprint = compute_variance(sprint_dist, sprint_points, dnf_penalty=dnf_penalty)
        d["std_dev_sprint"] = round(float(np.sqrt(var_sprint)), 2)

    d["ep_race"] = round(float(ep_race), 2)
    d["ep_sprint"] = round(float(ep_sprint), 2)
    d["ep_total"] = round(float(ep_race + ep_sprint), 2)
    d["std_dev"] = round(float(np.sqrt(var_race)), 2)
    return d


def backfill_file(path: Path, dry_run: bool = False) -> bool:
    """Rewrite one race-output JSON in place. Returns True if changed."""
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  SKIP {path}: {e}")
        return False

    if "drivers" not in data or "scoring" not in data:
        print(f"  SKIP {path}: not a race-output file")
        return False

    old_dnf = data["scoring"].get("dnf_penalty")
    if old_dnf == DNF_PENALTY and data["scoring"].get("exact_bonus") == EXACT_BONUS:
        print(f"  OK   {path} (already at DNF={DNF_PENALTY}, EXACT_BONUS={EXACT_BONUS})")
        return False

    race_points = _coerce_points_table(data["scoring"].get("race", RACE_POINTS))
    sprint_points = _coerce_points_table(data["scoring"].get("sprint", SPRINT_POINTS))
    is_sprint = bool(data.get("meta", {}).get("is_sprint", False))

    # Recompute per-driver EPs / std_dev.
    for d in data["drivers"]:
        _recompute_driver(d, race_points, sprint_points, is_sprint, DNF_PENALTY)
    data["drivers"].sort(key=lambda d: -d["ep_total"])

    # Recompute combined lineups. We can only do the combined slice for old
    # single-fit files (no per-event distributions). If a file already has
    # split distributions, recompute all three.
    has_dual = any("position_distribution_sprint" in d for d in data["drivers"])

    data["top_lineups"] = find_top_lineups(
        data["drivers"], top_n=len(data.get("top_lineups") or []) or 10,
        score_key="ep_total",
        dist_keys=(("position_distribution", "position_distribution_sprint")
                   if has_dual else ("position_distribution",)),
    )
    if has_dual:
        data["top_lineups_race"] = find_top_lineups(
            data["drivers"], top_n=len(data.get("top_lineups_race") or []) or 10,
            score_key="ep_race", dist_keys=("position_distribution",),
        )
        data["top_lineups_sprint"] = find_top_lineups(
            data["drivers"], top_n=len(data.get("top_lineups_sprint") or []) or 10,
            score_key="ep_sprint", dist_keys=("position_distribution_sprint",),
        )

    data["scoring"]["dnf_penalty"] = DNF_PENALTY
    data["scoring"]["exact_bonus"] = EXACT_BONUS
    data.setdefault("meta", {})["backfilled_at"] = datetime.now(timezone.utc).isoformat()
    data["meta"]["backfill_note"] = (
        f"DNF penalty corrected from {old_dnf} to {DNF_PENALTY}; per-driver EPs "
        f"and lineups recomputed from existing position_distribution arrays."
    )

    if dry_run:
        d0 = data["drivers"][0]
        print(f"  DRY  {path}: top driver {d0['abbr']} ep_total={d0['ep_total']}")
        return True

    with path.open("w") as f:
        json.dump(data, f, indent=2)
    d0 = data["drivers"][0]
    print(f"  OK   {path}: rewrote (top {d0['abbr']} ep_total={d0['ep_total']})")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="public/data",
                        help="Repo data directory (default: public/data)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute new values and report, but don't write files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Targets: public/data/latest.json + public/data/races/*.json (NOT the
    # timestamped snapshots under public/data/races/<slug>/<ts>.json).
    targets: List[Path] = []
    latest = data_dir / "latest.json"
    if latest.exists():
        targets.append(latest)
    races_dir = data_dir / "races"
    if races_dir.exists():
        for p in sorted(races_dir.glob("*.json")):
            if p.name == "index.json":
                continue
            targets.append(p)

    print(f"Backfilling {len(targets)} file(s) (dry_run={args.dry_run}):")
    changed = 0
    for path in targets:
        if backfill_file(path, dry_run=args.dry_run):
            changed += 1
    print(f"\n{changed}/{len(targets)} file(s) {'would be ' if args.dry_run else ''}updated.")


if __name__ == "__main__":
    main()
