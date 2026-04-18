#!/usr/bin/env python3
"""
Fetch historical F1 odds from The Odds API historical endpoint.

Downloads pre-race odds snapshots at multiple time points per race weekend,
prioritized by importance for backtesting.

Time point priorities:
  P1: before_qualifying   - Before first qualifying event of the weekend
  P2: after_sprint_qual   - After sprint qualifying (sprint weekends only)
  P3: after_fp2           - After FP2 / before next session
  P4: after_qualifying    - After race-grid qualifying / before race
  P5: before_fp2          - Before FP2
  P6: before_fp1          - Before FP1

Files are stored as: {season}_{round:02d}_{slug}_{timepoint}.json

Usage:
    python -m pipeline.backtest.fetch_historical_odds \
        --api-key YOUR_KEY \
        --output-dir pipeline/historical_odds/ \
        --max-priority 2

Cost: ~20 credits per race per time point (10 for outrights + 10 for placements).
$30/month plan = 20,000 credits. All 6 priorities across ~95 races fits comfortably.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Placement market variants to try. The exact API key names are uncertain,
# so we try multiple formats. These are batched in a single API call (no
# extra cost), and the API ignores unrecognized keys.
PLACEMENT_MARKET_VARIANTS = [
    ("outrights_top4", "top4"),
    ("outrights_top_4", "top4"),
    ("outrights_top6", "top6"),
    ("outrights_top_6", "top6"),
    ("outrights_top3", "podium"),
    ("outrights_top10", "top10"),
]

# Ergast/Jolpica session field names -> our internal names
ERGAST_SESSION_KEYS = {
    "FirstPractice": "fp1",
    "SecondPractice": "fp2",
    "ThirdPractice": "fp3",
    "Qualifying": "qualifying",
    "Sprint": "sprint",
    "SprintQualifying": "sprint_qualifying",
    "SprintShootout": "sprint_qualifying",
}

SESSION_DURATIONS = {
    "fp1": timedelta(hours=1),
    "fp2": timedelta(hours=1),
    "fp3": timedelta(hours=1),
    "qualifying": timedelta(hours=1, minutes=15),
    "sprint_qualifying": timedelta(minutes=45),
    "sprint": timedelta(minutes=45),
}


# ────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────


@dataclass
class SessionSchedule:
    """Session start times for a race weekend (all UTC datetimes)."""

    fp1: Optional[datetime] = None
    fp2: Optional[datetime] = None
    fp3: Optional[datetime] = None
    qualifying: Optional[datetime] = None
    sprint_qualifying: Optional[datetime] = None
    sprint: Optional[datetime] = None
    race: Optional[datetime] = None

    @property
    def is_sprint_weekend(self) -> bool:
        return self.sprint is not None or self.sprint_qualifying is not None


@dataclass
class TimePointDef:
    """Definition for a time point to fetch odds at."""

    name: str
    priority: int
    description: str
    sprint_only: bool = False


TIME_POINTS = [
    TimePointDef("before_qualifying", 1, "Before first qualifying event"),
    TimePointDef("after_sprint_qual", 2, "After sprint qualifying", sprint_only=True),
    TimePointDef("after_fp2", 3, "After FP2"),
    TimePointDef("after_qualifying", 4, "After race-grid qualifying"),
    TimePointDef("before_fp2", 5, "Before FP2"),
    TimePointDef("before_fp1", 6, "Before FP1"),
]


class BudgetExhausted(Exception):
    pass


@dataclass
class BudgetTracker:
    """Track API credits used and remaining."""

    limit: int = 20000
    used: int = 0
    remaining: Optional[int] = None
    reserve: int = 100

    def check(self, cost: int = 10):
        if self.remaining is not None and self.remaining < self.reserve:
            raise BudgetExhausted(
                f"Only {self.remaining} credits remaining (reserve={self.reserve})"
            )
        if self.used + cost > self.limit - self.reserve:
            raise BudgetExhausted(
                f"Would exceed budget: {self.used}+{cost} > {self.limit - self.reserve}"
            )

    def update_from_headers(self, headers: dict):
        rem = headers.get("x-requests-remaining")
        used = headers.get("x-requests-used")
        if rem is not None:
            self.remaining = int(rem)
        if used is not None:
            self.used = int(used)


# ────────────────────────────────────────────────────────────────
# Time point computation
# ────────────────────────────────────────────────────────────────


def compute_time_point(tp: TimePointDef, schedule: SessionSchedule) -> Optional[datetime]:
    """
    Compute the UTC datetime for a time point given the session schedule.
    Returns None if the time point is not applicable for this race.
    """
    if tp.name == "before_qualifying":
        first_qual = schedule.sprint_qualifying or schedule.qualifying
        if first_qual:
            return first_qual - timedelta(hours=1)

    elif tp.name == "after_sprint_qual":
        if not schedule.is_sprint_weekend:
            return None
        if schedule.sprint_qualifying:
            # 2023+: explicit sprint qualifying session
            dur = SESSION_DURATIONS["sprint_qualifying"]
            return schedule.sprint_qualifying + dur + timedelta(minutes=30)
        # 2021-2022: regular qualifying on Friday determines sprint grid
        if schedule.qualifying and schedule.sprint:
            if schedule.qualifying < schedule.sprint:
                dur = SESSION_DURATIONS["qualifying"]
                return schedule.qualifying + dur + timedelta(minutes=30)

    elif tp.name == "after_fp2":
        if schedule.fp2:
            dur = SESSION_DURATIONS["fp2"]
            return schedule.fp2 + dur + timedelta(minutes=30)

    elif tp.name == "after_qualifying":
        if schedule.is_sprint_weekend:
            # 2023+: both sprint_qualifying and qualifying exist
            if schedule.sprint_qualifying and schedule.qualifying:
                dur = SESSION_DURATIONS["qualifying"]
                return schedule.qualifying + dur + timedelta(minutes=30)
            # 2021-2022: sprint result determines race grid
            if schedule.sprint:
                dur = SESSION_DURATIONS["sprint"]
                return schedule.sprint + dur + timedelta(minutes=30)
        elif schedule.qualifying:
            dur = SESSION_DURATIONS["qualifying"]
            return schedule.qualifying + dur + timedelta(minutes=30)

    elif tp.name == "before_fp2":
        if schedule.fp2:
            return schedule.fp2 - timedelta(hours=1)

    elif tp.name == "before_fp1":
        if schedule.fp1:
            return schedule.fp1 - timedelta(hours=1)

    return None


# ────────────────────────────────────────────────────────────────
# HTTP helpers
# ────────────────────────────────────────────────────────────────


def _get_with_retry(
    url: str,
    params: dict,
    budget: Optional[BudgetTracker] = None,
    max_retries: int = 3,
) -> Optional[dict]:
    """Make a GET request with retry logic and budget tracking."""
    if budget:
        budget.check()

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)

            if budget:
                budget.update_from_headers(dict(resp.headers))
            remaining = resp.headers.get("x-requests-remaining", "?")
            used = resp.headers.get("x-requests-used", "?")
            print(f"      [credits: {used} used, {remaining} remaining]")

            if resp.status_code == 422:
                print(f"      No data (422)")
                return None
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"      Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"      Retry {attempt + 1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"      ERROR after {max_retries} attempts: {e}")
                return None
    return None


# ────────────────────────────────────────────────────────────────
# Odds API interaction
# ────────────────────────────────────────────────────────────────


def discover_f1_sport_key(
    api_key: str, timestamp: str, budget: Optional[BudgetTracker] = None
) -> Optional[str]:
    """Find the F1 sport key from the historical sports list."""
    data = _get_with_retry(
        f"{ODDS_API_BASE}/historical/sports",
        {"apiKey": api_key, "date": timestamp},
        budget,
    )
    if not data:
        return None

    sports = data.get("data", data)
    if isinstance(sports, dict) and "data" in sports:
        sports = sports["data"]
    if not isinstance(sports, list):
        print(f"      Unexpected sports response format")
        return None

    for sport in sports:
        key = sport.get("key", "")
        title = sport.get("title", "").lower()
        if ("formula" in title or "formula" in key or "f1" in key) and sport.get(
            "has_outrights"
        ):
            print(f"    Found F1: key={key}, title={sport.get('title')}")
            return key

    for sport in sports:
        key = sport.get("key", "")
        if "motorsport" in key and sport.get("has_outrights"):
            print(f"    Fallback motorsport: key={key}")
            return key

    return None


def _extract_best_odds(event: dict, market_key: str) -> Dict[str, float]:
    """Extract best odds per driver from bookmaker data."""
    driver_prices = defaultdict(list)
    for bookmaker in event.get("bookmakers", []):
        for mkt in bookmaker.get("markets", []):
            if mkt["key"] == market_key:
                for outcome in mkt["outcomes"]:
                    driver_prices[outcome["name"]].append(outcome["price"])

    odds = {}
    for name, prices in driver_prices.items():
        best = max(prices, key=_american_to_implied)
        odds[name] = best
    return odds


def _american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def _unwrap_historical_response(data: dict):
    """Unwrap the historical endpoint's {data: ...} wrapper."""
    result = data.get("data", data)
    if isinstance(result, dict) and "data" in result:
        result = result["data"]
    return result


def fetch_odds_at_timestamp(
    api_key: str,
    sport_key: str,
    timestamp: str,
    regions: str,
    budget: Optional[BudgetTracker] = None,
) -> Tuple[Dict[str, Dict[str, float]], dict]:
    """
    Fetch odds for all available markets at a given timestamp.

    Returns (markets_dict, event_info) where markets_dict maps our internal
    market names (win, top4, top6, etc.) to {driver_name: american_odds}.
    """
    raw_odds = {}
    event_info = {}

    # 1. Fetch outrights (race winner) — also gives us the event ID
    print(f"    Fetching outrights...")
    data = _get_with_retry(
        f"{ODDS_API_BASE}/historical/sports/{sport_key}/odds",
        {
            "apiKey": api_key,
            "date": timestamp,
            "regions": regions,
            "markets": "outrights",
            "oddsFormat": "american",
        },
        budget,
    )

    if not data:
        return {}, {}

    events = _unwrap_historical_response(data)
    if not isinstance(events, list):
        events = [events]
    if not events:
        return {}, {}

    event = events[0]
    event_info = {
        "id": event.get("id", ""),
        "title": event.get("sport_title", "F1"),
        "commence_time": event.get("commence_time", ""),
    }

    win_odds = _extract_best_odds(event, "outrights")
    if win_odds:
        raw_odds["win"] = win_odds
        print(f"      win: {len(win_odds)} drivers")

    # 2. Try placement markets via event-specific endpoint
    event_id = event.get("id")
    if event_id:
        batch_keys = ",".join(v[0] for v in PLACEMENT_MARKET_VARIANTS)
        print(f"    Fetching placement markets...")
        time.sleep(0.3)

        market_data = _get_with_retry(
            f"{ODDS_API_BASE}/historical/sports/{sport_key}/events/{event_id}/odds",
            {
                "apiKey": api_key,
                "date": timestamp,
                "regions": regions,
                "markets": batch_keys,
                "oddsFormat": "american",
            },
            budget,
        )

        if market_data:
            market_event = _unwrap_historical_response(market_data)
            if isinstance(market_event, list) and market_event:
                market_event = market_event[0]

            if isinstance(market_event, dict):
                seen_internal = set()
                for api_mkt_key, internal_key in PLACEMENT_MARKET_VARIANTS:
                    if internal_key in seen_internal:
                        continue
                    odds = _extract_best_odds(market_event, api_mkt_key)
                    if odds:
                        raw_odds[internal_key] = odds
                        seen_internal.add(internal_key)
                        print(f"      {internal_key}: {len(odds)} drivers")

    return raw_odds, event_info


# ────────────────────────────────────────────────────────────────
# Session schedule from Jolpica/Ergast API
# ────────────────────────────────────────────────────────────────


def _parse_session_datetime(session_data: dict) -> Optional[datetime]:
    """Parse {date, time} dict from Ergast API into a UTC datetime."""
    if not session_data or "date" not in session_data:
        return None
    date_str = session_data["date"]
    time_str = session_data.get("time", "14:00:00Z")
    if time_str:
        clean = time_str.rstrip("Z")
        return datetime.fromisoformat(f"{date_str}T{clean}+00:00")
    return datetime.fromisoformat(f"{date_str}T14:00:00+00:00")


def fetch_season_schedule(season: int, retries: int = 3) -> Dict[int, SessionSchedule]:
    """
    Fetch full session schedule for every race in a season from Jolpica API.
    Returns {round_number: SessionSchedule}.
    """
    url = f"https://api.jolpi.ca/ergast/f1/{season}.json"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            races = data["MRData"]["RaceTable"]["Races"]

            result = {}
            for r in races:
                rnd = int(r["round"])
                schedule = SessionSchedule()

                schedule.race = _parse_session_datetime(
                    {"date": r["date"], "time": r.get("time")}
                )

                for ergast_key, internal_key in ERGAST_SESSION_KEYS.items():
                    if ergast_key in r:
                        dt = _parse_session_datetime(r[ergast_key])
                        if dt and getattr(schedule, internal_key) is None:
                            setattr(schedule, internal_key, dt)

                result[rnd] = schedule
            return result
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                print(f"  ERROR fetching {season} schedule: {e}")
                return {}
    return {}


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────


def _race_slug(name: str) -> str:
    """Convert race name to a URL-safe slug."""
    slug = name.lower().replace("grand prix", "gp")
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def load_historical_results(filepath: str) -> list:
    """Load historical race results."""
    with open(filepath) as f:
        data = json.load(f)
    return data.get("races", [])


def compute_snapshot_time(
    race_date: str,
    race_time: str = None,
    hours_before: int = 3,
) -> str:
    """
    Compute a snapshot timestamp relative to race start.
    Kept for backward compatibility with other callers.
    """
    dt = datetime.fromisoformat(race_date)
    if race_time:
        clean = race_time.rstrip("Z")
        parts = clean.split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        second = int(parts[2]) if len(parts) > 2 else 0
        race_start = dt.replace(
            hour=hour, minute=minute, second=second, tzinfo=timezone.utc
        )
    else:
        race_start = dt.replace(hour=14, minute=0, second=0, tzinfo=timezone.utc)
    snapshot = race_start - timedelta(hours=hours_before)
    return snapshot.isoformat()


# ────────────────────────────────────────────────────────────────
# Main fetch logic
# ────────────────────────────────────────────────────────────────


def run_fetch(
    api_key: str,
    output_dir: str,
    historical_results_path: str = "pipeline/historical_results.json",
    seasons: Optional[List[int]] = None,
    max_priority: int = 6,
    regions: str = "us,uk,eu,au",
    dry_run: bool = False,
    budget_limit: int = 20000,
):
    """
    Fetch historical odds at multiple time points per race, in priority order.

    Processes all races at P1 first, then all at P2, etc. This ensures the
    most important snapshots are captured before the budget runs out.
    Supports checkpoint/resume -- skips races that already have output files.
    """
    if not HAS_REQUESTS:
        print("ERROR: requests library not installed")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    budget = BudgetTracker(limit=budget_limit)

    # Load historical results
    print(f"Loading results from {historical_results_path}...")
    races = load_historical_results(historical_results_path)
    if seasons:
        races = [r for r in races if r["season"] in seasons]
    print(f"  {len(races)} races to process")

    # Fetch full session schedules from Jolpica API
    schedules: Dict[int, Dict[int, SessionSchedule]] = {}
    needed_seasons = sorted(set(r["season"] for r in races))
    for season in needed_seasons:
        print(f"  Fetching {season} session schedule...")
        s = fetch_season_schedule(season)
        if s:
            schedules[season] = s
            sprint_count = sum(1 for sch in s.values() if sch.is_sprint_weekend)
            print(f"    {len(s)} rounds ({sprint_count} sprint weekends)")
        time.sleep(0.5)

    # Discover F1 sport key using the most recent race as probe
    sport_key = None
    if not dry_run:
        probe_race = races[-1]
        probe_schedule = schedules.get(probe_race["season"], {}).get(
            probe_race["round"]
        )
        if probe_schedule and probe_schedule.race:
            probe_ts = (probe_schedule.race - timedelta(hours=3)).isoformat()
        else:
            probe_ts = f"2024-06-01T11:00:00+00:00"

        print(f"\nDiscovering F1 sport key...")
        sport_key = discover_f1_sport_key(api_key, probe_ts, budget)
        if not sport_key:
            print("ERROR: Could not find F1 sport key in The Odds API")
            sys.exit(1)
        print(f"  Sport key: {sport_key}")

    # Estimate total work
    active_tps = [tp for tp in TIME_POINTS if tp.priority <= max_priority]
    total_possible = 0
    for tp in active_tps:
        for race in races:
            schedule = schedules.get(race["season"], {}).get(race["round"])
            if not schedule:
                continue
            if tp.sprint_only and not schedule.is_sprint_weekend:
                continue
            if compute_time_point(tp, schedule) is not None:
                total_possible += 1

    print(f"\n  Time points to fetch (max priority {max_priority}): {total_possible}")
    print(f"  Estimated cost: ~{total_possible * 20} credits")
    print(f"  Budget: {budget_limit} credits")

    # Process in priority order: all races at P1, then P2, etc.
    stats = {
        "fetched": 0,
        "skipped": 0,
        "no_data": 0,
        "no_session": 0,
        "errors": 0,
    }

    try:
        for tp in active_tps:
            print(f"\n{'=' * 60}")
            print(f"Priority {tp.priority}: {tp.name} -- {tp.description}")
            if tp.sprint_only:
                print(f"  (sprint weekends only)")
            print(f"{'=' * 60}")

            for race in races:
                season = race["season"]
                rnd = race["round"]
                name = race["name"]
                slug = _race_slug(name)
                filename = f"{season}_{rnd:02d}_{slug}_{tp.name}.json"
                filepath = output_path / filename

                # Checkpoint: skip existing files
                if filepath.exists():
                    stats["skipped"] += 1
                    continue

                # Get session schedule
                schedule = schedules.get(season, {}).get(rnd)
                if not schedule:
                    stats["no_session"] += 1
                    continue

                # Skip non-sprint races for sprint-only time points
                if tp.sprint_only and not schedule.is_sprint_weekend:
                    continue

                # Compute the timestamp for this time point
                ts_dt = compute_time_point(tp, schedule)
                if ts_dt is None:
                    stats["no_session"] += 1
                    continue

                timestamp = ts_dt.isoformat()
                print(f"\n  [{season} R{rnd:02d}] {name} @ {tp.name}")
                print(f"    Timestamp: {timestamp}")

                if dry_run:
                    print(f"    DRY RUN -> {filename}")
                    continue

                # Fetch odds at this timestamp
                try:
                    raw_odds, event_info = fetch_odds_at_timestamp(
                        api_key, sport_key, timestamp, regions, budget
                    )
                except BudgetExhausted as e:
                    print(f"\n  BUDGET EXHAUSTED: {e}")
                    raise
                except Exception as e:
                    print(f"    ERROR: {e}")
                    stats["errors"] += 1
                    continue

                race_date_str = (
                    schedule.race.date().isoformat() if schedule.race else None
                )

                if not raw_odds:
                    print(f"    No odds data available")
                    stats["no_data"] += 1
                    marker = {
                        "race": name,
                        "season": season,
                        "round": rnd,
                        "date": race_date_str,
                        "time_point": tp.name,
                        "snapshot_time": timestamp,
                        "source": "the-odds-api-historical",
                        "no_data": True,
                        "markets": {},
                    }
                    with open(filepath, "w") as f:
                        json.dump(marker, f, indent=2)
                    continue

                output = {
                    "race": name,
                    "season": season,
                    "round": rnd,
                    "date": race_date_str,
                    "time_point": tp.name,
                    "snapshot_time": timestamp,
                    "is_sprint": schedule.is_sprint_weekend,
                    "source": "the-odds-api-historical",
                    "event_info": event_info,
                    "markets": raw_odds,
                }

                with open(filepath, "w") as f:
                    json.dump(output, f, indent=2)
                market_names = ", ".join(sorted(raw_odds.keys()))
                print(f"    Wrote {filename} ({market_names})")
                stats["fetched"] += 1

                time.sleep(0.5)

    except BudgetExhausted:
        print(f"\n  Stopping early due to budget limit.")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Fetch complete:")
    print(f"  Fetched:    {stats['fetched']}")
    print(f"  Skipped:    {stats['skipped']} (already exist)")
    print(f"  No data:    {stats['no_data']} (API had no odds)")
    print(f"  No session: {stats['no_session']} (missing schedule info)")
    print(f"  Errors:     {stats['errors']}")
    if budget.remaining is not None:
        print(f"  Credits remaining: {budget.remaining}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical F1 odds at multiple time points per race weekend"
    )
    parser.add_argument(
        "--api-key",
        "-k",
        help="The Odds API key (or set ODDS_API_KEY env var)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="pipeline/historical_odds",
        help="Output directory for odds files (default: pipeline/historical_odds)",
    )
    parser.add_argument(
        "--results",
        "-r",
        default="pipeline/historical_results.json",
        help="Path to historical results JSON",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        help="Seasons to fetch (default: all in results file)",
    )
    parser.add_argument(
        "--max-priority",
        "-p",
        type=int,
        default=6,
        help="Max time point priority to fetch, 1-6 (default: 6 = all)",
    )
    parser.add_argument(
        "--regions",
        default="us,uk,eu,au",
        help="Comma-separated bookmaker regions (default: us,uk,eu,au)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=20000,
        help="API credit budget limit (default: 20000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without making API calls",
    )

    args = parser.parse_args()
    api_key = args.api_key or os.environ.get("ODDS_API_KEY")
    if not api_key and not args.dry_run:
        parser.error("--api-key or ODDS_API_KEY env var required")

    run_fetch(
        api_key=api_key or "",
        output_dir=args.output_dir,
        historical_results_path=args.results,
        seasons=args.seasons,
        max_priority=args.max_priority,
        regions=args.regions,
        dry_run=args.dry_run,
        budget_limit=args.budget,
    )


if __name__ == "__main__":
    main()
