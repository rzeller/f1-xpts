#!/usr/bin/env python3
"""
Fetch historical F1 odds from The Odds API historical endpoint.

Downloads pre-race odds snapshots for each race in historical_results.json,
storing one JSON per race in the output directory. Files use the same format
as manual odds input files so the backtest runner can consume them directly.

Usage:
    python -m pipeline.backtest.fetch_historical_odds \
        --api-key YOUR_KEY \
        --output-dir pipeline/historical_odds/ \
        --seasons 2022 2023 2024

The historical endpoint requires a paid plan. Each request costs ~10 quota
units per region per market. Budget ~1,500-3,800 requests for 95 races.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Markets to fetch (API key → our internal key)
MARKETS = {
    "outrights": "win",
    "outrights_top3": "podium",
    "outrights_top6": "top6",
    "outrights_top10": "top10",
}


def _get_with_retry(url: str, params: dict, max_retries: int = 3) -> Optional[dict]:
    """Make a GET request with retry logic and quota tracking."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            remaining = resp.headers.get("x-requests-remaining", "?")
            used = resp.headers.get("x-requests-used", "?")
            print(f"      API quota: {used} used, {remaining} remaining")

            if resp.status_code == 422:
                # No data for this request (e.g., sport not found at timestamp)
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


def discover_f1_sport_key(api_key: str, timestamp: str) -> Optional[str]:
    """
    Find the F1 sport key from the historical sports list at a given timestamp.

    Parameters
    ----------
    timestamp : ISO 8601 timestamp (e.g., "2024-03-01T12:00:00Z")
    """
    data = _get_with_retry(
        f"{ODDS_API_BASE}/historical/sports",
        {"apiKey": api_key, "date": timestamp},
    )
    if not data:
        return None

    # The historical endpoint wraps data differently
    sports = data.get("data", data)
    if isinstance(sports, dict) and "data" in sports:
        sports = sports["data"]

    if not isinstance(sports, list):
        print(f"      Unexpected sports response format")
        return None

    for sport in sports:
        key = sport.get("key", "")
        title = sport.get("title", "").lower()
        if ("formula" in title or "formula" in key or "f1" in key) and sport.get("has_outrights"):
            print(f"    Found F1: key={key}, title={sport.get('title')}")
            return key

    # Fallback: any motorsport with outrights
    for sport in sports:
        key = sport.get("key", "")
        if "motorsport" in key and sport.get("has_outrights"):
            print(f"    Fallback motorsport: key={key}")
            return key

    return None


def fetch_historical_odds(
    api_key: str,
    sport_key: str,
    timestamp: str,
    regions: str = "us,uk,eu,au",
) -> Tuple[Dict[str, Dict[str, float]], dict]:
    """
    Fetch historical odds for all available markets at a given timestamp.

    Returns (raw_odds_by_market, event_info)
    """
    raw_odds = {}
    event_info = {}

    # Fetch outrights (race winner) first
    print(f"    Fetching outrights at {timestamp}...")
    data = _get_with_retry(
        f"{ODDS_API_BASE}/historical/sports/{sport_key}/odds",
        {
            "apiKey": api_key,
            "date": timestamp,
            "regions": regions,
            "markets": "outrights",
            "oddsFormat": "american",
        },
    )

    if not data:
        return {}, {}

    # Historical endpoint wraps in {data: [...], timestamp: ...}
    events = data.get("data", data)
    if isinstance(events, dict) and "data" in events:
        events = events["data"]
    if not isinstance(events, list):
        events = [events]

    if not events:
        return {}, {}

    # Find the F1 race event (take first one)
    event = events[0]
    event_info = {
        "id": event.get("id", ""),
        "title": event.get("sport_title", "F1"),
        "commence_time": event.get("commence_time", ""),
    }

    # Extract odds from bookmakers
    win_odds = _extract_best_odds(event, "outrights")
    if win_odds:
        raw_odds["win"] = win_odds
        print(f"      win: {len(win_odds)} drivers")

    # Try additional markets via event-specific endpoint
    event_id = event.get("id")
    if event_id:
        for api_market, our_key in [
            ("outrights_top3", "podium"),
            ("outrights_top6", "top6"),
            ("outrights_top10", "top10"),
        ]:
            print(f"    Fetching {our_key}...")
            time.sleep(0.5)  # Be polite
            market_data = _get_with_retry(
                f"{ODDS_API_BASE}/historical/sports/{sport_key}/events/{event_id}/odds",
                {
                    "apiKey": api_key,
                    "date": timestamp,
                    "regions": regions,
                    "markets": api_market,
                    "oddsFormat": "american",
                },
            )
            if market_data:
                market_events = market_data.get("data", market_data)
                if isinstance(market_events, dict) and "data" in market_events:
                    market_events = market_events["data"]
                if isinstance(market_events, list) and market_events:
                    market_event = market_events[0]
                elif isinstance(market_events, dict):
                    market_event = market_events
                else:
                    continue
                odds = _extract_best_odds(market_event, api_market)
                if odds:
                    raw_odds[our_key] = odds
                    print(f"      {our_key}: {len(odds)} drivers")

    return raw_odds, event_info


def _extract_best_odds(event: dict, market_key: str) -> Dict[str, float]:
    """Extract best odds per driver from bookmaker data."""
    from collections import defaultdict

    driver_prices = defaultdict(list)
    for bookmaker in event.get("bookmakers", []):
        for mkt in bookmaker.get("markets", []):
            if mkt["key"] == market_key:
                for outcome in mkt["outcomes"]:
                    driver_prices[outcome["name"]].append(outcome["price"])

    odds = {}
    for name, prices in driver_prices.items():
        # Take the best (most favorable) price for each driver
        # For positive American odds: higher is better
        # For negative American odds: closer to 0 is better (less negative)
        # Best = highest implied probability = most favorable to bettor
        best = max(prices, key=lambda p: _american_to_implied(p))
        odds[name] = best

    return odds


def _american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def _race_slug(name: str) -> str:
    """Convert race name to a URL-safe slug."""
    import re
    slug = name.lower()
    slug = slug.replace("grand prix", "gp")
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    return slug


def compute_snapshot_time(
    race_date: str,
    race_time: str = None,
    hours_before: int = 3,
) -> str:
    """
    Compute the timestamp to fetch odds for (race start minus N hours).

    Parameters
    ----------
    race_date : ISO date string (e.g., "2024-03-02")
    race_time : UTC time string from Ergast API (e.g., "05:00:00Z").
        If None, falls back to 14:00 UTC.
    hours_before : hours before race start to snapshot odds

    Returns
    -------
    ISO 8601 timestamp string
    """
    dt = datetime.fromisoformat(race_date)

    if race_time:
        # Parse "HH:MM:SSZ" format from Ergast/Jolpica API
        clean = race_time.rstrip("Z")
        parts = clean.split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        second = int(parts[2]) if len(parts) > 2 else 0
        race_start = dt.replace(hour=hour, minute=minute, second=second, tzinfo=timezone.utc)
    else:
        race_start = dt.replace(hour=14, minute=0, second=0, tzinfo=timezone.utc)

    snapshot = race_start - timedelta(hours=hours_before)
    return snapshot.isoformat()


def load_historical_results(filepath: str) -> list:
    """Load historical race results and extract dates."""
    with open(filepath) as f:
        data = json.load(f)
    return data.get("races", [])


def fetch_race_dates_from_api(season: int, retries: int = 3) -> Dict[int, dict]:
    """
    Fetch race dates and start times for a season from the Jolpica/Ergast API.

    Returns {round: {"date": "2024-03-02", "time": "05:00:00Z"}}
    The time field is the actual race start time in UTC.
    """
    url = f"https://api.jolpi.ca/ergast/f1/{season}.json"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            races = data["MRData"]["RaceTable"]["Races"]
            return {
                int(r["round"]): {
                    "date": r["date"],
                    "time": r.get("time", None),  # e.g. "05:00:00Z"
                }
                for r in races
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                print(f"  ERROR fetching {season} calendar: {e}")
                return {}


def run_fetch(
    api_key: str,
    output_dir: str,
    historical_results_path: str = "pipeline/historical_results.json",
    seasons: Optional[List[int]] = None,
    hours_before: int = 3,
    regions: str = "us,uk,eu,au",
    dry_run: bool = False,
):
    """
    Main entry point: fetch historical odds for all races.

    Supports checkpoint/resume — skips races that already have output files.
    """
    if not HAS_REQUESTS:
        print("ERROR: requests library not installed")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load historical results
    print(f"Loading results from {historical_results_path}...")
    races = load_historical_results(historical_results_path)
    print(f"  Found {len(races)} total races")

    # Filter by seasons if specified
    if seasons:
        races = [r for r in races if r["season"] in seasons]
        print(f"  Filtered to {len(races)} races in seasons {seasons}")

    # Fetch race dates + start times from Jolpica API for each season
    season_dates = {}  # {season: {round: {"date": ..., "time": ...}}}
    needed_seasons = sorted(set(r["season"] for r in races))
    for season in needed_seasons:
        print(f"  Fetching {season} calendar...")
        dates = fetch_race_dates_from_api(season)
        if dates:
            season_dates[season] = dates
            print(f"    Got dates for {len(dates)} rounds")
        time.sleep(0.5)

    # Discover F1 sport key (probe with a recent timestamp)
    probe_race = races[-1]  # Most recent race
    probe_season = probe_race["season"]
    probe_round = probe_race["round"]
    probe_info = season_dates.get(probe_season, {}).get(probe_round, {})
    probe_date = probe_info.get("date", "2024-06-01") if isinstance(probe_info, dict) else probe_info
    probe_ts = compute_snapshot_time(probe_date)

    print(f"\nDiscovering F1 sport key (probing {probe_ts})...")
    sport_key = discover_f1_sport_key(api_key, probe_ts)
    if not sport_key:
        print("ERROR: Could not find F1 sport key in The Odds API")
        sys.exit(1)
    print(f"  Sport key: {sport_key}")

    # Process each race
    fetched = 0
    skipped = 0
    no_data = 0
    errors = 0

    for race in races:
        season = race["season"]
        rnd = race["round"]
        name = race["name"]
        slug = _race_slug(name)
        filename = f"{season}_{rnd:02d}_{slug}.json"
        filepath = output_path / filename

        # Checkpoint: skip if already exists
        if filepath.exists():
            skipped += 1
            continue

        # Get race date and start time
        race_info = season_dates.get(season, {}).get(rnd)
        if not race_info:
            print(f"\n  SKIP {season} R{rnd} {name}: no date found")
            no_data += 1
            continue

        race_date = race_info["date"] if isinstance(race_info, dict) else race_info
        race_time = race_info.get("time") if isinstance(race_info, dict) else None

        timestamp = compute_snapshot_time(race_date, race_time, hours_before)
        print(f"\n  [{season} R{rnd:02d}] {name} — snapshot at {timestamp}")

        if dry_run:
            print(f"    DRY RUN: would fetch → {filename}")
            continue

        # Fetch odds
        try:
            raw_odds, event_info = fetch_historical_odds(
                api_key, sport_key, timestamp, regions
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            errors += 1
            continue

        if not raw_odds:
            print(f"    No odds data available")
            no_data += 1
            # Write a marker file so we don't retry
            marker = {
                "race": name,
                "season": season,
                "round": rnd,
                "date": race_date,
                "snapshot_time": timestamp,
                "source": "the-odds-api-historical",
                "no_data": True,
                "markets": {},
            }
            with open(filepath, "w") as f:
                json.dump(marker, f, indent=2)
            continue

        # Build output in manual odds input format
        output = {
            "race": name,
            "season": season,
            "round": rnd,
            "date": race_date,
            "snapshot_time": timestamp,
            "is_sprint": False,
            "source": "the-odds-api-historical",
            "event_info": event_info,
            "markets": raw_odds,
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        print(f"    Wrote {filename}")
        fetched += 1

        # Rate limit: pause between races
        time.sleep(1)

    print(f"\n{'=' * 60}")
    print(f"Fetch complete:")
    print(f"  Fetched: {fetched}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  No data: {no_data}")
    print(f"  Errors: {errors}")
    print(f"  Total: {fetched + skipped + no_data + errors} / {len(races)}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical F1 odds from The Odds API"
    )
    parser.add_argument(
        "--api-key", "-k",
        help="The Odds API key (or set ODDS_API_KEY env var)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="pipeline/historical_odds",
        help="Output directory for odds files (default: pipeline/historical_odds)",
    )
    parser.add_argument(
        "--results", "-r",
        default="pipeline/historical_results.json",
        help="Path to historical results JSON",
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int,
        help="Seasons to fetch (default: all in results file)",
    )
    parser.add_argument(
        "--hours-before", type=int, default=3,
        help="Hours before race to snapshot odds (default: 3)",
    )
    parser.add_argument(
        "--regions", default="us,uk,eu,au",
        help="Comma-separated regions (default: us,uk,eu,au)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
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
        hours_before=args.hours_before,
        regions=args.regions,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
