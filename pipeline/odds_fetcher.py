"""
Odds fetcher: retrieves F1 race odds from The Odds API or manual JSON files.

The Odds API (the-odds-api.com) provides structured odds data.
Manual JSON input is the fallback for markets the API doesn't cover.
Both sources can be combined — manual overrides API for the same market.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from config import DRIVERS, DRIVER_NAME_MAP, N_DRIVERS
from devig import devig_market, american_to_implied, devig_shin


ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def resolve_driver_index(name: str, driver_name_map: dict = None) -> Optional[int]:
    """Match a name from odds data to our canonical driver index.

    Parameters
    ----------
    name : driver name string from odds source
    driver_name_map : optional override for name→index mapping.
        If None, uses the global DRIVER_NAME_MAP from config.py.
    """
    name_map = driver_name_map if driver_name_map is not None else DRIVER_NAME_MAP
    key = name.lower().strip()
    if key in name_map:
        return name_map[key]

    # Try each word (handles "George Russell" → match on "russell")
    for word in key.split():
        if word in name_map:
            return name_map[word]

    return None


def _get(url: str, params: dict) -> dict:
    """Make a GET request, print remaining API quota from headers."""
    resp = requests.get(url, params=params)
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    print(f"    API quota: {used} used, {remaining} remaining")
    resp.raise_for_status()
    return resp.json()


def discover_f1_sport_key(api_key: str) -> Optional[str]:
    """Find the correct sport key for F1 from The Odds API sports list."""
    data = _get(f"{ODDS_API_BASE}/sports", {"apiKey": api_key, "all": "true"})
    for sport in data:
        key = sport.get("key", "")
        title = sport.get("title", "").lower()
        if "formula" in title or "formula" in key or "f1" in key:
            print(f"  Found F1: key={key}, title={sport.get('title')}, active={sport.get('active')}")
            if sport.get("active"):
                return key
    # Fallback: check for any motorsport with outrights
    for sport in data:
        key = sport.get("key", "")
        if "motorsport" in key and sport.get("has_outrights"):
            print(f"  Fallback motorsport outright: key={key}, title={sport.get('title')}")
            return key
    return None


def fetch_f1_events(api_key: str, sport_key: str) -> list:
    """Get list of upcoming F1 events (races)."""
    data = _get(f"{ODDS_API_BASE}/sports/{sport_key}/events", {"apiKey": api_key})
    print(f"  Found {len(data)} F1 events")
    for evt in data[:3]:
        print(f"    {evt.get('id', '?')[:16]}... {evt.get('commence_time', '?')}")
    return data


def fetch_odds_for_market(
    api_key: str,
    sport_key: str,
    event_id: str = None,
    market: str = "outrights",
    regions: str = "us,uk,eu,au",
) -> Tuple[Dict[str, float], dict]:
    """
    Fetch odds for a specific F1 market.

    Returns (odds_dict, event_info) where odds_dict maps driver name → American odds.
    """
    if not HAS_REQUESTS:
        raise RuntimeError("requests library not installed")

    if event_id:
        url = f"{ODDS_API_BASE}/sports/{sport_key}/events/{event_id}/odds"
    else:
        url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": market,
        "oddsFormat": "american",
    }

    data = _get(url, params)

    events = data if isinstance(data, list) else [data]
    if not events:
        return {}, {}

    event = events[0]
    event_info = {
        "id": event.get("id", ""),
        "title": event.get("sport_title", "F1"),
        "commence_time": event.get("commence_time", ""),
    }

    # Aggregate across bookmakers: take best odds (highest implied prob)
    driver_prices = defaultdict(list)
    for bookmaker in event.get("bookmakers", []):
        for mkt in bookmaker.get("markets", []):
            if mkt["key"] == market:
                for outcome in mkt["outcomes"]:
                    driver_prices[outcome["name"]].append(outcome["price"])

    odds = {}
    for name, prices in driver_prices.items():
        best = max(prices, key=lambda p: american_to_implied(p))
        odds[name] = best

    n_books = len(event.get("bookmakers", []))
    print(f"  {market}: {len(odds)} drivers from {n_books} bookmakers")
    return odds, event_info


def fetch_all_f1_odds(api_key: str) -> Tuple[dict, dict]:
    """
    Fetch all available F1 odds from The Odds API.

    Returns (raw_odds_by_market, race_info)
    """
    print("Discovering F1 sport key...")
    sport_key = discover_f1_sport_key(api_key)
    if not sport_key:
        print("  ERROR: Could not find F1 in The Odds API")
        return {}, {}

    raw_odds = {}
    race_info = {"race": "Next F1 Race", "date": "", "is_sprint": False}

    # Fetch race winner (outrights)
    print(f"\nFetching race winner odds...")
    win_odds, evt_info = fetch_odds_for_market(api_key, sport_key, market="outrights")
    if win_odds:
        raw_odds["win"] = win_odds
        race_info["race"] = evt_info.get("title", "F1 Race")
        race_info["date"] = evt_info.get("commence_time", "")

    # Try events endpoint for additional markets
    try:
        events = fetch_f1_events(api_key, sport_key)
        if events:
            event_id = events[0]["id"]
            for api_market, our_key in [
                ("outrights_top3", "podium"),
                ("outrights_top6", "top6"),
                ("outrights_top10", "top10"),
            ]:
                try:
                    print(f"\nFetching {our_key} odds...")
                    odds, _ = fetch_odds_for_market(
                        api_key, sport_key, event_id, market=api_market
                    )
                    if odds:
                        raw_odds[our_key] = odds
                except Exception as e:
                    print(f"  {api_market} not available: {e}")
    except Exception as e:
        print(f"  Could not fetch events: {e}")

    return raw_odds, race_info


def load_manual_odds(filepath: str) -> dict:
    """
    Load manually-entered odds from a JSON file.

    Expected format: see CLAUDE.md "Manual Odds Input Template"
    """
    with open(filepath) as f:
        data = json.load(f)
    return data


def process_odds_to_fair_probs(
    raw_odds: dict,
    devig_method: str = "shin",
    driver_name_map: dict = None,
) -> Dict[str, Dict[int, float]]:
    """
    Convert raw odds (from API or manual) into fair probabilities
    mapped to canonical driver indices.
    """
    observed_probs = {}

    for market_name, market_odds in raw_odds.items():
        if not market_odds:
            continue

        # DNF is a binary market per driver, not a multi-runner market
        if market_name == "dnf":
            probs = {}
            for name, odds_val in market_odds.items():
                idx = resolve_driver_index(name, driver_name_map)
                if idx is None:
                    print(f"  Warning: could not match '{name}'")
                    continue
                implied = american_to_implied(odds_val)
                probs[idx] = implied * 0.95  # Light devig for binary
            observed_probs["dnf"] = probs
            continue

        # Placement markets (podium, top6, top10) are binary yes/no per driver:
        # "Will this driver finish in the top N?" Each driver's prob is independent,
        # so they should sum to N (not 1). Devig each as a binary market.
        #
        # Win market is a true multi-runner outright: exactly one winner,
        # probabilities should sum to 1. Use Shin's method.
        placement_slots = {"podium": 3, "top6": 6, "top10": 10}

        if market_name in placement_slots:
            probs = {}
            target_sum = placement_slots[market_name]
            # Devig as binary markets, then rescale so they sum to target
            raw_implied = {}
            for name, odds_val in market_odds.items():
                idx = resolve_driver_index(name, driver_name_map)
                if idx is None:
                    print(f"  Warning: could not match '{name}'")
                    continue
                raw_implied[idx] = american_to_implied(odds_val)

            # The overround is distributed across all runners
            total_implied = sum(raw_implied.values())
            scale = target_sum / total_implied
            for idx, imp in raw_implied.items():
                probs[idx] = imp * scale

            observed_probs[market_name] = probs
        else:
            # Win market: true multi-runner outright, devig with Shin's method
            fair = devig_market(market_odds, format="american", method=devig_method)

            probs = {}
            for name, prob in fair.items():
                idx = resolve_driver_index(name, driver_name_map)
                if idx is None:
                    print(f"  Warning: could not match '{name}'")
                    continue
                probs[idx] = prob

            observed_probs[market_name] = probs

    return observed_probs


def get_observed_probs(
    manual_file: Optional[str] = None,
    api_key: Optional[str] = None,
    devig_method: str = "shin",
) -> tuple:
    """
    Main entry point: get observed probabilities.

    Priority:
    1. Try The Odds API if api_key is provided
    2. Merge with manual file if provided (manual overrides API for same market)
    3. Use manual file alone if no API key
    """
    raw_odds = {}
    race_info = {"race": "Unknown", "date": "", "is_sprint": False}

    # Try API first
    if api_key:
        api_odds, api_race_info = fetch_all_f1_odds(api_key)
        if api_odds:
            raw_odds.update(api_odds)
            race_info = api_race_info
            print(f"  API markets: {list(api_odds.keys())}")

    # Load manual file (supplements or overrides API)
    if manual_file and os.path.exists(manual_file):
        print(f"\nLoading manual odds from {manual_file}")
        data = load_manual_odds(manual_file)
        for market, odds in data.get("markets", {}).items():
            if market in raw_odds:
                print(f"  Manual overrides API for: {market}")
            else:
                print(f"  Manual adds: {market}")
            raw_odds[market] = odds

        if race_info["race"] == "Unknown":
            race_info = {
                "race": data.get("race", "Unknown"),
                "date": data.get("date", ""),
                "is_sprint": data.get("is_sprint", False),
            }

    if not raw_odds:
        raise ValueError("No odds data from API or manual file")

    observed_probs = process_odds_to_fair_probs(raw_odds, devig_method)

    print(f"\nFinal markets: {list(observed_probs.keys())}")
    for market, probs in observed_probs.items():
        n = len(probs)
        top = sorted(probs.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{DRIVERS[i]['abbr']}={p:.3f}" for i, p in top)
        print(f"  {market} ({n} drivers): {top_str}")

    return observed_probs, race_info
