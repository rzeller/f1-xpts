#!/usr/bin/env python3
"""
Download historical F1 race results from the Jolpica (Ergast successor) API.

Usage:
    python download_historical.py --seasons 2022 2023 2024 --output historical_results.json

Run this on a machine with internet access. The calibration script reads
the output file.
"""

import argparse
import json
import time
import sys

import requests


API_BASE = "https://api.jolpi.ca/ergast/f1"


def fetch_season_results(season: int, retries: int = 3) -> list:
    """Fetch all race results for a season."""
    url = f"{API_BASE}/{season}/results.json?limit=600"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            races_raw = data["MRData"]["RaceTable"]["Races"]
            break
        except (requests.RequestException, KeyError) as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"  ERROR: Failed to fetch {season}: {e}")
                return []

    races = []
    for race in races_raw:
        results = []
        for r in race.get("Results", []):
            driver_code = r["Driver"].get("code", r["Driver"]["familyName"][:3].upper())
            constructor = r["Constructor"]["name"]
            position = int(r["position"])
            status = r["status"]
            # Classify DNF: anything not "Finished" or "+N Lap(s)" is a DNF
            is_dnf = status != "Finished" and "Lap" not in status
            results.append({
                "driver": driver_code,
                "team": constructor,
                "position": position,
                "status": status,
                "dnf": is_dnf,
            })
        races.append({
            "season": season,
            "round": int(race["round"]),
            "name": race["raceName"],
            "results": results,
        })
    return races


def main():
    parser = argparse.ArgumentParser(description="Download F1 historical results")
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=[2022, 2023, 2024],
        help="Seasons to download (default: 2022 2023 2024)",
    )
    parser.add_argument(
        "--output", "-o", default="pipeline/historical_results.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    all_races = []
    for season in args.seasons:
        print(f"Fetching {season} season...")
        races = fetch_season_results(season)
        print(f"  Got {len(races)} races")
        all_races.extend(races)
        time.sleep(1)  # be polite to the API

    print(f"\nTotal: {len(all_races)} races across {len(args.seasons)} seasons")

    with open(args.output, "w") as f:
        json.dump({"races": all_races}, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
