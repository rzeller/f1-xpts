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
    """Fetch all race results for a season, paginating as needed."""
    PAGE_SIZE = 100  # API caps at 100
    offset = 0
    races_by_round = {}

    while True:
        url = f"{API_BASE}/{season}/results.json?limit={PAGE_SIZE}&offset={offset}"
        for attempt in range(retries):
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                data = r.json()
                mr = data["MRData"]
                total = int(mr["total"])
                races_raw = mr["RaceTable"]["Races"]
                break
            except (requests.RequestException, KeyError) as e:
                if attempt < retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  Retry {attempt + 1}/{retries} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    print(f"  ERROR: Failed to fetch {season} (offset={offset}): {e}")
                    return list(races_by_round.values())

        for race in races_raw:
            rnd = int(race["round"])
            if rnd not in races_by_round:
                races_by_round[rnd] = {"season": season, "round": rnd,
                                       "name": race["raceName"], "results": []}
            for res in race.get("Results", []):
                driver_code = res["Driver"].get("code", res["Driver"]["familyName"][:3].upper())
                constructor = res["Constructor"]["name"]
                position = int(res["position"])
                status = res["status"]
                is_dnf = status != "Finished" and "Lap" not in status
                races_by_round[rnd]["results"].append({
                    "driver": driver_code,
                    "team": constructor,
                    "position": position,
                    "status": status,
                    "dnf": is_dnf,
                })

        offset += PAGE_SIZE
        time.sleep(0.25)  # be polite
        if offset >= total:
            break

    return [races_by_round[rnd] for rnd in sorted(races_by_round)]


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
