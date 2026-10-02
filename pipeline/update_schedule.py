"""
Refresh public/data/schedule.json from the Jolpica F1 API (the Ergast successor).

The calendar moves mid-season (cancellations, relocations — e.g. the 2026
Bahrain GP held at Sepang), and every downstream consumer keys off
schedule.json: the update.yml run-window gate, the scraper's next-race
discovery, and the site's Schedule page. A stale file means the pipeline
projects the wrong race. This script is run daily by
.github/workflows/update-schedule.yml.

Merge policy:
  - Races that have already finished are kept exactly as they are in the
    existing file. History doesn't change, and the API may still list
    cancelled/relocated rounds at their original dates.
  - Upcoming races come from the API. Fields the API doesn't provide
    (slug, saturday_race, ...) are carried over from the matching existing
    entry; new races get them derived.
  - Rounds are renumbered by race date.
  - Sanity guard: if the API returns far fewer upcoming races than we already
    have, nothing is written (protects against a partial/broken response).

Usage:
    python pipeline/update_schedule.py [--season 2026] [--schedule PATH] [--dry-run]
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

API_URL = "https://api.jolpi.ca/ergast/f1/{season}.json?limit=100"

SCHEDULE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "data", "schedule.json")
)

# A race counts as finished this long after its start (matches update.yml's
# pre-filter).
FINISHED_AFTER = timedelta(hours=6)

# API session key -> our session key.
SESSION_KEYS = [
    ("FirstPractice", "fp1"),
    ("SecondPractice", "fp2"),
    ("ThirdPractice", "fp3"),
    ("SprintQualifying", "sprint_qualifying"),
    ("SprintShootout", "sprint_qualifying"),
    ("Sprint", "sprint"),
    ("Qualifying", "qualifying"),
]

# IANA time zone per circuit locality, falling back to country. Only used to
# fill timezone/utc_offset/saturday_race for races not already in the file.
LOCALITY_TZ = {
    "Miami": "America/New_York",
    "Austin": "America/Chicago",
    "Las Vegas": "America/Los_Angeles",
    "Montreal": "America/Toronto",
    "Montréal": "America/Toronto",
}
COUNTRY_TZ = {
    "Australia": "Australia/Melbourne",
    "China": "Asia/Shanghai",
    "Japan": "Asia/Tokyo",
    "Bahrain": "Asia/Bahrain",
    "Saudi Arabia": "Asia/Riyadh",
    "Canada": "America/Toronto",
    "Monaco": "Europe/Monaco",
    "Spain": "Europe/Madrid",
    "Austria": "Europe/Vienna",
    "UK": "Europe/London",
    "United Kingdom": "Europe/London",
    "Belgium": "Europe/Brussels",
    "Hungary": "Europe/Budapest",
    "Netherlands": "Europe/Amsterdam",
    "Italy": "Europe/Rome",
    "Azerbaijan": "Asia/Baku",
    "Singapore": "Asia/Singapore",
    "Mexico": "America/Mexico_City",
    "Brazil": "America/Sao_Paulo",
    "Qatar": "Asia/Qatar",
    "UAE": "Asia/Dubai",
    "United Arab Emirates": "Asia/Dubai",
    "Malaysia": "Asia/Kuala_Lumpur",
    "Portugal": "Europe/Lisbon",
    "Turkey": "Europe/Istanbul",
    "Germany": "Europe/Berlin",
    "France": "Europe/Paris",
}
# API country names -> the spelling the existing file uses.
COUNTRY_DISPLAY = {"UK": "United Kingdom", "USA": "United States", "UAE": "United Arab Emirates"}


def _parse(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _iso(date: str, time: str) -> str:
    """'2026-10-04' + '07:00:00Z' -> '2026-10-04T07:00:00Z'."""
    t = time if time.endswith("Z") else time + "Z"
    return f"{date}T{t}"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def slug_for(name: str) -> str:
    """Same transform update.py applies to race names ('Bahrain Grand Prix' -> 'bahrain-gp')."""
    return _norm(name).replace(" ", "-").replace("grand-prix", "gp")


def fetch_api_races(season: int) -> List[dict]:
    resp = requests.get(API_URL.format(season=season), timeout=30)
    resp.raise_for_status()
    return resp.json()["MRData"]["RaceTable"]["Races"]


def _find_existing(api_race: dict, existing: List[dict]) -> Optional[dict]:
    name = _norm(api_race["raceName"])
    slug = slug_for(api_race["raceName"])
    locality = _norm(api_race["Circuit"]["Location"].get("locality", ""))
    for key in (
        lambda r: _norm(r["name"]) == name,
        lambda r: r.get("slug") == slug,
        lambda r: locality and _norm(r.get("location", "")) == locality,
    ):
        hits = [r for r in existing if key(r)]
        if len(hits) == 1:
            return hits[0]
    return None


def convert(api_race: dict, prev: Optional[dict]) -> Optional[dict]:
    """Convert one API race to our schema, carrying over fields from prev."""
    prev = prev or {}
    prev_sessions = prev.get("sessions", {})
    loc = api_race["Circuit"]["Location"]
    country = COUNTRY_DISPLAY.get(loc.get("country", ""), loc.get("country", ""))

    sessions: Dict[str, str] = {}
    for api_key, key in SESSION_KEYS:
        s = api_race.get(api_key)
        if not s or key in sessions:
            continue
        if s.get("time"):
            sessions[key] = _iso(s["date"], s["time"])
        elif key in prev_sessions:
            sessions[key] = prev_sessions[key]
    if api_race.get("time"):
        sessions["race"] = _iso(api_race["date"], api_race["time"])
    elif "race" in prev_sessions and prev_sessions["race"][:10] == api_race["date"]:
        sessions["race"] = prev_sessions["race"]
    else:
        print(f"  WARNING: {api_race['raceName']} has no race time yet — using 12:00Z placeholder")
        sessions["race"] = _iso(api_race["date"], "12:00:00Z")

    is_sprint = "sprint" in sessions
    if is_sprint and "sprint_qualifying" not in sessions:
        print(f"  WARNING: {api_race['raceName']} is a sprint weekend without a sprint qualifying time")

    race = dict(prev)  # keep any extra fields (e.g. saturday_race)
    race.update({
        "name": api_race["raceName"],
        "slug": prev.get("slug") or slug_for(api_race["raceName"]),
        "location": prev.get("location") or loc.get("locality", ""),
        "circuit": api_race["Circuit"].get("circuitName", prev.get("circuit", "")),
        "country": country or prev.get("country", ""),
        "is_sprint": is_sprint,
        "sessions": dict(sorted(sessions.items(), key=lambda kv: kv[1])),
    })

    # Time zone: recompute when we know the IANA zone; keep prev otherwise.
    tz_name = LOCALITY_TZ.get(loc.get("locality", "")) or COUNTRY_TZ.get(loc.get("country", ""))
    if tz_name:
        tz = ZoneInfo(tz_name)
        local = _parse(sessions["race"]).astimezone(tz)
        offset = local.utcoffset().total_seconds() / 3600
        race["utc_offset"] = int(offset) if offset == int(offset) else offset
        abbr = local.tzname()
        # zoneinfo yields '+08' style names for some zones; keep a nicer prev one.
        race["timezone"] = prev.get("timezone") if (abbr[0] in "+-" and prev.get("timezone")) else abbr
        if local.weekday() == 5:
            race["saturday_race"] = True
        else:
            race.pop("saturday_race", None)
    elif "timezone" not in race:
        print(f"  WARNING: no time zone known for {api_race['raceName']} ({loc}) — defaulting to UTC")
        race["timezone"], race["utc_offset"] = "UTC", 0
    return race


def merge(existing: dict, api_races: List[dict], now: datetime) -> dict:
    old_races = existing.get("races", [])
    finished = [r for r in old_races if _parse(r["sessions"]["race"]) + FINISHED_AFTER < now]
    old_upcoming = [r for r in old_races if r not in finished]

    upcoming = []
    for api_race in api_races:
        race = convert(api_race, _find_existing(api_race, old_races))
        if race and _parse(race["sessions"]["race"]) + FINISHED_AFTER >= now:
            upcoming.append(race)

    if len(upcoming) < len(old_upcoming) / 2:
        raise SystemExit(
            f"Refusing to update: API has {len(upcoming)} upcoming races vs "
            f"{len(old_upcoming)} in the current schedule"
        )

    races = sorted(finished + upcoming, key=lambda r: r["sessions"]["race"])
    out = []
    for i, r in enumerate(races, 1):
        r = dict(r)
        r["round"] = i
        # Stable key order matching the hand-written file.
        order = ["round", "name", "slug", "location", "circuit", "country",
                 "timezone", "utc_offset", "is_sprint"]
        ordered = {k: r[k] for k in order if k in r}
        ordered.update({k: v for k, v in r.items() if k not in order and k != "sessions"})
        ordered["sessions"] = r["sessions"]
        out.append(ordered)

    result = dict(existing)
    result["races"] = out
    return result


def summarize(old: dict, new: dict) -> List[str]:
    old_by = {r["slug"]: r for r in old.get("races", [])}
    new_by = {r["slug"]: r for r in new["races"]}
    lines = []
    for slug in new_by.keys() - old_by.keys():
        lines.append(f"+ added {new_by[slug]['name']} ({new_by[slug]['sessions']['race']})")
    for slug in old_by.keys() - new_by.keys():
        lines.append(f"- removed {old_by[slug]['name']} ({old_by[slug]['sessions']['race']})")
    for slug in new_by.keys() & old_by.keys():
        if new_by[slug] != old_by[slug]:
            changed = [k for k in set(new_by[slug]) | set(old_by[slug])
                       if new_by[slug].get(k) != old_by[slug].get(k)]
            lines.append(f"~ changed {new_by[slug]['name']}: {', '.join(sorted(changed))}")
    return sorted(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=None, help="defaults to the file's season")
    ap.add_argument("--schedule", default=SCHEDULE_PATH)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.schedule) as f:
        existing = json.load(f)
    season = args.season or existing.get("season") or datetime.now(timezone.utc).year

    print(f"Fetching {season} calendar from Jolpica...")
    api_races = fetch_api_races(season)
    print(f"  API returned {len(api_races)} races")

    new = merge(existing, api_races, datetime.now(timezone.utc))
    changes = summarize(existing, new)
    if not changes:
        print("Schedule is up to date.")
        return
    print("Schedule changes:")
    for line in changes:
        print("  " + line)
    if args.dry_run:
        return
    with open(args.schedule, "w") as f:
        json.dump(new, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {args.schedule}")


if __name__ == "__main__":
    sys.exit(main())
