"""
Driver + team roster fetcher.

Source of truth: Jolpica (Ergast-compatible) F1 API. We pull the most-recent
race results, since the latest race results reflect the actual driver →
constructor mapping after any mid-season swap (a reserve replacing a regular,
a contract change, etc.).

A separate API call gives us the canonical roster for the *next* race we'd
otherwise have to scrape from Oddschecker. Oddschecker shows just driver
names with no team info, so we pair its odds with this roster's team mapping
on the Python side.

Team colors live in config.TEAM_COLORS, keyed by Jolpica's stable
`constructorId` (e.g. "red_bull", "mclaren"). Constructor *display names*
come from the API too (and may include "F1 Team" suffixes — we trim those
for cleaner UI).
"""

from typing import Dict, List, Optional

import requests

from config import TEAM_COLORS, DEFAULT_TEAM_COLOR


JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
DEFAULT_TIMEOUT = 15


def _strip_team_suffix(name: str) -> str:
    """`Alpine F1 Team` → `Alpine`, `RB F1 Team` → `RB`. Cosmetic only."""
    suffix = " F1 Team"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def fetch_current_roster(
    season: str = "current",
    timeout: int = DEFAULT_TIMEOUT,
) -> List[Dict]:
    """
    Fetch the active driver + team roster from Jolpica's most recent race results.

    Why "last results" rather than `/drivers/`: the `/drivers/` endpoint is a
    season-long superset (includes anyone who's raced this season — both the
    driver who was dropped and their replacement). The latest race's results
    is the cleanest snapshot of who's currently driving for whom.

    Returns a list of dicts in finishing order from the most recent race:
      - name:        full canonical name ("George Russell")
      - given_name:  first name(s)
      - family_name: surname
      - abbr:        FIA 3-letter code
      - team_id:     constructor id ("mercedes")
      - team_name:   display name ("Mercedes" — F1-Team suffix stripped)
      - team_idx:    index into the deduplicated team list (assigned here)
      - team_color:  hex color from TEAM_COLORS, or DEFAULT_TEAM_COLOR
    """
    url = f"{JOLPICA_BASE}/{season}/last/results/"
    resp = requests.get(url, headers={"Accept": "application/json"}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        raise RuntimeError(f"No race results returned from {url}")

    results = races[0].get("Results", [])
    if not results:
        raise RuntimeError(f"Last race ({races[0].get('raceName')}) has no results yet")

    # Build the team_idx lookup as we walk results, in first-encountered order.
    team_idx_by_id: Dict[str, int] = {}
    roster: List[Dict] = []

    for r in results:
        drv = r.get("Driver", {})
        con = r.get("Constructor", {})
        team_id = con.get("constructorId", "")
        team_display = _strip_team_suffix(con.get("name", ""))

        if team_id and team_id not in team_idx_by_id:
            team_idx_by_id[team_id] = len(team_idx_by_id)

        roster.append({
            "name": f"{drv.get('givenName', '').strip()} {drv.get('familyName', '').strip()}".strip(),
            "given_name": drv.get("givenName", ""),
            "family_name": drv.get("familyName", ""),
            "abbr": drv.get("code") or drv.get("familyName", "")[:3].upper(),
            "team_id": team_id,
            "team_name": team_display,
            "team_idx": team_idx_by_id.get(team_id, -1),
            "team_color": TEAM_COLORS.get(team_id, DEFAULT_TEAM_COLOR),
        })

    return roster


def teams_from_roster(roster: List[Dict]) -> List[Dict]:
    """
    Collapse a roster down to a deduplicated team list, indexed by team_idx.

    Used to populate the `teams` array in the output JSON.
    """
    teams: Dict[int, Dict] = {}
    for d in roster:
        idx = d["team_idx"]
        if idx < 0 or idx in teams:
            continue
        teams[idx] = {
            "id": d["team_id"],
            "name": d["team_name"],
            "color": d["team_color"],
        }
    return [teams[i] for i in sorted(teams)]


def build_name_map(roster: List[Dict]) -> Dict[str, int]:
    """
    Build a case-insensitive lookup from name strings to roster index.

    Includes: full name, family name, abbr, and a handful of common variants
    (last-name-only with diacritics stripped).
    """
    name_map: Dict[str, int] = {}

    def add(key: str, idx: int) -> None:
        if not key:
            return
        k = key.lower().strip()
        if k and k not in name_map:
            name_map[k] = idx

    for i, d in enumerate(roster):
        add(d["name"], i)
        add(d["family_name"], i)
        add(d["abbr"], i)
        # Strip diacritics for resilience (Hülkenberg → hulkenberg).
        # No external deps — handle the common F1-grid characters explicitly.
        stripped = (
            d["family_name"]
            .lower()
            .replace("ü", "u")
            .replace("ö", "o")
            .replace("é", "e")
            .replace("ñ", "n")
            .replace("á", "a")
            .replace("í", "i")
        )
        add(stripped, i)

    return name_map


def resolve_driver_index(name: str, name_map: Dict[str, int]) -> Optional[int]:
    """Match an arbitrary odds-source name string to a roster index, or None."""
    key = name.lower().strip()
    if key in name_map:
        return name_map[key]
    # Fall back to per-word lookup (handles "George Russell" → "russell").
    for word in key.split():
        if word in name_map:
            return name_map[word]
    return None
