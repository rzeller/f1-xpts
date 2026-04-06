"""
Build dynamic per-race driver/team configurations for historical backtesting.

The 2019-2024 grids differ from the 2026 grid in config.py. This module builds
a configuration object for each historical race from the odds + results data.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RaceConfig:
    """Dynamic driver/team configuration for a single historical race."""
    drivers: List[dict]          # [{name, abbr, team_idx}, ...]
    n_drivers: int
    n_teams: int
    team_indices: List[int]      # team_idx per driver
    driver_name_map: Dict[str, int]  # lowercase name → driver index
    teams: List[dict]            # [{name}, ...]
    season: int = 0
    round_num: int = 0
    race_name: str = ""


def _normalize_team_name(name: str) -> str:
    """Normalize constructor names across seasons for consistent team matching."""
    name = name.strip()
    # Map historical constructor names to canonical forms
    aliases = {
        "Red Bull Racing": "Red Bull",
        "Red Bull Racing Honda": "Red Bull",
        "Red Bull Racing RBPT": "Red Bull",
        "Red Bull Racing Honda RBPT": "Red Bull",
        "Scuderia AlphaTauri": "AlphaTauri",
        "Scuderia AlphaTauri Honda": "AlphaTauri",
        "AlphaTauri": "AlphaTauri",
        "RB F1 Team": "RB",
        "Visa Cash App RB F1 Team": "RB",
        "Racing Bulls": "RB",
        "Aston Martin": "Aston Martin",
        "Aston Martin Aramco": "Aston Martin",
        "Alpine F1 Team": "Alpine",
        "BWT Alpine F1 Team": "Alpine",
        "Renault": "Alpine",
        "McLaren": "McLaren",
        "McLaren F1 Team": "McLaren",
        "McLaren Mercedes": "McLaren",
        "Ferrari": "Ferrari",
        "Scuderia Ferrari": "Ferrari",
        "Mercedes": "Mercedes",
        "Mercedes-AMG Petronas F1 Team": "Mercedes",
        "Williams": "Williams",
        "Williams Racing": "Williams",
        "Haas F1 Team": "Haas",
        "MoneyGram Haas F1 Team": "Haas",
        "Alfa Romeo": "Alfa Romeo",
        "Alfa Romeo Racing": "Alfa Romeo",
        "Kick Sauber": "Sauber",
        "Sauber": "Sauber",
        "Stake F1 Team Kick Sauber": "Sauber",
        "Racing Point": "Racing Point",
        "Toro Rosso": "Toro Rosso",
    }
    return aliases.get(name, name)


def _make_abbr(driver_code: str, name: str = "") -> str:
    """Make a 3-letter abbreviation from driver code or name."""
    if driver_code and len(driver_code) == 3:
        return driver_code.upper()
    if name:
        parts = name.split()
        return parts[-1][:3].upper()
    return "UNK"


def _build_name_map_for_driver(idx: int, name: str, abbr: str) -> Dict[str, int]:
    """Build all name variants → index mappings for a single driver."""
    entries = {}
    # Full name
    entries[name.lower()] = idx
    # Abbreviation
    entries[abbr.lower()] = idx
    # Last name
    parts = name.split()
    if parts:
        entries[parts[-1].lower()] = idx
    # First name (if > 1 part and first name is unique enough)
    if len(parts) > 1:
        entries[parts[0].lower()] = idx

    # Handle special characters (e.g., Hülkenberg → hulkenberg)
    ascii_name = name.lower()
    for src, dst in [("ü", "u"), ("é", "e"), ("ë", "e"), ("ö", "o"), ("á", "a"), ("ñ", "n")]:
        ascii_name = ascii_name.replace(src, dst)
    if ascii_name != name.lower():
        entries[ascii_name] = idx
        parts_ascii = ascii_name.split()
        if parts_ascii:
            entries[parts_ascii[-1]] = idx

    return entries


# Known driver full names by code (covers 2019-2024 grid)
HISTORICAL_DRIVER_NAMES = {
    "HAM": "Lewis Hamilton",
    "BOT": "Valtteri Bottas",
    "VER": "Max Verstappen",
    "PER": "Sergio Perez",
    "LEC": "Charles Leclerc",
    "SAI": "Carlos Sainz",
    "NOR": "Lando Norris",
    "RIC": "Daniel Ricciardo",
    "PIA": "Oscar Piastri",
    "ALO": "Fernando Alonso",
    "STR": "Lance Stroll",
    "OCO": "Esteban Ocon",
    "GAS": "Pierre Gasly",
    "TSU": "Yuki Tsunoda",
    "VET": "Sebastian Vettel",
    "RAI": "Kimi Räikkönen",
    "RUS": "George Russell",
    "LAT": "Nicholas Latifi",
    "MSC": "Mick Schumacher",
    "MAZ": "Nikita Mazepin",
    "ALB": "Alexander Albon",
    "MAG": "Kevin Magnussen",
    "HUL": "Nico Hülkenberg",
    "ZHO": "Guanyu Zhou",
    "DEV": "Nyck de Vries",
    "LAW": "Liam Lawson",
    "SAR": "Logan Sargeant",
    "BEA": "Oliver Bearman",
    "COL": "Franco Colapinto",
    "DOO": "Jack Doohan",
    "KVY": "Daniil Kvyat",
    "GIO": "Antonio Giovinazzi",
    "AIT": "Jack Aitken",
    "FIT": "Pietro Fittipaldi",
    "GRO": "Romain Grosjean",
}


def build_race_config(
    race_results: dict,
    odds_drivers: Optional[List[str]] = None,
) -> RaceConfig:
    """
    Build a RaceConfig from historical race results and optional odds data.

    Parameters
    ----------
    race_results : dict with keys 'season', 'round', 'name', 'results'
        where results is a list of {driver, team, position, status, dnf}
    odds_drivers : optional list of driver names from odds data (used to
        ensure all odds drivers are included even if they didn't race)

    Returns
    -------
    RaceConfig with dynamically-built driver list, team assignments, name map
    """
    # Build driver list from race results
    seen_drivers = {}  # code → {name, team}
    for res in race_results.get("results", []):
        code = res["driver"]
        team_raw = res["team"]
        team = _normalize_team_name(team_raw)
        full_name = HISTORICAL_DRIVER_NAMES.get(code, code)
        seen_drivers[code] = {"name": full_name, "team": team, "abbr": code}

    # Assign team indices
    teams_seen = []
    for info in seen_drivers.values():
        if info["team"] not in teams_seen:
            teams_seen.append(info["team"])

    team_to_idx = {t: i for i, t in enumerate(teams_seen)}

    # Build driver list in a stable order (by finishing position from results)
    driver_order = []
    for res in race_results.get("results", []):
        code = res["driver"]
        if code not in driver_order:
            driver_order.append(code)

    drivers = []
    driver_name_map = {}
    for idx, code in enumerate(driver_order):
        info = seen_drivers[code]
        team_idx = team_to_idx[info["team"]]
        driver = {
            "name": info["name"],
            "abbr": info["abbr"],
            "team_idx": team_idx,
        }
        drivers.append(driver)
        # Build name map entries
        name_entries = _build_name_map_for_driver(idx, info["name"], info["abbr"])
        driver_name_map.update(name_entries)

    teams = [{"name": t} for t in teams_seen]

    return RaceConfig(
        drivers=drivers,
        n_drivers=len(drivers),
        n_teams=len(teams),
        team_indices=[d["team_idx"] for d in drivers],
        driver_name_map=driver_name_map,
        teams=teams,
        season=race_results.get("season", 0),
        round_num=race_results.get("round", 0),
        race_name=race_results.get("name", ""),
    )


def get_actual_results(race_results: dict, race_config: RaceConfig) -> dict:
    """
    Extract actual race results mapped to driver indices from the race config.

    Returns
    -------
    dict with:
        positions: {driver_idx: finishing_position (1-based)}
        dnfs: {driver_idx: True/False}
        points: {driver_idx: fantasy_points_scored}
    """
    from config import RACE_POINTS, DNF_PENALTY

    positions = {}
    dnfs = {}
    points = {}

    for res in race_results.get("results", []):
        code = res["driver"]
        idx = race_config.driver_name_map.get(code.lower())
        if idx is None:
            continue
        pos = res["position"]
        is_dnf = res["dnf"]

        positions[idx] = pos
        dnfs[idx] = is_dnf

        if is_dnf:
            points[idx] = DNF_PENALTY
        else:
            points[idx] = RACE_POINTS.get(pos, 0)

    return {"positions": positions, "dnfs": dnfs, "points": points}
