"""
F1 Expected Points Pipeline — Configuration
2026 Season
"""

# Points scoring for family league
RACE_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
DNF_PENALTY = -20

# 2026 Grid
TEAMS = [
    {"name": "Mercedes", "color": "#27F4D2"},
    {"name": "Ferrari", "color": "#E8002D"},
    {"name": "Red Bull", "color": "#3671C6"},
    {"name": "McLaren", "color": "#FF8000"},
    {"name": "Aston Martin", "color": "#229971"},
    {"name": "Alpine", "color": "#FF87BC"},
    {"name": "Williams", "color": "#64C4FF"},
    {"name": "RB", "color": "#6692FF"},
    {"name": "Haas", "color": "#B6BABD"},
    {"name": "Kick Sauber", "color": "#52E252"},
    {"name": "Cadillac", "color": "#C0C0C0"},
]

DRIVERS = [
    {"name": "George Russell", "team_idx": 0, "abbr": "RUS"},
    {"name": "Kimi Antonelli", "team_idx": 0, "abbr": "ANT"},
    {"name": "Charles Leclerc", "team_idx": 1, "abbr": "LEC"},
    {"name": "Lewis Hamilton", "team_idx": 1, "abbr": "HAM"},
    {"name": "Max Verstappen", "team_idx": 2, "abbr": "VER"},
    {"name": "Liam Lawson", "team_idx": 2, "abbr": "LAW"},
    {"name": "Lando Norris", "team_idx": 3, "abbr": "NOR"},
    {"name": "Oscar Piastri", "team_idx": 3, "abbr": "PIA"},
    {"name": "Fernando Alonso", "team_idx": 4, "abbr": "ALO"},
    {"name": "Lance Stroll", "team_idx": 4, "abbr": "STR"},
    {"name": "Pierre Gasly", "team_idx": 5, "abbr": "GAS"},
    {"name": "Franco Colapinto", "team_idx": 5, "abbr": "COL"},
    {"name": "Alexander Albon", "team_idx": 6, "abbr": "ALB"},
    {"name": "Carlos Sainz", "team_idx": 6, "abbr": "SAI"},
    {"name": "Yuki Tsunoda", "team_idx": 7, "abbr": "TSU"},
    {"name": "Isack Hadjar", "team_idx": 7, "abbr": "HAD"},
    {"name": "Oliver Bearman", "team_idx": 8, "abbr": "BEA"},
    {"name": "Esteban Ocon", "team_idx": 8, "abbr": "OCO"},
    {"name": "Nico Hülkenberg", "team_idx": 9, "abbr": "HUL"},
    {"name": "Gabriel Bortoleto", "team_idx": 9, "abbr": "BOR"},
    {"name": "Sergio Perez", "team_idx": 10, "abbr": "PER"},
    {"name": "Valtteri Bottas", "team_idx": 10, "abbr": "BOT"},
]

N_DRIVERS = len(DRIVERS)
N_TEAMS = len(TEAMS)

# Name matching: Oddschecker/The Odds API may use different name formats
# Map from various spellings to our canonical driver index
DRIVER_NAME_MAP = {}
for i, d in enumerate(DRIVERS):
    # Add full name, last name, and abbreviation
    DRIVER_NAME_MAP[d["name"].lower()] = i
    DRIVER_NAME_MAP[d["name"].split()[-1].lower()] = i
    DRIVER_NAME_MAP[d["abbr"].lower()] = i

# Explicit aliases for names with special characters or common alternate spellings
_ALIASES = {
    "hulkenberg": "hülkenberg",
    "hulk": "hülkenberg",
}
for alias, canonical in _ALIASES.items():
    if canonical in DRIVER_NAME_MAP:
        DRIVER_NAME_MAP[alias] = DRIVER_NAME_MAP[canonical]

# Sprint weekends in 2026
SPRINT_WEEKENDS = [
    "chinese-gp",
    "miami-gp",
    "canadian-gp",
    "british-gp",
    "dutch-gp",
    "singapore-gp",
]

# 2026 Calendar (for scheduling)
CALENDAR = [
    {"round": 1, "name": "Australian GP", "slug": "australian-gp", "date": "2026-03-08"},
    {"round": 2, "name": "Chinese GP", "slug": "chinese-gp", "date": "2026-03-22"},
    {"round": 3, "name": "Japanese GP", "slug": "japanese-gp", "date": "2026-03-29"},
    # ... add remaining races
]
