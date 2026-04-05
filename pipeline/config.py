"""
F1 Expected Points Pipeline — Configuration
2026 Season
"""

# Points scoring for family league
RACE_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
DNF_PENALTY = -10

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
# Race-day correlation parameters (hierarchical noise model)
# Calibrated via pipeline/calibrate_correlation.py against F1 summary statistics.
# Re-run with --data flag on real historical results for precise values.
# sigma_team: Team race-day volatility (shared by teammates in each sim)
# sigma_global: Field-wide chaos scaling (log-normal multiplier on Gumbel noise)
# sigma_dnf: DNF correlation (log-normal multiplier on DNF probabilities per sim)
CORRELATION_DEFAULTS = {
    "sigma_team": 0.0188,
    "sigma_global": 1.5332,
    "sigma_dnf": 0.4823,
}

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
    {"round": 1,  "name": "Australian GP",           "slug": "australian-gp",  "date": "2026-03-08"},
    {"round": 2,  "name": "Chinese GP",              "slug": "chinese-gp",     "date": "2026-03-15"},
    {"round": 3,  "name": "Japanese GP",             "slug": "japanese-gp",    "date": "2026-03-29"},
    {"round": 4,  "name": "Miami GP",                "slug": "miami-gp",       "date": "2026-05-03"},
    {"round": 5,  "name": "Canadian GP",             "slug": "canadian-gp",    "date": "2026-05-24"},
    {"round": 6,  "name": "Monaco GP",               "slug": "monaco-gp",      "date": "2026-06-07"},
    {"round": 7,  "name": "Barcelona-Catalunya GP",  "slug": "barcelona-gp",   "date": "2026-06-14"},
    {"round": 8,  "name": "Austrian GP",             "slug": "austrian-gp",    "date": "2026-06-28"},
    {"round": 9,  "name": "British GP",              "slug": "british-gp",     "date": "2026-07-05"},
    {"round": 10, "name": "Belgian GP",              "slug": "belgian-gp",     "date": "2026-07-19"},
    {"round": 11, "name": "Hungarian GP",            "slug": "hungarian-gp",   "date": "2026-07-26"},
    {"round": 12, "name": "Dutch GP",                "slug": "dutch-gp",       "date": "2026-08-23"},
    {"round": 13, "name": "Italian GP",              "slug": "italian-gp",     "date": "2026-09-06"},
    {"round": 14, "name": "Spanish GP",              "slug": "spanish-gp",     "date": "2026-09-13"},
    {"round": 15, "name": "Azerbaijan GP",           "slug": "azerbaijan-gp",  "date": "2026-09-26"},
    {"round": 16, "name": "Singapore GP",            "slug": "singapore-gp",   "date": "2026-10-11"},
    {"round": 17, "name": "United States GP",        "slug": "us-gp",          "date": "2026-10-25"},
    {"round": 18, "name": "Mexico City GP",          "slug": "mexico-gp",      "date": "2026-11-01"},
    {"round": 19, "name": "São Paulo GP",            "slug": "sao-paulo-gp",   "date": "2026-11-08"},
    {"round": 20, "name": "Las Vegas GP",            "slug": "las-vegas-gp",   "date": "2026-11-22"},
    {"round": 21, "name": "Qatar GP",                "slug": "qatar-gp",       "date": "2026-11-29"},
    {"round": 22, "name": "Abu Dhabi GP",            "slug": "abu-dhabi-gp",   "date": "2026-12-06"},
]
