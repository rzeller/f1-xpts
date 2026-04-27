"""
F1 Expected Points Pipeline — Configuration

The driver roster + team mapping is no longer hard-coded here. It's fetched
each pipeline run from the Jolpica F1 API (see roster.py) so mid-season swaps
(e.g. a reserve replacing a regular, a contract change) are reflected
automatically. Only team *colors* are configured locally, since neither the
F1 API nor Oddschecker exposes a color palette.
"""

# Points scoring for family league
RACE_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
DNF_PENALTY = -20

# Team colors keyed by Jolpica's stable `constructorId` (lowercase, snake_case).
# When a new team appears (or a team's id changes), it falls back to
# DEFAULT_TEAM_COLOR. To re-key a team, look up its constructorId in the API
# response (e.g. https://api.jolpi.ca/ergast/f1/current/last/results/).
TEAM_COLORS = {
    "mercedes":     "#27F4D2",
    "ferrari":      "#E8002D",
    "red_bull":     "#3671C6",
    "mclaren":      "#FF8000",
    "aston_martin": "#229971",
    "alpine":       "#FF87BC",
    "williams":     "#64C4FF",
    "rb":           "#6692FF",
    "haas":         "#B6BABD",
    # Audi acquired the Sauber team for 2026 — Jolpica's constructorId is
    # "audi" but the previous "Kick Sauber" green still works as the brand.
    "audi":         "#52E252",
    "cadillac":     "#C0C0C0",
}
DEFAULT_TEAM_COLOR = "#888888"

# Race-day correlation parameters (hierarchical noise model).
# sigma_team: Team race-day volatility (shared by teammates in each sim)
# sigma_global: Chaos magnitude (interpretation depends on chaos_model)
# sigma_dnf: DNF correlation (log-normal multiplier on DNF probabilities per sim)
# chaos_model: "one_sided" (default) — drivers can fall back due to incidents
#   but can't gain pace they don't have; mathematically `utility -=
#   Exp(σ_global)` per driver per race. Decouples backmarker performance from
#   favorites (Bottas's draw doesn't affect Russell winning), and matches
#   F1 reality where chaos = mistakes/mechanical/weather, not surprise speed.
#   Tightens win residuals to ~1pp per top driver and eliminates backmarker
#   leakage (issue #36 follow-up).
# chaos_model: "symmetric" — legacy log-normal multiplier on Gumbel noise.
#   Calibrated against historical race-variance via
#   pipeline/calibrate_correlation.py (sigma_global=1.1715). Available for
#   backward compatibility / comparison runs.
CORRELATION_DEFAULTS = {
    "sigma_team": 0.6634,
    "sigma_global": 0.5,        # tuned for one_sided chaos; was 1.1715 for symmetric
    "sigma_dnf": 0.3285,
    "chaos_model": "one_sided",
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
