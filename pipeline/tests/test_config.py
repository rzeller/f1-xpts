"""Tests for config.py — grid consistency and scoring tables."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    DRIVERS, TEAMS, N_DRIVERS, N_TEAMS,
    RACE_POINTS, SPRINT_POINTS, DNF_PENALTY,
    DRIVER_NAME_MAP, SPRINT_WEEKENDS, CALENDAR,
)


class TestGrid:
    def test_22_drivers(self):
        assert N_DRIVERS == 22

    def test_11_teams(self):
        assert N_TEAMS == 11

    def test_drivers_match_n(self):
        assert len(DRIVERS) == N_DRIVERS

    def test_teams_match_n(self):
        assert len(TEAMS) == N_TEAMS

    def test_each_team_has_two_drivers(self):
        from collections import Counter
        team_counts = Counter(d["team_idx"] for d in DRIVERS)
        for t in range(N_TEAMS):
            assert team_counts[t] == 2, f"Team {TEAMS[t]['name']} has {team_counts[t]} drivers"

    def test_team_indices_valid(self):
        for d in DRIVERS:
            assert 0 <= d["team_idx"] < N_TEAMS

    def test_unique_abbreviations(self):
        abbrs = [d["abbr"] for d in DRIVERS]
        assert len(abbrs) == len(set(abbrs))

    def test_abbreviations_three_chars(self):
        for d in DRIVERS:
            assert len(d["abbr"]) == 3

    def test_all_teams_have_colors(self):
        for t in TEAMS:
            assert t["color"].startswith("#")
            assert len(t["color"]) == 7  # #RRGGBB


class TestScoring:
    def test_race_points_p1(self):
        assert RACE_POINTS[1] == 25

    def test_race_points_p10(self):
        assert RACE_POINTS[10] == 1

    def test_race_points_top10_only(self):
        assert set(RACE_POINTS.keys()) == set(range(1, 11))

    def test_sprint_points_p1(self):
        assert SPRINT_POINTS[1] == 8

    def test_sprint_points_p8(self):
        assert SPRINT_POINTS[8] == 1

    def test_sprint_points_top8_only(self):
        assert set(SPRINT_POINTS.keys()) == set(range(1, 9))

    def test_race_points_decreasing(self):
        for i in range(1, 10):
            assert RACE_POINTS[i] > RACE_POINTS[i + 1]

    def test_sprint_points_decreasing(self):
        for i in range(1, 8):
            assert SPRINT_POINTS[i] > SPRINT_POINTS[i + 1]

    def test_dnf_penalty_negative(self):
        assert DNF_PENALTY < 0


class TestDriverNameMap:
    def test_full_name_matches(self):
        assert DRIVER_NAME_MAP["george russell"] == 0

    def test_last_name_matches(self):
        assert DRIVER_NAME_MAP["russell"] == 0

    def test_abbreviation_matches(self):
        assert DRIVER_NAME_MAP["rus"] == 0

    def test_hulkenberg_alias(self):
        assert DRIVER_NAME_MAP["hulkenberg"] == DRIVER_NAME_MAP["hülkenberg"]

    def test_all_drivers_have_entries(self):
        for i, d in enumerate(DRIVERS):
            assert d["name"].lower() in DRIVER_NAME_MAP
            assert d["abbr"].lower() in DRIVER_NAME_MAP


class TestCalendar:
    def test_22_races(self):
        assert len(CALENDAR) == 22

    def test_rounds_sequential(self):
        rounds = [r["round"] for r in CALENDAR]
        assert rounds == list(range(1, 23))

    def test_sprint_weekends_in_calendar(self):
        slugs = {r["slug"] for r in CALENDAR}
        for sw in SPRINT_WEEKENDS:
            assert sw in slugs, f"Sprint weekend {sw} not in calendar"
