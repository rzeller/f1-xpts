"""Tests for config.py — scoring tables, team colors, calendar."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    RACE_POINTS, SPRINT_POINTS, DNF_PENALTY,
    TEAM_COLORS, DEFAULT_TEAM_COLOR,
    SPRINT_WEEKENDS, CALENDAR,
)


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


class TestTeamColors:
    def test_default_color_is_hex(self):
        assert DEFAULT_TEAM_COLOR.startswith("#")
        assert len(DEFAULT_TEAM_COLOR) == 7

    def test_all_team_colors_are_hex(self):
        for team_id, color in TEAM_COLORS.items():
            assert color.startswith("#"), f"{team_id} color {color!r} missing #"
            assert len(color) == 7, f"{team_id} color {color!r} not #RRGGBB"

    def test_team_ids_are_snake_case(self):
        # constructorId convention: lowercase, snake_case, no spaces.
        for team_id in TEAM_COLORS:
            assert team_id == team_id.lower()
            assert " " not in team_id


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
