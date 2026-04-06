"""Tests for historical config builder."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from backtest.config_historical import (
    build_race_config,
    get_actual_results,
    _normalize_team_name,
    _build_name_map_for_driver,
    RaceConfig,
)


SAMPLE_RACE = {
    "season": 2024,
    "round": 1,
    "name": "Australian Grand Prix",
    "results": [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1, "status": "Finished", "dnf": False},
        {"driver": "LEC", "team": "Ferrari", "position": 2, "status": "Finished", "dnf": False},
        {"driver": "NOR", "team": "McLaren", "position": 3, "status": "Finished", "dnf": False},
        {"driver": "HAM", "team": "Mercedes", "position": 4, "status": "Finished", "dnf": False},
        {"driver": "PER", "team": "Red Bull Racing", "position": 5, "status": "Finished", "dnf": False},
        {"driver": "SAI", "team": "Ferrari", "position": 6, "status": "+1 Lap", "dnf": False},
        {"driver": "RIC", "team": "RB F1 Team", "position": 7, "status": "+1 Lap", "dnf": False},
        {"driver": "STR", "team": "Aston Martin", "position": 8, "status": "+1 Lap", "dnf": False},
        {"driver": "ALO", "team": "Aston Martin", "position": 9, "status": "Retired", "dnf": True},
        {"driver": "HUL", "team": "Haas F1 Team", "position": 10, "status": "Retired", "dnf": True},
    ],
}


class TestNormalizeTeamName:
    def test_red_bull_variants(self):
        assert _normalize_team_name("Red Bull Racing") == "Red Bull"
        assert _normalize_team_name("Red Bull Racing Honda") == "Red Bull"
        assert _normalize_team_name("Red Bull Racing RBPT") == "Red Bull"

    def test_alphatauri_to_rb(self):
        assert _normalize_team_name("RB F1 Team") == "RB"
        assert _normalize_team_name("Scuderia AlphaTauri") == "AlphaTauri"

    def test_unknown_team_passthrough(self):
        assert _normalize_team_name("Unknown Team") == "Unknown Team"


class TestBuildNameMap:
    def test_includes_full_name(self):
        entries = _build_name_map_for_driver(0, "Max Verstappen", "VER")
        assert entries["max verstappen"] == 0

    def test_includes_last_name(self):
        entries = _build_name_map_for_driver(0, "Max Verstappen", "VER")
        assert entries["verstappen"] == 0

    def test_includes_abbreviation(self):
        entries = _build_name_map_for_driver(0, "Max Verstappen", "VER")
        assert entries["ver"] == 0

    def test_special_characters(self):
        entries = _build_name_map_for_driver(0, "Nico Hülkenberg", "HUL")
        assert entries["nico hülkenberg"] == 0
        assert entries["nico hulkenberg"] == 0  # ASCII fallback
        assert entries["hulkenberg"] == 0


class TestBuildRaceConfig:
    def test_returns_race_config(self):
        config = build_race_config(SAMPLE_RACE)
        assert isinstance(config, RaceConfig)

    def test_correct_driver_count(self):
        config = build_race_config(SAMPLE_RACE)
        assert config.n_drivers == 10

    def test_driver_order_matches_results(self):
        config = build_race_config(SAMPLE_RACE)
        assert config.drivers[0]["abbr"] == "VER"
        assert config.drivers[1]["abbr"] == "LEC"
        assert config.drivers[2]["abbr"] == "NOR"

    def test_teammates_share_team_idx(self):
        config = build_race_config(SAMPLE_RACE)
        # VER and PER are both Red Bull
        ver_idx = next(i for i, d in enumerate(config.drivers) if d["abbr"] == "VER")
        per_idx = next(i for i, d in enumerate(config.drivers) if d["abbr"] == "PER")
        assert config.drivers[ver_idx]["team_idx"] == config.drivers[per_idx]["team_idx"]

    def test_name_map_resolves_drivers(self):
        config = build_race_config(SAMPLE_RACE)
        assert config.driver_name_map["verstappen"] is not None
        assert config.driver_name_map["ver"] is not None

    def test_season_and_round(self):
        config = build_race_config(SAMPLE_RACE)
        assert config.season == 2024
        assert config.round_num == 1
        assert config.race_name == "Australian Grand Prix"


class TestGetActualResults:
    def test_positions(self):
        config = build_race_config(SAMPLE_RACE)
        actuals = get_actual_results(SAMPLE_RACE, config)
        # VER (index 0) finished P1
        assert actuals["positions"][0] == 1
        # LEC (index 1) finished P2
        assert actuals["positions"][1] == 2

    def test_dnfs(self):
        config = build_race_config(SAMPLE_RACE)
        actuals = get_actual_results(SAMPLE_RACE, config)
        # ALO (index 8) DNF'd
        alo_idx = next(i for i, d in enumerate(config.drivers) if d["abbr"] == "ALO")
        assert actuals["dnfs"][alo_idx] is True
        # VER did not DNF
        assert actuals["dnfs"][0] is False

    def test_points(self):
        config = build_race_config(SAMPLE_RACE)
        actuals = get_actual_results(SAMPLE_RACE, config)
        # VER P1 → 25 points
        assert actuals["points"][0] == 25
        # LEC P2 → 18 points
        assert actuals["points"][1] == 18
        # DNF → -10 points
        alo_idx = next(i for i, d in enumerate(config.drivers) if d["abbr"] == "ALO")
        assert actuals["points"][alo_idx] == -10
