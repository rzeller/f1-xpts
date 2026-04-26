"""Tests for odds_fetcher.py — driver resolution and odds processing.

These tests don't hit the live Jolpica API. We build a synthetic 4-driver
roster + name_map and pass it explicitly into the functions under test.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile
import pytest
from odds_fetcher import (
    resolve_driver_index,
    load_manual_odds,
    process_odds_to_fair_probs,
)
from roster import build_name_map


def _stub_roster():
    """A small synthetic roster — just enough to exercise name resolution."""
    return [
        {"name": "George Russell", "given_name": "George", "family_name": "Russell",
         "abbr": "RUS", "team_id": "mercedes", "team_name": "Mercedes", "team_idx": 0,
         "team_color": "#27F4D2"},
        {"name": "Andrea Kimi Antonelli", "given_name": "Andrea Kimi", "family_name": "Antonelli",
         "abbr": "ANT", "team_id": "mercedes", "team_name": "Mercedes", "team_idx": 0,
         "team_color": "#27F4D2"},
        {"name": "Charles Leclerc", "given_name": "Charles", "family_name": "Leclerc",
         "abbr": "LEC", "team_id": "ferrari", "team_name": "Ferrari", "team_idx": 1,
         "team_color": "#E8002D"},
        {"name": "Lewis Hamilton", "given_name": "Lewis", "family_name": "Hamilton",
         "abbr": "HAM", "team_id": "ferrari", "team_name": "Ferrari", "team_idx": 1,
         "team_color": "#E8002D"},
        {"name": "Max Verstappen", "given_name": "Max", "family_name": "Verstappen",
         "abbr": "VER", "team_id": "red_bull", "team_name": "Red Bull", "team_idx": 2,
         "team_color": "#3671C6"},
        {"name": "Nico Hülkenberg", "given_name": "Nico", "family_name": "Hülkenberg",
         "abbr": "HUL", "team_id": "audi", "team_name": "Audi", "team_idx": 3,
         "team_color": "#52E252"},
    ]


@pytest.fixture
def roster():
    return _stub_roster()


@pytest.fixture
def name_map(roster):
    return build_name_map(roster)


class TestResolveDriverIndex:
    def test_full_name(self, name_map):
        assert resolve_driver_index("George Russell", name_map) == 0

    def test_last_name(self, name_map):
        assert resolve_driver_index("Russell", name_map) == 0

    def test_abbreviation(self, name_map):
        assert resolve_driver_index("RUS", name_map) == 0

    def test_case_insensitive(self, name_map):
        assert resolve_driver_index("george russell", name_map) == 0
        assert resolve_driver_index("RUSSELL", name_map) == 0

    def test_all_drivers_resolvable(self, roster, name_map):
        for i, d in enumerate(roster):
            assert resolve_driver_index(d["name"], name_map) == i
            assert resolve_driver_index(d["abbr"], name_map) == i

    def test_hulkenberg_without_umlaut(self, name_map):
        assert resolve_driver_index("Hulkenberg", name_map) is not None

    def test_unknown_returns_none(self, name_map):
        assert resolve_driver_index("Michael Schumacher", name_map) is None

    def test_whitespace_stripped(self, name_map):
        assert resolve_driver_index("  Russell  ", name_map) == 0


class TestLoadManualOdds:
    def test_loads_valid_json(self):
        data = {
            "race": "Test GP",
            "date": "2026-01-01",
            "is_sprint": False,
            "markets": {"win": {"Russell": -150}},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = load_manual_odds(f.name)
        assert result["race"] == "Test GP"
        assert "win" in result["markets"]
        os.unlink(f.name)


class TestProcessOddsToFairProbs:
    @pytest.fixture
    def win_odds(self):
        return {
            "win": {
                "Russell": -150,
                "Antonelli": 200,
                "Leclerc": 600,
                "Hamilton": 800,
            }
        }

    def test_win_market_sums_to_one(self, win_odds, name_map):
        result = process_odds_to_fair_probs(win_odds, name_map)
        probs = result["win"]
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)

    def test_win_market_uses_driver_indices(self, win_odds, name_map):
        result = process_odds_to_fair_probs(win_odds, name_map)
        # Russell is roster index 0
        assert 0 in result["win"]

    def test_win_favorite_has_highest_prob(self, win_odds, name_map):
        result = process_odds_to_fair_probs(win_odds, name_map)
        probs = result["win"]
        assert probs[0] == max(probs.values())

    def test_placement_market(self, name_map):
        raw = {
            "podium": {
                "Russell": -300,
                "Antonelli": -200,
                "Leclerc": 120,
                "Hamilton": 150,
            }
        }
        result = process_odds_to_fair_probs(raw, name_map)
        probs = result["podium"]
        # Podium probs should sum to ~3 (3 podium slots), modulo small overround
        assert sum(probs.values()) == pytest.approx(3.0, abs=0.5)

    def test_dnf_market(self, name_map):
        raw = {
            "dnf": {
                "Russell": 1200,   # low DNF chance
                "Verstappen": 500, # higher DNF chance
            }
        }
        result = process_odds_to_fair_probs(raw, name_map)
        dnf = result["dnf"]
        ver_idx = resolve_driver_index("Verstappen", name_map)
        rus_idx = resolve_driver_index("Russell", name_map)
        assert dnf[ver_idx] > dnf[rus_idx]

    def test_unknown_driver_skipped(self, name_map):
        raw = {"win": {"Russell": -150, "UnknownDriver": 5000}}
        result = process_odds_to_fair_probs(raw, name_map)
        assert 0 in result["win"]
        assert len(result["win"]) == 1

    def test_empty_market_skipped(self, name_map):
        result = process_odds_to_fair_probs({"win": {}}, name_map)
        assert "win" not in result
