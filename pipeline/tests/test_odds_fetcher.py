"""Tests for odds_fetcher.py — driver resolution and odds processing."""

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
from config import DRIVERS, N_DRIVERS


class TestResolveDriverIndex:
    def test_full_name(self):
        assert resolve_driver_index("George Russell") == 0

    def test_last_name(self):
        assert resolve_driver_index("Russell") == 0

    def test_abbreviation(self):
        assert resolve_driver_index("RUS") == 0

    def test_case_insensitive(self):
        assert resolve_driver_index("george russell") == 0
        assert resolve_driver_index("RUSSELL") == 0

    def test_all_drivers_resolvable(self):
        for i, d in enumerate(DRIVERS):
            assert resolve_driver_index(d["name"]) == i
            assert resolve_driver_index(d["abbr"]) == i

    def test_hulkenberg_without_umlaut(self):
        assert resolve_driver_index("Hulkenberg") is not None

    def test_unknown_returns_none(self):
        assert resolve_driver_index("Michael Schumacher") is None

    def test_whitespace_stripped(self):
        assert resolve_driver_index("  Russell  ") == 0


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

    def test_win_market_sums_to_one(self, win_odds):
        result = process_odds_to_fair_probs(win_odds)
        probs = result["win"]
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)

    def test_win_market_uses_driver_indices(self, win_odds):
        result = process_odds_to_fair_probs(win_odds)
        # Russell is driver index 0
        assert 0 in result["win"]

    def test_win_favorite_has_highest_prob(self, win_odds):
        result = process_odds_to_fair_probs(win_odds)
        probs = result["win"]
        # Russell at -150 should be the favorite
        assert probs[0] == max(probs.values())

    def test_placement_market(self):
        raw = {
            "podium": {
                "Russell": -300,
                "Antonelli": -200,
                "Leclerc": 120,
                "Hamilton": 150,
            }
        }
        result = process_odds_to_fair_probs(raw)
        probs = result["podium"]
        # Podium probs should sum to ~3 (3 podium slots)
        assert sum(probs.values()) == pytest.approx(3.0, abs=0.5)

    def test_dnf_market(self):
        raw = {
            "dnf": {
                "Russell": 1200,  # low DNF chance
                "Verstappen": 500,  # higher DNF chance
            }
        }
        result = process_odds_to_fair_probs(raw)
        dnf = result["dnf"]
        # Verstappen at +500 should have higher DNF prob than Russell at +1200
        ver_idx = resolve_driver_index("Verstappen")
        rus_idx = resolve_driver_index("Russell")
        assert dnf[ver_idx] > dnf[rus_idx]

    def test_unknown_driver_skipped(self):
        raw = {"win": {"Russell": -150, "UnknownDriver": 5000}}
        result = process_odds_to_fair_probs(raw)
        # Should still work, just fewer drivers
        assert 0 in result["win"]  # Russell resolved
        assert len(result["win"]) == 1  # Unknown skipped

    def test_empty_market_skipped(self):
        result = process_odds_to_fair_probs({"win": {}})
        assert "win" not in result

    def test_full_manual_file(self):
        """Process the actual Japanese GP odds file."""
        filepath = os.path.join(
            os.path.dirname(__file__), "..", "..", "public", "data", "odds_input", "japanese-gp-2026.json"
        )
        if not os.path.exists(filepath):
            pytest.skip("Manual odds file not found")
        with open(filepath) as f:
            data = json.load(f)
        result = process_odds_to_fair_probs(data["markets"])
        assert "win" in result
        assert len(result["win"]) >= 20  # Most drivers should resolve
        assert sum(result["win"].values()) == pytest.approx(1.0, abs=1e-3)
