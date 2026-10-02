"""Tests for update.py helpers."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from update import _race_slug


def test_race_slug_prefers_schedule_slug():
    # Display name can differ from the slug (2026's relocated Bahrain GP).
    assert _race_slug({"race": "Bahrain Grand Prix in Malaysia", "slug": "bahrain-gp"}) == "bahrain-gp"


def test_race_slug_falls_back_to_name():
    assert _race_slug({"race": "Singapore Grand Prix"}) == "singapore-gp"
    assert _race_slug({"race": "Singapore Grand Prix", "slug": ""}) == "singapore-gp"
