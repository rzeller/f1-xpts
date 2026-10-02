"""Tests for update_schedule.py — merging the Jolpica calendar into schedule.json."""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from update_schedule import merge, slug_for, summarize

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _existing_race(round_, name, slug, location, country, race_iso, sprint=False):
    sessions = {"fp1": race_iso[:8] + "01T10:00:00Z", "qualifying": race_iso[:8] + "02T14:00:00Z", "race": race_iso}
    return {"round": round_, "name": name, "slug": slug, "location": location, "circuit": "C",
            "country": country, "timezone": "X", "utc_offset": 0, "is_sprint": sprint, "sessions": sessions}


def _api_race(name, locality, country, date, time="13:00:00Z", sprint=False):
    r = {"raceName": name, "date": date, "time": time,
         "Circuit": {"circuitName": f"{locality} Circuit", "Location": {"locality": locality, "country": country}},
         "FirstPractice": {"date": date[:8] + "%02d" % (int(date[8:]) - 2), "time": "10:00:00Z"},
         "Qualifying": {"date": date[:8] + "%02d" % (int(date[8:]) - 1), "time": "14:00:00Z"}}
    if sprint:
        r["SprintQualifying"] = {"date": r["FirstPractice"]["date"], "time": "14:00:00Z"}
        r["Sprint"] = {"date": r["Qualifying"]["date"], "time": "10:00:00Z"}
    return r


@pytest.fixture
def existing():
    return {"season": 2026, "races": [
        _existing_race(1, "Azerbaijan Grand Prix", "azerbaijan-gp", "Baku", "Azerbaijan", "2026-09-26T11:00:00Z"),
        _existing_race(2, "Singapore Grand Prix", "singapore-gp", "Singapore", "Singapore", "2026-10-11T12:00:00Z", True),
        _existing_race(3, "Mexico City Grand Prix", "mexico-gp", "Mexico City", "Mexico", "2026-11-01T20:00:00Z"),
    ]}


def test_slug_for_matches_update_py_convention():
    assert slug_for("Bahrain Grand Prix") == "bahrain-gp"
    assert slug_for("São Paulo Grand Prix") == "sao-paulo-gp"


def test_inserts_relocated_race_and_renumbers(existing):
    api = [
        _api_race("Bahrain Grand Prix", "Sakhir", "Bahrain", "2026-04-12"),  # stale/cancelled, in the past
        _api_race("Azerbaijan Grand Prix", "Baku", "Azerbaijan", "2026-09-27"),  # past: ignored
        _api_race("Bahrain Grand Prix", "Sepang", "Malaysia", "2026-10-04", "07:00:00Z"),
        _api_race("Singapore Grand Prix", "Marina Bay", "Singapore", "2026-10-11", "12:00:00Z", sprint=True),
        _api_race("Mexico City Grand Prix", "Mexico City", "Mexico", "2026-11-01", "20:00:00Z"),
    ]
    new = merge(existing, api, NOW)
    names = [(r["round"], r["slug"]) for r in new["races"]]
    assert names == [(1, "azerbaijan-gp"), (2, "bahrain-gp"), (3, "singapore-gp"), (4, "mexico-gp")]
    # Finished race untouched (API's different date ignored).
    assert new["races"][0]["sessions"]["race"] == "2026-09-26T11:00:00Z"
    bah = new["races"][1]
    assert bah["location"] == "Sepang" and bah["country"] == "Malaysia"
    assert bah["utc_offset"] == 8 and bah["is_sprint"] is False
    assert bah["sessions"]["race"] == "2026-10-04T07:00:00Z"
    sgp = new["races"][2]
    assert sgp["is_sprint"] is True and "sprint_qualifying" in sgp["sessions"]
    # Existing slug preserved even though the name-derived one differs.
    assert new["races"][3]["slug"] == "mexico-gp"
    assert any(l.startswith("+ added Bahrain") for l in summarize(existing, new))


def test_refuses_on_truncated_api_response(existing):
    with pytest.raises(SystemExit):
        merge(existing, [], NOW)


def test_missing_time_keeps_existing(existing):
    api = [
        _api_race("Singapore Grand Prix", "Marina Bay", "Singapore", "2026-10-11", time=None, sprint=True),
        _api_race("Mexico City Grand Prix", "Mexico City", "Mexico", "2026-11-01", "20:00:00Z"),
    ]
    del api[0]["time"]
    new = merge(existing, api, NOW)
    assert new["races"][1]["sessions"]["race"] == "2026-10-11T12:00:00Z"


def test_renamed_race_matches_on_circuit():
    existing = {"season": 2026, "races": [
        _existing_race(1, "Bahrain Grand Prix", "bahrain-gp", "Sepang", "Malaysia", "2026-10-04T07:00:00Z"),
    ]}
    existing["races"][0]["circuit"] = "Sepang International Circuit"
    api = [_api_race("Bahrain Grand Prix in Malaysia", "Kuala Lumpur", "Malaysia", "2026-10-04", "07:00:00Z")]
    api[0]["Circuit"]["circuitName"] = "Sepang International Circuit"
    race = merge(existing, api, NOW)["races"][0]
    assert race["name"] == "Bahrain Grand Prix"
    assert race["slug"] == "bahrain-gp" and race["location"] == "Sepang"
    assert race["timezone"] == "MYT"
