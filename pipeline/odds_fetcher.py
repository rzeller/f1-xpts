"""
Odds fetcher: scrapes F1 race odds from Oddschecker via headless Chromium.

Manual JSON input is the fallback / override for markets the scraper doesn't
cover. Both sources can be combined — manual overrides scraped for the same
market.

Markets scraped (when present):
  - win    → race winner (outright)
  - podium → podium finish (top 3)
  - top6   → top-6 finish
  - dnf    → driver to not be classified / retire
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from config import DRIVERS, DRIVER_NAME_MAP
from devig import american_to_implied, devig_market


ODDSCHECKER_BASE = "https://www.oddschecker.com/motorsport/formula-1"

# Each market may live under several URL slugs depending on the race weekend.
# We try them in order and use the first that yields odds.
MARKET_URL_CANDIDATES: Dict[str, List[str]] = {
    "win": ["winner", "race-winner", "outright-winner"],
    "podium": ["podium-finish", "to-finish-on-podium", "podium"],
    "top6": ["top-6-finish", "to-finish-in-top-6", "top-six-finish"],
    "dnf": [
        "to-not-be-classified",
        "not-to-be-classified",
        "driver-to-retire",
        "to-retire",
        "driver-not-to-finish",
    ],
}

# Default page nav timeout (ms). Oddschecker is heavy; give it room.
PAGE_TIMEOUT_MS = 45000
NAV_WAIT_MS = 15000


def resolve_driver_index(name: str) -> Optional[int]:
    """Match a name from odds data to our canonical driver index."""
    key = name.lower().strip()
    if key in DRIVER_NAME_MAP:
        return DRIVER_NAME_MAP[key]

    # Try each word (handles "George Russell" → match on "russell")
    for word in key.split():
        if word in DRIVER_NAME_MAP:
            return DRIVER_NAME_MAP[word]

    return None


# ---------------------------------------------------------------------------
# Odds format conversion
# ---------------------------------------------------------------------------


def fractional_to_american(num: float, den: float) -> float:
    """Convert fractional odds (e.g. 5/2) to American."""
    if den == 0:
        return 0.0
    if num >= den:
        return round(100.0 * num / den)
    return -round(100.0 * den / num)


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds to American."""
    if decimal <= 1.0:
        return 0.0
    if decimal >= 2.0:
        return round((decimal - 1.0) * 100.0)
    return -round(100.0 / (decimal - 1.0))


_FRACTIONAL_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")
_DECIMAL_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*$")


def parse_odds_string(odds_str: str) -> Optional[float]:
    """
    Parse an odds string into American odds.

    Accepts fractional ("5/2"), decimal ("3.50"), or "EVS"/"EVENS"/"EVEN".
    Returns None if the string isn't recognisable as odds.
    """
    if not odds_str:
        return None
    s = odds_str.strip().upper().replace(" ", "")
    if s in ("EVS", "EVENS", "EVEN", "1/1"):
        return 100.0

    m = _FRACTIONAL_RE.match(s)
    if m:
        return fractional_to_american(float(m.group(1)), float(m.group(2)))

    if _DECIMAL_RE.match(s):
        try:
            return decimal_to_american(float(s))
        except ValueError:
            return None

    return None


# ---------------------------------------------------------------------------
# Playwright scraping
# ---------------------------------------------------------------------------


def _dismiss_overlays(page) -> None:
    """Best-effort dismiss for cookie banners / age gates that block content."""
    selectors = [
        'button:has-text("Accept All")',
        'button:has-text("Accept all")',
        'button:has-text("Accept")',
        'button:has-text("I Accept")',
        'button:has-text("Agree")',
        '[id*="onetrust-accept"]',
        'button#onetrust-accept-btn-handler',
        'button[aria-label*="ccept"]',
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.is_visible(timeout=500):
                loc.click(timeout=1500)
                page.wait_for_timeout(300)
        except Exception:
            continue


def _wait_for_market_table(page) -> None:
    """Wait for the odds grid to render — try several selectors."""
    candidates = [
        '[data-testid*="market"]',
        '[data-testid*="selection"]',
        'tr[data-bname]',
        'table.eventTable',
        'table.diff-row',
        'table tbody tr',
    ]
    for sel in candidates:
        try:
            page.wait_for_selector(sel, timeout=NAV_WAIT_MS, state="attached")
            return
        except Exception:
            continue
    # Fall back to networkidle
    try:
        page.wait_for_load_state("networkidle", timeout=NAV_WAIT_MS)
    except Exception:
        pass


def _row_best_odds(row) -> Optional[float]:
    """Pick the best American odds from a row."""
    candidates: List[float] = []

    # Strategy A: explicit "best" cell
    for sel in [
        'td.bs',
        'td.bs-best',
        'td[class*="best"]',
        '[data-testid*="best-odds"]',
        '[class*="bestOdds"]',
    ]:
        try:
            els = row.locator(sel).all()
        except Exception:
            els = []
        for el in els:
            try:
                txt = (el.text_content() or "").strip()
            except Exception:
                continue
            am = parse_odds_string(txt)
            if am is not None:
                candidates.append(am)

    # Strategy B: data-odig attribute (legacy: implied probability as decimal)
    try:
        odig_els = row.locator('[data-odig]').all()
    except Exception:
        odig_els = []
    for el in odig_els:
        try:
            v = el.get_attribute("data-odig")
            if v:
                implied = float(v)
                if 0.0 < implied < 1.0:
                    decimal = 1.0 / implied
                    candidates.append(decimal_to_american(decimal))
        except Exception:
            continue

    # Strategy C: scan every cell for parseable odds
    if not candidates:
        try:
            cells = row.locator("td").all()
        except Exception:
            cells = []
        for cell in cells:
            try:
                txt = (cell.text_content() or "").strip()
            except Exception:
                continue
            am = parse_odds_string(txt)
            if am is not None:
                candidates.append(am)

    if not candidates:
        return None

    # Best odds for the punter = highest payout = highest American value
    # (where positive > negative, and -120 > -150).
    return max(candidates)


def _row_driver_name(row) -> Optional[str]:
    """Extract the driver name from a row."""
    # Attribute-based (legacy Oddschecker)
    for attr in ("data-bname", "data-name", "data-runner"):
        try:
            v = row.get_attribute(attr)
        except Exception:
            v = None
        if v and resolve_driver_index(v) is not None:
            return v.strip()

    # Common name-cell selectors
    for sel in [
        '[data-testid*="selection-name"]',
        '[data-testid*="participant"]',
        'td.popup',
        'td.beta-cell',
        'td.sel',
        'td[class*="name"]',
        'th[scope="row"]',
        'td:first-child',
    ]:
        try:
            el = row.locator(sel).first
            txt = (el.text_content() or "").strip()
        except Exception:
            txt = ""
        if txt and resolve_driver_index(txt) is not None:
            return txt

    # Last resort: scan all cells for one that resolves to a known driver
    try:
        cells = row.locator("td, th").all()
    except Exception:
        cells = []
    for cell in cells:
        try:
            txt = (cell.text_content() or "").strip()
        except Exception:
            continue
        if txt and resolve_driver_index(txt) is not None:
            return txt

    return None


def _scrape_market_page(page, url: str) -> Dict[str, float]:
    """
    Visit an Oddschecker market page and extract { driver_name: american_odds }.
    Returns {} on failure (404, no rows recognised, etc.).
    """
    print(f"  → {url}")
    try:
        resp = page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
    except Exception as e:
        print(f"    nav failed: {e}")
        return {}

    if resp is not None and resp.status >= 400:
        print(f"    HTTP {resp.status} — skipping")
        return {}

    _dismiss_overlays(page)
    _wait_for_market_table(page)

    odds: Dict[str, float] = {}

    # Iterate rows that look like selection rows. Try the most-specific
    # selector first, falling back to a generic table-row scan.
    row_selectors = [
        'tr[data-bname]',
        '[data-testid*="selection-row"]',
        '[data-testid*="selectionRow"]',
        '[data-testid*="market"] tbody tr',
        'table.eventTable tbody tr',
        'table tbody tr',
    ]

    rows = []
    for sel in row_selectors:
        try:
            found = page.locator(sel).all()
        except Exception:
            found = []
        if found:
            rows = found
            print(f"    matched {len(rows)} rows via `{sel}`")
            break

    if not rows:
        print("    WARNING: no candidate rows found on page")
        return {}

    for row in rows:
        try:
            name = _row_driver_name(row)
            if not name:
                continue
            best = _row_best_odds(row)
            if best is None:
                continue
            # Keep the best (highest American) if a driver appears twice
            prior = odds.get(name)
            if prior is None or best > prior:
                odds[name] = best
        except Exception:
            continue

    print(f"    extracted {len(odds)} drivers")
    return odds


def _find_next_race(page) -> Tuple[Optional[str], dict]:
    """
    Visit the F1 hub page and identify the upcoming race base URL.

    Returns (race_url, race_info) where race_url is e.g.
    'https://www.oddschecker.com/motorsport/formula-1/japanese-grand-prix'
    and race_info has the race name.
    """
    print("Discovering next F1 race on Oddschecker...")
    try:
        page.goto(ODDSCHECKER_BASE, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
    except Exception as e:
        print(f"  hub nav failed: {e}")
        return None, {"race": "Unknown", "date": "", "is_sprint": False}

    _dismiss_overlays(page)
    try:
        page.wait_for_load_state("networkidle", timeout=NAV_WAIT_MS)
    except Exception:
        pass

    # Find links pointing into a race page. Prefer ones that already include
    # a known market suffix (most reliable signal that the slug is a race slug,
    # not a navigation/category link).
    race_pattern = re.compile(
        r"/motorsport/formula-1/([a-z0-9][a-z0-9\-]*?-(?:grand-prix|gp))(?:/|$)"
    )

    found_slug: Optional[str] = None
    found_label = ""

    try:
        links = page.locator('a[href*="/motorsport/formula-1/"]').all()
    except Exception:
        links = []

    for link in links:
        try:
            href = link.get_attribute("href") or ""
            text = (link.text_content() or "").strip()
        except Exception:
            continue
        m = race_pattern.search(href)
        if not m:
            continue
        slug = m.group(1)
        if slug in ("formula-1", "outright", "outrights", "specials"):
            continue
        found_slug = slug
        if text and "winner" not in text.lower():
            found_label = text
        break

    if not found_slug:
        # Looser pattern: any non-trivial slug under /motorsport/formula-1/
        for link in links:
            try:
                href = link.get_attribute("href") or ""
            except Exception:
                continue
            m = re.search(r"/motorsport/formula-1/([a-z0-9][a-z0-9\-]+)", href)
            if not m:
                continue
            slug = m.group(1)
            if slug in ("formula-1", "outright", "outrights", "specials", "drivers-championship", "constructors-championship"):
                continue
            found_slug = slug
            break

    if not found_slug:
        print("  ERROR: no upcoming race link found on hub page")
        return None, {"race": "Unknown", "date": "", "is_sprint": False}

    race_url = f"{ODDSCHECKER_BASE}/{found_slug}"
    race_name = found_label or found_slug.replace("-", " ").title()
    print(f"  next race: {race_name} ({race_url})")
    return race_url, {"race": race_name, "date": "", "is_sprint": False}


def fetch_all_f1_odds() -> Tuple[Dict[str, Dict[str, float]], dict]:
    """
    Scrape all available F1 markets for the next race from Oddschecker.

    Returns (raw_odds_by_market, race_info) where raw_odds_by_market maps
    'win'/'podium'/'top6'/'dnf' → { driver_name: american_odds }.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "Playwright is required for Oddschecker scraping. "
            "Install with: pip install playwright && playwright install chromium"
        ) from e

    raw_odds: Dict[str, Dict[str, float]] = {}
    race_info: dict = {"race": "Unknown", "date": "", "is_sprint": False}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1366, "height": 900},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-GB",
        )
        page = context.new_page()
        page.set_default_timeout(PAGE_TIMEOUT_MS)

        try:
            race_url, race_info = _find_next_race(page)
            if not race_url:
                return {}, race_info

            for market_key, slugs in MARKET_URL_CANDIDATES.items():
                print(f"\nScraping {market_key} market...")
                for slug in slugs:
                    url = f"{race_url}/{slug}"
                    odds = _scrape_market_page(page, url)
                    if odds:
                        raw_odds[market_key] = odds
                        break
                if market_key not in raw_odds:
                    print(f"  {market_key}: no odds found across {len(slugs)} URL candidate(s)")
        finally:
            try:
                context.close()
            except Exception:
                pass
            browser.close()

    return raw_odds, race_info


# ---------------------------------------------------------------------------
# Manual file loader + odds → fair-prob processing (unchanged behaviour)
# ---------------------------------------------------------------------------


def load_manual_odds(filepath: str) -> dict:
    """
    Load manually-entered odds from a JSON file.

    Expected format: see CLAUDE.md "Manual Odds Input Template"
    """
    with open(filepath) as f:
        data = json.load(f)
    return data


def process_odds_to_fair_probs(
    raw_odds: dict,
    devig_method: str = "shin",
) -> Dict[str, Dict[int, float]]:
    """
    Convert raw odds (from scraper or manual) into fair probabilities
    mapped to canonical driver indices.
    """
    observed_probs = {}

    for market_name, market_odds in raw_odds.items():
        if not market_odds:
            continue

        # DNF is a binary market per driver, not a multi-runner market
        if market_name == "dnf":
            probs = {}
            for name, odds_val in market_odds.items():
                idx = resolve_driver_index(name)
                if idx is None:
                    print(f"  Warning: could not match '{name}'")
                    continue
                implied = american_to_implied(odds_val)
                probs[idx] = implied * 0.95  # Light devig for binary
            observed_probs["dnf"] = probs
            continue

        # Placement markets (podium, top6, top10) are binary yes/no per driver:
        # "Will this driver finish in the top N?" Each driver's prob is independent,
        # so they should sum to N (not 1). Devig each as a binary market.
        #
        # Win market is a true multi-runner outright: exactly one winner,
        # probabilities should sum to 1. Use Shin's method.
        placement_slots = {"podium": 3, "top6": 6, "top10": 10}

        if market_name in placement_slots:
            probs = {}
            target_sum = placement_slots[market_name]
            # Devig as binary markets, then rescale so they sum to target
            raw_implied = {}
            for name, odds_val in market_odds.items():
                idx = resolve_driver_index(name)
                if idx is None:
                    print(f"  Warning: could not match '{name}'")
                    continue
                raw_implied[idx] = american_to_implied(odds_val)

            # The overround is distributed across all runners
            total_implied = sum(raw_implied.values())
            scale = target_sum / total_implied
            for idx, imp in raw_implied.items():
                probs[idx] = imp * scale

            observed_probs[market_name] = probs
        else:
            # Win market: true multi-runner outright, devig with Shin's method
            fair = devig_market(market_odds, format="american", method=devig_method)

            probs = {}
            for name, prob in fair.items():
                idx = resolve_driver_index(name)
                if idx is None:
                    print(f"  Warning: could not match '{name}'")
                    continue
                probs[idx] = prob

            observed_probs[market_name] = probs

    return observed_probs


def get_observed_probs(
    manual_file: Optional[str] = None,
    scrape: bool = True,
    devig_method: str = "shin",
) -> tuple:
    """
    Main entry point: get observed probabilities.

    Priority:
    1. If `scrape` is True, attempt to fetch fresh odds from Oddschecker.
    2. Merge with manual file if provided (manual overrides scrape per-market).
    3. Fall back to manual file alone if scraping is disabled or yields nothing.
    """
    raw_odds: Dict[str, Dict[str, float]] = {}
    race_info = {"race": "Unknown", "date": "", "is_sprint": False}

    # Attempt scrape
    if scrape:
        try:
            scraped_odds, scraped_info = fetch_all_f1_odds()
        except Exception as e:
            print(f"  Scrape failed: {e}")
            scraped_odds, scraped_info = {}, race_info

        if scraped_odds:
            raw_odds.update(scraped_odds)
            race_info = scraped_info
            print(f"  Scraped markets: {list(scraped_odds.keys())}")

    # Load manual file (supplements or overrides scraped)
    if manual_file and os.path.exists(manual_file):
        print(f"\nLoading manual odds from {manual_file}")
        data = load_manual_odds(manual_file)
        for market, odds in data.get("markets", {}).items():
            if market in raw_odds:
                print(f"  Manual overrides scraped for: {market}")
            else:
                print(f"  Manual adds: {market}")
            raw_odds[market] = odds

        if race_info["race"] == "Unknown":
            race_info = {
                "race": data.get("race", "Unknown"),
                "date": data.get("date", ""),
                "is_sprint": data.get("is_sprint", False),
            }

    if not raw_odds:
        raise ValueError("No odds data from scraper or manual file")

    observed_probs = process_odds_to_fair_probs(raw_odds, devig_method)

    print(f"\nFinal markets: {list(observed_probs.keys())}")
    for market, probs in observed_probs.items():
        n = len(probs)
        top = sorted(probs.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{DRIVERS[i]['abbr']}={p:.3f}" for i, p in top)
        print(f"  {market} ({n} drivers): {top_str}")

    return observed_probs, race_info
