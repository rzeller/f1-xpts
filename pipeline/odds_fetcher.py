"""
Odds fetcher: scrapes F1 race odds from Oddschecker via headless Chromium.

Manual JSON input is the fallback / override for markets the scraper doesn't
cover. Both sources can be combined — manual overrides scraped for the same
market.

Markets scraped (when present):
  - win    → race winner (outright)
  - podium → podium finish (top 3)
  - top5   → top-5 finish        (Oddschecker market: Top 5 Finish)
  - top10  → points finish       (Oddschecker market: Points Finish)

Oddschecker no longer publishes a DNF market for F1. We fall back to a
default per-driver DNF probability in the model fitter when one isn't
provided.

Race discovery:
  We derive the next-race slug from public/data/schedule.json rather than
  crawling the Oddschecker hub. The hub crawl loaded enough JS / fired enough
  selector queries that Cloudflare fingerprinted the headless browser and
  served 403 challenges on every subsequent market URL. Going straight to a
  known race URL in a fresh browser context avoids that.
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from devig import american_to_implied, devig_market
from roster import resolve_driver_index as _roster_resolve


# Use the US regional path: `/us/motorsport/formula-one/`. The non-regional
# path `/motorsport/formula-1/` exists but only exposes a subset of markets
# (winner + podium-finish). The US path lists winner, podium-finish,
# top-5-finish, points-finish, winning-team and fastest-lap.
ODDSCHECKER_BASE = "https://www.oddschecker.com/us/motorsport/formula-one"

# Schedule lives at <repo>/public/data/schedule.json. odds_fetcher.py is at
# <repo>/pipeline/odds_fetcher.py — go up one level.
SCHEDULE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "data", "schedule.json")
)

# Oddschecker uses `<country>-grand-prix`. Our schedule uses `<country>-gp`.
# Most slugs translate directly via `-gp` → `-grand-prix`. The exceptions go
# here. Add new entries when a race weekend's scrape fails for slug reasons.
ODDSCHECKER_SLUG_OVERRIDES: Dict[str, str] = {
    "us-gp": "united-states-grand-prix",
}

# Each market may live under several URL slugs depending on the race weekend.
# We try them in order and use the first that yields odds.
MARKET_URL_CANDIDATES: Dict[str, List[str]] = {
    "win": ["winner"],
    "podium": ["podium-finish"],
    "top5": ["top-5-finish"],
    "top10": ["points-finish"],
}

# Each market URL on Oddschecker actually renders several market accordions
# stacked on the page (e.g. /podium-finish renders Points Finish, Fastest Lap,
# AND Podium Finish, in that order). We can't assume the first accordion is the
# one we asked for. Instead, match by the article's heading text — the heading
# always exactly matches one of the strings below.
MARKET_HEADING_TEXT: Dict[str, str] = {
    "win": "Winner",
    "podium": "Podium Finish",
    "top5": "Top 5 Finish",
    "top10": "Points Finish",
}

# Default page nav timeout (ms). Oddschecker is heavy; give it room.
PAGE_TIMEOUT_MS = 45000
NAV_WAIT_MS = 15000


def resolve_driver_index(name: str, name_map: Dict[str, int]) -> Optional[int]:
    """Match a name from odds data to a roster index using the given name_map."""
    return _roster_resolve(name, name_map)


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
_AMERICAN_RE = re.compile(r"^([+\-])(\d+)$")


def parse_odds_string(odds_str: str) -> Optional[float]:
    """
    Parse an odds string into American odds.

    Accepts:
      - American with explicit sign: "+150", "-200"
      - Fractional: "5/2"
      - Decimal: "3.50"
      - "EVS" / "EVENS" / "EVEN"
    Returns None if the string isn't recognisable as odds.

    Note: bare integers without a sign are treated as decimal odds, not
    American — e.g. "150" is decimal 150.0 (preposterously long shot in
    American), not +150. Real Oddschecker American pages always show a sign.
    """
    if not odds_str:
        return None
    s = odds_str.strip().upper().replace(" ", "")
    if s in ("EVS", "EVENS", "EVEN", "1/1"):
        return 100.0

    m = _AMERICAN_RE.match(s)
    if m:
        sign, digits = m.group(1), m.group(2)
        return float(digits) if sign == "+" else -float(digits)

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
    """Best-effort dismiss for cookie banners and webpush popups that block clicks."""
    # The webpush popup (`webpush-swal2-container`) intercepts pointer events on
    # the whole page — a CSS click handler won't dismiss it, so just remove the
    # node from the DOM. The site keeps working without it.
    try:
        page.evaluate(
            "() => document.querySelectorAll('.webpush-swal2-container, .webpush-swal2-shown').forEach(n => n.remove())"
        )
    except Exception:
        pass

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
                page.wait_for_timeout(200)
        except Exception:
            continue


def _wait_for_market_bets(page) -> None:
    """Wait for at least one bet row to render."""
    try:
        page.wait_for_selector(
            '[data-testid="market-bet"]', timeout=NAV_WAIT_MS, state="attached"
        )
    except Exception:
        try:
            page.wait_for_load_state("networkidle", timeout=NAV_WAIT_MS)
        except Exception:
            pass


def _expand_show_more(page, scope) -> None:
    """
    Click "Show More" within `scope` (a Locator) to reveal collapsed drivers.

    `scope` is the per-market AccordionWrapper — clicking the show-more inside
    it expands only that market, not other markets stacked on the same page.
    """
    # Defensive cap: each click may reveal another "Show More" (rare).
    for _ in range(5):
        try:
            buttons = scope.locator('[data-testid="show-more-less"]').all()
        except Exception:
            buttons = []
        clicked = 0
        for btn in buttons:
            try:
                txt = (btn.text_content() or "").strip().lower()
            except Exception:
                txt = ""
            if "more" not in txt:
                continue
            try:
                # JS-dispatched .click() does not fire React's synthetic event
                # listeners on this site, so use a real Playwright click.
                # scroll_into_view first because if the article is below the
                # fold, the click can be intercepted by an overlay; force=True
                # bypasses any residual overlay (e.g. webpush popup).
                btn.scroll_into_view_if_needed(timeout=2000)
                btn.click(force=True, timeout=3000)
                clicked += 1
            except Exception:
                continue
        if clicked == 0:
            return
        page.wait_for_timeout(400)


def _row_driver_name(row) -> Optional[str]:
    """
    Extract the raw bet-name text from a US-style market-bet row.

    The scraper does not try to resolve names against a known roster — it
    just returns whatever Oddschecker shows. Roster matching happens in
    process_odds_to_fair_probs against a name_map built from the Jolpica
    F1 API's current grid.
    """
    try:
        el = row.locator('[data-testid="bet-name"]').first
        txt = (el.text_content() or "").strip()
    except Exception:
        return None
    return txt or None


def _row_best_odds(row) -> Optional[float]:
    """Extract American odds from the row's bet-odds button."""
    # The button contains the odds in its first child div; siblings hold
    # tooltip/marketing content that we don't want to pull in.
    for sel in [
        '[data-testid="bet-odds"] .textWrapper_t1l74o75',
        '[data-testid="bet-odds"] > div',
        '[data-testid="bet-odds"]',
    ]:
        try:
            el = row.locator(sel).first
            txt = (el.text_content() or "").strip()
        except Exception:
            continue
        # "+150\n$10 wins ..." — keep only the leading odds token.
        first_token = txt.split()[0] if txt else ""
        am = parse_odds_string(first_token)
        if am is not None:
            return am
    return None


def _dump_debug(page, debug_dir: Optional[str], label: str) -> None:
    """Write the rendered HTML + a screenshot for post-mortem inspection."""
    if not debug_dir:
        return
    try:
        os.makedirs(debug_dir, exist_ok=True)
        safe = re.sub(r"[^a-z0-9_\-]+", "_", label.lower()).strip("_") or "page"
        html_path = os.path.join(debug_dir, f"{safe}.html")
        png_path = os.path.join(debug_dir, f"{safe}.png")
        try:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(page.content())
            print(f"    [debug] wrote {html_path}")
        except Exception as e:
            print(f"    [debug] HTML dump failed: {e}")
        try:
            page.screenshot(path=png_path, full_page=True)
            print(f"    [debug] wrote {png_path}")
        except Exception as e:
            print(f"    [debug] screenshot failed: {e}")
    except Exception as e:
        print(f"    [debug] dump failed: {e}")


def _scrape_market_page(
    page,
    url: str,
    expected_heading: str,
    debug_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Visit an Oddschecker market page and extract { driver_name: american_odds }.

    `expected_heading` is the exact text of the market's <h*> heading — e.g.
    "Podium Finish". Each market URL on Oddschecker stacks several markets on
    the same page in arbitrary order, so we have to find the right one by name
    rather than relying on position.

    Returns {} on failure (404, redirect to a different page, no rows, etc.).
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

    # If Oddschecker redirected us away from the URL we asked for (e.g. an
    # unknown market slug bounces to the F1 hub), treat it as a not-found.
    # Comparing URL prefixes ignores trailing-slash and query-string variation.
    landed = page.url.rstrip("/").split("?", 1)[0]
    requested = url.rstrip("/").split("?", 1)[0]
    if landed != requested:
        print(f"    redirected to {landed} — treating as not-found")
        return {}

    _dismiss_overlays(page)
    _wait_for_market_bets(page)

    # Find the MarketWrapper article whose heading text exactly matches the
    # market we want. Each article wraps one market's heading + AccordionWrapper.
    # We do the matching in JS because Playwright's `:has(:text-is())` chain is
    # finicky with quoting and the Locator filter API needs a sub-locator on a
    # base set we'd have to enumerate first anyway.
    article_index = page.evaluate(
        """(heading) => {
            const articles = document.querySelectorAll('article[class*="MarketWrapper"]');
            for (let i = 0; i < articles.length; i++) {
                const h = articles[i].querySelector('h1,h2,h3,h4');
                if (h && (h.textContent || '').trim() === heading) return i;
            }
            return -1;
        }""",
        expected_heading,
    )
    if article_index < 0:
        print(f"    WARNING: no article with heading {expected_heading!r} on page")
        return {}

    market_article = page.locator('article[class*="MarketWrapper"]').nth(article_index)
    market_scope = market_article.locator('[class*="AccordionWrapper"]').first

    # Wait for the show-more button to appear inside the right accordion.
    # When the target article is below the fold, the AccordionWrapper exists
    # but its show-more button may still be hydrating, and clicking it before
    # then is a no-op — leading to a stuck 6-row collapsed view.
    try:
        market_scope.locator('[data-testid="show-more-less"]').first.wait_for(
            state="attached", timeout=NAV_WAIT_MS
        )
    except Exception:
        pass

    _expand_show_more(page, market_scope)
    # The webpush overlay can reappear after async loads.
    _dismiss_overlays(page)

    odds: Dict[str, float] = {}

    try:
        rows = market_scope.locator('[data-testid="market-bet"]').all()
    except Exception:
        rows = []

    if not rows:
        print(f"    WARNING: no [data-testid=market-bet] rows in {expected_heading!r}")
        _dump_debug(page, debug_dir, f"market_norows_{url.rstrip('/').rsplit('/', 1)[-1]}")
        return {}

    print(f"    matched {len(rows)} bet rows in {expected_heading!r}")

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
    if not odds:
        _dump_debug(page, debug_dir, f"market_noresolved_{url.rstrip('/').rsplit('/', 1)[-1]}")
    return odds


def _oddschecker_slug(repo_slug: str) -> str:
    """Convert a schedule.json slug (e.g. 'miami-gp') to Oddschecker's URL form."""
    if repo_slug in ODDSCHECKER_SLUG_OVERRIDES:
        return ODDSCHECKER_SLUG_OVERRIDES[repo_slug]
    if repo_slug.endswith("-gp"):
        return repo_slug[: -len("-gp")] + "-grand-prix"
    return repo_slug


def _next_race_from_schedule(
    schedule_path: str = SCHEDULE_PATH,
    now: Optional[datetime] = None,
) -> Optional[dict]:
    """
    Pick the next upcoming race from the schedule file.

    Returns a dict with:
      - name:     human-readable race name
      - slug:     Oddschecker URL slug (e.g. 'miami-grand-prix')
      - date:     YYYY-MM-DD of race day (UTC)
      - is_sprint: bool
    or None if the schedule has no future races.

    "Next" = first race whose race-session start is >= now. We don't apply a
    grace window: by the time the GitHub Actions cron fires Friday morning,
    the previous race is days behind us and Oddschecker's lines for the
    upcoming race are already up.
    """
    if not os.path.exists(schedule_path):
        print(f"  schedule.json not found at {schedule_path}")
        return None

    with open(schedule_path) as f:
        schedule = json.load(f)

    now = now or datetime.now(timezone.utc)

    upcoming = []
    for r in schedule.get("races", []):
        race_iso = r.get("sessions", {}).get("race")
        if not race_iso:
            continue
        try:
            race_dt = datetime.fromisoformat(race_iso.replace("Z", "+00:00"))
        except ValueError:
            continue
        if race_dt >= now:
            upcoming.append((race_dt, r))

    if not upcoming:
        print("  No upcoming races found in schedule.json")
        return None

    upcoming.sort(key=lambda x: x[0])
    race_dt, r = upcoming[0]
    return {
        "name": r["name"],
        "slug": _oddschecker_slug(r["slug"]),
        "date": race_dt.date().isoformat(),
        "is_sprint": bool(r.get("is_sprint", False)),
    }


def fetch_all_f1_odds(headed: bool = False, debug_dir: Optional[str] = None) -> Tuple[Dict[str, Dict[str, float]], dict]:
    """
    Scrape all available F1 markets for the next race from Oddschecker.

    Returns (raw_odds_by_market, race_info) where raw_odds_by_market maps
    'win'/'podium'/'top6'/'dnf' → { driver_name: american_odds }.

    A fresh browser context is opened for each market URL attempt: a single
    bad URL (redirect, 403) on Oddschecker poisons the session via Cloudflare
    fingerprinting, so we don't reuse a context that's seen a failure.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "Playwright is required for Oddschecker scraping. "
            "Install with: pip install playwright && playwright install chromium"
        ) from e

    print("Discovering next F1 race from schedule.json...")
    next_race = _next_race_from_schedule()
    if not next_race:
        return {}, {"race": "Unknown", "date": "", "is_sprint": False}

    race_info = {
        "race": next_race["name"],
        "date": next_race["date"],
        "is_sprint": next_race["is_sprint"],
    }
    race_url = f"{ODDSCHECKER_BASE}/{next_race['slug']}"
    print(f"  next race: {next_race['name']} ({race_url})")

    raw_odds: Dict[str, Dict[str, float]] = {}

    user_agent = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed)
        try:
            for market_key, slugs in MARKET_URL_CANDIDATES.items():
                print(f"\nScraping {market_key} market...")
                heading = MARKET_HEADING_TEXT[market_key]
                for slug in slugs:
                    url = f"{race_url}/{slug}"
                    context = browser.new_context(
                        viewport={"width": 1366, "height": 900},
                        user_agent=user_agent,
                        locale="en-GB",
                    )
                    try:
                        page = context.new_page()
                        page.set_default_timeout(PAGE_TIMEOUT_MS)
                        odds = _scrape_market_page(page, url, heading, debug_dir=debug_dir)
                    finally:
                        try:
                            context.close()
                        except Exception:
                            pass
                    if odds:
                        raw_odds[market_key] = odds
                        break
                if market_key not in raw_odds:
                    print(f"  {market_key}: no odds found across {len(slugs)} URL candidate(s)")
        finally:
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
    name_map: Dict[str, int],
    devig_method: str = "shin",
) -> Dict[str, Dict[int, float]]:
    """
    Convert raw odds (from scraper or manual) into fair probabilities mapped to
    roster indices. `name_map` is built from the active roster via
    roster.build_name_map; any odds-source name that doesn't resolve to a
    roster slot is dropped with a warning.
    """
    observed_probs = {}

    for market_name, market_odds in raw_odds.items():
        if not market_odds:
            continue

        # DNF is a binary market per driver, not a multi-runner market
        if market_name == "dnf":
            probs = {}
            for name, odds_val in market_odds.items():
                idx = resolve_driver_index(name, name_map)
                if idx is None:
                    print(f"  Warning: could not match '{name}'")
                    continue
                implied = american_to_implied(odds_val)
                probs[idx] = implied * 0.95  # Light devig for binary
            observed_probs["dnf"] = probs
            continue

        # Placement markets (podium, top5, top6, top10) are binary yes/no per
        # driver: "Will this driver finish in the top N?" Each driver's prob is
        # independent, so they should sum to N (not 1). Devig as binary markets.
        #
        # Win market is a true multi-runner outright: exactly one winner,
        # probabilities should sum to 1. Use Shin's method.
        placement_slots = {"podium": 3, "top5": 5, "top6": 6, "top10": 10}

        if market_name in placement_slots:
            probs = {}
            target_sum = placement_slots[market_name]
            # Devig as binary markets, then rescale so they sum to target
            raw_implied = {}
            for name, odds_val in market_odds.items():
                idx = resolve_driver_index(name, name_map)
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
                idx = resolve_driver_index(name, name_map)
                if idx is None:
                    print(f"  Warning: could not match '{name}'")
                    continue
                probs[idx] = prob

            observed_probs[market_name] = probs

    return observed_probs


def get_observed_probs(
    roster: List[dict],
    name_map: Dict[str, int],
    manual_file: Optional[str] = None,
    scrape: bool = True,
    devig_method: str = "shin",
    headed: bool = False,
    debug_dir: Optional[str] = None,
) -> tuple:
    """
    Main entry point: get observed probabilities keyed by roster index.

    Priority:
    1. If `scrape` is True, attempt to fetch fresh odds from Oddschecker.
    2. Merge with manual file if provided (manual overrides scrape per-market).
    3. Fall back to manual file alone if scraping is disabled or yields nothing.

    Returns
    -------
    (observed_probs, race_info, raw_odds)
        observed_probs : {market: {driver_idx: fair_probability}}
        race_info : {race, date, is_sprint}
        raw_odds : {market: {driver_name: american_odds}} — pre-devig snapshot
            (post merge of scrape + manual). Saved to meta.raw_odds so future
            audits can tell scraper / devig / model failures apart (issue #36).
    """
    raw_odds: Dict[str, Dict[str, float]] = {}
    race_info = {"race": "Unknown", "date": "", "is_sprint": False}

    # Attempt scrape
    if scrape:
        try:
            scraped_odds, scraped_info = fetch_all_f1_odds(headed=headed, debug_dir=debug_dir)
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

    observed_probs = process_odds_to_fair_probs(raw_odds, name_map, devig_method)

    print(f"\nFinal markets: {list(observed_probs.keys())}")
    for market, probs in observed_probs.items():
        n = len(probs)
        top = sorted(probs.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{roster[i]['abbr']}={p:.3f}" for i, p in top)
        print(f"  {market} ({n} drivers): {top_str}")

    return observed_probs, race_info, raw_odds
