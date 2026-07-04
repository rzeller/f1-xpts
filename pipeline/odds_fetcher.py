"""
Odds fetcher: scrapes F1 race + sprint odds from Oddschecker via headless Chromium.

Manual JSON input is the fallback / override for markets the scraper doesn't
cover. Both sources can be combined — manual overrides scraped for the same
market.

Markets scraped (per event, when present):
  - win    → race winner (outright)
  - podium → podium finish (top 3)
  - top5   → top-5 finish        (Oddschecker market: Top 5 Finish)
  - top10  → points finish       (Oddschecker market: Points Finish)
  - dnf    → "Not To Be Classified" (binary per driver)

Sprint weekends:
  When the schedule flags is_sprint, the scraper also visits the
  `<race-slug>-sprint/...` paths (e.g. miami-grand-prix-sprint/winner) and
  returns sprint odds under a separate event key. The dual-fit pipeline
  fits a separate Plackett-Luce model per event.

Return shape (event-keyed):
  {"race":   {"win": {...}, "podium": {...}, ...},
   "sprint": {"win": {...}, ...}}     # only on sprint weekends

Race discovery:
  We derive the next-race slug from public/data/schedule.json rather than
  crawling the Oddschecker hub, to keep the browsing footprint minimal.

Anti-bot (Cloudflare):
  Oddschecker sits behind Cloudflare bot management. The scraper warms up ONE
  browser context on the event hub page, then reuses it (keeping the clearance
  cookies) across all of that event's market URLs, with human-like pacing and a
  referer. It prefers real Chrome via patchright (a stealth playwright drop-in).
  If a market is hard-blocked (403/challenge) it rotates to a fresh context once
  and retries. See _scrape_event_markets / _ScraperBrowser.

  Egress IP is the real lever, not fingerprint. Oddschecker HARD_BLOCKs both
  GitHub's datacenter IPs and raw residential-proxy pools (verified: DataImpulse
  US + EU exits all returned kind=HARD_BLOCK before any page loaded), and the
  compliance-heavy unblockers (Bright Data) policy-block gambling domains. A
  HARD_BLOCK precedes rendering, so no browser-side technique fixes it.

  Fetch backends, in precedence order:
    1. Firecrawl (FIRECRAWL_API_KEY) — PRIMARY CI PATH. A managed scrape API
       whose stealth/auto proxy clears Cloudflare and doesn't block gambling
       domains. We fetch each market page via Firecrawl (expanding the lazy
       "show more" rows with an executeJavascript action), load the returned
       HTML into a local page with set_content, and run the shared extractor.
       See _firecrawl_fetch_html / _scrape_event_markets_firecrawl.
    2. CDP unblocker (SCRAPER_BROWSER_CDP_URL) — connect over CDP to a managed
       browser that solves Cloudflare and egresses from a clean IP. Works
       technically but Bright Data (etc.) refuse gambling domains, so this is a
       fallback for a permissive provider.
    3. Raw proxy (SCRAPER_PROXY_*) / direct — local/self-hosted only.
"""

import json
import os
import random
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
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
    # Oddschecker re-introduced the DNF market under "Not To Be Classified".
    # The slug also appears with hyphenation variants on some race pages.
    "dnf": ["not-to-be-classified", "to-not-be-classified",
            "driver-not-to-finish", "driver-to-retire", "to-retire"],
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
    "dnf": "Not To Be Classified",
}

# Default page nav timeout (ms). Oddschecker is heavy; give it room.
PAGE_TIMEOUT_MS = 45000
NAV_WAIT_MS = 15000
# Timeout for connecting to a remote CDP scraping browser (ms). Provisioning a
# fresh remote browser + solving Cloudflare can take a while, so be generous.
CDP_CONNECT_TIMEOUT_MS = 120000

# Firecrawl managed-scrape backend. Oddschecker hard-blocks datacenter IPs and
# raw residential proxies (Cloudflare), and the big compliance-heavy unblockers
# (Bright Data) policy-block gambling domains. Firecrawl's stealth/auto proxy
# clears both. When FIRECRAWL_API_KEY is set the scraper fetches each market
# page through Firecrawl instead of driving a browser at Oddschecker directly.
FIRECRAWL_ENDPOINT = "https://api.firecrawl.dev/v2/scrape"
FIRECRAWL_TIMEOUT_S = 220
FIRECRAWL_MAX_ATTEMPTS = 2
# Oddschecker collapses each market to ~6 rows behind a "show more" toggle and
# lazy-loads the rest on click. A plain Firecrawl fetch (or a plain click
# action) leaves it collapsed; running the click via executeJavascript after a
# scroll expands every market on the page (verified: 36 → 112 bet rows). We then
# parse the returned, already-expanded HTML with the shared extractor.
FIRECRAWL_EXPAND_ACTIONS = [
    {"type": "scroll", "direction": "down"},
    {"type": "executeJavascript",
     "script": "document.querySelectorAll('[data-testid=\"show-more-less\"]')"
               ".forEach(b => b.click());"},
    {"type": "wait", "milliseconds": 3500},
]

# Human-like pacing between market navigations (seconds). Oddschecker sits
# behind Cloudflare bot management; back-to-back requests from a datacenter IP
# are a bot tell, so we dwell between pages. Volume is tiny (4-10 pages/run) so
# this is cheap.
PACE_MIN_S = 3.0
PACE_MAX_S = 8.0
# Dwell on the warm-up hub page so Cloudflare hands out clearance cookies in a
# low-suspicion context before we hit deep market URLs.
WARMUP_DWELL_MIN_S = 2.0
WARMUP_DWELL_MAX_S = 4.0


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


# Cloudflare interstitial / "are you human" markers. Kept specific so a normal
# market page (which may mention Cloudflare in scripts) is not misclassified.
_CHALLENGE_MARKERS = (
    "just a moment",
    "attention required",
    "verify you are human",
    "needs to review the security of your connection",
    "checking if the site connection is secure",
    "enable javascript and cookies to continue",
)


def _looks_like_challenge(page) -> bool:
    """True if the page is a Cloudflare challenge/interstitial rather than content."""
    try:
        title = (page.title() or "").lower()
    except Exception:
        title = ""
    if any(m in title for m in _CHALLENGE_MARKERS):
        return True
    try:
        text = page.evaluate(
            "() => (document.body && document.body.innerText || '').slice(0, 3000)"
        ) or ""
    except Exception:
        text = ""
    text = text.lower()
    return any(m in text for m in _CHALLENGE_MARKERS)


# Markers of an OUTRIGHT block (IP/ASN reputation, rate limit) as opposed to a
# solvable JS/managed challenge. If we see these, no browser-side tweak from
# this egress IP will help — only a different IP (proxy / self-hosted runner)
# or the manual-odds fallback will.
_HARD_BLOCK_MARKERS = (
    "you have been blocked",
    "sorry, you have been blocked",
    "access denied",
    "error 1006",
    "error 1007",
    "error 1008",
    "error 1009",
    "error 1010",
    "error 1015",
)


def _block_details(resp, page) -> str:
    """One-line diagnostic classifying a Cloudflare block: CHALLENGE (solvable,
    JS/managed) vs HARD_BLOCK (IP/ASN reputation — no browser fix helps) vs
    UNKNOWN, plus the cf-ray / cf-mitigated / server headers and a body snippet.
    """
    ray = mitigated = server = "?"
    try:
        h = resp.headers if resp is not None else {}
        ray = h.get("cf-ray", "?")
        mitigated = h.get("cf-mitigated", "?")
        server = h.get("server", "?")
    except Exception:
        pass
    try:
        title = (page.title() or "").strip()
    except Exception:
        title = ""
    try:
        text = page.evaluate("() => (document.body && document.body.innerText) || ''") or ""
    except Exception:
        text = ""
    low = (title + " " + text).lower()
    if any(m in low for m in _HARD_BLOCK_MARKERS):
        kind = "HARD_BLOCK"
    elif any(m in low for m in _CHALLENGE_MARKERS):
        kind = "CHALLENGE"
    else:
        kind = "UNKNOWN"
    snippet = " ".join(text.split())[:220]
    return (f"kind={kind} cf-ray={ray} cf-mitigated={mitigated} server={server} "
            f"title={title!r} body={snippet!r}")


def _slug_tail(url: str) -> str:
    """Last path segment of a URL, for use in debug-dump filenames."""
    return url.rstrip("/").rsplit("/", 1)[-1] or "page"


def _scrape_market_page(
    page,
    url: str,
    expected_heading: str,
    referer: Optional[str] = None,
    debug_dir: Optional[str] = None,
    max_attempts: int = 2,
) -> Tuple[Dict[str, float], bool]:
    """
    Visit an Oddschecker market page and extract { driver_name: american_odds }.

    `expected_heading` is the exact text of the market's <h*> heading — e.g.
    "Podium Finish". Each market URL on Oddschecker stacks several markets on
    the same page in arbitrary order, so we have to find the right one by name
    rather than relying on position.

    Returns (odds, blocked). `blocked` is True when the page was a Cloudflare
    403/429/challenge — the caller can then rotate to a fresh context and retry.
    A plain empty result (404, redirect, no rows) returns ({}, False).
    """
    print(f"  → {url}")

    # Navigate, retrying on a Cloudflare block with backoff. A 403/challenge on
    # the same warmed session rarely clears on retry, but a short backoff gives
    # Cloudflare's rate window room; if it stays blocked we surface blocked=True
    # so the caller can rotate the context once.
    for attempt in range(1, max_attempts + 1):
        try:
            resp = page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=PAGE_TIMEOUT_MS,
                referer=referer,
            )
        except Exception as e:
            print(f"    nav failed: {e}")
            return {}, False

        status = resp.status if resp is not None else 0
        if status in (403, 429) or _looks_like_challenge(page):
            print(f"    SCRAPER_BLOCKED status={status} url={url}")
            if attempt == 1:
                print(f"    block-detail: {_block_details(resp, page)}")
            _dump_debug(page, debug_dir, f"blocked_{status}_{_slug_tail(url)}")
            if attempt < max_attempts:
                backoff = 3.0 * attempt + random.uniform(0.0, 2.0)
                print(f"    blocked — backing off {backoff:.1f}s "
                      f"(attempt {attempt}/{max_attempts})")
                time.sleep(backoff)
                continue
            return {}, True

        if status >= 400:
            print(f"    HTTP {status} — skipping")
            return {}, False

        break  # got a usable (<400, non-challenge) response

    # If Oddschecker redirected us away from the URL we asked for (e.g. an
    # unknown market slug bounces to the F1 hub), treat it as a not-found.
    # Comparing URL prefixes ignores trailing-slash and query-string variation.
    landed = page.url.rstrip("/").split("?", 1)[0]
    requested = url.rstrip("/").split("?", 1)[0]
    if landed != requested:
        print(f"    redirected to {landed} — treating as not-found")
        _dump_debug(page, debug_dir, f"redirect_{_slug_tail(url)}")
        return {}, False

    odds = _extract_market_odds(
        page, expected_heading, url, debug_dir=debug_dir, interactive=True
    )
    return odds, False


def _extract_market_odds(
    page,
    expected_heading: str,
    url: str,
    debug_dir: Optional[str] = None,
    interactive: bool = True,
) -> Dict[str, float]:
    """Extract { driver_name: american_odds } for one market from a loaded page.

    Shared by both fetch backends:
      - interactive=True (live browser): dismiss overlays, wait for hydration,
        and click "show more" to expand the lazy-loaded rows.
      - interactive=False (Firecrawl): the page was filled via set_content from
        HTML that Firecrawl already expanded with a click-all-show-more action,
        so we skip all interaction and just read the static DOM.
    """
    if interactive:
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

    if interactive:
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
        _dump_debug(page, debug_dir, f"market_norows_{_slug_tail(url)}")
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
        _dump_debug(page, debug_dir, f"market_noresolved_{_slug_tail(url)}")
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


def _import_sync_playwright():
    """Import the sync Playwright API, preferring patchright (stealth) if present.

    patchright is a drop-in for playwright's sync_api with anti-detection
    patches (masks navigator.webdriver, avoids the CDP Runtime.enable leak). If
    it isn't installed we fall back to vanilla playwright so the pipeline still
    runs. Returns (sync_playwright, is_patchright).
    """
    try:
        from patchright.sync_api import sync_playwright
        return sync_playwright, True
    except ImportError:
        pass
    try:
        from playwright.sync_api import sync_playwright
        return sync_playwright, False
    except ImportError as e:
        raise RuntimeError(
            "Playwright is required for Oddschecker scraping. Install with: "
            "pip install patchright && patchright install chrome"
        ) from e


def _resolve_headless(headed: bool) -> bool:
    """Run headful when `headed` is set or SCRAPER_HEADFUL is truthy.

    SCRAPER_HEADFUL lets CI flip to an xvfb-run headful browser (a stronger
    anti-bot posture) without a code change.
    """
    if headed:
        return False
    env = os.environ.get("SCRAPER_HEADFUL", "").strip().lower()
    if env not in ("", "0", "false", "no"):
        return False
    return True


# Resource types we don't need for reading odds tables. Blocking them cuts
# proxy bandwidth and load time by a large factor without touching the odds
# markup or Cloudflare's JS challenge (documents, scripts and XHR still load).
_BLOCKED_RESOURCE_TYPES = {"image", "media", "font"}


def _block_heavy_resources(route) -> None:
    """Playwright route handler: abort image/media/font requests, allow the rest."""
    try:
        if route.request.resource_type in _BLOCKED_RESOURCE_TYPES:
            route.abort()
            return
    except Exception:
        pass
    try:
        route.continue_()
    except Exception:
        pass


def _should_block_resources() -> bool:
    """Block heavy resources unless SCRAPER_LOAD_IMAGES is set (escape hatch in
    case a future Cloudflare check ever expects image/font loads)."""
    env = os.environ.get("SCRAPER_LOAD_IMAGES", "").strip().lower()
    return env in ("", "0", "false", "no")


def _proxy_from_env() -> Optional[dict]:
    """Build a Playwright proxy config from env vars, or None if unset.

    Oddschecker firewall-blocks datacenter IP ranges (GitHub Actions), so to
    scrape from CI we route the browser through a residential/mobile proxy.
    Configure via repository secrets:
      SCRAPER_PROXY_SERVER   e.g. "http://gate.example.com:7000" or "socks5://host:port"
      SCRAPER_PROXY_USERNAME (optional; residential providers often encode the
                              session/country/sticky-IP here)
      SCRAPER_PROXY_PASSWORD (optional)
    Only SCRAPER_PROXY_SERVER is required to enable the proxy.
    """
    server = os.environ.get("SCRAPER_PROXY_SERVER", "").strip()
    if not server:
        return None
    proxy = {"server": server}
    user = os.environ.get("SCRAPER_PROXY_USERNAME", "").strip()
    pwd = os.environ.get("SCRAPER_PROXY_PASSWORD", "")
    if user:
        proxy["username"] = user
        proxy["password"] = pwd
    return proxy


def _firecrawl_api_key() -> Optional[str]:
    """Firecrawl API key from env, or None. When set, the Firecrawl fetch
    backend is used and takes precedence over the CDP/proxy browser paths."""
    return os.environ.get("FIRECRAWL_API_KEY", "").strip() or None


def _block_all_resources(route) -> None:
    """Abort every network request. Used for the local browser that only parses
    Firecrawl-returned HTML via set_content: the document is injected directly
    (not fetched), so we don't want it loading scripts/styles/images that could
    mutate the static DOM or hang the page."""
    try:
        route.abort()
    except Exception:
        try:
            route.continue_()
        except Exception:
            pass


class _ScraperBrowser:
    """Owns the Playwright browser/context lifecycle for one scrape run.

    Two transports, chosen by env:

    - **Remote CDP scraping browser** (SCRAPER_BROWSER_CDP_URL set): connect over
      CDP to a managed unblocker's browser (Bright Data Scraping Browser,
      Browserless, ZenRows, etc.). That service egresses from a clean residential
      IP and transparently solves Cloudflare, so all the navigation, warm-up and
      selector logic downstream runs unchanged. This is the CI path — Oddschecker
      hard-blocks both GitHub's datacenter IPs and raw residential-proxy pools
      (confirmed HARD_BLOCK across US + EU DataImpulse exits), which no
      browser-side technique can fix; the unblocker's IP reputation is the lever.
      When CDP is set the local proxy env is ignored (the remote browser owns
      egress).

    - **Local browser** (no CDP URL): prefer a persistent real-Chrome context
      (most human-like), degrading to persistent bundled-Chromium, then a plain
      launch — so it still runs where Chrome isn't installed. Routes through
      SCRAPER_PROXY_* if configured. Fine for local runs; blocked from CI.

    The first strategy that works is pinned for the rest of the run.
    `new_context()` hands out a fresh context (used both for the normal per-event
    context and the one-time rotate-after-block recovery); `close()` tears
    everything down, including temp profile dirs.

    Per patchright's guidance we do NOT set a custom user-agent, custom headers,
    or a fixed viewport — letting real Chrome present its native, coherent
    fingerprint (the old hardcoded Linux Chrome/124 UA was itself a tell).
    """

    def __init__(self, p, headless: bool, local_only: bool = False):
        self._p = p
        self._headless = headless
        self._strategy = None            # pinned once one works
        self._contexts = []
        self._browser = None
        self._profile_dirs = []
        # local_only: a plain local browser used ONLY to parse Firecrawl HTML
        # via set_content. It never touches Oddschecker, so no CDP/proxy egress
        # and we abort all subresource loads (see _register).
        self._local_only = local_only
        self._cdp_url = None if local_only else (
            os.environ.get("SCRAPER_BROWSER_CDP_URL", "").strip() or None)
        # CDP and a local proxy are mutually exclusive: the remote scraping
        # browser owns its own (clean) egress, so a local proxy would be ignored
        # at best and conflicting at worst.
        self._proxy = None if (self._cdp_url or local_only) else _proxy_from_env()
        if local_only:
            print("  browser transport: local (parsing Firecrawl-returned HTML)")
        elif self._cdp_url:
            print("  browser transport: remote CDP scraping browser "
                  "(unblocker handles Cloudflare + egress)")
        elif self._proxy:
            print(f"  proxy: routing through {self._proxy['server']}"
                  f"{' (authenticated)' if 'username' in self._proxy else ''}")

    def _persistent(self, channel):
        profile = tempfile.mkdtemp(prefix="oc-profile-")
        self._profile_dirs.append(profile)
        kwargs = dict(
            user_data_dir=profile,
            headless=self._headless,
            no_viewport=True,
            locale="en-US",
        )
        if channel:
            kwargs["channel"] = channel
        if self._proxy:
            kwargs["proxy"] = self._proxy
        return self._p.chromium.launch_persistent_context(**kwargs)

    def _persistent_chrome(self):
        return self._persistent(channel="chrome")

    def _persistent_chromium(self):
        return self._persistent(channel=None)

    def _plain_launch(self):
        if self._browser is None:
            launch_kwargs = {"headless": self._headless}
            if self._proxy:
                launch_kwargs["proxy"] = self._proxy
            self._browser = self._p.chromium.launch(**launch_kwargs)
        return self._browser.new_context(locale="en-US")

    def _cdp_connect(self):
        """Connect to a remote unblocker's browser over CDP and hand out a context.

        The remote service handles Cloudflare and egresses from a clean IP, so
        the rest of the pipeline is transport-agnostic. We try new_context()
        (isolated context, needed for the rotate-after-block recovery); some
        scraping browsers expose only a single default context, so we fall back
        to that.
        """
        if self._browser is None:
            self._browser = self._p.chromium.connect_over_cdp(
                self._cdp_url, timeout=CDP_CONNECT_TIMEOUT_MS
            )
        try:
            return self._browser.new_context(locale="en-US")
        except Exception:
            contexts = self._browser.contexts
            if contexts:
                return contexts[0]
            raise

    def _strategies(self):
        if self._cdp_url:
            return [self._cdp_connect]
        return [self._persistent_chrome, self._persistent_chromium, self._plain_launch]

    def _register(self, ctx):
        """Track the context for cleanup and apply resource blocking."""
        if self._local_only:
            # Parsing static Firecrawl HTML — block every request so injected
            # scripts/styles/images can't load and mutate the captured DOM.
            try:
                ctx.route("**/*", _block_all_resources)
            except Exception as e:
                print(f"  resource blocking unavailable: {e}")
        elif _should_block_resources():
            try:
                ctx.route("**/*", _block_heavy_resources)
            except Exception as e:
                print(f"  resource blocking unavailable: {e}")
        self._contexts.append(ctx)
        return ctx

    def new_context(self):
        if self._strategy is not None:
            return self._register(self._strategy())
        last_err = None
        for strat in self._strategies():
            try:
                ctx = strat()
            except Exception as e:
                last_err = e
                print(f"  launch strategy {strat.__name__} unavailable: {e}")
                continue
            self._strategy = strat
            print(f"  browser: {strat.__name__} (headless={self._headless})")
            return self._register(ctx)
        raise RuntimeError(f"Could not launch any browser context: {last_err}")

    def close_context(self, ctx):
        try:
            ctx.close()
        except Exception:
            pass
        if ctx in self._contexts:
            self._contexts.remove(ctx)

    def close(self):
        for ctx in list(self._contexts):
            try:
                ctx.close()
            except Exception:
                pass
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
        for d in self._profile_dirs:
            shutil.rmtree(d, ignore_errors=True)


def _warm_up(page, hub_url: str) -> bool:
    """Visit the event hub first so Cloudflare grants clearance cookies in a
    low-suspicion context before we request deep market URLs.

    Doubles as a reachability probe: returns True if the hub itself is blocked.
    When the hub is blocked there's no point hammering every market/slug for
    that event, so the caller short-circuits.
    """
    print(f"  warm-up: {hub_url}")
    try:
        resp = page.goto(hub_url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
    except Exception as e:
        print(f"    warm-up nav failed: {e}")
        return False  # unknown; let the markets try
    status = resp.status if resp is not None else 0
    if status in (403, 429) or _looks_like_challenge(page):
        print(f"    SCRAPER_BLOCKED status={status} url={hub_url} (warm-up)")
        print(f"    block-detail: {_block_details(resp, page)}")
        return True
    _dismiss_overlays(page)
    time.sleep(random.uniform(WARMUP_DWELL_MIN_S, WARMUP_DWELL_MAX_S))
    return False


def _scrape_event_markets(
    browser: "_ScraperBrowser",
    event_label: str,
    base_url: str,
    debug_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Scrape every market for one event under base_url, reusing ONE warmed
    browser context across all markets.

    Reusing the context keeps the Cloudflare clearance cookies the warm-up
    earned — the previous per-URL fresh-context approach threw those away, so
    every request after the first looked like a new cookieless visitor and got
    a 403. Markets are hit with human-like pacing between navigations, `win`
    first (it establishes the session and is the most valuable market for a
    partial run). On a hard block we rotate to a fresh context ONCE and retry
    the blocked market — the old "poisoned session" escape hatch, kept as a
    recovery path rather than the default.

    base_url is e.g. ".../miami-grand-prix" (race) or
    ".../miami-grand-prix-sprint" (sprint). Returns {market: {driver: odds}}.
    """
    out: Dict[str, Dict[str, float]] = {}

    context = browser.new_context()
    page = context.new_page()
    page.set_default_timeout(PAGE_TIMEOUT_MS)
    rotated = False

    # Warm-up doubles as a reachability probe. If the hub is blocked, rotate the
    # context once and re-probe; if it's still blocked, the whole egress IP is
    # being refused — skip this event's markets rather than burning minutes on
    # ~10 URLs that will all 403.
    if _warm_up(page, base_url):
        print("  hub blocked at warm-up — rotating context once...")
        browser.close_context(context)
        context = browser.new_context()
        page = context.new_page()
        page.set_default_timeout(PAGE_TIMEOUT_MS)
        time.sleep(random.uniform(PACE_MIN_S, PACE_MAX_S))
        rotated = True  # the one rotation is spent
        if _warm_up(page, base_url):
            print(f"  {event_label}: hub still blocked after rotation — "
                  f"skipping all markets (egress IP appears blocked)")
            return out

    first_nav = True

    for market_key, slugs in MARKET_URL_CANDIDATES.items():
        print(f"\nScraping {event_label}/{market_key}...")
        heading = MARKET_HEADING_TEXT[market_key]
        for slug in slugs:
            url = f"{base_url}/{slug}"
            if not first_nav:
                time.sleep(random.uniform(PACE_MIN_S, PACE_MAX_S))
            first_nav = False

            odds, blocked = _scrape_market_page(
                page, url, heading, referer=base_url, debug_dir=debug_dir,
            )

            if blocked and not rotated:
                rotated = True
                print("  rotating to a fresh browser context after block...")
                browser.close_context(context)
                context = browser.new_context()
                page = context.new_page()
                page.set_default_timeout(PAGE_TIMEOUT_MS)
                time.sleep(random.uniform(PACE_MIN_S, PACE_MAX_S))
                _warm_up(page, base_url)
                odds, blocked = _scrape_market_page(
                    page, url, heading, referer=base_url, debug_dir=debug_dir,
                )

            if odds:
                out[market_key] = odds
                break
        if market_key not in out:
            print(f"  {event_label}/{market_key}: no odds found across {len(slugs)} URL candidate(s)")
    return out


def _firecrawl_fetch_html(
    url: str,
    api_key: str,
    debug_dir: Optional[str] = None,
) -> Optional[str]:
    """Fetch one Oddschecker market page through Firecrawl, expanded.

    Returns the page HTML (with every market's rows expanded via the
    click-all-show-more action) or None on error/block. Uses proxy=auto so
    Firecrawl only escalates to the pricier stealth proxy if the cheap attempt
    is blocked.
    """
    payload = json.dumps({
        "url": url,
        "formats": ["html"],
        "proxy": "auto",
        "onlyMainContent": False,
        "actions": FIRECRAWL_EXPAND_ACTIONS,
    }).encode()

    for attempt in range(1, FIRECRAWL_MAX_ATTEMPTS + 1):
        req = urllib.request.Request(
            FIRECRAWL_ENDPOINT, data=payload, method="POST",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=FIRECRAWL_TIMEOUT_S) as resp:
                body = resp.read().decode("utf-8", "replace")
            data = json.loads(body)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300] if e.fp else ""
            print(f"    FIRECRAWL_ERROR status={e.code} url={url} {detail}")
            data = None
        except Exception as e:
            print(f"    FIRECRAWL_ERROR url={url} {e!r}")
            data = None

        if data is not None:
            html = (data.get("data", data) or {}).get("html") or ""
            if html:
                return html
            print(f"    Firecrawl returned no html for {url}")

        if attempt < FIRECRAWL_MAX_ATTEMPTS:
            backoff = 3.0 * attempt + random.uniform(0.0, 2.0)
            print(f"    Firecrawl retry in {backoff:.1f}s "
                  f"(attempt {attempt}/{FIRECRAWL_MAX_ATTEMPTS})")
            time.sleep(backoff)

    return None


def _scrape_event_markets_firecrawl(
    browser: "_ScraperBrowser",
    event_label: str,
    base_url: str,
    api_key: str,
    debug_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Firecrawl variant of _scrape_event_markets.

    For each market/slug, fetch the (expanded) page HTML via Firecrawl, load it
    into a local page with set_content, and run the shared extractor. No
    warm-up/pacing/rotation — Firecrawl owns the egress and Cloudflare handling,
    and an invalid slug simply yields a page with no matching market heading
    (extractor returns {}), so we fall through to the next candidate.
    """
    out: Dict[str, Dict[str, float]] = {}
    context = browser.new_context()
    page = context.new_page()
    page.set_default_timeout(PAGE_TIMEOUT_MS)

    for market_key, slugs in MARKET_URL_CANDIDATES.items():
        print(f"\nScraping {event_label}/{market_key}...")
        heading = MARKET_HEADING_TEXT[market_key]
        for slug in slugs:
            url = f"{base_url}/{slug}"
            print(f"  → {url} (via Firecrawl)")
            html = _firecrawl_fetch_html(url, api_key, debug_dir=debug_dir)
            if not html:
                continue
            try:
                page.set_content(
                    html, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS
                )
            except Exception as e:
                print(f"    set_content failed: {e}")
                continue
            odds = _extract_market_odds(
                page, heading, url, debug_dir=debug_dir, interactive=False
            )
            if odds:
                out[market_key] = odds
                break
        if market_key not in out:
            print(f"  {event_label}/{market_key}: no odds found across "
                  f"{len(slugs)} URL candidate(s)")
    return out


def fetch_all_f1_odds(headed: bool = False, debug_dir: Optional[str] = None) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], dict]:
    """
    Scrape all available F1 markets for the next race from Oddschecker.

    Returns (raw_odds_by_event, race_info) where raw_odds_by_event maps
    'race'/'sprint' → {market: {driver_name: american_odds}}. The 'sprint'
    key is only populated on sprint weekends where Oddschecker exposes a
    /…-sprint/* market path.

    A single browser context is warmed on the event hub and reused across all
    of that event's market URLs so Cloudflare's clearance cookies persist. If a
    market is hard-blocked, the context is rotated once and the market retried.
    """
    sync_playwright, is_patchright = _import_sync_playwright()

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
    sprint_url = f"{race_url}-sprint" if next_race["is_sprint"] else None
    print(f"  next race: {next_race['name']} ({race_url})")
    print(f"  scraper engine: {'patchright' if is_patchright else 'playwright'}")
    if sprint_url:
        print(f"  sprint event: {sprint_url}")

    api_key = _firecrawl_api_key()
    headless = _resolve_headless(headed)
    raw_odds: Dict[str, Dict[str, Dict[str, float]]] = {}

    if api_key:
        print("  fetch backend: Firecrawl (auto proxy; expands markets via actions)")

    with sync_playwright() as p:
        browser = _ScraperBrowser(p, headless=headless, local_only=bool(api_key))
        try:
            if api_key:
                scrape_event = lambda label, base: _scrape_event_markets_firecrawl(
                    browser, label, base, api_key, debug_dir=debug_dir,
                )
            else:
                scrape_event = lambda label, base: _scrape_event_markets(
                    browser, label, base, debug_dir=debug_dir,
                )

            race_markets = scrape_event("race", race_url)
            if race_markets:
                raw_odds["race"] = race_markets

            if sprint_url:
                sprint_markets = scrape_event("sprint", sprint_url)
                if sprint_markets:
                    raw_odds["sprint"] = sprint_markets
        finally:
            browser.close()

    return raw_odds, race_info


# ---------------------------------------------------------------------------
# Manual file loader + odds → fair-prob processing (unchanged behaviour)
# ---------------------------------------------------------------------------


def load_manual_odds(filepath: str) -> dict:
    """
    Load manually-entered odds from a JSON file.

    Two supported shapes:
      1. Single-event (race-only):
         { "markets": { "win": {...}, "podium": {...}, ... } }
      2. Dual-event (sprint weekend, separate sprint odds):
         { "markets_race": {...}, "markets_sprint": {...} }
         (or `markets` may still be present and is treated as race odds)

    See CLAUDE.md "Manual Odds Input Template" for examples.
    """
    with open(filepath) as f:
        data = json.load(f)
    return data


def _process_one_event(event_odds: dict, name_map: Dict[str, int], devig_method: str) -> Dict[str, Dict[int, float]]:
    """Devig + map names→idx for one event's markets."""
    observed_probs: Dict[str, Dict[int, float]] = {}
    for market_name, market_odds in event_odds.items():
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


def process_odds_to_fair_probs(
    raw_odds: dict,
    name_map: Dict[str, int],
    devig_method: str = "shin",
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Convert raw event-keyed odds into fair probabilities mapped to roster
    indices. Returns {event: {market: {driver_idx: fair_prob}}}.
    """
    return {
        event: _process_one_event(event_odds, name_map, devig_method)
        for event, event_odds in raw_odds.items()
    }


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
    Main entry point: get event-keyed observed probabilities.

    Priority:
    1. If `scrape` is True, attempt to fetch fresh odds from Oddschecker.
    2. Merge with manual file if provided (manual overrides scrape per-market,
       per-event).
    3. Fall back to manual file alone if scraping is disabled or yields nothing.

    Returns
    -------
    (observed_probs, race_info, raw_odds)
        observed_probs : {event: {market: {driver_idx: fair_probability}}}
            event ∈ {"race", "sprint"}; "sprint" only present on sprint
            weekends with sprint-specific odds.
        race_info : {race, date, is_sprint}
        raw_odds : {event: {market: {driver_name: american_odds}}} — pre-devig
            snapshot (post merge of scrape + manual). Saved to meta.raw_odds.
    """
    raw_odds: Dict[str, Dict[str, Dict[str, float]]] = {}
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
            print(f"  Scraped events: {list(scraped_odds.keys())}")
            for event, markets in scraped_odds.items():
                print(f"    {event}: {list(markets.keys())}")

    # Load manual file (supplements or overrides scraped per event/market).
    if manual_file and os.path.exists(manual_file):
        print(f"\nLoading manual odds from {manual_file}")
        data = load_manual_odds(manual_file)

        # Three accepted shapes (in priority order):
        #   markets_race / markets_sprint  → explicit per-event sections
        #   markets                        → race-only (legacy)
        manual_events: Dict[str, Dict[str, Dict[str, float]]] = {}
        if "markets_race" in data or "markets_sprint" in data:
            if "markets_race" in data:
                manual_events["race"] = data["markets_race"]
            if "markets_sprint" in data:
                manual_events["sprint"] = data["markets_sprint"]
        elif "markets" in data:
            manual_events["race"] = data["markets"]

        for event, event_markets in manual_events.items():
            event_bucket = raw_odds.setdefault(event, {})
            for market, odds in event_markets.items():
                if market in event_bucket:
                    print(f"  Manual overrides scraped for: {event}/{market}")
                else:
                    print(f"  Manual adds: {event}/{market}")
                event_bucket[market] = odds

        if race_info["race"] == "Unknown":
            race_info = {
                "race": data.get("race", "Unknown"),
                "date": data.get("date", ""),
                "is_sprint": data.get("is_sprint", False),
            }

    if not raw_odds:
        raise ValueError("No odds data from scraper or manual file")

    observed_probs = process_odds_to_fair_probs(raw_odds, name_map, devig_method)

    print(f"\nFinal events: {list(observed_probs.keys())}")
    for event, event_probs in observed_probs.items():
        print(f"\n  [{event}] markets: {list(event_probs.keys())}")
        for market, probs in event_probs.items():
            n = len(probs)
            top = sorted(probs.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{roster[i]['abbr']}={p:.3f}" for i, p in top)
            print(f"    {market} ({n} drivers): {top_str}")

    return observed_probs, race_info, raw_odds
