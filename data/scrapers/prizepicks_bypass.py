"""
PrizePicks Golf Lines Fetcher — PerimeterX Bypass
6-step fallback: DB -> curl_cffi -> cloudscraper -> Relay -> Disk Cache -> Direct

Bypasses:
  - curl_cffi with Chrome120 TLS impersonation (primary)
  - cloudscraper for Cloudflare JS challenges
  - Relay server (pp_relay.py running on residential IP)
  - Scraper service SQLite DB
  - Disk cache fallback
"""

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── API Endpoints ─────────────────────────────────────────────────────────────
PRIZEPICKS_API = "https://api.prizepicks.com/projections"
PRIZEPICKS_LEAGUES_API = "https://api.prizepicks.com/leagues"

# ── Golf league defaults ─────────────────────────────────────────────────────
GOLF_LEAGUE_ID = 9  # PGA default, auto-discovered at runtime

# ── File paths ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_FILE = _PROJECT_ROOT / ".pp_golf_cache.json"
_DB_PATH = _PROJECT_ROOT / "data" / "golf_quant.db"

# ── Chrome 120 browser fingerprint headers ────────────────────────────────────
_CHROME120_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://app.prizepicks.com/",
    "Origin": "https://app.prizepicks.com",
    "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "Connection": "keep-alive",
    "DNT": "1",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNAL:  Low-level HTTP request with 3-tier fallback
# ═══════════════════════════════════════════════════════════════════════════════

def _pp_request(
    per_page: int = 250,
    cookies_str: str = "",
    league_id: int = GOLF_LEAGUE_ID,
) -> tuple[Any, Optional[str]]:
    """
    Execute a single PrizePicks API request with 3-layer bypass fallback.

    Attempt 1: curl_cffi with Chrome 120 TLS impersonation
    Attempt 2: cloudscraper (Cloudflare JS challenge solver)
    Attempt 3: plain requests (last resort)

    Returns:
        (response_object, error_string_or_None)
    """
    params = {
        "league_id": league_id,
        "per_page": per_page,
        "single_stat": "true",
    }
    headers = dict(_CHROME120_HEADERS)
    if cookies_str:
        headers["Cookie"] = cookies_str

    # ── Attempt 1: curl_cffi (Chrome 120 TLS fingerprint) ─────────────────
    try:
        from curl_cffi import requests as cffi_requests
        log.debug("pp_request: trying curl_cffi impersonate=chrome120")
        resp = cffi_requests.get(
            PRIZEPICKS_API,
            params=params,
            headers=headers,
            impersonate="chrome120",
            timeout=20,
        )
        if resp.status_code == 200:
            log.info("pp_request: curl_cffi succeeded (200)")
            return resp, None
        log.warning(f"pp_request: curl_cffi returned HTTP {resp.status_code}")
    except ImportError:
        log.debug("pp_request: curl_cffi not installed, skipping")
    except Exception as exc:
        log.warning(f"pp_request: curl_cffi error: {exc}")

    # ── Attempt 2: cloudscraper (JS challenge bypass) ─────────────────────
    try:
        import cloudscraper
        log.debug("pp_request: trying cloudscraper")
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "darwin", "desktop": True}
        )
        scraper.headers.update(headers)
        resp = scraper.get(PRIZEPICKS_API, params=params, timeout=20)
        if resp.status_code == 200:
            log.info("pp_request: cloudscraper succeeded (200)")
            return resp, None
        log.warning(f"pp_request: cloudscraper returned HTTP {resp.status_code}")
    except ImportError:
        log.debug("pp_request: cloudscraper not installed, skipping")
    except Exception as exc:
        log.warning(f"pp_request: cloudscraper error: {exc}")

    # ── Attempt 3: plain requests ─────────────────────────────────────────
    try:
        import requests
        log.debug("pp_request: trying plain requests")
        resp = requests.get(
            PRIZEPICKS_API,
            params=params,
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 200:
            log.info("pp_request: plain requests succeeded (200)")
            return resp, None
        return resp, f"HTTP {resp.status_code}"
    except Exception as exc:
        log.error(f"pp_request: all 3 methods failed. Last error: {exc}")
        return None, f"All request methods failed: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNAL:  Parse PrizePicks JSON:API response
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_pp_response(data: dict) -> list[dict]:
    """
    Parse the PrizePicks JSON:API envelope into a flat list of line dicts.

    PrizePicks format:
      data["data"]     -> list of projection objects
      data["included"] -> related resources (players, leagues) keyed by type+id

    Returns:
        List of dicts with keys:
          player, stat_type, stat_display, line, pp_id, start_time, is_promo
    """
    if not data or not isinstance(data, dict):
        return []

    included = data.get("included", [])

    # Build player lookup: id -> display_name
    player_map: dict[str, str] = {}
    for obj in included:
        if obj.get("type") == "new_player":
            attrs = obj.get("attributes", {})
            display_name = (
                attrs.get("display_name")
                or attrs.get("name")
                or "Unknown"
            )
            player_map[obj["id"]] = display_name

    rows: list[dict] = []
    for item in data.get("data", []):
        if item.get("type") != "projection":
            continue

        attrs = item.get("attributes", {})
        rels = item.get("relationships", {})

        # Resolve player name from included resources
        player_rel = rels.get("new_player", {}).get("data", {})
        player_id = player_rel.get("id", "")
        player_name = player_map.get(
            player_id,
            attrs.get("description", "Unknown"),
        )

        line_score = attrs.get("line_score")
        if line_score is None:
            continue

        # Parse start_time safely
        start_time_raw = attrs.get("start_time", "")
        start_time_iso = ""
        if start_time_raw:
            try:
                dt = datetime.fromisoformat(
                    start_time_raw.replace("Z", "+00:00")
                )
                start_time_iso = dt.isoformat()
            except (ValueError, TypeError):
                start_time_iso = start_time_raw

        stat_display = attrs.get("stat_type", "")
        is_promo = bool(attrs.get("is_promo", False))

        rows.append({
            "player": player_name,
            "stat_type": stat_display.lower().replace(" ", "_"),
            "stat_display": stat_display,
            "line": float(line_score),
            "pp_id": str(item.get("id", "")),
            "start_time": start_time_iso,
            "is_promo": is_promo,
        })

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNAL:  Single fetch with 429 retry
# ═══════════════════════════════════════════════════════════════════════════════

def _pp_fetch_one(
    per_page: int = 250,
    cookies_str: str = "",
    league_id: int = GOLF_LEAGUE_ID,
    max_retries: int = 3,
) -> tuple[list[dict], Optional[str]]:
    """
    Fetch one page of PrizePicks lines with automatic retry on 429 (rate limit).

    Returns:
        (rows_list, error_string_or_None)
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        resp, err = _pp_request(
            per_page=per_page,
            cookies_str=cookies_str,
            league_id=league_id,
        )

        if err and resp is None:
            last_err = err
            log.warning(f"_pp_fetch_one: attempt {attempt}/{max_retries} "
                        f"failed entirely: {err}")
            time.sleep(2 * attempt)
            continue

        if resp is not None and hasattr(resp, "status_code"):
            if resp.status_code == 429:
                backoff = 10 * attempt
                log.warning(f"_pp_fetch_one: 429 rate limited, "
                            f"backing off {backoff}s (attempt {attempt})")
                time.sleep(backoff)
                last_err = "HTTP 429 rate limited"
                continue

            if resp.status_code == 403:
                last_err = "HTTP 403 Forbidden (PerimeterX block)"
                log.warning(f"_pp_fetch_one: 403 blocked (attempt {attempt})")
                time.sleep(5 * attempt)
                continue

            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}"
                log.warning(f"_pp_fetch_one: HTTP {resp.status_code} "
                            f"(attempt {attempt})")
                time.sleep(2 * attempt)
                continue

        # Success path — parse JSON
        try:
            data = resp.json()
        except Exception as exc:
            last_err = f"JSON decode error: {exc}"
            log.warning(f"_pp_fetch_one: {last_err}")
            continue

        rows = _parse_pp_response(data)
        if rows:
            log.info(f"_pp_fetch_one: parsed {len(rows)} lines")
            return rows, None
        else:
            log.info("_pp_fetch_one: response parsed OK but 0 lines "
                     "(slate may be empty)")
            return [], None

    return [], last_err


# ═══════════════════════════════════════════════════════════════════════════════
#  DISK CACHE:  JSON file fallback
# ═══════════════════════════════════════════════════════════════════════════════

def _save_pp_disk_cache(rows: list[dict]) -> None:
    """Persist rows to the JSON disk cache for offline fallback."""
    try:
        payload = {
            "ts": time.time(),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "count": len(rows),
            "rows": rows,
        }
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "w") as f:
            json.dump(payload, f, indent=2)
        log.debug(f"Disk cache saved: {len(rows)} rows -> {_CACHE_FILE}")
    except Exception as exc:
        log.warning(f"Failed to write disk cache: {exc}")


def _load_pp_disk_cache(max_age_sec: int = 3600) -> tuple[list[dict], Optional[str]]:
    """
    Load rows from disk cache if the file is fresher than max_age_sec.

    Returns:
        (rows_list, error_string_or_None)
    """
    if not _CACHE_FILE.exists():
        return [], "No disk cache file"

    try:
        with open(_CACHE_FILE, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        return [], f"Disk cache read error: {exc}"

    cached_ts = payload.get("ts", 0)
    age = time.time() - cached_ts
    if age > max_age_sec:
        return [], f"Disk cache too old ({int(age)}s > {max_age_sec}s)"

    rows = payload.get("rows", [])
    if not rows:
        return [], "Disk cache is empty"

    log.info(f"Loaded {len(rows)} lines from disk cache (age={int(age)}s)")
    return rows, None


# ═══════════════════════════════════════════════════════════════════════════════
#  SCRAPER DB FALLBACK:  Read from SQLite prizepicks_lines table
# ═══════════════════════════════════════════════════════════════════════════════

def _load_pp_from_scraper_db(
    max_age_sec: int = 7200,
) -> tuple[list[dict], Optional[str]]:
    """
    Read the latest batch of PrizePicks lines from the scraper service's
    SQLite database (golf_quant.db -> prizepicks_lines table).

    Only returns rows where is_latest=1 and fetched_at is within max_age_sec.

    Returns:
        (rows_list, error_string_or_None)
    """
    db_path = str(_DB_PATH)
    if not os.path.exists(db_path):
        return [], f"Scraper DB not found: {db_path}"

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Check if the table exists
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='prizepicks_lines'"
        )
        if not cur.fetchone():
            conn.close()
            return [], "prizepicks_lines table does not exist"

        # Fetch latest batch
        cur.execute(
            "SELECT pp_id, player_name, stat_type, stat_display, "
            "       line_score, is_promo, start_time, fetched_at "
            "FROM prizepicks_lines "
            "WHERE is_latest = 1 "
            "ORDER BY player_name, stat_type"
        )
        db_rows = cur.fetchall()
        conn.close()

        if not db_rows:
            return [], "No is_latest=1 rows in prizepicks_lines"

        # Check staleness of the batch using the first row's fetched_at
        fetched_at_str = db_rows[0]["fetched_at"]
        if fetched_at_str:
            try:
                fetched_dt = datetime.fromisoformat(fetched_at_str)
                age = (datetime.utcnow() - fetched_dt).total_seconds()
                if age > max_age_sec:
                    return [], (f"Scraper DB data too old "
                                f"({int(age)}s > {max_age_sec}s)")
            except (ValueError, TypeError):
                pass  # Can't parse — use the data anyway

        rows = []
        for r in db_rows:
            rows.append({
                "player": r["player_name"],
                "stat_type": r["stat_type"] or "",
                "stat_display": r["stat_display"] or "",
                "line": float(r["line_score"]),
                "pp_id": r["pp_id"] or "",
                "start_time": r["start_time"] or "",
                "is_promo": bool(r["is_promo"]),
            })

        log.info(f"Loaded {len(rows)} lines from scraper DB")
        return rows, None

    except Exception as exc:
        return [], f"Scraper DB error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO-DISCOVER:  Find the golf/PGA league ID dynamically
# ═══════════════════════════════════════════════════════════════════════════════

def _discover_golf_league_id(cookies_str: str = "") -> int:
    """
    Query the PrizePicks /leagues endpoint and find the active PGA/Golf
    league ID. Falls back to GOLF_LEAGUE_ID (9) on any failure.
    """
    headers = dict(_CHROME120_HEADERS)
    if cookies_str:
        headers["Cookie"] = cookies_str

    # Try curl_cffi first, then plain requests
    data = None

    try:
        from curl_cffi import requests as cffi_requests
        resp = cffi_requests.get(
            PRIZEPICKS_LEAGUES_API,
            headers=headers,
            impersonate="chrome120",
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
    except Exception:
        pass

    if data is None:
        try:
            import requests
            resp = requests.get(
                PRIZEPICKS_LEAGUES_API,
                headers=headers,
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
        except Exception:
            pass

    if not data:
        log.warning("_discover_golf_league_id: could not fetch leagues, "
                     f"defaulting to {GOLF_LEAGUE_ID}")
        return GOLF_LEAGUE_ID

    # Search for PGA / Golf in the leagues list
    golf_keywords = {"PGA", "GOLF", "PGA TOUR"}
    for league in data.get("data", []):
        attrs = league.get("attributes", {})
        name = (attrs.get("name") or "").upper()
        active = attrs.get("active", False)
        if active and any(kw in name for kw in golf_keywords):
            found_id = int(league.get("id", GOLF_LEAGUE_ID))
            log.info(f"Auto-discovered golf league: "
                     f"{attrs.get('name')} (id={found_id})")
            return found_id

    log.warning(f"No active golf league found in /leagues response, "
                f"defaulting to {GOLF_LEAGUE_ID}")
    return GOLF_LEAGUE_ID


# ═══════════════════════════════════════════════════════════════════════════════
#  RELAY:  Fetch via remote relay server on residential IP
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_via_relay(
    relay_url: str,
    league_id: int = GOLF_LEAGUE_ID,
    per_page: int = 250,
    timeout: int = 30,
) -> tuple[list[dict], Optional[str]]:
    """
    Fetch PrizePicks lines through a relay server (pp_relay.py) running
    on a residential IP to bypass datacenter IP blocks.

    The relay is expected to accept GET /fetch?league_id=N&per_page=N
    and return the raw PrizePicks JSON:API response.

    Returns:
        (rows_list, error_string_or_None)
    """
    if not relay_url:
        return [], "No relay URL configured"

    url = relay_url.rstrip("/")
    if not url.endswith("/fetch"):
        url += "/fetch"

    try:
        import requests
        resp = requests.get(
            url,
            params={"league_id": league_id, "per_page": per_page},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return [], f"Relay returned HTTP {resp.status_code}"

        data = resp.json()

        # Relay may wrap in its own envelope or pass through raw
        if "data" in data and isinstance(data["data"], list):
            # Raw PrizePicks JSON:API passthrough
            rows = _parse_pp_response(data)
        elif "rows" in data:
            # Relay pre-parsed format
            rows = data["rows"]
        else:
            return [], "Relay response format unrecognized"

        if rows:
            log.info(f"Relay returned {len(rows)} lines")
        return rows, None

    except Exception as exc:
        return [], f"Relay error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_golf_prizepicks_lines(
    cookies_str: str = "",
    relay_url: str = "",
    per_page: int = 250,
    cache_max_age: int = 3600,
    db_max_age: int = 7200,
) -> tuple[list[dict], Optional[str]]:
    """
    Fetch live PrizePicks golf projection lines using a multi-step
    fallback chain designed to survive PerimeterX bot detection.

    Fallback order:
      Step 1: Relay server (residential IP bypass)
      Step 2: Scraper DB (SQLite prizepicks_lines table)
      Step 3: Disk cache (JSON file)
      Step 4: Direct fetch (curl_cffi -> cloudscraper -> requests)
      Step 5: Return empty with accumulated error

    Args:
        cookies_str:   Optional browser cookies for authenticated sessions
        relay_url:     Optional URL of pp_relay.py server
        per_page:      Number of projections to request per page
        cache_max_age: Maximum age of disk cache in seconds
        db_max_age:    Maximum age of scraper DB data in seconds

    Returns:
        (rows_list, error_string_or_None)
        rows_list is a list of dicts with keys:
          player, stat_type, stat_display, line, pp_id, start_time, is_promo
    """
    errors: list[str] = []

    # Auto-discover the golf league ID
    league_id = _discover_golf_league_id(cookies_str)

    # ── Step 1: Relay server ──────────────────────────────────────────────
    if relay_url:
        log.info("Step 1: Trying relay server...")
        rows, err = _fetch_via_relay(
            relay_url, league_id=league_id, per_page=per_page,
        )
        if rows:
            _save_pp_disk_cache(rows)
            return rows, None
        if err:
            errors.append(f"Relay: {err}")
            log.warning(f"Step 1 failed: {err}")
    else:
        log.debug("Step 1: No relay URL, skipping")

    # ── Step 2: Scraper DB ────────────────────────────────────────────────
    log.info("Step 2: Checking scraper DB...")
    rows, err = _load_pp_from_scraper_db(max_age_sec=db_max_age)
    if rows:
        return rows, None
    if err:
        errors.append(f"DB: {err}")
        log.debug(f"Step 2: {err}")

    # ── Step 3: Disk cache ────────────────────────────────────────────────
    log.info("Step 3: Checking disk cache...")
    rows, err = _load_pp_disk_cache(max_age_sec=cache_max_age)
    if rows:
        return rows, None
    if err:
        errors.append(f"Cache: {err}")
        log.debug(f"Step 3: {err}")

    # ── Step 4: Direct fetch (curl_cffi / cloudscraper / requests) ────────
    log.info("Step 4: Direct API fetch (curl_cffi -> cloudscraper -> requests)...")
    rows, err = _pp_fetch_one(
        per_page=per_page,
        cookies_str=cookies_str,
        league_id=league_id,
    )
    if rows:
        _save_pp_disk_cache(rows)
        return rows, None
    if err:
        errors.append(f"Direct: {err}")
        log.warning(f"Step 4 failed: {err}")

    # ── Step 5: All steps exhausted ───────────────────────────────────────
    combined_err = " | ".join(errors) if errors else "All fetch methods failed"
    log.error(f"fetch_golf_prizepicks_lines: {combined_err}")
    return [], combined_err


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND AUTO-FETCHER (singleton daemon thread)
# ═══════════════════════════════════════════════════════════════════════════════

_auto_state: dict[str, Any] = {
    "enabled": False,
    "interval_sec": 300,
    "cookies": "",
    "relay_url": "",
    "thread": None,
    "lock": threading.Lock(),
    "last_rows": [],
    "last_ts": 0.0,
    "last_err": None,
    "stop_event": threading.Event(),
}


def _pp_auto_loop(state: dict) -> None:
    """
    Daemon loop that periodically fetches PrizePicks lines in the background.
    Controlled via the module-level _auto_state dict.
    """
    log.info("PrizePicks auto-fetch thread started")

    while not state["stop_event"].is_set():
        with state["lock"]:
            if not state["enabled"]:
                break
            cookies = state["cookies"]
            relay_url = state["relay_url"]
            interval = state["interval_sec"]

        try:
            rows, err = fetch_golf_prizepicks_lines(
                cookies_str=cookies,
                relay_url=relay_url,
            )
            with state["lock"]:
                state["last_rows"] = rows
                state["last_ts"] = time.time()
                state["last_err"] = err
            if rows:
                log.info(f"Auto-fetch: {len(rows)} lines updated")
            elif err:
                log.warning(f"Auto-fetch: {err}")
        except Exception as exc:
            with state["lock"]:
                state["last_err"] = str(exc)
            log.error(f"Auto-fetch exception: {exc}", exc_info=True)

        # Sleep in small increments to allow quick shutdown
        for _ in range(int(interval)):
            if state["stop_event"].is_set():
                break
            time.sleep(1)

    log.info("PrizePicks auto-fetch thread stopped")


def _ensure_pp_auto_thread() -> None:
    """Start the auto-fetch daemon thread if it's not already running."""
    state = _auto_state
    with state["lock"]:
        thread = state["thread"]
        if thread is not None and thread.is_alive():
            return  # Already running
        state["stop_event"].clear()
        t = threading.Thread(
            target=_pp_auto_loop,
            args=(state,),
            daemon=True,
            name="pp-golf-auto-fetch",
        )
        t.start()
        state["thread"] = t
        log.info("Auto-fetch daemon thread launched")


def set_pp_auto_fetch(
    enabled: bool = True,
    interval_sec: int = 300,
    cookies: str = "",
    relay_url: str = "",
) -> None:
    """
    Enable or disable the background auto-fetch thread.

    Args:
        enabled:      True to start, False to stop
        interval_sec: Seconds between fetches (default 300 = 5 min)
        cookies:      Browser cookies string for authenticated sessions
        relay_url:    URL of the relay server
    """
    state = _auto_state
    with state["lock"]:
        state["enabled"] = enabled
        state["interval_sec"] = max(30, interval_sec)  # Floor at 30s
        state["cookies"] = cookies
        state["relay_url"] = relay_url

    if enabled:
        _ensure_pp_auto_thread()
        log.info(f"Auto-fetch ENABLED (interval={interval_sec}s)")
    else:
        state["stop_event"].set()
        thread = state.get("thread")
        if thread and thread.is_alive():
            thread.join(timeout=5)
        with state["lock"]:
            state["thread"] = None
        log.info("Auto-fetch DISABLED")


def get_pp_auto_lines() -> tuple[list[dict], float, Optional[str]]:
    """
    Retrieve the most recent auto-fetched lines without triggering a new fetch.

    Returns:
        (rows, age_seconds, error_or_None)
        - rows: list of line dicts from the last successful fetch
        - age_seconds: seconds since last fetch (0.0 if never fetched)
        - error_or_None: last error string, or None if last fetch succeeded
    """
    state = _auto_state
    with state["lock"]:
        rows = list(state["last_rows"])
        ts = state["last_ts"]
        err = state["last_err"]

    age = (time.time() - ts) if ts > 0 else 0.0
    return rows, age, err


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    print("=" * 60)
    print("PrizePicks Golf Lines — PerimeterX Bypass Fetcher")
    print("=" * 60)

    rows, err = fetch_golf_prizepicks_lines()

    if err:
        print(f"\nError: {err}")

    if rows:
        print(f"\nFetched {len(rows)} lines:\n")
        for r in rows[:20]:
            promo_tag = " [PROMO]" if r.get("is_promo") else ""
            print(
                f"  {r['player']:<25s}  "
                f"{r['stat_display']:<22s}  "
                f"Line: {r['line']:>6.1f}"
                f"{promo_tag}"
            )
        if len(rows) > 20:
            print(f"  ... and {len(rows) - 20} more")
    else:
        print("\nNo lines available.")
