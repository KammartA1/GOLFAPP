"""
services/clv_system/odds_ingestion.py
======================================
Continuous odds capture from multiple sources for CLV tracking.

Ingests lines from:
  - PrizePicks (golf props: birdies, GIR, fantasy score, etc.)
  - The Odds API (golf tournament markets: outrights, h2h, totals)

Stores EVERY line movement with millisecond-precision timestamps in the
clv_line_movements table.  Runs on a 15-minute schedule for Golf.

This module wraps the existing workers/odds_worker.py fetch functions and
pipes the results into the CLV-specific line_movements table for
high-resolution tracking separate from the main app's line_movements table.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from quant_system.db.schema import get_engine, get_session
from services.clv_system.models import CLVLineMovement, Base

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY_GOLF = "golf_pga_championship"
REGION_US = "us"

PRIZEPICKS_API = "https://api.prizepicks.com/projections"
GOLF_LEAGUES = {"PGA", "PGA TOUR", "GOLF", "PGA TOUR CHAMPIONS", "LIV GOLF", "LPGA"}

# Golf-specific Odds API market keys (tournament-level markets)
ODDS_API_MARKET_KEYS = [
    "outrights",
    "h2h",
    "totals",
]

# Golf sport keys to auto-discover from The Odds API
GOLF_SPORT_KEYS = [
    "golf_pga_championship",
    "golf_us_open",
    "golf_the_open_championship",
    "golf_pga_tour",
    "golf_masters_tournament",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

_PP_STAT_MAP: Dict[str, str] = {
    # Birdies
    "Birdies": "Birdies", "birdies": "Birdies", "Birdies Made": "Birdies",
    "Total Birdies": "Birdies",
    # Bogeys
    "Bogeys": "Bogeys", "bogeys": "Bogeys", "Bogeys Made": "Bogeys",
    "Total Bogeys": "Bogeys", "Bogeys+": "Bogeys",
    # Fantasy Score
    "Fantasy Score": "Fantasy Score", "fantasy score": "Fantasy Score",
    "Fantasy Points": "Fantasy Score", "Fpts": "Fantasy Score",
    "FPTS": "Fantasy Score",
    # GIR (Greens in Regulation)
    "Greens in Regulation": "GIR", "GIR": "GIR", "gir": "GIR",
    "Greens In Regulation": "GIR", "GIRs": "GIR",
    # Strokes
    "Strokes": "Strokes", "strokes": "Strokes", "Total Strokes": "Strokes",
    "Score": "Strokes", "Round Score": "Strokes",
    # Pars
    "Pars": "Pars", "pars": "Pars", "Pars Made": "Pars",
    "Total Pars": "Pars",
    # Eagles
    "Eagles": "Eagles", "eagles": "Eagles", "Eagles Made": "Eagles",
    # Fairways Hit
    "Fairways Hit": "Fairways Hit", "FIR": "Fairways Hit",
    "Fairways in Regulation": "Fairways Hit", "Fairways": "Fairways Hit",
    # Putts
    "Putts": "Putts", "putts": "Putts", "Total Putts": "Putts",
    # Hole-in-one / Aces
    "Aces": "Aces", "Hole in One": "Aces", "Hole-in-One": "Aces",
    # Tournament finish
    "Finish Position": "Finish Position", "Position": "Finish Position",
    # Birdie-free holes / streak stats
    "Longest Drive": "Longest Drive",
    "Driving Distance": "Driving Distance",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _odds_api_key() -> str:
    return os.environ.get("ODDS_API_KEY", "")


def _normalize_stat(raw: str) -> str:
    s = str(raw).strip()
    for k, v in _PP_STAT_MAP.items():
        if k.lower() == s.lower():
            return v
    return s


def _http_get(url: str, params: Optional[dict] = None,
              headers: Optional[dict] = None, timeout: int = 20,
              retries: int = 3) -> Tuple[Optional[Any], Optional[str]]:
    """GET with retry + backoff."""
    hdrs = dict(_HEADERS)
    if headers:
        hdrs.update(headers)
    for attempt in range(retries):
        try:
            try:
                from curl_cffi import requests as cffi_req
                r = cffi_req.get(url, params=params, headers=hdrs,
                                 impersonate="chrome120", timeout=timeout)
                if r.status_code not in (403, 429):
                    r.raise_for_status()
                    return r.json(), None
                if r.status_code == 429:
                    wait = 8 * (attempt + 1)
                    log.warning("Rate limited %s, waiting %ds", url, wait)
                    time.sleep(wait)
                    continue
            except ImportError:
                pass
            except Exception:
                pass

            r = requests.get(url, params=params, headers=hdrs, timeout=timeout)
            if r.status_code == 429:
                time.sleep(8 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json(), None
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return None, f"{type(exc).__name__}: {exc}"
    return None, "All retries exhausted"


def _american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds == 0:
        return 0.5
    elif odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


class CLVOddsIngestion:
    """Continuous odds capture engine for CLV tracking (Golf).

    Ingests from all available golf sources, stores every line movement with
    millisecond timestamps in the clv_line_movements table.

    Schedule: every 15 minutes.
    """

    def __init__(self, sport: str = "GOLF", db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create CLV tables if they don't exist."""
        engine = get_engine(self._db_path)
        Base.metadata.create_all(engine)

    def _session(self):
        return get_session(self._db_path)

    # ── Master ingest method ──────────────────────────────────────────

    def ingest_all(self) -> Dict[str, Any]:
        """Fetch odds from all sources and store in clv_line_movements.

        Returns summary dict with counts per source and any errors.
        """
        results: Dict[str, Any] = {
            "ok": True,
            "timestamp": _utcnow().isoformat(),
            "source_counts": {},
            "total_lines": 0,
            "errors": [],
        }

        # 1. PrizePicks (golf props)
        try:
            pp_count = self._ingest_prizepicks()
            results["source_counts"]["prizepicks"] = pp_count
            results["total_lines"] += pp_count
            log.info("CLV ingestion — PrizePicks Golf: %d lines", pp_count)
        except Exception as exc:
            results["errors"].append(f"PrizePicks: {exc}")
            log.warning("CLV PrizePicks failed: %s", exc)

        # 2. The Odds API (golf tournament markets)
        try:
            oa_count = self._ingest_odds_api()
            results["source_counts"]["odds_api"] = oa_count
            results["total_lines"] += oa_count
            log.info("CLV ingestion — Odds API Golf: %d lines", oa_count)
        except Exception as exc:
            results["errors"].append(f"Odds API: {exc}")
            log.warning("CLV Odds API failed: %s", exc)

        if results["total_lines"] == 0:
            results["ok"] = False

        return results

    # ── PrizePicks (Golf) ─────────────────────────────────────────────

    def _ingest_prizepicks(self) -> int:
        """Fetch and store PrizePicks golf lines."""
        rows = self._fetch_prizepicks_raw()
        return self._store_rows(rows, book="prizepicks")

    def _fetch_prizepicks_raw(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen = set()
        for single_stat in ("true", "false"):
            params = {
                "per_page": "500",
                "single_stat": single_stat,
                "in_play": "false",
            }
            pp_headers = dict(_HEADERS)
            pp_headers["Referer"] = "https://app.prizepicks.com/"
            pp_headers["Origin"] = "https://app.prizepicks.com"

            data, err = _http_get(PRIZEPICKS_API, params=params, headers=pp_headers)
            if err or data is None:
                log.warning("PrizePicks fetch error: %s", err)
                continue

            included = {
                item["id"]: item
                for item in data.get("included", [])
                if isinstance(item, dict) and "id" in item
            }

            for proj in data.get("data", []):
                if not isinstance(proj, dict):
                    continue
                attrs = proj.get("attributes", {}) or {}
                league = str(attrs.get("league", "") or "").upper().strip()
                # Filter for golf leagues only
                if not league or league not in GOLF_LEAGUES:
                    continue

                stat_type = attrs.get("stat_type", "")
                line_score = attrs.get("line_score")
                if not stat_type or line_score is None:
                    continue

                rels = proj.get("relationships", {}) or {}
                player_id = (rels.get("new_player", {}).get("data", {}) or {}).get("id")
                if not player_id:
                    player_id = (rels.get("player", {}).get("data", {}) or {}).get("id")
                player_attrs = (
                    included.get(player_id, {}).get("attributes", {})
                    if player_id else {}
                )
                player_name = (
                    player_attrs.get("name", "")
                    or player_attrs.get("display_name", "")
                    or attrs.get("name", "")
                    or attrs.get("description", "")
                )
                if not player_name:
                    continue

                key = (player_name.lower(), _normalize_stat(stat_type))
                if key in seen:
                    continue
                seen.add(key)

                start_time = attrs.get("start_time", "")
                try:
                    rows.append({
                        "player": player_name,
                        "market_type": _normalize_stat(stat_type),
                        "line": float(line_score),
                        "event_id": start_time,
                        "odds_american": -110,
                    })
                except (TypeError, ValueError):
                    pass

            time.sleep(2)

        return rows

    # ── The Odds API (Golf) ──────────────────────────────────────────

    def _ingest_odds_api(self) -> int:
        rows = self._fetch_odds_api_raw()
        return self._store_rows_bulk(rows)

    def _discover_active_golf_keys(self) -> List[str]:
        """Auto-discover which golf sport keys are currently active on The Odds API."""
        key = _odds_api_key()
        if not key:
            return []

        active_keys = []
        sports_data, err = _http_get(
            f"{ODDS_BASE}/sports",
            params={"apiKey": key},
        )
        if err or not isinstance(sports_data, list):
            log.warning("Odds API sports discovery error: %s", err)
            return GOLF_SPORT_KEYS[:1]  # Fallback to default

        for sport in sports_data:
            sport_key = sport.get("key", "")
            if sport_key.startswith("golf") and sport.get("active", False):
                active_keys.append(sport_key)

        if not active_keys:
            # Fallback: try the known golf keys
            for candidate in GOLF_SPORT_KEYS:
                for sport in sports_data:
                    if sport.get("key") == candidate:
                        active_keys.append(candidate)
                        break

        return active_keys if active_keys else [SPORT_KEY_GOLF]

    def _fetch_odds_api_raw(self) -> List[Dict[str, Any]]:
        key = _odds_api_key()
        if not key:
            log.info("No ODDS_API_KEY — skipping The Odds API")
            return []

        rows: List[Dict[str, Any]] = []

        # Auto-discover active golf keys, falling back to default
        active_keys = self._discover_active_golf_keys()
        log.info("Active golf sport keys: %s", active_keys)

        for sport_key in active_keys:
            # Fetch events for this golf sport key
            events_data, err = _http_get(
                f"{ODDS_BASE}/sports/{sport_key}/events",
                params={"apiKey": key},
            )
            if err or not isinstance(events_data, list):
                log.warning("Odds API events error for %s: %s", sport_key, err)
                continue

            market_str = ",".join(ODDS_API_MARKET_KEYS)
            for ev in events_data:
                eid = ev.get("id", "")
                if not eid:
                    continue
                event_name = (
                    ev.get("description", "")
                    or ev.get("home_team", "")
                    or f"{ev.get('away_team', '')} vs {ev.get('home_team', '')}"
                )
                commence_time = ev.get("commence_time", "")

                # Fetch odds for each event
                odds_data, err2 = _http_get(
                    f"{ODDS_BASE}/sports/{sport_key}/events/{eid}/odds",
                    params={
                        "apiKey": key,
                        "regions": REGION_US,
                        "markets": market_str,
                        "oddsFormat": "decimal",
                        "dateFormat": "iso",
                    },
                )
                if err2 or not isinstance(odds_data, dict):
                    # Try fetching odds at the sport level instead
                    odds_data, err2 = _http_get(
                        f"{ODDS_BASE}/sports/{sport_key}/odds",
                        params={
                            "apiKey": key,
                            "regions": REGION_US,
                            "markets": market_str,
                            "oddsFormat": "decimal",
                            "dateFormat": "iso",
                        },
                    )
                    if err2 or odds_data is None:
                        continue
                    # Sport-level odds returns a list of events
                    if isinstance(odds_data, list):
                        for ev_odds in odds_data:
                            rows.extend(
                                self._parse_odds_api_event(ev_odds, sport_key)
                            )
                        continue
                    if not isinstance(odds_data, dict):
                        continue

                rows.extend(self._parse_odds_api_event(odds_data, sport_key))
                time.sleep(0.5)

            time.sleep(1)

        return rows

    def _parse_odds_api_event(
        self, odds_data: Dict[str, Any], sport_key: str,
    ) -> List[Dict[str, Any]]:
        """Parse a single event's odds data from The Odds API into row dicts."""
        rows: List[Dict[str, Any]] = []
        event_name = (
            odds_data.get("description", "")
            or odds_data.get("home_team", "")
            or odds_data.get("id", "unknown")
        )
        commence_time = odds_data.get("commence_time", "")

        for bookmaker in odds_data.get("bookmakers", []):
            book_key = bookmaker.get("key", "unknown")
            for market_obj in bookmaker.get("markets", []):
                market_key = market_obj.get("key", "")
                for outcome in market_obj.get("outcomes", []):
                    name = outcome.get("name", "")
                    desc = outcome.get("description", "")
                    point = outcome.get("point")
                    price = outcome.get("price")

                    # For outrights, the player name is in 'name'
                    player_name = desc if desc else name
                    if not player_name:
                        continue

                    odds_dec = float(price) if price else None
                    odds_am = None
                    if odds_dec and odds_dec > 0:
                        if odds_dec >= 2.0:
                            odds_am = int(round((odds_dec - 1) * 100))
                        else:
                            odds_am = int(round(-100 / (odds_dec - 1)))

                    # For outrights, there's no point/line — use odds as the line
                    line_val = float(point) if point is not None else odds_dec

                    if line_val is not None:
                        rows.append({
                            "player": player_name,
                            "market_type": market_key,
                            "line": line_val,
                            "book": book_key,
                            "event_id": event_name,
                            "odds_american": odds_am,
                            "odds_decimal": odds_dec,
                        })

        return rows

    # ── Storage helpers ───────────────────────────────────────────────

    def _store_rows(self, rows: List[Dict[str, Any]], book: str) -> int:
        """Store a batch of line rows from a single book."""
        if not rows:
            return 0

        now = _utcnow()
        session = self._session()
        count = 0
        try:
            for row in rows:
                odds_am = row.get("odds_american")
                implied = _american_to_implied_prob(odds_am) if odds_am else None

                lm = CLVLineMovement(
                    sport=self.sport,
                    event_id=row.get("event_id", ""),
                    market_type=row.get("market_type", ""),
                    book=book,
                    player=row.get("player", ""),
                    line=row["line"],
                    odds_american=odds_am,
                    odds_decimal=row.get("odds_decimal"),
                    implied_prob=implied,
                    timestamp=now,
                )
                session.add(lm)
                count += 1
            session.commit()
        except Exception:
            session.rollback()
            log.exception("Failed to store CLV lines for book=%s", book)
            raise
        finally:
            session.close()

        return count

    def _store_rows_bulk(self, rows: List[Dict[str, Any]]) -> int:
        """Store a batch of line rows with per-row book info (Odds API)."""
        if not rows:
            return 0

        now = _utcnow()
        session = self._session()
        count = 0
        try:
            for row in rows:
                odds_am = row.get("odds_american")
                implied = _american_to_implied_prob(odds_am) if odds_am else None

                lm = CLVLineMovement(
                    sport=self.sport,
                    event_id=row.get("event_id", ""),
                    market_type=row.get("market_type", ""),
                    book=row.get("book", "odds_api"),
                    player=row.get("player", ""),
                    line=row["line"],
                    odds_american=odds_am,
                    odds_decimal=row.get("odds_decimal"),
                    implied_prob=implied,
                    timestamp=now,
                )
                session.add(lm)
                count += 1
            session.commit()
        except Exception:
            session.rollback()
            log.exception("Failed to store CLV Odds API lines")
            raise
        finally:
            session.close()

        return count

    # ── Query helpers (used by other CLV modules) ─────────────────────

    def get_latest_lines(self, player: str, market_type: str,
                         limit: int = 20) -> List[Dict[str, Any]]:
        """Get the most recent line observations for a player/market."""
        session = self._session()
        try:
            rows = (
                session.query(CLVLineMovement)
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.player == player,
                    CLVLineMovement.market_type == market_type,
                )
                .order_by(CLVLineMovement.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in rows]
        finally:
            session.close()

    def get_line_history(self, player: str, market_type: str,
                         hours: int = 24) -> List[Dict[str, Any]]:
        """Get line history for a player/market over the last N hours."""
        from datetime import timedelta
        cutoff = _utcnow() - timedelta(hours=hours)
        session = self._session()
        try:
            rows = (
                session.query(CLVLineMovement)
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.player == player,
                    CLVLineMovement.market_type == market_type,
                    CLVLineMovement.timestamp >= cutoff,
                )
                .order_by(CLVLineMovement.timestamp.asc())
                .all()
            )
            return [r.to_dict() for r in rows]
        finally:
            session.close()

    def get_all_current_lines(self) -> List[Dict[str, Any]]:
        """Get the most recent line for each player/market/book combination."""
        from sqlalchemy import func as sa_func

        session = self._session()
        try:
            subq = (
                session.query(
                    CLVLineMovement.player,
                    CLVLineMovement.market_type,
                    CLVLineMovement.book,
                    sa_func.max(CLVLineMovement.timestamp).label("max_ts"),
                )
                .filter(CLVLineMovement.sport == self.sport)
                .group_by(
                    CLVLineMovement.player,
                    CLVLineMovement.market_type,
                    CLVLineMovement.book,
                )
                .subquery()
            )

            rows = (
                session.query(CLVLineMovement)
                .join(
                    subq,
                    (CLVLineMovement.player == subq.c.player)
                    & (CLVLineMovement.market_type == subq.c.market_type)
                    & (CLVLineMovement.book == subq.c.book)
                    & (CLVLineMovement.timestamp == subq.c.max_ts),
                )
                .all()
            )
            return [r.to_dict() for r in rows]
        finally:
            session.close()
