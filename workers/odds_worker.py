"""
Golf Quant Engine -- Odds Worker
=================================
Fetches and stores odds from multiple sources on a 15-minute schedule:
  - PrizePicks (golf props: birdies, strokes, etc.)
  - The Odds API (tournament outright, matchups, props)
  - DraftKings / FanDuel (via Odds API bookmaker data)

All lines are normalised into the ``line_movements`` table for downstream
signal generation and CLV tracking.

Run standalone:
    python -m workers.odds_worker
"""
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root on sys.path for standalone execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workers.base import BaseWorker
from database.connection import get_session_factory, init_db
from database.models import LineMovement, Player, Event

log = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    """Lowercase, strip suffixes, collapse whitespace."""
    if not name:
        return ""
    n = name.strip()
    n = re.sub(r"\s+(Jr\.|Sr\.|III|IV|II)\s*$", "", n, flags=re.IGNORECASE)
    n = re.sub(r"\s+", " ", n).strip().lower()
    return n


class OddsWorker(BaseWorker):
    name = "odds_worker"
    interval_seconds = int(os.environ.get("ODDS_WORKER_INTERVAL", 900))  # 15 min
    max_retries = 3
    retry_delay = 20.0
    description = "Fetches odds from PrizePicks, Odds API, and sportsbooks"

    # ------------------------------------------------------------------ #
    # Core execute
    # ------------------------------------------------------------------ #
    def execute(self) -> dict:
        init_db()
        factory = get_session_factory()

        total_lines = 0
        errors: list[str] = []

        # 1. Detect current tournament
        tournament_info = self._detect_tournament(factory)
        event_name = tournament_info.get("name", "")
        event_id = tournament_info.get("db_event_id")

        # 2. Fetch PrizePicks
        pp_count = 0
        try:
            pp_count = self._fetch_prizepicks(factory, event_name, event_id)
            total_lines += pp_count
        except Exception as exc:
            msg = f"PrizePicks fetch failed: {exc}"
            self._logger.error(msg, exc_info=True)
            errors.append(msg)

        # 3. Fetch Odds API
        odds_count = 0
        try:
            odds_count = self._fetch_odds_api(factory, event_name, event_id)
            total_lines += odds_count
        except Exception as exc:
            msg = f"Odds API fetch failed: {exc}"
            self._logger.error(msg, exc_info=True)
            errors.append(msg)

        # If ALL sources failed, raise so retry logic kicks in
        if total_lines == 0 and errors:
            raise RuntimeError(
                f"All odds sources failed: {'; '.join(errors)}"
            )

        return {
            "items_processed": total_lines,
            "prizepicks_lines": pp_count,
            "odds_api_lines": odds_count,
            "event": event_name,
            "errors": errors,
        }

    # ------------------------------------------------------------------ #
    # Tournament detection
    # ------------------------------------------------------------------ #
    def _detect_tournament(self, factory) -> dict:
        """Detect the current PGA tournament and ensure it exists in the events table."""
        from scrapers.tournament_detector import detect_current_tournament

        info = detect_current_tournament()
        if info.get("error"):
            self._logger.warning("Tournament detection: %s", info["error"])
            return {"name": "", "db_event_id": None}

        session = factory()
        try:
            espn_id = info.get("espn_id", "")
            event_row = session.query(Event).filter_by(
                event_name=info.get("name", ""),
                sport="GOLF",
            ).first()

            if event_row is None:
                event_row = Event(
                    sport="GOLF",
                    event_name=info.get("name", ""),
                    course_name=info.get("course_name", ""),
                    venue=info.get("course_name", ""),
                    start_time=_parse_dt(info.get("start_date")),
                    status=_map_status(info.get("status", "upcoming")),
                    metadata_json=json.dumps({
                        "espn_id": espn_id,
                        "par": info.get("par", 72),
                        "course_lat": info.get("course_lat", 0),
                        "course_lon": info.get("course_lon", 0),
                        "current_round": info.get("current_round", 0),
                    }),
                )
                session.add(event_row)
                session.commit()
                self._logger.info("Created event: %s (id=%d)", event_row.event_name, event_row.id)
            else:
                # Update status if changed
                new_status = _map_status(info.get("status", "upcoming"))
                if event_row.status != new_status:
                    old = event_row.status
                    event_row.status = new_status
                    meta = json.loads(event_row.metadata_json) if event_row.metadata_json else {}
                    meta["current_round"] = info.get("current_round", 0)
                    event_row.metadata_json = json.dumps(meta)
                    session.commit()
                    self._logger.info(
                        "Event [%s] status changed: %s -> %s",
                        event_row.event_name, old, new_status,
                    )

            # Upsert players from the field
            self._upsert_field_players(session, info.get("field", []))

            db_event_id = event_row.id
            session.commit()

            return {
                "name": info.get("name", ""),
                "db_event_id": db_event_id,
                "course_name": info.get("course_name", ""),
                "status": info.get("status", ""),
                "current_round": info.get("current_round", 0),
            }
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _upsert_field_players(self, session, field: list[dict]):
        """Ensure every player in the field exists in the players table."""
        for p in field:
            name = p.get("name", "")
            if not name or name == "Unknown":
                continue
            existing = session.query(Player).filter_by(name=name, sport="GOLF").first()
            if existing is None:
                player = Player(
                    name=name,
                    sport="GOLF",
                    active=True,
                    metadata_json=json.dumps({
                        "espn_id": p.get("espn_id", ""),
                    }),
                )
                session.add(player)
        session.flush()

    # ------------------------------------------------------------------ #
    # PrizePicks
    # ------------------------------------------------------------------ #
    def _fetch_prizepicks(self, factory, event_name: str, event_id: int | None) -> int:
        """Fetch PrizePicks golf lines and store as line_movements."""
        from scrapers.prizepicks_scraper import PrizePicksScraper

        scraper = PrizePicksScraper()
        result = scraper.fetch_golf_lines()

        if result.get("error") and not result.get("lines"):
            self._logger.warning("PrizePicks: %s", result["error"])
            return 0

        lines = result.get("lines", [])
        if not lines:
            return 0

        session = factory()
        count = 0
        try:
            for line in lines:
                player_name = line.get("player_name", "Unknown")
                stat_type = line.get("stat_type", "")
                line_value = line.get("line_value", 0)

                if not player_name or player_name == "Unknown" or not stat_type:
                    continue

                # PrizePicks lines are effectively -110/-110 => implied ~52.4%
                implied_prob = 0.524
                odds_dec = 1.0 / implied_prob if implied_prob > 0 else 2.0

                lm = LineMovement(
                    sport="GOLF",
                    event=event_name,
                    market=stat_type,
                    book="PrizePicks",
                    player=player_name,
                    line=line_value,
                    odds=odds_dec,
                    is_opening=False,
                    is_closing=False,
                    timestamp=datetime.utcnow(),
                )
                session.add(lm)
                count += 1

                # Ensure player exists
                self._ensure_player(session, player_name)

            session.commit()
            self._logger.info("PrizePicks: stored %d lines for %s", count, event_name)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return count

    # ------------------------------------------------------------------ #
    # Odds API
    # ------------------------------------------------------------------ #
    def _fetch_odds_api(self, factory, event_name: str, event_id: int | None) -> int:
        """Fetch odds from The Odds API and store as line_movements."""
        from config.settings import ODDS_API_KEY
        if not ODDS_API_KEY:
            self._logger.info("Odds API key not configured -- skipping")
            return 0

        from scrapers.odds_api_scraper import OddsAPIScraper

        scraper = OddsAPIScraper(api_key=ODDS_API_KEY)

        count = 0
        session = factory()
        try:
            # Fetch outright/winner odds
            for market_type in ["outrights", "h2h", "totals"]:
                try:
                    result = scraper.fetch_odds(markets=market_type)
                except Exception as exc:
                    self._logger.warning("Odds API market=%s failed: %s", market_type, exc)
                    continue

                if result.get("error") and not result.get("lines"):
                    self._logger.debug("Odds API market=%s: %s", market_type, result["error"])
                    continue

                for line_data in result.get("lines", []):
                    player_name = line_data.get("player_name", "Unknown")
                    bookmaker = line_data.get("bookmaker", "Unknown")
                    odds_american = line_data.get("odds_american", 0)
                    odds_decimal = line_data.get("odds_decimal", 2.0)
                    implied_prob = line_data.get("implied_prob", 0.0)

                    if not player_name or player_name == "Unknown":
                        continue

                    lm = LineMovement(
                        sport="GOLF",
                        event=event_name or line_data.get("event_name", ""),
                        market=line_data.get("market_type", market_type),
                        book=bookmaker,
                        player=player_name,
                        line=odds_american,
                        odds=odds_decimal,
                        is_opening=False,
                        is_closing=False,
                        timestamp=datetime.utcnow(),
                    )
                    session.add(lm)
                    count += 1

                    # Ensure player exists
                    self._ensure_player(session, player_name)

                # Log quota usage
                quota = result.get("quota", {})
                if quota:
                    self._logger.info(
                        "Odds API quota: remaining=%s, used=%s",
                        quota.get("remaining", "?"), quota.get("used", "?"),
                    )

            session.commit()
            self._logger.info("Odds API: stored %d lines for %s", count, event_name)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return count

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _ensure_player(self, session, player_name: str):
        """Create a Player record if one does not already exist."""
        existing = session.query(Player).filter_by(name=player_name, sport="GOLF").first()
        if existing is None:
            session.add(Player(name=player_name, sport="GOLF", active=True))
            session.flush()


def _parse_dt(val) -> datetime | None:
    if not val:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val).replace("Z", "+00:00").replace("+00:00", ""))
    except Exception:
        return None


def _map_status(status: str) -> str:
    mapping = {
        "upcoming": "scheduled",
        "in_progress": "live",
        "completed": "completed",
        "delayed": "scheduled",
        "suspended": "scheduled",
    }
    return mapping.get(status, "scheduled")


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    )
    import argparse
    parser = argparse.ArgumentParser(description="Golf Odds Worker")
    parser.add_argument("--loop", action="store_true", help="Run in loop mode")
    args = parser.parse_args()

    worker = OddsWorker()
    if args.loop:
        worker.run_loop()
    else:
        success = worker.run_once()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
