#!/usr/bin/env python3
"""
Golf PrizePicks Scraper Service
================================
Runs 24/7 as a systemd service. Pulls live golf lines from PrizePicks API
and stores them in the SQLite database.

Schedule:
  - Tournament weeks (Thu–Sun): every 30 minutes
  - Off-week (Mon–Wed):         every 2 hours

Usage:
    python -m services.prizepicks_scraper          # foreground
    systemctl start golf-prizepicks-scraper        # via systemd
"""
import sys
import os
import time
import signal
import logging
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import LOG_DIR
from data.scrapers.prizepicks import PrizePicksScraper
from data.storage.database import (
    get_session, init_db, PrizePicksLine, ScraperStatus, AuditLog,
)

# ── Logging ────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "prizepicks_scraper.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("golf_pp_scraper")

# ── Schedule constants ─────────────────────────────────────────────────────
TOURNAMENT_DAYS = {3, 4, 5, 6}   # Thursday=3, Friday=4, Saturday=5, Sunday=6
INTERVAL_TOURNAMENT = 30 * 60    # 30 minutes
INTERVAL_OFFWEEK    = 2 * 60 * 60  # 2 hours

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    log.info(f"Received signal {signum}, shutting down gracefully...")
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def is_tournament_day() -> bool:
    """Thursday (3) through Sunday (6) = tournament day."""
    return datetime.now().weekday() in TOURNAMENT_DAYS


def get_interval() -> int:
    """Return sleep interval in seconds based on day of week."""
    return INTERVAL_TOURNAMENT if is_tournament_day() else INTERVAL_OFFWEEK


def store_lines(projections: list) -> int:
    """Save fetched PrizePicks lines to the database."""
    session = get_session()
    try:
        # Mark all previous lines as not-latest
        session.query(PrizePicksLine).filter(
            PrizePicksLine.is_latest == True
        ).update({"is_latest": False})

        now = datetime.utcnow()
        count = 0
        for p in projections:
            line = PrizePicksLine(
                pp_id=p.pp_id,
                player_name=p.player_name,
                player_id=p.player_id,
                stat_type=p.stat_type,
                stat_display=p.stat_display,
                line_score=p.line_score,
                is_promo=p.is_promo,
                start_time=p.start_time,
                description=p.description,
                league=p.league,
                flash_sale_line=p.flash_sale_line_score,
                fetched_at=now,
                is_latest=True,
            )
            session.add(line)
            count += 1

        session.commit()
        return count
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()


def update_status(success: bool, lines_count: int = 0, error: str = None):
    """Update the scraper_status monitoring row."""
    session = get_session()
    try:
        status = session.query(ScraperStatus).filter_by(
            scraper_name="golf_prizepicks"
        ).first()
        if not status:
            status = ScraperStatus(scraper_name="golf_prizepicks")
            session.add(status)

        now = datetime.utcnow()
        status.last_attempt = now
        status.total_runs = (status.total_runs or 0) + 1
        if success:
            status.last_success = now
            status.lines_fetched = lines_count
            status.last_error = None
        else:
            status.last_error = error
            status.total_errors = (status.total_errors or 0) + 1

        session.commit()
    except Exception as e:
        session.rollback()
        log.error(f"Failed to update scraper status: {e}")
    finally:
        session.close()


def log_audit(event_type: str, description: str, data: dict = None):
    """Write an audit log entry."""
    session = get_session()
    try:
        import json
        entry = AuditLog(
            event_type=event_type,
            description=description,
            data_json=json.dumps(data) if data else None,
        )
        session.add(entry)
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()


def run_once() -> bool:
    """Execute a single scrape cycle. Returns True on success."""
    scraper = PrizePicksScraper()
    try:
        log.info("Fetching PrizePicks golf projections...")
        projections = scraper.fetch_golf_projections(single_stat_only=False)

        if not projections:
            log.warning("No projections returned (slate may be empty)")
            update_status(success=True, lines_count=0)
            return True

        count = store_lines(projections)
        log.info(f"Stored {count} PrizePicks lines in database")
        update_status(success=True, lines_count=count)
        log_audit("pp_scrape_success", f"Stored {count} golf lines", {
            "count": count,
            "stat_types": list({p.stat_type for p in projections}),
        })
        return True

    except Exception as e:
        log.error(f"Scrape failed: {e}", exc_info=True)
        update_status(success=False, error=str(e))
        log_audit("pp_scrape_error", f"Golf scrape failed: {e}")
        return False


def main():
    log.info("=" * 60)
    log.info("Golf PrizePicks Scraper Service starting")
    log.info(f"  Project root: {PROJECT_ROOT}")
    log.info(f"  Log file:     {LOG_FILE}")
    log.info("=" * 60)

    init_db()

    while not _shutdown:
        run_once()
        interval = get_interval()
        day_type = "tournament" if is_tournament_day() else "off-week"
        log.info(f"Next pull in {interval // 60} min ({day_type} schedule)")

        # Sleep in small increments so we can respond to SIGTERM quickly
        waited = 0
        while waited < interval and not _shutdown:
            time.sleep(min(10, interval - waited))
            waited += 10

    log.info("Scraper service stopped.")


if __name__ == "__main__":
    main()
