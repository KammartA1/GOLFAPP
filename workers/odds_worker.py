"""Odds Worker — Scrape PrizePicks and other sources on schedule.

Golf: Every 15 minutes during tournament hours (Thu-Sun 7am-8pm ET)
Writes to: line_movements table
Updates: scraper_status table

Integrates with existing scrapers but writes to unified DB.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from database.connection import DatabaseManager
from database.models import LineMovement, Player, normalize_player_name
from .base_worker import BaseWorker

logger = logging.getLogger(__name__)


class OddsWorker(BaseWorker):
    WORKER_NAME = "odds_worker_golf"
    SPORT = "golf"
    DEFAULT_INTERVAL_SECONDS = 900  # 15 minutes

    def __init__(self, interval_seconds: int | None = None):
        super().__init__(interval_seconds)
        self._scraper = None

    def _get_scraper(self):
        """Lazy-load the PrizePicks scraper."""
        if self._scraper is None:
            try:
                from scrapers.prizepicks import PrizePicksScraper
                self._scraper = PrizePicksScraper()
            except ImportError:
                try:
                    from data.scrapers.prizepicks_scraper import PrizePicksScraper
                    self._scraper = PrizePicksScraper()
                except ImportError:
                    logger.warning("No PrizePicks scraper found — using API fallback")
                    self._scraper = PrizePicksFallbackScraper()
        return self._scraper

    def execute(self) -> dict:
        """Scrape all golf prop sources and store in line_movements."""
        results = {"sources": {}, "total_lines": 0, "errors": []}

        # 1. PrizePicks
        pp_result = self._scrape_prizepicks()
        results["sources"]["prizepicks"] = pp_result
        results["total_lines"] += pp_result.get("lines_stored", 0)
        if pp_result.get("error"):
            results["errors"].append(f"prizepicks: {pp_result['error']}")

        # 2. Mark old lines as inactive
        self._deactivate_stale_lines()

        # Update scraper status
        self._update_scraper_status(
            success=len(results["errors"]) == 0,
            lines_count=results["total_lines"],
            error="; ".join(results["errors"]) if results["errors"] else None,
        )

        return results

    def _scrape_prizepicks(self) -> dict:
        """Scrape PrizePicks golf props."""
        try:
            scraper = self._get_scraper()
            raw_lines = scraper.fetch_golf_lines()

            if not raw_lines:
                return {"lines_fetched": 0, "lines_stored": 0}

            stored = 0
            with DatabaseManager.session_scope() as session:
                for line_data in raw_lines:
                    player_name = line_data.get("player_name", "")
                    if not player_name:
                        continue

                    norm_name = normalize_player_name(player_name)

                    # Ensure player exists
                    player = session.query(Player).filter_by(
                        normalized_name=norm_name
                    ).first()
                    if player is None:
                        player = Player(
                            name=player_name,
                            normalized_name=norm_name,
                            sport="golf",
                        )
                        session.add(player)
                        session.flush()

                    # Store line movement
                    movement = LineMovement(
                        sport="golf",
                        player=player_name,
                        stat_type=line_data.get("stat_type", "unknown"),
                        source="prizepicks",
                        line=float(line_data.get("line", 0)),
                        odds_type=line_data.get("odds_type", "standard"),
                        is_flash_sale=line_data.get("is_flash_sale", False),
                        discount_pct=line_data.get("discount_pct"),
                        captured_at=datetime.utcnow(),
                        is_active=True,
                        raw_data=json.dumps(line_data),
                    )
                    session.add(movement)
                    stored += 1

            self._update_scraper_status(
                success=True,
                lines_count=stored,
                scraper_name="prizepicks_golf",
            )

            return {"lines_fetched": len(raw_lines), "lines_stored": stored}

        except Exception as e:
            logger.exception("PrizePicks golf scrape failed")
            return {"lines_fetched": 0, "lines_stored": 0, "error": str(e)}

    def _deactivate_stale_lines(self):
        """Mark lines older than 4 hours as inactive."""
        try:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(hours=4)

            with DatabaseManager.session_scope() as session:
                stale = (
                    session.query(LineMovement)
                    .filter(
                        LineMovement.sport == "golf",
                        LineMovement.is_active == True,
                        LineMovement.captured_at < cutoff,
                    )
                    .all()
                )
                for line in stale:
                    line.is_active = False

                if stale:
                    logger.info("Deactivated %d stale golf lines", len(stale))

        except Exception:
            logger.exception("Failed to deactivate stale lines")


class PrizePicksFallbackScraper:
    """Minimal fallback scraper using the PrizePicks API directly."""

    PP_API_URL = "https://api.prizepicks.com/projections"

    def fetch_golf_lines(self) -> list[dict]:
        """Fetch golf projections from PrizePicks API."""
        try:
            import requests
            resp = requests.get(
                self.PP_API_URL,
                params={"league_id": 19},  # PGA Tour
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            lines = []
            included = {item["id"]: item for item in data.get("included", [])}

            for proj in data.get("data", []):
                attrs = proj.get("attributes", {})
                player_rel = proj.get("relationships", {}).get("new_player", {}).get("data", {})
                player_id = player_rel.get("id", "")
                player_info = included.get(player_id, {}).get("attributes", {})

                lines.append({
                    "player_name": player_info.get("display_name", player_info.get("name", "")),
                    "stat_type": attrs.get("stat_type", "unknown"),
                    "line": attrs.get("line_score", 0),
                    "odds_type": attrs.get("odds_type", "standard"),
                    "is_flash_sale": attrs.get("is_promo", False),
                    "discount_pct": attrs.get("discount_percentage"),
                    "start_time": attrs.get("start_time"),
                })

            return lines

        except Exception as e:
            logger.exception("PrizePicks API fallback failed")
            return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    worker = OddsWorker()
    worker.run_forever()
