"""
Golf Quant Engine -- Stats Worker
===================================
Daily data ingestion pipeline for player stats and tournament data.

Runs once daily (default 6 AM UTC / 2 AM ET):
  1. Refresh PGA Tour player database
  2. Update SG stats from PGA Tour / ShotLink
  3. Sync tournament field for upcoming events
  4. Update player metadata

This worker writes directly to the players, sg_stats, events,
and tournament_field tables via the database ORM.

Run standalone:
    python -m workers.stats_worker
    python -m workers.stats_worker --loop
"""
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workers.base import BaseWorker
from database.connection import get_session_factory, init_db
from database.models import (
    Player, Event, SGStat, TournamentField,
)

log = logging.getLogger(__name__)


class StatsWorker(BaseWorker):
    name = "stats_worker"
    interval_seconds = int(os.environ.get("STATS_WORKER_INTERVAL", 86400))  # 24 hours
    max_retries = 2
    retry_delay = 60.0
    description = "Daily data ingestion: player stats, SG data, tournament fields"

    def execute(self) -> dict:
        init_db()
        factory = get_session_factory()

        players_updated = 0
        sg_records_added = 0
        field_entries = 0
        errors: list[str] = []

        # 1. Refresh player database from PGA Tour
        try:
            players_updated = self._refresh_players(factory)
        except Exception as exc:
            msg = f"Player refresh failed: {exc}"
            self._logger.error(msg, exc_info=True)
            errors.append(msg)

        # 2. Update SG stats
        try:
            sg_records_added = self._refresh_sg_stats(factory)
        except Exception as exc:
            msg = f"SG stats refresh failed: {exc}"
            self._logger.error(msg, exc_info=True)
            errors.append(msg)

        # 3. Sync tournament field for upcoming events
        try:
            field_entries = self._sync_tournament_field(factory)
        except Exception as exc:
            msg = f"Tournament field sync failed: {exc}"
            self._logger.error(msg, exc_info=True)
            errors.append(msg)

        # 4. Update player metadata (world rankings, etc.)
        try:
            self._update_player_metadata(factory)
        except Exception as exc:
            msg = f"Player metadata update failed: {exc}"
            self._logger.error(msg, exc_info=True)
            errors.append(msg)

        total = players_updated + sg_records_added + field_entries
        if total == 0 and errors:
            raise RuntimeError(f"All stats sources failed: {'; '.join(errors)}")

        return {
            "items_processed": total,
            "players_updated": players_updated,
            "sg_records_added": sg_records_added,
            "field_entries": field_entries,
            "errors": errors,
        }

    # ------------------------------------------------------------------ #
    # 1. Refresh players from PGA Tour
    # ------------------------------------------------------------------ #
    def _refresh_players(self, factory) -> int:
        """Fetch current PGA Tour player list and upsert into players table."""
        try:
            from data.scrapers.pga_tour import PGATourScraper
            scraper = PGATourScraper()
            player_list = scraper.fetch_player_list()
        except ImportError:
            self._logger.info("PGA Tour scraper not available — trying ESPN fallback")
            player_list = self._fetch_players_espn()
        except Exception:
            player_list = self._fetch_players_espn()

        if not player_list:
            self._logger.info("No player data available from any source")
            return 0

        session = factory()
        count = 0
        try:
            for p in player_list:
                name = p.get("name", "").strip()
                if not name or name == "Unknown":
                    continue

                existing = session.query(Player).filter_by(
                    name=name, sport="GOLF"
                ).first()

                if existing is None:
                    player = Player(
                        name=name,
                        team=p.get("tour", "PGA"),
                        sport="GOLF",
                        active=True,
                        last_updated=datetime.utcnow(),
                        metadata_json=json.dumps({
                            "pga_id": p.get("player_id", ""),
                            "country": p.get("country", ""),
                        }),
                    )
                    session.add(player)
                    count += 1
                else:
                    # Update activity status
                    existing.active = True
                    existing.last_updated = datetime.utcnow()

            session.commit()
            self._logger.info("Player refresh: %d new players added", count)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return count

    def _fetch_players_espn(self) -> list[dict]:
        """Fallback player fetch via ESPN Golf API."""
        try:
            import requests
            resp = requests.get(
                "https://site.api.espn.com/apis/site/v2/sports/golf/pga/athletes",
                params={"limit": 300},
                timeout=15,
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            athletes = data.get("items", [])
            players = []
            for a in athletes:
                name = a.get("displayName", a.get("fullName", ""))
                if name:
                    players.append({
                        "name": name,
                        "player_id": str(a.get("id", "")),
                        "tour": "PGA",
                        "country": a.get("citizenship", ""),
                    })
            return players
        except Exception as exc:
            self._logger.warning("ESPN player fetch failed: %s", exc)
            return []

    # ------------------------------------------------------------------ #
    # 2. Refresh SG stats
    # ------------------------------------------------------------------ #
    def _refresh_sg_stats(self, factory) -> int:
        """Fetch latest SG stats from PGA Tour or ShotLink."""
        sg_data = []

        # Try PGA Tour GraphQL first
        try:
            from data.scrapers.pga_tour import PGATourScraper
            scraper = PGATourScraper()
            sg_data = scraper.fetch_sg_stats()
        except Exception as exc:
            self._logger.warning("PGA Tour SG fetch failed: %s", exc)

        # Try ShotLink if PGA Tour failed
        if not sg_data:
            try:
                from data.scrapers.shotlink import get_season_sg_snapshot
                snapshot = get_season_sg_snapshot()
                if snapshot is not None and hasattr(snapshot, "to_dict"):
                    records = snapshot.to_dict(orient="records")
                    sg_data = records
            except Exception as exc:
                self._logger.warning("ShotLink SG fetch failed: %s", exc)

        if not sg_data:
            self._logger.info("No SG data available from any source")
            return 0

        session = factory()
        count = 0
        try:
            now = datetime.utcnow()
            current_season = str(now.year)

            for entry in sg_data:
                player_name = entry.get("player_name", entry.get("name", "")).strip()
                if not player_name:
                    continue

                # Find player
                player = session.query(Player).filter_by(
                    name=player_name, sport="GOLF"
                ).first()
                if player is None:
                    # Auto-create player
                    player = Player(
                        name=player_name, sport="GOLF", active=True,
                        last_updated=now,
                    )
                    session.add(player)
                    session.flush()

                # Check for duplicate (same player + season, recent insert)
                recent_sg = (
                    session.query(SGStat)
                    .filter_by(player_id=player.id, season=current_season)
                    .filter(SGStat.created_at >= now - timedelta(days=1))
                    .first()
                )
                if recent_sg:
                    continue  # Already refreshed today

                sg = SGStat(
                    player_id=player.id,
                    sg_total=entry.get("sg_total"),
                    sg_ott=entry.get("sg_ott", entry.get("sg_off_the_tee")),
                    sg_app=entry.get("sg_app", entry.get("sg_approach")),
                    sg_atg=entry.get("sg_atg", entry.get("sg_around_the_green")),
                    sg_putt=entry.get("sg_putt", entry.get("sg_putting")),
                    rounds_played=entry.get("rounds_played", entry.get("rounds")),
                    season=current_season,
                    source="stats_worker",
                )
                session.add(sg)
                count += 1

            session.commit()
            self._logger.info("SG refresh: %d records added", count)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return count

    # ------------------------------------------------------------------ #
    # 3. Sync tournament field
    # ------------------------------------------------------------------ #
    def _sync_tournament_field(self, factory) -> int:
        """Sync tournament field for active/upcoming events."""
        session = factory()
        count = 0

        try:
            # Find active/upcoming events
            now = datetime.utcnow()
            week_ahead = now + timedelta(days=7)

            active_events = (
                session.query(Event)
                .filter(Event.sport == "GOLF")
                .filter(Event.status.in_(["scheduled", "live"]))
                .filter(Event.start_time <= week_ahead)
                .all()
            )

            if not active_events:
                self._logger.info("No upcoming events to sync field for")
                return 0

            for event in active_events:
                # Get field from tournament detector
                try:
                    from scrapers.tournament_detector import detect_current_tournament
                    info = detect_current_tournament()
                    field = info.get("field", [])
                except Exception as exc:
                    self._logger.warning(
                        "Field fetch failed for %s: %s", event.event_name, exc
                    )
                    continue

                for p in field:
                    name = p.get("name", "")
                    if not name or name == "Unknown":
                        continue

                    # Find or create player
                    player = session.query(Player).filter_by(
                        name=name, sport="GOLF"
                    ).first()
                    if player is None:
                        player = Player(
                            name=name, sport="GOLF", active=True,
                            last_updated=now,
                        )
                        session.add(player)
                        session.flush()

                    # Check for existing field entry
                    existing = session.query(TournamentField).filter_by(
                        tournament_id=event.id, player_id=player.id
                    ).first()

                    if existing is None:
                        tf = TournamentField(
                            tournament_id=event.id,
                            player_id=player.id,
                            status="confirmed",
                            metadata_json=json.dumps({
                                "espn_id": p.get("espn_id", ""),
                                "position": p.get("position", ""),
                            }),
                        )
                        session.add(tf)
                        count += 1

            session.commit()
            self._logger.info("Tournament field sync: %d entries added", count)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return count

    # ------------------------------------------------------------------ #
    # 4. Update player metadata
    # ------------------------------------------------------------------ #
    def _update_player_metadata(self, factory) -> None:
        """Update player metadata (rankings, status, etc.)."""
        session = factory()
        try:
            # Mark players as inactive if not seen in 90 days
            cutoff = datetime.utcnow() - timedelta(days=90)
            stale_players = (
                session.query(Player)
                .filter(Player.sport == "GOLF")
                .filter(Player.active == True)
                .filter(Player.last_updated < cutoff)
                .all()
            )

            for p in stale_players:
                p.active = False

            if stale_players:
                self._logger.info(
                    "Marked %d players as inactive (no updates in 90 days)",
                    len(stale_players),
                )

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    )
    import argparse
    parser = argparse.ArgumentParser(description="Golf Stats Worker")
    parser.add_argument("--loop", action="store_true", help="Run in loop mode")
    args = parser.parse_args()

    worker = StatsWorker()
    if args.loop:
        worker.run_loop()
    else:
        success = worker.run_once()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
