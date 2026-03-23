"""Odds ingestion — Ingest odds from available sources, store with timestamps."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from database.connection import DatabaseManager
from services.clv_system.models import OddsSnapshot

logger = logging.getLogger(__name__)


class OddsIngestionService:
    """Ingest and store odds from multiple sources."""

    def ingest_odds(
        self,
        event_id: str,
        player: str,
        market_type: str,
        source: str,
        odds_american: Optional[int] = None,
        odds_decimal: Optional[float] = None,
        line: Optional[float] = None,
        is_opening: bool = False,
        is_closing: bool = False,
        raw_data: Optional[str] = None,
    ) -> int:
        """Ingest a single odds snapshot.

        Returns the snapshot ID.
        """
        if odds_decimal is None and odds_american is not None:
            if odds_american > 0:
                odds_decimal = 1.0 + odds_american / 100.0
            elif odds_american < 0:
                odds_decimal = 1.0 + 100.0 / abs(odds_american)
            else:
                odds_decimal = 2.0

        implied_prob = 1.0 / odds_decimal if odds_decimal and odds_decimal > 1.0 else None

        with DatabaseManager.session_scope() as session:
            snapshot = OddsSnapshot(
                event_id=event_id,
                player=player,
                market_type=market_type,
                source=source,
                odds_american=odds_american,
                odds_decimal=odds_decimal,
                implied_prob=implied_prob,
                line=line,
                captured_at=datetime.utcnow(),
                is_opening=is_opening,
                is_closing=is_closing,
                raw_data=raw_data,
            )
            session.add(snapshot)
            session.flush()
            snap_id = snapshot.id

        logger.debug("Ingested odds: %s/%s/%s from %s (%.3f)",
                      event_id, player, market_type, source, odds_decimal or 0)
        return snap_id

    def ingest_batch(self, odds_list: List[Dict]) -> int:
        """Ingest a batch of odds snapshots.

        Each dict should have: event_id, player, market_type, source, odds_american/odds_decimal.
        Returns count of ingested records.
        """
        count = 0
        with DatabaseManager.session_scope() as session:
            for odds in odds_list:
                odds_dec = odds.get("odds_decimal")
                odds_am = odds.get("odds_american")

                if odds_dec is None and odds_am is not None:
                    if odds_am > 0:
                        odds_dec = 1.0 + odds_am / 100.0
                    elif odds_am < 0:
                        odds_dec = 1.0 + 100.0 / abs(odds_am)
                    else:
                        odds_dec = 2.0

                implied = 1.0 / odds_dec if odds_dec and odds_dec > 1.0 else None

                snapshot = OddsSnapshot(
                    event_id=odds["event_id"],
                    player=odds["player"],
                    market_type=odds["market_type"],
                    source=odds["source"],
                    odds_american=odds_am,
                    odds_decimal=odds_dec,
                    implied_prob=implied,
                    line=odds.get("line"),
                    captured_at=datetime.utcnow(),
                    is_opening=odds.get("is_opening", False),
                    is_closing=odds.get("is_closing", False),
                    raw_data=odds.get("raw_data"),
                )
                session.add(snapshot)
                count += 1

        logger.info("Ingested %d odds snapshots", count)
        return count

    def get_latest_odds(
        self,
        event_id: str,
        player: str,
        market_type: str,
        source: Optional[str] = None,
    ) -> Optional[dict]:
        """Get the most recent odds for a player/market."""
        with DatabaseManager.session_scope() as session:
            q = (
                session.query(OddsSnapshot)
                .filter(
                    OddsSnapshot.event_id == event_id,
                    OddsSnapshot.player == player,
                    OddsSnapshot.market_type == market_type,
                )
            )
            if source:
                q = q.filter(OddsSnapshot.source == source)
            snap = q.order_by(OddsSnapshot.captured_at.desc()).first()

            if not snap:
                return None

            return {
                "odds_american": snap.odds_american,
                "odds_decimal": snap.odds_decimal,
                "implied_prob": snap.implied_prob,
                "source": snap.source,
                "captured_at": snap.captured_at.isoformat(),
            }
