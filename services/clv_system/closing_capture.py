"""Closing line capture — Capture closing lines at first tee time."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from database.connection import DatabaseManager
from services.clv_system.models import ClosingLineRecord, OddsSnapshot, BetPriceSnapshot

logger = logging.getLogger(__name__)


class ClosingLineCapture:
    """Capture closing lines at tournament start (first tee time)."""

    def capture_closing_lines(
        self,
        event_id: str,
        first_tee_time: Optional[datetime] = None,
    ) -> int:
        """Capture closing lines for all players/markets in an event.

        Should be called just before the first tee time.
        Returns count of closing lines captured.
        """
        if first_tee_time is None:
            first_tee_time = datetime.utcnow()

        count = 0
        with DatabaseManager.session_scope() as session:
            # Get the latest odds for each player/market/source combination
            from sqlalchemy import func

            latest_subq = (
                session.query(
                    OddsSnapshot.player,
                    OddsSnapshot.market_type,
                    OddsSnapshot.source,
                    func.max(OddsSnapshot.id).label("max_id"),
                )
                .filter(OddsSnapshot.event_id == event_id)
                .group_by(
                    OddsSnapshot.player,
                    OddsSnapshot.market_type,
                    OddsSnapshot.source,
                )
                .subquery()
            )

            latest_snaps = (
                session.query(OddsSnapshot)
                .join(latest_subq, OddsSnapshot.id == latest_subq.c.max_id)
                .all()
            )

            for snap in latest_snaps:
                # Mark the original snapshot as closing
                snap.is_closing = True

                # Create closing line record
                closing = ClosingLineRecord(
                    event_id=event_id,
                    player=snap.player,
                    market_type=snap.market_type,
                    source=snap.source,
                    closing_odds_american=snap.odds_american,
                    closing_odds_decimal=snap.odds_decimal,
                    closing_implied_prob=snap.implied_prob,
                    closing_line=snap.line,
                    first_tee_time=first_tee_time,
                    captured_at=datetime.utcnow(),
                )
                session.add(closing)
                count += 1

        logger.info("Captured %d closing lines for event %s", count, event_id)
        return count

    def update_bet_snapshots_with_closing(self, event_id: str) -> int:
        """Update bet price snapshots with closing line data.

        Should be called after capture_closing_lines.
        Returns count of updated bet snapshots.
        """
        updated = 0
        with DatabaseManager.session_scope() as session:
            # Get all bet snapshots for this event
            bet_snaps = (
                session.query(BetPriceSnapshot)
                .filter(BetPriceSnapshot.event_id == event_id)
                .filter(BetPriceSnapshot.closing_odds_decimal.is_(None))
                .all()
            )

            for bet_snap in bet_snaps:
                # Find best closing line for this player/market
                closing = (
                    session.query(ClosingLineRecord)
                    .filter(
                        ClosingLineRecord.event_id == event_id,
                        ClosingLineRecord.player == bet_snap.player,
                        ClosingLineRecord.market_type == bet_snap.market_type,
                    )
                    .order_by(ClosingLineRecord.closing_odds_decimal.desc())
                    .first()
                )

                if closing:
                    bet_snap.closing_odds_american = closing.closing_odds_american
                    bet_snap.closing_odds_decimal = closing.closing_odds_decimal
                    bet_snap.closing_implied_prob = closing.closing_implied_prob
                    bet_snap.closing_line = closing.closing_line
                    bet_snap.closing_captured_at = closing.captured_at

                    # Calculate CLV
                    if bet_snap.bet_implied_prob and closing.closing_implied_prob:
                        bet_snap.clv_cents = round(
                            (closing.closing_implied_prob - bet_snap.bet_implied_prob) * 100, 2
                        )
                        bet_snap.beat_close = bet_snap.bet_implied_prob < closing.closing_implied_prob

                    updated += 1

        logger.info("Updated %d bet snapshots with closing lines for event %s", updated, event_id)
        return updated

    def get_closing_line(
        self,
        event_id: str,
        player: str,
        market_type: str,
        source: Optional[str] = None,
    ) -> Optional[dict]:
        """Get closing line for a specific player/market."""
        with DatabaseManager.session_scope() as session:
            q = (
                session.query(ClosingLineRecord)
                .filter(
                    ClosingLineRecord.event_id == event_id,
                    ClosingLineRecord.player == player,
                    ClosingLineRecord.market_type == market_type,
                )
            )
            if source:
                q = q.filter(ClosingLineRecord.source == source)

            record = q.order_by(ClosingLineRecord.captured_at.desc()).first()

            if not record:
                return None

            return {
                "closing_odds_decimal": record.closing_odds_decimal,
                "closing_odds_american": record.closing_odds_american,
                "closing_implied_prob": record.closing_implied_prob,
                "source": record.source,
                "first_tee_time": record.first_tee_time.isoformat() if record.first_tee_time else None,
            }
