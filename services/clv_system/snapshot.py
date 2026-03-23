"""Bet-time price snapshot — Capture the exact odds when a bet is placed."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from database.connection import DatabaseManager
from services.clv_system.models import BetPriceSnapshot, OddsSnapshot

logger = logging.getLogger(__name__)


class BetSnapshotService:
    """Capture and store the full pricing context at bet placement time."""

    def capture_bet_snapshot(
        self,
        bet_id: str,
        event_id: str,
        player: str,
        market_type: str,
        bet_odds_american: int,
        bet_odds_decimal: float,
        bet_line: Optional[float] = None,
    ) -> int:
        """Capture a snapshot of all available prices at bet time.

        Returns the snapshot ID.
        """
        bet_implied = 1.0 / bet_odds_decimal if bet_odds_decimal > 1.0 else 0.5

        # Find best available odds at this moment
        best_odds = bet_odds_decimal
        best_source = "bet_book"

        with DatabaseManager.session_scope() as session:
            # Get latest odds from all sources
            from sqlalchemy import func
            latest = (
                session.query(OddsSnapshot)
                .filter(
                    OddsSnapshot.event_id == event_id,
                    OddsSnapshot.player == player,
                    OddsSnapshot.market_type == market_type,
                )
                .order_by(OddsSnapshot.captured_at.desc())
                .limit(20)
                .all()
            )

            seen_sources = set()
            for snap in latest:
                if snap.source not in seen_sources:
                    seen_sources.add(snap.source)
                    if snap.odds_decimal and snap.odds_decimal > best_odds:
                        best_odds = snap.odds_decimal
                        best_source = snap.source

            snapshot = BetPriceSnapshot(
                bet_id=bet_id,
                event_id=event_id,
                player=player,
                market_type=market_type,
                bet_odds_american=bet_odds_american,
                bet_odds_decimal=bet_odds_decimal,
                bet_implied_prob=bet_implied,
                bet_line=bet_line,
                best_odds_decimal=best_odds,
                best_source=best_source,
                bet_placed_at=datetime.utcnow(),
            )
            session.add(snapshot)
            session.flush()
            snap_id = snapshot.id

        logger.info("Captured bet snapshot for %s: odds=%.3f, best=%.3f (%s)",
                     bet_id, bet_odds_decimal, best_odds, best_source)
        return snap_id

    def get_bet_snapshot(self, bet_id: str) -> Optional[dict]:
        """Retrieve bet-time snapshot."""
        with DatabaseManager.session_scope() as session:
            snap = (
                session.query(BetPriceSnapshot)
                .filter_by(bet_id=bet_id)
                .first()
            )
            if not snap:
                return None

            return {
                "bet_id": snap.bet_id,
                "bet_odds_decimal": snap.bet_odds_decimal,
                "bet_implied_prob": snap.bet_implied_prob,
                "best_odds_decimal": snap.best_odds_decimal,
                "best_source": snap.best_source,
                "closing_odds_decimal": snap.closing_odds_decimal,
                "closing_implied_prob": snap.closing_implied_prob,
                "clv_cents": snap.clv_cents,
                "beat_close": snap.beat_close,
                "bet_placed_at": snap.bet_placed_at.isoformat(),
            }
