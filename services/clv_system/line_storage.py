"""Time-series line movement storage and retrieval."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np

from database.connection import DatabaseManager
from services.clv_system.models import OddsSnapshot

logger = logging.getLogger(__name__)


class LineStorageService:
    """Time-series storage and analysis of line movements."""

    def get_line_history(
        self,
        event_id: str,
        player: str,
        market_type: str,
        source: Optional[str] = None,
        hours: int = 168,
    ) -> List[dict]:
        """Get full line movement history for a player/market."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        with DatabaseManager.session_scope() as session:
            q = (
                session.query(OddsSnapshot)
                .filter(
                    OddsSnapshot.event_id == event_id,
                    OddsSnapshot.player == player,
                    OddsSnapshot.market_type == market_type,
                    OddsSnapshot.captured_at >= cutoff,
                )
            )
            if source:
                q = q.filter(OddsSnapshot.source == source)

            snapshots = q.order_by(OddsSnapshot.captured_at.asc()).all()

            return [
                {
                    "odds_decimal": s.odds_decimal,
                    "odds_american": s.odds_american,
                    "implied_prob": s.implied_prob,
                    "source": s.source,
                    "captured_at": s.captured_at.isoformat(),
                    "is_opening": s.is_opening,
                    "is_closing": s.is_closing,
                }
                for s in snapshots
            ]

    def get_line_movement(
        self,
        event_id: str,
        player: str,
        market_type: str,
    ) -> dict:
        """Compute line movement summary: opening, current, total movement."""
        history = self.get_line_history(event_id, player, market_type)

        if not history:
            return {"has_data": False}

        opening = history[0]
        current = history[-1]

        opening_prob = opening.get("implied_prob", 0.5)
        current_prob = current.get("implied_prob", 0.5)
        movement = current_prob - opening_prob

        # Detect sharp movements (large single moves)
        probs = [h.get("implied_prob", 0.5) for h in history if h.get("implied_prob")]
        if len(probs) >= 2:
            diffs = np.diff(probs)
            max_single_move = float(np.max(np.abs(diffs)))
        else:
            max_single_move = 0.0

        return {
            "has_data": True,
            "opening_prob": round(opening_prob, 4),
            "current_prob": round(current_prob, 4),
            "total_movement": round(movement, 4),
            "total_movement_cents": round(movement * 100, 2),
            "n_snapshots": len(history),
            "max_single_move": round(max_single_move, 4),
            "direction": "shortening" if movement > 0 else "lengthening" if movement < 0 else "stable",
            "opening_time": opening.get("captured_at"),
            "latest_time": current.get("captured_at"),
        }

    def get_consensus_line(
        self,
        event_id: str,
        player: str,
        market_type: str,
    ) -> Optional[float]:
        """Get consensus implied probability across all sources."""
        with DatabaseManager.session_scope() as session:
            # Get latest from each source
            from sqlalchemy import func
            subq = (
                session.query(
                    OddsSnapshot.source,
                    func.max(OddsSnapshot.captured_at).label("max_time"),
                )
                .filter(
                    OddsSnapshot.event_id == event_id,
                    OddsSnapshot.player == player,
                    OddsSnapshot.market_type == market_type,
                )
                .group_by(OddsSnapshot.source)
                .subquery()
            )

            latest = (
                session.query(OddsSnapshot)
                .join(subq, (OddsSnapshot.source == subq.c.source) &
                      (OddsSnapshot.captured_at == subq.c.max_time))
                .filter(
                    OddsSnapshot.event_id == event_id,
                    OddsSnapshot.player == player,
                    OddsSnapshot.market_type == market_type,
                )
                .all()
            )

            if not latest:
                return None

            probs = [s.implied_prob for s in latest if s.implied_prob]
            if not probs:
                return None

            return round(float(np.mean(probs)), 4)
