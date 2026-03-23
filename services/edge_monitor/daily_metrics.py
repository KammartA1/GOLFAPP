"""Daily edge metrics computation."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np

from database.connection import DatabaseManager
from database.models import Bet, CLVLog, CalibrationLog

logger = logging.getLogger(__name__)


class DailyEdgeMetrics:
    """Compute daily edge metrics for monitoring."""

    def __init__(self, sport: str = "golf"):
        self.sport = sport

    def compute(self) -> dict:
        """Compute current daily edge metrics."""
        with DatabaseManager.session_scope() as session:
            now = datetime.utcnow()

            # Recent settled bets
            settled = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.status.in_(["won", "lost"]),
                )
                .order_by(Bet.settled_at.desc())
                .limit(500)
                .all()
            )

            if not settled:
                return {"has_data": False}

            # CLV rolling windows
            clv_records = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport)
                .order_by(CLVLog.calculated_at.desc())
                .limit(500)
                .all()
            )

            clv_windows = {}
            for window in [25, 50, 100, 250]:
                subset = clv_records[:window]
                if subset:
                    clvs = [c.clv_cents for c in subset]
                    clv_windows[f"clv_{window}"] = {
                        "avg": round(float(np.mean(clvs)), 2),
                        "beat_rate": round(float(np.mean([1.0 if c.beat_close else 0.0 for c in subset])), 4),
                        "n": len(subset),
                    }

            # ROI
            total_staked = sum(b.stake for b in settled)
            total_pnl = sum(b.pnl for b in settled)
            roi = total_pnl / max(total_staked, 1)

            # Win rate
            wins = sum(1 for b in settled if b.status == "won")
            win_rate = wins / max(len(settled), 1)

            # Recent performance (last 7 days)
            week_ago = now - timedelta(days=7)
            recent = [b for b in settled if b.settled_at and b.settled_at >= week_ago]
            recent_pnl = sum(b.pnl for b in recent)
            recent_wr = sum(1 for b in recent if b.status == "won") / max(len(recent), 1)

            return {
                "has_data": True,
                "report_date": now.isoformat(),
                "total_bets": len(settled),
                "total_pnl": round(total_pnl, 2),
                "roi": round(roi, 4),
                "win_rate": round(win_rate, 4),
                "clv_windows": clv_windows,
                "recent_7d": {
                    "n_bets": len(recent),
                    "pnl": round(recent_pnl, 2),
                    "win_rate": round(recent_wr, 4),
                },
            }
