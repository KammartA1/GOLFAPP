"""CLV Calculator — CLV in cents, probability, beat-close rate, by segment."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from database.connection import DatabaseManager
from database.models import CLVLog, Bet
from services.clv_system.models import BetPriceSnapshot

logger = logging.getLogger(__name__)


class CLVCalculator:
    """Calculate and analyze Closing Line Value across all bets."""

    def __init__(self, sport: str = "golf"):
        self.sport = sport

    def calculate_clv_for_bet(
        self,
        bet_id: str,
        closing_odds_decimal: float,
        bet_odds_decimal: float,
    ) -> dict:
        """Calculate CLV for a single bet.

        Returns:
            {
                'clv_cents': float,
                'clv_probability': float,
                'beat_close': bool,
                'bet_implied_prob': float,
                'closing_implied_prob': float,
            }
        """
        bet_prob = 1.0 / bet_odds_decimal if bet_odds_decimal > 1.0 else 0.5
        close_prob = 1.0 / closing_odds_decimal if closing_odds_decimal > 1.0 else 0.5

        clv_prob = close_prob - bet_prob
        clv_cents = clv_prob * 100
        beat_close = bet_prob < close_prob

        return {
            "clv_cents": round(clv_cents, 2),
            "clv_probability": round(clv_prob, 4),
            "beat_close": beat_close,
            "bet_implied_prob": round(bet_prob, 4),
            "closing_implied_prob": round(close_prob, 4),
        }

    def rolling_clv(self, window: int = 100) -> dict:
        """Get rolling CLV metrics over last N bets."""
        with DatabaseManager.session_scope() as session:
            records = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport)
                .order_by(CLVLog.calculated_at.desc())
                .limit(window)
                .all()
            )

            if not records:
                return {
                    "avg_clv_cents": 0.0,
                    "beat_close_rate": 0.0,
                    "n_bets": 0,
                    "std_clv": 0.0,
                    "median_clv": 0.0,
                }

            clvs = [r.clv_cents for r in records]
            clv_arr = np.array(clvs)

            return {
                "avg_clv_cents": round(float(np.mean(clv_arr)), 2),
                "median_clv_cents": round(float(np.median(clv_arr)), 2),
                "std_clv": round(float(np.std(clv_arr)), 2),
                "beat_close_rate": round(float(np.mean([1.0 if r.beat_close else 0.0 for r in records])), 4),
                "n_bets": len(records),
                "min_clv": round(float(np.min(clv_arr)), 2),
                "max_clv": round(float(np.max(clv_arr)), 2),
                "pct_positive": round(float(np.mean(clv_arr > 0)), 4),
            }

    def clv_by_segment(self) -> dict:
        """Break down CLV by market type, book, time of day, etc."""
        with DatabaseManager.session_scope() as session:
            bet_snaps = (
                session.query(BetPriceSnapshot)
                .filter(BetPriceSnapshot.clv_cents.isnot(None))
                .all()
            )

            if not bet_snaps:
                return {"by_market_type": {}, "by_book": {}}

            # By market type
            by_market = defaultdict(list)
            for snap in bet_snaps:
                by_market[snap.market_type].append(snap.clv_cents)

            market_results = {}
            for mtype, clvs in by_market.items():
                arr = np.array(clvs)
                market_results[mtype] = {
                    "n_bets": len(clvs),
                    "avg_clv": round(float(np.mean(arr)), 2),
                    "beat_close_rate": round(float(np.mean(arr > 0)), 4),
                }

            return {
                "by_market_type": market_results,
                "total_bets_with_clv": len(bet_snaps),
            }

    def clv_trend(self, windows: List[int] = None) -> dict:
        """CLV trend over multiple rolling windows."""
        if windows is None:
            windows = [25, 50, 100, 250, 500]

        trends = {}
        for w in windows:
            result = self.rolling_clv(w)
            trends[f"last_{w}"] = result

        # Determine trend direction
        if len(trends) >= 2:
            short_clv = trends.get("last_50", {}).get("avg_clv_cents", 0)
            long_clv = trends.get("last_250", {}).get("avg_clv_cents", 0)
            if short_clv > long_clv + 0.5:
                trend_direction = "improving"
            elif short_clv < long_clv - 0.5:
                trend_direction = "declining"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "unknown"

        trends["trend_direction"] = trend_direction
        return trends
