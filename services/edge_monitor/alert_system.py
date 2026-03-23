"""Alert system — Daily EDGE = YES/NO verdict with detailed reasoning."""

from __future__ import annotations

import logging
from datetime import datetime

from services.edge_monitor.daily_metrics import DailyEdgeMetrics
from services.edge_monitor.trend_detection import EdgeTrendDetector
from database.connection import DatabaseManager
from database.models import CLVLog

logger = logging.getLogger(__name__)


class EdgeAlertSystem:
    """Daily edge alert: EDGE = YES or EDGE = NO with reasoning."""

    def __init__(self, sport: str = "golf"):
        self.sport = sport
        self.metrics = DailyEdgeMetrics(sport)
        self.trend_detector = EdgeTrendDetector()

    def daily_verdict(self) -> dict:
        """Generate daily EDGE = YES/NO verdict.

        Returns:
            {
                'edge_exists': bool,
                'verdict': str ("YES" or "NO"),
                'confidence': float (0-1),
                'reasons': list of str,
                'metrics': dict,
                'recommended_action': str,
            }
        """
        daily = self.metrics.compute()
        if not daily.get("has_data"):
            return {
                "edge_exists": False,
                "verdict": "UNKNOWN",
                "confidence": 0.0,
                "reasons": ["No settled bet data available"],
                "recommended_action": "Wait for more data before betting",
            }

        reasons_yes = []
        reasons_no = []

        # 1. CLV check (most important)
        clv_100 = daily.get("clv_windows", {}).get("clv_100", {})
        clv_avg = clv_100.get("avg", 0)
        beat_rate = clv_100.get("beat_rate", 0.5)

        if clv_avg > 1.0:
            reasons_yes.append(f"CLV positive: {clv_avg:.1f}c (last 100)")
        elif clv_avg > 0:
            reasons_yes.append(f"CLV slightly positive: {clv_avg:.1f}c")
        else:
            reasons_no.append(f"CLV negative: {clv_avg:.1f}c (last 100)")

        if beat_rate > 0.52:
            reasons_yes.append(f"Beating closing line {beat_rate:.1%} of the time")
        elif beat_rate < 0.48:
            reasons_no.append(f"Not beating closing line ({beat_rate:.1%})")

        # 2. ROI check
        roi = daily.get("roi", 0)
        if roi > 0.02:
            reasons_yes.append(f"Positive ROI: {roi:.1%}")
        elif roi < -0.03:
            reasons_no.append(f"Negative ROI: {roi:.1%}")

        # 3. Trend detection (CUSUM)
        clv_values = self._get_clv_series()
        if len(clv_values) >= 50:
            edge_death = self.trend_detector.detect_edge_death(clv_values)
            if edge_death.get("edge_dead"):
                reasons_no.append("CUSUM detected edge death — CLV consistently negative")
            elif edge_death.get("edge_warning"):
                reasons_no.append("CUSUM warning — edge may be declining")

            trend = self.trend_detector.trend_analysis(clv_values)
            if trend.get("trend") == "declining" and trend.get("is_significant"):
                reasons_no.append(f"Significant declining trend (slope={trend['slope']:.3f})")
            elif trend.get("trend") == "improving":
                reasons_yes.append("Edge trend is improving")

        # 4. Recent performance
        recent = daily.get("recent_7d", {})
        if recent.get("n_bets", 0) >= 5:
            if recent["pnl"] > 0:
                reasons_yes.append(f"Recent 7d: +${recent['pnl']:.0f}")
            elif recent["pnl"] < -50:
                reasons_no.append(f"Recent 7d: -${abs(recent['pnl']):.0f}")

        # Verdict
        yes_score = len(reasons_yes) * 1.5
        no_score = len(reasons_no) * 2.0  # Weight "no" reasons more heavily

        if clv_avg > 0.5 and beat_rate > 0.50 and no_score < yes_score:
            edge_exists = True
            verdict = "YES"
            confidence = min(0.95, 0.5 + (yes_score - no_score) * 0.1)
        elif clv_avg < -0.5 or beat_rate < 0.45 or no_score > yes_score * 1.5:
            edge_exists = False
            verdict = "NO"
            confidence = min(0.95, 0.5 + (no_score - yes_score) * 0.1)
        else:
            edge_exists = True  # Give benefit of doubt
            verdict = "MARGINAL"
            confidence = 0.4

        # Recommended action
        if verdict == "YES":
            action = "Continue betting with current approach"
        elif verdict == "MARGINAL":
            action = "Reduce bet sizing by 50% and monitor closely"
        else:
            action = "Suspend betting. Investigate model performance."

        return {
            "edge_exists": edge_exists,
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "reasons_for": reasons_yes,
            "reasons_against": reasons_no,
            "metrics": daily,
            "recommended_action": action,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _get_clv_series(self, limit: int = 500) -> list:
        """Get CLV series from database."""
        with DatabaseManager.session_scope() as session:
            records = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport)
                .order_by(CLVLog.calculated_at.asc())
                .limit(limit)
                .all()
            )
            return [r.clv_cents for r in records]
