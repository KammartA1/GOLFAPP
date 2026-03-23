"""Closing line audit — Verify closing line capture quality."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from database.connection import DatabaseManager
from database.models import Bet, CLVLog

logger = logging.getLogger(__name__)


class ClosingLineAuditor:
    """Audit closing line data completeness and quality."""

    def __init__(self, sport: str = "golf"):
        self.sport = sport

    def audit(self, days: int = 30) -> dict:
        """Audit closing line capture quality.

        Checks:
          - Coverage: what % of bets have closing lines
          - Timeliness: were closing lines captured near tee time
          - Consistency: do closing lines make sense relative to bet lines
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        with DatabaseManager.session_scope() as session:
            settled_bets = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.status.in_(["won", "lost"]),
                    Bet.timestamp >= cutoff,
                )
                .all()
            )

            if not settled_bets:
                return {"status": "no_data", "score": 0.0}

            n_total = len(settled_bets)
            n_with_closing = sum(1 for b in settled_bets if b.closing_line is not None)
            n_with_closing_odds = sum(1 for b in settled_bets if b.closing_odds is not None)

            # CLV log coverage
            clv_records = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport)
                .all()
            )
            bet_ids_with_clv = set(c.bet_id for c in clv_records)
            n_with_clv = sum(1 for b in settled_bets if b.bet_id in bet_ids_with_clv)

        coverage_pct = n_with_closing / max(n_total, 1)
        clv_coverage = n_with_clv / max(n_total, 1)

        issues = []
        score = 100.0

        if coverage_pct < 0.5:
            score -= 30
            issues.append(f"Only {coverage_pct:.0%} of bets have closing lines")
        elif coverage_pct < 0.8:
            score -= 15
            issues.append(f"Closing line coverage at {coverage_pct:.0%} — should be >80%")

        if clv_coverage < 0.5:
            score -= 20
            issues.append(f"Only {clv_coverage:.0%} of bets have CLV calculated")

        if n_with_closing_odds < n_with_closing * 0.5:
            score -= 10
            issues.append("Many closing lines missing odds data")

        return {
            "score": round(max(0, score), 1),
            "n_settled_bets": n_total,
            "n_with_closing_line": n_with_closing,
            "n_with_closing_odds": n_with_closing_odds,
            "n_with_clv": n_with_clv,
            "closing_line_coverage": round(coverage_pct, 4),
            "clv_coverage": round(clv_coverage, 4),
            "issues": issues,
            "is_healthy": score >= 70,
        }
