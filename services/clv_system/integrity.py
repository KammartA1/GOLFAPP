"""Data quality validation and integrity scoring for CLV data."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from database.connection import DatabaseManager
from services.clv_system.models import OddsSnapshot, BetPriceSnapshot, ClosingLineRecord

logger = logging.getLogger(__name__)


class CLVIntegrityChecker:
    """Validate CLV data quality and compute integrity score."""

    def check_integrity(self, event_id: str) -> dict:
        """Run full integrity check on CLV data for an event.

        Returns dict with scores and issues.
        """
        with DatabaseManager.session_scope() as session:
            # Count odds snapshots
            n_snapshots = (
                session.query(OddsSnapshot)
                .filter(OddsSnapshot.event_id == event_id)
                .count()
            )

            # Count closing lines
            n_closing = (
                session.query(ClosingLineRecord)
                .filter(ClosingLineRecord.event_id == event_id)
                .count()
            )

            # Count bet snapshots
            n_bet_snaps = (
                session.query(BetPriceSnapshot)
                .filter(BetPriceSnapshot.event_id == event_id)
                .count()
            )

            # Bet snapshots with closing data
            n_with_closing = (
                session.query(BetPriceSnapshot)
                .filter(
                    BetPriceSnapshot.event_id == event_id,
                    BetPriceSnapshot.closing_odds_decimal.isnot(None),
                )
                .count()
            )

            # Check for stale odds (no snapshot in last 2 hours when event is upcoming)
            two_hours_ago = datetime.utcnow() - timedelta(hours=2)
            recent_snaps = (
                session.query(OddsSnapshot)
                .filter(
                    OddsSnapshot.event_id == event_id,
                    OddsSnapshot.captured_at >= two_hours_ago,
                )
                .count()
            )

            # Check for duplicate snapshots
            from sqlalchemy import func
            dupes = (
                session.query(func.count(OddsSnapshot.id))
                .filter(OddsSnapshot.event_id == event_id)
                .group_by(
                    OddsSnapshot.player,
                    OddsSnapshot.market_type,
                    OddsSnapshot.source,
                    OddsSnapshot.captured_at,
                )
                .having(func.count(OddsSnapshot.id) > 1)
                .count()
            )

        # Compute integrity score (0-100)
        issues = []
        score = 100.0

        if n_snapshots < 10:
            score -= 30
            issues.append(f"Very few odds snapshots ({n_snapshots})")
        elif n_snapshots < 50:
            score -= 10
            issues.append(f"Low odds snapshot count ({n_snapshots})")

        if n_closing == 0:
            score -= 25
            issues.append("No closing lines captured")

        if n_bet_snaps > 0 and n_with_closing == 0:
            score -= 20
            issues.append("No bet snapshots have closing line data")
        elif n_bet_snaps > 0:
            closing_pct = n_with_closing / n_bet_snaps
            if closing_pct < 0.8:
                score -= 10
                issues.append(f"Only {closing_pct:.0%} of bets have closing lines")

        if recent_snaps == 0 and n_snapshots > 0:
            score -= 10
            issues.append("No recent odds snapshots (data may be stale)")

        if dupes > 0:
            score -= 5
            issues.append(f"{dupes} duplicate snapshot groups found")

        score = max(0, min(100, score))

        return {
            "event_id": event_id,
            "integrity_score": round(score, 1),
            "n_odds_snapshots": n_snapshots,
            "n_closing_lines": n_closing,
            "n_bet_snapshots": n_bet_snaps,
            "n_bets_with_closing": n_with_closing,
            "n_recent_snapshots": recent_snaps,
            "n_duplicate_groups": dupes,
            "issues": issues,
            "is_healthy": score >= 70,
        }
