"""
services/data_audit/closing_line_audit.py
==========================================
Validates closing line data quality for golf.

Checks:
  - Are closing lines true closes or stale pre-close data?
  - Compare our closing lines with independent sources
  - Is closing line capture timing consistent? (always at first tee,
    not sometimes hours before)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from sqlalchemy import func as sa_func

from quant_system.db.schema import get_engine, get_session, BetLog, CLVLog
from services.clv_system.models import (
    CLVLineMovement,
    CLVClosingLine,
    Base,
)

log = logging.getLogger(__name__)

STALE_CLOSE_THRESHOLD_HOURS = 6
CONSISTENT_CAPTURE_WINDOW_SEC = 300


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ClosingLineAuditor:
    """Validates closing line data for accuracy and consistency.

    Produces a detailed findings dict with:
      - true_close_pct: % of closing lines that are genuine market closes
      - stale_close_count: number of closes that are likely stale data
      - capture_consistency_pct: % of closes captured within consistent window
      - cross_source_match_pct: % of closes matching across books
      - issues: list of human-readable issue descriptions
      - score: 0-100 composite closing line quality score
    """

    def __init__(self, sport: str = "GOLF", db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)

    def _session(self):
        return get_session(self._db_path)

    def audit(self) -> Dict[str, Any]:
        """Run all closing line checks and return consolidated results."""
        staleness = self._check_close_staleness()
        consistency = self._check_capture_consistency()
        cross_source = self._check_cross_source_agreement()
        movement = self._check_close_vs_last_movement()

        all_issues: List[str] = []
        all_issues.extend(staleness.get("issues", []))
        all_issues.extend(consistency.get("issues", []))
        all_issues.extend(cross_source.get("issues", []))
        all_issues.extend(movement.get("issues", []))

        true_close_pct = staleness.get("true_close_pct", 100.0)
        consistency_pct = consistency.get("consistency_pct", 100.0)
        cross_match_pct = cross_source.get("match_pct", 100.0)
        movement_pct = movement.get("active_market_pct", 100.0)

        score = max(0.0, min(100.0,
            true_close_pct * 0.35
            + consistency_pct * 0.25
            + cross_match_pct * 0.20
            + movement_pct * 0.20
        ))

        return {
            "score": round(score, 1),
            "true_close_pct": round(true_close_pct, 1),
            "stale_close_count": staleness.get("stale_count", 0),
            "total_closing_lines": staleness.get("total", 0),
            "capture_consistency_pct": round(consistency_pct, 1),
            "capture_std_dev_sec": round(consistency.get("std_dev_sec", 0.0), 1),
            "cross_source_match_pct": round(cross_match_pct, 1),
            "cross_source_checked": cross_source.get("checked", 0),
            "active_market_pct": round(movement_pct, 1),
            "issues": all_issues,
        }

    def _check_close_staleness(self) -> Dict[str, Any]:
        """Check if closing lines are true market closes or stale data."""
        session = self._session()
        try:
            issues = []
            closing_lines = (
                session.query(CLVClosingLine)
                .filter(CLVClosingLine.sport == self.sport)
                .all()
            )

            total = len(closing_lines)
            stale_count = 0
            true_count = 0

            for cl in closing_lines:
                last_movement = (
                    session.query(CLVLineMovement)
                    .filter(
                        CLVLineMovement.sport == self.sport,
                        CLVLineMovement.player == cl.player,
                        CLVLineMovement.market_type == cl.market_type,
                        CLVLineMovement.book == cl.book,
                        CLVLineMovement.timestamp < cl.captured_at,
                    )
                    .order_by(CLVLineMovement.timestamp.desc())
                    .first()
                )

                if last_movement:
                    hours_since = (
                        cl.captured_at - last_movement.timestamp
                    ).total_seconds() / 3600
                    if hours_since > STALE_CLOSE_THRESHOLD_HOURS:
                        stale_count += 1
                    else:
                        true_count += 1
                else:
                    true_count += 1

            true_close_pct = (true_count / total * 100) if total > 0 else 100.0

            if stale_count > 0:
                issues.append(
                    f"STALE_CLOSES: {stale_count}/{total} closing lines had no "
                    f"line movement within {STALE_CLOSE_THRESHOLD_HOURS} hours "
                    f"before capture — likely stale cached data, not true closes"
                )

            return {
                "true_close_pct": true_close_pct,
                "stale_count": stale_count,
                "true_count": true_count,
                "total": total,
                "issues": issues,
            }
        finally:
            session.close()

    def _check_capture_consistency(self) -> Dict[str, Any]:
        """Check if closing lines are consistently captured at first tee time."""
        session = self._session()
        try:
            issues = []
            closing_lines = (
                session.query(CLVClosingLine)
                .filter(
                    CLVClosingLine.sport == self.sport,
                    CLVClosingLine.event_start_time.isnot(None),
                    CLVClosingLine.captured_at.isnot(None),
                )
                .all()
            )

            if not closing_lines:
                return {"consistency_pct": 100.0, "std_dev_sec": 0.0, "issues": []}

            deltas = []
            for cl in closing_lines:
                delta = (cl.captured_at - cl.event_start_time).total_seconds()
                deltas.append(delta)

            if not deltas:
                return {"consistency_pct": 100.0, "std_dev_sec": 0.0, "issues": []}

            import numpy as np
            mean_delta = float(np.mean(deltas))
            std_delta = float(np.std(deltas))

            within_window = sum(
                1 for d in deltas if abs(d) <= CONSISTENT_CAPTURE_WINDOW_SEC
            )
            consistency_pct = (within_window / len(deltas) * 100) if deltas else 100.0

            if std_delta > 600:
                issues.append(
                    f"INCONSISTENT_CAPTURE_TIMING: closing line capture has "
                    f"std dev of {std_delta:.0f}s ({std_delta/60:.1f} min) — "
                    f"should be consistently at first tee time"
                )

            if abs(mean_delta) > 600:
                direction = "before" if mean_delta < 0 else "after"
                issues.append(
                    f"CAPTURE_OFFSET: closing lines captured on average "
                    f"{abs(mean_delta)/60:.1f} min {direction} first tee time"
                )

            return {
                "consistency_pct": consistency_pct,
                "std_dev_sec": std_delta,
                "mean_delta_sec": mean_delta,
                "within_window": within_window,
                "total_checked": len(deltas),
                "issues": issues,
            }
        finally:
            session.close()

    def _check_cross_source_agreement(self) -> Dict[str, Any]:
        """Compare closing lines across multiple books for the same event."""
        session = self._session()
        try:
            issues = []
            closing_lines = (
                session.query(CLVClosingLine)
                .filter(CLVClosingLine.sport == self.sport)
                .all()
            )

            groups: Dict[str, List[CLVClosingLine]] = defaultdict(list)
            for cl in closing_lines:
                key = f"{cl.event_id}:{cl.player}:{cl.market_type}"
                groups[key].append(cl)

            multi_book_groups = {k: v for k, v in groups.items() if len(v) >= 2}

            if not multi_book_groups:
                return {"match_pct": 100.0, "checked": 0, "issues": []}

            matching = 0
            mismatching = 0

            for key, cls_list in multi_book_groups.items():
                lines = [cl.closing_line for cl in cls_list]
                line_range = max(lines) - min(lines)
                if line_range <= 2.0:
                    matching += 1
                else:
                    mismatching += 1

            total = matching + mismatching
            match_pct = (matching / total * 100) if total > 0 else 100.0

            if mismatching > 0:
                issues.append(
                    f"CROSS_SOURCE_DISAGREEMENT: {mismatching}/{total} "
                    f"player/market/event groups have closing lines that "
                    f"differ by >2 points across books"
                )

            return {
                "match_pct": match_pct,
                "matching": matching,
                "mismatching": mismatching,
                "checked": total,
                "issues": issues,
            }
        finally:
            session.close()

    def _check_close_vs_last_movement(self) -> Dict[str, Any]:
        """Check if the market was actually active near closing time."""
        session = self._session()
        try:
            issues = []
            closing_lines = (
                session.query(CLVClosingLine)
                .filter(CLVClosingLine.sport == self.sport)
                .all()
            )

            if not closing_lines:
                return {"active_market_pct": 100.0, "issues": []}

            active_count = 0
            inactive_count = 0

            for cl in closing_lines:
                recent_movements = (
                    session.query(sa_func.count(CLVLineMovement.id))
                    .filter(
                        CLVLineMovement.sport == self.sport,
                        CLVLineMovement.player == cl.player,
                        CLVLineMovement.market_type == cl.market_type,
                        CLVLineMovement.timestamp >= cl.captured_at - timedelta(hours=4),
                        CLVLineMovement.timestamp <= cl.captured_at,
                    )
                    .scalar()
                ) or 0

                if recent_movements >= 2:
                    active_count += 1
                else:
                    inactive_count += 1

            total = active_count + inactive_count
            active_pct = (active_count / total * 100) if total > 0 else 100.0

            if inactive_count > 0:
                issues.append(
                    f"INACTIVE_MARKETS: {inactive_count}/{total} closing lines "
                    f"had fewer than 2 line observations in the 4 hours before "
                    f"close — market may not have been liquid"
                )

            return {
                "active_market_pct": active_pct,
                "active_count": active_count,
                "inactive_count": inactive_count,
                "issues": issues,
            }
        finally:
            session.close()
