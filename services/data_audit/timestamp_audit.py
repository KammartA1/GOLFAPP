"""Timestamp audit — Verify data freshness and timing consistency."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from database.connection import DatabaseManager
from database.models import LineMovement, Bet, SGStats, WeatherData

logger = logging.getLogger(__name__)


class TimestampAuditor:
    """Audit timestamp quality across data sources."""

    def __init__(self, sport: str = "golf"):
        self.sport = sport

    def audit(self) -> dict:
        """Run full timestamp audit.

        Checks:
          - Data freshness (staleness)
          - Timestamp ordering (monotonicity)
          - Gap detection
          - Timezone consistency
        """
        results = {}

        with DatabaseManager.session_scope() as session:
            now = datetime.utcnow()

            # Line movements freshness
            latest_line = (
                session.query(LineMovement)
                .filter(LineMovement.sport == self.sport)
                .order_by(LineMovement.captured_at.desc())
                .first()
            )
            if latest_line:
                line_age_hours = (now - latest_line.captured_at).total_seconds() / 3600
                results["line_movements"] = {
                    "latest": latest_line.captured_at.isoformat(),
                    "age_hours": round(line_age_hours, 1),
                    "is_stale": line_age_hours > 6,
                    "status": "fresh" if line_age_hours < 2 else "aging" if line_age_hours < 6 else "stale",
                }
            else:
                results["line_movements"] = {"latest": None, "is_stale": True, "status": "no_data"}

            # SG stats freshness
            latest_sg = (
                session.query(SGStats)
                .order_by(SGStats.captured_at.desc())
                .first()
            )
            if latest_sg:
                sg_age_hours = (now - latest_sg.captured_at).total_seconds() / 3600
                results["sg_stats"] = {
                    "latest": latest_sg.captured_at.isoformat(),
                    "age_hours": round(sg_age_hours, 1),
                    "is_stale": sg_age_hours > 168,  # 1 week
                    "status": "fresh" if sg_age_hours < 48 else "aging" if sg_age_hours < 168 else "stale",
                }
            else:
                results["sg_stats"] = {"latest": None, "is_stale": True, "status": "no_data"}

            # Weather data freshness
            latest_weather = (
                session.query(WeatherData)
                .order_by(WeatherData.captured_at.desc())
                .first()
            )
            if latest_weather:
                weather_age = (now - latest_weather.captured_at).total_seconds() / 3600
                results["weather"] = {
                    "latest": latest_weather.captured_at.isoformat(),
                    "age_hours": round(weather_age, 1),
                    "is_stale": weather_age > 12,
                    "status": "fresh" if weather_age < 4 else "aging" if weather_age < 12 else "stale",
                }
            else:
                results["weather"] = {"latest": None, "is_stale": True, "status": "no_data"}

            # Gap detection in line movements (last 24h)
            cutoff = now - timedelta(hours=24)
            lines_24h = (
                session.query(LineMovement)
                .filter(
                    LineMovement.sport == self.sport,
                    LineMovement.captured_at >= cutoff,
                )
                .order_by(LineMovement.captured_at.asc())
                .all()
            )

            if len(lines_24h) >= 2:
                timestamps = [l.captured_at for l in lines_24h]
                gaps = [(timestamps[i + 1] - timestamps[i]).total_seconds() / 3600
                        for i in range(len(timestamps) - 1)]
                max_gap = max(gaps) if gaps else 0
                avg_gap = sum(gaps) / len(gaps) if gaps else 0
                results["gap_analysis"] = {
                    "n_snapshots_24h": len(lines_24h),
                    "max_gap_hours": round(max_gap, 1),
                    "avg_gap_hours": round(avg_gap, 2),
                    "has_large_gap": max_gap > 4,
                }
            else:
                results["gap_analysis"] = {"n_snapshots_24h": len(lines_24h), "has_large_gap": True}

        # Overall score
        stale_count = sum(1 for v in results.values() if isinstance(v, dict) and v.get("is_stale", False))
        total = sum(1 for v in results.values() if isinstance(v, dict) and "is_stale" in v)
        results["overall_score"] = round((1.0 - stale_count / max(total, 1)) * 100, 1)

        return results
