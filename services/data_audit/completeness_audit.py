"""Completeness audit — Check data completeness across all tables."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from database.connection import DatabaseManager
from database.models import (
    Player, Event, Bet, LineMovement, Signal, SGStats,
    TournamentResult, Projection, WeatherData,
)

logger = logging.getLogger(__name__)


class CompletenessAuditor:
    """Audit data completeness across all database tables."""

    def __init__(self, sport: str = "golf"):
        self.sport = sport

    def audit(self) -> dict:
        """Run completeness checks on all critical tables."""
        with DatabaseManager.session_scope() as session:
            results = {}

            # Players
            n_players = session.query(Player).filter(Player.sport == self.sport).count()
            n_active = session.query(Player).filter(Player.sport == self.sport, Player.active == True).count()
            n_with_rank = session.query(Player).filter(
                Player.sport == self.sport, Player.world_rank.isnot(None)
            ).count()
            results["players"] = {
                "total": n_players,
                "active": n_active,
                "with_ranking": n_with_rank,
                "completeness": round(n_with_rank / max(n_players, 1), 4),
            }

            # Events
            n_events = session.query(Event).filter(Event.sport == self.sport).count()
            n_with_course = session.query(Event).filter(
                Event.sport == self.sport, Event.course_name.isnot(None)
            ).count()
            results["events"] = {
                "total": n_events,
                "with_course_info": n_with_course,
                "completeness": round(n_with_course / max(n_events, 1), 4),
            }

            # SG Stats
            n_sg = session.query(SGStats).count()
            n_complete_sg = session.query(SGStats).filter(
                SGStats.sg_total.isnot(None),
                SGStats.sg_ott.isnot(None),
                SGStats.sg_app.isnot(None),
                SGStats.sg_putt.isnot(None),
            ).count()
            results["sg_stats"] = {
                "total": n_sg,
                "complete_records": n_complete_sg,
                "completeness": round(n_complete_sg / max(n_sg, 1), 4),
            }

            # Bets
            n_bets = session.query(Bet).filter(Bet.sport == self.sport).count()
            n_settled = session.query(Bet).filter(
                Bet.sport == self.sport, Bet.status.in_(["won", "lost"])
            ).count()
            results["bets"] = {
                "total": n_bets,
                "settled": n_settled,
                "pending": n_bets - n_settled,
            }

            # Projections
            n_proj = session.query(Projection).count()
            results["projections"] = {"total": n_proj}

            # Weather
            n_weather = session.query(WeatherData).count()
            results["weather"] = {"total": n_weather}

            # Lines
            cutoff = datetime.utcnow() - timedelta(days=7)
            n_recent_lines = session.query(LineMovement).filter(
                LineMovement.sport == self.sport,
                LineMovement.captured_at >= cutoff,
            ).count()
            results["line_movements"] = {"recent_7d": n_recent_lines}

        # Overall completeness score
        scores = [
            v.get("completeness", 1.0) for v in results.values()
            if isinstance(v, dict) and "completeness" in v
        ]
        overall = float(np.mean(scores)) * 100 if scores else 0.0

        import numpy as np
        overall = float(np.mean(scores)) * 100 if scores else 0.0

        results["overall_score"] = round(overall, 1)
        return results
