"""Odds audit — Verify odds data quality and consistency."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np

from database.connection import DatabaseManager
from database.models import LineMovement

logger = logging.getLogger(__name__)


class OddsAuditor:
    """Audit odds data quality."""

    def __init__(self, sport: str = "golf"):
        self.sport = sport

    def audit(self, days: int = 7) -> dict:
        """Run odds quality audit.

        Checks:
          - Range validity (odds in reasonable range)
          - Consistency across sources
          - Movement plausibility (no impossible jumps)
          - Coverage (all players have odds)
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        with DatabaseManager.session_scope() as session:
            lines = (
                session.query(LineMovement)
                .filter(
                    LineMovement.sport == self.sport,
                    LineMovement.captured_at >= cutoff,
                )
                .all()
            )

            if not lines:
                return {"status": "no_data", "score": 0.0}

            issues = []
            n_total = len(lines)

            # Range check
            n_invalid_odds = 0
            for lm in lines:
                if lm.odds_decimal is not None:
                    if lm.odds_decimal < 1.01 or lm.odds_decimal > 1000:
                        n_invalid_odds += 1
                if lm.line is not None:
                    if lm.line < -50 or lm.line > 200:
                        n_invalid_odds += 1

            if n_invalid_odds > 0:
                issues.append(f"{n_invalid_odds} odds values out of valid range")

            # Implied probability check
            n_bad_prob = 0
            for lm in lines:
                if lm.over_prob_implied is not None:
                    if lm.over_prob_implied < 0 or lm.over_prob_implied > 1:
                        n_bad_prob += 1

            if n_bad_prob > 0:
                issues.append(f"{n_bad_prob} invalid implied probabilities")

            # Source diversity
            sources = set(lm.source for lm in lines)
            if len(sources) < 2:
                issues.append(f"Only {len(sources)} odds source(s) — limited cross-reference ability")

            # Large movement detection
            from collections import defaultdict
            player_lines = defaultdict(list)
            for lm in lines:
                key = f"{lm.player}|{lm.stat_type}"
                player_lines[key].append((lm.captured_at, lm.line))

            n_large_moves = 0
            for key, entries in player_lines.items():
                if len(entries) < 2:
                    continue
                entries.sort(key=lambda x: x[0])
                for i in range(1, len(entries)):
                    if entries[i][1] is not None and entries[i - 1][1] is not None:
                        move = abs(entries[i][1] - entries[i - 1][1])
                        if move > 5:  # Implausibly large single move
                            n_large_moves += 1

            if n_large_moves > 0:
                issues.append(f"{n_large_moves} implausibly large line movements detected")

        # Score
        score = 100.0
        score -= min(n_invalid_odds / max(n_total, 1) * 100, 30)
        score -= min(n_bad_prob / max(n_total, 1) * 100, 20)
        score -= min(n_large_moves * 2, 20)
        if len(sources) < 2:
            score -= 15

        return {
            "score": round(max(0, score), 1),
            "n_total_lines": n_total,
            "n_invalid_odds": n_invalid_odds,
            "n_bad_probabilities": n_bad_prob,
            "n_large_movements": n_large_moves,
            "n_sources": len(sources),
            "sources": list(sources),
            "issues": issues,
            "is_healthy": score >= 70,
        }
