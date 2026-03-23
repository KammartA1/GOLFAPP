"""Fatigue Source — Travel schedule, consecutive starts.

Golf has no off-days between events. Players who play many consecutive
weeks show measurable performance decline. Travel across time zones
adds additional fatigue. This source captures the cumulative fatigue effect.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from edge_analysis.edge_sources import EdgeSource

logger = logging.getLogger(__name__)


class FatigueSource(EdgeSource):
    """Model fatigue from consecutive starts and travel schedule."""

    name = "fatigue"
    category = "informational"
    description = "Consecutive starts, travel distance, and schedule density fatigue effects"

    def get_signal(self, player: str, tournament_context: Dict[str, Any]) -> float:
        """Signal based on player's fatigue level.

        Negative = player is fatigued (expect underperformance).
        Positive = well-rested (expect better than baseline).
        """
        schedule = tournament_context.get("player_schedule", {})

        consecutive_starts = schedule.get("consecutive_starts", 0)
        weeks_since_break = schedule.get("weeks_since_break", 0)
        travel_miles_last_2w = schedule.get("travel_miles_2w", 0)
        timezone_changes = schedule.get("timezone_changes", 0)
        events_last_6w = schedule.get("events_last_6w", 0)

        fatigue_score = 0.0

        # Consecutive starts: penalty increases non-linearly
        if consecutive_starts >= 5:
            fatigue_score -= 0.35 + (consecutive_starts - 5) * 0.1
        elif consecutive_starts >= 4:
            fatigue_score -= 0.20
        elif consecutive_starts >= 3:
            fatigue_score -= 0.10
        elif consecutive_starts <= 1:
            fatigue_score += 0.05  # Well-rested bonus

        # Travel fatigue
        if travel_miles_last_2w > 5000:
            fatigue_score -= 0.15
        elif travel_miles_last_2w > 3000:
            fatigue_score -= 0.08

        # Timezone changes
        if timezone_changes >= 3:
            fatigue_score -= 0.12
        elif timezone_changes >= 2:
            fatigue_score -= 0.06

        # Schedule density
        if events_last_6w >= 5:
            fatigue_score -= 0.10
        elif events_last_6w <= 2:
            fatigue_score += 0.05

        # Week off before a major: many players rest the week before
        is_major_week = tournament_context.get("event_type", "regular") == "major"
        if is_major_week and consecutive_starts == 0:
            fatigue_score += 0.10  # Fresh for the major

        return round(float(np.clip(fatigue_score, -1.5, 0.5)), 4)

    def get_mechanism(self) -> str:
        return (
            "Golf performance degrades measurably with consecutive starts. "
            "Research shows ~0.1-0.2 SG decline per consecutive week after week 3. "
            "Markets partially price this (odds shorten for rested players) but "
            "underweight the cumulative effect of 4+ consecutive starts, especially "
            "combined with transcontinental travel."
        )

    def get_decay_risk(self) -> str:
        return (
            "low — Fatigue is a real physiological effect that cannot be arbitraged "
            "away. The challenge is that schedule data is public, but quantifying "
            "the exact fatigue penalty requires historical analysis that most "
            "market participants do not perform."
        )

    def validate(self, historical_data: List[Dict[str, Any]]) -> dict:
        if len(historical_data) < 30:
            return {"is_valid": False, "reason": "insufficient data", "n_samples": len(historical_data)}

        signals = []
        outcomes = []
        for rec in historical_data:
            sig = self.get_signal(rec.get("player", ""), rec)
            signals.append(sig)
            outcomes.append(rec.get("actual_finish_pct", 0.5))

        sig_arr = np.array(signals)
        out_arr = np.array(outcomes)

        # Focus on fatigued entries (negative signals)
        mask = sig_arr < -0.1
        if mask.sum() < 10:
            return {"is_valid": False, "reason": "insufficient fatigue instances", "n_samples": int(mask.sum())}

        fatigued_outcomes = out_arr[mask]
        rested_outcomes = out_arr[~mask]

        # Fatigued players should have lower finish percentile
        fatigue_effect = float(np.mean(rested_outcomes) - np.mean(fatigued_outcomes))

        from scipy import stats
        t_stat, p_val = stats.ttest_ind(rested_outcomes, fatigued_outcomes)

        corr = float(np.corrcoef(sig_arr, out_arr)[0, 1]) if len(sig_arr) > 2 else 0.0
        returns = sig_arr * (out_arr - 0.5)
        sharpe = float(np.mean(returns) / max(np.std(returns), 0.001))

        return {
            "is_valid": fatigue_effect > 0.02 and p_val < 0.15,
            "sharpe": round(sharpe, 4),
            "hit_rate": 0.0,
            "fatigue_effect": round(fatigue_effect, 4),
            "n_samples": len(historical_data),
            "n_fatigued": int(mask.sum()),
            "p_value": round(float(p_val), 4),
        }
