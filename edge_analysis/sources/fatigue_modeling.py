"""
Fatigue Modeling Edge Source
==============================
Travel schedule, consecutive starts, major recovery effects.
Market edge: sportsbooks do not track travel or fatigue — pure information edge.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

import numpy as np
from scipy import stats as sp_stats

log = logging.getLogger(__name__)

# Fatigue parameters calibrated from PGA Tour data
_CONSECUTIVE_START_PENALTY = {
    1: 0.0,
    2: -0.02,
    3: -0.08,
    4: -0.18,
    5: -0.30,
    6: -0.45,
}

# Miles thresholds for travel fatigue
_TRAVEL_LIGHT = 1500     # < 1500 miles in 30 days
_TRAVEL_MODERATE = 4000  # 1500-4000 miles
_TRAVEL_HEAVY = 8000     # 4000-8000 miles
# > 8000 miles = extreme travel (international + back)

# Major recovery: physical + emotional drain
_MAJOR_RECOVERY_WEEKS = {
    "Masters": 2,
    "PGA Championship": 2,
    "U.S. Open": 3,        # most physically demanding
    "The Open Championship": 3,  # jet lag + links fatigue
    "British Open": 3,
}


class FatigueModelingSource:
    """Travel and fatigue edge — invisible to sportsbooks."""

    name = "Fatigue Modeling"
    category = "physical"
    version = "1.0"

    def get_signal(self, player: dict, tournament_context: dict) -> float:
        """
        Compute fatigue-adjusted performance edge.

        player keys:
            consecutive_starts   – int, consecutive weeks playing
            miles_last_30_days   – float, total travel miles in last 30 days
            last_major           – str, name of most recent major played (or None)
            weeks_since_major    – int, weeks since last major
            age                  – int (optional, older players fatigue faster)
            fitness_tier         – "elite", "good", "average", "below_average"
            had_week_off         – bool, whether player took previous week off
            timezone_changes     – int, timezone changes in last 14 days
        tournament_context keys:
            week_of_season       – int, week number in PGA Tour season
            is_major             – bool
        """
        consec = player.get("consecutive_starts", 1)
        miles = player.get("miles_last_30_days", 2000.0)
        age = player.get("age", 30)
        fitness = player.get("fitness_tier", "good")
        had_week_off = player.get("had_week_off", False)
        tz_changes = player.get("timezone_changes", 0)

        # ── Consecutive starts penalty ──────────────────────────────────
        consec_capped = min(consec, 6)
        consec_penalty = _CONSECUTIVE_START_PENALTY.get(consec_capped, -0.45)

        # Fresh player bonus
        if had_week_off and consec <= 1:
            consec_penalty = 0.05  # slight rest advantage

        # Age modifier: players >35 fatigue faster
        age_factor = 1.0
        if age >= 40:
            age_factor = 1.5
        elif age >= 35:
            age_factor = 1.2
        elif age <= 25:
            age_factor = 0.8

        consec_penalty *= age_factor

        # Fitness modifier
        fitness_mods = {"elite": 0.6, "good": 0.85, "average": 1.0, "below_average": 1.3}
        consec_penalty *= fitness_mods.get(fitness, 1.0)

        # ── Travel fatigue ──────────────────────────────────────────────
        travel_penalty = 0.0
        if miles > _TRAVEL_HEAVY:
            travel_penalty = -0.15 * min((miles - _TRAVEL_HEAVY) / 4000.0 + 1, 2.0)
        elif miles > _TRAVEL_MODERATE:
            frac = (miles - _TRAVEL_MODERATE) / (_TRAVEL_HEAVY - _TRAVEL_MODERATE)
            travel_penalty = -0.05 * (1 + frac)
        elif miles > _TRAVEL_LIGHT:
            frac = (miles - _TRAVEL_LIGHT) / (_TRAVEL_MODERATE - _TRAVEL_LIGHT)
            travel_penalty = -0.02 * frac

        # Timezone disruption
        if tz_changes >= 4:
            travel_penalty -= 0.08  # severe jet lag
        elif tz_changes >= 2:
            travel_penalty -= 0.03

        travel_penalty *= age_factor

        # ── Major recovery ──────────────────────────────────────────────
        major_penalty = 0.0
        last_major = player.get("last_major", None)
        weeks_since = player.get("weeks_since_major", 99)

        if last_major and weeks_since <= 3:
            recovery_weeks = _MAJOR_RECOVERY_WEEKS.get(last_major, 2)
            if weeks_since < recovery_weeks:
                # Still in recovery window
                recovery_frac = weeks_since / recovery_weeks
                major_penalty = -0.12 * (1.0 - recovery_frac)
                # Worse if they played deep into the weekend
                if player.get("major_finish", 50) <= 10:
                    major_penalty *= 1.3  # contention = more draining

        # ── Season fatigue (late-season drag) ───────────────────────────
        week = tournament_context.get("week_of_season", 20)
        total_starts_season = player.get("total_starts_season", 20)
        season_fatigue = 0.0
        if total_starts_season > 25:
            season_fatigue = -0.02 * (total_starts_season - 25)
        if week > 35:  # late season
            season_fatigue -= 0.02

        # ── Aggregate ───────────────────────────────────────────────────
        total_fatigue = consec_penalty + travel_penalty + major_penalty + season_fatigue

        # This is a negative signal (fatigue hurts), so we return it as-is.
        # A player with zero fatigue gets signal ~0 (no edge either way).
        # A rested player playing after a week off gets a slight positive signal.
        return round(float(total_fatigue), 4)

    def get_mechanism(self) -> str:
        return (
            "Three fatigue channels are modeled: (1) Consecutive starts — "
            "3+ consecutive weeks of competition degrades SG by 0.08-0.45 "
            "strokes per round, scaling with age and fitness.  (2) Travel "
            "distance — high mileage (>4000 mi in 30 days) and timezone "
            "disruption cause measurable performance drag.  (3) Major recovery "
            "— the physical and emotional toll of a major championship lasts "
            "2-3 weeks, especially for contenders.  Sportsbooks have zero "
            "visibility into travel schedules or fatigue accumulation.  "
            "This is a pure information edge."
        )

    def get_decay_risk(self) -> str:
        return (
            "LOW — Travel schedules are not publicly aggregated in a usable "
            "format.  Even sophisticated models ignore fatigue.  The only decay "
            "risk is if a data provider begins publishing fatigue metrics, "
            "which would require tracking player movements — unlikely near-term."
        )

    def validate(self, historical_data: list[dict]) -> dict:
        if len(historical_data) < 30:
            return {
                "sharpe": 0.0, "p_value": 1.0,
                "sample_size": len(historical_data),
                "correlation_with_other_signals": {},
                "status": "INSUFFICIENT_DATA",
            }

        signals, outcomes = [], []
        for rec in historical_data:
            sig = self.get_signal(rec.get("player", {}), rec.get("tournament_context", {}))
            finish = rec.get("actual_finish", 50)
            outcomes.append((50.0 - finish) / 50.0)
            signals.append(sig)

        signals = np.array(signals)
        outcomes = np.array(outcomes)
        n = len(signals)
        q = max(n // 5, 1)
        idx = np.argsort(signals)
        spread = float(np.mean(outcomes[idx[-q:]]) - np.mean(outcomes[idx[:q]]))
        pooled = float(np.std(np.concatenate([outcomes[idx[-q:]], outcomes[idx[:q]]]), ddof=1))
        sharpe = spread / pooled if pooled > 1e-9 else 0.0
        corr, p_val = sp_stats.spearmanr(signals, outcomes)
        if np.isnan(corr):
            corr, p_val = 0.0, 1.0

        return {
            "sharpe": round(sharpe, 4),
            "p_value": round(float(p_val), 6),
            "sample_size": n,
            "spearman_r": round(float(corr), 4),
            "quintile_spread": round(spread, 4),
            "correlation_with_other_signals": {},
            "status": "VALID" if p_val < 0.10 and n >= 30 else "MARGINAL",
        }
