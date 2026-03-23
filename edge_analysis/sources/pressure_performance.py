"""Pressure Performance Source — Major championship vs regular tour.

Some players consistently outperform or underperform their baseline
in high-pressure situations (majors, playoff events, Ryder Cup).
This source captures pressure-adjusted performance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from edge_analysis.edge_sources import EdgeSource

logger = logging.getLogger(__name__)

PRESSURE_EVENTS = {
    "major": 1.0,        # Full pressure multiplier
    "players": 0.8,      # Near-major pressure
    "playoff": 0.7,      # FedEx playoff events
    "signature": 0.5,    # Signature events
    "regular": 0.0,      # No pressure effect
    "opposite": -0.2,    # Opposite-field (lower pressure, smaller fields)
}


class PressurePerformanceSource(EdgeSource):
    """Predict pressure-adjusted performance at majors and big events."""

    name = "pressure_performance"
    category = "predictive"
    description = "Major championship and high-pressure event performance differential"

    def get_signal(self, player: str, tournament_context: Dict[str, Any]) -> float:
        """Signal based on player's pressure performance history.

        Positive = player historically outperforms in this pressure level.
        """
        event_type = tournament_context.get("event_type", "regular")
        pressure_level = PRESSURE_EVENTS.get(event_type, 0.0)

        if pressure_level == 0.0:
            return 0.0  # No pressure edge for regular events

        # Player's pressure stats
        pressure_stats = tournament_context.get("player_pressure_stats", {})
        major_sg_diff = pressure_stats.get("major_sg_differential", 0.0)
        big_event_wins = pressure_stats.get("big_event_wins", 0)
        major_top10_rate = pressure_stats.get("major_top10_rate", 0.0)
        regular_top10_rate = pressure_stats.get("regular_top10_rate", 0.0)
        n_majors = pressure_stats.get("n_majors_played", 0)

        if n_majors < 4:
            # Insufficient pressure data — can't assess
            return 0.0

        # Primary signal: SG differential in big events
        signal = major_sg_diff * pressure_level * 1.2

        # Secondary: top-10 rate differential
        top10_diff = major_top10_rate - regular_top10_rate
        signal += top10_diff * pressure_level * 2.0

        # Proven big-event winners get a bonus
        if big_event_wins >= 3:
            signal += 0.15 * pressure_level
        elif big_event_wins >= 1:
            signal += 0.08 * pressure_level

        # Confidence: more major experience = more reliable signal
        confidence = min(n_majors / 20.0, 1.0)
        signal *= confidence

        return round(float(np.clip(signal, -2.0, 2.0)), 4)

    def get_mechanism(self) -> str:
        return (
            "Markets price major championship odds based on overall rankings and "
            "recent form, but underweight player-specific pressure performance. "
            "Some players consistently rise in majors (e.g., Tiger, Jack, Koepka) "
            "while others consistently underperform. This alpha comes from the "
            "difficulty of quantifying 'clutch' performance in thin-market contexts."
        )

    def get_decay_risk(self) -> str:
        return (
            "medium — Pressure performance is increasingly studied, but the signal "
            "requires many years of major championship data per player. With only "
            "4 majors per year, it takes 5+ years to get reliable estimates. This "
            "data constraint preserves the edge."
        )

    def validate(self, historical_data: List[Dict[str, Any]]) -> dict:
        if len(historical_data) < 20:
            return {"is_valid": False, "reason": "insufficient data", "n_samples": len(historical_data)}

        signals = []
        outcomes = []
        for rec in historical_data:
            sig = self.get_signal(rec.get("player", ""), rec)
            signals.append(sig)
            outcomes.append(rec.get("actual_finish_pct", 0.5))

        sig_arr = np.array(signals)
        out_arr = np.array(outcomes)
        mask = np.abs(sig_arr) > 0.05
        if mask.sum() < 10:
            return {"is_valid": False, "reason": "insufficient pressure events", "n_samples": int(mask.sum())}

        sig_active = sig_arr[mask]
        out_active = out_arr[mask]
        corr = float(np.corrcoef(sig_active, out_active)[0, 1]) if len(sig_active) > 2 else 0.0
        returns = sig_active * (out_active - 0.5)
        sharpe = float(np.mean(returns) / max(np.std(returns), 0.001))
        hit_rate = float(np.mean((sig_active > 0) == (out_active > 0.5)))

        from scipy import stats
        _, p_val = stats.pearsonr(sig_active, out_active) if len(sig_active) >= 10 else (0, 1.0)

        return {
            "is_valid": corr > 0.03,
            "sharpe": round(sharpe, 4),
            "hit_rate": round(hit_rate, 4),
            "correlation": round(corr, 4),
            "n_samples": int(mask.sum()),
            "p_value": round(float(p_val), 4),
        }
