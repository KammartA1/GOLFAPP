"""Course History Source — Venue-specific over/under-performance.

Some players consistently outperform their baseline at specific venues.
This goes beyond course-fit modeling — it captures intangible factors like
comfort level, local knowledge, accommodation quality, and psychological
associations with past success.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
from scipy import stats as scipy_stats

from edge_analysis.edge_sources import EdgeSource

logger = logging.getLogger(__name__)


class CourseHistorySource(EdgeSource):
    """Venue-specific over/under-performance beyond course-fit modeling."""

    name = "course_history"
    category = "predictive"
    description = "Venue-specific performance history — horse-for-course effects"

    def get_signal(self, player: str, tournament_context: Dict[str, Any]) -> float:
        """Signal based on player's specific venue history.

        Positive = player historically outperforms at this venue.
        """
        course_history = tournament_context.get("player_course_history", {})
        n_visits = course_history.get("n_visits", 0)

        if n_visits < 2:
            return 0.0  # Insufficient venue history

        # SG differential at this venue vs overall baseline
        venue_sg_diff = course_history.get("venue_sg_differential", 0.0)
        venue_avg_finish_pct = course_history.get("avg_finish_percentile", 0.5)
        overall_avg_finish_pct = course_history.get("overall_avg_finish_percentile", 0.5)
        venue_top10_rate = course_history.get("venue_top10_rate", 0.0)
        overall_top10_rate = course_history.get("overall_top10_rate", 0.0)
        made_cut_rate_venue = course_history.get("venue_cut_rate", 0.65)

        # Primary signal: SG differential at venue
        signal = venue_sg_diff * 0.8

        # Secondary: finish percentile differential
        finish_diff = venue_avg_finish_pct - overall_avg_finish_pct
        signal += finish_diff * 1.5

        # Tertiary: top-10 rate differential
        top10_diff = venue_top10_rate - overall_top10_rate
        signal += top10_diff * 2.0

        # Strong venue performers (high cut rate at venue)
        if made_cut_rate_venue > 0.85 and n_visits >= 5:
            signal += 0.15

        # Regression: small samples regress more
        # Effective sample: sqrt(n_visits) / sqrt(10) as confidence factor
        confidence = min(np.sqrt(n_visits) / np.sqrt(10), 1.0)

        # Apply Bayesian shrinkage
        # With 2 visits, mostly baseline. With 10+, mostly venue-specific.
        shrinkage = 1.0 - confidence * 0.6
        signal *= (1.0 - shrinkage)

        # Recent venue history matters more
        recent_venue_sg = course_history.get("last_2_visits_sg_diff", None)
        if recent_venue_sg is not None:
            signal = signal * 0.6 + recent_venue_sg * 0.4

        return round(float(np.clip(signal, -2.0, 2.0)), 4)

    def get_mechanism(self) -> str:
        return (
            "Venue-specific performance (horse-for-course) captures factors beyond "
            "measurable course fit: local knowledge of breaks and grain on greens, "
            "comfort with specific green complexes, positive mental associations from "
            "past success, preferred accommodation and routine. Markets partially price "
            "this for well-known cases (e.g., Rahm at Torrey Pines) but underweight "
            "it for mid-tier players with strong venue histories."
        )

    def get_decay_risk(self) -> str:
        return (
            "medium — Course history data is publicly available (past results), but "
            "the analytical step of extracting venue-specific alpha after controlling "
            "for overall skill requires specialized modeling. Risk increases as "
            "DataGolf and similar tools make this information more accessible."
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

        # Only validate where we have venue history (non-zero signal)
        mask = np.abs(sig_arr) > 0.05
        if mask.sum() < 15:
            return {"is_valid": False, "reason": "insufficient venue events", "n_samples": int(mask.sum())}

        sig_active = sig_arr[mask]
        out_active = out_arr[mask]
        corr = float(np.corrcoef(sig_active, out_active)[0, 1]) if len(sig_active) > 2 else 0.0
        returns = sig_active * (out_active - 0.5)
        sharpe = float(np.mean(returns) / max(np.std(returns), 0.001))
        hit_rate = float(np.mean((sig_active > 0) == (out_active > 0.5)))

        _, p_val = scipy_stats.pearsonr(sig_active, out_active) if len(sig_active) >= 10 else (0, 1.0)

        return {
            "is_valid": corr > 0.04 and hit_rate > 0.52,
            "sharpe": round(sharpe, 4),
            "hit_rate": round(hit_rate, 4),
            "correlation": round(corr, 4),
            "n_samples": int(mask.sum()),
            "p_value": round(float(p_val), 4),
        }
