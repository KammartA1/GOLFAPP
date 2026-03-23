"""Driving Distance Source — Distance advantage on specific layouts.

Certain courses provide outsized reward for distance off the tee.
This source identifies when a player's driving distance creates
a structural advantage on a specific layout.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from edge_analysis.edge_sources import EdgeSource

logger = logging.getLogger(__name__)


class DrivingDistanceSource(EdgeSource):
    """Distance advantage on layouts that reward length."""

    name = "driving_distance"
    category = "predictive"
    description = "Driving distance advantage on layouts where length creates scoring separation"

    def get_signal(self, player: str, tournament_context: Dict[str, Any]) -> float:
        """Signal based on driving distance advantage at this course.

        Positive = player's distance creates advantage on this layout.
        """
        driving_dist = tournament_context.get("driving_distance", 290.0)
        field_avg_dist = tournament_context.get("field_avg_driving_distance", 293.0)
        course = tournament_context.get("course_profile", {})

        course_length = course.get("total_yardage", 7200)
        n_par5 = course.get("n_par5", 4)
        n_driveable_par4 = course.get("n_driveable_par4", 0)
        avg_par4_length = course.get("avg_par4_length", 440)
        fairway_width = course.get("avg_fairway_width", 30)

        # Distance advantage in yards
        dist_advantage = driving_dist - field_avg_dist

        # Course distance sensitivity: how much does extra distance matter here?
        sensitivity = 0.0

        # Long courses reward distance more
        if course_length > 7400:
            sensitivity += 0.4
        elif course_length > 7200:
            sensitivity += 0.2

        # Many par-5s = more reachable in two for long hitters
        if n_par5 >= 5:
            sensitivity += 0.3
        elif n_par5 >= 4:
            sensitivity += 0.15

        # Driveable par-4s are huge for long hitters
        sensitivity += n_driveable_par4 * 0.15

        # Long par-4s: distance helps reach in regulation
        if avg_par4_length > 460:
            sensitivity += 0.2

        # Wide fairways: bombers can let it rip without penalty
        if fairway_width > 35:
            sensitivity += 0.1
        elif fairway_width < 25:
            sensitivity -= 0.15  # Narrow fairways penalize bombers

        # Convert to signal
        # Each yard of distance advantage * course sensitivity
        signal = (dist_advantage / 10.0) * sensitivity

        # Altitude adjustment: ball flies further at altitude, reducing distance advantage
        elevation = course.get("elevation", 0)
        if elevation > 4000:
            signal *= 0.7  # Distance advantage is worth less at altitude

        return round(float(np.clip(signal, -2.0, 2.0)), 4)

    def get_mechanism(self) -> str:
        return (
            "Distance off the tee creates non-linear advantages on certain layouts. "
            "Reaching par-5s in two gives birdie/eagle opportunities. Driving the green "
            "on short par-4s creates massive scoring edges. Markets price this partially "
            "through overall rankings but underweight the course-specific interaction "
            "between a player's distance and the specific layout demands."
        )

    def get_decay_risk(self) -> str:
        return (
            "medium — Distance data is publicly available, but the course-specific "
            "sensitivity model requires detailed hole-by-hole analysis. As the tour "
            "moves to longer courses and equipment advances, the landscape changes."
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
        corr = float(np.corrcoef(sig_arr, out_arr)[0, 1]) if len(sig_arr) > 2 else 0.0
        returns = sig_arr * (out_arr - 0.5)
        sharpe = float(np.mean(returns) / max(np.std(returns), 0.001))
        hit_rate = float(np.mean((sig_arr > 0) == (out_arr > 0.5)))

        from scipy import stats
        _, p_val = stats.pearsonr(sig_arr, out_arr) if len(sig_arr) >= 10 else (0, 1.0)

        return {
            "is_valid": corr > 0.03,
            "sharpe": round(sharpe, 4),
            "hit_rate": round(hit_rate, 4),
            "correlation": round(corr, 4),
            "n_samples": len(historical_data),
            "p_value": round(float(p_val), 4),
        }
