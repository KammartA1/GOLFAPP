"""Course Fit Source — Correlate SG profile with course demands.

Goes beyond basic SG decomposition to model specific course features:
  - Fairway width vs driving accuracy
  - Green size/firmness vs approach accuracy
  - Bermuda vs bentgrass putting
  - Elevation changes
  - Par-3/Par-5 scoring requirements
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from edge_analysis.edge_sources import EdgeSource

logger = logging.getLogger(__name__)


class CourseFitSource(EdgeSource):
    """Correlate player SG profile with specific course feature demands."""

    name = "course_fit"
    category = "predictive"
    description = "Match player SG profile to granular course features beyond basic type"

    def get_signal(self, player: str, tournament_context: Dict[str, Any]) -> float:
        """Signal based on detailed course-player fit.

        Considers: fairway width, green size, surface type, par distribution.
        """
        sg = tournament_context.get("player_sg", {})
        course = tournament_context.get("course_profile", {})

        # Course features (defaults for balanced course)
        fairway_width = course.get("avg_fairway_width", 30)  # yards
        green_size = course.get("avg_green_size", 5500)  # sq ft
        is_bermuda = course.get("is_bermuda", False)
        n_par5 = course.get("n_par5", 4)
        n_par3 = course.get("n_par3", 4)
        avg_par4_length = course.get("avg_par4_length", 440)  # yards
        elevation = course.get("elevation", 0)  # feet

        # Player SG components
        sg_ott = sg.get("sg_ott", 0.0)
        sg_app = sg.get("sg_app", 0.0)
        sg_atg = sg.get("sg_atg", 0.0)
        sg_putt = sg.get("sg_putt", 0.0)

        # Player stats
        driving_dist = tournament_context.get("driving_distance", 290)
        driving_acc = tournament_context.get("driving_accuracy", 0.60)
        gir = tournament_context.get("gir_pct", 0.65)
        scrambling = tournament_context.get("scrambling_pct", 0.60)

        signal = 0.0

        # Narrow fairways penalize bombers who miss fairways
        if fairway_width < 28:
            # Tight course — accuracy matters more
            signal += driving_acc * 0.5 - (1 - driving_acc) * sg_ott * 0.3
        elif fairway_width > 35:
            # Wide open — bombers thrive
            signal += sg_ott * 0.4

        # Small greens reward approach accuracy
        if green_size < 5000:
            signal += sg_app * 0.5
        elif green_size > 6500:
            # Large greens — less separation on approach
            signal += sg_app * 0.2

        # Bermuda greens require specific putting skill
        if is_bermuda:
            bermuda_skill = tournament_context.get("bermuda_putting_sg", sg_putt * 0.8)
            signal += bermuda_skill * 0.3
        else:
            signal += sg_putt * 0.25

        # Par-5 heavy courses reward bombers
        if n_par5 >= 5:
            signal += sg_ott * 0.2 + (driving_dist - 290) * 0.005

        # Short par-4s and many par-3s reward precision
        if n_par3 >= 5 or avg_par4_length < 420:
            signal += sg_app * 0.3

        # Long courses reward distance
        if avg_par4_length > 460:
            signal += (driving_dist - 290) * 0.008

        # Altitude: ball flies further, benefits distance players less
        if elevation > 4000:
            signal -= (driving_dist - 290) * 0.003  # Distance advantage reduced

        # Scrambling at courses with tough greens complexes
        if course.get("scrambling_difficulty", 0.5) > 0.6:
            signal += scrambling * 0.3 + sg_atg * 0.3

        return round(float(np.clip(signal, -3.0, 3.0)), 4)

    def get_mechanism(self) -> str:
        return (
            "Markets price players on overall skill level but underweight specific "
            "course-player interactions. A mid-ranked player who happens to be an "
            "elite bermuda putter at a bermuda course is systematically undervalued. "
            "This source captures granular course feature matching beyond basic SG."
        )

    def get_decay_risk(self) -> str:
        return (
            "low — Course-specific feature matching requires detailed proprietary "
            "course data and player skill profiles that most bettors do not have. "
            "The granularity of this source provides durable edge."
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
            "is_valid": corr > 0.05 and hit_rate > 0.52,
            "sharpe": round(sharpe, 4),
            "hit_rate": round(hit_rate, 4),
            "correlation": round(corr, 4),
            "n_samples": len(historical_data),
            "p_value": round(float(p_val), 4),
        }
