"""Edge decay model — Track how our edge decays over time."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class EdgeDecayAnalyzer:
    """Analyze and project edge decay over time."""

    def analyze_decay(
        self,
        monthly_clv: List[float],
        monthly_roi: List[float],
    ) -> dict:
        """Analyze edge decay trend from monthly CLV/ROI data.

        Args:
            monthly_clv: CLV in cents per month (oldest first).
            monthly_roi: ROI per month (oldest first).

        Returns:
            Decay analysis with trend, half-life, and projections.
        """
        if len(monthly_clv) < 3:
            return {"has_data": False, "reason": "Need at least 3 months of data"}

        clv_arr = np.array(monthly_clv)
        months = np.arange(len(clv_arr))

        # Linear trend
        slope, intercept, r_val, p_val, std_err = scipy_stats.linregress(months, clv_arr)

        # Exponential decay fit: CLV(t) = a * exp(-lambda * t)
        positive_clv = clv_arr[clv_arr > 0]
        if len(positive_clv) >= 3:
            log_clv = np.log(np.clip(clv_arr, 0.01, None))
            pos_months = months[clv_arr > 0]
            if len(pos_months) >= 3:
                exp_slope, exp_intercept, _, _, _ = scipy_stats.linregress(pos_months, log_clv[clv_arr > 0])
                decay_rate = -exp_slope
                half_life = np.log(2) / max(decay_rate, 0.001) if decay_rate > 0 else 999
            else:
                decay_rate = 0.0
                half_life = 999
        else:
            decay_rate = 0.0
            half_life = 999

        # Months until edge hits zero (linear projection)
        if slope < 0 and intercept > 0:
            months_to_zero = -intercept / slope
        else:
            months_to_zero = 999

        # Current edge estimate (last 3 months average)
        current_edge = float(np.mean(clv_arr[-3:]))

        # 6-month projection
        projected_6m = intercept + slope * (len(clv_arr) + 6)

        # Is edge decaying?
        if slope < -0.1 and p_val < 0.10:
            decay_status = "decaying"
        elif slope < -0.05:
            decay_status = "possibly_decaying"
        elif slope > 0.05:
            decay_status = "improving"
        else:
            decay_status = "stable"

        return {
            "has_data": True,
            "n_months": len(monthly_clv),
            "current_edge_cents": round(current_edge, 2),
            "linear_slope": round(slope, 4),
            "linear_intercept": round(intercept, 2),
            "linear_r_squared": round(r_val ** 2, 4),
            "linear_p_value": round(p_val, 4),
            "decay_rate": round(decay_rate, 4),
            "half_life_months": round(min(half_life, 999), 1),
            "months_to_zero": round(min(months_to_zero, 999), 1),
            "projected_6m_clv": round(projected_6m, 2),
            "decay_status": decay_status,
        }
