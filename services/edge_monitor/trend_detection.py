"""Trend detection — CUSUM change-point detection for edge monitoring."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EdgeTrendDetector:
    """Detect changes in edge using CUSUM and other methods."""

    def __init__(
        self,
        cusum_threshold: float = 5.0,
        cusum_drift: float = 0.5,
    ):
        """
        Args:
            cusum_threshold: CUSUM alarm threshold (h parameter).
            cusum_drift: Allowable drift before signaling (k parameter).
        """
        self.threshold = cusum_threshold
        self.drift = cusum_drift

    def cusum_detection(
        self,
        values: List[float],
        target_mean: float = 0.0,
    ) -> dict:
        """Run CUSUM change-point detection.

        Detects both upward and downward shifts from target mean.

        Args:
            values: Time series of CLV or edge values.
            target_mean: Expected mean (0 for CLV = no edge).

        Returns:
            {
                'change_detected': bool,
                'change_direction': str,
                'change_point_index': int or None,
                'cusum_upper': list,
                'cusum_lower': list,
            }
        """
        if len(values) < 10:
            return {"change_detected": False, "reason": "insufficient data"}

        arr = np.array(values)
        n = len(arr)

        # CUSUM for upward shift
        s_upper = np.zeros(n)
        # CUSUM for downward shift
        s_lower = np.zeros(n)

        upper_alarm = None
        lower_alarm = None

        for i in range(1, n):
            s_upper[i] = max(0, s_upper[i - 1] + (arr[i] - target_mean) - self.drift)
            s_lower[i] = max(0, s_lower[i - 1] - (arr[i] - target_mean) - self.drift)

            if s_upper[i] > self.threshold and upper_alarm is None:
                upper_alarm = i
            if s_lower[i] > self.threshold and lower_alarm is None:
                lower_alarm = i

        change_detected = upper_alarm is not None or lower_alarm is not None

        if upper_alarm is not None and lower_alarm is not None:
            change_direction = "up" if upper_alarm < lower_alarm else "down"
            change_idx = min(upper_alarm, lower_alarm)
        elif upper_alarm is not None:
            change_direction = "up"
            change_idx = upper_alarm
        elif lower_alarm is not None:
            change_direction = "down"
            change_idx = lower_alarm
        else:
            change_direction = "none"
            change_idx = None

        # Current CUSUM values (for monitoring)
        current_upper = float(s_upper[-1])
        current_lower = float(s_lower[-1])

        return {
            "change_detected": change_detected,
            "change_direction": change_direction,
            "change_point_index": change_idx,
            "current_cusum_upper": round(current_upper, 2),
            "current_cusum_lower": round(current_lower, 2),
            "threshold": self.threshold,
            "cusum_upper": [round(float(x), 3) for x in s_upper],
            "cusum_lower": [round(float(x), 3) for x in s_lower],
        }

    def detect_edge_death(
        self,
        clv_series: List[float],
        window: int = 50,
    ) -> dict:
        """Detect if edge has died (CLV consistently below zero).

        Uses rolling average and CUSUM together.
        """
        if len(clv_series) < window:
            return {"edge_dead": False, "reason": "insufficient data"}

        arr = np.array(clv_series)

        # Rolling average
        recent = arr[-window:]
        avg_recent = float(np.mean(recent))

        # Percentage below zero
        pct_negative = float(np.mean(recent < 0))

        # CUSUM for downward shift
        cusum = self.cusum_detection(clv_series, target_mean=1.0)  # Target = 1 cent CLV

        # Edge death criteria:
        # 1. Rolling average below 0 for last 50 bets
        # 2. >60% of recent bets have negative CLV
        # 3. CUSUM detected downward shift
        is_dead = (avg_recent < 0 and pct_negative > 0.60) or cusum["change_direction"] == "down"
        is_warning = avg_recent < 0.5 or pct_negative > 0.50

        return {
            "edge_dead": is_dead,
            "edge_warning": is_warning,
            "recent_avg_clv": round(avg_recent, 2),
            "pct_negative_clv": round(pct_negative, 4),
            "cusum_result": cusum,
            "window": window,
        }

    def trend_analysis(
        self,
        values: List[float],
    ) -> dict:
        """Simple linear trend analysis of edge metrics."""
        if len(values) < 10:
            return {"has_trend": False}

        arr = np.array(values)
        x = np.arange(len(arr))

        from scipy import stats
        slope, intercept, r_val, p_val, std_err = stats.linregress(x, arr)

        if slope > 0.05 and p_val < 0.10:
            trend = "improving"
        elif slope < -0.05 and p_val < 0.10:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "has_trend": True,
            "trend": trend,
            "slope": round(slope, 4),
            "r_squared": round(r_val ** 2, 4),
            "p_value": round(p_val, 4),
            "is_significant": p_val < 0.05,
        }
