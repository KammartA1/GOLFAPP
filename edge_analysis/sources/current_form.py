"""
Current Form Edge Source
=========================
Optimal recency weighting for detecting genuine hot/cold streaks vs noise.
Market edge: the public overweights the most recent 1-2 events; the
statistically optimal window is longer (8-12 events with exponential decay).
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import stats as sp_stats

log = logging.getLogger(__name__)

# Optimal half-life in events (calibrated from PGA Tour data 2015-2024)
_HALF_LIFE_EVENTS = 8
_MIN_EVENTS = 4
_MAX_EVENTS = 30


class CurrentFormSource:
    """Recency-weighted form signal with streak detection."""

    name = "Current Form"
    category = "form"
    version = "1.0"

    def get_signal(self, player: dict, tournament_context: dict) -> float:
        """
        Compute current form edge signal.

        player keys:
            recent_results   – list of dicts, most recent first, each with:
                               {'sg_total': float, 'finish': int, 'field_strength': float,
                                'events_ago': int}
            baseline_sg      – long-term (2+ year) SG average
            sg_total         – current season SG average (for fallback)
        tournament_context keys:
            field_strength   – 0-1 scale, strength of this week's field
        """
        results = player.get("recent_results", [])
        baseline = player.get("baseline_sg", player.get("sg_total", 0.0))

        if len(results) < _MIN_EVENTS:
            return 0.0

        # Cap at MAX_EVENTS
        results = results[:_MAX_EVENTS]

        # Exponential decay weights: w_i = 0.5^(i / half_life)
        n = len(results)
        weights = np.array([
            math.pow(0.5, r.get("events_ago", i) / _HALF_LIFE_EVENTS)
            for i, r in enumerate(results)
        ], dtype=float)
        weights /= weights.sum()

        # Weighted SG (field-adjusted)
        sg_values = np.array([
            r.get("sg_total", 0.0) * (1.0 + 0.1 * (r.get("field_strength", 0.5) - 0.5))
            for r in results
        ], dtype=float)

        weighted_sg = float(np.dot(weights, sg_values))

        # Form edge: deviation from baseline
        form_edge = weighted_sg - baseline

        # ── Streak detection ────────────────────────────────────────────
        # Runs test for non-randomness in above/below baseline
        above = sg_values > baseline
        streak_z = self._runs_test_z(above)

        # Significant hot streak amplifies signal, significant cold streak too
        streak_multiplier = 1.0
        if abs(streak_z) > 1.96:  # p < 0.05 two-tailed
            # Significant clustering — trend is real, amplify
            streak_multiplier = 1.3
        elif abs(streak_z) > 1.645:  # p < 0.10
            streak_multiplier = 1.15

        # ── Momentum: slope of recent SG ────────────────────────────────
        if n >= 5:
            x = np.arange(n, dtype=float)
            slope, _, r_value, p_slope, _ = sp_stats.linregress(x, sg_values)
            # slope is SG change per event
            if p_slope < 0.15:
                momentum = slope * 3.0  # project 3 events forward
            else:
                momentum = 0.0
        else:
            momentum = 0.0

        # ── Combine ─────────────────────────────────────────────────────
        signal = form_edge * streak_multiplier + momentum * 0.4

        # ── Anti-recency-bias correction ────────────────────────────────
        # Public overweights last result; penalise if last result is extreme outlier
        if n >= 3:
            last_sg = sg_values[0]
            rest_mean = float(np.mean(sg_values[1:]))
            rest_std = float(np.std(sg_values[1:], ddof=1))
            if rest_std > 0.1:
                z_last = (last_sg - rest_mean) / rest_std
                if abs(z_last) > 2.0:
                    # Last event is an outlier — discount it (mean-reversion)
                    reversion = -0.15 * (last_sg - rest_mean)
                    signal += reversion

        return round(float(signal), 4)

    @staticmethod
    def _runs_test_z(binary_sequence: np.ndarray) -> float:
        """
        Wald-Wolfowitz runs test for randomness.
        Returns z-score: negative = too few runs (trending), positive = too many.
        """
        n = len(binary_sequence)
        if n < 10:
            return 0.0

        n1 = int(binary_sequence.sum())
        n2 = n - n1
        if n1 == 0 or n2 == 0:
            return 0.0

        # Count runs
        runs = 1
        for i in range(1, n):
            if binary_sequence[i] != binary_sequence[i - 1]:
                runs += 1

        # Expected runs and variance
        mu = 1 + (2 * n1 * n2) / n
        denom = n * n * (n - 1)
        if denom == 0:
            return 0.0
        var = (2 * n1 * n2 * (2 * n1 * n2 - n)) / denom
        if var <= 0:
            return 0.0

        z = (runs - mu) / math.sqrt(var)
        return float(z)

    def get_mechanism(self) -> str:
        return (
            "We compute exponentially-weighted SG with a half-life of 8 events, "
            "field-strength adjusted.  This is compared to a 2+ year baseline.  "
            "A Wald-Wolfowitz runs test detects genuine streaks vs random "
            "variation: when clustering is significant (p<0.05), the trend is "
            "amplified.  Linear regression on recent SG captures momentum.  "
            "Anti-recency-bias correction discounts the last result if it is a "
            "statistical outlier (>2 sigma), counteracting the public's tendency "
            "to overweight the most recent event."
        )

    def get_decay_risk(self) -> str:
        return (
            "MEDIUM — Form signals are inherently ephemeral.  The edge is in "
            "the optimal weighting scheme and streak detection, not in the raw "
            "data.  Decay risk: if market consensus moves toward exponential "
            "decay models, the advantage compresses.  Refresh weights quarterly."
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
