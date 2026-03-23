"""Form vs Baseline Source — Optimal recency weighting window.

Markets tend to overweight very recent form (last 1-2 events).
Research shows optimal weighting uses ~12-24 event windows with
decay. This source identifies when recent form deviates from
baseline in a way the market has overreacted to.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
from scipy import stats as scipy_stats

from edge_analysis.edge_sources import EdgeSource

logger = logging.getLogger(__name__)


class FormVsBaselineSource(EdgeSource):
    """Detect market overreaction to recent form vs true baseline skill."""

    name = "form_vs_baseline"
    category = "predictive"
    description = "Optimal recency weighting — fade market overreaction to recent form"

    def get_signal(self, player: str, tournament_context: Dict[str, Any]) -> float:
        """Signal based on form vs baseline deviation.

        Positive = market likely overreacted to poor recent form (buy low).
        Negative = market likely overreacted to hot streak (sell high).
        """
        sg_history = tournament_context.get("sg_history", [])

        if len(sg_history) < 8:
            return 0.0

        sg_values = np.array([h.get("sg_total", 0.0) for h in sg_history])

        # Baseline: last 24 events (or all available)
        baseline_window = min(24, len(sg_values))
        baseline = float(np.mean(sg_values[-baseline_window:]))

        # Recent form: last 4 events
        recent_window = min(4, len(sg_values))
        recent = float(np.mean(sg_values[-recent_window:]))

        # Form deviation from baseline
        deviation = recent - baseline

        # Standard deviation of baseline for normalization
        baseline_std = float(np.std(sg_values[-baseline_window:])) if baseline_window > 2 else 1.5
        z_deviation = deviation / max(baseline_std, 0.5)

        # The signal is CONTRARIAN: when recent form is much worse than baseline,
        # the market overreacts → player is undervalued → positive signal (buy).
        # When recent form is much better than baseline,
        # the market overreacts → player is overvalued → negative signal (sell).

        # Only generate signal when deviation is meaningful (>0.5 z-scores)
        if abs(z_deviation) < 0.5:
            return 0.0

        # Contrarian signal: negative z_deviation (cold streak) → positive signal
        signal = -z_deviation * 0.6

        # Confidence adjustment: more history = more confident in baseline
        sample_factor = min(len(sg_values) / 20.0, 1.0)
        signal *= sample_factor

        # Check if the deviation is statistically significant
        if len(sg_values) >= 12:
            recent_vals = sg_values[-recent_window:]
            baseline_vals = sg_values[-baseline_window:-recent_window]
            if len(baseline_vals) >= 5:
                t_stat, p_val = scipy_stats.ttest_ind(recent_vals, baseline_vals)
                # If not significant, reduce signal
                if p_val > 0.20:
                    signal *= 0.5

        return round(float(np.clip(signal, -2.0, 2.0)), 4)

    def get_mechanism(self) -> str:
        return (
            "Markets and bettors systematically overweight the last 1-2 events "
            "(recency bias). Research shows optimal form windows are 12-24 events. "
            "When a player has a bad week, odds lengthen more than justified by their "
            "true skill baseline. This creates systematic mean-reversion opportunities "
            "that our model captures by properly weighting the full sample."
        )

    def get_decay_risk(self) -> str:
        return (
            "low — Recency bias is a behavioral phenomenon deeply rooted in human "
            "psychology. Even sophisticated bettors tend to overweight recent results. "
            "This edge source is among the most durable."
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

        # Only validate on non-zero signals
        mask = np.abs(sig_arr) > 0.1
        if mask.sum() < 15:
            return {"is_valid": False, "reason": "insufficient triggered signals", "n_samples": int(mask.sum())}

        sig_active = sig_arr[mask]
        out_active = out_arr[mask]
        corr = float(np.corrcoef(sig_active, out_active)[0, 1]) if len(sig_active) > 2 else 0.0
        returns = sig_active * (out_active - 0.5)
        sharpe = float(np.mean(returns) / max(np.std(returns), 0.001))
        hit_rate = float(np.mean((sig_active > 0) == (out_active > 0.5)))

        _, p_val = scipy_stats.pearsonr(sig_active, out_active) if len(sig_active) >= 10 else (0, 1.0)

        return {
            "is_valid": hit_rate > 0.52 and sharpe > 0.05,
            "sharpe": round(sharpe, 4),
            "hit_rate": round(hit_rate, 4),
            "correlation": round(corr, 4),
            "n_samples": int(mask.sum()),
            "p_value": round(float(p_val), 4),
        }
