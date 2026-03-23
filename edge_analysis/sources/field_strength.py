"""Field Strength Source — Normalize stats for field quality.

PGA Tour fields vary enormously in strength. A T10 at THE PLAYERS
is far more impressive than T10 at the Barracuda Championship.
This source adjusts player projections for field quality.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from edge_analysis.edge_sources import EdgeSource

logger = logging.getLogger(__name__)


class FieldStrengthSource(EdgeSource):
    """Normalize projections for field quality differences."""

    name = "field_strength"
    category = "structural"
    description = "Adjust expectations based on field strength — weak/strong field effects"

    def get_signal(self, player: str, tournament_context: Dict[str, Any]) -> float:
        """Signal based on player strength relative to field.

        Positive = player is stronger relative to field than overall ranking suggests.
        """
        player_sg = tournament_context.get("player_sg", {}).get("sg_total", 0.0)
        field_sg_values = tournament_context.get("field_sg_values", [])
        player_rank = tournament_context.get("world_rank", 50)
        field_avg_rank = tournament_context.get("field_avg_rank", 80)

        if not field_sg_values:
            return 0.0

        field_mean = float(np.mean(field_sg_values))
        field_std = float(np.std(field_sg_values)) if len(field_sg_values) > 1 else 1.0
        field_size = len(field_sg_values)

        # Z-score within this specific field
        z_in_field = (player_sg - field_mean) / max(field_std, 0.3)

        # Expected z-score based on world rank
        # Rank 1 -> ~2.0 SG, Rank 50 -> ~0.5 SG, Rank 150 -> ~-0.3 SG
        expected_sg = max(2.5 - player_rank * 0.02, -1.0)
        expected_z = (expected_sg - field_mean) / max(field_std, 0.3)

        # Signal: difference between actual field-relative strength and expected
        rank_based_edge = z_in_field - expected_z

        # Weak field bonus: player is better relative to field than rank suggests
        # This happens in opposite-field events
        field_strength_category = self._classify_field(field_mean, field_size)
        if field_strength_category == "weak" and player_sg > field_mean + 0.5:
            rank_based_edge += 0.2
        elif field_strength_category == "strong" and player_sg < field_mean - 0.3:
            rank_based_edge -= 0.15

        return round(float(np.clip(rank_based_edge, -2.0, 2.0)), 4)

    def _classify_field(self, field_mean_sg: float, field_size: int) -> str:
        """Classify field strength."""
        if field_mean_sg > 0.3 and field_size > 120:
            return "strong"
        elif field_mean_sg > 0.1:
            return "average"
        elif field_mean_sg < -0.1 or field_size < 100:
            return "weak"
        return "average"

    def get_mechanism(self) -> str:
        return (
            "Outright and top-finish odds are calibrated to the overall PGA Tour "
            "but not perfectly adjusted for the specific field each week. In weak "
            "fields (opposite-field events), top players are systematically undervalued "
            "for win/top-5 finishes because odds reflect their overall rank rather "
            "than their dominance within this specific field."
        )

    def get_decay_risk(self) -> str:
        return (
            "medium — Field strength adjustments are increasingly common in "
            "commercial models. However, the specific normalization and its "
            "interaction with market type (outrights vs matchups) retains value."
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
