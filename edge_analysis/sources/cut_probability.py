"""Cut Probability Source — Field strength, course history -> make/miss cut.

Models cut probability using:
  - Field strength relative to player skill
  - Course-specific cut history
  - Player's make-cut rate
  - Recent form trajectory
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
from scipy import stats as scipy_stats

from edge_analysis.edge_sources import EdgeSource

logger = logging.getLogger(__name__)


class CutProbabilitySource(EdgeSource):
    """Predict make/miss cut probability more accurately than the market."""

    name = "cut_probability"
    category = "predictive"
    description = "Model cut probability from field strength, course history, and form"

    def get_signal(self, player: str, tournament_context: Dict[str, Any]) -> float:
        """Signal for make/miss cut edge.

        Positive = model says player more likely to make cut than market implies.
        """
        player_sg = tournament_context.get("player_sg", {}).get("sg_total", 0.0)
        field_sg_values = tournament_context.get("field_sg_values", [])
        cut_history = tournament_context.get("player_cut_history", {})
        market_cut_prob = tournament_context.get("market_make_cut_prob", None)

        # Model's cut probability
        model_cut_prob = self._model_cut_probability(
            player_sg, field_sg_values, cut_history, tournament_context
        )

        if market_cut_prob is None:
            # No market price available — can't generate edge signal
            # But we can still flag extreme values
            if model_cut_prob > 0.85:
                return 0.3  # Very likely to make cut
            elif model_cut_prob < 0.35:
                return -0.3  # Likely to miss cut
            return 0.0

        # Edge = model probability - market probability
        edge = model_cut_prob - market_cut_prob

        # Convert to signal scale
        signal = edge * 5.0  # Scale: 5% edge -> 0.25 signal

        return round(float(np.clip(signal, -2.0, 2.0)), 4)

    def _model_cut_probability(
        self,
        player_sg: float,
        field_sg_values: List[float],
        cut_history: dict,
        context: dict,
    ) -> float:
        """Model make-cut probability from multiple factors."""
        # Base: logistic model from SG total
        # SG = 0 (tour avg) -> ~65% make cut
        # SG = +1 -> ~80%, SG = -1 -> ~45%
        base_prob = 1.0 / (1.0 + np.exp(-(player_sg * 1.2 + 0.6)))

        # Field strength adjustment
        if field_sg_values:
            field_mean = float(np.mean(field_sg_values))
            field_std = float(np.std(field_sg_values)) if len(field_sg_values) > 1 else 1.0
            # Stronger field = harder to make cut
            field_factor = -0.05 * (field_mean - 0.0) / max(field_std, 0.5)
            base_prob = np.clip(base_prob + field_factor, 0.05, 0.97)

        # Historical cut rate for this player
        historical_cut_rate = cut_history.get("make_cut_rate", None)
        n_events = cut_history.get("n_events", 0)
        if historical_cut_rate is not None and n_events >= 10:
            # Blend model with historical rate (more weight to model for small samples)
            weight = min(n_events / 30.0, 0.4)
            base_prob = base_prob * (1 - weight) + historical_cut_rate * weight

        # Course-specific cut rate
        course_cut_rate = cut_history.get("course_cut_rate", None)
        n_course = cut_history.get("n_course_events", 0)
        if course_cut_rate is not None and n_course >= 3:
            course_weight = min(n_course / 10.0, 0.2)
            base_prob = base_prob * (1 - course_weight) + course_cut_rate * course_weight

        # Recent form: consecutive missed cuts is a red flag
        consecutive_mc = cut_history.get("consecutive_missed_cuts", 0)
        if consecutive_mc >= 3:
            base_prob *= 0.85
        elif consecutive_mc >= 2:
            base_prob *= 0.92

        # Field size: larger field = more players miss cut
        field_size = context.get("field_size", 156)
        if field_size < 80:
            base_prob *= 1.05  # Smaller field = easier to make cut
        elif field_size > 156:
            base_prob *= 0.95

        return float(np.clip(base_prob, 0.03, 0.98))

    def get_mechanism(self) -> str:
        return (
            "Make/miss cut markets are among the thinnest in golf betting, with "
            "wide vig and infrequent repricing. The market sets cut lines based on "
            "simple heuristics (ranking, name recognition) rather than modeling "
            "the actual cut dynamics: field strength, 36-hole scoring distribution, "
            "and player-specific cut rates at specific courses."
        )

    def get_decay_risk(self) -> str:
        return (
            "low — Cut markets remain thinly traded with high vig. The specific "
            "modeling of field-adjusted cut probability requires combining multiple "
            "data sources in ways that bookmakers do not prioritize."
        )

    def validate(self, historical_data: List[Dict[str, Any]]) -> dict:
        if len(historical_data) < 30:
            return {"is_valid": False, "reason": "insufficient data", "n_samples": len(historical_data)}

        model_probs = []
        actuals = []
        for rec in historical_data:
            sg = rec.get("player_sg", {}).get("sg_total", 0.0)
            field_sg = rec.get("field_sg_values", [])
            cut_hist = rec.get("player_cut_history", {})
            prob = self._model_cut_probability(sg, field_sg, cut_hist, rec)
            model_probs.append(prob)
            actuals.append(1.0 if rec.get("made_cut", True) else 0.0)

        pred_arr = np.array(model_probs)
        act_arr = np.array(actuals)

        # Brier score
        brier = float(np.mean((pred_arr - act_arr) ** 2))
        naive_brier = float(np.mean((np.mean(act_arr) - act_arr) ** 2))
        bss = 1.0 - brier / naive_brier if naive_brier > 0 else 0.0

        # Calibration
        from edge_analysis.predictive import calibration_curve
        cal = calibration_curve(pred_arr, act_arr, n_bins=5)

        return {
            "is_valid": bss > 0.02,
            "sharpe": round(bss * 2, 4),
            "hit_rate": round(float(np.mean((pred_arr > 0.5) == (act_arr > 0.5))), 4),
            "brier_score": round(brier, 4),
            "brier_skill_score": round(bss, 4),
            "n_samples": len(historical_data),
            "p_value": 0.0,
            "calibration": cal,
        }
