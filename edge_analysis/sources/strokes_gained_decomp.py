"""SG Component Prediction per Course Type.

Decomposes strokes gained into OTT/APP/ATG/PUTT components and predicts
performance based on course type demands. Different courses weight
SG components differently — a bomber's course rewards OTT while a
putting course rewards PUTT.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from edge_analysis.edge_sources import EdgeSource

logger = logging.getLogger(__name__)

# Course type -> SG component weights (how much each component matters)
COURSE_TYPE_WEIGHTS = {
    "bomber": {"sg_ott": 0.40, "sg_app": 0.25, "sg_atg": 0.15, "sg_putt": 0.20},
    "precision": {"sg_ott": 0.15, "sg_app": 0.35, "sg_atg": 0.25, "sg_putt": 0.25},
    "balanced": {"sg_ott": 0.25, "sg_app": 0.30, "sg_atg": 0.20, "sg_putt": 0.25},
    "putting": {"sg_ott": 0.15, "sg_app": 0.20, "sg_atg": 0.25, "sg_putt": 0.40},
    "links": {"sg_ott": 0.20, "sg_app": 0.35, "sg_atg": 0.25, "sg_putt": 0.20},
    "resort": {"sg_ott": 0.25, "sg_app": 0.25, "sg_atg": 0.25, "sg_putt": 0.25},
}

DEFAULT_WEIGHTS = {"sg_ott": 0.25, "sg_app": 0.30, "sg_atg": 0.20, "sg_putt": 0.25}


class StrokesGainedDecompSource(EdgeSource):
    """SG component decomposition matched to course demands."""

    name = "strokes_gained_decomp"
    category = "predictive"
    description = "Predict performance by matching SG component profile to course type demands"

    def get_signal(self, player: str, tournament_context: Dict[str, Any]) -> float:
        """Signal based on SG component fit to course type.

        Positive signal = player's SG strengths match course demands.
        """
        sg_components = tournament_context.get("player_sg", {})
        course_type = tournament_context.get("course_type", "balanced")
        field_sg_values = tournament_context.get("field_sg_values", [])

        sg_ott = sg_components.get("sg_ott", 0.0)
        sg_app = sg_components.get("sg_app", 0.0)
        sg_atg = sg_components.get("sg_atg", 0.0)
        sg_putt = sg_components.get("sg_putt", 0.0)

        weights = COURSE_TYPE_WEIGHTS.get(course_type, DEFAULT_WEIGHTS)

        # Weighted SG total for this course type
        weighted_sg = (
            sg_ott * weights["sg_ott"] +
            sg_app * weights["sg_app"] +
            sg_atg * weights["sg_atg"] +
            sg_putt * weights["sg_putt"]
        )

        # Normalize: compare weighted SG to equal-weight SG
        equal_weight_sg = (sg_ott + sg_app + sg_atg + sg_putt) / 4.0
        course_fit_advantage = weighted_sg - equal_weight_sg

        # Also compare to field
        if field_sg_values:
            field_mean = float(np.mean(field_sg_values))
            field_std = float(np.std(field_sg_values)) if len(field_sg_values) > 1 else 1.0
            z_score = (weighted_sg * 4 - field_mean) / max(field_std, 0.5)
            # Blend course fit advantage with field-relative z-score
            signal = course_fit_advantage * 2.0 + z_score * 0.3
        else:
            signal = course_fit_advantage * 2.0

        return round(float(np.clip(signal, -3.0, 3.0)), 4)

    def get_mechanism(self) -> str:
        return (
            "Markets often price golfers based on overall SG total without properly "
            "weighting which components matter at a specific course. A player with "
            "+2.0 SG:OTT and -0.5 SG:PUTT is undervalued at bomber courses and "
            "overvalued at putting courses. This source captures the mispricing from "
            "inadequate component weighting."
        )

    def get_decay_risk(self) -> str:
        return (
            "medium — As course-fit models become more common in public tools "
            "(DataGolf, etc.), this edge is slowly being priced in. However, the "
            "specific weights and interaction effects remain proprietary."
        )

    def validate(self, historical_data: List[Dict[str, Any]]) -> dict:
        """Validate: do SG component predictions beat equal-weight SG?"""
        if len(historical_data) < 20:
            return {"is_valid": False, "reason": "insufficient data", "n_samples": len(historical_data)}

        correct = 0
        signals = []
        outcomes = []

        for rec in historical_data:
            sg = rec.get("player_sg", {})
            course_type = rec.get("course_type", "balanced")
            actual = rec.get("actual_finish_pct", 0.5)  # 0-1 percentile finish

            weights = COURSE_TYPE_WEIGHTS.get(course_type, DEFAULT_WEIGHTS)
            weighted = sum(sg.get(k, 0) * v for k, v in weights.items())
            equal = sum(sg.get(k, 0) for k in ["sg_ott", "sg_app", "sg_atg", "sg_putt"]) / 4.0

            signal = weighted - equal
            signals.append(signal)
            outcomes.append(actual)

            # Did course-fit prediction improve accuracy?
            if (signal > 0 and actual > 0.5) or (signal < 0 and actual < 0.5):
                correct += 1

        hit_rate = correct / len(historical_data)
        sig_arr = np.array(signals)
        out_arr = np.array(outcomes)

        # Correlation between signal and outcome
        corr = float(np.corrcoef(sig_arr, out_arr)[0, 1]) if len(sig_arr) > 2 else 0.0
        returns = sig_arr * (out_arr - 0.5)
        sharpe = float(np.mean(returns) / max(np.std(returns), 0.001))

        # P-value via permutation test
        from scipy import stats
        if len(sig_arr) >= 10:
            _, p_val = stats.pearsonr(sig_arr, out_arr)
        else:
            p_val = 1.0

        return {
            "is_valid": hit_rate > 0.52 and corr > 0.05,
            "sharpe": round(sharpe, 4),
            "hit_rate": round(hit_rate, 4),
            "correlation": round(corr, 4),
            "n_samples": len(historical_data),
            "p_value": round(p_val, 4),
        }
