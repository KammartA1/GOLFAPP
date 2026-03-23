"""Altitude Conditions Source — Ball flight at altitude, desert vs links.

Ball flight changes significantly at altitude (Denver, Mexico City) and
in different climates (desert heat vs links cold). This source captures
player-specific adaptation to these conditions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from edge_analysis.edge_sources import EdgeSource

logger = logging.getLogger(__name__)

# Altitude thresholds
HIGH_ALTITUDE = 4000  # feet
VERY_HIGH_ALTITUDE = 6000

# Course type climate profiles
CLIMATE_PROFILES = {
    "links": {"wind_factor": 1.5, "temp_range": (45, 65), "altitude": 0},
    "desert": {"wind_factor": 0.8, "temp_range": (75, 105), "altitude": 1000},
    "mountain": {"wind_factor": 1.0, "temp_range": (50, 80), "altitude": 5000},
    "tropical": {"wind_factor": 1.2, "temp_range": (75, 95), "altitude": 0},
    "temperate": {"wind_factor": 1.0, "temp_range": (55, 85), "altitude": 500},
}


class AltitudeConditionsSource(EdgeSource):
    """Model altitude and climate effects on player performance."""

    name = "altitude_conditions"
    category = "informational"
    description = "Ball flight at altitude, desert vs links climate adaptation"

    def get_signal(self, player: str, tournament_context: Dict[str, Any]) -> float:
        """Signal based on player adaptation to altitude/climate conditions.

        Positive = player performs well in these specific conditions.
        """
        course = tournament_context.get("course_profile", {})
        elevation = course.get("elevation", 0)
        climate = course.get("climate_type", "temperate")
        player_history = tournament_context.get("player_condition_history", {})

        signal = 0.0

        # Altitude effect
        if elevation > HIGH_ALTITUDE:
            altitude_sg_diff = player_history.get("high_altitude_sg_diff", 0.0)
            n_altitude_events = player_history.get("n_altitude_events", 0)

            if n_altitude_events >= 3:
                altitude_factor = min((elevation - HIGH_ALTITUDE) / 3000.0, 1.0)
                signal += altitude_sg_diff * altitude_factor * 1.2

                # At altitude: ball flies further, spin is reduced
                # Players who rely on high spin are disadvantaged
                spin_rate = tournament_context.get("player_spin_rate", "average")
                if spin_rate == "high" and elevation > VERY_HIGH_ALTITUDE:
                    signal -= 0.15  # High-spin players struggle
                elif spin_rate == "low":
                    signal += 0.08  # Low-spin players adapt better

        # Climate adaptation
        climate_sg_diff = player_history.get(f"{climate}_sg_diff", 0.0)
        n_climate_events = player_history.get(f"n_{climate}_events", 0)
        if n_climate_events >= 3:
            signal += climate_sg_diff * 0.8

        # Links-specific: wind tolerance is critical
        if climate == "links":
            wind_skill = player_history.get("wind_sg_differential", 0.0)
            signal += wind_skill * 0.5

        # Desert-specific: heat tolerance, dry conditions
        if climate == "desert":
            heat_tolerance = player_history.get("heat_sg_differential", 0.0)
            signal += heat_tolerance * 0.4

        # Confidence: more events in these conditions = more reliable
        total_condition_events = max(
            player_history.get("n_altitude_events", 0),
            player_history.get(f"n_{climate}_events", 0),
        )
        confidence = min(total_condition_events / 10.0, 1.0)
        signal *= confidence

        return round(float(np.clip(signal, -1.5, 1.5)), 4)

    def get_mechanism(self) -> str:
        return (
            "Ball flight physics change significantly at altitude: reduced air density "
            "means 5-10% more carry distance, less spin, and different club selection. "
            "Similarly, links conditions require specific wind skills that most "
            "American-based players lack. Markets price these events based on overall "
            "rankings without fully accounting for condition-specific skill."
        )

    def get_decay_risk(self) -> str:
        return (
            "low — Altitude and climate effects are physical realities that cannot "
            "be arbitraged. The edge comes from player-specific adaptation data "
            "that requires years of condition-stratified performance analysis."
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
        mask = np.abs(sig_arr) > 0.05
        if mask.sum() < 10:
            return {"is_valid": False, "reason": "insufficient condition events", "n_samples": int(mask.sum())}

        sig_active = sig_arr[mask]
        out_active = out_arr[mask]
        corr = float(np.corrcoef(sig_active, out_active)[0, 1]) if len(sig_active) > 2 else 0.0
        returns = sig_active * (out_active - 0.5)
        sharpe = float(np.mean(returns) / max(np.std(returns), 0.001))

        from scipy import stats
        _, p_val = stats.pearsonr(sig_active, out_active) if len(sig_active) >= 10 else (0, 1.0)

        return {
            "is_valid": corr > 0.03,
            "sharpe": round(sharpe, 4),
            "hit_rate": round(float(np.mean((sig_active > 0) == (out_active > 0.5))), 4),
            "correlation": round(corr, 4),
            "n_samples": int(mask.sum()),
            "p_value": round(float(p_val), 4),
        }
