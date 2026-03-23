"""
Altitude & Conditions Edge Source
===================================
Ball flight at altitude (Denver, Mexico City), desert vs links,
humidity effects on ball spin.
Market edge: sportsbooks do not granularly adjust for altitude or
atmospheric conditions beyond basic weather.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import stats as sp_stats

log = logging.getLogger(__name__)

# Altitude reference points
_SEA_LEVEL = 0
_MILD_ALTITUDE = 1500    # ft — noticeable carry increase
_HIGH_ALTITUDE = 4000    # ft — significant carry increase
_EXTREME_ALTITUDE = 7000  # ft — Mexico City, Denver

# Ball carry increase per 1000 ft of elevation: ~1.8-2.2%
_CARRY_PCT_PER_1000FT = 0.020

# Spin reduction at altitude: ~3% per 1000 ft
_SPIN_REDUCTION_PER_1000FT = 0.030

# Course condition types and their effects
_CONDITION_PROFILES = {
    "links": {
        "ground_firmness": 0.8,  # firm, ball runs out
        "wind_exposure": 0.9,
        "rough_severity": 0.7,
        "green_speed": 0.6,
        "bounce_factor": 1.3,    # ball bounces more
    },
    "parkland": {
        "ground_firmness": 0.5,
        "wind_exposure": 0.4,
        "rough_severity": 0.6,
        "green_speed": 0.7,
        "bounce_factor": 0.8,
    },
    "desert": {
        "ground_firmness": 0.7,
        "wind_exposure": 0.6,
        "rough_severity": 0.3,   # desert rough is sparse
        "green_speed": 0.8,
        "bounce_factor": 1.0,
    },
    "mountain": {
        "ground_firmness": 0.6,
        "wind_exposure": 0.7,
        "rough_severity": 0.5,
        "green_speed": 0.7,
        "bounce_factor": 0.9,
    },
    "coastal": {
        "ground_firmness": 0.6,
        "wind_exposure": 0.85,
        "rough_severity": 0.65,
        "green_speed": 0.65,
        "bounce_factor": 1.1,
    },
}


def _load_course_profiles() -> dict:
    try:
        from config.courses import COURSE_PROFILES
        return COURSE_PROFILES
    except ImportError:
        return {}


class AltitudeConditionsSource:
    """Altitude, atmosphere, and course condition edge."""

    name = "Altitude & Conditions"
    category = "conditions"
    version = "1.0"

    def __init__(self):
        self._profiles = _load_course_profiles()

    def get_signal(self, player: dict, tournament_context: dict) -> float:
        """
        Compute altitude/conditions edge.

        player keys:
            driving_distance      – avg driving distance (yards)
            spin_rate             – avg ball spin (rpm), or "high"/"low"/"medium"
            sg_app                – approach SG (affected by carry control)
            altitude_sg           – SG differential at altitude courses (optional)
            links_sg              – SG differential on links courses (optional)
            desert_sg             – SG differential on desert courses (optional)
            trajectory            – "high", "mid", "low" ball flight
        tournament_context keys:
            course                – venue name
            elevation_ft          – course elevation in feet
            humidity_pct          – relative humidity (0-100)
            temperature_f         – temperature in F
            course_condition      – "links", "parkland", "desert", "mountain", "coastal"
        """
        course = tournament_context.get("course", "")
        profile = self._profiles.get(course, {})

        elevation = tournament_context.get(
            "elevation_ft", profile.get("elevation_ft", 0)
        )
        humidity = tournament_context.get("humidity_pct", 50.0)
        temp_f = tournament_context.get("temperature_f", 72.0)
        condition = tournament_context.get(
            "course_condition",
            profile.get("typical_conditions", "parkland"),
        )
        # Normalise condition string
        condition_lower = condition.lower().replace(" ", "_")
        for key in _CONDITION_PROFILES:
            if key in condition_lower:
                condition = key
                break
        else:
            condition = "parkland"

        cond_profile = _CONDITION_PROFILES.get(condition, _CONDITION_PROFILES["parkland"])

        # ── Altitude carry adjustment ───────────────────────────────────
        carry_increase_pct = elevation / 1000.0 * _CARRY_PCT_PER_1000FT
        spin_reduction_pct = elevation / 1000.0 * _SPIN_REDUCTION_PER_1000FT

        altitude_edge = 0.0

        # High-trajectory players benefit MORE at altitude (carry advantage)
        trajectory = player.get("trajectory", "mid")
        traj_mod = {"high": 1.3, "mid": 1.0, "low": 0.7}.get(trajectory, 1.0)

        if elevation >= _EXTREME_ALTITUDE:
            # Extreme altitude: everyone hits further, but spin control changes
            # Players who rely on spin to stop the ball are disadvantaged
            spin_type = player.get("spin_rate", "medium")
            if isinstance(spin_type, str):
                spin_factor = {"high": -0.10, "medium": 0.0, "low": 0.08}.get(spin_type, 0.0)
            else:
                # Numeric spin rate: high spin > 2800 rpm is a disadvantage
                if spin_type > 2800:
                    spin_factor = -0.08
                elif spin_type < 2400:
                    spin_factor = 0.06
                else:
                    spin_factor = 0.0

            altitude_edge = spin_factor * traj_mod

            # Player-specific altitude experience
            alt_sg = player.get("altitude_sg", 0.0)
            altitude_edge += alt_sg * 0.5

        elif elevation >= _HIGH_ALTITUDE:
            spin_type = player.get("spin_rate", "medium")
            if isinstance(spin_type, str):
                spin_factor = {"high": -0.05, "medium": 0.0, "low": 0.04}.get(spin_type, 0.0)
            else:
                spin_factor = -0.03 if spin_type > 2800 else (0.03 if spin_type < 2400 else 0.0)
            altitude_edge = spin_factor * traj_mod * 0.7
            alt_sg = player.get("altitude_sg", 0.0)
            altitude_edge += alt_sg * 0.3

        elif elevation >= _MILD_ALTITUDE:
            altitude_edge = player.get("altitude_sg", 0.0) * 0.15

        # ── Humidity / air density ──────────────────────────────────────
        # Humid air is LESS dense (water vapor is lighter than N2/O2)
        # Less dense air = more carry but less spin bite
        humidity_edge = 0.0
        if humidity > 80:
            # High humidity: ball carries slightly more, less spin
            sg_app = player.get("sg_app", 0.0)
            # Good approach players who rely on spin are slightly disadvantaged
            if sg_app > 0.5:
                humidity_edge = -0.02 * (humidity - 80) / 20.0
            else:
                humidity_edge = 0.01 * (humidity - 80) / 20.0
        elif humidity < 20:
            # Very dry: dense air, more spin, benefits spin players
            humidity_edge = 0.01

        # ── Course condition fit ────────────────────────────────────────
        condition_edge = 0.0

        if condition == "links":
            # Links: wind management, low ball flight, bump-and-run
            links_sg = player.get("links_sg", 0.0)
            condition_edge += links_sg * 0.5
            # Low trajectory is an advantage on links
            traj_links = {"low": 0.06, "mid": 0.0, "high": -0.05}.get(trajectory, 0.0)
            condition_edge += traj_links
            # Wind skill matters more
            wind_sg = player.get("wind_sg", 0.0)
            condition_edge += wind_sg * cond_profile["wind_exposure"] * 0.2

        elif condition == "desert":
            desert_sg = player.get("desert_sg", 0.0)
            condition_edge += desert_sg * 0.4
            # Desert: wide fairways, less rough, favours aggressive play
            sg_ott = player.get("sg_ott", 0.0)
            if sg_ott > 0.3:
                condition_edge += 0.03  # bombers can swing freely

        elif condition == "mountain":
            # Mountain: uneven lies, elevation changes
            sg_atg = player.get("sg_atg", 0.0)
            condition_edge += sg_atg * 0.15  # short game from weird lies

        # ── Temperature / air density physics ───────────────────────────
        # Cold air is denser = less carry, hot air = more carry
        # Standard: ~2 yards per 10F from 70F baseline
        temp_edge = 0.0
        if temp_f < 50:
            # Cold: everyone loses carry, but distance-dependent players hurt more
            dd = player.get("driving_distance", 295)
            if dd > 310:
                temp_edge = -0.02  # long hitters affected more by carry loss
            elif dd < 280:
                temp_edge = 0.01   # short hitters relatively less affected
        elif temp_f > 95:
            dd = player.get("driving_distance", 295)
            if dd > 310:
                temp_edge = 0.01   # long hitters gain even more carry

        signal = altitude_edge + humidity_edge + condition_edge + temp_edge

        return round(float(signal), 4)

    def get_mechanism(self) -> str:
        return (
            "Three atmospheric channels: (1) Altitude — ball carries 2% further "
            "per 1000ft but spin reduces 3%, disadvantaging high-spin approach "
            "players.  (2) Humidity/air density — humid air is less dense, "
            "affecting carry and spin bite differently by player style.  "
            "(3) Course condition type (links/desert/mountain) — each surface "
            "type rewards different skill sets.  Low-trajectory players gain "
            "on links; bombers gain on open desert courses.  Sportsbooks use "
            "generic course difficulty without modeling atmospheric physics."
        )

    def get_decay_risk(self) -> str:
        return (
            "LOW — Atmospheric physics does not change.  Course elevation and "
            "condition type are permanent features.  Player-specific adjustments "
            "(trajectory, spin rate) are stable within seasons.  The edge "
            "persists as long as the market ignores granular atmospheric modeling."
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
