"""
Driving Distance Edge Source
==============================
Course-specific driving distance advantage.
Long courses favour bombers; tight courses favour accuracy.
Market edge: sportsbooks use generic distance rankings, not course-specific
distance advantage calculations.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import stats as sp_stats

log = logging.getLogger(__name__)

# Course yardage thresholds (par 72 equivalent)
_SHORT_COURSE = 7050
_MEDIUM_COURSE = 7250
_LONG_COURSE = 7450

# Average PGA Tour driving distance (yards, 2024 baseline)
_TOUR_AVG_DISTANCE = 299.5
_TOUR_STD_DISTANCE = 12.0


def _load_course_profiles() -> dict:
    try:
        from config.courses import COURSE_PROFILES
        return COURSE_PROFILES
    except ImportError:
        return {}


class DrivingDistanceSource:
    """Course-specific driving distance advantage edge."""

    name = "Driving Distance"
    category = "skill"
    version = "1.0"

    def __init__(self):
        self._profiles = _load_course_profiles()

    def get_signal(self, player: dict, tournament_context: dict) -> float:
        """
        Compute driving distance edge for player at this course.

        player keys:
            driving_distance     – average driving distance (yards)
            driving_accuracy     – fairway hit % (0-1)
            sg_ott               – strokes gained off the tee
        tournament_context keys:
            course               – venue name
            course_yardage       – total yardage (par 72 equivalent)
            avg_fairway_width    – average fairway width (yards, optional)
            par5_reachable       – number of reachable par 5s in two (optional)
        """
        course = tournament_context.get("course", "")
        profile = self._profiles.get(course, {})

        player_dist = player.get("driving_distance", _TOUR_AVG_DISTANCE)
        player_acc = player.get("driving_accuracy", 0.62)
        sg_ott = player.get("sg_ott", 0.0)

        # ── Course distance demand ──────────────────────────────────────
        yardage = tournament_context.get("course_yardage", 7200)
        distance_bonus = profile.get("distance_bonus", 0.5)
        accuracy_penalty = profile.get("accuracy_penalty", 0.5)

        # Z-score of player distance vs tour average
        dist_z = (player_dist - _TOUR_AVG_DISTANCE) / _TOUR_STD_DISTANCE

        # Long course amplifies distance advantage
        if yardage >= _LONG_COURSE:
            course_dist_factor = 1.4
        elif yardage >= _MEDIUM_COURSE:
            course_dist_factor = 1.0
        elif yardage >= _SHORT_COURSE:
            course_dist_factor = 0.6
        else:
            course_dist_factor = 0.3

        # Distance edge: how much this course rewards the player's length
        dist_edge = dist_z * distance_bonus * course_dist_factor * 0.15

        # ── Accuracy constraint ─────────────────────────────────────────
        # On tight courses, distance is only useful if you can hit fairways
        fairway_width = tournament_context.get("avg_fairway_width", 30.0)
        tightness = max(0.0, (35.0 - fairway_width) / 15.0)  # 0-1, tight=1

        # Accuracy z-score: higher accuracy = less penalty on tight courses
        tour_avg_acc = 0.62
        acc_z = (player_acc - tour_avg_acc) / 0.07  # std ~7%

        acc_adjustment = 0.0
        if tightness > 0.3:
            # Tight course: penalise inaccurate long hitters
            if dist_z > 0.5 and acc_z < -0.5:
                acc_adjustment = -accuracy_penalty * tightness * 0.12
            # Tight course: reward accurate players
            if acc_z > 0.5:
                acc_adjustment += accuracy_penalty * tightness * 0.08

        # ── Par 5 reachability ──────────────────────────────────────────
        reachable = tournament_context.get("par5_reachable", 2)
        # Long hitters can reach more par 5s in two = birdie/eagle opportunities
        par5_edge = 0.0
        if dist_z > 0.5 and reachable >= 3:
            par5_edge = dist_z * 0.04 * reachable
        elif dist_z < -0.5 and reachable >= 3:
            par5_edge = dist_z * 0.02 * reachable  # short hitters miss opportunities

        # ── Altitude adjustment ─────────────────────────────────────────
        elevation = profile.get("elevation_ft", 0)
        # Ball carries ~2% further per 1000ft of elevation
        altitude_carry_bonus = elevation / 1000.0 * 0.02
        # This reduces the distance advantage (everyone hits further at altitude)
        dist_edge *= (1.0 - altitude_carry_bonus * 0.5)

        signal = dist_edge + acc_adjustment + par5_edge

        return round(float(signal), 4)

    def get_mechanism(self) -> str:
        return (
            "Driving distance is course-dependent: a 310-yard average is worth "
            "much more at a 7,500-yard course with reachable par 5s than a "
            "tight 7,000-yard course.  We compute a player's distance z-score "
            "and scale it by the course's distance bonus factor, yardage "
            "category, and par-5 reachability.  An accuracy constraint ensures "
            "we don't over-value bombers on tight tracks.  Altitude adjustments "
            "normalize for courses where everyone hits further.  The market uses "
            "generic distance rankings without course-specific calibration."
        )

    def get_decay_risk(self) -> str:
        return (
            "LOW — Driving distance advantages are structural and tied to course "
            "architecture.  Player distances are stable within a season.  Risk: "
            "rollback of golf ball or driver regulations could compress distance "
            "variance, reducing the signal's magnitude."
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
