"""
Course Fit Edge Source
=======================
Scores each player's SG profile against 50+ course weight vectors using
cosine similarity and Euclidean distance.  Market edge: the public sees
"good golfer" not "good golfer FOR THIS COURSE."
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import stats as sp_stats
from scipy.spatial.distance import cosine as cosine_dist

log = logging.getLogger(__name__)

_SG_KEYS = ["sg_ott", "sg_app", "sg_atg", "sg_putt"]


def _load_course_profiles() -> dict:
    try:
        from config.courses import COURSE_PROFILES
        return COURSE_PROFILES
    except ImportError:
        return {}


class CourseFitSource:
    """Player-course fit via SG profile similarity."""

    name = "Course Fit"
    category = "skill"
    version = "1.0"

    def __init__(self):
        self._profiles = _load_course_profiles()

    def get_signal(self, player: dict, tournament_context: dict) -> float:
        """
        Compute course-fit score.

        Returns a value in [-1, 1].  Positive = player's SG strengths align
        with what the course rewards.  Negative = mismatch.
        """
        course = tournament_context.get("course", "")
        profile = self._profiles.get(course, {})
        weights = profile.get("sg_weights", {k: 0.25 for k in _SG_KEYS})

        # Course demand vector (what the course rewards)
        demand = np.array([weights.get(k, 0.25) for k in _SG_KEYS], dtype=float)
        demand_sum = demand.sum()
        if demand_sum > 1e-9:
            demand = demand / demand_sum

        # Player SG vector (z-score normalised across the field if possible)
        sg_raw = np.array([player.get(k, 0.0) for k in _SG_KEYS], dtype=float)

        # Normalise player vector to unit so cosine similarity is meaningful
        sg_norm = np.linalg.norm(sg_raw)
        if sg_norm < 1e-9:
            return 0.0

        # Cosine similarity: how well player strengths align with course demands
        cos_sim = 1.0 - cosine_dist(sg_raw, demand)

        # Weight by player's overall SG magnitude (better player + good fit = bigger edge)
        total_sg = float(sg_raw.sum())
        fit_score = cos_sim * total_sg

        # Additional adjustments from course profile
        dist_bonus = profile.get("distance_bonus", 0.5)
        acc_penalty = profile.get("accuracy_penalty", 0.5)
        player_dist_sg = player.get("sg_ott", 0.0)
        player_acc = player.get("driving_accuracy", 0.5)  # 0-1 scale

        # Distance adjustment: if course rewards distance, bonus for long hitters
        dist_adj = dist_bonus * player_dist_sg * 0.3

        # Accuracy adjustment: if course penalises misses, penalty for inaccurate
        acc_adj = -acc_penalty * (1.0 - player_acc) * 0.2

        # Key skills bonus
        key_skills = profile.get("key_skills", [])
        skill_bonus = 0.0
        skill_map = {
            "distance": player.get("sg_ott", 0.0) * 0.1,
            "precise_irons": player.get("sg_app", 0.0) * 0.1,
            "precision_irons": player.get("sg_app", 0.0) * 0.1,
            "scrambling": player.get("sg_atg", 0.0) * 0.12,
            "bermuda_putting": player.get("bermuda_putt_sg", player.get("sg_putt", 0.0)) * 0.08,
            "bentgrass_putting": player.get("sg_putt", 0.0) * 0.08,
            "wind_management": player.get("wind_sg", 0.0) * 0.1,
            "course_management": player.get("sg_app", 0.0) * 0.05,
            "accurate_driving": player.get("driving_accuracy", 0.5) * 0.1,
            "sand_play": player.get("sg_atg", 0.0) * 0.08,
        }
        for skill in key_skills:
            skill_bonus += skill_map.get(skill, 0.0)

        signal = fit_score + dist_adj + acc_adj + skill_bonus

        return round(float(signal), 4)

    def get_mechanism(self) -> str:
        return (
            "Each PGA Tour course has a distinct demand profile across OTT, APP, "
            "ATG, and PUTT.  We score every player's SG vector against the course "
            "demand vector using cosine similarity, then scale by overall SG "
            "magnitude.  Additional adjustments for distance bonus, accuracy "
            "penalty, and course-specific key skills.  The market prices players "
            "by generic ranking; we identify players whose skill shape is "
            "specifically rewarded by this week's venue."
        )

    def get_decay_risk(self) -> str:
        return (
            "LOW — Course profiles change slowly.  The fit calculation itself "
            "is straightforward but the comprehensive course database covering "
            "50+ venues with calibrated weight vectors is the moat.  Risk: major "
            "course renovations (rare, ~1 per year across PGA Tour)."
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
