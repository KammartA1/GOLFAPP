"""
SG Decomposition Edge Source
=============================
Decomposes Strokes Gained into OTT / APP / ATG / PUTT and weights each
component by course-specific demands.  Market edge: sportsbooks use total SG;
we weight each component by how much that specific course rewards it.

Signal: positive when a player's SG profile is stronger than their total SG
suggests, after applying course-specific weights.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from scipy import stats as sp_stats

log = logging.getLogger(__name__)

# Default course weights when no profile is available
_DEFAULT_WEIGHTS = {"sg_ott": 0.25, "sg_app": 0.35, "sg_atg": 0.20, "sg_putt": 0.20}

_SG_KEYS = ["sg_ott", "sg_app", "sg_atg", "sg_putt"]


def _load_course_profiles() -> dict:
    """Lazy-load from config/courses.py."""
    try:
        from config.courses import COURSE_PROFILES
        return COURSE_PROFILES
    except ImportError:
        return {}


class SGDecompositionSource:
    """Course-weighted SG decomposition edge."""

    name = "SG Decomposition"
    category = "skill"
    version = "1.0"

    def __init__(self):
        self._profiles = _load_course_profiles()

    # ── public API ───────────────────────────────────────────────────────

    def get_signal(self, player: dict, tournament_context: dict) -> float:
        """
        Return edge signal in strokes-gained units.

        Parameters
        ----------
        player : dict
            Must contain sg_ott, sg_app, sg_atg, sg_putt (floats, per-round averages).
        tournament_context : dict
            Must contain 'course' (str).  Optionally 'field_avg_sg' for centering.

        Returns
        -------
        float   Positive = player has hidden edge at this course.
        """
        course = tournament_context.get("course", "")
        weights = self._course_weights(course)

        sg_vec = np.array([player.get(k, 0.0) for k in _SG_KEYS], dtype=float)
        w_vec = np.array([weights[k] for k in _SG_KEYS], dtype=float)

        # Course-weighted SG (rescaled so weights sum to 4, i.e. full SG scale)
        weighted_sg = float(np.dot(sg_vec, w_vec) * 4.0)

        # Market perception: simple total SG (equal weight)
        total_sg = float(sg_vec.sum())

        # Edge = how much better the player is when you weight properly
        edge = weighted_sg - total_sg

        # Apply recency multiplier if available
        recency = player.get("form_factor", 1.0)
        edge *= recency

        return round(edge, 4)

    def get_mechanism(self) -> str:
        return (
            "Sportsbooks price golfers using total Strokes Gained, which treats "
            "all four SG components equally.  In reality, course architecture "
            "heavily rewards specific components: approach-dominated courses like "
            "Augusta inflate approach-strong players while de-emphasizing putting.  "
            "By weighting SG components against the course's historical demand "
            "profile, we identify players whose true expected performance diverges "
            "from the market's flat-weighted view."
        )

    def get_decay_risk(self) -> str:
        return (
            "LOW — Course architecture changes slowly (once every 5-10 years for "
            "major renovations).  SG component data is publicly available but "
            "decomposing it by course is not standard market practice.  Edge "
            "decays only if books begin publishing course-weighted SG."
        )

    def validate(self, historical_data: list[dict]) -> dict:
        """
        Validate on historical tournament results.

        Each entry in *historical_data* must contain:
            player (dict with SG keys), tournament_context (dict with 'course'),
            actual_finish (int, lower = better).

        Returns dict with sharpe, p_value, sample_size, correlation_with_other_signals.
        """
        if len(historical_data) < 30:
            return {
                "sharpe": 0.0,
                "p_value": 1.0,
                "sample_size": len(historical_data),
                "correlation_with_other_signals": {},
                "status": "INSUFFICIENT_DATA",
            }

        signals = []
        outcomes = []
        for rec in historical_data:
            sig = self.get_signal(rec.get("player", {}), rec.get("tournament_context", {}))
            finish = rec.get("actual_finish", 50)
            # Convert finish position to a return-like metric: lower finish = higher return
            ret = (50.0 - finish) / 50.0
            signals.append(sig)
            outcomes.append(ret)

        signals = np.array(signals, dtype=float)
        outcomes = np.array(outcomes, dtype=float)

        # Quintile spread: top quintile signal vs bottom quintile outcome
        n = len(signals)
        q = max(n // 5, 1)
        idx = np.argsort(signals)
        top_outcomes = outcomes[idx[-q:]]
        bot_outcomes = outcomes[idx[:q]]

        mean_diff = float(np.mean(top_outcomes) - np.mean(bot_outcomes))
        pooled_std = float(np.std(np.concatenate([top_outcomes, bot_outcomes]), ddof=1))
        sharpe = mean_diff / pooled_std if pooled_std > 1e-9 else 0.0

        # Spearman rank correlation test
        corr, p_val = sp_stats.spearmanr(signals, outcomes)
        if np.isnan(corr):
            corr, p_val = 0.0, 1.0

        return {
            "sharpe": round(float(sharpe), 4),
            "p_value": round(float(p_val), 6),
            "sample_size": n,
            "spearman_r": round(float(corr), 4),
            "quintile_spread": round(mean_diff, 4),
            "correlation_with_other_signals": {},
            "status": "VALID" if p_val < 0.10 and n >= 30 else "MARGINAL",
        }

    # ── internals ────────────────────────────────────────────────────────

    def _course_weights(self, course: str) -> dict:
        """Look up course weight vector; fall back to default."""
        profile = self._profiles.get(course, {})
        raw = profile.get("sg_weights", _DEFAULT_WEIGHTS)
        # Normalise to sum=1
        total = sum(raw.values())
        if total < 1e-9:
            return _DEFAULT_WEIGHTS
        return {k: raw.get(k, 0.25) / total for k in _SG_KEYS}
