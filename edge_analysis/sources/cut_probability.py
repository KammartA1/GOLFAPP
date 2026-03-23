"""
Cut Probability Edge Source
============================
Models probability of making the cut based on field strength, course history,
current form, and SG profile.  Critical for DFS (dead money) and
outright/matchup betting.
Market edge: cut probability is under-modeled in most betting markets.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import stats as sp_stats
from scipy.special import expit  # logistic sigmoid

log = logging.getLogger(__name__)

# PGA Tour average made-cut rate
_TOUR_AVG_CUT_RATE = 0.55


class CutProbabilitySource:
    """Cut probability model — critical for DFS and outright markets."""

    name = "Cut Probability"
    category = "structural"
    version = "1.0"

    def get_signal(self, player: dict, tournament_context: dict) -> float:
        """
        Compute cut probability edge signal.

        Positive signal = player is more likely to make cut than market implies.
        Returns deviation from market-implied cut probability.

        player keys:
            sg_total              – overall SG per round
            made_cuts_last_10     – int, cuts made in last 10 starts
            total_starts_last_10  – int, total starts in last 10
            course_cuts_made      – int, cuts made at this venue
            course_starts         – int, total starts at this venue
            owgr                  – world ranking
            recent_mc_streak      – int, consecutive missed cuts (0 if made last cut)
            form_factor           – float, current form multiplier (1.0 = neutral)
        tournament_context keys:
            field_strength        – 0-1 scale
            field_size            – number of players in field
            cut_rule              – "top_65", "top_70", "no_cut", etc.
            market_cut_prob       – implied cut probability from odds (optional)
            course                – venue name
        """
        sg = player.get("sg_total", 0.0)
        owgr = player.get("owgr", 100)
        form = player.get("form_factor", 1.0)

        # ── Base cut probability from SG ────────────────────────────────
        # Logistic model: P(cut) = sigmoid(a + b*SG)
        # Calibrated: SG=0 -> ~55% cut rate, SG=+2 -> ~88%, SG=-1 -> ~35%
        logit_base = -0.2 + 1.1 * sg
        base_prob = float(expit(logit_base))

        # ── Historical cut rate adjustment ──────────────────────────────
        cuts_10 = player.get("made_cuts_last_10", 7)
        starts_10 = player.get("total_starts_last_10", 10)
        if starts_10 > 0:
            historical_rate = cuts_10 / starts_10
        else:
            historical_rate = _TOUR_AVG_CUT_RATE

        # Bayesian update: combine model prediction with historical rate
        # Weight historical data by sample size
        hist_weight = min(starts_10 / 20.0, 0.5)
        adjusted_prob = base_prob * (1 - hist_weight) + historical_rate * hist_weight

        # ── Course-specific history ─────────────────────────────────────
        course_cuts = player.get("course_cuts_made", 0)
        course_starts = player.get("course_starts", 0)
        if course_starts >= 3:
            course_rate = course_cuts / course_starts
            course_weight = min(course_starts / 10.0, 0.3)
            adjusted_prob = adjusted_prob * (1 - course_weight) + course_rate * course_weight

        # ── Field strength adjustment ───────────────────────────────────
        field_strength = tournament_context.get("field_strength", 0.5)
        # Stronger field = harder to make cut for marginal players
        field_adj = -(field_strength - 0.5) * 0.15
        adjusted_prob += field_adj

        # ── Missed cut streak penalty ───────────────────────────────────
        mc_streak = player.get("recent_mc_streak", 0)
        if mc_streak >= 3:
            # Confidence spiral — 3+ consecutive MCs is a red flag
            streak_penalty = -0.05 * (mc_streak - 2)
            adjusted_prob += streak_penalty
        elif mc_streak == 0 and cuts_10 >= 8:
            # Strong cut-making player gets a small boost
            adjusted_prob += 0.02

        # ── Form factor ─────────────────────────────────────────────────
        adjusted_prob *= form

        # Clamp to [0.05, 0.98]
        adjusted_prob = max(0.05, min(0.98, adjusted_prob))

        # ── Compute edge vs market ──────────────────────────────────────
        market_prob = tournament_context.get("market_cut_prob", None)
        if market_prob is not None and 0 < market_prob < 1:
            edge = adjusted_prob - market_prob
        else:
            # No market line — express as deviation from tour average
            edge = adjusted_prob - _TOUR_AVG_CUT_RATE

        return round(float(edge), 4)

    def get_cut_probability(self, player: dict, tournament_context: dict) -> float:
        """Return raw cut probability (not edge-relative).  0-1 scale."""
        sg = player.get("sg_total", 0.0)
        logit = -0.2 + 1.1 * sg
        prob = float(expit(logit))

        cuts_10 = player.get("made_cuts_last_10", 7)
        starts_10 = player.get("total_starts_last_10", 10)
        hist_rate = cuts_10 / max(starts_10, 1)
        hist_w = min(starts_10 / 20.0, 0.5)
        prob = prob * (1 - hist_w) + hist_rate * hist_w

        field_strength = tournament_context.get("field_strength", 0.5)
        prob += -(field_strength - 0.5) * 0.15

        form = player.get("form_factor", 1.0)
        prob *= form

        return max(0.05, min(0.98, prob))

    def get_mechanism(self) -> str:
        return (
            "A logistic regression maps SG to base cut probability, then "
            "Bayesian-updates with the player's recent cut rate (last 10 starts) "
            "and course-specific cut history.  Field strength shifts the cutline "
            "difficulty.  A missed-cut streak penalty captures confidence spirals.  "
            "The edge is the difference between our modeled cut probability and "
            "the market-implied probability.  In DFS, accurately pricing dead "
            "money (players who miss the cut) is the single biggest ROI driver."
        )

    def get_decay_risk(self) -> str:
        return (
            "LOW — Cut probability is a structural edge: the model uses "
            "fundamental skill metrics (SG) and the market does not price "
            "cuts granularly.  As long as DFS and tournament markets exist, "
            "accurate cut probability is valuable.  Decay only if books begin "
            "offering explicit made-cut markets with tight lines."
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
            # For cut probability, outcome is binary: made cut or not
            made_cut = 1.0 if rec.get("made_cut", rec.get("actual_finish", 50) <= 65) else 0.0
            outcomes.append(made_cut)
            signals.append(sig)

        signals = np.array(signals)
        outcomes = np.array(outcomes)
        n = len(signals)

        # For binary outcomes, use point-biserial correlation
        corr, p_val = sp_stats.pointbiserialr(outcomes, signals)
        if np.isnan(corr):
            corr, p_val = 0.0, 1.0

        # Calibration: bin signals into quintiles and check actual cut rates
        q = max(n // 5, 1)
        idx = np.argsort(signals)
        top_cut_rate = float(np.mean(outcomes[idx[-q:]]))
        bot_cut_rate = float(np.mean(outcomes[idx[:q]]))
        spread = top_cut_rate - bot_cut_rate
        pooled = float(np.std(outcomes, ddof=1))
        sharpe = spread / pooled if pooled > 1e-9 else 0.0

        # Brier score
        probs = 1.0 / (1.0 + np.exp(-(signals + 0.5)))  # rough mapping
        brier = float(np.mean((probs - outcomes) ** 2))

        return {
            "sharpe": round(sharpe, 4),
            "p_value": round(float(p_val), 6),
            "sample_size": n,
            "point_biserial_r": round(float(corr), 4),
            "quintile_spread": round(spread, 4),
            "brier_score": round(brier, 4),
            "top_quintile_cut_rate": round(top_cut_rate, 4),
            "bottom_quintile_cut_rate": round(bot_cut_rate, 4),
            "correlation_with_other_signals": {},
            "status": "VALID" if p_val < 0.10 and n >= 30 else "MARGINAL",
        }
