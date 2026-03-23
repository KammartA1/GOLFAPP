"""
Tournament History Edge Source
================================
Some players repeatedly over/under-perform at specific venues beyond
what SG predicts.  Course memory, comfort, local knowledge.
Uses Bayesian shrinkage for small samples.
Market edge: sportsbooks weight course history too simply.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import stats as sp_stats

log = logging.getLogger(__name__)

# Bayesian shrinkage parameters
# We assume course-specific true talent is drawn from a population with
# mean 0 (no course effect) and variance tau^2.
# Observed differential has variance sigma^2 / n.
# Shrinkage factor = tau^2 / (tau^2 + sigma^2/n)
_TAU_SQ = 0.25       # prior variance of course-specific talent
_SIGMA_SQ = 4.0      # per-round SG variance (typical golfer)
_MIN_STARTS = 2      # minimum starts to produce any signal
_FULL_WEIGHT_STARTS = 12  # enough data for minimal shrinkage


class TournamentHistorySource:
    """Venue-specific over/under-performance with Bayesian shrinkage."""

    name = "Tournament History"
    category = "venue"
    version = "1.0"

    def get_signal(self, player: dict, tournament_context: dict) -> float:
        """
        Compute tournament-history edge.

        player keys:
            course_history    – list of dicts, each with:
                                {'sg_total': float, 'finish': int, 'year': int,
                                 'rounds': int, 'field_strength': float}
            baseline_sg       – long-term SG baseline
            sg_total          – current SG (fallback baseline)
        tournament_context keys:
            course            – venue name
            years_weight      – how much to decay older results (optional)
        """
        history = player.get("course_history", [])
        baseline = player.get("baseline_sg", player.get("sg_total", 0.0))

        if len(history) < _MIN_STARTS:
            return 0.0

        # ── Compute course-specific SG differential ─────────────────────
        # Weight more recent appearances higher (exponential decay by year)
        current_year = 2026
        differentials = []
        weights = []
        total_rounds = 0

        for entry in history:
            sg = entry.get("sg_total", 0.0)
            year = entry.get("year", current_year - 5)
            rounds = entry.get("rounds", 4)
            fs = entry.get("field_strength", 0.5)

            # SG differential vs baseline, field-adjusted
            diff = sg - baseline
            # Adjust for field strength: weak field inflates SG
            diff *= (fs / 0.7)  # normalize to ~regular-field

            # Time decay: half-life of 3 years
            years_ago = current_year - year
            time_weight = math.pow(0.5, years_ago / 3.0)

            # Weight by rounds played (more rounds = more reliable)
            round_weight = math.sqrt(rounds)

            w = time_weight * round_weight
            differentials.append(diff)
            weights.append(w)
            total_rounds += rounds

        if not differentials:
            return 0.0

        differentials = np.array(differentials, dtype=float)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

        # Weighted mean differential
        raw_diff = float(np.dot(weights, differentials))

        # ── Bayesian shrinkage ──────────────────────────────────────────
        # Effective sample size (number of appearances, weighted)
        n_eff = len(history)
        per_appearance_var = _SIGMA_SQ / max(total_rounds / 4.0, 1.0)

        # Shrinkage factor: how much to trust the observed differential
        shrinkage = _TAU_SQ / (_TAU_SQ + per_appearance_var / n_eff)

        # Shrunk estimate: pulled toward zero (population mean)
        shrunk_diff = raw_diff * shrinkage

        # ── Consistency bonus ───────────────────────────────────────────
        # If player consistently over-performs (low variance in differential),
        # the signal is more reliable
        if len(differentials) >= 4:
            diff_std = float(np.std(differentials, ddof=1))
            # Low variance = consistent over-performance = more trustworthy
            consistency = max(0.0, 1.0 - diff_std / 2.0)
            # Boost signal for consistent performers
            shrunk_diff *= (1.0 + 0.2 * consistency)

        # ── Finish position cross-check ─────────────────────────────────
        finishes = [e.get("finish", 50) for e in history]
        if finishes:
            avg_finish = np.mean(finishes)
            # Top-20 average finish at a venue is a strong signal
            finish_signal = max(0.0, (40 - avg_finish) / 40.0) * 0.1
            # Only add if it aligns with SG differential
            if (shrunk_diff > 0 and finish_signal > 0) or (shrunk_diff < 0 and finish_signal < 0):
                shrunk_diff += finish_signal
            elif shrunk_diff > 0 and finish_signal == 0:
                # SG says good but finishes are average — discount
                shrunk_diff *= 0.85

        return round(float(shrunk_diff), 4)

    def get_mechanism(self) -> str:
        return (
            "Some players have persistent venue-specific edges that SG alone "
            "does not explain: course familiarity, putting green type comfort, "
            "local knowledge, psychological attachment.  We compute a "
            "time-weighted SG differential at each venue, then apply Bayesian "
            "shrinkage toward zero to handle small samples.  Shrinkage factor "
            "= tau^2 / (tau^2 + sigma^2/n), calibrated so that 2 starts get "
            "heavy shrinkage while 12+ starts are minimally shrunk.  A "
            "consistency bonus amplifies the signal for players with low "
            "variance in their course-specific differential.  The market uses "
            "raw course history stats without shrinkage or consistency modeling."
        )

    def get_decay_risk(self) -> str:
        return (
            "MEDIUM — Course renovations (e.g. Pinehurst 2024 redesign) can "
            "invalidate historical data.  Player age and swing changes reduce "
            "relevance of old data.  Our time-decay (3-year half-life) and "
            "Bayesian shrinkage mitigate this, but the signal should be "
            "monitored for courses undergoing major changes."
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
