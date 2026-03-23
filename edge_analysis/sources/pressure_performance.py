"""
Pressure Performance Edge Source
=================================
Major championship vs regular tour event performance differential.
Some players elevate in majors, others shrink.
Market edge: sportsbooks weight all events equally in their models.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import stats as sp_stats

log = logging.getLogger(__name__)

_MAJOR_NAMES = {
    "Masters", "The Masters", "Masters Tournament",
    "PGA Championship", "The PGA Championship",
    "U.S. Open", "US Open",
    "The Open Championship", "The Open", "British Open",
}

_HIGH_PRESSURE_EVENTS = _MAJOR_NAMES | {
    "THE PLAYERS Championship", "Players Championship",
    "Tour Championship",
    "WGC-Match Play",
    "Memorial Tournament",
    "Arnold Palmer Invitational",
}

# Pressure tiers: majors = 1.0, signature = 0.7, regular = 0.3
_PRESSURE_TIER = {
    "major": 1.0,
    "signature": 0.7,
    "playoff": 0.9,
    "regular": 0.3,
}


class PressurePerformanceSource:
    """Major/high-pressure performance differential edge."""

    name = "Pressure Performance"
    category = "psychological"
    version = "1.0"

    def get_signal(self, player: dict, tournament_context: dict) -> float:
        """
        Compute pressure-performance edge.

        player keys:
            major_sg           – average SG in majors
            regular_sg         – average SG in regular events
            major_starts       – number of major starts
            pressure_sg        – SG in all high-pressure events (optional)
            contention_sg      – SG when within 5 shots of lead entering final round
            baseline_sg        – overall SG baseline
        tournament_context keys:
            tournament_name    – name of tournament
            event_type         – "major", "signature", "regular", "playoff"
            is_major           – bool (alternative to event_type)
            round_number       – 1-4
            player_position    – current leaderboard position (for rounds 2-4)
            shots_behind_lead  – strokes back of leader
        """
        event_type = tournament_context.get("event_type", "regular")
        tourney_name = tournament_context.get("tournament_name", "")
        is_major = tournament_context.get("is_major", False)

        # Detect major from name if not explicitly set
        if tourney_name in _MAJOR_NAMES or tourney_name in _HIGH_PRESSURE_EVENTS:
            is_major = True
        if is_major and event_type == "regular":
            event_type = "major"

        pressure_level = _PRESSURE_TIER.get(event_type, 0.3)

        # Core differential: how player performs under pressure vs baseline
        major_sg = player.get("major_sg", None)
        regular_sg = player.get("regular_sg", player.get("baseline_sg", 0.0))
        baseline = player.get("baseline_sg", regular_sg)

        major_starts = player.get("major_starts", 0)

        if major_sg is not None and major_starts >= 8:
            # Enough data for reliable major SG estimate
            pressure_diff = major_sg - baseline
        elif major_sg is not None and major_starts >= 4:
            # Partial data — shrink toward zero (Bayesian)
            shrinkage = major_starts / 12.0  # full weight at 12 starts
            pressure_diff = (major_sg - baseline) * shrinkage
        else:
            # Use pressure_sg if available, else assume zero differential
            pressure_sg = player.get("pressure_sg", baseline)
            pressure_diff = (pressure_sg - baseline) * 0.5

        # Scale by how much pressure this event carries
        edge = pressure_diff * pressure_level

        # ── Contention bonus: in-contention performance ─────────────────
        round_num = tournament_context.get("round_number", 1)
        shots_back = tournament_context.get("shots_behind_lead", 99)

        if round_num >= 3 and shots_back <= 5:
            contention_sg = player.get("contention_sg", baseline)
            contention_diff = (contention_sg - baseline) * 0.5
            # Amplify for final round
            if round_num == 4:
                contention_diff *= 1.5
            edge += contention_diff * 0.3

        # ── Sunday performance factor ───────────────────────────────────
        sunday_sg = player.get("sunday_sg", None)
        if round_num == 4 and sunday_sg is not None:
            sunday_diff = (sunday_sg - baseline) * 0.3
            edge += sunday_diff

        return round(float(edge), 4)

    def get_mechanism(self) -> str:
        return (
            "Some players consistently perform above their baseline in high-"
            "pressure environments (majors, contention, Sunday final groups) "
            "while others shrink.  We track major SG differential with "
            "Bayesian shrinkage for small samples.  The market weights all "
            "events equally in its SG models, so a player whose +2.0 SG comes "
            "mostly from weak fields is priced the same as one who earns it "
            "in majors.  We also model in-contention performance (within 5 "
            "shots entering round 3-4) and Sunday-specific SG."
        )

    def get_decay_risk(self) -> str:
        return (
            "LOW — Pressure performance is deeply psychological and highly "
            "persistent across careers.  Top pressure performers (Tiger, Rory, "
            "Scheffler) show consistent elevation over 10+ year spans.  The "
            "data is publicly available but the differential calculation with "
            "Bayesian shrinkage and contention modeling is not standard."
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
