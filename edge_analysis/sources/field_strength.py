"""
Field Strength Edge Source
============================
Normalizes SG for field quality using OWGR-based field strength ratings.
A +2.0 SG in a weak field is not the same as +2.0 in a major.
Market edge: the public overvalues stats from weak fields.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import stats as sp_stats

log = logging.getLogger(__name__)

# OWGR field strength tiers (approximate)
_FIELD_TIERS = {
    "major":          {"strength": 0.95, "sg_deflator": 0.70},
    "signature":      {"strength": 0.85, "sg_deflator": 0.80},
    "elevated":       {"strength": 0.75, "sg_deflator": 0.85},
    "regular":        {"strength": 0.55, "sg_deflator": 1.00},
    "alternate":      {"strength": 0.35, "sg_deflator": 1.20},
    "opposite_field": {"strength": 0.25, "sg_deflator": 1.35},
}

# Average OWGR of top-50 players in different field types
_OWGR_BENCHMARKS = {
    "major": 15.0,
    "signature": 25.0,
    "elevated": 40.0,
    "regular": 70.0,
    "alternate": 120.0,
    "opposite_field": 150.0,
}


class FieldStrengthSource:
    """Field-strength adjusted SG normalization edge."""

    name = "Field Strength"
    category = "structural"
    version = "1.0"

    def get_signal(self, player: dict, tournament_context: dict) -> float:
        """
        Compute field-strength-adjusted edge.

        Positive = player's true SG is higher than their raw stats suggest
        (they've been playing in strong fields).
        Negative = player's SG is inflated from weak-field events.

        player keys:
            sg_total                 – raw SG per round (unadjusted)
            recent_results           – list of recent event dicts with
                                       'sg_total' and 'field_strength'
            avg_field_strength       – weighted avg field strength of events played
            owgr                     – world ranking
        tournament_context keys:
            field_strength           – 0-1 scale for THIS week's event
            event_type               – "major", "signature", "regular", etc.
            field_avg_owgr           – average OWGR of the field (optional)
        """
        sg_raw = player.get("sg_total", 0.0)
        avg_fs = player.get("avg_field_strength", 0.5)
        this_fs = tournament_context.get("field_strength", 0.5)
        event_type = tournament_context.get("event_type", "regular")

        # ── SG deflation factor ─────────────────────────────────────────
        # If player has been playing weak fields, their SG is inflated
        tier_info = _FIELD_TIERS.get(event_type, _FIELD_TIERS["regular"])
        deflator = tier_info["sg_deflator"]

        # Compare player's average field strength to this event's field
        fs_diff = this_fs - avg_fs

        # ── Compute field-adjusted SG ───────────────────────────────────
        # Core insight: SG in a 0.25-strength field is worth only ~70% of
        # SG in a 0.95-strength field.  We normalize.
        if avg_fs > 0.05:
            # Adjustment factor: how much harder is THIS field vs player's avg
            adjustment_ratio = this_fs / avg_fs
            # Dampen extreme adjustments
            adjustment_ratio = max(0.5, min(2.0, adjustment_ratio))
        else:
            adjustment_ratio = 1.0

        # Recent results field-adjusted SG
        results = player.get("recent_results", [])
        if results:
            adjusted_sgs = []
            for r in results[:20]:
                r_sg = r.get("sg_total", 0.0)
                r_fs = r.get("field_strength", 0.5)
                # Normalize: weak-field SG is deflated
                if r_fs > 0.05:
                    # Scale SG by field strength relative to a major-level field
                    normalized = r_sg * (r_fs / 0.90)  # 0.90 as major benchmark
                else:
                    normalized = r_sg * 0.5
                adjusted_sgs.append(normalized)

            adjusted_mean = float(np.mean(adjusted_sgs))
        else:
            adjusted_mean = sg_raw * (avg_fs / 0.90) if avg_fs > 0.05 else sg_raw * 0.5

        # ── Edge: adjusted vs raw ──────────────────────────────────────
        # If adjusted > raw: player is UNDER-valued (played strong fields)
        # If adjusted < raw: player is OVER-valued (inflated by weak fields)
        edge = adjusted_mean - sg_raw

        # ── OWGR validation ─────────────────────────────────────────────
        owgr = player.get("owgr", 100)
        field_avg_owgr = tournament_context.get("field_avg_owgr", 70)

        # OWGR cross-check: if player's OWGR is much better than the field,
        # they're a big fish in a small pond — their raw SG will be inflated
        if field_avg_owgr > 0:
            owgr_ratio = owgr / field_avg_owgr
            if owgr_ratio < 0.5:
                # Player is much better than the field — SG will be high but
                # partly due to weak competition
                edge -= 0.05 * (0.5 - owgr_ratio)

        return round(float(edge), 4)

    def get_mechanism(self) -> str:
        return (
            "PGA Tour SG is computed relative to the field in each event.  "
            "A +2.0 SG against a 150-average-OWGR field is fundamentally "
            "different from +2.0 against a major-strength field.  We normalize "
            "each result by field strength: SG is scaled by (field_strength / 0.90), "
            "using major fields as the benchmark.  Players who pad stats in "
            "weak-field 'opposite field' events are over-priced by the market; "
            "players who consistently compete in majors/signature events are "
            "under-priced.  The edge is the gap between field-adjusted SG and "
            "raw SG."
        )

    def get_decay_risk(self) -> str:
        return (
            "LOW — Field strength disparity is a structural feature of the "
            "PGA Tour schedule.  As long as opposite-field events coexist with "
            "majors, this normalization adds value.  Risk: if the Tour eliminates "
            "weak fields entirely (unlikely given the 2024 schedule expansion)."
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
