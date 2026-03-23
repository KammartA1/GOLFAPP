"""Informational edge component — Timing analysis per data source.

Measures the timing advantage from different information sources:
  - Weather data (forecasts available before market adjusts)
  - Injury/withdrawal news
  - Strokes gained updates (recent form data)
  - Course history lookups
  - Tee time / wave assignments

The key question: Does having information X earlier than the market
translate into actual betting edge?
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import timedelta
from typing import List

import numpy as np

from edge_analysis.schemas import GolfBetRecord, EdgeComponent

logger = logging.getLogger(__name__)

# Data source categories and their typical edge windows
DATA_SOURCES = {
    "weather": {
        "name": "Weather Forecasts",
        "typical_lead_hours": 24,
        "decay_hours": 6,
        "mechanism": "Wind/rain forecasts before market adjusts lines",
    },
    "injury": {
        "name": "Injury / Withdrawal",
        "typical_lead_hours": 2,
        "decay_hours": 1,
        "mechanism": "WD/injury news before field odds update",
    },
    "strokes_gained": {
        "name": "Strokes Gained Updates",
        "typical_lead_hours": 48,
        "decay_hours": 24,
        "mechanism": "Fresh SG data from recent tournaments before models reprice",
    },
    "course_history": {
        "name": "Course History",
        "typical_lead_hours": 168,
        "decay_hours": 72,
        "mechanism": "Venue-specific performance data not fully priced by books",
    },
    "tee_times": {
        "name": "Tee Time / Wave",
        "typical_lead_hours": 12,
        "decay_hours": 4,
        "mechanism": "AM/PM wave assignment + weather interaction",
    },
    "ownership": {
        "name": "DFS Ownership Projections",
        "typical_lead_hours": 6,
        "decay_hours": 2,
        "mechanism": "Leverage spots where public is overweight/underweight",
    },
}


class InformationalAnalyzer:
    """Analyze information timing advantage per data source."""

    def analyze(self, records: List[GolfBetRecord]) -> EdgeComponent:
        """Compute informational edge — which data sources drive value?"""
        if not records:
            return EdgeComponent(
                name="informational", value=0.0, confidence=0.0,
                verdict="No data to analyze.",
            )

        source_analysis = {}
        total_info_edge = 0.0
        n_with_timing = 0

        # Analyze each data source's contribution
        for source_key, source_meta in DATA_SOURCES.items():
            result = self._analyze_source(records, source_key, source_meta)
            source_analysis[source_key] = result
            total_info_edge += result["edge_contribution"]
            if result["n_bets_with_source"] > 0:
                n_with_timing += 1

        # Timing analysis: bets placed early vs late
        timing_analysis = self._timing_analysis(records)

        # Overall informational edge in cents
        info_edge_cents = total_info_edge * 100

        confidence = min(len(records) / 150.0, 1.0) * min(n_with_timing / 3.0, 1.0)

        # Verdict
        best_source = max(source_analysis.items(),
                          key=lambda x: x[1]["edge_contribution"],
                          default=("none", {"edge_contribution": 0}))

        if info_edge_cents > 2.0:
            verdict = (f"Strong informational edge ({info_edge_cents:.1f}c). "
                       f"Best source: {best_source[0]} ({best_source[1]['edge_contribution']*100:.1f}c)")
        elif info_edge_cents > 0.5:
            verdict = (f"Moderate informational edge ({info_edge_cents:.1f}c). "
                       f"Best source: {best_source[0]}")
        else:
            verdict = f"Minimal informational edge ({info_edge_cents:.1f}c)"

        return EdgeComponent(
            name="informational",
            value=round(info_edge_cents, 2),
            confidence=round(confidence, 3),
            details={
                "by_source": source_analysis,
                "timing_analysis": timing_analysis,
                "n_sources_contributing": n_with_timing,
                "best_source": best_source[0],
            },
            verdict=verdict,
        )

    def _analyze_source(
        self,
        records: List[GolfBetRecord],
        source_key: str,
        source_meta: dict,
    ) -> dict:
        """Analyze edge contribution from a specific data source."""
        # Partition bets by whether this source was available
        with_source = [r for r in records if source_key in r.data_sources_available]
        without_source = [r for r in records if source_key not in r.data_sources_available]

        n_with = len(with_source)
        n_without = len(without_source)

        if n_with < 5:
            return {
                "n_bets_with_source": n_with,
                "n_bets_without_source": n_without,
                "edge_contribution": 0.0,
                "clv_with": 0.0,
                "clv_without": 0.0,
                "mechanism": source_meta["mechanism"],
                "confidence": 0.0,
                "verdict": "Insufficient data",
            }

        # CLV comparison: bets with this source vs without
        clv_with = np.mean([r.clv_cents for r in with_source])
        clv_without = np.mean([r.clv_cents for r in without_source]) if without_source else 0.0
        clv_diff = clv_with - clv_without

        # Win rate comparison
        wr_with = np.mean([r.actual_outcome for r in with_source])
        wr_without = np.mean([r.actual_outcome for r in without_source]) if without_source else 0.5

        # Edge contribution: portion of CLV attributable to this source
        # Estimated as CLV difference normalized by total CLV
        total_clv = np.mean([r.clv_cents for r in records])
        if total_clv > 0 and clv_diff > 0:
            edge_contribution = clv_diff / 100.0  # Convert cents to probability
        else:
            edge_contribution = max(0.0, clv_diff / 100.0)

        # Timing within source: did early bets (relative to source availability) do better?
        timing_edge = self._source_timing_edge(with_source, source_meta)

        confidence = min(n_with / 50.0, 1.0)

        return {
            "n_bets_with_source": n_with,
            "n_bets_without_source": n_without,
            "edge_contribution": round(edge_contribution, 4),
            "clv_with": round(float(clv_with), 2),
            "clv_without": round(float(clv_without), 2),
            "clv_diff": round(float(clv_diff), 2),
            "win_rate_with": round(float(wr_with), 4),
            "win_rate_without": round(float(wr_without), 4),
            "mechanism": source_meta["mechanism"],
            "timing_edge": timing_edge,
            "confidence": round(confidence, 3),
        }

    def _source_timing_edge(self, records: List[GolfBetRecord], source_meta: dict) -> dict:
        """Analyze if acting faster on a data source yields more edge."""
        timed = [r for r in records if r.bet_timestamp and r.signal_timestamp]
        if len(timed) < 10:
            return {"has_data": False}

        # Split into early vs late actors
        lead_hours = []
        for r in timed:
            dt = (r.bet_timestamp - r.signal_timestamp).total_seconds() / 3600.0
            lead_hours.append(dt)

        lead_arr = np.array(lead_hours)
        median_lead = float(np.median(lead_arr))

        early = [r for r, lh in zip(timed, lead_hours) if lh <= median_lead]
        late = [r for r, lh in zip(timed, lead_hours) if lh > median_lead]

        clv_early = float(np.mean([r.clv_cents for r in early])) if early else 0.0
        clv_late = float(np.mean([r.clv_cents for r in late])) if late else 0.0

        return {
            "has_data": True,
            "median_lead_hours": round(median_lead, 1),
            "clv_early_bets": round(clv_early, 2),
            "clv_late_bets": round(clv_late, 2),
            "timing_premium": round(clv_early - clv_late, 2),
            "decay_hours": source_meta["decay_hours"],
        }

    def _timing_analysis(self, records: List[GolfBetRecord]) -> dict:
        """Overall bet timing analysis: do earlier bets perform better?"""
        timed = [r for r in records if r.bet_timestamp and r.closing_timestamp]
        if len(timed) < 10:
            return {"has_data": False}

        # Hours before close
        hours_before = []
        clvs = []
        for r in timed:
            dt = (r.closing_timestamp - r.bet_timestamp).total_seconds() / 3600.0
            if dt > 0:
                hours_before.append(dt)
                clvs.append(r.clv_cents)

        if len(hours_before) < 10:
            return {"has_data": False}

        hours_arr = np.array(hours_before)
        clv_arr = np.array(clvs)

        # Correlation between time-to-close and CLV
        corr = float(np.corrcoef(hours_arr, clv_arr)[0, 1]) if len(hours_arr) > 2 else 0.0

        # Bucket analysis
        buckets = []
        for low, high, label in [(0, 6, "0-6h"), (6, 24, "6-24h"), (24, 72, "1-3d"), (72, 500, "3d+")]:
            mask = (hours_arr >= low) & (hours_arr < high)
            n = int(mask.sum())
            if n > 0:
                buckets.append({
                    "window": label,
                    "n_bets": n,
                    "avg_clv": round(float(clv_arr[mask].mean()), 2),
                    "avg_hours": round(float(hours_arr[mask].mean()), 1),
                })

        return {
            "has_data": True,
            "timing_clv_correlation": round(corr, 4),
            "avg_hours_before_close": round(float(hours_arr.mean()), 1),
            "buckets": buckets,
        }
