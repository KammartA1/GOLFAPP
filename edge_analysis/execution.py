"""Execution edge component — Price quality and line shopping effectiveness.

Measures how well we execute bets:
  - Price quality: did we get good odds relative to true probability?
  - Line shopping: did we find the best available price?
  - Speed: did execution delay cost us edge?
  - Slippage: difference between expected and actual fill price
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List

import numpy as np

from edge_analysis.schemas import GolfBetRecord, EdgeComponent

logger = logging.getLogger(__name__)


class ExecutionAnalyzer:
    """Analyze execution quality of placed bets."""

    def analyze(self, records: List[GolfBetRecord]) -> EdgeComponent:
        """Compute execution edge — how well did we capture available edge?"""
        if not records:
            return EdgeComponent(
                name="execution", value=0.0, confidence=0.0,
                verdict="No data to analyze.",
            )

        # Price quality: compare bet price to signal price
        price_quality = self._price_quality_analysis(records)

        # Line shopping: compare our price to what was available
        shop_analysis = self._line_shopping_analysis(records)

        # Execution speed
        speed_analysis = self._speed_analysis(records)

        # Per-book analysis
        book_analysis = self._per_book_analysis(records)

        # Execution edge in cents: difference between signal edge and captured edge
        signal_edges = np.array([r.predicted_prob - (1.0 / r.odds_decimal) for r in records
                                 if r.odds_decimal > 1.0])
        captured_edges = np.array([r.clv_cents for r in records])

        if len(signal_edges) > 0 and len(captured_edges) > 0:
            # Execution cost = how much edge we lose in execution
            avg_signal_edge = float(np.mean(signal_edges)) * 100  # to cents
            avg_captured = float(np.mean(captured_edges))
            execution_cost = avg_signal_edge - avg_captured
            execution_edge = -execution_cost  # Negative cost = positive execution
        else:
            execution_cost = 0.0
            execution_edge = 0.0

        confidence = min(len(records) / 100.0, 1.0)

        if execution_edge > 0:
            verdict = f"Good execution (+{execution_edge:.1f}c above expected)"
        elif execution_edge > -1.0:
            verdict = f"Acceptable execution ({execution_edge:.1f}c)"
        else:
            verdict = f"Poor execution ({execution_edge:.1f}c leakage)"

        return EdgeComponent(
            name="execution",
            value=round(execution_edge, 2),
            confidence=round(confidence, 3),
            details={
                "price_quality": price_quality,
                "line_shopping": shop_analysis,
                "speed": speed_analysis,
                "by_book": book_analysis,
                "execution_cost_cents": round(execution_cost, 2),
            },
            verdict=verdict,
        )

    def _price_quality_analysis(self, records: List[GolfBetRecord]) -> dict:
        """Analyze how good the prices were that we got."""
        slippages = []
        for r in records:
            # Slippage = difference between signal line and bet line
            slip = r.bet_line - r.signal_line
            slippages.append(slip)

        slippages_arr = np.array(slippages)

        # Price quality score: percentage of bets where we got signal price or better
        got_signal_or_better = float(np.mean(slippages_arr <= 0))

        return {
            "avg_slippage": round(float(np.mean(slippages_arr)), 3),
            "median_slippage": round(float(np.median(slippages_arr)), 3),
            "max_slippage": round(float(np.max(np.abs(slippages_arr))), 3) if len(slippages_arr) > 0 else 0.0,
            "pct_at_signal_or_better": round(got_signal_or_better, 4),
            "n_bets": len(records),
        }

    def _line_shopping_analysis(self, records: List[GolfBetRecord]) -> dict:
        """Analyze line shopping effectiveness across books."""
        # Group by tournament + player to see if we found the best line
        by_opportunity = defaultdict(list)
        for r in records:
            key = f"{r.tournament}|{r.player}|{r.market_type}"
            by_opportunity[key].append(r)

        n_shopped = 0
        savings_cents = []

        for key, recs in by_opportunity.items():
            if len(recs) > 1:
                # Multiple bets on same opportunity — compare prices
                best_odds = max(r.odds_decimal for r in recs)
                for r in recs:
                    saving = (1.0 / r.odds_decimal - 1.0 / best_odds) * 100
                    savings_cents.append(saving)
                n_shopped += 1

        if not savings_cents:
            return {
                "has_data": False,
                "n_unique_opportunities": len(by_opportunity),
            }

        return {
            "has_data": True,
            "n_unique_opportunities": len(by_opportunity),
            "n_shopped_across_books": n_shopped,
            "avg_shopping_savings_cents": round(float(np.mean(savings_cents)), 2),
            "total_shopping_savings_cents": round(float(np.sum(savings_cents)), 2),
        }

    def _speed_analysis(self, records: List[GolfBetRecord]) -> dict:
        """Analyze execution speed — does delay cost edge?"""
        timed = [r for r in records if r.signal_timestamp and r.bet_timestamp]
        if len(timed) < 10:
            return {"has_data": False}

        delays = []
        for r in timed:
            delay_minutes = (r.bet_timestamp - r.signal_timestamp).total_seconds() / 60.0
            if delay_minutes >= 0:
                delays.append({
                    "delay_minutes": delay_minutes,
                    "clv_cents": r.clv_cents,
                    "edge": r.edge,
                })

        if len(delays) < 10:
            return {"has_data": False}

        delay_arr = np.array([d["delay_minutes"] for d in delays])
        clv_arr = np.array([d["clv_cents"] for d in delays])

        # Split into fast vs slow execution
        median_delay = float(np.median(delay_arr))
        fast = [d for d in delays if d["delay_minutes"] <= median_delay]
        slow = [d for d in delays if d["delay_minutes"] > median_delay]

        clv_fast = float(np.mean([d["clv_cents"] for d in fast])) if fast else 0.0
        clv_slow = float(np.mean([d["clv_cents"] for d in slow])) if slow else 0.0

        # Correlation between delay and CLV
        corr = float(np.corrcoef(delay_arr, clv_arr)[0, 1]) if len(delay_arr) > 2 else 0.0

        return {
            "has_data": True,
            "avg_delay_minutes": round(float(np.mean(delay_arr)), 1),
            "median_delay_minutes": round(median_delay, 1),
            "clv_fast_execution": round(clv_fast, 2),
            "clv_slow_execution": round(clv_slow, 2),
            "speed_premium_cents": round(clv_fast - clv_slow, 2),
            "delay_clv_correlation": round(corr, 4),
        }

    def _per_book_analysis(self, records: List[GolfBetRecord]) -> dict:
        """Analyze execution quality per sportsbook."""
        grouped = defaultdict(list)
        for r in records:
            grouped[r.book].append(r)

        result = {}
        for book, recs in grouped.items():
            clvs = [r.clv_cents for r in recs]
            edges = [r.edge for r in recs]
            slippages = [r.bet_line - r.signal_line for r in recs]

            result[book] = {
                "n_bets": len(recs),
                "avg_clv": round(float(np.mean(clvs)), 2),
                "avg_edge": round(float(np.mean(edges)), 4),
                "avg_slippage": round(float(np.mean(slippages)), 3),
                "beat_close_rate": round(float(np.mean([1.0 if r.beat_close else 0.0 for r in recs])), 4),
            }

        return result
