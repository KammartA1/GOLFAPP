"""Market inefficiency component — CLV analysis per market type.

Measures where the market is structurally inefficient:
  - Outright winner markets (thin, high vig)
  - Matchup markets (head-to-head)
  - Top 5/10/20 finish markets
  - Make/miss cut markets

Key metric: Closing Line Value (CLV) by segment.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List

import numpy as np
from scipy import stats as scipy_stats

from edge_analysis.schemas import GolfBetRecord, EdgeComponent

logger = logging.getLogger(__name__)


class MarketInefficiencyAnalyzer:
    """Analyze market inefficiency across golf betting market types."""

    def analyze(self, records: List[GolfBetRecord]) -> EdgeComponent:
        """Compute market edge via CLV decomposition per market type."""
        if not records:
            return EdgeComponent(
                name="market", value=0.0, confidence=0.0,
                verdict="No data to analyze.",
            )

        # Overall CLV
        all_clv = np.array([r.clv_cents for r in records])
        overall_clv = float(np.mean(all_clv))
        beat_close_rate = float(np.mean([1.0 if r.beat_close else 0.0 for r in records]))

        # Per market type breakdown
        by_market = self._per_market_clv(records)

        # Statistical significance of overall CLV
        if len(all_clv) >= 20:
            t_stat, p_val = scipy_stats.ttest_1samp(all_clv, 0)
            p_value = p_val / 2 if t_stat > 0 else 1.0
        else:
            p_value = 1.0

        # Identify most inefficient market
        best_market = "none"
        best_clv = 0.0
        for mtype, data in by_market.items():
            if data["avg_clv"] > best_clv and data["n_bets"] >= 10:
                best_clv = data["avg_clv"]
                best_market = mtype

        confidence = min(len(records) / 200.0, 1.0) * (1.0 - min(p_value, 1.0))

        # Line movement analysis
        line_movement = self._line_movement_analysis(records)

        # Vig analysis per market
        vig_analysis = self._vig_analysis(records)

        if overall_clv > 3.0 and p_value < 0.05:
            verdict = (f"Strong market edge ({overall_clv:.1f}c CLV, p={p_value:.4f}). "
                       f"Best market: {best_market} ({best_clv:.1f}c)")
        elif overall_clv > 1.0 and p_value < 0.10:
            verdict = f"Moderate market edge ({overall_clv:.1f}c CLV). Best: {best_market}"
        elif overall_clv > 0:
            verdict = f"Weak market edge ({overall_clv:.1f}c CLV, not significant)"
        else:
            verdict = f"No market edge detected ({overall_clv:.1f}c CLV)"

        return EdgeComponent(
            name="market",
            value=round(overall_clv, 2),
            confidence=round(confidence, 3),
            details={
                "overall_clv_cents": round(overall_clv, 2),
                "beat_close_rate": round(beat_close_rate, 4),
                "p_value": round(p_value, 4),
                "n_bets": len(records),
                "by_market_type": by_market,
                "best_market": best_market,
                "line_movement": line_movement,
                "vig_analysis": vig_analysis,
            },
            verdict=verdict,
        )

    def _per_market_clv(self, records: List[GolfBetRecord]) -> dict:
        """CLV breakdown per market type."""
        grouped = defaultdict(list)
        for r in records:
            grouped[r.market_type].append(r)

        result = {}
        for mtype, recs in grouped.items():
            clvs = np.array([r.clv_cents for r in recs])
            pnls = np.array([r.pnl for r in recs])
            edges = np.array([r.edge for r in recs])
            outcomes = np.array([r.actual_outcome for r in recs])

            # T-test on CLV
            if len(clvs) >= 10:
                t_stat, p_val = scipy_stats.ttest_1samp(clvs, 0)
                p_value = p_val / 2 if t_stat > 0 else 1.0
            else:
                p_value = 1.0

            result[mtype] = {
                "n_bets": len(recs),
                "avg_clv": round(float(np.mean(clvs)), 2),
                "median_clv": round(float(np.median(clvs)), 2),
                "std_clv": round(float(np.std(clvs)), 2),
                "beat_close_rate": round(float(np.mean([1.0 if r.beat_close else 0.0 for r in recs])), 4),
                "avg_edge": round(float(np.mean(edges)), 4),
                "win_rate": round(float(np.mean(outcomes)), 4),
                "total_pnl": round(float(np.sum(pnls)), 2),
                "roi": round(float(np.sum(pnls) / max(sum(r.stake for r in recs), 1)), 4),
                "p_value": round(p_value, 4),
                "is_significant": p_value < 0.05,
            }

        return result

    def _line_movement_analysis(self, records: List[GolfBetRecord]) -> dict:
        """Analyze line movement patterns and their predictive value."""
        movements = []
        for r in records:
            move = r.closing_line - r.bet_line
            if move != 0:
                movements.append({
                    "movement": move,
                    "clv_cents": r.clv_cents,
                    "outcome": r.actual_outcome,
                    "market_type": r.market_type,
                })

        if len(movements) < 10:
            return {"has_data": False}

        move_arr = np.array([m["movement"] for m in movements])
        clv_arr = np.array([m["clv_cents"] for m in movements])

        # Bets where line moved in our favor vs against
        moved_favor = [m for m in movements if m["clv_cents"] > 0]
        moved_against = [m for m in movements if m["clv_cents"] <= 0]

        return {
            "has_data": True,
            "n_with_movement": len(movements),
            "avg_movement": round(float(np.mean(move_arr)), 3),
            "pct_moved_in_favor": round(len(moved_favor) / len(movements), 4),
            "avg_clv_when_favor": round(float(np.mean([m["clv_cents"] for m in moved_favor])), 2) if moved_favor else 0.0,
            "avg_clv_when_against": round(float(np.mean([m["clv_cents"] for m in moved_against])), 2) if moved_against else 0.0,
            "movement_clv_corr": round(float(np.corrcoef(move_arr, clv_arr)[0, 1]), 4) if len(move_arr) > 2 else 0.0,
        }

    def _vig_analysis(self, records: List[GolfBetRecord]) -> dict:
        """Analyze market vig (overround) per market type."""
        grouped = defaultdict(list)
        for r in records:
            grouped[r.market_type].append(r)

        result = {}
        for mtype, recs in grouped.items():
            # Estimate vig from odds — typical golf market overrounds
            implied_probs = [1.0 / r.odds_decimal for r in recs if r.odds_decimal > 1.0]
            if not implied_probs:
                continue

            avg_implied = float(np.mean(implied_probs))

            # Estimated overround by market type
            typical_vigs = {
                "outright": 0.20,   # 20% overround typical for outrights
                "matchup": 0.05,    # 5% for matchups (most efficient)
                "top5": 0.10,
                "top10": 0.08,
                "top20": 0.08,
                "make_cut": 0.06,
            }

            est_vig = typical_vigs.get(mtype, 0.10)
            true_prob_est = avg_implied / (1 + est_vig)

            result[mtype] = {
                "avg_implied_prob": round(avg_implied, 4),
                "estimated_vig": est_vig,
                "estimated_true_prob": round(true_prob_est, 4),
                "n_bets": len(recs),
            }

        return result
