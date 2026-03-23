"""
edge_analysis/market_inefficiency.py
======================================
Component 3: MARKET INEFFICIENCY CAPTURE — Golf CLV analysis.

Golf-specific: outrights are notoriously inefficient markets with thin
liquidity and wide spreads. Quantifies CLV per market type to identify
where the real pricing errors live.

Uses the CLV system built in Section 5.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

from edge_analysis.schemas import GolfBetRecord, EdgeComponentResult

log = logging.getLogger(__name__)


def _compute_clv_for_bet(bet: GolfBetRecord) -> float:
    """Compute CLV in cents/points for a single golf bet.

    For outrights/futures: CLV is in odds space (probability difference).
    For props (over/under): CLV is in line space (closing - bet line).
    """
    if bet.market_type in ("outright", "top5", "top10", "top20", "make_cut"):
        # For position markets, CLV is probability-based
        # Lower closing odds = we got value (probability increased)
        if bet.market_prob_at_close is not None and bet.market_prob_at_bet > 0:
            return bet.market_prob_at_close - bet.market_prob_at_bet
        # Fallback: use line movement
        return bet.closing_line - bet.bet_line
    elif bet.market_type == "matchup":
        # For matchups, CLV is probability-based
        if bet.market_prob_at_close is not None:
            return bet.market_prob_at_close - bet.market_prob_at_bet
        return bet.closing_line - bet.bet_line
    else:
        # Props: direction-based line CLV
        if bet.direction.upper() == "OVER":
            return bet.closing_line - bet.bet_line
        else:
            return bet.bet_line - bet.closing_line


def _american_to_implied(odds: int) -> float:
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def compute_market_inefficiency_edge(
    bets: List[GolfBetRecord],
    total_roi: float,
) -> EdgeComponentResult:
    """Analyze golf market inefficiency via CLV.

    Golf-specific insights:
    - Outrights: notoriously inefficient, thin liquidity, wide spreads
    - Matchups: more liquid but still exploitable with course-fit models
    - Top-N markets: moderate liquidity, often mispriced early in week
    - Make cut: relatively efficient, high volume

    Returns EdgeComponentResult with CLV analysis per market type.
    """
    valid = [b for b in bets if b.closing_line is not None and b.closing_line != 0]
    if len(valid) < 15:
        return EdgeComponentResult(
            name="market_inefficiency",
            edge_pct_of_roi=0.0, absolute_value=0.0, p_value=1.0,
            is_significant=False, is_positive=False,
            sample_size=len(valid),
            verdict="Insufficient CLV data for market inefficiency analysis (need 15+ bets)",
        )

    # Compute CLV for every bet
    clv_values = np.array([_compute_clv_for_bet(b) for b in valid])
    avg_clv = float(np.mean(clv_values))
    median_clv = float(np.median(clv_values))
    std_clv = float(np.std(clv_values))
    beat_close_rate = float(np.mean(clv_values > 0))

    # Significance
    t_stat, p_two = sp_stats.ttest_1samp(clv_values, 0.0)
    p_value = float(p_two / 2 if t_stat > 0 else 1.0 - p_two / 2)
    is_positive = avg_clv > 0
    is_significant = p_value < 0.05

    # Per market type breakdown
    market_groups: Dict[str, List[float]] = defaultdict(list)
    for b in valid:
        market_groups[b.market_type].append(_compute_clv_for_bet(b))

    clv_by_market = {}
    for mtype, vals in market_groups.items():
        arr = np.array(vals)
        t, p = (0.0, 1.0)
        if len(arr) >= 5 and np.std(arr) > 0:
            t, p = sp_stats.ttest_1samp(arr, 0.0)
        clv_by_market[mtype] = {
            "avg_clv": round(float(np.mean(arr)), 4),
            "median_clv": round(float(np.median(arr)), 4),
            "beat_close_pct": round(float(np.mean(arr > 0)), 4),
            "n_bets": len(vals),
            "is_positive": float(np.mean(arr)) > 0,
            "p_value": round(float(p / 2 if t > 0 else 1.0 - p / 2), 4),
            "is_significant": float(p / 2 if t > 0 else 1.0 - p / 2) < 0.05,
        }

    # Per-tournament CLV
    tournament_groups: Dict[str, List[float]] = defaultdict(list)
    for b in valid:
        tournament_groups[b.tournament].append(_compute_clv_for_bet(b))

    clv_by_tournament = {}
    for tourn, vals in tournament_groups.items():
        arr = np.array(vals)
        clv_by_tournament[tourn] = {
            "avg_clv": round(float(np.mean(arr)), 4),
            "n_bets": len(vals),
        }

    # Distribution percentiles
    pcts = np.percentile(clv_values, [5, 10, 25, 50, 75, 90, 95])

    # Attribution
    if is_positive and is_significant:
        market_pct = min(70.0, max(30.0, beat_close_rate * 100.0))
    elif is_positive:
        market_pct = min(25.0, beat_close_rate * 50.0)
    else:
        market_pct = 0.0

    # Verdict
    verdict_parts = []
    if is_positive and is_significant:
        verdict_parts.append(
            f"REAL market inefficiency edge. Avg CLV = {avg_clv:+.4f}, "
            f"beat-close rate = {beat_close_rate:.1%}. p={p_value:.4f}."
        )
    elif is_positive:
        verdict_parts.append(
            f"Positive CLV ({avg_clv:+.4f}) but NOT significant (p={p_value:.4f})."
        )
    else:
        verdict_parts.append(
            f"NO market inefficiency edge. Avg CLV = {avg_clv:+.4f}. "
            f"Beat-close rate {beat_close_rate:.1%}."
        )

    # Highlight outright inefficiency
    if "outright" in clv_by_market:
        outright = clv_by_market["outright"]
        if outright["is_positive"]:
            verdict_parts.append(
                f"Outrights show CLV={outright['avg_clv']:+.4f} "
                f"(beat-close {outright['beat_close_pct']:.1%}, n={outright['n_bets']}). "
                f"Confirming outright market inefficiency."
            )

    # Best/worst market
    real = {k: v for k, v in clv_by_market.items() if v["n_bets"] >= 5}
    if real:
        best = max(real.items(), key=lambda x: x[1]["avg_clv"])
        worst = min(real.items(), key=lambda x: x[1]["avg_clv"])
        verdict_parts.append(
            f"Most inefficient: {best[0]} (CLV={best[1]['avg_clv']:+.4f}). "
            f"Most efficient: {worst[0]} (CLV={worst[1]['avg_clv']:+.4f})."
        )

    return EdgeComponentResult(
        name="market_inefficiency",
        edge_pct_of_roi=round(market_pct, 2),
        absolute_value=round(avg_clv, 4),
        p_value=round(p_value, 4),
        is_significant=is_significant,
        is_positive=is_positive,
        sample_size=len(valid),
        details={
            "avg_clv": round(avg_clv, 4),
            "median_clv": round(median_clv, 4),
            "std_clv": round(std_clv, 4),
            "beat_close_rate": round(beat_close_rate, 4),
            "t_stat": round(float(t_stat), 3),
            "clv_percentiles": {
                "p5": round(float(pcts[0]), 4), "p10": round(float(pcts[1]), 4),
                "p25": round(float(pcts[2]), 4), "p50": round(float(pcts[3]), 4),
                "p75": round(float(pcts[4]), 4), "p90": round(float(pcts[5]), 4),
                "p95": round(float(pcts[6]), 4),
            },
            "clv_by_market": clv_by_market,
            "clv_by_tournament": clv_by_tournament,
            "clv_distribution": clv_values.tolist(),
        },
        verdict=" ".join(verdict_parts),
    )
