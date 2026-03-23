"""
edge_analysis/execution.py
============================
Component 4: EXECUTION EDGE — Golf line shopping effectiveness.

Golf-specific execution analysis:
  - Line shopping across multiple books (outright odds vary dramatically)
  - Timing of bet placement relative to market moves
  - Slippage in thin golf markets
  - Book-specific execution quality
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy import stats as sp_stats

from edge_analysis.schemas import GolfBetRecord, EdgeComponentResult

log = logging.getLogger(__name__)


def _compute_execution_cost(bet: GolfBetRecord) -> float:
    """Execution cost (slippage) for a golf bet.

    For outright/futures: compare signal odds to bet odds in probability space.
    For props: compare signal line to bet line.
    """
    if bet.market_type in ("outright", "top5", "top10", "top20", "make_cut", "matchup"):
        # Odds-based: signal implied prob vs bet implied prob
        # Worse price = higher implied prob to win (paying more vig)
        signal_prob = bet.predicted_prob  # Signal identified this probability
        if bet.market_prob_at_bet > 0 and bet.predicted_prob > 0:
            # Cost = market prob at bet - signal implied price
            # Positive = we paid more than signal indicated
            return bet.market_prob_at_bet - bet.predicted_prob
        # Line-based fallback
        return bet.bet_line - bet.signal_line
    else:
        if bet.direction.upper() == "OVER":
            return bet.bet_line - bet.signal_line
        else:
            return bet.signal_line - bet.bet_line


def _compute_execution_vs_close(bet: GolfBetRecord) -> float:
    """Fraction of signal-to-close move captured. 1.0 = captured all value."""
    if bet.direction.upper() in ("OVER", "WIN", "PLACE"):
        total_move = bet.closing_line - bet.signal_line
        captured = bet.closing_line - bet.bet_line
    else:
        total_move = bet.signal_line - bet.closing_line
        captured = bet.bet_line - bet.closing_line

    if abs(total_move) < 0.001:
        return 1.0
    return captured / total_move if total_move != 0 else 0.0


def compute_execution_edge(
    bets: List[GolfBetRecord],
    total_roi: float,
) -> EdgeComponentResult:
    """Analyze golf execution quality.

    Golf execution is critical because:
    1. Outright markets have wide spreads (5-10% vig on some books)
    2. Line shopping can save 2-5% on outright futures
    3. Early-week prices are often better than tournament-week
    4. Withdrawal announcements cause rapid line movement
    """
    valid = [b for b in bets if b.signal_line is not None and b.signal_line != 0]
    if len(valid) < 15:
        return EdgeComponentResult(
            name="execution",
            edge_pct_of_roi=0.0, absolute_value=0.0, p_value=1.0,
            is_significant=False, is_positive=False,
            sample_size=len(valid),
            verdict="Insufficient data for execution edge analysis (need 15+ bets)",
        )

    costs = np.array([_compute_execution_cost(b) for b in valid])
    captures = np.array([_compute_execution_vs_close(b) for b in valid])

    avg_cost = float(np.mean(costs))
    median_cost = float(np.median(costs))
    pct_improved = float(np.mean(costs < 0))
    avg_capture = float(np.mean(captures))

    t_stat, p_two = sp_stats.ttest_1samp(costs, 0.0)
    p_value = float(p_two)
    is_positive = avg_cost < 0
    is_significant = p_value < 0.05

    # Per market type
    market_groups: Dict[str, List[float]] = defaultdict(list)
    for b in valid:
        market_groups[b.market_type].append(_compute_execution_cost(b))

    cost_by_market = {}
    for mtype, vals in market_groups.items():
        arr = np.array(vals)
        cost_by_market[mtype] = {
            "avg_slippage": round(float(np.mean(arr)), 4),
            "pct_improved": round(float(np.mean(arr < 0)), 4),
            "n_bets": len(vals),
        }

    total_drag = float(np.sum(costs))

    # Attribution
    if is_positive and is_significant:
        exec_pct = min(15.0, pct_improved * 20.0)
    elif is_positive:
        exec_pct = min(8.0, pct_improved * 10.0)
    elif is_significant:
        exec_pct = max(-20.0, -abs(avg_cost) * 10.0)
    else:
        exec_pct = max(-10.0, -abs(avg_cost) * 5.0)

    # Verdict
    verdict_parts = []
    if is_positive and is_significant:
        verdict_parts.append(
            f"POSITIVE execution edge. Line shopping effective — avg improvement: "
            f"{abs(avg_cost):.4f}. {pct_improved:.0%} executed at better-than-signal price."
        )
    elif is_positive:
        verdict_parts.append(
            f"Slight execution advantage ({abs(avg_cost):.4f} avg improvement) "
            f"but NOT significant (p={p_value:.4f})."
        )
    elif is_significant:
        verdict_parts.append(
            f"EXECUTION DRAG detected. Avg slippage: {avg_cost:+.4f}. "
            f"Only {pct_improved:.0%} of bets improved. Golf markets are thin — "
            f"consider earlier betting and multi-book line shopping."
        )
    else:
        verdict_parts.append(
            f"Neutral execution. Avg slippage: {avg_cost:+.4f} (p={p_value:.4f}). "
            f"Capture rate: {avg_capture:.0%}."
        )

    # Market-specific execution notes
    if "outright" in cost_by_market:
        out = cost_by_market["outright"]
        verdict_parts.append(
            f"Outright execution: avg slippage {out['avg_slippage']:+.4f} "
            f"({out['pct_improved']:.0%} improved, n={out['n_bets']})."
        )

    return EdgeComponentResult(
        name="execution",
        edge_pct_of_roi=round(exec_pct, 2),
        absolute_value=round(avg_cost, 4),
        p_value=round(p_value, 4),
        is_significant=is_significant,
        is_positive=is_positive,
        sample_size=len(valid),
        details={
            "avg_slippage": round(avg_cost, 4),
            "median_slippage": round(median_cost, 4),
            "pct_price_improved": round(pct_improved, 4),
            "avg_capture_rate": round(avg_capture, 4),
            "total_drag": round(total_drag, 4),
            "cost_by_market": cost_by_market,
        },
        verdict=" ".join(verdict_parts),
    )
