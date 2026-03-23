"""Best bet removal test — Is profitability concentrated in a few lucky bets?

A robust system should remain profitable even after removing the best bets.
If removing the top 5% of bets makes the system unprofitable, the edge may
be illusory — driven by luck rather than skill.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class BestBetRemovalTest:
    """Test whether profitability survives removal of best outcomes.

    Key Question: If we remove the top N% of bets by PnL, is the
    remaining portfolio still profitable? If not, our "edge" might
    just be a few lucky longshots.

    This is especially relevant for golf betting where outright winners
    at 20:1+ can mask hundreds of losing bets.
    """

    REMOVAL_PERCENTAGES = [1, 2, 5, 10, 15, 20]

    def run(self, bet_records: List[dict]) -> dict:
        """Run the best-bet removal analysis.

        Args:
            bet_records: List of dicts with keys:
                pnl, stake, status, bet_type, odds_decimal, model_prob

        Returns:
            {
                'n_bets': int,
                'original_roi': float,
                'original_pnl': float,
                'removal_results': list of dicts,
                'concentration_index': float,
                'verdict': str,
                'by_market_type': dict,
            }
        """
        if len(bet_records) < 20:
            return {"error": "Need at least 20 bet records"}

        pnl = np.array([r["pnl"] for r in bet_records])
        stakes = np.array([r["stake"] for r in bet_records])
        total_pnl = float(np.sum(pnl))
        total_staked = float(np.sum(stakes))
        original_roi = total_pnl / max(total_staked, 1)

        # Sort by PnL descending
        sorted_indices = np.argsort(pnl)[::-1]

        removal_results = []
        for pct in self.REMOVAL_PERCENTAGES:
            n_remove = max(1, int(len(bet_records) * pct / 100))
            keep_indices = sorted_indices[n_remove:]

            remaining_pnl = float(np.sum(pnl[keep_indices]))
            remaining_staked = float(np.sum(stakes[keep_indices]))
            remaining_roi = remaining_pnl / max(remaining_staked, 1)

            removed_pnl = float(np.sum(pnl[sorted_indices[:n_remove]]))

            removal_results.append({
                "pct_removed": pct,
                "n_removed": n_remove,
                "n_remaining": len(keep_indices),
                "removed_pnl": round(removed_pnl, 2),
                "remaining_pnl": round(remaining_pnl, 2),
                "remaining_roi": round(remaining_roi, 4),
                "roi_delta": round(remaining_roi - original_roi, 4),
                "still_profitable": remaining_pnl > 0,
                "pnl_concentration": round(removed_pnl / max(abs(total_pnl), 1), 4),
            })

        # Concentration index: what % of total PnL comes from top 5% of bets?
        n_top5pct = max(1, int(len(bet_records) * 0.05))
        top5_pnl = float(np.sum(pnl[sorted_indices[:n_top5pct]]))
        concentration_index = top5_pnl / max(abs(total_pnl), 1)

        # Gini coefficient of PnL distribution
        gini = self._gini_coefficient(pnl)

        # By market type analysis
        by_market = self._analyze_by_market_type(bet_records)

        # Verdict
        # Find at what removal % we become unprofitable
        first_unprofitable = None
        for r in removal_results:
            if not r["still_profitable"]:
                first_unprofitable = r["pct_removed"]
                break

        if first_unprofitable is None:
            verdict = "robust"
            verdict_detail = "Profitable even after removing top 20% of bets"
        elif first_unprofitable >= 10:
            verdict = "acceptable"
            verdict_detail = f"Unprofitable after removing top {first_unprofitable}%"
        elif first_unprofitable >= 5:
            verdict = "fragile"
            verdict_detail = f"Unprofitable after removing just top {first_unprofitable}%"
        else:
            verdict = "illusory"
            verdict_detail = (
                f"Unprofitable after removing top {first_unprofitable}% — "
                f"edge may be driven by luck"
            )

        return {
            "n_bets": len(bet_records),
            "original_pnl": round(total_pnl, 2),
            "original_roi": round(original_roi, 4),
            "removal_results": removal_results,
            "concentration_index": round(concentration_index, 4),
            "gini_coefficient": round(gini, 4),
            "first_unprofitable_removal_pct": first_unprofitable,
            "verdict": verdict,
            "verdict_detail": verdict_detail,
            "by_market_type": by_market,
        }

    def _analyze_by_market_type(self, bet_records: List[dict]) -> dict:
        """Run concentration analysis per market type.

        Golf-specific: outright bets are inherently more concentrated
        (big payoffs on rare events) vs make_cut/top20 which should be
        more consistently profitable.
        """
        from collections import defaultdict
        by_type = defaultdict(list)
        for r in bet_records:
            mtype = r.get("bet_type", r.get("market_type", "unknown"))
            by_type[mtype].append(r)

        results = {}
        for mtype, records in by_type.items():
            if len(records) < 5:
                continue

            pnl = np.array([r["pnl"] for r in records])
            stakes = np.array([r["stake"] for r in records])
            total_pnl = float(np.sum(pnl))
            total_staked = float(np.sum(stakes))

            # Top 10% removal for this market type
            n_remove = max(1, int(len(records) * 0.10))
            sorted_idx = np.argsort(pnl)[::-1]
            remaining_pnl = float(np.sum(pnl[sorted_idx[n_remove:]]))
            remaining_staked = float(np.sum(stakes[sorted_idx[n_remove:]]))

            results[mtype] = {
                "n_bets": len(records),
                "total_pnl": round(total_pnl, 2),
                "roi": round(total_pnl / max(total_staked, 1), 4),
                "after_10pct_removal_pnl": round(remaining_pnl, 2),
                "after_10pct_removal_roi": round(
                    remaining_pnl / max(remaining_staked, 1), 4
                ),
                "still_profitable": remaining_pnl > 0,
            }

        return results

    @staticmethod
    def _gini_coefficient(values: np.ndarray) -> float:
        """Compute Gini coefficient of PnL distribution.

        Gini = 0: perfectly equal (all bets have same PnL)
        Gini = 1: perfectly concentrated (one bet has all PnL)
        """
        if len(values) < 2:
            return 0.0

        # Shift to non-negative for Gini calculation
        shifted = values - values.min() + 1e-10
        sorted_vals = np.sort(shifted)
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_vals))) / (n * cumsum[-1]) - (n + 1) / n
        return float(np.clip(gini, 0, 1))
