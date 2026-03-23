"""ExecutionSimulator — Chain all execution models to simulate real-world friction."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from services.execution.slippage_model import SlippageModel
from services.execution.limit_model import LimitModel
from services.execution.latency_model import LatencyModel
from services.execution.rejection_model import RejectionModel

logger = logging.getLogger(__name__)


class ExecutionSimulator:
    """Chain all execution models to simulate real-world betting friction.

    Usage:
        sim = ExecutionSimulator()
        result = sim.simulate_execution(
            edge=0.05, stake=100, book="draftkings",
            market_type="outright", latency_seconds=30
        )
    """

    def __init__(self):
        self.slippage = SlippageModel()
        self.limits = LimitModel()
        self.latency = LatencyModel()
        self.rejection = RejectionModel()

    def simulate_execution(
        self,
        edge: float,
        stake: float,
        book: str = "draftkings",
        market_type: str = "outright",
        latency_seconds: float = 30.0,
        account_profit_pct: float = 0.0,
        account_age_months: int = 6,
        is_line_moving: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Simulate full execution chain and return reality-adjusted edge.

        Returns:
            {
                'original_edge': float,
                'effective_edge': float,
                'edge_retained_pct': float,
                'original_stake': float,
                'effective_stake': float,
                'rejection_prob': float,
                'expected_value': float,
                'should_bet': bool,
                'details': {...},
            }
        """
        if rng is None:
            rng = np.random.default_rng()

        # 1. Latency cost
        latency_result = self.latency.estimate_latency_cost(
            edge, latency_seconds, market_type, is_line_moving
        )
        edge_after_latency = latency_result["edge_after_latency"]

        # 2. Slippage cost
        slippage_result = self.slippage.estimate_slippage(
            market_type, stake, latency_seconds / 60.0, rng
        )
        edge_after_slippage = max(0, edge_after_latency - slippage_result["slippage_prob"])

        # 3. Limit check
        limit_result = self.limits.get_max_stake(
            book, market_type, account_profit_pct
        )
        effective_stake = min(stake, limit_result["effective_limit"])

        # 4. Rejection probability
        rejection_result = self.rejection.rejection_probability(
            book, effective_stake, edge, account_age_months, account_profit_pct
        )

        # Effective edge after all friction
        effective_edge = edge_after_slippage * (1 - rejection_result["rejection_probability"])

        # Expected value
        expected_value = effective_edge * effective_stake

        # Should we still bet?
        min_viable_edge = 0.015  # Minimum edge after friction
        should_bet = effective_edge > min_viable_edge and effective_stake > 5.0

        edge_retained = effective_edge / max(edge, 0.001)

        return {
            "original_edge": round(edge, 4),
            "effective_edge": round(effective_edge, 4),
            "edge_retained_pct": round(edge_retained, 4),
            "original_stake": round(stake, 2),
            "effective_stake": round(effective_stake, 2),
            "rejection_prob": round(rejection_result["rejection_probability"], 4),
            "expected_value": round(expected_value, 2),
            "should_bet": should_bet,
            "details": {
                "latency": latency_result,
                "slippage": slippage_result,
                "limits": limit_result,
                "rejection": rejection_result,
            },
        }

    def simulate_portfolio_execution(
        self,
        bets: list[dict],
        book: str = "draftkings",
        account_profit_pct: float = 0.0,
    ) -> dict:
        """Simulate execution for a portfolio of bets.

        Each bet dict: {edge, stake, market_type, latency_seconds}
        """
        results = []
        total_original_ev = 0.0
        total_effective_ev = 0.0
        total_rejected = 0

        for bet in bets:
            result = self.simulate_execution(
                edge=bet["edge"],
                stake=bet["stake"],
                book=book,
                market_type=bet.get("market_type", "outright"),
                latency_seconds=bet.get("latency_seconds", 30),
                account_profit_pct=account_profit_pct,
            )
            results.append(result)
            total_original_ev += bet["edge"] * bet["stake"]
            total_effective_ev += result["expected_value"]
            if not result["should_bet"]:
                total_rejected += 1

        friction_cost = total_original_ev - total_effective_ev

        return {
            "n_bets": len(bets),
            "n_viable": len(bets) - total_rejected,
            "n_rejected": total_rejected,
            "total_original_ev": round(total_original_ev, 2),
            "total_effective_ev": round(total_effective_ev, 2),
            "total_friction_cost": round(friction_cost, 2),
            "friction_pct": round(friction_cost / max(total_original_ev, 0.01), 4),
            "individual_results": results,
        }
