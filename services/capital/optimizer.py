"""Capital optimizer — Integrate Kelly, risk, and portfolio for optimal sizing."""

from __future__ import annotations

import logging
from typing import Dict, List

from services.capital.kelly import KellyCriterion
from services.capital.risk_adjusted import RiskMetrics
from services.capital.portfolio import PortfolioManager

logger = logging.getLogger(__name__)


class CapitalOptimizer:
    """Optimize capital allocation across the betting portfolio."""

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_bet_pct: float = 0.05,
        max_daily_loss_pct: float = 0.10,
    ):
        self.kelly = KellyCriterion(kelly_fraction, max_bet_pct)
        self.risk_metrics = RiskMetrics()
        self.portfolio = PortfolioManager()
        self.max_daily_loss_pct = max_daily_loss_pct

    def optimize_bet(
        self,
        win_prob: float,
        odds_decimal: float,
        bankroll: float,
        pending_bets: List[dict],
        player: str,
        tournament: str = "unknown",
        market_type: str = "outright",
        prob_uncertainty: float = 0.05,
        clv_avg_cents: float = 0.0,
        calibration_mae: float = 0.0,
        daily_pnl: float = 0.0,
    ) -> dict:
        """Compute optimal bet size considering all constraints.

        Returns fully optimized sizing decision.
        """
        # 1. Kelly sizing
        kelly_result = self.kelly.optimal_stake(
            win_prob, odds_decimal, bankroll,
            prob_uncertainty, clv_avg_cents, calibration_mae,
        )

        if kelly_result["blocked"]:
            return {
                "approved": False,
                "reason": kelly_result["block_reason"],
                "stake": 0.0,
                "kelly": kelly_result,
            }

        proposed_stake = kelly_result["stake_dollars"]

        # 2. Daily loss limit check
        daily_loss_limit = bankroll * self.max_daily_loss_pct
        if daily_pnl < 0 and abs(daily_pnl) + proposed_stake > daily_loss_limit:
            remaining = max(0, daily_loss_limit - abs(daily_pnl))
            if remaining < 5:
                return {
                    "approved": False,
                    "reason": f"Daily loss limit reached (${abs(daily_pnl):.0f} lost, limit=${daily_loss_limit:.0f})",
                    "stake": 0.0,
                    "kelly": kelly_result,
                }
            proposed_stake = min(proposed_stake, remaining)

        # 3. Portfolio concentration check
        new_bet = {
            "player": player,
            "tournament": tournament,
            "market_type": market_type,
            "stake": proposed_stake,
        }
        concentration = self.portfolio.check_concentration(pending_bets, new_bet, bankroll)

        if not concentration["allowed"]:
            if concentration["max_allowed_stake"] > 5:
                proposed_stake = concentration["max_allowed_stake"]
            else:
                return {
                    "approved": False,
                    "reason": f"Concentration limit: {'; '.join(concentration['violations'])}",
                    "stake": 0.0,
                    "kelly": kelly_result,
                    "concentration": concentration,
                }

        return {
            "approved": True,
            "reason": "",
            "stake": round(proposed_stake, 2),
            "stake_pct": round(proposed_stake / max(bankroll, 1) * 100, 3),
            "kelly": kelly_result,
            "concentration": concentration,
            "edge": kelly_result["edge"],
            "expected_value": round(proposed_stake * kelly_result["edge"], 2),
        }
