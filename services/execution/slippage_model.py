"""Slippage model — Price deterioration between signal and execution."""

from __future__ import annotations

import numpy as np


class SlippageModel:
    """Model price slippage between signal generation and bet placement."""

    def __init__(
        self,
        avg_slippage_cents: float = 0.5,
        slippage_std: float = 0.3,
        market_type_adjustments: dict | None = None,
    ):
        self.avg_slippage = avg_slippage_cents
        self.slippage_std = slippage_std
        self.market_adjustments = market_type_adjustments or {
            "outright": 1.5,   # Thin markets, more slippage
            "matchup": 0.8,    # Tighter spreads
            "top5": 1.2,
            "top10": 1.0,
            "top20": 0.9,
            "make_cut": 0.7,   # Most liquid
        }

    def estimate_slippage(
        self,
        market_type: str = "outright",
        stake_dollars: float = 50.0,
        time_delay_minutes: float = 5.0,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """Estimate slippage for a bet.

        Returns:
            {
                'slippage_cents': float (positive = cost),
                'slippage_prob': float,
                'effective_odds_adjustment': float,
            }
        """
        if rng is None:
            rng = np.random.default_rng()

        # Base slippage
        market_mult = self.market_adjustments.get(market_type, 1.0)
        base = self.avg_slippage * market_mult

        # Stake impact: larger bets may move the line
        stake_factor = 1.0
        if stake_dollars > 500:
            stake_factor = 1.0 + (stake_dollars - 500) / 5000.0
        elif stake_dollars > 200:
            stake_factor = 1.0 + (stake_dollars - 200) / 3000.0

        # Time decay: longer delay = more slippage
        time_factor = 1.0 + np.log1p(time_delay_minutes) * 0.1

        expected_slippage = base * stake_factor * time_factor
        actual_slippage = max(0, rng.normal(expected_slippage, self.slippage_std))

        return {
            "slippage_cents": round(float(actual_slippage), 2),
            "slippage_prob": round(float(actual_slippage / 100.0), 4),
            "effective_odds_adjustment": round(float(actual_slippage / 100.0), 4),
            "market_type_factor": market_mult,
            "stake_factor": round(stake_factor, 3),
            "time_factor": round(time_factor, 3),
        }
