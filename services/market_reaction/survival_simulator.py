"""12-month MarketSurvivalSimulator — Project viability of the operation."""

from __future__ import annotations

import logging

import numpy as np

from services.market_reaction.book_behavior import BookBehaviorModel
from services.market_reaction.limit_progression import LimitProgressionTracker
from services.market_reaction.edge_decay import EdgeDecayAnalyzer

logger = logging.getLogger(__name__)


class MarketSurvivalSimulator:
    """Simulate 12-month survival of the betting operation.

    Models the interaction of:
      - Edge decay over time
      - Limit progression (accounts getting restricted)
      - Total capacity decline
      - Profitability trajectory
    """

    def __init__(self, n_sims: int = 1000):
        self.n_sims = n_sims
        self.book_model = BookBehaviorModel()
        self.limit_tracker = LimitProgressionTracker()
        self.decay_analyzer = EdgeDecayAnalyzer()

    def simulate_12_month_survival(
        self,
        current_monthly_clv_cents: float = 3.0,
        current_monthly_volume: float = 5000.0,
        monthly_profit_pct: float = 0.08,
        months_active: int = 6,
        monthly_clv_history: list | None = None,
        bankroll: float = 5000.0,
    ) -> dict:
        """Simulate 12 months of operation.

        Returns monthly projections of edge, volume, and profitability.
        """
        rng = np.random.default_rng(42)

        # Edge decay parameters
        if monthly_clv_history and len(monthly_clv_history) >= 3:
            decay = self.decay_analyzer.analyze_decay(monthly_clv_history, [0.0] * len(monthly_clv_history))
            monthly_decay_rate = -decay.get("linear_slope", 0.0)
        else:
            monthly_decay_rate = 0.15  # Default: 0.15 cents CLV decay per month

        # Simulate N paths
        monthly_results = {m: {"edge": [], "volume": [], "profit": [], "bankroll": []}
                          for m in range(1, 13)}

        for sim in range(self.n_sims):
            edge = current_monthly_clv_cents
            volume = current_monthly_volume
            br = bankroll

            for month in range(1, 13):
                # Edge decay with noise
                decay = monthly_decay_rate + rng.normal(0, 0.1)
                edge = max(0, edge - decay)

                # Limit progression reduces volume
                future_month = months_active + month
                capacity = self.limit_tracker.project_total_capacity(
                    monthly_profit_pct, future_month
                )
                capacity_factor = min(1.0, capacity["total_current_capacity"] / max(current_monthly_volume / 4, 1))
                volume = current_monthly_volume * capacity_factor

                # Add volume noise
                volume *= (1 + rng.normal(0, 0.15))
                volume = max(0, volume)

                # Monthly profit
                profit = volume * (edge / 100.0) + rng.normal(0, volume * 0.03)
                br += profit

                monthly_results[month]["edge"].append(edge)
                monthly_results[month]["volume"].append(volume)
                monthly_results[month]["profit"].append(profit)
                monthly_results[month]["bankroll"].append(br)

        # Aggregate results
        projections = []
        for month in range(1, 13):
            edges = np.array(monthly_results[month]["edge"])
            volumes = np.array(monthly_results[month]["volume"])
            profits = np.array(monthly_results[month]["profit"])
            bankrolls = np.array(monthly_results[month]["bankroll"])

            projections.append({
                "month": month,
                "median_edge_cents": round(float(np.median(edges)), 2),
                "p10_edge": round(float(np.percentile(edges, 10)), 2),
                "p90_edge": round(float(np.percentile(edges, 90)), 2),
                "median_volume": round(float(np.median(volumes)), 0),
                "median_profit": round(float(np.median(profits)), 2),
                "p10_profit": round(float(np.percentile(profits, 10)), 2),
                "p90_profit": round(float(np.percentile(profits, 90)), 2),
                "median_bankroll": round(float(np.median(bankrolls)), 2),
                "prob_profitable": round(float(np.mean(profits > 0)), 4),
                "prob_edge_alive": round(float(np.mean(edges > 0.5)), 4),
            })

        # Summary statistics
        final_bankrolls = np.array(monthly_results[12]["bankroll"])
        final_edges = np.array(monthly_results[12]["edge"])

        return {
            "projections": projections,
            "summary": {
                "prob_surviving_12m": round(float(np.mean(final_edges > 0.5)), 4),
                "prob_profitable_12m": round(float(np.mean(final_bankrolls > bankroll)), 4),
                "median_final_bankroll": round(float(np.median(final_bankrolls)), 2),
                "p10_final_bankroll": round(float(np.percentile(final_bankrolls, 10)), 2),
                "p90_final_bankroll": round(float(np.percentile(final_bankrolls, 90)), 2),
                "median_total_profit": round(float(np.median(final_bankrolls - bankroll)), 2),
                "expected_edge_at_12m": round(float(np.median(final_edges)), 2),
                "months_until_edge_death_median": self._median_edge_death(monthly_results),
            },
            "parameters": {
                "starting_clv": current_monthly_clv_cents,
                "starting_volume": current_monthly_volume,
                "decay_rate": round(monthly_decay_rate, 3),
                "starting_bankroll": bankroll,
                "n_simulations": self.n_sims,
            },
        }

    def _median_edge_death(self, monthly_results: dict) -> int:
        """Find median month when edge drops below 0.5 cents."""
        for month in range(1, 13):
            edges = np.array(monthly_results[month]["edge"])
            if float(np.median(edges)) < 0.5:
                return month
        return 13  # Edge survives all 12 months
