"""Rejection model — Bet rejection probability by book."""

from __future__ import annotations

import numpy as np


class RejectionModel:
    """Model bet rejection/voiding probability."""

    def __init__(self):
        # Base rejection rates by book type
        self.base_rejection_rates = {
            "pinnacle": 0.02,      # Pinnacle rarely rejects
            "draftkings": 0.05,
            "fanduel": 0.05,
            "betmgm": 0.08,
            "caesars": 0.10,
            "bet365": 0.07,
            "prizepicks": 0.03,     # DFS, different rejection mechanism
        }

    def rejection_probability(
        self,
        book: str,
        stake_dollars: float,
        edge_pct: float,
        account_age_months: int = 6,
        account_profit_pct: float = 0.0,
        is_sharp_line: bool = False,
    ) -> dict:
        """Estimate probability that a bet will be rejected.

        Factors:
          - Book's base rejection rate
          - Stake size (larger = more scrutiny)
          - Edge size (sharper bets = more likely flagged)
          - Account profitability (winners get limited)
          - Line sharpness (betting into sharp moves = flagged)
        """
        base_rate = self.base_rejection_rates.get(book.lower(), 0.05)

        # Stake factor: higher stakes = more scrutiny
        stake_factor = 1.0
        if stake_dollars > 1000:
            stake_factor = 1.0 + (stake_dollars - 1000) / 5000.0
        elif stake_dollars > 500:
            stake_factor = 1.0 + (stake_dollars - 500) / 5000.0

        # Edge factor: sharper bets attract attention
        edge_factor = 1.0
        if edge_pct > 0.10:
            edge_factor = 2.0
        elif edge_pct > 0.05:
            edge_factor = 1.5

        # Profitability factor: winning accounts get limited
        profit_factor = 1.0
        if account_profit_pct > 0.20:
            profit_factor = 3.0
        elif account_profit_pct > 0.10:
            profit_factor = 2.0
        elif account_profit_pct > 0.05:
            profit_factor = 1.5

        # Sharp line factor
        sharp_factor = 1.5 if is_sharp_line else 1.0

        # New account factor (more scrutiny)
        age_factor = 1.3 if account_age_months < 3 else 1.0

        rejection_prob = min(0.95, base_rate * stake_factor * edge_factor *
                             profit_factor * sharp_factor * age_factor)

        return {
            "rejection_probability": round(rejection_prob, 4),
            "base_rate": base_rate,
            "stake_factor": round(stake_factor, 3),
            "edge_factor": round(edge_factor, 3),
            "profit_factor": round(profit_factor, 3),
            "expected_to_be_rejected": rejection_prob > 0.30,
            "recommendation": self._recommendation(rejection_prob, book),
        }

    def _recommendation(self, prob: float, book: str) -> str:
        if prob > 0.50:
            return f"High rejection risk at {book}. Consider alternative book or reduced stake."
        if prob > 0.20:
            return f"Moderate rejection risk. Proceed with caution."
        return "Low rejection risk. Proceed normally."

    def expected_fill_rate(self, book: str, n_bets: int = 100, avg_edge: float = 0.05) -> float:
        """Expected percentage of bets that will actually be filled."""
        result = self.rejection_probability(book, 100, avg_edge)
        return round(1.0 - result["rejection_probability"], 4)
