"""Limit model — Maximum bet sizes by book and market."""

from __future__ import annotations

import numpy as np


# Typical limits by book and market type (in USD)
DEFAULT_LIMITS = {
    "pinnacle": {"outright": 5000, "matchup": 2000, "top5": 3000, "top10": 3000, "top20": 2000, "make_cut": 2000},
    "draftkings": {"outright": 1000, "matchup": 500, "top5": 500, "top10": 500, "top20": 500, "make_cut": 500},
    "fanduel": {"outright": 1000, "matchup": 500, "top5": 500, "top10": 500, "top20": 500, "make_cut": 500},
    "betmgm": {"outright": 500, "matchup": 250, "top5": 300, "top10": 300, "top20": 300, "make_cut": 300},
    "caesars": {"outright": 500, "matchup": 250, "top5": 300, "top10": 300, "top20": 300, "make_cut": 300},
    "prizepicks": {"outright": 0, "matchup": 0, "top5": 0, "top10": 0, "top20": 0, "make_cut": 0},
}

# Account limit reduction factor based on profitability
LIMIT_REDUCTION_SCHEDULE = {
    # profit_pct -> remaining_limit_pct
    0.0: 1.0,     # New account
    0.05: 0.90,   # Slightly profitable
    0.10: 0.70,   # Clearly profitable
    0.20: 0.40,   # Very profitable
    0.30: 0.20,   # Heavily limited
    0.50: 0.05,   # Effectively banned
}


class LimitModel:
    """Model betting limits and account restrictions."""

    def __init__(self, limits: dict | None = None):
        self.limits = limits or DEFAULT_LIMITS

    def get_max_stake(
        self,
        book: str,
        market_type: str,
        account_profit_pct: float = 0.0,
        months_active: int = 0,
    ) -> dict:
        """Get maximum stake for a book/market considering limit reductions.

        Returns:
            {
                'base_limit': float,
                'effective_limit': float,
                'reduction_factor': float,
                'is_limited': bool,
            }
        """
        book_limits = self.limits.get(book.lower(), {})
        base_limit = book_limits.get(market_type, 100)

        # Apply limit reduction based on profitability
        reduction = 1.0
        for threshold, factor in sorted(LIMIT_REDUCTION_SCHEDULE.items()):
            if account_profit_pct >= threshold:
                reduction = factor

        # Time-based limit relaxation (new accounts may have lower initial limits)
        if months_active < 1:
            reduction *= 0.5  # Half limits for new accounts
        elif months_active < 3:
            reduction *= 0.8

        effective_limit = base_limit * reduction

        return {
            "base_limit": base_limit,
            "effective_limit": round(effective_limit, 2),
            "reduction_factor": round(reduction, 3),
            "is_limited": reduction < 0.5,
            "book": book,
            "market_type": market_type,
        }

    def estimate_total_capacity(
        self,
        market_type: str,
        account_profit_pct: float = 0.0,
    ) -> dict:
        """Estimate total betting capacity across all books."""
        total_capacity = 0.0
        book_details = {}

        for book in self.limits:
            result = self.get_max_stake(book, market_type, account_profit_pct)
            total_capacity += result["effective_limit"]
            book_details[book] = result

        return {
            "total_capacity": round(total_capacity, 2),
            "by_book": book_details,
            "n_books_available": sum(1 for d in book_details.values() if d["effective_limit"] > 10),
        }
