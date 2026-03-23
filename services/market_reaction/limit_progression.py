"""Limit progression — Track and project account limit changes over time."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from services.market_reaction.book_behavior import BookBehaviorModel, BOOK_PROFILES

logger = logging.getLogger(__name__)


class LimitProgressionTracker:
    """Track limit progression across all books over time."""

    def __init__(self):
        self.book_model = BookBehaviorModel()
        self._history: List[dict] = []

    def record_limit_observation(
        self,
        book: str,
        max_accepted_stake: float,
        was_rejected: bool = False,
        month: int = 0,
    ) -> None:
        """Record an observed limit data point."""
        self._history.append({
            "book": book,
            "max_stake": max_accepted_stake,
            "rejected": was_rejected,
            "month": month,
        })

    def project_total_capacity(
        self,
        monthly_profit_pct: float = 0.08,
        months_active: int = 6,
    ) -> dict:
        """Project total betting capacity across all books."""
        total_now = 0.0
        total_6m = 0.0
        book_details = {}

        for book_name in BOOK_PROFILES:
            trajectory = self.book_model.predict_limit_trajectory(
                book_name, monthly_profit_pct, months_active
            )
            total_now += trajectory["current_limit_usd"]
            total_6m += trajectory["future_6m_limit_usd"]
            book_details[book_name] = trajectory

        return {
            "total_current_capacity": round(total_now, 2),
            "total_6m_capacity": round(total_6m, 2),
            "capacity_decline_pct": round(1 - total_6m / max(total_now, 1), 4),
            "n_books_available": sum(1 for d in book_details.values() if d["current_limit_usd"] > 50),
            "by_book": book_details,
        }

    def sustainable_monthly_volume(
        self,
        monthly_profit_pct: float = 0.08,
        months_active: int = 6,
    ) -> float:
        """Estimate sustainable monthly betting volume considering limits."""
        capacity = self.project_total_capacity(monthly_profit_pct, months_active)
        # Assume we can cycle through capacity ~4x per month (weekly tournaments)
        return round(capacity["total_current_capacity"] * 4, 2)
