"""Book behavior model — How different books respond to winning accounts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BookProfile:
    """Profile of a sportsbook's behavior toward sharp bettors."""
    name: str
    limit_speed: str  # fast/medium/slow — how quickly they limit
    initial_limit_usd: float
    min_limit_after_restriction: float
    months_to_restriction: float  # Avg months before limits hit
    allows_arb: bool
    sharp_friendly: bool
    golf_market_depth: str  # shallow/moderate/deep


BOOK_PROFILES = {
    "pinnacle": BookProfile("Pinnacle", "slow", 5000, 500, 24, True, True, "deep"),
    "draftkings": BookProfile("DraftKings", "medium", 1000, 25, 6, False, False, "moderate"),
    "fanduel": BookProfile("FanDuel", "medium", 1000, 25, 6, False, False, "moderate"),
    "betmgm": BookProfile("BetMGM", "fast", 500, 10, 3, False, False, "shallow"),
    "caesars": BookProfile("Caesars", "fast", 500, 10, 3, False, False, "shallow"),
    "bet365": BookProfile("Bet365", "medium", 500, 25, 4, False, False, "moderate"),
}


class BookBehaviorModel:
    """Model how books react to profitable betting accounts."""

    def predict_limit_trajectory(
        self,
        book: str,
        monthly_profit_pct: float,
        months_active: int,
    ) -> dict:
        """Predict the limit trajectory for an account.

        Returns dict with current limit estimate and future projections.
        """
        profile = BOOK_PROFILES.get(book.lower())
        if not profile:
            return {"error": f"Unknown book: {book}", "current_limit_pct": 1.0}

        # Months until restriction based on profitability
        if monthly_profit_pct > 0.15:
            months_to_limit = profile.months_to_restriction * 0.5
        elif monthly_profit_pct > 0.08:
            months_to_limit = profile.months_to_restriction * 0.7
        elif monthly_profit_pct > 0.03:
            months_to_limit = profile.months_to_restriction
        else:
            months_to_limit = profile.months_to_restriction * 2.0

        # Current limit as percentage of initial
        if months_active < months_to_limit * 0.5:
            current_pct = 1.0
        elif months_active < months_to_limit:
            progress = (months_active - months_to_limit * 0.5) / (months_to_limit * 0.5)
            current_pct = 1.0 - progress * 0.6
        else:
            overage = (months_active - months_to_limit) / months_to_limit
            current_pct = max(0.05, 0.4 * (1.0 - overage))

        current_limit = profile.initial_limit_usd * current_pct

        # 6-month projection
        future_months = months_active + 6
        if future_months < months_to_limit:
            future_pct = 1.0 - max(0, (future_months - months_to_limit * 0.5) / (months_to_limit * 0.5)) * 0.6
        else:
            overage = (future_months - months_to_limit) / months_to_limit
            future_pct = max(0.05, 0.4 * (1.0 - overage))

        return {
            "book": book,
            "current_limit_pct": round(current_pct, 3),
            "current_limit_usd": round(current_limit, 2),
            "months_to_restriction": round(months_to_limit, 1),
            "is_restricted": current_pct < 0.5,
            "future_6m_limit_pct": round(future_pct, 3),
            "future_6m_limit_usd": round(profile.initial_limit_usd * future_pct, 2),
            "sharp_friendly": profile.sharp_friendly,
        }
