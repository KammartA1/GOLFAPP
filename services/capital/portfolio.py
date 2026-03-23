"""Portfolio management — Correlation matrix, optimal allocation, concentration checks."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manage portfolio of golf bets — diversification and concentration."""

    def __init__(
        self,
        max_player_concentration: float = 0.20,
        max_tournament_concentration: float = 0.40,
        max_market_type_concentration: float = 0.50,
    ):
        self.max_player = max_player_concentration
        self.max_tournament = max_tournament_concentration
        self.max_market_type = max_market_type_concentration

    def check_concentration(
        self,
        pending_bets: List[dict],
        new_bet: dict,
        bankroll: float,
    ) -> dict:
        """Check if a new bet would create dangerous concentration.

        Args:
            pending_bets: List of {player, tournament, market_type, stake}.
            new_bet: The proposed bet with same keys.
            bankroll: Current bankroll.

        Returns:
            {
                'allowed': bool,
                'violations': list,
                'max_allowed_stake': float,
                'current_exposure': dict,
            }
        """
        all_bets = pending_bets + [new_bet]
        total_exposure = sum(b["stake"] for b in all_bets)

        # Player concentration
        by_player = defaultdict(float)
        for b in all_bets:
            by_player[b["player"]] += b["stake"]

        # Tournament concentration
        by_tournament = defaultdict(float)
        for b in all_bets:
            by_tournament[b.get("tournament", "unknown")] += b["stake"]

        # Market type concentration
        by_market = defaultdict(float)
        for b in all_bets:
            by_market[b.get("market_type", "outright")] += b["stake"]

        violations = []
        max_allowed = new_bet["stake"]

        # Player check
        player_pct = by_player[new_bet["player"]] / max(bankroll, 1)
        if player_pct > self.max_player:
            max_for_player = bankroll * self.max_player - (by_player[new_bet["player"]] - new_bet["stake"])
            max_allowed = min(max_allowed, max(0, max_for_player))
            violations.append(f"Player {new_bet['player']} concentration: {player_pct:.1%} > {self.max_player:.0%}")

        # Tournament check
        tourn = new_bet.get("tournament", "unknown")
        tourn_pct = by_tournament[tourn] / max(bankroll, 1)
        if tourn_pct > self.max_tournament:
            max_for_tourn = bankroll * self.max_tournament - (by_tournament[tourn] - new_bet["stake"])
            max_allowed = min(max_allowed, max(0, max_for_tourn))
            violations.append(f"Tournament concentration: {tourn_pct:.1%} > {self.max_tournament:.0%}")

        # Market type check
        mtype = new_bet.get("market_type", "outright")
        market_pct = by_market[mtype] / max(bankroll, 1)
        if market_pct > self.max_market_type:
            max_for_market = bankroll * self.max_market_type - (by_market[mtype] - new_bet["stake"])
            max_allowed = min(max_allowed, max(0, max_for_market))
            violations.append(f"Market {mtype} concentration: {market_pct:.1%} > {self.max_market_type:.0%}")

        return {
            "allowed": len(violations) == 0,
            "violations": violations,
            "max_allowed_stake": round(max(0, max_allowed), 2),
            "current_exposure": {
                "total": round(total_exposure, 2),
                "total_pct_bankroll": round(total_exposure / max(bankroll, 1), 4),
                "by_player": dict(by_player),
                "by_tournament": dict(by_tournament),
                "by_market_type": dict(by_market),
            },
        }

    def correlation_matrix(
        self,
        bet_outcomes: Dict[str, List[float]],
    ) -> dict:
        """Compute correlation matrix between bet categories.

        Args:
            bet_outcomes: {category_name: [list of returns]}

        Returns:
            Correlation matrix and diversification score.
        """
        categories = sorted(bet_outcomes.keys())
        if len(categories) < 2:
            return {"matrix": {}, "diversification_score": 1.0}

        min_len = min(len(bet_outcomes[c]) for c in categories)
        if min_len < 5:
            return {"matrix": {}, "diversification_score": 1.0, "reason": "insufficient data"}

        data = np.column_stack([np.array(bet_outcomes[c][:min_len]) for c in categories])
        corr = np.corrcoef(data.T)

        matrix = {}
        for i, ci in enumerate(categories):
            matrix[ci] = {}
            for j, cj in enumerate(categories):
                matrix[ci][cj] = round(float(corr[i, j]), 3)

        # Diversification score: lower average correlation = better diversification
        off_diag = corr[np.triu_indices(len(categories), k=1)]
        avg_corr = float(np.mean(np.abs(off_diag))) if len(off_diag) > 0 else 0.0
        div_score = max(0, 1.0 - avg_corr)

        return {
            "matrix": matrix,
            "categories": categories,
            "avg_abs_correlation": round(avg_corr, 3),
            "diversification_score": round(div_score, 3),
        }

    def optimal_allocation(
        self,
        expected_returns: Dict[str, float],
        volatilities: Dict[str, float],
        correlations: Dict[str, Dict[str, float]] | None = None,
    ) -> Dict[str, float]:
        """Compute optimal allocation across categories using inverse-volatility weighting.

        Simple but robust: allocate inversely proportional to volatility,
        adjusted by expected return.
        """
        categories = sorted(expected_returns.keys())
        if not categories:
            return {}

        weights = {}
        total_weight = 0.0

        for cat in categories:
            er = expected_returns.get(cat, 0.0)
            vol = max(volatilities.get(cat, 1.0), 0.01)

            # Risk-adjusted weight: return / volatility (Sharpe-like)
            if er > 0:
                w = er / vol
            else:
                w = 0.0  # Don't allocate to negative expected return

            weights[cat] = w
            total_weight += w

        # Normalize to sum to 1.0
        if total_weight > 0:
            return {cat: round(w / total_weight, 4) for cat, w in weights.items()}
        else:
            # Equal weight if no positive expected returns
            n = len(categories)
            return {cat: round(1.0 / n, 4) for cat in categories}
