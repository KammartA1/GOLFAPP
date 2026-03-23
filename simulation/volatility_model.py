"""
Golf Quant Engine — Volatility Model
======================================
Per-player scoring volatility analysis.  High-variance players are
better outright bets; low-variance players are better top-20 bets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from simulation.player_model import PlayerModel


@dataclass
class VolatilityProfile:
    """Volatility analysis for a single player."""
    player_name: str
    volatility: float           # Raw volatility multiplier
    scoring_std: float          # Expected round scoring std dev
    upside_percentile: float    # Expected score at 5th percentile (best rounds)
    downside_percentile: float  # Expected score at 95th percentile (worst rounds)
    boom_probability: float     # P(round score < -4 relative to field avg)
    bust_probability: float     # P(round score > +4 relative to field avg)
    category: str               # "low", "medium", "high", "extreme"

    # Market recommendations
    best_bet_type: str          # Recommended bet type based on profile
    outright_edge_factor: float # Multiplier on outright value (>1 = good for outrights)
    top20_edge_factor: float    # Multiplier on top-20 value (>1 = good for top 20)
    h2h_suitability: float      # How suitable for H2H (0-1, lower vol = better)


class VolatilityModel:
    """Analyze and apply player volatility to market selection.

    Key insight: volatility is the most under-appreciated factor in
    golf betting markets. High-variance players are systematically
    undervalued for outright bets and overvalued for top-20 bets.

    This happens because:
    - Outright odds are set based on average skill level
    - But winning requires an outlier performance
    - High-vol players produce more outlier rounds
    - Therefore high-vol players outperform their outright odds
    """

    # Volatility category thresholds
    CATEGORY_THRESHOLDS = {
        "low": (0.0, 0.85),
        "medium": (0.85, 1.10),
        "high": (1.10, 1.35),
        "extreme": (1.35, float("inf")),
    }

    def __init__(self):
        pass

    def analyze_player(
        self,
        player: PlayerModel,
        field_avg_score: float = 71.5,
    ) -> VolatilityProfile:
        """Produce a full volatility profile for a player.

        Parameters
        ----------
        player : PlayerModel
            The player to analyze.
        field_avg_score : float
            Expected field average score (for boom/bust calculation).
        """
        vol = player.volatility
        scoring_std = player.overall_std

        # Category
        category = "medium"
        for cat, (lo, hi) in self.CATEGORY_THRESHOLDS.items():
            if lo <= vol < hi:
                category = cat
                break

        # Expected best and worst rounds (5th and 95th percentiles)
        # Score relative to par, so negative = under par = good
        mean_score = -player.sg_total_mean  # SG to score conversion (approx)
        upside = mean_score - 1.645 * scoring_std  # 5th percentile
        downside = mean_score + 1.645 * scoring_std  # 95th percentile

        # Boom probability: P(round score more than 4 under field avg)
        # In SG terms, a "boom" is SG > +4 for the day
        boom_z = (4.0 - player.sg_total_mean) / max(scoring_std, 0.01)
        boom_prob = 1.0 - float(_norm_cdf(boom_z))

        # Bust probability: P(round score more than 4 over field avg)
        bust_z = (-4.0 - player.sg_total_mean) / max(scoring_std, 0.01)
        bust_prob = float(_norm_cdf(bust_z))

        # Market recommendations
        outright_edge = self._calc_outright_edge_factor(vol, scoring_std)
        top20_edge = self._calc_top20_edge_factor(vol)
        h2h_suit = self._calc_h2h_suitability(vol)
        best_bet = self._recommend_bet_type(vol, player.sg_total_mean)

        return VolatilityProfile(
            player_name=player.name,
            volatility=vol,
            scoring_std=round(scoring_std, 3),
            upside_percentile=round(upside, 2),
            downside_percentile=round(downside, 2),
            boom_probability=round(boom_prob, 4),
            bust_probability=round(bust_prob, 4),
            category=category,
            best_bet_type=best_bet,
            outright_edge_factor=round(outright_edge, 3),
            top20_edge_factor=round(top20_edge, 3),
            h2h_suitability=round(h2h_suit, 3),
        )

    def analyze_field(
        self,
        players: list[PlayerModel],
    ) -> list[VolatilityProfile]:
        """Analyze volatility for an entire field."""
        return [self.analyze_player(p) for p in players]

    def rank_by_outright_value(
        self,
        players: list[PlayerModel],
    ) -> list[VolatilityProfile]:
        """Rank players by outright betting value (skill + volatility)."""
        profiles = self.analyze_field(players)
        # Sort by outright edge factor * overall skill
        profiles.sort(
            key=lambda p: p.outright_edge_factor * (1.0 + max(0, -p.upside_percentile)),
            reverse=True,
        )
        return profiles

    def rank_by_top20_value(
        self,
        players: list[PlayerModel],
    ) -> list[VolatilityProfile]:
        """Rank players by top-20 consistency value."""
        profiles = self.analyze_field(players)
        profiles.sort(key=lambda p: p.top20_edge_factor, reverse=True)
        return profiles

    def get_optimal_market(
        self,
        player: PlayerModel,
    ) -> dict:
        """Determine which market type offers the most edge for a player.

        Returns dict with market recommendations and reasoning.
        """
        profile = self.analyze_player(player)

        markets = {
            "outright": {
                "edge_factor": profile.outright_edge_factor,
                "reasoning": self._outright_reasoning(profile),
            },
            "top5": {
                "edge_factor": (profile.outright_edge_factor + profile.top20_edge_factor) / 2,
                "reasoning": "Blend of upside and consistency",
            },
            "top10": {
                "edge_factor": profile.top20_edge_factor * 0.9 + profile.outright_edge_factor * 0.1,
                "reasoning": "Mostly consistency with some upside",
            },
            "top20": {
                "edge_factor": profile.top20_edge_factor,
                "reasoning": self._top20_reasoning(profile),
            },
            "h2h": {
                "edge_factor": profile.h2h_suitability,
                "reasoning": self._h2h_reasoning(profile),
            },
        }

        best_market = max(markets, key=lambda k: markets[k]["edge_factor"])

        return {
            "player": player.name,
            "best_market": best_market,
            "profile": profile,
            "all_markets": markets,
        }

    def calculate_volatility_adjusted_probability(
        self,
        base_probability: float,
        volatility: float,
        market_type: str,
    ) -> float:
        """Adjust a probability estimate based on volatility.

        This is the key function for generating edges:
        - Sportsbooks set lines based on average skill (mean)
        - But market outcomes depend on the distribution shape
        - High-vol players win outrights MORE than their skill suggests

        Parameters
        ----------
        base_probability : float
            Probability based on mean skill alone.
        volatility : float
            Player's volatility multiplier.
        market_type : str
            Type of market: "outright", "top5", "top10", "top20", "make_cut".
        """
        if market_type == "outright":
            # High vol boosts outright probability superlinearly
            vol_factor = 1.0 + (volatility - 1.0) * 0.8
            return base_probability * max(vol_factor, 0.3)
        elif market_type in ("top5", "top10"):
            # Moderate vol benefit for top finishes
            vol_factor = 1.0 + (volatility - 1.0) * 0.3
            return base_probability * max(vol_factor, 0.5)
        elif market_type == "top20":
            # Low vol is BETTER for top 20 (consistency wins)
            vol_factor = 1.0 - (volatility - 1.0) * 0.4
            return base_probability * max(vol_factor, 0.3)
        elif market_type == "make_cut":
            # Low vol strongly favors making cuts
            vol_factor = 1.0 - (volatility - 1.0) * 0.6
            return base_probability * max(vol_factor, 0.2)
        return base_probability

    # --- Private helpers ---

    def _calc_outright_edge_factor(self, vol: float, scoring_std: float) -> float:
        """Higher vol = better for outrights.  Nonlinear relationship."""
        # Base factor from volatility
        base = 1.0 + (vol - 1.0) * 1.5
        # Bonus for high scoring std
        std_bonus = max(0, (scoring_std - 2.5) * 0.15)
        return max(0.3, base + std_bonus)

    def _calc_top20_edge_factor(self, vol: float) -> float:
        """Lower vol = better for top 20.  Inverse relationship."""
        return max(0.3, 1.0 - (vol - 1.0) * 1.2)

    def _calc_h2h_suitability(self, vol: float) -> float:
        """Low-to-medium vol is best for H2H (predictable)."""
        if vol < 0.9:
            return 0.95
        elif vol < 1.1:
            return 0.85
        elif vol < 1.3:
            return 0.65
        return 0.45

    def _recommend_bet_type(self, vol: float, sg_total_mean: float) -> str:
        """Recommend the single best bet type for this player."""
        if sg_total_mean > 1.5 and vol > 1.1:
            return "outright"
        elif sg_total_mean > 1.0 and vol >= 0.95:
            return "top5"
        elif sg_total_mean > 0.5 and vol < 1.0:
            return "top20"
        elif vol < 0.90:
            return "make_cut"
        elif vol > 1.2:
            return "outright"
        return "top10"

    def _outright_reasoning(self, profile: VolatilityProfile) -> str:
        if profile.category in ("high", "extreme"):
            return "High variance creates outright value — boom rounds win tournaments"
        elif profile.category == "medium":
            return "Average variance — outright value depends on skill level"
        return "Low variance player — limited outright value despite consistency"

    def _top20_reasoning(self, profile: VolatilityProfile) -> str:
        if profile.category == "low":
            return "Consistency monster — ideal top-20 candidate, rarely has blow-up rounds"
        elif profile.category == "medium":
            return "Moderate consistency — solid top-20 candidate"
        return "High variance hurts top-20 probability — too many bust rounds"

    def _h2h_reasoning(self, profile: VolatilityProfile) -> str:
        if profile.h2h_suitability > 0.8:
            return "Predictable scorer — H2H lines will be accurate, look for small edges"
        elif profile.h2h_suitability > 0.6:
            return "Moderate predictability — H2H viable if edge is clear"
        return "Too volatile for reliable H2H — outcomes are coin-flippy"


def _norm_cdf(z: float) -> float:
    """Standard normal CDF without scipy import at module level."""
    from scipy.stats import norm
    return float(norm.cdf(z))
