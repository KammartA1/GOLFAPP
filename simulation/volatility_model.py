"""Volatility model — Per-player scoring volatility (boom/bust vs consistent)."""

from __future__ import annotations

from typing import List

import numpy as np


class VolatilityModel:
    """Model per-player scoring volatility for simulation."""

    TOUR_AVG_STD = 2.75  # Tour average round-to-round SG std

    def estimate_volatility(
        self,
        sg_history: List[float],
        min_rounds: int = 8,
    ) -> dict:
        """Estimate player volatility from SG history.

        Returns dict with volatility parameters for simulation.
        """
        if len(sg_history) < min_rounds:
            return {
                "round_std": self.TOUR_AVG_STD,
                "volatility_multiplier": 1.0,
                "is_boom_bust": False,
                "is_consistent": False,
                "confidence": 0.0,
            }

        sg_arr = np.array(sg_history)
        player_std = float(np.std(sg_arr))

        # Bayesian shrinkage toward tour average
        n = len(sg_arr)
        shrinkage = min(n / 20.0, 1.0)
        estimated_std = player_std * shrinkage + self.TOUR_AVG_STD * (1.0 - shrinkage)
        estimated_std = max(1.0, min(5.0, estimated_std))

        vol_multiplier = estimated_std / self.TOUR_AVG_STD

        # Classify player type
        is_boom_bust = vol_multiplier > 1.2  # High variance
        is_consistent = vol_multiplier < 0.85  # Low variance

        # Skewness: some players have more upside than downside
        skewness = float(np.mean(((sg_arr - np.mean(sg_arr)) / max(player_std, 0.1)) ** 3))

        # Kurtosis: fat tails (extreme performances)
        kurtosis = float(np.mean(((sg_arr - np.mean(sg_arr)) / max(player_std, 0.1)) ** 4))

        return {
            "round_std": round(estimated_std, 3),
            "volatility_multiplier": round(vol_multiplier, 3),
            "is_boom_bust": is_boom_bust,
            "is_consistent": is_consistent,
            "skewness": round(skewness, 3),
            "kurtosis": round(kurtosis, 3),
            "raw_std": round(player_std, 3),
            "n_rounds": n,
            "confidence": round(shrinkage, 3),
        }

    def apply_volatility_to_round(
        self,
        base_sg: float,
        round_std: float,
        volatility_multiplier: float,
        rng: np.random.Generator,
    ) -> float:
        """Generate a round score with player-specific volatility.

        Returns total SG for the round.
        """
        # Base noise
        noise = rng.normal(0.0, round_std)

        # Apply volatility multiplier
        noise *= volatility_multiplier

        return base_sg + noise

    def upside_probability(self, sg_total: float, round_std: float, threshold_sg: float = 3.0) -> float:
        """Probability of a 'boom' round (SG > threshold).

        Useful for GPP DFS analysis — need upside for tournaments.
        """
        from scipy import stats
        if round_std <= 0:
            return 0.0
        return float(1.0 - stats.norm.cdf(threshold_sg, loc=sg_total, scale=round_std))

    def downside_probability(self, sg_total: float, round_std: float, threshold_sg: float = -2.0) -> float:
        """Probability of a 'bust' round (SG < threshold)."""
        from scipy import stats
        if round_std <= 0:
            return 0.0
        return float(stats.norm.cdf(threshold_sg, loc=sg_total, scale=round_std))
