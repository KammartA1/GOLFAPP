"""
Golf Quant Engine — Hole Engine
================================
Simulates a single hole outcome for a player, given their SG draw,
hole characteristics, and weather conditions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from simulation.config import SimulationConfig
from simulation.course_model import HoleSpec, CourseModel
from simulation.player_model import PlayerSGComponents
from simulation.weather_model import WeatherConditions


# Outcome indices in probability arrays
EAGLE = 0
BIRDIE = 1
PAR = 2
BOGEY = 3
DOUBLE_PLUS = 4

# Score relative to par for each outcome
OUTCOME_SCORES = {
    3: [-2, -1, 0, 1, 2],  # Par 3: ace, birdie, par, bogey, double
    4: [-2, -1, 0, 1, 2],  # Par 4: eagle, birdie, par, bogey, double
    5: [-2, -1, 0, 1, 2],  # Par 5: eagle(albatross=-3 lumped), birdie, par, bogey, double
}


class HoleEngine:
    """Simulate the outcome of a single hole.

    The engine combines:
    1. Player's drawn SG components, weighted by hole-specific SG importance
    2. Hole difficulty (base scoring distribution)
    3. Weather penalty
    4. Momentum from previous holes (hot/cold streak)
    5. Pin position / green firmness (difficulty modifier)

    Output: a discrete score relative to par (typically -2 to +3).
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()

    def simulate_hole(
        self,
        player_sg: PlayerSGComponents,
        hole: HoleSpec,
        rng: np.random.Generator,
        weather: Optional[WeatherConditions] = None,
        weather_penalty: float = 0.0,
        momentum: float = 0.0,
        difficulty_modifier: float = 0.0,
    ) -> int:
        """Simulate one hole and return score relative to par.

        Parameters
        ----------
        player_sg : PlayerSGComponents
            Player's drawn SG values for this round.
        hole : HoleSpec
            Hole specification.
        rng : np.random.Generator
            Seeded RNG.
        weather : WeatherConditions, optional
            Current weather (used if weather_penalty not pre-calculated).
        weather_penalty : float
            Pre-calculated weather penalty for this hole.
        momentum : float
            Momentum from recent holes (-0.1 to +0.1 typical).
            Positive = playing well recently.
        difficulty_modifier : float
            Additional difficulty modifier (pin position, green firmness).
            Positive = harder.

        Returns
        -------
        int : Score relative to par (e.g., -1 = birdie, 0 = par, +1 = bogey).
        """
        # Calculate player's effective SG for this hole
        hole_sg = self._calculate_hole_sg(player_sg, hole)

        # Apply momentum
        hole_sg += momentum * 0.5

        # Get scoring distribution
        probs = self._get_outcome_probabilities(
            hole_sg=hole_sg,
            hole=hole,
            weather_penalty=weather_penalty + difficulty_modifier,
        )

        # Draw outcome
        outcome_idx = rng.choice(5, p=probs)

        # Map to score relative to par
        scores = OUTCOME_SCORES[hole.par]
        return scores[outcome_idx]

    def simulate_hole_batch(
        self,
        player_sg_array: np.ndarray,
        hole: HoleSpec,
        rng: np.random.Generator,
        weather_penalties: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Simulate a hole for many players (or many simulations) at once.

        Parameters
        ----------
        player_sg_array : np.ndarray
            Shape (n, 4) with columns [sg_ott, sg_app, sg_atg, sg_putt].
        hole : HoleSpec
            Hole specification.
        rng : np.random.Generator
            Seeded RNG.
        weather_penalties : np.ndarray, optional
            Shape (n,) per-player weather penalties.

        Returns
        -------
        np.ndarray of int, shape (n,) — scores relative to par.
        """
        n = player_sg_array.shape[0]

        # Calculate hole SG for each player
        weights = np.array([
            hole.sg_weight_ott,
            hole.sg_weight_app,
            hole.sg_weight_atg,
            hole.sg_weight_putt,
        ])
        hole_sgs = player_sg_array @ weights  # (n,)

        if weather_penalties is None:
            weather_penalties = np.zeros(n)

        # Get scoring distribution for each player
        results = np.zeros(n, dtype=np.int32)
        scores_for_par = OUTCOME_SCORES[hole.par]

        for i in range(n):
            probs = self._get_outcome_probabilities(
                hole_sg=hole_sgs[i],
                hole=hole,
                weather_penalty=weather_penalties[i],
            )
            outcome_idx = rng.choice(5, p=probs)
            results[i] = scores_for_par[outcome_idx]

        return results

    def _calculate_hole_sg(
        self,
        player_sg: PlayerSGComponents,
        hole: HoleSpec,
    ) -> float:
        """Calculate the player's effective SG contribution for this hole.

        Weights the player's SG components by the hole's SG importance.
        Par 3s have zero OTT weight.  Par 5s have high OTT weight.
        """
        return (
            player_sg.sg_ott * hole.sg_weight_ott
            + player_sg.sg_app * hole.sg_weight_app
            + player_sg.sg_atg * hole.sg_weight_atg
            + player_sg.sg_putt * hole.sg_weight_putt
        )

    def _get_outcome_probabilities(
        self,
        hole_sg: float,
        hole: HoleSpec,
        weather_penalty: float = 0.0,
    ) -> np.ndarray:
        """Get the 5-outcome probability distribution for a hole.

        Adjusts base rates by player skill and weather.
        Returns normalized probability array: [eagle, birdie, par, bogey, double+].
        """
        # Get base rates from config
        base_rates = np.array(
            self.config.scoring_rates.get_rates(hole.par, hole.difficulty_tier),
            dtype=np.float64,
        )

        # Net SG: positive = better than average
        net_sg = hole_sg - weather_penalty

        # Shift probabilities
        shift = net_sg * 0.20

        if shift > 0:
            # Better: more birdies/eagles, fewer bogeys/doubles
            adjusted = np.array([
                base_rates[0] + shift * 0.08,
                base_rates[1] + shift * 0.55,
                base_rates[2] + shift * 0.05,
                base_rates[3] - shift * 0.45,
                base_rates[4] - shift * 0.23,
            ])
        else:
            # Worse: fewer birdies, more bogeys/doubles
            abs_shift = abs(shift)
            adjusted = np.array([
                base_rates[0] - abs_shift * 0.05,
                base_rates[1] - abs_shift * 0.45,
                base_rates[2] - abs_shift * 0.05,
                base_rates[3] + abs_shift * 0.38,
                base_rates[4] + abs_shift * 0.17,
            ])

        # Green complexity: affects putting/ATG variance
        if hole.green_complexity > 0.7:
            complexity_effect = (hole.green_complexity - 0.5) * 0.03
            adjusted[1] -= complexity_effect
            adjusted[3] += complexity_effect * 0.7
            adjusted[4] += complexity_effect * 0.3

        # Clamp and normalize
        adjusted = np.maximum(adjusted, 0.001)
        adjusted /= adjusted.sum()

        return adjusted

    def estimate_expected_score(
        self,
        player_sg: PlayerSGComponents,
        hole: HoleSpec,
        weather_penalty: float = 0.0,
    ) -> float:
        """Calculate expected score relative to par (without randomness).

        Useful for analytical calculations and validation.
        """
        probs = self._get_outcome_probabilities(
            hole_sg=self._calculate_hole_sg(player_sg, hole),
            hole=hole,
            weather_penalty=weather_penalty,
        )
        scores = np.array(OUTCOME_SCORES[hole.par], dtype=np.float64)
        return float(np.dot(probs, scores))
