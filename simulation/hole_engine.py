"""Hole-by-hole discrete outcome model.

Models each hole as a discrete outcome: eagle / birdie / par / bogey / double+
Probabilities are derived from player SG components and hole characteristics.
"""

from __future__ import annotations

import numpy as np

from simulation.config import SimulationConfig


class HoleEngine:
    """Simulate discrete outcomes for a single hole."""

    def __init__(self, config: SimulationConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    def simulate_hole(
        self,
        hole_par: int,
        hole_difficulty: float,
        hole_distance: int,
        player_sg_per_hole: float,
        player_volatility: float,
        weather_adjustment: float = 0.0,
        pressure_adjustment: float = 0.0,
        momentum: float = 0.0,
    ) -> int:
        """Simulate a single hole outcome.

        Args:
            hole_par: 3, 4, or 5
            hole_difficulty: 0.0 = average, positive = harder
            hole_distance: Hole distance in yards
            player_sg_per_hole: Player's expected SG per hole (total/18)
            player_volatility: Player-specific volatility multiplier (1.0 = average)
            weather_adjustment: Weather effect on scoring (positive = harder)
            pressure_adjustment: Pressure effect (positive = worse under pressure)
            momentum: Previous hole momentum effect

        Returns:
            Score relative to par (e.g., -1 = birdie, 0 = par, +1 = bogey)
        """
        # Effective player skill for this hole
        effective_sg = player_sg_per_hole - hole_difficulty - weather_adjustment - pressure_adjustment
        effective_sg += momentum * self.config.hole_to_hole_correlation

        # Convert SG to outcome probabilities
        probs = self._sg_to_probabilities(hole_par, effective_sg, player_volatility)

        # Sample outcome
        outcomes = list(probs.keys())
        probabilities = list(probs.values())
        result = self.rng.choice(outcomes, p=probabilities)

        return int(result)

    def _sg_to_probabilities(
        self,
        hole_par: int,
        effective_sg: float,
        volatility: float,
    ) -> dict:
        """Convert effective SG per hole to discrete outcome probabilities.

        Uses logistic-based model calibrated to PGA Tour scoring distributions.
        """
        # Base rates by par
        if hole_par == 3:
            base = {
                -2: 0.001,   # Hole in one / eagle
                -1: 0.14,    # Birdie
                0: 0.62,     # Par
                1: 0.19,     # Bogey
                2: 0.049,    # Double+
            }
        elif hole_par == 5:
            base = {
                -2: 0.06,    # Eagle
                -1: 0.35,    # Birdie
                0: 0.45,     # Par
                1: 0.11,     # Bogey
                2: 0.03,     # Double+
            }
        else:  # par 4
            base = {
                -2: self.config.base_eagle_rate,
                -1: self.config.base_birdie_rate,
                0: self.config.base_par_rate,
                1: self.config.base_bogey_rate,
                2: self.config.base_double_plus_rate,
            }

        # Adjust for player skill (SG)
        # Positive SG -> more birdies/eagles, fewer bogeys
        skill_shift = effective_sg * 18.0  # Scale to per-hole impact

        adjusted = {}
        # Birdie/eagle probability increases with skill
        adjusted[-2] = base[-2] * np.exp(skill_shift * 0.8)
        adjusted[-1] = base[-1] * np.exp(skill_shift * 0.5)
        adjusted[0] = base[0]
        # Bogey/double probability decreases with skill
        adjusted[1] = base[1] * np.exp(-skill_shift * 0.4)
        adjusted[2] = base[2] * np.exp(-skill_shift * 0.6)

        # Volatility adjustment: high-volatility players have more extreme outcomes
        if volatility != 1.0:
            vol_factor = volatility
            mean_score = sum(k * v for k, v in adjusted.items())
            for score in adjusted:
                deviation = score - mean_score
                if abs(deviation) > 0.5:
                    adjusted[score] *= vol_factor ** (abs(deviation) * 0.5)

        # Normalize to sum to 1.0
        total = sum(adjusted.values())
        if total <= 0:
            return {-2: 0.005, -1: 0.18, 0: 0.60, 1: 0.16, 2: 0.055}

        normalized = {k: max(v / total, 0.0001) for k, v in adjusted.items()}
        # Re-normalize after clamping
        total2 = sum(normalized.values())
        return {k: v / total2 for k, v in normalized.items()}
