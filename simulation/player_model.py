"""Player model — SG components as inputs, daily variance.

Converts player SG projections into per-hole and per-round parameters
for the simulation engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SimPlayer:
    """Player parameters for simulation."""

    name: str
    sg_total: float = 0.0
    sg_ott: float = 0.0
    sg_app: float = 0.0
    sg_atg: float = 0.0
    sg_putt: float = 0.0
    # Variance
    round_std: float = 2.75  # Round-to-round SG standard deviation
    volatility_multiplier: float = 1.0  # >1 = boom/bust, <1 = consistent
    # Context
    world_rank: int = 100
    course_fit_score: float = 50.0
    wave: str = "unknown"  # AM / PM
    # Weather sensitivity
    wind_sg_diff: float = 0.0
    rain_sg_diff: float = 0.0
    # Pressure
    pressure_coefficient: float = 0.0  # Positive = chokes, negative = thrives
    closer_index: float = 0.0  # Positive = thrives under Sunday pressure, negative = chokes

    @property
    def sg_per_hole(self) -> float:
        """Expected SG per hole (18 holes per round)."""
        return self.sg_total / 18.0

    @property
    def expected_round_score(self) -> float:
        """Expected round score relative to par (negative = under par)."""
        # Tour average on a par-72 is roughly 71.5
        # SG = 0 -> scores ~71.5 = -0.5 relative to par
        # Each SG = 1 stroke better per round
        return -self.sg_total - 0.5


class PlayerModel:
    """Convert player projections into simulation-ready parameters."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def generate_daily_form(
        self,
        player: SimPlayer,
        prev_form: float | None = None,
        round_correlation: float = 0.0,
    ) -> float:
        """Generate a random daily form factor for one round.

        Models day-to-day variation in player performance.
        When prev_form is provided, carries over round-to-round correlation.
        Returns SG adjustment for this specific day.
        """
        # Daily form is normally distributed around 0 with player-specific std
        daily_noise = self.rng.normal(0.0, player.round_std)

        # Apply volatility multiplier
        daily_noise *= player.volatility_multiplier

        # Round-to-round correlation (C8): carry over portion of previous form
        if prev_form is not None and round_correlation > 0:
            daily_noise += round_correlation * prev_form

        return float(daily_noise)

    def effective_sg_for_round(
        self,
        player: SimPlayer,
        daily_form: float,
        weather_adjustment: float = 0.0,
        wave_adjustment: float = 0.0,
        pressure_adjustment: float = 0.0,
    ) -> float:
        """Compute effective SG for a round including all adjustments."""
        return (
            player.sg_total +
            daily_form +
            weather_adjustment +
            wave_adjustment -
            pressure_adjustment
        )

    def from_projection(self, projection: dict) -> SimPlayer:
        """Create SimPlayer from a projection dict (from ProjectionEngine)."""
        return SimPlayer(
            name=projection.get("name", "Unknown"),
            sg_total=projection.get("proj_sg_total", 0.0),
            sg_ott=projection.get("proj_sg_ott", 0.0),
            sg_app=projection.get("proj_sg_app", 0.0),
            sg_atg=projection.get("proj_sg_atg", 0.0),
            sg_putt=projection.get("proj_sg_putt", 0.0),
            round_std=projection.get("player_variance", 2.75),
            volatility_multiplier=projection.get("volatility_multiplier", 1.0),
            world_rank=projection.get("world_rank", 100),
            course_fit_score=projection.get("course_fit_score", 50.0),
        )
