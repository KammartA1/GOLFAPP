"""Pressure model — Leaderboard pressure effects, player-specific coefficient."""

from __future__ import annotations

import numpy as np

from simulation.player_model import SimPlayer
from simulation.config import SimulationConfig


class PressureModel:
    """Model leaderboard pressure effects on player performance."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def pressure_adjustment(
        self,
        player: SimPlayer,
        round_num: int,
        current_position: int,
        field_size: int,
        strokes_off_lead: float,
    ) -> float:
        """Calculate pressure-induced SG adjustment.

        Args:
            player: Player with pressure coefficient.
            round_num: 1-4.
            current_position: Current leaderboard position (1 = leader).
            field_size: Total field size.
            strokes_off_lead: Strokes behind the leader.

        Returns:
            SG adjustment (positive = pressure hurts, negative = pressure helps).
        """
        if not self.config.pressure_enabled:
            return 0.0

        # Pressure intensity increases in later rounds
        round_pressure = {1: 0.1, 2: 0.2, 3: 0.5, 4: 1.0}.get(round_num, 0.0)

        # Position pressure: higher for contenders
        if current_position <= 5:
            position_pressure = 1.0
        elif current_position <= 15:
            position_pressure = 0.6
        elif current_position <= 30:
            position_pressure = 0.3
        else:
            position_pressure = 0.05

        # Close to the lead increases pressure
        if strokes_off_lead <= 2:
            closeness_factor = 1.0
        elif strokes_off_lead <= 5:
            closeness_factor = 0.5
        else:
            closeness_factor = 0.1

        # Total pressure intensity
        pressure_intensity = round_pressure * position_pressure * closeness_factor

        # Player-specific response
        # pressure_coefficient: positive = chokes, negative = thrives
        adjustment = player.pressure_coefficient * pressure_intensity

        # Closer index (R9): in round 4, players in contention get bonus/penalty
        if (self.config.closer_index_enabled
                and round_num == 4
                and player.closer_index != 0.0
                and current_position <= 10):
            # closer_index > 0 means player thrives, so subtract from adjustment (helps)
            closer_bonus = player.closer_index * round_pressure * position_pressure * closeness_factor
            adjustment -= closer_bonus

        # Clamp to reasonable range
        max_effect = self.config.max_pressure_effect_sg
        return float(np.clip(adjustment, -max_effect, max_effect))

    def sunday_back_nine_pressure(
        self,
        player: SimPlayer,
        hole_number: int,
        current_position: int,
        strokes_off_lead: float,
    ) -> float:
        """Extra pressure on the back nine on Sunday (round 4).

        The back nine on Sunday is where tournaments are won and lost.
        Pressure peaks on holes 15-18 when in contention.
        """
        if hole_number < 10:
            return 0.0

        # Increasing pressure toward hole 18
        hole_factor = (hole_number - 9) / 9.0  # 0.11 to 1.0

        # Only intense for contenders
        if current_position > 10 or strokes_off_lead > 4:
            return 0.0

        intensity = hole_factor * (1.0 - strokes_off_lead / 5.0)
        adjustment = player.pressure_coefficient * intensity * 0.5

        # Closer index: thrives under Sunday back nine pressure
        if self.config.closer_index_enabled and player.closer_index != 0.0:
            adjustment -= player.closer_index * intensity * 0.5

        max_effect = self.config.max_pressure_effect_sg / 18.0  # Per hole
        return float(np.clip(adjustment, -max_effect, max_effect))
