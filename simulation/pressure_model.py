"""
Golf Quant Engine — Pressure Model
====================================
Models leaderboard pressure effects on player performance.
Players near the lead in rounds 3-4 experience amplified variance
and mean shifts based on their pressure coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from simulation.config import SimulationConfig, PressureCoefficients
from simulation.player_model import PlayerModel, PlayerSGComponents


@dataclass
class PressureState:
    """Current pressure context for a player in a round."""
    round_number: int          # 1-4
    shots_off_lead: float      # Current strokes behind leader
    is_leader: bool            # Currently leading or co-leading
    in_contention: bool        # Within contention_threshold of lead
    is_back_nine: bool         # Currently on back 9 (holes 10-18)
    current_hole: int          # Current hole number (1-18)

    @property
    def is_sunday(self) -> bool:
        return self.round_number == 4

    @property
    def is_sunday_back9(self) -> bool:
        return self.is_sunday and self.is_back_nine

    @property
    def pressure_description(self) -> str:
        if not self.in_contention:
            return "no_pressure"
        if self.is_sunday_back9 and self.is_leader:
            return "sunday_back9_leader"
        if self.is_sunday_back9:
            return "sunday_back9_contender"
        if self.is_leader:
            return "leader"
        return "contender"


class PressureModel:
    """Apply pressure-based adjustments to player SG components.

    Core mechanics:
    1. Pressure only activates when a player is in contention (within N shots)
    2. Pressure scales with round number (R1=none, R4=full)
    3. Sunday back 9 amplifies pressure by 1.5x
    4. Player's pressure_coeff determines if they thrive or fold:
       - Positive = closer (SG shifts up under pressure)
       - Negative = choker (SG shifts down under pressure)
    5. Everyone's variance increases under pressure
    6. First-time contenders face additional penalty
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.pc = self.config.pressure_coefficients

    def calculate_pressure_state(
        self,
        round_number: int,
        player_score: float,
        leader_score: float,
        current_hole: int = 1,
    ) -> PressureState:
        """Determine the pressure context for a player.

        Parameters
        ----------
        round_number : int
            Current round (1-4).
        player_score : float
            Player's cumulative score relative to par.
        leader_score : float
            Leader's cumulative score relative to par.
        current_hole : int
            Current hole (1-18).
        """
        shots_off = player_score - leader_score  # Positive = behind
        in_contention = shots_off <= self.pc.contention_threshold
        is_leader = shots_off <= 0

        return PressureState(
            round_number=round_number,
            shots_off_lead=shots_off,
            is_leader=is_leader,
            in_contention=in_contention,
            is_back_nine=current_hole > 9,
            current_hole=current_hole,
        )

    def apply_pressure(
        self,
        sg_components: PlayerSGComponents,
        player: PlayerModel,
        pressure_state: PressureState,
        rng: np.random.Generator,
    ) -> PlayerSGComponents:
        """Apply pressure adjustments to a player's SG draw.

        Returns a new PlayerSGComponents with pressure effects applied.

        The modification is:
        1. Mean shift = pressure_coeff * pressure_intensity * sg_mean_shift
        2. Variance multiplier = base_multiplier * pressure_intensity
        3. First-time contender penalty applied if applicable
        """
        if not pressure_state.in_contention:
            return sg_components

        # Calculate pressure intensity
        intensity = self._calculate_intensity(pressure_state)

        if intensity <= 0:
            return sg_components

        # Mean shift: based on player's pressure coefficient
        mean_shift = (
            player.pressure_coeff
            * intensity
            * self.pc.pressure_sg_mean_shift
        )

        # First-time contender penalty
        if player.is_first_time_contender and player.career_wins < 2:
            mean_shift += self.pc.first_time_contender_penalty * intensity

        # Variance increase under pressure
        var_mult = 1.0 + (self.pc.pressure_variance_multiplier - 1.0) * intensity

        # Apply: shift all components equally, add noise from increased variance
        extra_noise_std = 0.10 * intensity * var_mult

        new_ott = sg_components.sg_ott + mean_shift * 0.25 + rng.normal(0, extra_noise_std)
        new_app = sg_components.sg_app + mean_shift * 0.35 + rng.normal(0, extra_noise_std)
        new_atg = sg_components.sg_atg + mean_shift * 0.20 + rng.normal(0, extra_noise_std)
        new_putt = sg_components.sg_putt + mean_shift * 0.20 + rng.normal(0, extra_noise_std * 1.3)

        return PlayerSGComponents(
            sg_ott=new_ott,
            sg_app=new_app,
            sg_atg=new_atg,
            sg_putt=new_putt,
        )

    def _calculate_intensity(self, pressure_state: PressureState) -> float:
        """Calculate pressure intensity on 0-1+ scale.

        Combines round-based scaling, Sunday back 9 amplifier,
        and proximity to the lead.
        """
        # Round-based scaling
        round_idx = pressure_state.round_number - 1
        if round_idx < len(self.pc.round_pressure_scale):
            round_factor = self.pc.round_pressure_scale[round_idx]
        else:
            round_factor = 1.0

        # Sunday back 9 amplifier
        amplifier = 1.0
        if pressure_state.is_sunday_back9:
            amplifier = self.pc.sunday_back9_amplifier
        elif pressure_state.is_back_nine and pressure_state.round_number >= 3:
            amplifier = 1.2

        # Proximity to lead: closer = more pressure
        if pressure_state.is_leader:
            proximity = 1.0
        else:
            # Linear decay: at contention_threshold shots back, proximity = 0.3
            proximity = max(
                0.3,
                1.0 - (pressure_state.shots_off_lead / self.pc.contention_threshold) * 0.7,
            )

        return round_factor * amplifier * proximity

    def get_pressure_adjustment_preview(
        self,
        player: PlayerModel,
        round_number: int,
        shots_off_lead: float,
    ) -> dict:
        """Preview the pressure adjustment without applying it.

        Useful for analysis and UI display.
        """
        ps = PressureState(
            round_number=round_number,
            shots_off_lead=shots_off_lead,
            is_leader=shots_off_lead <= 0,
            in_contention=shots_off_lead <= self.pc.contention_threshold,
            is_back_nine=False,
            current_hole=10,
        )

        intensity = self._calculate_intensity(ps) if ps.in_contention else 0.0

        mean_shift = player.pressure_coeff * intensity * self.pc.pressure_sg_mean_shift
        if player.is_first_time_contender and player.career_wins < 2:
            mean_shift += self.pc.first_time_contender_penalty * intensity

        return {
            "player": player.name,
            "pressure_coeff": player.pressure_coeff,
            "in_contention": ps.in_contention,
            "intensity": round(intensity, 3),
            "mean_shift_sg": round(mean_shift, 4),
            "description": ps.pressure_description,
            "is_first_time_contender": player.is_first_time_contender,
        }

    def classify_player_pressure_type(self, player: PlayerModel) -> str:
        """Classify a player's pressure response for UI display."""
        coeff = player.pressure_coeff
        if coeff >= 0.5:
            return "Elite Closer"
        elif coeff >= 0.2:
            return "Closer"
        elif coeff >= -0.1:
            return "Neutral"
        elif coeff >= -0.4:
            return "Wilts Under Pressure"
        return "Severe Choker"
