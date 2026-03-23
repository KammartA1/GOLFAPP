"""
Golf Quant Engine — Round Engine
==================================
Simulates a complete 18-hole round for a player, tracking cumulative
score, hole-by-hole outcomes, momentum, weather evolution, and fatigue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from simulation.config import SimulationConfig
from simulation.course_model import CourseModel, HoleSpec
from simulation.hole_engine import HoleEngine
from simulation.player_model import PlayerModel, PlayerSGComponents
from simulation.pressure_model import PressureModel, PressureState
from simulation.weather_model import RoundWeather, WeatherConditions, WeatherModel


@dataclass
class RoundResult:
    """Complete result of simulating one round for one player."""
    player_name: str
    round_number: int
    total_score: int                  # Total strokes relative to par
    hole_scores: List[int]            # Per-hole scores relative to par
    front_nine_score: int             # Front 9 total relative to par
    back_nine_score: int              # Back 9 total relative to par
    sg_components: PlayerSGComponents  # SG draw used for this round
    birdies: int
    pars: int
    bogeys: int
    eagles: int
    doubles_plus: int
    wave_adjustment: float = 0.0
    pressure_applied: bool = False

    @property
    def gross_score(self) -> int:
        """Approximate gross score (assuming par 72)."""
        return 72 + self.total_score

    def to_dict(self) -> dict:
        return {
            "player_name": self.player_name,
            "round_number": self.round_number,
            "total_score": self.total_score,
            "front_nine": self.front_nine_score,
            "back_nine": self.back_nine_score,
            "birdies": self.birdies,
            "pars": self.pars,
            "bogeys": self.bogeys,
            "eagles": self.eagles,
            "doubles_plus": self.doubles_plus,
            "sg_total": self.sg_components.sg_total,
        }


class RoundEngine:
    """Simulate a complete 18-hole round.

    Orchestrates the hole engine across all 18 holes, applying:
    - Weather changes through the round
    - Intra-round momentum (hot/cold streaks)
    - Pressure adjustments based on leaderboard
    - Fatigue on long courses in extreme conditions
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.hole_engine = HoleEngine(self.config)
        self.pressure_model = PressureModel(self.config)
        self.weather_model = WeatherModel(self.config)

    def simulate_round(
        self,
        player: PlayerModel,
        course: CourseModel,
        rng: np.random.Generator,
        round_number: int = 1,
        round_weather: Optional[RoundWeather] = None,
        cumulative_score: float = 0.0,
        leader_score: float = 0.0,
        wave_adjustment: float = 0.0,
    ) -> RoundResult:
        """Simulate a full 18-hole round.

        Parameters
        ----------
        player : PlayerModel
            Player skill model.
        course : CourseModel
            Course specification.
        rng : np.random.Generator
            Seeded RNG.
        round_number : int
            Round number (1-4).
        round_weather : RoundWeather, optional
            Weather conditions for the round.
        cumulative_score : float
            Player's cumulative score entering this round (for pressure calc).
        leader_score : float
            Leader's cumulative score (for pressure calc).
        wave_adjustment : float
            Wave-based scoring adjustment (from WaveModel).

        Returns
        -------
        RoundResult with full round details.
        """
        # Draw SG components for this round
        sg_draw = player.sample_round_sg_components(
            rng=rng,
            surface_bermuda=course.bermuda_greens,
        )

        # Calculate weather penalties per hole
        if round_weather is not None:
            weather_penalties = self.weather_model.calculate_round_weather_penalties(
                round_weather=round_weather,
                holes=course.holes,
                course_type=course.course_type,
            )
        else:
            weather_penalties = np.zeros(len(course.holes))

        # Simulate each hole
        hole_scores: list[int] = []
        momentum = 0.0
        running_score = cumulative_score

        for i, hole in enumerate(course.holes):
            current_hole = i + 1

            # Pressure check (primarily for rounds 3-4)
            pressure_state = self.pressure_model.calculate_pressure_state(
                round_number=round_number,
                player_score=running_score,
                leader_score=leader_score,
                current_hole=current_hole,
            )

            # Apply pressure to SG if in contention
            if pressure_state.in_contention and round_number >= 3:
                effective_sg = self.pressure_model.apply_pressure(
                    sg_components=sg_draw,
                    player=player,
                    pressure_state=pressure_state,
                    rng=rng,
                )
                pressure_applied = True
            else:
                effective_sg = sg_draw
                pressure_applied = False

            # Fatigue adjustment
            fatigue = self._calculate_fatigue(
                hole_number=current_hole,
                temperature=self._get_hole_temp(round_weather, i),
            )

            # Simulate the hole
            score = self.hole_engine.simulate_hole(
                player_sg=effective_sg,
                hole=hole,
                rng=rng,
                weather_penalty=weather_penalties[i] + fatigue,
                momentum=momentum,
            )

            hole_scores.append(score)
            running_score += score

            # Update momentum (exponential decay with latest hole)
            momentum = (
                momentum * (1 - self.config.intra_round_momentum)
                + (-score) * self.config.intra_round_momentum
            )
            # Clamp momentum
            momentum = np.clip(momentum, -0.15, 0.15)

        # Compile results
        total = sum(hole_scores)
        front_nine = sum(hole_scores[:9])
        back_nine = sum(hole_scores[9:])

        return RoundResult(
            player_name=player.name,
            round_number=round_number,
            total_score=total,
            hole_scores=hole_scores,
            front_nine_score=front_nine,
            back_nine_score=back_nine,
            sg_components=sg_draw,
            birdies=sum(1 for s in hole_scores if s == -1),
            pars=sum(1 for s in hole_scores if s == 0),
            bogeys=sum(1 for s in hole_scores if s == 1),
            eagles=sum(1 for s in hole_scores if s <= -2),
            doubles_plus=sum(1 for s in hole_scores if s >= 2),
            wave_adjustment=wave_adjustment,
            pressure_applied=pressure_applied,
        )

    def simulate_round_fast(
        self,
        player: PlayerModel,
        course: CourseModel,
        rng: np.random.Generator,
        round_number: int = 1,
        weather_penalties: Optional[np.ndarray] = None,
        cumulative_score: float = 0.0,
        leader_score: float = 0.0,
    ) -> int:
        """Fast round simulation returning only total score.

        Skips detailed tracking for speed in bulk simulations.
        """
        sg_draw = player.sample_round_sg_components(
            rng=rng,
            surface_bermuda=course.bermuda_greens,
        )

        if weather_penalties is None:
            weather_penalties = np.zeros(len(course.holes))

        total = 0
        momentum = 0.0

        for i, hole in enumerate(course.holes):
            score = self.hole_engine.simulate_hole(
                player_sg=sg_draw,
                hole=hole,
                rng=rng,
                weather_penalty=weather_penalties[i],
                momentum=momentum,
            )
            total += score
            momentum = (
                momentum * (1 - self.config.intra_round_momentum)
                + (-score) * self.config.intra_round_momentum
            )
            momentum = np.clip(momentum, -0.15, 0.15)

        return total

    def _calculate_fatigue(
        self,
        hole_number: int,
        temperature: float = 72.0,
    ) -> float:
        """Calculate fatigue penalty for a given hole.

        Golf fatigue is minimal but non-zero, especially:
        - In extreme heat (>95F)
        - On hilly courses
        - Late in the round (holes 15-18)
        """
        base_fatigue = hole_number * self.config.fatigue_per_hole

        # Heat fatigue amplifier
        if temperature > self.config.heat_fatigue_threshold_f:
            heat_excess = temperature - self.config.heat_fatigue_threshold_f
            base_fatigue *= 1.0 + heat_excess * 0.05 * self.config.heat_fatigue_multiplier

        return base_fatigue

    def _get_hole_temp(
        self,
        round_weather: Optional[RoundWeather],
        hole_index: int,
    ) -> float:
        """Get temperature for a specific hole from weather data."""
        if round_weather is None:
            return 72.0
        if hole_index < len(round_weather.hole_conditions):
            return round_weather.hole_conditions[hole_index].temperature_f
        return round_weather.base_conditions.temperature_f
