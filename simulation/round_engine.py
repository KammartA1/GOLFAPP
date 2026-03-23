"""Round engine — 18-hole round simulation."""

from __future__ import annotations

from typing import List

import numpy as np

from simulation.config import SimulationConfig
from simulation.hole_engine import HoleEngine
from simulation.player_model import SimPlayer, PlayerModel
from simulation.course_model import CourseProfile
from simulation.weather_model import WeatherConditions, WeatherModel
from simulation.pressure_model import PressureModel


class RoundEngine:
    """Simulate a single 18-hole round for a player."""

    def __init__(self, config: SimulationConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.hole_engine = HoleEngine(config, rng)
        self.player_model = PlayerModel(rng)
        self.weather_model = WeatherModel()
        self.pressure_model = PressureModel(config)

    def simulate_round(
        self,
        player: SimPlayer,
        course: CourseProfile,
        round_num: int = 1,
        weather: WeatherConditions | None = None,
        current_position: int = 75,
        strokes_off_lead: float = 0.0,
        field_size: int = 156,
    ) -> dict:
        """Simulate one 18-hole round for a player.

        Returns:
            {
                "total_score": int (relative to par),
                "gross_score": int (actual strokes),
                "hole_scores": list of int (relative to par per hole),
                "sg_this_round": float,
                "birdies": int,
                "bogeys": int,
                "eagles": int,
                "doubles_plus": int,
            }
        """
        # Generate daily form
        daily_form = self.player_model.generate_daily_form(player)

        # Weather adjustment
        weather_adj = 0.0
        player_weather_adj = 0.0
        if weather:
            weather_adj = self.weather_model.round_scoring_adjustment(weather)
            player_weather_adj = self.weather_model.player_weather_adjustment(player, weather)

        # Effective SG for this round
        pressure_adj = self.pressure_model.pressure_adjustment(
            player, round_num, current_position, field_size, strokes_off_lead
        )

        effective_sg = self.player_model.effective_sg_for_round(
            player, daily_form, player_weather_adj, 0.0, pressure_adj
        )

        sg_per_hole = effective_sg / 18.0

        # Simulate each hole
        hole_scores = []
        momentum = 0.0
        birdies = 0
        bogeys = 0
        eagles = 0
        doubles_plus = 0

        for hole in course.holes:
            # Additional pressure on Sunday back nine
            extra_pressure = 0.0
            if round_num == 4:
                extra_pressure = self.pressure_model.sunday_back_nine_pressure(
                    player, hole.number, current_position, strokes_off_lead
                )

            # Per-hole weather
            hole_weather = 0.0
            if weather:
                hole_weather = self.weather_model.per_hole_adjustment(
                    weather, hole.wind_exposed
                )

            score = self.hole_engine.simulate_hole(
                hole_par=hole.par,
                hole_difficulty=hole.difficulty + hole_weather,
                hole_distance=hole.distance,
                player_sg_per_hole=sg_per_hole,
                player_volatility=player.volatility_multiplier,
                weather_adjustment=0.0,  # Already in sg_per_hole
                pressure_adjustment=extra_pressure,
                momentum=momentum,
            )

            hole_scores.append(score)
            momentum = float(score)  # Last hole result influences next

            # Count outcomes
            if score <= -2:
                eagles += 1
            elif score == -1:
                birdies += 1
            elif score == 1:
                bogeys += 1
            elif score >= 2:
                doubles_plus += 1

        total_to_par = sum(hole_scores)
        gross_score = course.total_par + total_to_par

        return {
            "total_score": total_to_par,
            "gross_score": gross_score,
            "hole_scores": hole_scores,
            "sg_this_round": round(effective_sg, 3),
            "daily_form": round(daily_form, 3),
            "birdies": birdies,
            "bogeys": bogeys,
            "eagles": eagles,
            "doubles_plus": doubles_plus,
            "pars": 18 - birdies - bogeys - eagles - doubles_plus,
            "weather_adjustment": round(weather_adj, 3),
            "pressure_adjustment": round(pressure_adj, 3),
        }
