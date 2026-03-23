"""Weather model — Wind, rain, temperature effects on scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.player_model import SimPlayer


@dataclass
class WeatherConditions:
    """Weather conditions for a round."""

    wind_speed: float = 10.0  # mph
    wind_gust: float = 15.0  # mph
    precip_chance: float = 0.0  # 0-1
    precip_amount: float = 0.0  # inches
    temperature: float = 75.0  # Fahrenheit
    humidity: float = 50.0  # percent
    conditions: str = "sunny"  # sunny/cloudy/rain/storm


class WeatherModel:
    """Model weather effects on scoring and individual players."""

    def round_scoring_adjustment(self, weather: WeatherConditions) -> float:
        """Calculate field-wide scoring adjustment from weather.

        Returns positive value = harder scoring conditions (strokes above normal).
        """
        adjustment = 0.0

        # Wind effect: primary difficulty driver
        if weather.wind_speed > 20:
            adjustment += (weather.wind_speed - 10) * 0.06
        elif weather.wind_speed > 12:
            adjustment += (weather.wind_speed - 10) * 0.04

        # Rain effect
        if weather.precip_chance > 0.5:
            adjustment += weather.precip_chance * 0.8
            if weather.precip_amount > 0.5:
                adjustment += 0.3

        # Temperature extremes
        if weather.temperature < 50:
            adjustment += (50 - weather.temperature) * 0.02
        elif weather.temperature > 95:
            adjustment += (weather.temperature - 90) * 0.015

        # Gusty conditions add more difficulty than steady wind
        gust_factor = weather.wind_gust - weather.wind_speed
        if gust_factor > 10:
            adjustment += 0.15

        return round(adjustment, 3)

    def player_weather_adjustment(
        self,
        player: SimPlayer,
        weather: WeatherConditions,
    ) -> float:
        """Calculate player-specific weather adjustment.

        Returns SG adjustment (positive = player benefits from weather).
        """
        adjustment = 0.0

        # Wind skill differential
        if weather.wind_speed > 15:
            wind_factor = min((weather.wind_speed - 15) / 10.0, 1.0)
            adjustment += player.wind_sg_diff * wind_factor

        # Rain skill differential
        if weather.precip_chance > 0.4:
            rain_factor = min((weather.precip_chance - 0.4) / 0.4, 1.0)
            adjustment += player.rain_sg_diff * rain_factor

        return round(adjustment, 3)

    def per_hole_adjustment(
        self,
        weather: WeatherConditions,
        hole_wind_exposed: bool = False,
    ) -> float:
        """Per-hole weather adjustment.

        Wind-exposed holes get extra difficulty in windy conditions.
        """
        base = self.round_scoring_adjustment(weather) / 18.0

        if hole_wind_exposed and weather.wind_speed > 15:
            base *= 1.5

        return base
