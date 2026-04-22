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
    wind_direction: float = 0.0  # degrees, 0=N
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

        # Wind effect: primary difficulty driver (continuous function)
        if weather.wind_speed > 12:
            base_wind = (weather.wind_speed - 10) * 0.04
            if weather.wind_speed > 20:
                base_wind += (weather.wind_speed - 20) * 0.02  # Extra difficulty above 20
            adjustment += base_wind

        # Rain effect
        if weather.precip_chance > 0.5:
            adjustment += weather.precip_chance * 0.8
            if weather.precip_amount > 0.5:
                adjustment += 0.3

        # Temperature extremes
        if weather.temperature < 50:
            adjustment += (50 - weather.temperature) * 0.02
        elif weather.temperature > 90:
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

    def per_hole_wind_direction_adjustment(
        self,
        weather: WeatherConditions,
        hole_orientation: float,
        wind_direction: float,
    ) -> float:
        """Compute hole-specific wind impact based on orientation (R7).

        Args:
            weather: Current weather conditions.
            hole_orientation: Hole tee-to-green orientation in degrees (0=N).
            wind_direction: Wind direction in degrees (0=N, direction wind comes FROM).

        Returns:
            SG adjustment (positive = harder due to headwind).
        """
        if weather.wind_speed < 8:
            return 0.0

        # Angle between wind direction and hole orientation
        # Headwind: wind coming from the direction the hole faces
        angle_diff = abs(wind_direction - hole_orientation) % 360
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # cos(0) = 1.0 (pure headwind), cos(90) = 0 (crosswind), cos(180) = -1 (tailwind)
        wind_alignment = np.cos(np.radians(angle_diff))

        # Scale by wind speed above baseline
        wind_factor = (weather.wind_speed - 8) * 0.005

        # Headwind hurts (positive adj), tailwind helps (negative adj)
        # Crosswind has slight penalty due to difficulty
        crosswind_component = abs(np.sin(np.radians(angle_diff)))
        crosswind_penalty = crosswind_component * wind_factor * 0.3

        return round(wind_alignment * wind_factor + crosswind_penalty, 4)
