"""
Golf Quant Engine — Weather Model
===================================
Models wind, rain, and temperature effects on scoring.
Generates per-round and per-hole weather conditions with
intra-round evolution (morning calm -> afternoon wind).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from simulation.config import SimulationConfig, WeatherCoefficients, COURSE_TYPES


@dataclass
class WeatherConditions:
    """Weather state at a specific point in time."""
    wind_speed_mph: float = 8.0
    wind_direction_deg: float = 180.0   # 0=N, 90=E, 180=S, 270=W
    rain_mm_hr: float = 0.0
    temperature_f: float = 72.0
    humidity_pct: float = 50.0

    @property
    def is_raining(self) -> bool:
        return self.rain_mm_hr > 0.5

    @property
    def rain_category(self) -> str:
        if self.rain_mm_hr < 0.5:
            return "none"
        elif self.rain_mm_hr < 2.0:
            return "light"
        elif self.rain_mm_hr < 5.0:
            return "moderate"
        return "heavy"

    @property
    def wind_category(self) -> str:
        if self.wind_speed_mph < 8:
            return "calm"
        elif self.wind_speed_mph < 15:
            return "moderate"
        elif self.wind_speed_mph < 22:
            return "strong"
        return "gale"

    def to_dict(self) -> dict:
        return {
            "wind_speed_mph": self.wind_speed_mph,
            "wind_direction_deg": self.wind_direction_deg,
            "rain_mm_hr": self.rain_mm_hr,
            "temperature_f": self.temperature_f,
            "humidity_pct": self.humidity_pct,
            "wind_category": self.wind_category,
            "rain_category": self.rain_category,
        }


@dataclass
class RoundWeather:
    """Weather conditions for a full round, with per-hole detail."""
    round_number: int
    base_conditions: WeatherConditions
    hole_conditions: List[WeatherConditions] = field(default_factory=list)

    @property
    def avg_wind(self) -> float:
        if not self.hole_conditions:
            return self.base_conditions.wind_speed_mph
        return np.mean([h.wind_speed_mph for h in self.hole_conditions])

    @property
    def avg_temp(self) -> float:
        if not self.hole_conditions:
            return self.base_conditions.temperature_f
        return np.mean([h.temperature_f for h in self.hole_conditions])

    @property
    def total_rain(self) -> float:
        if not self.hole_conditions:
            return self.base_conditions.rain_mm_hr * 4.5  # ~4.5 hours
        return sum(h.rain_mm_hr * 0.25 for h in self.hole_conditions)  # ~15 min/hole


class WeatherModel:
    """Generate and apply weather effects to golf scoring.

    Weather evolves within a round: front 9 is typically calmer than back 9
    as afternoon winds pick up.  Rain comes in bands.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.wc = self.config.weather_coefficients

    def generate_tournament_weather(
        self,
        rng: np.random.Generator,
        n_rounds: int = 4,
        base_wind: float = 10.0,
        base_temp: float = 72.0,
        rain_probability: float = 0.15,
        base_wind_direction: float = 180.0,
        course_type: str = "parkland",
    ) -> list[RoundWeather]:
        """Generate weather conditions for all rounds of a tournament.

        Parameters
        ----------
        rng : np.random.Generator
            Seeded random generator.
        n_rounds : int
            Number of tournament rounds (typically 4).
        base_wind : float
            Average wind speed in mph for the week.
        base_temp : float
            Average temperature in F for the week.
        rain_probability : float
            Probability of rain in any given round.
        base_wind_direction : float
            Prevailing wind direction (degrees).
        course_type : str
            Course type for wave/weather sensitivity lookup.

        Returns
        -------
        List of RoundWeather, one per round.
        """
        rounds = []
        ct = COURSE_TYPES.get(course_type, COURSE_TYPES["parkland"])
        wind_sens = ct["wind_sensitivity"]

        for r in range(n_rounds):
            # Day-to-day wind variation
            day_wind = base_wind + rng.normal(0, base_wind * 0.25)
            day_wind = max(0.0, day_wind)

            # Day-to-day temperature variation
            day_temp = base_temp + rng.normal(0, 4.0)

            # Wind direction drifts day-to-day
            day_wind_dir = (base_wind_direction + rng.normal(0, 30)) % 360

            # Rain decision
            is_rainy = rng.random() < rain_probability
            rain_intensity = 0.0
            if is_rainy:
                rain_intensity = rng.exponential(2.0)
                rain_intensity = min(rain_intensity, 10.0)

            base_cond = WeatherConditions(
                wind_speed_mph=day_wind,
                wind_direction_deg=day_wind_dir,
                rain_mm_hr=rain_intensity,
                temperature_f=day_temp,
                humidity_pct=50.0 + rng.normal(0, 15),
            )

            # Generate per-hole conditions with intra-round evolution
            hole_conds = self._evolve_within_round(
                rng, base_cond, wind_sens, is_rainy
            )

            rounds.append(RoundWeather(
                round_number=r + 1,
                base_conditions=base_cond,
                hole_conditions=hole_conds,
            ))

        return rounds

    def _evolve_within_round(
        self,
        rng: np.random.Generator,
        base: WeatherConditions,
        wind_sensitivity: float,
        is_rainy: bool,
    ) -> list[WeatherConditions]:
        """Model weather evolution over 18 holes.

        Pattern: morning is calmer, wind picks up through the day.
        Rain comes in bands of 3-6 holes.
        """
        conditions = []

        # Wind ramp: starts at 70% of base, increases to 130%
        wind_ramp = np.linspace(0.70, 1.30, 18)

        # Add random gusts
        gust_noise = rng.normal(0, base.wind_speed_mph * 0.10, size=18)

        # Rain bands (if rainy)
        rain_profile = np.zeros(18)
        if is_rainy:
            # Rain starts at a random hole, lasts 3-8 holes
            rain_start = rng.integers(0, 14)
            rain_duration = rng.integers(3, 9)
            rain_end = min(rain_start + rain_duration, 18)
            # Rain intensity ramps up and down
            for h in range(rain_start, rain_end):
                mid = (rain_start + rain_end) / 2
                dist_from_mid = abs(h - mid) / max(rain_duration / 2, 1)
                rain_profile[h] = base.rain_mm_hr * (1.0 - 0.5 * dist_from_mid)
                rain_profile[h] += rng.normal(0, 0.3)
                rain_profile[h] = max(0.0, rain_profile[h])

        # Temperature: slight warming through morning, cooling if rain
        temp_curve = np.linspace(-2, 3, 18)

        # Wind direction: gradual shift through the round
        dir_drift = np.linspace(0, rng.normal(0, 15), 18)

        for h in range(18):
            wind = base.wind_speed_mph * wind_ramp[h] + gust_noise[h]
            wind = max(0.0, wind)

            temp = base.temperature_f + temp_curve[h]
            if rain_profile[h] > 0:
                temp -= 3.0  # Rain cools things down

            direction = (base.wind_direction_deg + dir_drift[h]) % 360

            conditions.append(WeatherConditions(
                wind_speed_mph=wind,
                wind_direction_deg=direction,
                rain_mm_hr=rain_profile[h],
                temperature_f=temp,
                humidity_pct=base.humidity_pct + (10 if rain_profile[h] > 0 else 0),
            ))

        return conditions

    def calculate_hole_weather_penalty(
        self,
        weather: WeatherConditions,
        hole_wind_exposure: float,
        hole_orientation_deg: float,
        course_type: str = "parkland",
    ) -> float:
        """Calculate the stroke penalty for weather on a specific hole.

        Parameters
        ----------
        weather : WeatherConditions
            Current weather at the hole.
        hole_wind_exposure : float
            0-1 scale of how exposed the hole is to wind.
        hole_orientation_deg : float
            Compass heading of the hole (tee to green).
        course_type : str
            Type of course (affects sensitivity).

        Returns
        -------
        float : Additional strokes of difficulty (0.0 = no effect).
        """
        penalty = 0.0
        ct = COURSE_TYPES.get(course_type, COURSE_TYPES["parkland"])

        # --- Wind penalty ---
        effective_wind = weather.wind_speed_mph * hole_wind_exposure
        if effective_wind > self.wc.wind_threshold_mph:
            excess = effective_wind - self.wc.wind_threshold_mph
            wind_pen = excess * self.wc.wind_stroke_penalty_per_mph

            # Headwind/crosswind is worse than tailwind
            # Calculate angle between wind direction and hole orientation
            angle_diff = abs(weather.wind_direction_deg - hole_orientation_deg) % 360
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            # 0 = pure headwind (worst), 90 = crosswind, 180 = tailwind (least)
            if angle_diff < 60:
                # Near headwind: full penalty + bonus
                wind_pen *= 1.3
            elif angle_diff > 120:
                # Near tailwind: reduced penalty (still affects control)
                wind_pen *= 0.5
            # Crosswind: standard penalty

            wind_pen *= ct["wind_sensitivity"]
            penalty += min(wind_pen, self.wc.wind_max_penalty_per_hole)

        # --- Rain penalty ---
        if weather.rain_mm_hr >= 0.5:
            if weather.rain_mm_hr < 2.0:
                penalty += self.wc.rain_light_penalty * ct["rain_sensitivity"]
            elif weather.rain_mm_hr < 5.0:
                penalty += self.wc.rain_moderate_penalty * ct["rain_sensitivity"]
            else:
                penalty += self.wc.rain_heavy_penalty * ct["rain_sensitivity"]

        # --- Temperature penalty ---
        if weather.temperature_f < self.wc.temp_baseline_f:
            cold_deg = self.wc.temp_baseline_f - weather.temperature_f
            penalty += cold_deg * self.wc.cold_penalty_per_degree * ct["temp_sensitivity"]
        elif weather.temperature_f > 95:
            heat_deg = weather.temperature_f - 95
            penalty += heat_deg * self.wc.heat_penalty_per_degree * ct["temp_sensitivity"]

        return min(penalty, self.wc.max_weather_penalty_per_hole)

    def calculate_round_weather_penalties(
        self,
        round_weather: RoundWeather,
        holes: list,
        course_type: str = "parkland",
    ) -> np.ndarray:
        """Calculate per-hole weather penalties for an entire round.

        Parameters
        ----------
        round_weather : RoundWeather
            Weather for the round.
        holes : list of HoleSpec
            Course holes.
        course_type : str
            Course type.

        Returns
        -------
        np.ndarray of shape (18,) with per-hole penalty values.
        """
        penalties = np.zeros(len(holes))

        for i, hole in enumerate(holes):
            if i < len(round_weather.hole_conditions):
                weather = round_weather.hole_conditions[i]
            else:
                weather = round_weather.base_conditions

            penalties[i] = self.calculate_hole_weather_penalty(
                weather=weather,
                hole_wind_exposure=hole.wind_exposure,
                hole_orientation_deg=hole.hole_orientation_deg,
                course_type=course_type,
            )

        return penalties

    def estimate_round_scoring_impact(
        self,
        round_weather: RoundWeather,
        course_type: str = "parkland",
    ) -> float:
        """Quick estimate of total scoring impact for a round.

        Returns approximate total strokes added to par from weather.
        """
        base = round_weather.base_conditions
        ct = COURSE_TYPES.get(course_type, COURSE_TYPES["parkland"])

        impact = 0.0

        # Wind impact (rough estimate across 18 holes)
        if base.wind_speed_mph > self.wc.wind_threshold_mph:
            excess = base.wind_speed_mph - self.wc.wind_threshold_mph
            impact += excess * self.wc.wind_stroke_penalty_per_mph * 18 * 0.5 * ct["wind_sensitivity"]

        # Rain
        if base.is_raining:
            if base.rain_category == "light":
                impact += self.wc.rain_light_penalty * 18 * 0.5 * ct["rain_sensitivity"]
            elif base.rain_category == "moderate":
                impact += self.wc.rain_moderate_penalty * 18 * 0.5 * ct["rain_sensitivity"]
            else:
                impact += self.wc.rain_heavy_penalty * 18 * 0.5 * ct["rain_sensitivity"]

        # Temperature
        if base.temperature_f < self.wc.temp_baseline_f:
            cold = self.wc.temp_baseline_f - base.temperature_f
            impact += cold * self.wc.cold_penalty_per_degree * 18 * ct["temp_sensitivity"]

        return impact
