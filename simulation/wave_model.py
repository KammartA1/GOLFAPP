"""Wave model — AM/PM wave conditions and advantage."""

from __future__ import annotations

from dataclasses import dataclass

from simulation.weather_model import WeatherConditions, WeatherModel
from simulation.player_model import SimPlayer


@dataclass
class WaveConditions:
    """Weather conditions split by AM/PM wave."""

    am_weather: WeatherConditions
    pm_weather: WeatherConditions


class WaveModel:
    """Model AM/PM wave advantage from differential weather conditions."""

    def __init__(self):
        self.weather_model = WeatherModel()

    def wave_advantage(self, wave_conditions: WaveConditions, player_wave: str) -> float:
        """Calculate the wave advantage in SG for a player's wave.

        Args:
            wave_conditions: AM and PM weather conditions.
            player_wave: "AM" or "PM"

        Returns:
            SG advantage (positive = player's wave is easier).
        """
        am_difficulty = self.weather_model.round_scoring_adjustment(wave_conditions.am_weather)
        pm_difficulty = self.weather_model.round_scoring_adjustment(wave_conditions.pm_weather)

        if player_wave == "AM":
            # Player is in AM wave: advantage = PM_difficulty - AM_difficulty
            return round(pm_difficulty - am_difficulty, 3)
        elif player_wave == "PM":
            return round(am_difficulty - pm_difficulty, 3)
        return 0.0

    def wave_scoring_differential(self, wave_conditions: WaveConditions) -> float:
        """Expected scoring differential between waves.

        Returns positive value = PM wave scores worse.
        """
        am_diff = self.weather_model.round_scoring_adjustment(wave_conditions.am_weather)
        pm_diff = self.weather_model.round_scoring_adjustment(wave_conditions.pm_weather)
        return round(pm_diff - am_diff, 3)

    def round2_wave_flip(self, round1_wave: str) -> str:
        """In PGA Tour events, waves flip in round 2."""
        if round1_wave == "AM":
            return "PM"
        elif round1_wave == "PM":
            return "AM"
        return "unknown"

    def net_36_hole_advantage(
        self,
        wave_conditions_r1: WaveConditions,
        wave_conditions_r2: WaveConditions,
        player_r1_wave: str,
    ) -> float:
        """Net 36-hole (2-round) wave advantage accounting for wave flip.

        Round 1 AM -> Round 2 PM.
        """
        r1_advantage = self.wave_advantage(wave_conditions_r1, player_r1_wave)
        r2_wave = self.round2_wave_flip(player_r1_wave)
        r2_advantage = self.wave_advantage(wave_conditions_r2, r2_wave)
        return round(r1_advantage + r2_advantage, 3)
