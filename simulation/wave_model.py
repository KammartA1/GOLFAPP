"""
Golf Quant Engine — Wave Model
================================
AM/PM wave advantage modeling.  In PGA Tour events, half the field
tees off in the morning and half in the afternoon.  Weather differences
(especially wind) create structural scoring advantages of 1-3 strokes
at exposed courses.  This is a key betting edge in round-level markets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from simulation.config import SimulationConfig, WaveParameters, COURSE_TYPES
from simulation.weather_model import RoundWeather, WeatherConditions


@dataclass
class WaveAssignment:
    """Tracks which wave a player is in for each round."""
    player_name: str
    # Round number -> wave ("AM" or "PM")
    round_waves: Dict[int, str] = field(default_factory=dict)

    def get_wave(self, round_number: int) -> str:
        return self.round_waves.get(round_number, "AM")


@dataclass
class WaveAdvantage:
    """Quantified scoring advantage for each wave in a round."""
    round_number: int
    am_scoring_adjustment: float  # Positive = easier, negative = harder
    pm_scoring_adjustment: float
    advantage_wave: str           # "AM" or "PM" — which wave has the edge
    advantage_strokes: float      # Magnitude of the advantage

    @property
    def edge_description(self) -> str:
        return (
            f"R{self.round_number}: {self.advantage_wave} wave has "
            f"{self.advantage_strokes:+.1f} stroke advantage"
        )


class WaveModel:
    """Model AM/PM wave advantages based on weather evolution.

    Key insight: afternoon wind typically increases, creating 1-3 stroke
    advantages at links/coastal courses.  At desert courses, afternoon
    heat can create a similar but smaller effect.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.wp = self.config.wave_parameters

    def assign_waves(
        self,
        player_names: list[str],
        n_rounds: int = 4,
        rng: Optional[np.random.Generator] = None,
    ) -> list[WaveAssignment]:
        """Assign players to AM/PM waves for each round.

        PGA Tour convention: waves alternate.  If you're AM in R1, you're PM
        in R2, and typically the cut happens after R2 so R3/R4 are single-wave
        (everyone plays at similar times by tee-time groupings based on score).

        Parameters
        ----------
        player_names : list of str
            Names of all players in the field.
        n_rounds : int
            Number of rounds.
        rng : np.random.Generator, optional
            Random generator for initial wave assignment.

        Returns
        -------
        list of WaveAssignment, one per player.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        n_players = len(player_names)
        # Shuffle for initial wave assignment
        indices = np.arange(n_players)
        rng.shuffle(indices)

        half = n_players // 2
        am_r1 = set(indices[:half])
        pm_r1 = set(indices[half:])

        assignments = []
        for i, name in enumerate(player_names):
            waves = {}
            if i in am_r1:
                waves[1] = "AM"
                waves[2] = "PM"  # Alternate in R2
            else:
                waves[1] = "PM"
                waves[2] = "AM"  # Alternate in R2
            # R3 and R4: no wave effect (grouped by score, tee times spread)
            waves[3] = "NONE"
            waves[4] = "NONE"

            assignments.append(WaveAssignment(
                player_name=name,
                round_waves=waves,
            ))

        return assignments

    def calculate_wave_advantages(
        self,
        round_weathers: list[RoundWeather],
        course_type: str = "parkland",
    ) -> list[WaveAdvantage]:
        """Calculate the AM/PM wave advantage for each round.

        Logic: compare average weather conditions for front 9 (holes 1-9,
        representing AM tee times) vs back 9 (holes 10-18, representing PM
        start — PM wave starts when AM wave is on back 9).

        In practice, the PM wave faces afternoon conditions on the
        front 9 and late afternoon on the back 9.

        Parameters
        ----------
        round_weathers : list of RoundWeather
            Weather for each round.
        course_type : str
            Course type for sensitivity multipliers.

        Returns
        -------
        list of WaveAdvantage, one per round.
        """
        ct = COURSE_TYPES.get(course_type, COURSE_TYPES["parkland"])
        wave_mult = ct.get("wave_multiplier", 1.0)

        advantages = []

        for rw in round_weathers:
            if not rw.hole_conditions or rw.round_number > 2:
                # R3/R4 have no wave effect
                advantages.append(WaveAdvantage(
                    round_number=rw.round_number,
                    am_scoring_adjustment=0.0,
                    pm_scoring_adjustment=0.0,
                    advantage_wave="NONE",
                    advantage_strokes=0.0,
                ))
                continue

            # AM wave conditions: holes 1-9 in the morning
            am_winds = [h.wind_speed_mph for h in rw.hole_conditions[:9]]
            am_rain = [h.rain_mm_hr for h in rw.hole_conditions[:9]]
            am_temp = [h.temperature_f for h in rw.hole_conditions[:9]]

            # PM wave conditions: holes 1-9 in the afternoon (conditions of back 9)
            # The PM wave plays front 9 when AM wave is on back 9
            pm_winds = [h.wind_speed_mph for h in rw.hole_conditions[9:]]
            pm_rain = [h.rain_mm_hr for h in rw.hole_conditions[9:]]
            pm_temp = [h.temperature_f for h in rw.hole_conditions[9:]]

            # Average conditions per wave
            am_avg_wind = np.mean(am_winds)
            pm_avg_wind = np.mean(pm_winds)
            am_avg_rain = np.mean(am_rain)
            pm_avg_rain = np.mean(pm_rain)

            # Wind scoring differential
            wind_diff = (pm_avg_wind - am_avg_wind)
            wind_impact = wind_diff * 0.035 * 9 * wave_mult  # Per 9 holes

            # Rain differential
            rain_diff = (pm_avg_rain - am_avg_rain)
            rain_impact = rain_diff * 0.05 * 9 * wave_mult

            # Total advantage (positive = AM is better)
            total_advantage = wind_impact + rain_impact

            # Clamp to reasonable range
            total_advantage = np.clip(
                total_advantage,
                -self.wp.max_wave_advantage,
                self.wp.max_wave_advantage,
            )

            # AM adjustment: if total_advantage > 0, AM has it easier
            # We split the advantage: AM gets a bonus, PM gets a penalty
            am_adj = -total_advantage / 2  # Negative = easier (fewer strokes)
            pm_adj = total_advantage / 2   # Positive = harder (more strokes)

            adv_wave = "AM" if total_advantage > 0 else "PM"
            adv_mag = abs(total_advantage)

            advantages.append(WaveAdvantage(
                round_number=rw.round_number,
                am_scoring_adjustment=am_adj,
                pm_scoring_adjustment=pm_adj,
                advantage_wave=adv_wave if adv_mag > 0.1 else "NONE",
                advantage_strokes=adv_mag,
            ))

        return advantages

    def get_player_wave_adjustment(
        self,
        player_wave: str,
        wave_advantage: WaveAdvantage,
    ) -> float:
        """Get the scoring adjustment for a player based on their wave.

        Returns stroke adjustment to add to the player's round score.
        Positive = harder conditions, negative = easier.
        """
        if player_wave == "AM":
            return wave_advantage.am_scoring_adjustment
        elif player_wave == "PM":
            return wave_advantage.pm_scoring_adjustment
        return 0.0  # NONE wave (R3/R4)

    def estimate_wave_edge(
        self,
        wave_advantages: list[WaveAdvantage],
        player_wave: WaveAssignment,
    ) -> dict[int, float]:
        """Estimate the wave-based scoring edge for a player across all rounds.

        Returns dict of {round_number: scoring_edge}.
        Positive edge = favorable wave, negative = unfavorable.
        """
        edges = {}
        for adv in wave_advantages:
            rnd = adv.round_number
            wave = player_wave.get_wave(rnd)
            adj = self.get_player_wave_adjustment(wave, adv)
            # Negate: a positive adjustment means harder, so the edge is negative
            edges[rnd] = -adj
        return edges

    def summarize_wave_impact(
        self,
        wave_advantages: list[WaveAdvantage],
    ) -> dict:
        """Summary statistics of wave effects for the tournament."""
        total_am = sum(a.am_scoring_adjustment for a in wave_advantages)
        total_pm = sum(a.pm_scoring_adjustment for a in wave_advantages)
        max_adv = max(a.advantage_strokes for a in wave_advantages) if wave_advantages else 0

        return {
            "total_am_adjustment": total_am,
            "total_pm_adjustment": total_pm,
            "max_single_round_advantage": max_adv,
            "rounds_with_advantage": sum(
                1 for a in wave_advantages if a.advantage_strokes > 0.3
            ),
            "per_round": [
                {
                    "round": a.round_number,
                    "advantage_wave": a.advantage_wave,
                    "advantage_strokes": round(a.advantage_strokes, 2),
                }
                for a in wave_advantages
            ],
        }
