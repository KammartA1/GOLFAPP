"""
Golf Quant Engine — Simulation Configuration
=============================================
All tunable simulation parameters in a single dataclass.
Defaults are calibrated to PGA Tour historical distributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ScoringRates:
    """Hole outcome probability rates by par value and difficulty tier.

    Keys: (par, difficulty_tier) -> dict of outcome probabilities.
    difficulty_tier: 0 = easy, 1 = medium, 2 = hard.
    Probabilities are for a TOUR-AVERAGE player in calm conditions.
    """
    # Par 3 distributions  — [eagle, birdie, par, bogey, double_plus]
    par3_easy:   list[float] = field(default_factory=lambda: [0.002, 0.18, 0.62, 0.155, 0.043])
    par3_medium: list[float] = field(default_factory=lambda: [0.001, 0.13, 0.60, 0.20, 0.069])
    par3_hard:   list[float] = field(default_factory=lambda: [0.0005, 0.08, 0.56, 0.27, 0.0895])

    # Par 4 distributions
    par4_easy:   list[float] = field(default_factory=lambda: [0.005, 0.25, 0.56, 0.145, 0.040])
    par4_medium: list[float] = field(default_factory=lambda: [0.002, 0.18, 0.58, 0.185, 0.053])
    par4_hard:   list[float] = field(default_factory=lambda: [0.001, 0.10, 0.55, 0.26, 0.089])

    # Par 5 distributions
    par5_easy:   list[float] = field(default_factory=lambda: [0.04, 0.42, 0.44, 0.075, 0.025])
    par5_medium: list[float] = field(default_factory=lambda: [0.02, 0.35, 0.48, 0.11, 0.040])
    par5_hard:   list[float] = field(default_factory=lambda: [0.008, 0.25, 0.52, 0.17, 0.052])

    def get_rates(self, par: int, difficulty_tier: int) -> list[float]:
        """Return [eagle, birdie, par, bogey, double_plus] for given par and tier."""
        key = f"par{par}_{'easy' if difficulty_tier == 0 else 'medium' if difficulty_tier == 1 else 'hard'}"
        return list(getattr(self, key))


@dataclass
class WeatherCoefficients:
    """Stroke penalties per unit of weather effect."""
    # Wind: additional strokes per hole per mph above threshold
    wind_threshold_mph: float = 8.0
    wind_stroke_penalty_per_mph: float = 0.035
    wind_max_penalty_per_hole: float = 0.80

    # Rain: flat penalty per hole when raining
    rain_light_penalty: float = 0.08       # < 2 mm/hr
    rain_moderate_penalty: float = 0.18    # 2-5 mm/hr
    rain_heavy_penalty: float = 0.35       # > 5 mm/hr

    # Temperature: stroke adjustment per degree F below/above baseline
    temp_baseline_f: float = 72.0
    cold_penalty_per_degree: float = 0.005   # Per degree below baseline
    heat_penalty_per_degree: float = 0.002   # Per degree above 95F

    # Combined cap
    max_weather_penalty_per_hole: float = 1.2


@dataclass
class PressureCoefficients:
    """Pressure effect parameters for contention scenarios."""
    # Contention threshold: shots back from lead
    contention_threshold: int = 5

    # SG mean shift under pressure (multiplied by player's pressure_coeff)
    # Positive pressure_coeff = closer (benefits), negative = choker (hurts)
    pressure_sg_mean_shift: float = 0.04

    # SG variance multiplier under pressure
    # Pressure increases variance for everyone
    pressure_variance_multiplier: float = 1.25

    # Sunday back 9 amplifier (pressure is 1.5x on back 9 Sunday)
    sunday_back9_amplifier: float = 1.50

    # Round-specific pressure scaling (R1=0, R2=0.3, R3=0.7, R4=1.0)
    round_pressure_scale: list[float] = field(
        default_factory=lambda: [0.0, 0.3, 0.7, 1.0]
    )

    # First-time contender penalty (less experience under pressure)
    first_time_contender_penalty: float = -0.15


@dataclass
class WaveParameters:
    """AM/PM wave advantage parameters."""
    # Default wave advantage in strokes (AM advantage when PM is windier)
    default_wave_advantage: float = 0.0

    # Maximum historical wave advantage observed
    max_wave_advantage: float = 3.0

    # Wind ramp: typical afternoon wind increase (mph)
    afternoon_wind_increase_mph: float = 5.0

    # Links courses have larger wave effects
    links_wave_multiplier: float = 1.8
    desert_wave_multiplier: float = 1.3
    parkland_wave_multiplier: float = 1.0


COURSE_TYPES = {
    "links": {
        "wind_sensitivity": 0.90,
        "rain_sensitivity": 0.60,
        "temp_sensitivity": 0.30,
        "wave_multiplier": 1.8,
        "elevation_effect": 0.0,
    },
    "parkland": {
        "wind_sensitivity": 0.45,
        "rain_sensitivity": 0.50,
        "temp_sensitivity": 0.40,
        "wave_multiplier": 1.0,
        "elevation_effect": 0.0,
    },
    "desert": {
        "wind_sensitivity": 0.55,
        "rain_sensitivity": 0.20,
        "temp_sensitivity": 0.60,
        "wave_multiplier": 1.3,
        "elevation_effect": 0.02,
    },
    "coastal_parkland": {
        "wind_sensitivity": 0.70,
        "rain_sensitivity": 0.50,
        "temp_sensitivity": 0.35,
        "wave_multiplier": 1.4,
        "elevation_effect": 0.0,
    },
    "links_coastal": {
        "wind_sensitivity": 0.85,
        "rain_sensitivity": 0.55,
        "temp_sensitivity": 0.30,
        "wave_multiplier": 1.7,
        "elevation_effect": 0.0,
    },
    "parkland_links_hybrid": {
        "wind_sensitivity": 0.60,
        "rain_sensitivity": 0.55,
        "temp_sensitivity": 0.35,
        "wave_multiplier": 1.3,
        "elevation_effect": 0.0,
    },
    "parkland_coastal": {
        "wind_sensitivity": 0.60,
        "rain_sensitivity": 0.50,
        "temp_sensitivity": 0.35,
        "wave_multiplier": 1.3,
        "elevation_effect": 0.0,
    },
    "parkland_tropical": {
        "wind_sensitivity": 0.35,
        "rain_sensitivity": 0.65,
        "temp_sensitivity": 0.55,
        "wave_multiplier": 0.9,
        "elevation_effect": 0.0,
    },
}


@dataclass
class SimulationConfig:
    """Master configuration for the tournament simulator."""

    # Core simulation params
    n_simulations: int = 10_000
    rounds_per_tournament: int = 4
    holes_per_round: int = 18
    cut_after_round: int = 2
    random_seed: int = 42

    # SG component weights (matching config/settings.py)
    sg_weights: Dict[str, float] = field(default_factory=lambda: {
        "sg_ott": 0.25,
        "sg_app": 0.38,
        "sg_atg": 0.22,
        "sg_putt": 0.15,
    })

    # Cut rules
    cut_top_n: int = 65        # Top N + ties make the cut
    cut_ties: bool = True      # Include ties at the cut number
    mdf_enabled: bool = True   # Made-cut-didn't-finish (top 65 + ties)

    # Field size
    default_field_size: int = 156

    # Sub-configs
    scoring_rates: ScoringRates = field(default_factory=ScoringRates)
    weather_coefficients: WeatherCoefficients = field(default_factory=WeatherCoefficients)
    pressure_coefficients: PressureCoefficients = field(default_factory=PressureCoefficients)
    wave_parameters: WaveParameters = field(default_factory=WaveParameters)

    # Fatigue (minimal in golf but non-zero in extreme heat or altitude)
    fatigue_per_hole: float = 0.001
    heat_fatigue_threshold_f: float = 95.0
    heat_fatigue_multiplier: float = 2.5

    # Autocorrelation within a round (hot/cold streaks)
    intra_round_momentum: float = 0.10  # Correlation of consecutive hole performance
