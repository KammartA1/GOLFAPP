"""Simulation configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration for tournament simulation."""

    n_simulations: int = 10000
    n_rounds: int = 4
    n_holes: int = 18
    cut_after_round: int = 2  # Cut happens after round 2
    cut_top_n: int = 65  # Top 65 + ties make cut (PGA Tour standard)
    cut_plus_ties: bool = True

    # Scoring variance parameters
    tour_avg_round_std: float = 2.75  # SG std per round (tour average)
    hole_to_hole_correlation: float = 0.05  # Mild within-round momentum
    round_to_round_correlation: float = 0.15  # Day-to-day consistency

    # Outcome probabilities (tour average on par-4)
    base_eagle_rate: float = 0.005
    base_birdie_rate: float = 0.18
    base_par_rate: float = 0.60
    base_bogey_rate: float = 0.16
    base_double_plus_rate: float = 0.055

    # Pressure model
    pressure_enabled: bool = True
    max_pressure_effect_sg: float = 0.30  # Max pressure impact in SG

    # Weather model
    weather_enabled: bool = True

    # Random seed (None = truly random)
    random_seed: int | None = None

    # Minimum field size for meaningful simulation
    min_field_size: int = 10
