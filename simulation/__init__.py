"""Hole-by-hole tournament simulation engine.

Full Monte Carlo simulation of PGA Tour tournaments:
  - 18-hole round simulation with discrete outcomes per hole
  - 4-round tournament with cut dynamics
  - Player-specific SG-based scoring models
  - Weather, wave, and pressure effects
  - N=10,000 simulations for outcome distributions

Usage:
    from simulation.tournament_engine import TournamentSimulator

    sim = TournamentSimulator(n_sims=10000)
    results = sim.simulate_tournament(players, course, weather)
"""

from simulation.tournament_engine import TournamentSimulator
from simulation.config import SimulationConfig

__all__ = ["TournamentSimulator", "SimulationConfig"]
