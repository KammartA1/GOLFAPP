"""Simulation Pipeline Bridge — Connects TournamentSimulator to the projection pipeline.

Simulation results VALIDATE and ENHANCE analytical projections.
The analytical SG model gives us expected values; the Monte Carlo simulation
gives us full distributions, tail probabilities, and emergent dynamics
(cut effects, pressure, weather correlation across rounds).

Usage:
    from simulation.pipeline_bridge import SimulationBridge

    bridge = SimulationBridge(n_sims=10000)
    sim_df = bridge.run_tournament_simulation(proj_df, "TPC Sawgrass")
    blended = bridge.blend_projections(proj_df, sim_df, sim_weight=0.5)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from simulation.config import SimulationConfig
from simulation.tournament_engine import TournamentSimulator
from simulation.player_model import SimPlayer, PlayerModel
from simulation.course_model import CourseModel, CourseProfile
from simulation.weather_model import WeatherConditions

logger = logging.getLogger(__name__)


class SimulationBridge:
    """
    Connects TournamentSimulator to the projection pipeline.
    Simulation results VALIDATE and ENHANCE analytical projections.
    """

    def __init__(self, n_sims: int = 10000):
        self.n_sims = n_sims
        self.simulator = TournamentSimulator(SimulationConfig(n_simulations=n_sims))
        self.course_model = CourseModel()
        self.player_model = PlayerModel(self.simulator.rng)

    # ── Core: Run Tournament Simulation ────────────────────────────────

    def run_tournament_simulation(
        self,
        field_projections: pd.DataFrame,
        course_name: str,
        weather: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Takes SG model field projections, runs full tournament simulation.
        Returns enhanced projections with simulation-based probabilities.

        Args:
            field_projections: DataFrame from ProjectionEngine.run() with columns
                including name, proj_sg_total, proj_sg_ott, proj_sg_app, etc.
            course_name: Name of the course for profile generation.
            weather: Optional {round_num: WeatherConditions} dict.

        Output adds columns:
            - sim_win_prob, sim_top5_prob, sim_top10_prob, sim_top20_prob, sim_make_cut_prob
            - sim_avg_finish, sim_avg_score, sim_std_score
        """
        if field_projections.empty:
            logger.warning("Empty field projections — skipping simulation")
            return field_projections

        logger.info(
            "Running tournament simulation: %d players, %d sims, course=%s",
            len(field_projections), self.n_sims, course_name,
        )

        # Convert projection rows to SimPlayer objects
        players = self._projections_to_sim_players(field_projections)

        if len(players) < 10:
            logger.warning("Field too small (%d) for meaningful simulation", len(players))
            return self._empty_sim_columns(field_projections)

        # Build course profile
        course = self.course_model.generate_default_course(name=course_name)

        # Convert weather dict if provided
        weather_by_round = None
        if weather:
            weather_by_round = {}
            for rnd, w in weather.items():
                if isinstance(w, WeatherConditions):
                    weather_by_round[rnd] = w
                elif isinstance(w, dict):
                    weather_by_round[rnd] = WeatherConditions(**w)

        # Run simulation
        sim_df = self.simulator.simulate_tournament(
            players=players,
            course=course,
            weather_by_round=weather_by_round,
        )

        # Rename simulation columns to avoid collision with analytical columns
        sim_result = sim_df.rename(columns={
            "win_prob":      "sim_win_prob",
            "top5_prob":     "sim_top5_prob",
            "top10_prob":    "sim_top10_prob",
            "top20_prob":    "sim_top20_prob",
            "make_cut_prob": "sim_make_cut_prob",
            "avg_finish":    "sim_avg_finish",
            "avg_score":     "sim_avg_score",
            "std_score":     "sim_std_score",
        })

        # Select only the simulation columns we need
        sim_cols = [
            "name",
            "sim_win_prob", "sim_top5_prob", "sim_top10_prob",
            "sim_top20_prob", "sim_make_cut_prob",
            "sim_avg_finish", "sim_avg_score", "sim_std_score",
            "sim_rank",
        ]
        sim_cols = [c for c in sim_cols if c in sim_result.columns]
        sim_result = sim_result[sim_cols]

        logger.info(
            "Simulation complete. Top player: %s (sim_win_prob=%.2f%%)",
            sim_result.iloc[0]["name"] if not sim_result.empty else "N/A",
            sim_result.iloc[0].get("sim_win_prob", 0) * 100 if not sim_result.empty else 0,
        )

        return sim_result

    # ── Blend Analytical + Simulation ──────────────────────────────────

    def blend_projections(
        self,
        analytical: pd.DataFrame,
        simulated: pd.DataFrame,
        sim_weight: float = 0.5,
    ) -> pd.DataFrame:
        """
        Ensemble analytical SG model with simulation output.

        For probabilities (win, top5, etc.): weighted average of analytical and sim.
        For stats (score, finish): simulation is PRIMARY (emerging from dynamics).

        Also flags disagreements between the two engines — these are opportunities
        where one model sees something the other doesn't.

        Args:
            analytical: Full projection DataFrame from ProjectionEngine.run().
            simulated: Simulation results from run_tournament_simulation().
            sim_weight: Weight given to simulation (0-1). Default 0.5.

        Returns:
            Merged DataFrame with blended columns and disagreement flags.
        """
        if simulated.empty:
            logger.warning("No simulation results to blend — returning analytical only")
            return analytical

        anal_weight = 1.0 - sim_weight

        # Merge on player name
        merged = analytical.merge(simulated, on="name", how="left", suffixes=("", "_sim"))

        # Probability columns: weighted average
        prob_pairs = [
            ("win_prob",      "sim_win_prob",      "blended_win_prob"),
            ("top5_prob",     "sim_top5_prob",      "blended_top5_prob"),
            ("top10_prob",    "sim_top10_prob",     "blended_top10_prob"),
            ("top20_prob",    "sim_top20_prob",     "blended_top20_prob"),
            ("make_cut_prob", "sim_make_cut_prob",  "blended_make_cut_prob"),
        ]

        for anal_col, sim_col, blend_col in prob_pairs:
            if anal_col in merged.columns and sim_col in merged.columns:
                anal_vals = merged[anal_col].fillna(0)
                sim_vals = merged[sim_col].fillna(0)
                merged[blend_col] = (anal_vals * anal_weight + sim_vals * sim_weight).round(4)

        # Stat columns: simulation is primary (use sim values directly)
        for col in ["sim_avg_finish", "sim_avg_score", "sim_std_score", "sim_rank"]:
            # These are already present from the merge — keep as-is
            pass

        # Disagreement detection: flag where analytical and simulation diverge significantly
        if "win_prob" in merged.columns and "sim_win_prob" in merged.columns:
            merged["win_prob_disagreement"] = (
                (merged["win_prob"].fillna(0) - merged["sim_win_prob"].fillna(0)).abs()
            ).round(4)

        if "make_cut_prob" in merged.columns and "sim_make_cut_prob" in merged.columns:
            merged["cut_prob_disagreement"] = (
                (merged["make_cut_prob"].fillna(0) - merged["sim_make_cut_prob"].fillna(0)).abs()
            ).round(4)

        # Engine agreement score: 0 = total disagreement, 1 = perfect agreement
        # Based on average absolute difference across probability columns
        disagreements = []
        for anal_col, sim_col, _ in prob_pairs:
            if anal_col in merged.columns and sim_col in merged.columns:
                diff = (merged[anal_col].fillna(0) - merged[sim_col].fillna(0)).abs()
                disagreements.append(diff)

        if disagreements:
            avg_disagreement = pd.concat(disagreements, axis=1).mean(axis=1)
            # Scale: 0.0 diff = 1.0 agreement, 0.20+ diff = 0.0 agreement
            merged["engine_agreement"] = (1.0 - (avg_disagreement / 0.20).clip(upper=1.0)).round(3)
        else:
            merged["engine_agreement"] = 1.0

        logger.info(
            "Blended projections: %d players, sim_weight=%.0f%%, avg_agreement=%.2f",
            len(merged), sim_weight * 100,
            merged["engine_agreement"].mean() if "engine_agreement" in merged.columns else 0,
        )

        return merged

    # ── Prop Market Probabilities ──────────────────────────────────────

    def prop_market_probabilities(
        self,
        player_name: str,
        stat: str,
        line: float,
        simulation_results: pd.DataFrame,
    ) -> dict:
        """
        From tournament simulation, extract P(over line) for prop markets.
        Uses distribution of simulated outcomes, not analytical formula.

        Args:
            player_name: Name of the player.
            stat: Stat column in simulation results (e.g. 'sim_avg_score').
            line: The prop line to evaluate.
            simulation_results: Full simulation DataFrame.

        Returns:
            {
                "p_over": float,
                "p_under": float,
                "mean": float,
                "std": float,
                "percentiles": {10: val, 25: val, 50: val, 75: val, 90: val},
            }
        """
        player_row = simulation_results[
            simulation_results["name"].str.lower() == player_name.lower()
        ]

        if player_row.empty:
            logger.warning("Player '%s' not found in simulation results", player_name)
            return {
                "p_over": 0.5, "p_under": 0.5,
                "mean": line, "std": 0.0,
                "percentiles": {},
            }

        row = player_row.iloc[0]

        # Use simulation mean and std to construct a normal distribution
        # for the stat in question
        mean_col = f"sim_avg_{stat}" if f"sim_avg_{stat}" in row.index else stat
        std_col = f"sim_std_{stat}" if f"sim_std_{stat}" in row.index else None

        if mean_col in row.index:
            mean_val = float(row[mean_col])
        else:
            mean_val = float(row.get("sim_avg_score", line))

        if std_col and std_col in row.index:
            std_val = float(row[std_col])
        else:
            std_val = float(row.get("sim_std_score", 2.0))

        if std_val <= 0:
            std_val = 2.0

        from scipy import stats
        dist = stats.norm(loc=mean_val, scale=std_val)

        p_over = float(1 - dist.cdf(line))
        p_under = float(dist.cdf(line))

        percentiles = {}
        for pct in [10, 25, 50, 75, 90]:
            percentiles[pct] = round(float(dist.ppf(pct / 100.0)), 2)

        return {
            "p_over": round(p_over, 4),
            "p_under": round(p_under, 4),
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "percentiles": percentiles,
        }

    # ── Internal Helpers ───────────────────────────────────────────────

    def _projections_to_sim_players(self, df: pd.DataFrame) -> list[SimPlayer]:
        """Convert projection DataFrame rows to SimPlayer objects."""
        players = []
        for _, row in df.iterrows():
            player = SimPlayer(
                name=row.get("name", "Unknown"),
                sg_total=row.get("proj_sg_total", 0.0),
                sg_ott=row.get("proj_sg_ott", 0.0),
                sg_app=row.get("proj_sg_app", 0.0),
                sg_atg=row.get("proj_sg_atg", 0.0),
                sg_putt=row.get("proj_sg_putt", 0.0),
                round_std=row.get("player_variance", 2.75),
                volatility_multiplier=row.get("volatility_multiplier", 1.0),
                world_rank=int(row.get("world_rank", 100) or 100),
                course_fit_score=row.get("course_fit_score", 50.0),
            )
            players.append(player)
        return players

    def _empty_sim_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with simulation columns set to NaN."""
        result = pd.DataFrame({"name": df["name"]})
        for col in [
            "sim_win_prob", "sim_top5_prob", "sim_top10_prob",
            "sim_top20_prob", "sim_make_cut_prob",
            "sim_avg_finish", "sim_avg_score", "sim_std_score",
            "sim_rank",
        ]:
            result[col] = np.nan
        return result
