"""
Golf Quant Engine — Tournament Engine (THE CORE)
==================================================
Full Monte Carlo tournament simulator.  Runs N simulations of a
4-round tournament with cut, pressure, weather, and wave effects.
Produces win/top-N/cut probabilities and head-to-head edges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from simulation.config import SimulationConfig
from simulation.course_model import CourseModel
from simulation.cut_model import CutModel, CutResult
from simulation.player_model import PlayerModel, PlayerSGComponents
from simulation.pressure_model import PressureModel
from simulation.round_engine import RoundEngine, RoundResult
from simulation.volatility_model import VolatilityModel
from simulation.wave_model import WaveModel, WaveAssignment, WaveAdvantage
from simulation.weather_model import WeatherModel, RoundWeather

log = logging.getLogger(__name__)


@dataclass
class PlayerTournamentResult:
    """Aggregated results for one player across all simulations."""
    player_name: str
    n_simulations: int

    # Position probabilities
    win_prob: float = 0.0
    top3_prob: float = 0.0
    top5_prob: float = 0.0
    top10_prob: float = 0.0
    top20_prob: float = 0.0
    top30_prob: float = 0.0
    make_cut_prob: float = 0.0

    # Score distributions
    avg_total_score: float = 0.0     # Average 4-round total vs par
    score_std: float = 0.0
    avg_round_score: float = 0.0
    round_scores_r1: List[float] = field(default_factory=list)
    round_scores_r2: List[float] = field(default_factory=list)
    round_scores_r3: List[float] = field(default_factory=list)
    round_scores_r4: List[float] = field(default_factory=list)

    # Finishing position distribution
    position_distribution: Dict[int, float] = field(default_factory=dict)
    avg_finish_position: float = 0.0
    median_finish_position: float = 0.0

    # Underlying data (for further analysis)
    finishing_positions: Optional[np.ndarray] = None
    total_scores: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "player_name": self.player_name,
            "win_prob": round(self.win_prob, 4),
            "top5_prob": round(self.top5_prob, 4),
            "top10_prob": round(self.top10_prob, 4),
            "top20_prob": round(self.top20_prob, 4),
            "make_cut_prob": round(self.make_cut_prob, 4),
            "avg_total_score": round(self.avg_total_score, 2),
            "score_std": round(self.score_std, 2),
            "avg_finish_position": round(self.avg_finish_position, 1),
        }


@dataclass
class TournamentResult:
    """Complete tournament simulation output."""
    tournament_name: str
    course_name: str
    n_simulations: int
    n_players: int

    player_results: Dict[str, PlayerTournamentResult]

    # Metadata
    config: Optional[SimulationConfig] = None
    avg_cut_line: float = 0.0
    cut_line_std: float = 0.0
    avg_winning_score: float = 0.0
    winning_score_std: float = 0.0

    def get_leaderboard(self, sort_by: str = "win_prob") -> pd.DataFrame:
        """Return a leaderboard DataFrame sorted by the specified metric."""
        rows = []
        for name, pr in self.player_results.items():
            rows.append({
                "Player": name,
                "Win%": round(pr.win_prob * 100, 2),
                "Top5%": round(pr.top5_prob * 100, 2),
                "Top10%": round(pr.top10_prob * 100, 2),
                "Top20%": round(pr.top20_prob * 100, 2),
                "MakeCut%": round(pr.make_cut_prob * 100, 2),
                "AvgScore": round(pr.avg_total_score, 2),
                "ScoreStd": round(pr.score_std, 2),
                "AvgFinish": round(pr.avg_finish_position, 1),
            })

        df = pd.DataFrame(rows)
        col_map = {
            "win_prob": "Win%",
            "top5_prob": "Top5%",
            "top10_prob": "Top10%",
            "top20_prob": "Top20%",
            "make_cut_prob": "MakeCut%",
            "avg_score": "AvgScore",
            "avg_finish": "AvgFinish",
        }
        sort_col = col_map.get(sort_by, "Win%")
        ascending = sort_col in ("AvgScore", "AvgFinish")
        df = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "Rank"
        return df

    def get_h2h_probability(self, player_a: str, player_b: str) -> float:
        """Get probability that player_a finishes ahead of player_b.

        Computed from the stored finishing position arrays across simulations.
        """
        res_a = self.player_results.get(player_a)
        res_b = self.player_results.get(player_b)

        if res_a is None or res_b is None:
            raise ValueError(
                f"Player not found. Available: {list(self.player_results.keys())}"
            )

        if res_a.total_scores is None or res_b.total_scores is None:
            return 0.5  # No data

        # Lower total score = better finish
        a_wins = np.sum(res_a.total_scores < res_b.total_scores)
        ties = np.sum(res_a.total_scores == res_b.total_scores)
        total = len(res_a.total_scores)

        return float((a_wins + 0.5 * ties) / total)

    def get_position_probability(self, player: str, position: int) -> float:
        """Get probability that a player finishes at or above a position.

        Parameters
        ----------
        player : str
            Player name.
        position : int
            Target position (e.g., 10 = top 10).
        """
        res = self.player_results.get(player)
        if res is None:
            raise ValueError(f"Player {player!r} not found.")

        if res.finishing_positions is None:
            return 0.0

        return float(np.mean(res.finishing_positions <= position))

    def get_h2h_matrix(self, player_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Build a head-to-head probability matrix for selected players."""
        if player_names is None:
            player_names = list(self.player_results.keys())[:20]

        data = {}
        for a in player_names:
            row = {}
            for b in player_names:
                if a == b:
                    row[b] = 0.5
                else:
                    row[b] = round(self.get_h2h_probability(a, b), 3)
            data[a] = row

        return pd.DataFrame(data, index=player_names)


class TournamentEngine:
    """Monte Carlo tournament simulator — the core engine.

    Simulates N complete tournaments (4 rounds each) with:
    - Full field of players (each with a PlayerModel)
    - Course model with hole-by-hole specs
    - Weather generation per round
    - AM/PM wave effects (rounds 1-2)
    - 36-hole cut (top 65 + ties)
    - Pressure model (rounds 3-4)
    - Fatigue and momentum

    Usage::

        engine = TournamentEngine()
        result = engine.run_simulation(
            players=[...],
            course=CourseModel(...),
            n_simulations=10000,
        )
        print(result.get_leaderboard())
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.round_engine = RoundEngine(self.config)
        self.cut_model = CutModel(
            cut_top_n=self.config.cut_top_n,
            cut_ties=self.config.cut_ties,
            mdf_enabled=self.config.mdf_enabled,
        )
        self.wave_model = WaveModel(self.config)
        self.weather_model = WeatherModel(self.config)
        self.volatility_model = VolatilityModel()

    def run_simulation(
        self,
        players: list[PlayerModel],
        course: CourseModel,
        n_simulations: Optional[int] = None,
        seed: Optional[int] = None,
        weather_base_wind: float = 10.0,
        weather_base_temp: float = 72.0,
        weather_rain_prob: float = 0.15,
        weather_wind_direction: float = 180.0,
        tournament_name: str = "Simulation",
        progress_callback=None,
    ) -> TournamentResult:
        """Run the full Monte Carlo tournament simulation.

        Parameters
        ----------
        players : list of PlayerModel
            Field of players.
        course : CourseModel
            Course specification.
        n_simulations : int, optional
            Number of simulations (default from config).
        seed : int, optional
            Random seed.
        weather_base_wind : float
            Base wind speed.
        weather_base_temp : float
            Base temperature.
        weather_rain_prob : float
            Rain probability per round.
        weather_wind_direction : float
            Prevailing wind direction.
        tournament_name : str
            Name for display.
        progress_callback : callable, optional
            Called with (sim_number, n_total) for progress tracking.

        Returns
        -------
        TournamentResult with all probabilities and distributions.
        """
        if n_simulations is None:
            n_simulations = self.config.n_simulations
        if seed is None:
            seed = self.config.random_seed

        n_players = len(players)
        player_names = [p.name for p in players]
        n_rounds = self.config.rounds_per_tournament

        log.info(
            "Starting simulation: %d sims, %d players, %s at %s",
            n_simulations, n_players, tournament_name, course.name,
        )

        # Storage arrays
        # finishing_positions[sim, player] = final position in that sim
        finishing_positions = np.zeros((n_simulations, n_players), dtype=np.int32)
        total_scores = np.zeros((n_simulations, n_players), dtype=np.float64)
        round_scores = np.zeros((n_simulations, n_players, n_rounds), dtype=np.float64)
        made_cut_flags = np.zeros((n_simulations, n_players), dtype=np.bool_)
        cut_lines = np.zeros(n_simulations, dtype=np.float64)
        winning_scores = np.zeros(n_simulations, dtype=np.float64)

        # Master RNG
        master_rng = np.random.default_rng(seed)

        for sim in range(n_simulations):
            # Seed for this simulation
            sim_rng = np.random.default_rng(master_rng.integers(0, 2**32))

            # Generate weather for this simulation
            tournament_weather = self.weather_model.generate_tournament_weather(
                rng=sim_rng,
                n_rounds=n_rounds,
                base_wind=weather_base_wind,
                base_temp=weather_base_temp,
                rain_probability=weather_rain_prob,
                base_wind_direction=weather_wind_direction,
                course_type=course.course_type,
            )

            # Generate wave assignments
            wave_assignments = self.wave_model.assign_waves(
                player_names=player_names,
                n_rounds=n_rounds,
                rng=sim_rng,
            )
            wave_advantages = self.wave_model.calculate_wave_advantages(
                round_weathers=tournament_weather,
                course_type=course.course_type,
            )

            # --- Rounds 1-2 (pre-cut) ---
            r1_scores = np.zeros(n_players, dtype=np.float64)
            r2_scores = np.zeros(n_players, dtype=np.float64)

            # Round 1
            r1_weather = tournament_weather[0] if len(tournament_weather) > 0 else None
            r1_penalties = self._get_weather_penalties(r1_weather, course)
            r1_wave_adv = wave_advantages[0] if len(wave_advantages) > 0 else None

            for pi, player in enumerate(players):
                wave = wave_assignments[pi].get_wave(1) if pi < len(wave_assignments) else "AM"
                wave_adj = self.wave_model.get_player_wave_adjustment(
                    wave, r1_wave_adv
                ) if r1_wave_adv else 0.0

                score = self.round_engine.simulate_round_fast(
                    player=player,
                    course=course,
                    rng=sim_rng,
                    round_number=1,
                    weather_penalties=r1_penalties,
                )
                r1_scores[pi] = score + wave_adj

            round_scores[sim, :, 0] = r1_scores

            # Round 2
            r2_weather = tournament_weather[1] if len(tournament_weather) > 1 else None
            r2_penalties = self._get_weather_penalties(r2_weather, course)
            r2_wave_adv = wave_advantages[1] if len(wave_advantages) > 1 else None

            for pi, player in enumerate(players):
                wave = wave_assignments[pi].get_wave(2) if pi < len(wave_assignments) else "AM"
                wave_adj = self.wave_model.get_player_wave_adjustment(
                    wave, r2_wave_adv
                ) if r2_wave_adv else 0.0

                score = self.round_engine.simulate_round_fast(
                    player=player,
                    course=course,
                    rng=sim_rng,
                    round_number=2,
                    weather_penalties=r2_penalties,
                )
                r2_scores[pi] = score + wave_adj

            round_scores[sim, :, 1] = r2_scores

            # --- Apply cut ---
            scores_after_r2 = r1_scores + r2_scores
            cut_result = self.cut_model.calculate_cut_line(scores_after_r2)
            cut_lines[sim] = cut_result.cut_line

            made_cut_mask = scores_after_r2 <= cut_result.cut_line
            made_cut_flags[sim] = made_cut_mask

            # --- Rounds 3-4 (post-cut, with pressure) ---
            r3_scores = np.full(n_players, np.nan)
            r4_scores = np.full(n_players, np.nan)

            active_indices = np.where(made_cut_mask)[0]
            cumulative_active = scores_after_r2.copy()

            # Round 3
            if len(active_indices) > 0:
                r3_weather = tournament_weather[2] if len(tournament_weather) > 2 else None
                r3_penalties = self._get_weather_penalties(r3_weather, course)
                leader_score_r3 = np.min(cumulative_active[active_indices])

                for pi in active_indices:
                    score = self.round_engine.simulate_round_fast(
                        player=players[pi],
                        course=course,
                        rng=sim_rng,
                        round_number=3,
                        weather_penalties=r3_penalties,
                        cumulative_score=cumulative_active[pi],
                        leader_score=leader_score_r3,
                    )
                    r3_scores[pi] = score
                    cumulative_active[pi] += score

            round_scores[sim, :, 2] = r3_scores

            # Round 4
            if len(active_indices) > 0:
                r4_weather = tournament_weather[3] if len(tournament_weather) > 3 else None
                r4_penalties = self._get_weather_penalties(r4_weather, course)
                leader_score_r4 = np.nanmin(cumulative_active[active_indices])

                for pi in active_indices:
                    score = self.round_engine.simulate_round_fast(
                        player=players[pi],
                        course=course,
                        rng=sim_rng,
                        round_number=4,
                        weather_penalties=r4_penalties,
                        cumulative_score=cumulative_active[pi],
                        leader_score=leader_score_r4,
                    )
                    r4_scores[pi] = score
                    cumulative_active[pi] += score

            round_scores[sim, :, 3] = r4_scores

            # --- Calculate final positions ---
            final_scores = np.full(n_players, 999.0)  # Missed cut = 999
            for pi in active_indices:
                final_scores[pi] = cumulative_active[pi]

            # Rank: lower score = better position
            sorted_indices = np.argsort(final_scores)
            positions = np.zeros(n_players, dtype=np.int32)

            current_pos = 1
            i = 0
            while i < n_players:
                # Find all players tied at this score
                tie_score = final_scores[sorted_indices[i]]
                j = i
                while j < n_players and final_scores[sorted_indices[j]] == tie_score:
                    j += 1
                # All tied players get the same position
                for k in range(i, j):
                    positions[sorted_indices[k]] = current_pos
                current_pos = j + 1
                i = j

            # Players who missed cut get position = n_players
            for pi in range(n_players):
                if not made_cut_mask[pi]:
                    positions[pi] = n_players

            finishing_positions[sim] = positions
            total_scores[sim] = final_scores
            winning_scores[sim] = np.min(final_scores)

            if progress_callback and (sim + 1) % max(1, n_simulations // 20) == 0:
                progress_callback(sim + 1, n_simulations)

        # --- Aggregate results ---
        player_results = {}
        for pi, player in enumerate(players):
            name = player.name
            pos_array = finishing_positions[:, pi]
            score_array = total_scores[:, pi]
            valid_scores = score_array[score_array < 999]

            # Position probabilities
            win_p = float(np.mean(pos_array == 1))
            top3_p = float(np.mean(pos_array <= 3))
            top5_p = float(np.mean(pos_array <= 5))
            top10_p = float(np.mean(pos_array <= 10))
            top20_p = float(np.mean(pos_array <= 20))
            top30_p = float(np.mean(pos_array <= 30))
            mc_p = float(np.mean(made_cut_flags[:, pi]))

            # Score stats
            avg_score = float(np.mean(valid_scores)) if len(valid_scores) > 0 else 999.0
            score_std_val = float(np.std(valid_scores)) if len(valid_scores) > 0 else 0.0

            # Position distribution (binned)
            pos_dist = {}
            for pos in range(1, n_players + 1):
                prob = float(np.mean(pos_array == pos))
                if prob > 0.001:
                    pos_dist[pos] = round(prob, 4)

            # Round score distributions
            r1_scores_player = round_scores[:, pi, 0]
            r2_scores_player = round_scores[:, pi, 1]
            r3_scores_player = round_scores[:, pi, 2]
            r4_scores_player = round_scores[:, pi, 3]

            pr = PlayerTournamentResult(
                player_name=name,
                n_simulations=n_simulations,
                win_prob=win_p,
                top3_prob=top3_p,
                top5_prob=top5_p,
                top10_prob=top10_p,
                top20_prob=top20_p,
                top30_prob=top30_p,
                make_cut_prob=mc_p,
                avg_total_score=avg_score,
                score_std=score_std_val,
                avg_round_score=avg_score / 4 if avg_score < 999 else 0,
                round_scores_r1=r1_scores_player.tolist(),
                round_scores_r2=r2_scores_player.tolist(),
                round_scores_r3=[s for s in r3_scores_player.tolist() if not np.isnan(s)],
                round_scores_r4=[s for s in r4_scores_player.tolist() if not np.isnan(s)],
                position_distribution=pos_dist,
                avg_finish_position=float(np.mean(pos_array)),
                median_finish_position=float(np.median(pos_array)),
                finishing_positions=pos_array,
                total_scores=score_array,
            )

            player_results[name] = pr

        result = TournamentResult(
            tournament_name=tournament_name,
            course_name=course.name,
            n_simulations=n_simulations,
            n_players=n_players,
            player_results=player_results,
            config=self.config,
            avg_cut_line=float(np.mean(cut_lines)),
            cut_line_std=float(np.std(cut_lines)),
            avg_winning_score=float(np.mean(winning_scores)),
            winning_score_std=float(np.std(winning_scores)),
        )

        log.info(
            "Simulation complete. Avg winning score: %.1f, Avg cut line: %.1f",
            result.avg_winning_score,
            result.avg_cut_line,
        )

        return result

    def _get_weather_penalties(
        self,
        round_weather: Optional[RoundWeather],
        course: CourseModel,
    ) -> np.ndarray:
        """Get per-hole weather penalties for a round."""
        if round_weather is None:
            return np.zeros(len(course.holes))
        return self.weather_model.calculate_round_weather_penalties(
            round_weather=round_weather,
            holes=course.holes,
            course_type=course.course_type,
        )

    def run_quick_simulation(
        self,
        players: list[PlayerModel],
        course: CourseModel,
        n_simulations: int = 1000,
        seed: int = 42,
    ) -> TournamentResult:
        """Run a quick simulation with default weather and fewer sims.

        Useful for rapid iteration and testing.
        """
        return self.run_simulation(
            players=players,
            course=course,
            n_simulations=n_simulations,
            seed=seed,
            tournament_name="Quick Simulation",
        )

    def compare_h2h(
        self,
        player_a: PlayerModel,
        player_b: PlayerModel,
        course: CourseModel,
        n_simulations: int = 5000,
        seed: int = 42,
    ) -> dict:
        """Run a dedicated head-to-head comparison between two players.

        Returns detailed H2H statistics.
        """
        result = self.run_simulation(
            players=[player_a, player_b],
            course=course,
            n_simulations=n_simulations,
            seed=seed,
            tournament_name=f"H2H: {player_a.name} vs {player_b.name}",
        )

        a_prob = result.get_h2h_probability(player_a.name, player_b.name)
        b_prob = 1.0 - a_prob

        res_a = result.player_results[player_a.name]
        res_b = result.player_results[player_b.name]

        return {
            "player_a": player_a.name,
            "player_b": player_b.name,
            "a_win_prob": round(a_prob, 4),
            "b_win_prob": round(b_prob, 4),
            "a_avg_score": round(res_a.avg_total_score, 2),
            "b_avg_score": round(res_b.avg_total_score, 2),
            "a_score_std": round(res_a.score_std, 2),
            "b_score_std": round(res_b.score_std, 2),
            "edge": round(abs(a_prob - 0.5) * 2, 4),
            "recommended_side": player_a.name if a_prob > 0.5 else player_b.name,
        }
