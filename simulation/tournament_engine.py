"""Tournament engine — Full 4-round tournament simulation with cut dynamics.

Monte Carlo simulation of N=10,000 tournaments to produce:
  - Win/top-5/top-10/top-20/make-cut probabilities
  - Finishing position distributions
  - Expected DFS scoring distributions
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from simulation.config import SimulationConfig
from simulation.round_engine import RoundEngine
from simulation.player_model import SimPlayer, PlayerModel
from simulation.course_model import CourseProfile, CourseModel
from simulation.weather_model import WeatherConditions
from simulation.wave_model import WaveConditions, WaveModel
from simulation.cut_model import CutModel

logger = logging.getLogger(__name__)


class TournamentSimulator:
    """Full tournament Monte Carlo simulator."""

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        seed = self.config.random_seed
        self.rng = np.random.default_rng(seed)
        self.round_engine = RoundEngine(self.config, self.rng)
        self.cut_model = CutModel(self.config)
        self.wave_model = WaveModel()
        self.course_model = CourseModel()

    def simulate_tournament(
        self,
        players: List[SimPlayer],
        course: CourseProfile,
        weather_by_round: Optional[Dict[int, WeatherConditions]] = None,
        wave_conditions: Optional[Dict[int, WaveConditions]] = None,
    ) -> pd.DataFrame:
        """Simulate a full tournament N times and aggregate results.

        Args:
            players: List of SimPlayer objects.
            course: CourseProfile for the venue.
            weather_by_round: Optional {round_num: WeatherConditions}.
            wave_conditions: Optional {round_num: WaveConditions}.

        Returns:
            DataFrame with columns:
                name, win_prob, top5_prob, top10_prob, top20_prob,
                make_cut_prob, avg_finish, median_finish, avg_score,
                avg_birdies, finish_distribution
        """
        if len(players) < self.config.min_field_size:
            logger.warning("Field size %d below minimum %d", len(players), self.config.min_field_size)

        n_sims = self.config.n_simulations
        n_players = len(players)

        logger.info("Simulating %d tournaments with %d players at %s",
                     n_sims, n_players, course.name)

        # Pre-allocate result arrays
        finish_positions = np.zeros((n_sims, n_players), dtype=np.int32)
        total_scores = np.zeros((n_sims, n_players), dtype=np.float64)
        made_cuts = np.zeros((n_sims, n_players), dtype=bool)

        for sim_idx in range(n_sims):
            sim_result = self._simulate_one_tournament(
                players, course, weather_by_round, wave_conditions
            )

            for p_idx, player in enumerate(players):
                pname = player.name
                if pname in sim_result:
                    finish_positions[sim_idx, p_idx] = sim_result[pname]["finish_position"]
                    total_scores[sim_idx, p_idx] = sim_result[pname]["total_score"]
                    made_cuts[sim_idx, p_idx] = sim_result[pname]["made_cut"]
                else:
                    finish_positions[sim_idx, p_idx] = n_players
                    total_scores[sim_idx, p_idx] = 999
                    made_cuts[sim_idx, p_idx] = False

        # Aggregate results
        results = []
        for p_idx, player in enumerate(players):
            finishes = finish_positions[:, p_idx]
            scores = total_scores[:, p_idx]
            cuts = made_cuts[:, p_idx]

            results.append({
                "name": player.name,
                "sg_total": player.sg_total,
                "win_prob": round(float(np.mean(finishes == 1)), 4),
                "top5_prob": round(float(np.mean(finishes <= 5)), 4),
                "top10_prob": round(float(np.mean(finishes <= 10)), 4),
                "top20_prob": round(float(np.mean(finishes <= 20)), 4),
                "make_cut_prob": round(float(np.mean(cuts)), 4),
                "avg_finish": round(float(np.mean(finishes)), 1),
                "median_finish": int(np.median(finishes)),
                "p10_finish": int(np.percentile(finishes, 10)),
                "p90_finish": int(np.percentile(finishes, 90)),
                "avg_score": round(float(np.mean(scores[cuts])), 2) if cuts.any() else 0.0,
                "std_score": round(float(np.std(scores[cuts])), 2) if cuts.any() else 0.0,
            })

        df = pd.DataFrame(results)
        df = df.sort_values("avg_finish").reset_index(drop=True)
        df["sim_rank"] = range(1, len(df) + 1)

        logger.info("Simulation complete. Leader: %s (win_prob=%.2f%%)",
                     df.iloc[0]["name"], df.iloc[0]["win_prob"] * 100)

        return df

    def _simulate_one_tournament(
        self,
        players: List[SimPlayer],
        course: CourseProfile,
        weather_by_round: Optional[Dict[int, WeatherConditions]],
        wave_conditions: Optional[Dict[int, WaveConditions]],
    ) -> Dict[str, dict]:
        """Simulate a single tournament instance."""
        n_players = len(players)
        player_data = {}

        for player in players:
            player_data[player.name] = {
                "player": player,
                "round_scores": [],
                "total_score": 0,
                "made_cut": True,
                "finish_position": n_players,
                "prev_form": None,  # Track previous round form for C8 correlation
            }

        # Rounds 1-2: everyone plays
        for round_num in [1, 2]:
            weather = weather_by_round.get(round_num) if weather_by_round else None

            # Wave adjustments for rounds 1-2
            wave_adj = {}
            if wave_conditions and round_num in wave_conditions:
                for player in players:
                    wave = player.wave
                    if round_num == 2:
                        wave = "PM" if wave == "AM" else "AM" if wave == "PM" else wave
                    wave_adj[player.name] = self.wave_model.wave_advantage(
                        wave_conditions[round_num], wave
                    )

            for player in players:
                result = self.round_engine.simulate_round(
                    player=player,
                    course=course,
                    round_num=round_num,
                    weather=weather,
                    current_position=75,  # Approximate for early rounds
                    field_size=n_players,
                    wave_adjustment=wave_adj.get(player.name, 0.0),
                    prev_form=player_data[player.name]["prev_form"],
                )
                player_data[player.name]["round_scores"].append(result["total_score"])
                player_data[player.name]["total_score"] += result["total_score"]
                player_data[player.name]["prev_form"] = result["daily_form"]

        # Apply cut
        scores_after_r2 = [
            (name, data["total_score"])
            for name, data in player_data.items()
        ]
        cut_line, made_cut_names = self.cut_model.determine_cut_line(scores_after_r2)
        made_cut_set = set(made_cut_names)

        for name, data in player_data.items():
            data["made_cut"] = name in made_cut_set

        # Rounds 3-4: only players who made cut
        active_players = [p for p in players if player_data[p.name]["made_cut"]]

        for round_num in [3, 4]:
            weather = weather_by_round.get(round_num) if weather_by_round else None

            # Compute leaderboard positions for pressure
            current_scores = [(p.name, player_data[p.name]["total_score"]) for p in active_players]
            current_scores.sort(key=lambda x: x[1])
            position_map = {name: pos + 1 for pos, (name, _) in enumerate(current_scores)}
            lead_score = current_scores[0][1] if current_scores else 0

            for player in active_players:
                name = player.name
                pos = position_map.get(name, 75)
                off_lead = player_data[name]["total_score"] - lead_score

                result = self.round_engine.simulate_round(
                    player=player,
                    course=course,
                    round_num=round_num,
                    weather=weather,
                    current_position=pos,
                    strokes_off_lead=off_lead,
                    field_size=len(active_players),
                    prev_form=player_data[name]["prev_form"],
                )
                player_data[name]["round_scores"].append(result["total_score"])
                player_data[name]["total_score"] += result["total_score"]
                player_data[name]["prev_form"] = result["daily_form"]

        # Determine final positions
        final_scores = []
        for name, data in player_data.items():
            if data["made_cut"]:
                final_scores.append((name, data["total_score"]))

        final_scores.sort(key=lambda x: x[1])

        # Assign positions (handling ties)
        pos = 1
        i = 0
        while i < len(final_scores):
            j = i
            while j < len(final_scores) and final_scores[j][1] == final_scores[i][1]:
                j += 1
            # All tied players get the same position
            for k in range(i, j):
                player_data[final_scores[k][0]]["finish_position"] = pos
            pos = j + 1
            i = j

        # Players who missed cut get position = field size
        for name, data in player_data.items():
            if not data["made_cut"]:
                data["finish_position"] = n_players

        return player_data
