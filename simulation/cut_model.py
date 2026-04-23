"""Cut model — Cut line dynamics after round 2."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from simulation.config import SimulationConfig


class CutModel:
    """Model the cut after round 2."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def determine_cut_line(self, scores_after_r2: List[Tuple[str, int]]) -> Tuple[int, List[str]]:
        """Determine the cut line and who makes the cut.

        PGA Tour rule: Top 65 and ties make the cut.

        Args:
            scores_after_r2: List of (player_name, total_score_to_par) after 2 rounds.

        Returns:
            (cut_line_score, list_of_players_who_made_cut)
        """
        if not scores_after_r2:
            return 0, []

        # Sort by score (lowest = best in golf)
        sorted_scores = sorted(scores_after_r2, key=lambda x: x[1])

        if len(sorted_scores) <= self.config.cut_top_n:
            # Everyone makes the cut
            return sorted_scores[-1][1], [name for name, _ in sorted_scores]

        # Cut at position 65 (0-indexed: position 64)
        cut_idx = min(self.config.cut_top_n - 1, len(sorted_scores) - 1)
        cut_score = sorted_scores[cut_idx][1]

        if self.config.cut_plus_ties:
            # Include ties at the cut line
            made_cut = [name for name, score in sorted_scores if score <= cut_score]
        else:
            made_cut = [name for name, score in sorted_scores[:self.config.cut_top_n]]

        return cut_score, made_cut

    def projected_cut_line(
        self,
        field_sg_values: List[float],
        course_par: int = 72,
        weather_difficulty_r1: float = 0.0,
        weather_difficulty_r2: float = 0.0,
    ) -> float:
        """Project the expected cut line for a tournament.

        Used for pre-tournament cut probability estimation.

        Returns:
            Projected cut line as score relative to par (36 holes).
        """
        if not field_sg_values:
            # Default: E to +2 for a typical PGA Tour event
            return 1.0

        # The cut line depends on field strength and conditions
        field_arr = np.array(field_sg_values)

        # The 65th best player's expected 36-hole performance
        sorted_sg = np.sort(field_arr)[::-1]  # Best to worst
        if len(sorted_sg) > self.config.cut_top_n:
            cut_player_sg = sorted_sg[self.config.cut_top_n - 1]
        else:
            cut_player_sg = sorted_sg[-1]

        # Convert SG to score relative to par (36 holes)
        # SG per round * 2 rounds = total SG
        # Each SG = approximately 1 stroke better than par
        expected_cut_score = -(cut_player_sg * 2) + weather_difficulty_r1 + weather_difficulty_r2

        # Add variance: cut lines have ~1.5 stroke variance
        return round(expected_cut_score, 1)

    def make_cut_probability(
        self,
        player_sg: float,
        projected_cut_line: float,
        player_round_std: float = 2.75,
    ) -> float:
        """Estimate probability of making the cut.

        Uses normal distribution of 36-hole scoring.
        """
        from scipy import stats

        # Player's expected 36-hole score relative to par
        expected_36h = -(player_sg * 2)

        # 36-hole scoring variance
        # Variance of sum of 2 rounds (partially correlated)
        round_var = player_round_std ** 2
        correlation = self.config.round_to_round_correlation
        total_var_36h = 2 * round_var * (1 + correlation)
        total_std_36h = np.sqrt(total_var_36h)

        # P(36h_score <= cut_line) = CDF at cut line
        prob = stats.norm.cdf(projected_cut_line, loc=expected_36h, scale=total_std_36h)

        return float(np.clip(prob, 0.01, 0.99))
