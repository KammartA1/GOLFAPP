"""
Golf Quant Engine — Cut Model
===============================
Models the 36-hole cut line, bubble behavior, and MDF scenarios.
After round 2, typically top 65 + ties make the cut.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class CutResult:
    """Result of cut calculation for one simulation."""
    cut_line: float             # Score relative to par at the cut
    players_making_cut: int     # Number of players who made the cut
    players_on_bubble: int      # Players within 1 stroke of cut
    mdf_players: int            # Made-cut-didn't-finish (if >78 make cut)

    @property
    def had_mdf(self) -> bool:
        return self.mdf_players > 0


@dataclass
class CutProbabilities:
    """Per-player cut probabilities from simulation."""
    player_name: str
    make_cut_prob: float        # P(makes the cut)
    miss_cut_prob: float        # P(misses the cut)
    mdf_prob: float             # P(makes cut but MDF'd)
    bubble_prob: float          # P(within 1 stroke of cut line)
    expected_cut_position: float  # Expected position at cut time

    @property
    def effective_make_cut_prob(self) -> float:
        """Probability of making cut AND not being MDF'd."""
        return self.make_cut_prob - self.mdf_prob


# Historical cut line distributions by course difficulty
# (mean score relative to par, std dev)
HISTORICAL_CUT_LINES: dict[str, Tuple[float, float]] = {
    "Augusta National": (3.0, 1.8),
    "TPC Sawgrass": (1.0, 1.5),
    "Pebble Beach": (2.0, 2.0),
    "St Andrews Old Course": (-1.0, 2.5),
    "TPC Scottsdale": (-4.0, 1.5),
    "Pinehurst No. 2": (4.0, 2.0),
    "Torrey Pines South": (1.0, 1.8),
    "Riviera CC": (0.0, 1.5),
    "Harbour Town": (-2.0, 1.8),
    "Bay Hill": (2.0, 1.8),
    # Default for unknown courses
    "default": (1.0, 2.0),
}


class CutModel:
    """Model cut line dynamics and their effect on player behavior.

    Key mechanics:
    1. After round 2, project cut line from field scoring
    2. Top 65 + ties make the cut (standard PGA Tour)
    3. If more than 78 make the cut, MDF applies (65th place finishers + ties
       make cut but only top 65 are officially "made cut" for ranking points)
    4. Players on the bubble (within 1-2 shots) behave differently:
       aggressive play to ensure making it
    5. Historical cut line distributions by course/field strength
    """

    def __init__(
        self,
        cut_top_n: int = 65,
        cut_ties: bool = True,
        mdf_enabled: bool = True,
        mdf_threshold: int = 78,
    ):
        self.cut_top_n = cut_top_n
        self.cut_ties = cut_ties
        self.mdf_enabled = mdf_enabled
        self.mdf_threshold = mdf_threshold

    def calculate_cut_line(
        self,
        scores_after_r2: np.ndarray,
    ) -> CutResult:
        """Determine the cut line from 36-hole scores.

        Parameters
        ----------
        scores_after_r2 : np.ndarray
            Array of total scores (relative to par) after 2 rounds, one per player.

        Returns
        -------
        CutResult with cut line details.
        """
        sorted_scores = np.sort(scores_after_r2)
        n_players = len(sorted_scores)

        if n_players <= self.cut_top_n:
            # Entire field makes the cut
            return CutResult(
                cut_line=sorted_scores[-1] if n_players > 0 else 0.0,
                players_making_cut=n_players,
                players_on_bubble=0,
                mdf_players=0,
            )

        # Cut line is the score at position cut_top_n (0-indexed: cut_top_n - 1)
        cut_line_score = sorted_scores[self.cut_top_n - 1]

        if self.cut_ties:
            # Include all players tied at the cut line
            players_making = int(np.sum(scores_after_r2 <= cut_line_score))
        else:
            players_making = self.cut_top_n

        # Players on the bubble: within 1 stroke of cut
        bubble_players = int(np.sum(
            (scores_after_r2 >= cut_line_score - 1) &
            (scores_after_r2 <= cut_line_score + 1)
        ))

        # MDF: if too many make the cut
        mdf_count = 0
        if self.mdf_enabled and players_making > self.mdf_threshold:
            mdf_count = players_making - self.cut_top_n
            # Only top cut_top_n get full benefits; extras are MDF

        return CutResult(
            cut_line=float(cut_line_score),
            players_making_cut=players_making,
            players_on_bubble=bubble_players,
            mdf_players=mdf_count,
        )

    def get_players_making_cut(
        self,
        player_names: list[str],
        scores_after_r2: np.ndarray,
    ) -> Tuple[list[str], list[str], list[str]]:
        """Split field into made-cut, missed-cut, and MDF players.

        Parameters
        ----------
        player_names : list of str
        scores_after_r2 : np.ndarray
            Parallel array of scores.

        Returns
        -------
        Tuple of (made_cut, missed_cut, mdf) player name lists.
        """
        cut_result = self.calculate_cut_line(scores_after_r2)

        made_cut = []
        missed_cut = []
        mdf = []

        # Sort by score to determine MDF
        indices_by_score = np.argsort(scores_after_r2)

        for rank, idx in enumerate(indices_by_score):
            name = player_names[idx]
            score = scores_after_r2[idx]

            if score <= cut_result.cut_line:
                if self.mdf_enabled and rank >= self.cut_top_n and cut_result.had_mdf:
                    mdf.append(name)
                else:
                    made_cut.append(name)
            else:
                missed_cut.append(name)

        return made_cut, missed_cut, mdf

    def get_bubble_sg_adjustment(
        self,
        player_score: float,
        projected_cut_line: float,
    ) -> float:
        """Calculate SG adjustment for players near the bubble.

        Players on the bubble tend to play more aggressively (higher variance,
        slightly lower mean) as they try to make the cut.

        Parameters
        ----------
        player_score : float
            Player's current score relative to par.
        projected_cut_line : float
            Projected cut line.

        Returns
        -------
        float : SG mean adjustment (typically negative = slightly worse).
        """
        margin = projected_cut_line - player_score  # Positive = safely inside cut

        if margin > 3:
            # Comfortably inside: no adjustment
            return 0.0
        elif margin > 1:
            # Inside but not comfortable: slightly aggressive
            return -0.02
        elif margin >= -1:
            # On the bubble: most aggressive, slightly worse scoring
            return -0.05
        elif margin >= -3:
            # Likely missing: pressing, worst adjustment
            return -0.08
        else:
            # Well outside: already mentally checked out
            return -0.03

    def get_bubble_variance_multiplier(
        self,
        player_score: float,
        projected_cut_line: float,
    ) -> float:
        """Variance multiplier for bubble players.

        On the bubble, players take more risks -> higher variance.
        """
        margin = projected_cut_line - player_score

        if margin > 3:
            return 1.0
        elif margin > 1:
            return 1.05
        elif margin >= -1:
            return 1.15  # Maximum variance on the bubble
        elif margin >= -3:
            return 1.20  # Pressing hard
        return 1.0  # Well outside, normal play

    def project_cut_line(
        self,
        round1_scores: np.ndarray,
        field_strength_adjustment: float = 0.0,
    ) -> float:
        """Project the cut line after round 1.

        Useful for in-tournament live modeling. The cut line after R1 can
        be projected based on R1 scoring and historical patterns.

        Parameters
        ----------
        round1_scores : np.ndarray
            R1 scores relative to par.
        field_strength_adjustment : float
            Adjustment for field strength (weaker field = higher cut).

        Returns
        -------
        float : Projected cut line after R2.
        """
        # Sort and find the R1 cut-line-equivalent position
        sorted_r1 = np.sort(round1_scores)
        n = len(sorted_r1)
        cut_pos = min(self.cut_top_n - 1, n - 1)

        # R1 score at cut position
        r1_cut_score = sorted_r1[cut_pos]

        # Project R2: typically players near the cut regress slightly
        # and R2 scoring is slightly harder (less adrenaline)
        r2_projection = r1_cut_score * 1.05 + 0.5

        projected = r1_cut_score + r2_projection + field_strength_adjustment

        return projected

    def estimate_cut_probability(
        self,
        player_expected_score: float,
        player_score_std: float,
        projected_cut_line: float,
    ) -> float:
        """Estimate probability of a player making the cut.

        Uses normal CDF based on player's expected 36-hole score
        vs. projected cut line.

        Parameters
        ----------
        player_expected_score : float
            Player's expected 36-hole score (relative to par).
        player_score_std : float
            Standard deviation of player's 36-hole score.
        projected_cut_line : float
            Projected cut line.

        Returns
        -------
        float : Probability of making the cut (0-1).
        """
        if player_score_std <= 0:
            return 1.0 if player_expected_score <= projected_cut_line else 0.0

        # P(score <= cut_line) using normal CDF
        z = (projected_cut_line - player_expected_score) / player_score_std
        return float(stats.norm.cdf(z))

    def get_historical_cut_line(self, course_name: str) -> Tuple[float, float]:
        """Get historical cut line mean and std for a course.

        Returns (mean_cut_line, std_cut_line) relative to par.
        """
        return HISTORICAL_CUT_LINES.get(
            course_name,
            HISTORICAL_CUT_LINES["default"],
        )
