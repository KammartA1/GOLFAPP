"""
Golf Quant Engine — Player Skill Model
=======================================
Per-player SG-based skill model with variance, course fit, pressure,
form, and surface preferences.  Draws daily SG components for simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PlayerSGComponents:
    """A single draw of SG components for one round."""
    sg_ott: float = 0.0
    sg_app: float = 0.0
    sg_atg: float = 0.0
    sg_putt: float = 0.0

    @property
    def sg_total(self) -> float:
        return self.sg_ott + self.sg_app + self.sg_atg + self.sg_putt

    def to_dict(self) -> dict[str, float]:
        return {
            "sg_ott": self.sg_ott,
            "sg_app": self.sg_app,
            "sg_atg": self.sg_atg,
            "sg_putt": self.sg_putt,
            "sg_total": self.sg_total,
        }


@dataclass
class PlayerModel:
    """Full statistical model of a single golfer's ability.

    SG means are strokes-gained per round relative to tour average (0.0).
    Positive = better than average.  E.g., Scottie Scheffler ~ +2.0 total.
    Variance represents round-to-round consistency in each component.
    """

    name: str

    # --- SG component means (per round) ---
    sg_ott_mean: float = 0.0
    sg_app_mean: float = 0.0
    sg_atg_mean: float = 0.0
    sg_putt_mean: float = 0.0

    # --- SG component standard deviations (round-to-round) ---
    sg_ott_std: float = 0.60
    sg_app_std: float = 0.80
    sg_atg_std: float = 0.55
    sg_putt_std: float = 0.90

    # --- Volatility (overall boom-or-bust multiplier on std) ---
    volatility: float = 1.0  # >1 = boom/bust, <1 = consistent

    # --- Course history fit score ---
    # 0.0 = no fit data, positive = historically good at this course
    course_fit: float = 0.0

    # --- Pressure coefficient (-1 to +1) ---
    # +1.0 = elite closer (thrives under pressure)
    # -1.0 = severe choker (folds under pressure)
    #  0.0 = neutral
    pressure_coeff: float = 0.0

    # --- Recent form adjustment (SG shift from baseline) ---
    recent_form: float = 0.0

    # --- Surface preference ---
    # Positive = prefers bermuda, negative = prefers bentgrass
    # Typical range: -0.3 to +0.3
    bermuda_preference: float = 0.0

    # --- Experience flags ---
    career_wins: int = 0
    major_wins: int = 0
    is_first_time_contender: bool = True

    # --- Latent daily form variance ---
    # Models "good day / bad day" — a hidden variable that shifts ALL SG
    # components together on a given day
    daily_form_std: float = 0.40

    # --- Internal tracking ---
    player_id: Optional[int] = None

    @property
    def sg_total_mean(self) -> float:
        """Expected strokes gained per round (baseline + form + fit)."""
        return (
            self.sg_ott_mean
            + self.sg_app_mean
            + self.sg_atg_mean
            + self.sg_putt_mean
            + self.recent_form
            + self.course_fit
        )

    @property
    def overall_std(self) -> float:
        """Approximate total SG standard deviation per round."""
        raw_var = (
            self.sg_ott_std ** 2
            + self.sg_app_std ** 2
            + self.sg_atg_std ** 2
            + self.sg_putt_std ** 2
            + self.daily_form_std ** 2
        )
        return np.sqrt(raw_var) * self.volatility

    def sample_round_sg_components(
        self,
        rng: np.random.Generator,
        surface_bermuda: bool = False,
    ) -> PlayerSGComponents:
        """Draw SG components for one round from this player's distributions.

        Steps:
          1. Draw a latent 'daily form' variable that shifts all components
          2. Draw each SG component independently
          3. Apply volatility multiplier to the deviations
          4. Apply surface preference
          5. Apply recent form (already baked into means but form is additive)

        Parameters
        ----------
        rng : np.random.Generator
            Seeded random generator for reproducibility.
        surface_bermuda : bool
            Whether the course has bermuda greens.

        Returns
        -------
        PlayerSGComponents with drawn values for this round.
        """
        # Step 1: latent daily form (good day / bad day)
        daily_form = rng.normal(0.0, self.daily_form_std)

        # Step 2 & 3: draw each component, scale deviation by volatility
        def _draw(mean: float, std: float) -> float:
            deviation = rng.normal(0.0, std) * self.volatility
            return mean + deviation + daily_form * 0.25  # form partially shared

        sg_ott = _draw(self.sg_ott_mean, self.sg_ott_std)
        sg_app = _draw(self.sg_app_mean, self.sg_app_std)
        sg_atg = _draw(self.sg_atg_mean, self.sg_atg_std)
        sg_putt = _draw(self.sg_putt_mean, self.sg_putt_std)

        # Step 4: surface preference — adjust putting on bermuda vs bentgrass
        if surface_bermuda:
            sg_putt += self.bermuda_preference
        else:
            sg_putt -= self.bermuda_preference

        # Step 5: recent form is a shift to overall daily performance
        form_share = self.recent_form * 0.25  # distribute across components
        sg_ott += form_share
        sg_app += form_share
        sg_atg += form_share
        sg_putt += form_share

        return PlayerSGComponents(
            sg_ott=sg_ott,
            sg_app=sg_app,
            sg_atg=sg_atg,
            sg_putt=sg_putt,
        )

    def sample_round_sg_components_batch(
        self,
        rng: np.random.Generator,
        n: int,
        surface_bermuda: bool = False,
    ) -> np.ndarray:
        """Vectorized batch draw of SG components for N rounds.

        Returns shape (n, 4) array: columns [ott, app, atg, putt].
        """
        daily_form = rng.normal(0.0, self.daily_form_std, size=n)

        means = np.array([
            self.sg_ott_mean, self.sg_app_mean,
            self.sg_atg_mean, self.sg_putt_mean,
        ])
        stds = np.array([
            self.sg_ott_std, self.sg_app_std,
            self.sg_atg_std, self.sg_putt_std,
        ])

        # Draw deviations: (n, 4)
        deviations = rng.normal(0.0, 1.0, size=(n, 4)) * stds * self.volatility

        # Build result
        result = means + deviations + (daily_form[:, np.newaxis] * 0.25)

        # Surface preference on putting (column 3)
        if surface_bermuda:
            result[:, 3] += self.bermuda_preference
        else:
            result[:, 3] -= self.bermuda_preference

        # Recent form distributed across components
        form_share = self.recent_form * 0.25
        result += form_share

        return result

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sg_ott_mean": self.sg_ott_mean,
            "sg_app_mean": self.sg_app_mean,
            "sg_atg_mean": self.sg_atg_mean,
            "sg_putt_mean": self.sg_putt_mean,
            "sg_total_mean": self.sg_total_mean,
            "volatility": self.volatility,
            "pressure_coeff": self.pressure_coeff,
            "course_fit": self.course_fit,
            "recent_form": self.recent_form,
        }

    def __repr__(self) -> str:
        return (
            f"<PlayerModel name={self.name!r} "
            f"sg_total={self.sg_total_mean:+.2f} vol={self.volatility:.2f}>"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_tour_average_player(name: str = "Tour Average") -> PlayerModel:
    """Create a perfectly average PGA Tour player (all SG = 0)."""
    return PlayerModel(name=name)


def create_elite_player(
    name: str,
    sg_total: float = 2.0,
    profile: str = "balanced",
) -> PlayerModel:
    """Create an elite player with a given SG total and style profile.

    Profiles:
      - balanced: even distribution matching default weights
      - bomber: heavy OTT advantage
      - iron_player: heavy approach advantage
      - scrambler: heavy around-the-green advantage
      - putter: heavy putting advantage (high variance)
    """
    profiles = {
        "balanced": (0.25, 0.38, 0.22, 0.15),
        "bomber": (0.45, 0.25, 0.15, 0.15),
        "iron_player": (0.15, 0.55, 0.15, 0.15),
        "scrambler": (0.15, 0.25, 0.45, 0.15),
        "putter": (0.15, 0.25, 0.15, 0.45),
    }
    weights = profiles.get(profile, profiles["balanced"])

    return PlayerModel(
        name=name,
        sg_ott_mean=sg_total * weights[0],
        sg_app_mean=sg_total * weights[1],
        sg_atg_mean=sg_total * weights[2],
        sg_putt_mean=sg_total * weights[3],
        volatility=0.85 if profile == "putter" else 1.0,
        career_wins=5 if sg_total > 1.5 else 1,
        is_first_time_contender=sg_total < 1.0,
    )


# ---------------------------------------------------------------------------
# Default field of well-known players for quick testing
# ---------------------------------------------------------------------------

DEFAULT_PLAYER_MODELS: dict[str, PlayerModel] = {
    "Scottie Scheffler": PlayerModel(
        name="Scottie Scheffler",
        sg_ott_mean=0.65, sg_app_mean=1.10, sg_atg_mean=0.30, sg_putt_mean=0.15,
        sg_ott_std=0.50, sg_app_std=0.65, sg_atg_std=0.50, sg_putt_std=0.85,
        volatility=0.90, pressure_coeff=0.70, career_wins=14, major_wins=2,
        is_first_time_contender=False, bermuda_preference=0.05,
    ),
    "Rory McIlroy": PlayerModel(
        name="Rory McIlroy",
        sg_ott_mean=0.80, sg_app_mean=0.70, sg_atg_mean=0.10, sg_putt_mean=-0.05,
        sg_ott_std=0.45, sg_app_std=0.70, sg_atg_std=0.55, sg_putt_std=0.95,
        volatility=1.05, pressure_coeff=0.20, career_wins=25, major_wins=4,
        is_first_time_contender=False, bermuda_preference=-0.10,
    ),
    "Jon Rahm": PlayerModel(
        name="Jon Rahm",
        sg_ott_mean=0.55, sg_app_mean=0.85, sg_atg_mean=0.25, sg_putt_mean=0.20,
        sg_ott_std=0.50, sg_app_std=0.70, sg_atg_std=0.50, sg_putt_std=0.80,
        volatility=0.95, pressure_coeff=0.55, career_wins=12, major_wins=2,
        is_first_time_contender=False, bermuda_preference=0.0,
    ),
    "Xander Schauffele": PlayerModel(
        name="Xander Schauffele",
        sg_ott_mean=0.50, sg_app_mean=0.65, sg_atg_mean=0.30, sg_putt_mean=0.25,
        sg_ott_std=0.50, sg_app_std=0.65, sg_atg_std=0.50, sg_putt_std=0.80,
        volatility=0.88, pressure_coeff=0.45, career_wins=10, major_wins=2,
        is_first_time_contender=False, bermuda_preference=0.0,
    ),
    "Collin Morikawa": PlayerModel(
        name="Collin Morikawa",
        sg_ott_mean=0.20, sg_app_mean=1.00, sg_atg_mean=0.10, sg_putt_mean=-0.10,
        sg_ott_std=0.55, sg_app_std=0.60, sg_atg_std=0.55, sg_putt_std=0.90,
        volatility=0.85, pressure_coeff=0.40, career_wins=7, major_wins=2,
        is_first_time_contender=False, bermuda_preference=-0.05,
    ),
    "Viktor Hovland": PlayerModel(
        name="Viktor Hovland",
        sg_ott_mean=0.45, sg_app_mean=0.70, sg_atg_mean=-0.15, sg_putt_mean=0.10,
        sg_ott_std=0.50, sg_app_std=0.70, sg_atg_std=0.60, sg_putt_std=0.90,
        volatility=1.10, pressure_coeff=0.15, career_wins=6, major_wins=0,
        is_first_time_contender=True, bermuda_preference=-0.10,
    ),
    "Patrick Cantlay": PlayerModel(
        name="Patrick Cantlay",
        sg_ott_mean=0.25, sg_app_mean=0.60, sg_atg_mean=0.20, sg_putt_mean=0.45,
        sg_ott_std=0.50, sg_app_std=0.65, sg_atg_std=0.50, sg_putt_std=0.85,
        volatility=0.80, pressure_coeff=0.50, career_wins=8, major_wins=0,
        is_first_time_contender=False, bermuda_preference=0.05,
    ),
    "Wyndham Clark": PlayerModel(
        name="Wyndham Clark",
        sg_ott_mean=0.55, sg_app_mean=0.45, sg_atg_mean=0.05, sg_putt_mean=0.15,
        sg_ott_std=0.55, sg_app_std=0.75, sg_atg_std=0.55, sg_putt_std=0.90,
        volatility=1.15, pressure_coeff=0.10, career_wins=3, major_wins=1,
        is_first_time_contender=False, bermuda_preference=0.0,
    ),
    "Tour Average": PlayerModel(
        name="Tour Average",
        sg_ott_mean=0.0, sg_app_mean=0.0, sg_atg_mean=0.0, sg_putt_mean=0.0,
        volatility=1.0, pressure_coeff=0.0, career_wins=0,
        is_first_time_contender=True,
    ),
}
