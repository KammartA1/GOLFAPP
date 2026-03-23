"""
Golf Quant Engine — Course Model
=================================
Hole-by-hole course specification with SG component weights, wind exposure,
green complexity, and scoring distribution calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from simulation.config import SimulationConfig, ScoringRates


@dataclass
class HoleSpec:
    """Specification for a single hole on the course."""
    number: int
    par: int
    yardage: int
    difficulty_rating: float  # 0.0 (easiest) to 1.0 (hardest)
    scoring_average: float    # Historical scoring avg (e.g., 4.15 for a par 4)

    # SG component weights for this hole (must sum to 1.0)
    sg_weight_ott: float = 0.25
    sg_weight_app: float = 0.40
    sg_weight_atg: float = 0.20
    sg_weight_putt: float = 0.15

    # Environmental exposure
    wind_exposure: float = 0.5       # 0.0 (sheltered) to 1.0 (fully exposed)
    hole_orientation_deg: float = 0.0  # Compass heading tee-to-green (0=N, 90=E)
    elevation_change_ft: float = 0.0   # Positive = uphill, negative = downhill

    # Green characteristics
    green_complexity: float = 0.5    # 0.0 (flat/simple) to 1.0 (severe slopes)
    green_speed_stimp: float = 12.0  # Stimpmeter reading

    # Fairway
    fairway_width_yards: float = 30.0  # Average width in landing zone

    @property
    def difficulty_tier(self) -> int:
        """Convert continuous difficulty to tier: 0=easy, 1=medium, 2=hard."""
        if self.difficulty_rating < 0.33:
            return 0
        elif self.difficulty_rating < 0.67:
            return 1
        return 2

    @property
    def sg_weights(self) -> dict[str, float]:
        return {
            "sg_ott": self.sg_weight_ott,
            "sg_app": self.sg_weight_app,
            "sg_atg": self.sg_weight_atg,
            "sg_putt": self.sg_weight_putt,
        }


def _default_sg_weights_for_par(par: int) -> tuple[float, float, float, float]:
    """Return (ott, app, atg, putt) defaults based on par value."""
    if par == 3:
        return (0.0, 0.50, 0.25, 0.25)
    elif par == 5:
        return (0.35, 0.30, 0.15, 0.20)
    else:  # par 4
        return (0.25, 0.40, 0.18, 0.17)


@dataclass
class CourseModel:
    """Full course specification for simulation."""

    name: str
    holes: List[HoleSpec]
    course_type: str = "parkland"       # links, parkland, desert, etc.
    bermuda_greens: bool = False
    elevation_ft: float = 0.0
    wind_sensitivity: float = 0.50
    distance_bonus: float = 0.50
    accuracy_penalty: float = 0.50

    # Course-level SG weights (override default if provided)
    course_sg_weights: Optional[Dict[str, float]] = None

    @property
    def par(self) -> int:
        return sum(h.par for h in self.holes)

    @property
    def total_yardage(self) -> int:
        return sum(h.yardage for h in self.holes)

    @property
    def avg_difficulty(self) -> float:
        return np.mean([h.difficulty_rating for h in self.holes])

    def get_hole(self, number: int) -> HoleSpec:
        """Get hole by number (1-indexed)."""
        for h in self.holes:
            if h.number == number:
                return h
        raise ValueError(f"Hole {number} not found on {self.name}")

    def get_hole_scoring_distribution(
        self,
        player_sg_total: float,
        hole: HoleSpec,
        weather_penalty: float = 0.0,
        config: Optional[SimulationConfig] = None,
    ) -> np.ndarray:
        """Compute discrete outcome probabilities for a player on a hole.

        Returns array of 5 probabilities: [eagle, birdie, par, bogey, double+].
        Adjusted by player skill and weather conditions.

        Parameters
        ----------
        player_sg_total : float
            Player's hole-level SG (weighted by hole's SG component weights).
        hole : HoleSpec
            The hole specification.
        weather_penalty : float
            Additional strokes of difficulty from weather (positive = harder).
        config : SimulationConfig, optional
            Configuration with base scoring rates.
        """
        if config is None:
            config = SimulationConfig()

        # Get base rates for this par/difficulty
        base_rates = np.array(
            config.scoring_rates.get_rates(hole.par, hole.difficulty_tier),
            dtype=np.float64,
        )

        # Net SG effect: positive SG = better scoring, weather penalty = worse
        net_sg = player_sg_total - weather_penalty

        # Shift probabilities based on net SG
        # Each +0.1 SG shifts ~2% probability toward better outcomes
        shift_factor = net_sg * 0.20

        if shift_factor > 0:
            # Better player: shift probability mass from bogey/double toward birdie/eagle
            # Eagle gets small share, birdie gets most
            eagle_gain = shift_factor * 0.08
            birdie_gain = shift_factor * 0.55
            par_change = shift_factor * 0.05
            bogey_loss = shift_factor * 0.45
            double_loss = shift_factor * 0.23

            adjusted = np.array([
                base_rates[0] + eagle_gain,
                base_rates[1] + birdie_gain,
                base_rates[2] + par_change,
                base_rates[3] - bogey_loss,
                base_rates[4] - double_loss,
            ])
        else:
            # Worse player: shift from birdie/eagle toward bogey/double
            abs_shift = abs(shift_factor)
            eagle_loss = abs_shift * 0.05
            birdie_loss = abs_shift * 0.45
            par_change = -abs_shift * 0.05
            bogey_gain = abs_shift * 0.38
            double_gain = abs_shift * 0.17

            adjusted = np.array([
                base_rates[0] - eagle_loss,
                base_rates[1] - birdie_loss,
                base_rates[2] + par_change,
                base_rates[3] + bogey_gain,
                base_rates[4] + double_gain,
            ])

        # Green complexity amplifies putting variance
        if hole.green_complexity > 0.7:
            # Complex greens slightly increase bogey risk, decrease birdie rate
            complexity_effect = (hole.green_complexity - 0.5) * 0.03
            adjusted[1] -= complexity_effect
            adjusted[3] += complexity_effect * 0.7
            adjusted[4] += complexity_effect * 0.3

        # Narrow fairways on par 4/5 increase bogey risk
        if hole.par >= 4 and hole.fairway_width_yards < 25:
            narrow_effect = (25 - hole.fairway_width_yards) / 25 * 0.02
            adjusted[1] -= narrow_effect * 0.5
            adjusted[3] += narrow_effect * 0.7
            adjusted[4] += narrow_effect * 0.3

        # Clamp to valid probabilities and renormalize
        adjusted = np.maximum(adjusted, 0.001)
        adjusted /= adjusted.sum()

        return adjusted


# ---------------------------------------------------------------------------
# Default course profiles
# ---------------------------------------------------------------------------

def _generate_holes(
    pars: list[int],
    yardages: list[int],
    difficulties: list[float],
    scoring_avgs: list[float],
    wind_exposures: Optional[list[float]] = None,
    orientations: Optional[list[float]] = None,
    green_complexities: Optional[list[float]] = None,
    fairway_widths: Optional[list[float]] = None,
) -> list[HoleSpec]:
    """Helper to generate a list of HoleSpecs from parallel arrays."""
    n = len(pars)
    if wind_exposures is None:
        wind_exposures = [0.5] * n
    if orientations is None:
        orientations = [float(i * 20 % 360) for i in range(n)]
    if green_complexities is None:
        green_complexities = [0.5] * n
    if fairway_widths is None:
        fairway_widths = [30.0] * n

    holes = []
    for i in range(n):
        ott, app, atg, putt = _default_sg_weights_for_par(pars[i])
        holes.append(HoleSpec(
            number=i + 1,
            par=pars[i],
            yardage=yardages[i],
            difficulty_rating=difficulties[i],
            scoring_average=scoring_avgs[i],
            sg_weight_ott=ott,
            sg_weight_app=app,
            sg_weight_atg=atg,
            sg_weight_putt=putt,
            wind_exposure=wind_exposures[i],
            hole_orientation_deg=orientations[i],
            green_complexity=green_complexities[i],
            fairway_width_yards=fairway_widths[i],
        ))
    return holes


# Pre-built course models for major PGA Tour venues

AUGUSTA_NATIONAL = CourseModel(
    name="Augusta National",
    course_type="parkland",
    bermuda_greens=True,
    elevation_ft=330,
    wind_sensitivity=0.45,
    distance_bonus=0.65,
    accuracy_penalty=0.30,
    course_sg_weights={"sg_ott": 0.20, "sg_app": 0.42, "sg_atg": 0.22, "sg_putt": 0.16},
    holes=_generate_holes(
        pars=       [4, 5, 4, 3, 4, 3, 4, 5, 4, 4, 4, 3, 5, 4, 5, 3, 4, 4],
        yardages=   [445, 575, 350, 240, 495, 180, 450, 570, 460, 495, 520, 155, 510, 440, 530, 170, 440, 465],
        difficulties=[0.45, 0.30, 0.35, 0.55, 0.50, 0.40, 0.55, 0.35, 0.50, 0.65, 0.75, 0.85, 0.40, 0.50, 0.35, 0.60, 0.55, 0.70],
        scoring_avgs=[4.10, 4.70, 3.95, 3.25, 4.15, 3.10, 4.20, 4.85, 4.15, 4.30, 4.40, 3.35, 4.80, 4.15, 4.90, 3.15, 4.25, 4.35],
        wind_exposures=[0.3, 0.4, 0.3, 0.5, 0.4, 0.5, 0.6, 0.3, 0.4, 0.5, 0.7, 0.8, 0.6, 0.4, 0.5, 0.6, 0.4, 0.5],
        green_complexities=[0.7, 0.6, 0.5, 0.8, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.6, 0.9, 0.5, 0.7, 0.6, 0.8, 0.6, 0.7],
        fairway_widths=[35, 30, 28, 0, 28, 0, 25, 32, 30, 27, 25, 0, 30, 28, 35, 0, 27, 30],
    ),
)

TPC_SAWGRASS = CourseModel(
    name="TPC Sawgrass",
    course_type="parkland",
    bermuda_greens=True,
    elevation_ft=20,
    wind_sensitivity=0.70,
    distance_bonus=0.35,
    accuracy_penalty=0.60,
    course_sg_weights={"sg_ott": 0.18, "sg_app": 0.40, "sg_atg": 0.25, "sg_putt": 0.17},
    holes=_generate_holes(
        pars=       [4, 5, 3, 4, 4, 4, 4, 3, 5, 4, 5, 4, 3, 4, 4, 5, 3, 4],
        yardages=   [423, 532, 177, 384, 466, 393, 442, 219, 583, 424, 542, 358, 181, 467, 449, 523, 137, 462],
        difficulties=[0.40, 0.35, 0.45, 0.50, 0.55, 0.60, 0.55, 0.50, 0.40, 0.50, 0.35, 0.65, 0.45, 0.55, 0.50, 0.40, 0.90, 0.70],
        scoring_avgs=[4.05, 4.70, 3.10, 4.15, 4.20, 4.25, 4.15, 3.10, 4.75, 4.15, 4.65, 4.30, 3.10, 4.20, 4.15, 4.70, 3.25, 4.35],
        wind_exposures=[0.5, 0.6, 0.4, 0.5, 0.7, 0.6, 0.5, 0.7, 0.6, 0.5, 0.4, 0.6, 0.5, 0.7, 0.6, 0.5, 0.9, 0.8],
        green_complexities=[0.5, 0.5, 0.6, 0.6, 0.7, 0.5, 0.6, 0.7, 0.5, 0.6, 0.5, 0.8, 0.6, 0.6, 0.5, 0.5, 0.95, 0.7],
        fairway_widths=[28, 30, 0, 25, 26, 24, 28, 0, 32, 27, 30, 22, 0, 26, 28, 30, 0, 25],
    ),
)

PEBBLE_BEACH = CourseModel(
    name="Pebble Beach",
    course_type="links_coastal",
    bermuda_greens=False,
    elevation_ft=50,
    wind_sensitivity=0.85,
    distance_bonus=0.40,
    accuracy_penalty=0.65,
    course_sg_weights={"sg_ott": 0.20, "sg_app": 0.38, "sg_atg": 0.20, "sg_putt": 0.22},
    holes=_generate_holes(
        pars=       [4, 5, 4, 4, 3, 5, 3, 4, 4, 4, 4, 3, 4, 5, 4, 3, 4, 5],
        yardages=   [381, 502, 404, 331, 195, 523, 106, 428, 505, 446, 390, 202, 445, 580, 397, 178, 178, 543],
        difficulties=[0.35, 0.40, 0.50, 0.45, 0.50, 0.55, 0.65, 0.80, 0.75, 0.70, 0.55, 0.50, 0.60, 0.45, 0.55, 0.50, 0.60, 0.65],
        scoring_avgs=[4.00, 4.75, 4.10, 4.05, 3.15, 4.80, 3.20, 4.40, 4.40, 4.30, 4.15, 3.10, 4.25, 4.80, 4.15, 3.10, 4.20, 4.90],
        wind_exposures=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.95, 0.9, 0.85, 0.6, 0.5, 0.5, 0.6, 0.5, 0.5, 0.6, 0.8],
        green_complexities=[0.5, 0.5, 0.6, 0.7, 0.6, 0.5, 0.8, 0.7, 0.6, 0.7, 0.5, 0.6, 0.6, 0.5, 0.5, 0.6, 0.6, 0.6],
        fairway_widths=[30, 28, 25, 28, 0, 32, 0, 22, 25, 24, 28, 0, 26, 30, 28, 0, 30, 28],
    ),
)

ST_ANDREWS = CourseModel(
    name="St Andrews Old Course",
    course_type="links",
    bermuda_greens=False,
    elevation_ft=5,
    wind_sensitivity=0.90,
    distance_bonus=0.80,
    accuracy_penalty=0.25,
    course_sg_weights={"sg_ott": 0.28, "sg_app": 0.30, "sg_atg": 0.22, "sg_putt": 0.20},
    holes=_generate_holes(
        pars=       [4, 4, 4, 4, 5, 4, 4, 3, 4, 4, 3, 4, 4, 5, 4, 4, 4, 4],
        yardages=   [376, 453, 397, 480, 570, 412, 371, 188, 352, 386, 174, 348, 465, 618, 456, 423, 495, 357],
        difficulties=[0.25, 0.50, 0.40, 0.55, 0.35, 0.50, 0.45, 0.40, 0.35, 0.50, 0.65, 0.60, 0.55, 0.30, 0.45, 0.50, 0.75, 0.20],
        scoring_avgs=[3.90, 4.15, 4.05, 4.20, 4.70, 4.15, 4.10, 3.10, 4.00, 4.15, 3.20, 4.20, 4.25, 4.75, 4.10, 4.15, 4.40, 3.85],
        wind_exposures=[0.7, 0.8, 0.7, 0.9, 0.85, 0.8, 0.75, 0.9, 0.7, 0.8, 0.95, 0.85, 0.8, 0.9, 0.75, 0.8, 0.85, 0.5],
        green_complexities=[0.7, 0.8, 0.7, 0.6, 0.7, 0.8, 0.6, 0.7, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.7, 0.6, 0.8, 0.4],
        fairway_widths=[45, 40, 50, 35, 40, 38, 45, 0, 42, 35, 0, 30, 35, 45, 38, 35, 30, 85],
    ),
)

TPC_SCOTTSDALE = CourseModel(
    name="TPC Scottsdale",
    course_type="desert",
    bermuda_greens=True,
    elevation_ft=1500,
    wind_sensitivity=0.30,
    distance_bonus=0.75,
    accuracy_penalty=0.25,
    course_sg_weights={"sg_ott": 0.22, "sg_app": 0.36, "sg_atg": 0.20, "sg_putt": 0.22},
    holes=_generate_holes(
        pars=       [4, 4, 5, 4, 4, 3, 3, 5, 4, 4, 4, 3, 5, 4, 5, 3, 4, 4],
        yardages=   [424, 442, 573, 422, 453, 195, 215, 515, 414, 464, 469, 178, 584, 389, 553, 162, 332, 447],
        difficulties=[0.30, 0.40, 0.25, 0.45, 0.50, 0.40, 0.55, 0.20, 0.45, 0.55, 0.60, 0.35, 0.25, 0.40, 0.30, 0.50, 0.35, 0.55],
        scoring_avgs=[3.95, 4.05, 4.60, 4.10, 4.15, 3.05, 3.15, 4.55, 4.10, 4.20, 4.25, 3.00, 4.60, 4.05, 4.65, 3.10, 3.95, 4.20],
        wind_exposures=[0.3, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.3, 0.4, 0.3, 0.3, 0.2, 0.3, 0.2, 0.4, 0.2, 0.4],
        green_complexities=[0.4, 0.5, 0.4, 0.5, 0.6, 0.5, 0.5, 0.4, 0.5, 0.6, 0.5, 0.4, 0.4, 0.5, 0.4, 0.6, 0.5, 0.6],
        fairway_widths=[35, 30, 35, 28, 28, 0, 0, 38, 30, 27, 26, 0, 35, 32, 35, 0, 32, 28],
    ),
)


# Registry of all built-in courses
DEFAULT_COURSES: dict[str, CourseModel] = {
    "Augusta National": AUGUSTA_NATIONAL,
    "TPC Sawgrass": TPC_SAWGRASS,
    "Pebble Beach": PEBBLE_BEACH,
    "St Andrews Old Course": ST_ANDREWS,
    "TPC Scottsdale": TPC_SCOTTSDALE,
}


def get_default_course(name: str) -> Optional[CourseModel]:
    """Look up a pre-built course model by name."""
    return DEFAULT_COURSES.get(name)


def create_generic_course(
    name: str = "Generic PGA Course",
    par: int = 72,
    course_type: str = "parkland",
    bermuda: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> CourseModel:
    """Generate a generic 18-hole course with reasonable random parameters.

    Useful for testing when no specific course data is available.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Standard par distribution: 4 par-3s, 10 par-4s, 4 par-5s
    pars = [4, 4, 4, 3, 4, 4, 3, 4, 5, 4, 4, 3, 5, 4, 4, 3, 5, 5]
    rng.shuffle(pars)

    yardage_ranges = {3: (155, 240), 4: (370, 490), 5: (510, 610)}
    yardages = [int(rng.integers(*yardage_ranges[p])) for p in pars]
    difficulties = list(rng.uniform(0.2, 0.8, size=18))
    scoring_avgs = [p + (d - 0.5) * 0.4 for p, d in zip(pars, difficulties)]

    holes = _generate_holes(
        pars=pars,
        yardages=yardages,
        difficulties=difficulties,
        scoring_avgs=scoring_avgs,
        wind_exposures=list(rng.uniform(0.2, 0.8, size=18)),
        green_complexities=list(rng.uniform(0.3, 0.8, size=18)),
        fairway_widths=[0.0 if p == 3 else float(rng.integers(22, 40)) for p in pars],
    )

    return CourseModel(
        name=name,
        holes=holes,
        course_type=course_type,
        bermuda_greens=bermuda,
    )
