"""Course model — Course-specific hole parameters.

Defines per-hole difficulty, par, distance, and scoring distribution
for tournament simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class HoleProfile:
    """Single hole on a golf course."""

    number: int
    par: int
    distance: int  # yards
    difficulty: float = 0.0  # 0.0 = average, positive = harder
    avg_score_to_par: float = 0.0  # Historical average
    has_water: bool = False
    has_deep_bunkers: bool = False
    wind_exposed: bool = False


@dataclass
class CourseProfile:
    """Full 18-hole course profile for simulation."""

    name: str
    total_par: int = 72
    total_yardage: int = 7200
    holes: List[HoleProfile] = field(default_factory=list)
    elevation: int = 0  # feet above sea level
    is_links: bool = False
    is_bermuda: bool = False
    avg_field_scoring: float = -2.0  # Field average relative to par per round

    @property
    def n_par3(self) -> int:
        return sum(1 for h in self.holes if h.par == 3)

    @property
    def n_par4(self) -> int:
        return sum(1 for h in self.holes if h.par == 4)

    @property
    def n_par5(self) -> int:
        return sum(1 for h in self.holes if h.par == 5)

    @property
    def total_difficulty(self) -> float:
        return sum(h.difficulty for h in self.holes)


class CourseModel:
    """Generate course profiles for simulation."""

    def generate_default_course(
        self,
        name: str = "Default Course",
        total_par: int = 72,
        total_yardage: int = 7200,
        n_par3: int = 4,
        n_par5: int = 4,
        rng: np.random.Generator = None,
    ) -> CourseProfile:
        """Generate a standard course profile with realistic hole distribution."""
        if rng is None:
            rng = np.random.default_rng(42)

        holes = []
        n_par4 = 18 - n_par3 - n_par5

        # Par-3 holes
        par3_distances = np.linspace(170, 230, n_par3).astype(int)
        for i, dist in enumerate(par3_distances):
            holes.append(HoleProfile(
                number=0,  # Will be reassigned
                par=3,
                distance=int(dist),
                difficulty=rng.uniform(-0.1, 0.2),
                avg_score_to_par=rng.uniform(0.05, 0.20),
            ))

        # Par-4 holes
        par4_distances = np.linspace(380, 490, n_par4).astype(int)
        for i, dist in enumerate(par4_distances):
            holes.append(HoleProfile(
                number=0,
                par=4,
                distance=int(dist),
                difficulty=rng.uniform(-0.15, 0.25),
                avg_score_to_par=rng.uniform(-0.05, 0.15),
            ))

        # Par-5 holes
        par5_distances = np.linspace(520, 600, n_par5).astype(int)
        for i, dist in enumerate(par5_distances):
            holes.append(HoleProfile(
                number=0,
                par=5,
                distance=int(dist),
                difficulty=rng.uniform(-0.2, 0.1),
                avg_score_to_par=rng.uniform(-0.30, -0.10),
            ))

        # Shuffle to create realistic routing
        rng.shuffle(holes)
        for i, hole in enumerate(holes):
            hole.number = i + 1

        return CourseProfile(
            name=name,
            total_par=total_par,
            total_yardage=total_yardage,
            holes=holes,
        )

    def from_event_data(self, event: dict) -> CourseProfile:
        """Build course profile from event database record."""
        name = event.get("course_name", event.get("name", "Unknown"))
        par = event.get("course_par", 72)
        yardage = event.get("course_yardage", 7200)

        # If we have hole-by-hole data, use it
        hole_data = event.get("holes", [])
        if hole_data:
            holes = [
                HoleProfile(
                    number=h.get("number", i + 1),
                    par=h.get("par", 4),
                    distance=h.get("distance", 430),
                    difficulty=h.get("difficulty", 0.0),
                    avg_score_to_par=h.get("avg_score_to_par", 0.0),
                    has_water=h.get("has_water", False),
                    wind_exposed=h.get("wind_exposed", False),
                )
                for i, h in enumerate(hole_data)
            ]
        else:
            # Generate default holes based on par/yardage
            course = self.generate_default_course(name, par, yardage)
            holes = course.holes

        return CourseProfile(
            name=name,
            total_par=par,
            total_yardage=yardage,
            holes=holes,
            elevation=event.get("elevation", 0),
            is_links=event.get("is_links", False),
            is_bermuda=event.get("is_bermuda", False),
        )
