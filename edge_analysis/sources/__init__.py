"""Golf edge sources — 12 independent signal generators.

Each source implements:
  - get_signal(player, tournament_context) -> float
  - get_mechanism() -> str
  - get_decay_risk() -> str
  - validate(historical_data) -> dict
"""

from edge_analysis.sources.strokes_gained_decomp import StrokesGainedDecompSource
from edge_analysis.sources.course_fit import CourseFitSource
from edge_analysis.sources.weather_advantage import WeatherAdvantageSource
from edge_analysis.sources.wave_advantage import WaveAdvantageSource
from edge_analysis.sources.form_vs_baseline import FormVsBaselineSource
from edge_analysis.sources.pressure_performance import PressurePerformanceSource
from edge_analysis.sources.driving_distance import DrivingDistanceSource
from edge_analysis.sources.cut_probability import CutProbabilitySource
from edge_analysis.sources.fatigue import FatigueSource
from edge_analysis.sources.field_strength import FieldStrengthSource
from edge_analysis.sources.altitude_conditions import AltitudeConditionsSource
from edge_analysis.sources.course_history import CourseHistorySource

ALL_SOURCES = [
    StrokesGainedDecompSource,
    CourseFitSource,
    WeatherAdvantageSource,
    WaveAdvantageSource,
    FormVsBaselineSource,
    PressurePerformanceSource,
    DrivingDistanceSource,
    CutProbabilitySource,
    FatigueSource,
    FieldStrengthSource,
    AltitudeConditionsSource,
    CourseHistorySource,
]

__all__ = [
    "ALL_SOURCES",
    "StrokesGainedDecompSource",
    "CourseFitSource",
    "WeatherAdvantageSource",
    "WaveAdvantageSource",
    "FormVsBaselineSource",
    "PressurePerformanceSource",
    "DrivingDistanceSource",
    "CutProbabilitySource",
    "FatigueSource",
    "FieldStrengthSource",
    "AltitudeConditionsSource",
    "CourseHistorySource",
]
