"""
Edge source modules — twelve independent signal generators.
Each module exports a class implementing:
    get_signal(player, tournament_context) -> float
    get_mechanism() -> str
    get_decay_risk() -> str
    validate(historical_data) -> dict
"""

from edge_analysis.sources.sg_decomposition import SGDecompositionSource
from edge_analysis.sources.course_fit import CourseFitSource
from edge_analysis.sources.weather_advantage import WeatherAdvantageSource
from edge_analysis.sources.wave_advantage import WaveAdvantageSource
from edge_analysis.sources.current_form import CurrentFormSource
from edge_analysis.sources.pressure_performance import PressurePerformanceSource
from edge_analysis.sources.driving_distance import DrivingDistanceSource
from edge_analysis.sources.cut_probability import CutProbabilitySource
from edge_analysis.sources.fatigue_modeling import FatigueModelingSource
from edge_analysis.sources.field_strength import FieldStrengthSource
from edge_analysis.sources.altitude_conditions import AltitudeConditionsSource
from edge_analysis.sources.tournament_history import TournamentHistorySource

ALL_SOURCES = [
    SGDecompositionSource,
    CourseFitSource,
    WeatherAdvantageSource,
    WaveAdvantageSource,
    CurrentFormSource,
    PressurePerformanceSource,
    DrivingDistanceSource,
    CutProbabilitySource,
    FatigueModelingSource,
    FieldStrengthSource,
    AltitudeConditionsSource,
    TournamentHistorySource,
]

__all__ = [
    "SGDecompositionSource",
    "CourseFitSource",
    "WeatherAdvantageSource",
    "WaveAdvantageSource",
    "CurrentFormSource",
    "PressurePerformanceSource",
    "DrivingDistanceSource",
    "CutProbabilitySource",
    "FatigueModelingSource",
    "FieldStrengthSource",
    "AltitudeConditionsSource",
    "TournamentHistorySource",
    "ALL_SOURCES",
]
