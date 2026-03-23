"""
edge_analysis/schemas.py
========================
Dataclasses for the Golf Edge Decomposition system.

GolfBetRecord — every historical bet with golf-specific context.
EdgeComponentResult — output from a single edge component analysis.
EdgeReport — full 5-component attribution report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass(frozen=True)
class GolfBetRecord:
    """Complete record of a single golf bet with all fields for edge decomposition."""
    bet_id: str
    timestamp: datetime                     # When the bet was placed
    signal_generated_at: Optional[datetime]  # When the model generated the signal
    # Golf context
    tournament: str                         # Tournament name
    player: str                             # Golfer name
    course_id: Optional[str] = None         # Course identifier
    market_type: str = "outright"           # outright/matchup/top5/top10/top20/make_cut
    direction: str = "over"                 # OVER/UNDER/WIN/PLACE
    wave: Optional[str] = None              # "AM" or "PM" (R1/R2 tee wave)
    # Weather at bet time
    weather_conditions: Optional[str] = None  # JSON or summary string
    wind_speed_mph: Optional[float] = None
    temperature_f: Optional[float] = None
    precipitation_mm: Optional[float] = None
    # Lines
    signal_line: float = 0.0                # Line/odds when signal fired
    bet_line: float = 0.0                   # Line/odds at actual bet placement
    closing_line: float = 0.0               # Closing line/odds
    opening_line: Optional[float] = None    # Opening line (if available)
    # Probabilities
    predicted_prob: float = 0.5             # Model's predicted probability
    market_prob_at_bet: float = 0.5         # Market implied probability at bet time
    market_prob_at_close: Optional[float] = None  # Market implied prob at close
    # Outcome
    actual_outcome: Optional[float] = None  # Actual result (position, strokes, etc.)
    won: Optional[bool] = None              # Did the bet win?
    # Financial
    stake: float = 0.0
    pnl: float = 0.0
    odds_american: int = -110
    odds_decimal: float = 1.909
    kelly_fraction: float = 0.0
    # Model context
    model_projection: float = 0.0
    model_std: float = 1.0
    confidence_score: float = 0.0
    # Timing
    line_moved_at: Optional[datetime] = None  # When did the market line first move?
    weather_fetched_at: Optional[datetime] = None  # When weather data was captured
    injury_withdrawal_at: Optional[datetime] = None  # When injury/WD info was known


@dataclass
class EdgeComponentResult:
    """Output from one edge decomposition component."""
    name: str                               # "predictive", "informational", etc.
    edge_pct_of_roi: float                  # This component's share of total ROI
    absolute_value: float                   # Raw metric value
    p_value: float                          # Statistical significance
    is_significant: bool                    # p_value < 0.05
    is_positive: bool                       # Does this component add value?
    sample_size: int = 0
    details: dict = field(default_factory=dict)
    verdict: str = ""                       # Human-readable conclusion


@dataclass
class CalibrationPoint:
    """Single point on a calibration curve."""
    bucket_lower: float
    bucket_upper: float
    predicted_avg: float
    actual_rate: float
    n_bets: int
    calibration_error: float


@dataclass
class EdgeReport:
    """Full 5-component edge attribution report for Golf."""
    generated_at: datetime
    sport: str
    total_roi: float
    total_bets: int
    total_pnl: float
    # Component percentages
    predictive_pct: float
    informational_pct: float
    market_pct: float
    execution_pct: float
    structural_pct: float
    # Component details
    predictive: Optional[EdgeComponentResult] = None
    informational: Optional[EdgeComponentResult] = None
    market_inefficiency: Optional[EdgeComponentResult] = None
    execution: Optional[EdgeComponentResult] = None
    structural: Optional[EdgeComponentResult] = None
    # Calibration
    calibration_curve: List[CalibrationPoint] = field(default_factory=list)
    brier_score: float = 0.0
    log_loss: float = 0.0
    brier_baseline: float = 0.25
    log_loss_baseline: float = 0.693
    # Final verdict
    verdict: str = ""
    heavy_lifter: str = ""
    illusions: List[str] = field(default_factory=list)
