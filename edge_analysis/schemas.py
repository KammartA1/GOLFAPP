"""Schemas for golf edge analysis — bet records and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


MARKET_TYPES = ("outright", "matchup", "top5", "top10", "top20", "make_cut")


@dataclass
class GolfBetRecord:
    """A single golf bet with full context for edge decomposition."""

    bet_id: str
    tournament: str
    player: str
    market_type: str  # outright / matchup / top5 / top10 / top20 / make_cut
    signal_line: float  # Line when signal was generated
    bet_line: float  # Line when bet was placed
    closing_line: float  # Closing line at first tee time
    predicted_prob: float  # Model probability at bet time
    actual_outcome: float  # 1.0 = win, 0.0 = loss (for binary markets)
    # Context
    weather_conditions: str = "normal"  # calm / windy / rainy / extreme
    wave: str = "unknown"  # AM / PM / unknown
    course_id: str = ""
    # Timing
    bet_timestamp: Optional[datetime] = None
    signal_timestamp: Optional[datetime] = None
    closing_timestamp: Optional[datetime] = None
    # Pricing
    odds_american: int = -110
    odds_decimal: float = 1.909
    closing_odds_american: Optional[int] = None
    closing_odds_decimal: Optional[float] = None
    # Source info
    data_sources_available: list = field(default_factory=list)
    # Book
    book: str = "unknown"
    # Settlement
    pnl: float = 0.0
    stake: float = 0.0

    def __post_init__(self):
        if self.market_type not in MARKET_TYPES:
            raise ValueError(
                f"Invalid market_type '{self.market_type}'. "
                f"Must be one of: {MARKET_TYPES}"
            )
        if self.odds_decimal <= 1.0:
            if self.odds_american > 0:
                self.odds_decimal = 1.0 + self.odds_american / 100.0
            elif self.odds_american < 0:
                self.odds_decimal = 1.0 + 100.0 / abs(self.odds_american)
            else:
                self.odds_decimal = 2.0

    @property
    def market_prob(self) -> float:
        """Implied probability from bet-time odds."""
        return 1.0 / self.odds_decimal if self.odds_decimal > 0 else 0.5

    @property
    def closing_prob(self) -> float:
        """Implied probability from closing odds."""
        if self.closing_odds_decimal and self.closing_odds_decimal > 0:
            return 1.0 / self.closing_odds_decimal
        if self.closing_line != 0 and self.bet_line != 0:
            # Approximate from line movement
            return self.market_prob + (self.closing_line - self.bet_line) * 0.02
        return self.market_prob

    @property
    def edge(self) -> float:
        """Model edge = predicted_prob - market_prob."""
        return self.predicted_prob - self.market_prob

    @property
    def clv_cents(self) -> float:
        """CLV in cents per dollar wagered."""
        return (self.closing_prob - self.market_prob) * 100

    @property
    def beat_close(self) -> bool:
        """Did we get a better price than closing?"""
        return self.market_prob < self.closing_prob


@dataclass
class EdgeComponent:
    """One component of the 5-way edge decomposition."""

    name: str
    value: float  # Contribution in cents
    confidence: float  # 0-1 confidence in measurement
    details: dict = field(default_factory=dict)
    verdict: str = ""  # human-readable


@dataclass
class EdgeReport:
    """Full edge attribution report with verdict."""

    report_date: datetime
    n_bets: int
    total_pnl: float
    total_roi: float
    # 5-component decomposition
    predictive: EdgeComponent = field(default_factory=lambda: EdgeComponent("predictive", 0.0, 0.0))
    informational: EdgeComponent = field(default_factory=lambda: EdgeComponent("informational", 0.0, 0.0))
    market: EdgeComponent = field(default_factory=lambda: EdgeComponent("market", 0.0, 0.0))
    execution: EdgeComponent = field(default_factory=lambda: EdgeComponent("execution", 0.0, 0.0))
    structural: EdgeComponent = field(default_factory=lambda: EdgeComponent("structural", 0.0, 0.0))
    # Aggregate
    total_edge_cents: float = 0.0
    edge_is_real: bool = False
    p_value: float = 1.0
    verdict: str = ""
    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    # Per-market breakdown
    by_market_type: dict = field(default_factory=dict)

    def components_list(self) -> list:
        return [self.predictive, self.informational, self.market,
                self.execution, self.structural]

    def dominant_source(self) -> str:
        """Which component contributes most edge?"""
        components = self.components_list()
        if not components:
            return "none"
        best = max(components, key=lambda c: c.value)
        return best.name if best.value > 0 else "none"
