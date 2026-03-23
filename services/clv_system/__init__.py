"""CLV (Closing Line Value) tracking system.

Tracks odds from signal generation through closing to measure
whether we consistently beat the closing line — the gold standard
metric for sports betting edge.
"""

from services.clv_system.clv_calculator import CLVCalculator
from services.clv_system.odds_ingestion import OddsIngestionService
from services.clv_system.closing_capture import ClosingLineCapture

__all__ = ["CLVCalculator", "OddsIngestionService", "ClosingLineCapture"]
