"""Edge Analysis — Golf-specific edge decomposition and attribution engine.

Decomposes betting edge into 5 components:
  1. Predictive — Model accuracy (Brier, log loss, calibration)
  2. Informational — Timing advantage per data source
  3. Market — CLV per market type, market inefficiency
  4. Execution — Price quality, line shopping effectiveness
  5. Structural — Field correlation, wave advantage

Usage:
    from edge_analysis.decomposer import GolfEdgeDecomposer
    from edge_analysis.attribution import EdgeAttributionEngine

    decomposer = GolfEdgeDecomposer()
    report = decomposer.full_decomposition(bet_records)

    attribution = EdgeAttributionEngine()
    result = attribution.run_full_attribution(bet_records)
"""

from edge_analysis.schemas import GolfBetRecord, EdgeReport
from edge_analysis.decomposer import GolfEdgeDecomposer
from edge_analysis.attribution import EdgeAttributionEngine

__all__ = [
    "GolfBetRecord",
    "EdgeReport",
    "GolfEdgeDecomposer",
    "EdgeAttributionEngine",
]
