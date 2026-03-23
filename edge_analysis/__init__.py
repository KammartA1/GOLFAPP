"""
Edge Analysis System
====================
Twelve independent edge sources, each validated for statistical significance,
ranked by Sharpe ratio, and checked for pairwise independence.

Usage:
    from edge_analysis.source_registry import EdgeSourceRegistry
    registry = EdgeSourceRegistry()
    registry.load_all()
    rankings = registry.rank_sources()
"""

from edge_analysis.source_registry import EdgeSourceRegistry
from edge_analysis.attribution import EdgeAttributionEngine

__all__ = ["EdgeSourceRegistry", "EdgeAttributionEngine"]
