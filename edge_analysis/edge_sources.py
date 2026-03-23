"""Edge source base class and registry of all 12 golf edge sources.

Every edge source must implement the EdgeSource protocol:
  - get_signal(player, tournament_context) -> float
  - get_mechanism() -> str
  - get_decay_risk() -> str
  - validate(historical_data) -> dict
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EdgeSource(ABC):
    """Abstract base class for all golf edge sources."""

    name: str = "base"
    category: str = "unknown"  # predictive / informational / structural
    description: str = ""

    @abstractmethod
    def get_signal(self, player: str, tournament_context: Dict[str, Any]) -> float:
        """Generate a signal for a player in a tournament context.

        Args:
            player: Player name.
            tournament_context: Dict with keys like course_name, course_id,
                weather, wave, field_sg_values, event_id, etc.

        Returns:
            Signal value. Positive = favorable, negative = unfavorable.
            Scale: standard deviations from neutral (0 = neutral).
        """
        raise NotImplementedError

    @abstractmethod
    def get_mechanism(self) -> str:
        """Describe the economic mechanism behind this edge source.

        Why does this signal persist? Why hasn't the market fully priced it?
        """
        raise NotImplementedError

    @abstractmethod
    def get_decay_risk(self) -> str:
        """Assess the risk that this edge source will decay over time.

        Returns one of: 'low', 'medium', 'high'
        Plus explanation of why.
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, historical_data: List[Dict[str, Any]]) -> dict:
        """Validate this source against historical data.

        Args:
            historical_data: List of dicts with player performance data.

        Returns:
            {
                'is_valid': bool,
                'sharpe': float,
                'hit_rate': float,
                'n_samples': int,
                'p_value': float,
                'details': dict,
            }
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<EdgeSource: {self.name}>"


class EdgeSourceRegistry:
    """Registry for all edge sources. Loads, validates, and ranks sources."""

    def __init__(self):
        self._sources: Dict[str, EdgeSource] = {}

    def register(self, source: EdgeSource) -> None:
        """Register an edge source."""
        self._sources[source.name] = source
        logger.info("Registered edge source: %s", source.name)

    def register_all_defaults(self) -> None:
        """Register all 12 default golf edge sources."""
        from edge_analysis.sources import ALL_SOURCES
        for source_cls in ALL_SOURCES:
            self.register(source_cls())

    def get_source(self, name: str) -> Optional[EdgeSource]:
        """Get a source by name."""
        return self._sources.get(name)

    def all_sources(self) -> List[EdgeSource]:
        """Get all registered sources."""
        return list(self._sources.values())

    def get_all_signals(
        self, player: str, tournament_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get signals from all registered sources for a player."""
        signals = {}
        for name, source in self._sources.items():
            try:
                signal = source.get_signal(player, tournament_context)
                signals[name] = signal
            except Exception as e:
                logger.warning("Source %s failed for %s: %s", name, player, e)
                signals[name] = 0.0
        return signals

    def validate_all(self, historical_data: List[Dict[str, Any]]) -> Dict[str, dict]:
        """Validate all sources against historical data."""
        results = {}
        for name, source in self._sources.items():
            try:
                results[name] = source.validate(historical_data)
            except Exception as e:
                logger.warning("Validation failed for %s: %s", name, e)
                results[name] = {"is_valid": False, "error": str(e)}
        return results
