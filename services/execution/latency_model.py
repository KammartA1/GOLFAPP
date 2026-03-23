"""Latency model — Time from signal to execution and its cost."""

from __future__ import annotations

import numpy as np


class LatencyModel:
    """Model execution latency and its impact on edge capture."""

    def __init__(
        self,
        avg_latency_seconds: float = 30.0,
        edge_decay_per_minute: float = 0.003,
    ):
        self.avg_latency = avg_latency_seconds
        self.edge_decay_rate = edge_decay_per_minute

    def estimate_latency_cost(
        self,
        edge_at_signal: float,
        latency_seconds: float | None = None,
        market_type: str = "outright",
        is_line_moving: bool = False,
    ) -> dict:
        """Estimate edge lost due to execution latency.

        Returns:
            {
                'latency_seconds': float,
                'edge_at_signal': float,
                'edge_after_latency': float,
                'edge_lost': float,
                'edge_lost_pct': float,
            }
        """
        if latency_seconds is None:
            latency_seconds = self.avg_latency

        minutes = latency_seconds / 60.0

        # Edge decay rate varies by market type
        decay_multipliers = {
            "outright": 0.8,    # Outright odds move slowly
            "matchup": 1.5,    # H2H can move fast
            "top5": 1.0,
            "top10": 1.0,
            "top20": 0.9,
            "make_cut": 0.7,
        }
        decay_mult = decay_multipliers.get(market_type, 1.0)

        # Accelerated decay when line is actively moving
        if is_line_moving:
            decay_mult *= 2.0

        edge_lost_per_min = self.edge_decay_rate * decay_mult
        total_edge_lost = edge_lost_per_min * minutes
        edge_after = max(0, edge_at_signal - total_edge_lost)

        return {
            "latency_seconds": round(latency_seconds, 1),
            "edge_at_signal": round(edge_at_signal, 4),
            "edge_after_latency": round(edge_after, 4),
            "edge_lost": round(total_edge_lost, 4),
            "edge_lost_pct": round(total_edge_lost / max(edge_at_signal, 0.001), 4),
            "edge_survived": edge_after > 0,
        }

    def optimal_latency_threshold(self, min_edge: float = 0.02) -> float:
        """Maximum acceptable latency (seconds) before edge falls below minimum."""
        if self.edge_decay_rate <= 0:
            return 3600.0
        max_minutes = min_edge / self.edge_decay_rate
        return max_minutes * 60.0
