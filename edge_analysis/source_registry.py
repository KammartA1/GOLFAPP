"""Source registry — Load sources, compute independence, rank by Sharpe, reject correlated.

Manages the portfolio of edge sources:
  - Loads all 12 sources
  - Computes pairwise signal correlation matrix
  - Ranks sources by Sharpe ratio
  - Rejects highly correlated sources to avoid double-counting
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from edge_analysis.edge_sources import EdgeSource, EdgeSourceRegistry

logger = logging.getLogger(__name__)


class SourcePortfolioManager:
    """Manages the portfolio of edge sources — independence, ranking, selection."""

    def __init__(self, max_correlation: float = 0.60, min_sharpe: float = 0.10):
        """
        Args:
            max_correlation: Maximum allowed pairwise correlation between sources.
            min_sharpe: Minimum Sharpe ratio to keep a source active.
        """
        self.max_correlation = max_correlation
        self.min_sharpe = min_sharpe
        self.registry = EdgeSourceRegistry()
        self.registry.register_all_defaults()
        self._signal_history: Dict[str, List[float]] = {}
        self._outcome_history: List[float] = []

    def load_sources(self) -> List[EdgeSource]:
        """Load and return all registered sources."""
        return self.registry.all_sources()

    def record_signals(
        self,
        signals: Dict[str, float],
        outcome: float,
    ) -> None:
        """Record a set of signals and the corresponding outcome for analysis.

        Call this after each bet settles to build the signal history.
        """
        for name, value in signals.items():
            if name not in self._signal_history:
                self._signal_history[name] = []
            self._signal_history[name].append(value)
        self._outcome_history.append(outcome)

    def compute_independence_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Compute pairwise correlation matrix between all source signals.

        Returns:
            (correlation_matrix, source_names)
        """
        names = sorted(self._signal_history.keys())
        if len(names) < 2:
            return np.eye(len(names)), names

        # Align histories to same length
        min_len = min(len(self._signal_history[n]) for n in names)
        if min_len < 10:
            logger.warning("Insufficient signal history (%d) for correlation matrix", min_len)
            return np.eye(len(names)), names

        matrix = np.zeros((len(names), len(names)))
        signals_matrix = np.column_stack([
            np.array(self._signal_history[n][:min_len]) for n in names
        ])

        for i in range(len(names)):
            for j in range(len(names)):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    corr = np.corrcoef(signals_matrix[:, i], signals_matrix[:, j])[0, 1]
                    matrix[i, j] = corr if not np.isnan(corr) else 0.0

        return matrix, names

    def rank_by_sharpe(self) -> List[Tuple[str, float]]:
        """Rank sources by their signal-to-outcome Sharpe ratio.

        Sharpe = mean(signal * outcome) / std(signal * outcome)
        Higher = more predictive and consistent.
        """
        if not self._outcome_history:
            return [(s.name, 0.0) for s in self.registry.all_sources()]

        outcomes = np.array(self._outcome_history)
        rankings = []

        for name, signals in self._signal_history.items():
            n = min(len(signals), len(outcomes))
            if n < 10:
                rankings.append((name, 0.0))
                continue

            sig_arr = np.array(signals[:n])
            out_arr = outcomes[:n]

            # Signal-weighted returns
            returns = sig_arr * (out_arr - 0.5)  # Centered outcomes
            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns))

            sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
            rankings.append((name, round(sharpe, 4)))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def reject_correlated(
        self,
        rankings: Optional[List[Tuple[str, float]]] = None,
    ) -> List[str]:
        """Select independent sources by rejecting highly correlated ones.

        Greedy algorithm:
          1. Start with highest-Sharpe source
          2. Add next source only if correlation with all selected < threshold
          3. Repeat until all sources checked

        Returns:
            List of selected source names (independent, high-Sharpe).
        """
        if rankings is None:
            rankings = self.rank_by_sharpe()

        corr_matrix, names = self.compute_independence_matrix()
        name_to_idx = {n: i for i, n in enumerate(names)}

        selected = []
        for source_name, sharpe in rankings:
            if sharpe < self.min_sharpe:
                continue

            if source_name not in name_to_idx:
                selected.append(source_name)
                continue

            idx = name_to_idx[source_name]

            # Check correlation with all already-selected sources
            is_independent = True
            for sel_name in selected:
                if sel_name not in name_to_idx:
                    continue
                sel_idx = name_to_idx[sel_name]
                if abs(corr_matrix[idx, sel_idx]) > self.max_correlation:
                    logger.info(
                        "Rejecting %s (corr=%.2f with %s)",
                        source_name,
                        corr_matrix[idx, sel_idx],
                        sel_name,
                    )
                    is_independent = False
                    break

            if is_independent:
                selected.append(source_name)

        logger.info("Selected %d/%d independent sources", len(selected), len(rankings))
        return selected

    def get_active_sources(self) -> List[EdgeSource]:
        """Get the current set of active (independent, high-Sharpe) sources."""
        rankings = self.rank_by_sharpe()
        active_names = self.reject_correlated(rankings)
        return [
            self.registry.get_source(name)
            for name in active_names
            if self.registry.get_source(name) is not None
        ]

    def summary(self) -> dict:
        """Summary of source portfolio status."""
        rankings = self.rank_by_sharpe()
        corr_matrix, names = self.compute_independence_matrix()
        active = self.reject_correlated(rankings)

        return {
            "total_sources": len(self.registry.all_sources()),
            "active_sources": len(active),
            "active_names": active,
            "rankings": rankings,
            "n_signal_observations": len(self._outcome_history),
            "correlation_matrix_shape": corr_matrix.shape,
            "max_pairwise_correlation": float(np.max(np.abs(corr_matrix - np.eye(len(names))))) if len(names) > 1 else 0.0,
        }
