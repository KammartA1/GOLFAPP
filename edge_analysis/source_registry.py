"""
Edge Source Registry
=====================
Loads all 12 edge sources, computes independence matrix, ranks by Sharpe,
and rejects sources failing validation gates.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from edge_analysis.edge_sources import (
    compute_correlation_matrix,
    compute_vif,
    build_signal_matrix,
    get_expected_correlations,
)

log = logging.getLogger(__name__)

# Validation gates: a source must pass ALL gates to be accepted
_VALIDATION_GATES = {
    "min_sample_size": 30,
    "max_p_value": 0.15,          # lenient for individual sources
    "min_sharpe": 0.10,
    "max_vif": 10.0,              # multicollinearity limit
    "min_quintile_spread": 0.01,
}


class EdgeSourceRegistry:
    """
    Central registry for all edge sources.

    Loads sources, validates them, computes independence,
    ranks by Sharpe, and provides the active source set.
    """

    def __init__(self):
        self._sources: list = []
        self._names: list[str] = []
        self._validation_results: dict[str, dict] = {}
        self._independence: dict | None = None
        self._rankings: list[dict] | None = None
        self._rejected: list[dict] = []
        self._accepted: list[dict] = []

    # ── Loading ──────────────────────────────────────────────────────────

    def load_all(self) -> None:
        """Instantiate all 12 edge sources."""
        from edge_analysis.sources import ALL_SOURCES

        self._sources = []
        self._names = []
        for cls in ALL_SOURCES:
            try:
                inst = cls()
                self._sources.append(inst)
                self._names.append(getattr(inst, "name", cls.__name__))
            except Exception as exc:
                log.warning("Failed to load source %s: %s", cls.__name__, exc)

        log.info("Loaded %d edge sources", len(self._sources))

    def get_sources(self) -> list:
        """Return all loaded source instances."""
        return list(self._sources)

    def get_source_by_name(self, name: str):
        """Look up a source by name."""
        for src in self._sources:
            if getattr(src, "name", "") == name:
                return src
        return None

    # ── Validation ───────────────────────────────────────────────────────

    def validate_all(self, historical_data: list[dict]) -> dict[str, dict]:
        """
        Run validate() on every source and store results.

        Parameters
        ----------
        historical_data : list[dict]
            Each entry has 'player', 'tournament_context', 'actual_finish'.

        Returns
        -------
        dict mapping source name -> validation result dict
        """
        self._validation_results = {}
        for src in self._sources:
            name = getattr(src, "name", "?")
            try:
                result = src.validate(historical_data)
                self._validation_results[name] = result
            except Exception as exc:
                log.warning("Validation failed for %s: %s", name, exc)
                self._validation_results[name] = {
                    "sharpe": 0.0, "p_value": 1.0,
                    "sample_size": 0,
                    "correlation_with_other_signals": {},
                    "status": "ERROR",
                    "error": str(exc),
                }

        return dict(self._validation_results)

    # ── Independence ─────────────────────────────────────────────────────

    def compute_independence(self, observations: list[dict]) -> dict:
        """
        Compute pairwise correlation and VIF across all sources.

        Parameters
        ----------
        observations : list[dict]
            Each dict has 'player' and 'tournament_context'.

        Returns
        -------
        dict with correlation_matrix, flagged_pairs, vif_results, etc.
        """
        if not self._sources:
            self.load_all()

        matrix, names = build_signal_matrix(self._sources, observations)

        corr_result = compute_correlation_matrix(matrix, names)
        vif_result = compute_vif(matrix, names)

        self._independence = {
            **corr_result,
            "vif_results": vif_result,
            "signal_matrix": matrix,
        }

        # Cross-fill correlation info into validation results
        for vif_entry in vif_result:
            src_name = vif_entry["source"]
            if src_name in self._validation_results:
                self._validation_results[src_name]["vif"] = vif_entry["vif"]
                self._validation_results[src_name]["vif_status"] = vif_entry["status"]

        return self._independence

    # ── Ranking ──────────────────────────────────────────────────────────

    def rank_sources(self) -> list[dict]:
        """
        Rank all sources by composite score: Sharpe * independence * sample quality.

        Returns
        -------
        list of dicts sorted by composite score (best first).
        Each dict: {name, sharpe, p_value, sample_size, vif, composite, status, accepted}
        """
        rankings = []

        for src in self._sources:
            name = getattr(src, "name", "?")
            val = self._validation_results.get(name, {})

            sharpe = val.get("sharpe", 0.0)
            p_val = val.get("p_value", 1.0)
            n = val.get("sample_size", 0)
            vif = val.get("vif", 1.0)
            spread = val.get("quintile_spread", 0.0)

            # ── Gate checks ─────────────────────────────────────────────
            gates_passed = True
            rejection_reasons = []

            if n < _VALIDATION_GATES["min_sample_size"]:
                gates_passed = False
                rejection_reasons.append(f"sample_size={n} < {_VALIDATION_GATES['min_sample_size']}")

            if p_val > _VALIDATION_GATES["max_p_value"]:
                gates_passed = False
                rejection_reasons.append(f"p_value={p_val:.4f} > {_VALIDATION_GATES['max_p_value']}")

            if abs(sharpe) < _VALIDATION_GATES["min_sharpe"]:
                gates_passed = False
                rejection_reasons.append(f"|sharpe|={abs(sharpe):.4f} < {_VALIDATION_GATES['min_sharpe']}")

            if vif > _VALIDATION_GATES["max_vif"]:
                gates_passed = False
                rejection_reasons.append(f"vif={vif:.2f} > {_VALIDATION_GATES['max_vif']}")

            # ── Composite score ─────────────────────────────────────────
            # Higher is better.  Penalise high VIF, reward low p-value.
            if gates_passed:
                p_bonus = max(0.0, 1.0 - p_val)  # 0-1, higher when p is low
                vif_penalty = max(0.0, 1.0 - (vif - 1.0) / 10.0)  # 1 at VIF=1, 0 at VIF=11
                sample_bonus = min(1.0, n / 200.0)  # saturates at 200 samples
                composite = abs(sharpe) * p_bonus * vif_penalty * sample_bonus
            else:
                composite = 0.0

            entry = {
                "name": name,
                "category": getattr(src, "category", "unknown"),
                "sharpe": round(sharpe, 4),
                "p_value": round(p_val, 6),
                "sample_size": n,
                "quintile_spread": round(spread, 4),
                "vif": round(vif, 2) if not np.isinf(vif) else 999.0,
                "composite_score": round(composite, 4),
                "status": val.get("status", "UNKNOWN"),
                "accepted": gates_passed,
                "rejection_reasons": rejection_reasons,
                "mechanism": src.get_mechanism(),
                "decay_risk": src.get_decay_risk(),
            }
            rankings.append(entry)

        rankings.sort(key=lambda x: -x["composite_score"])
        self._rankings = rankings

        self._accepted = [r for r in rankings if r["accepted"]]
        self._rejected = [r for r in rankings if not r["accepted"]]

        return rankings

    # ── Accessors ────────────────────────────────────────────────────────

    def get_accepted_sources(self) -> list[dict]:
        """Return only sources that passed all validation gates."""
        if self._rankings is None:
            return []
        return list(self._accepted)

    def get_rejected_sources(self) -> list[dict]:
        """Return sources that failed validation."""
        if self._rankings is None:
            return []
        return list(self._rejected)

    def get_independence_report(self) -> dict | None:
        """Return the full independence analysis."""
        return self._independence

    def get_source_detail(self, name: str) -> dict | None:
        """Return validation + ranking detail for a single source."""
        if self._rankings is None:
            return None
        for r in self._rankings:
            if r["name"] == name:
                return r
        return None

    def get_composite_signal(
        self,
        player: dict,
        tournament_context: dict,
        weighting: str = "sharpe",
    ) -> dict:
        """
        Compute a composite edge signal from all accepted sources.

        Parameters
        ----------
        player : dict
        tournament_context : dict
        weighting : str
            "equal" — equal weight to all accepted sources
            "sharpe" — weight by Sharpe ratio (default)
            "composite" — weight by composite score

        Returns
        -------
        dict with:
            composite_signal  – float
            source_signals    – dict mapping source name -> individual signal
            weights           – dict mapping source name -> weight used
        """
        if not self._sources:
            self.load_all()

        accepted = self._accepted if self._accepted else [
            {"name": getattr(s, "name", "?")} for s in self._sources
        ]
        accepted_names = {r["name"] for r in accepted}

        source_signals = {}
        raw_weights = {}

        for src in self._sources:
            name = getattr(src, "name", "?")
            if name not in accepted_names:
                continue

            try:
                sig = src.get_signal(player, tournament_context)
            except Exception:
                sig = 0.0
            source_signals[name] = sig

            # Determine weight
            ranking = next((r for r in (self._rankings or []) if r["name"] == name), {})
            if weighting == "sharpe":
                raw_weights[name] = abs(ranking.get("sharpe", 0.1))
            elif weighting == "composite":
                raw_weights[name] = max(ranking.get("composite_score", 0.01), 0.01)
            else:
                raw_weights[name] = 1.0

        # Normalise weights
        total_w = sum(raw_weights.values())
        if total_w < 1e-9:
            total_w = 1.0
        norm_weights = {k: v / total_w for k, v in raw_weights.items()}

        composite = sum(source_signals[k] * norm_weights.get(k, 0.0) for k in source_signals)

        return {
            "composite_signal": round(float(composite), 4),
            "source_signals": {k: round(v, 4) for k, v in source_signals.items()},
            "weights": {k: round(v, 4) for k, v in norm_weights.items()},
            "n_active_sources": len(source_signals),
        }

    # ── Summary ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """High-level summary of the registry state."""
        return {
            "total_sources": len(self._sources),
            "accepted": len(self._accepted),
            "rejected": len(self._rejected),
            "independence_score": (
                self._independence.get("independence_score", None)
                if self._independence else None
            ),
            "n_flagged_pairs": (
                len(self._independence.get("flagged_pairs", []))
                if self._independence else 0
            ),
            "top_source": self._rankings[0]["name"] if self._rankings else None,
            "expected_correlations": get_expected_correlations(),
        }
