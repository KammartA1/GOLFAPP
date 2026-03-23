"""
Edge Sources — Correlation & Independence Analysis
=====================================================
Documents all 12 edge sources, computes pairwise correlation matrix,
and flags non-independent pairs that should not be double-counted.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats as sp_stats

log = logging.getLogger(__name__)

# All twelve edge source classes
_SOURCE_NAMES = [
    "SG Decomposition",
    "Course Fit",
    "Weather Advantage",
    "Wave Advantage",
    "Current Form",
    "Pressure Performance",
    "Driving Distance",
    "Cut Probability",
    "Fatigue Modeling",
    "Field Strength",
    "Altitude & Conditions",
    "Tournament History",
]

# Known correlation groups (sources that share inputs / mechanisms):
# These pairs should be checked and potentially down-weighted.
_EXPECTED_CORRELATIONS = {
    ("SG Decomposition", "Course Fit"): "Both use SG components + course profiles",
    ("SG Decomposition", "Driving Distance"): "OTT component overlaps with distance",
    ("Course Fit", "Driving Distance"): "Course fit includes distance bonus",
    ("Weather Advantage", "Wave Advantage"): "Both depend on wind conditions",
    ("Weather Advantage", "Altitude & Conditions"): "Atmospheric overlap",
    ("Current Form", "Pressure Performance"): "Both use SG history",
    ("Cut Probability", "Field Strength"): "Both consider field quality",
    ("Cut Probability", "Current Form"): "Form feeds cut probability",
}

# Independence threshold: |r| > this is flagged as non-independent
_INDEPENDENCE_THRESHOLD = 0.40


def get_all_source_names() -> list[str]:
    """Return names of all 12 edge sources."""
    return list(_SOURCE_NAMES)


def get_expected_correlations() -> dict[tuple[str, str], str]:
    """Return known pairs that share inputs/mechanisms."""
    return dict(_EXPECTED_CORRELATIONS)


def compute_correlation_matrix(
    signal_matrix: np.ndarray,
    source_names: list[str] | None = None,
) -> dict:
    """
    Compute pairwise Spearman rank correlation between all edge sources.

    Parameters
    ----------
    signal_matrix : np.ndarray
        Shape (n_observations, n_sources).  Each column is one source's signals
        across the same set of player-tournament observations.
    source_names : list[str], optional
        Labels for columns.  Defaults to canonical source names.

    Returns
    -------
    dict with:
        correlation_matrix  – np.ndarray (n_sources x n_sources)
        p_value_matrix      – np.ndarray (n_sources x n_sources)
        source_names        – list[str]
        flagged_pairs       – list of dicts for non-independent pairs
        independence_score  – float (0-1), overall independence of the signal set
    """
    if source_names is None:
        source_names = _SOURCE_NAMES[:signal_matrix.shape[1]]

    n = signal_matrix.shape[1]
    corr_mat = np.zeros((n, n), dtype=float)
    pval_mat = np.ones((n, n), dtype=float)

    for i in range(n):
        for j in range(i, n):
            if i == j:
                corr_mat[i, j] = 1.0
                pval_mat[i, j] = 0.0
            else:
                # Remove NaN pairs
                mask = ~(np.isnan(signal_matrix[:, i]) | np.isnan(signal_matrix[:, j]))
                if mask.sum() < 10:
                    corr_mat[i, j] = 0.0
                    corr_mat[j, i] = 0.0
                    continue

                r, p = sp_stats.spearmanr(
                    signal_matrix[mask, i],
                    signal_matrix[mask, j],
                )
                if np.isnan(r):
                    r, p = 0.0, 1.0
                corr_mat[i, j] = r
                corr_mat[j, i] = r
                pval_mat[i, j] = p
                pval_mat[j, i] = p

    # Flag non-independent pairs
    flagged = []
    for i in range(n):
        for j in range(i + 1, n):
            abs_r = abs(corr_mat[i, j])
            if abs_r > _INDEPENDENCE_THRESHOLD:
                pair = (source_names[i], source_names[j])
                reason = _EXPECTED_CORRELATIONS.get(pair, "")
                if not reason:
                    reason = _EXPECTED_CORRELATIONS.get((pair[1], pair[0]), "Unexpected correlation")
                flagged.append({
                    "source_a": source_names[i],
                    "source_b": source_names[j],
                    "correlation": round(float(corr_mat[i, j]), 4),
                    "p_value": round(float(pval_mat[i, j]), 6),
                    "abs_correlation": round(abs_r, 4),
                    "reason": reason,
                    "recommendation": _recommend_action(abs_r),
                })

    # Overall independence score: fraction of off-diagonal pairs below threshold
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append(abs(corr_mat[i, j]))
    independence_score = float(np.mean([1.0 if r < _INDEPENDENCE_THRESHOLD else 0.0 for r in off_diag]))

    return {
        "correlation_matrix": corr_mat,
        "p_value_matrix": pval_mat,
        "source_names": source_names,
        "flagged_pairs": sorted(flagged, key=lambda x: -x["abs_correlation"]),
        "independence_score": round(independence_score, 4),
        "n_sources": n,
        "n_observations": signal_matrix.shape[0],
    }


def compute_vif(signal_matrix: np.ndarray, source_names: list[str] | None = None) -> list[dict]:
    """
    Compute Variance Inflation Factor for each source.

    VIF > 5 indicates problematic multicollinearity.
    VIF > 10 indicates severe multicollinearity.

    Parameters
    ----------
    signal_matrix : np.ndarray
        Shape (n_obs, n_sources).

    Returns
    -------
    list of dicts with {source, vif, status}
    """
    if source_names is None:
        source_names = _SOURCE_NAMES[:signal_matrix.shape[1]]

    n = signal_matrix.shape[1]
    # Remove rows with NaN
    mask = ~np.any(np.isnan(signal_matrix), axis=1)
    clean = signal_matrix[mask]

    if clean.shape[0] < n + 2:
        return [{"source": s, "vif": float("nan"), "status": "INSUFFICIENT_DATA"} for s in source_names]

    results = []
    for i in range(n):
        y = clean[:, i]
        X = np.delete(clean, i, axis=1)

        # Add intercept
        X_int = np.column_stack([np.ones(X.shape[0]), X])

        # OLS: y = X_int @ beta
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X_int, y, rcond=None)
            y_hat = X_int @ beta
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            r_sq = max(0.0, min(r_sq, 0.9999))
            vif = 1.0 / (1.0 - r_sq)
        except np.linalg.LinAlgError:
            vif = float("inf")

        if vif > 10:
            status = "SEVERE"
        elif vif > 5:
            status = "MODERATE"
        elif vif > 2.5:
            status = "MILD"
        else:
            status = "OK"

        results.append({
            "source": source_names[i],
            "vif": round(float(vif), 2),
            "r_squared": round(float(r_sq) if not math.isinf(vif) else 1.0, 4),
            "status": status,
        })

    return sorted(results, key=lambda x: -x["vif"])


def _recommend_action(abs_r: float) -> str:
    """Recommend action for a correlated pair."""
    if abs_r > 0.80:
        return "MERGE: sources are near-duplicates; combine into single weighted signal"
    elif abs_r > 0.60:
        return "REDUCE: down-weight both sources by 30% when used together"
    elif abs_r > _INDEPENDENCE_THRESHOLD:
        return "MONITOR: mild correlation; apply 10% discount when combining"
    else:
        return "OK: sources are sufficiently independent"


# ── Convenience: build signal matrix from source instances ──────────────

def build_signal_matrix(
    sources: list,
    observations: list[dict],
) -> tuple[np.ndarray, list[str]]:
    """
    Build a (n_obs, n_sources) signal matrix by running each source on each
    observation.

    Parameters
    ----------
    sources : list of source instances
        Each must have get_signal(player, tournament_context).
    observations : list of dicts
        Each dict has 'player' and 'tournament_context' keys.

    Returns
    -------
    (signal_matrix, source_names)
    """
    n_obs = len(observations)
    n_src = len(sources)
    matrix = np.zeros((n_obs, n_src), dtype=float)
    names = [getattr(s, "name", f"Source_{i}") for i, s in enumerate(sources)]

    for j, src in enumerate(sources):
        for i, obs in enumerate(observations):
            try:
                matrix[i, j] = src.get_signal(
                    obs.get("player", {}),
                    obs.get("tournament_context", {}),
                )
            except Exception:
                matrix[i, j] = 0.0

    return matrix, names


import math  # noqa: E402 (already at top but keeping import explicit)
