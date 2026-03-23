"""Predictive edge component — Model accuracy metrics.

Measures how well the model predicts outcomes:
  - Brier score per market type
  - Log loss per market type
  - Calibration curves
  - Skill score vs naive baseline
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List

import numpy as np
from scipy import stats as scipy_stats

from edge_analysis.schemas import GolfBetRecord, EdgeComponent

logger = logging.getLogger(__name__)


def brier_score(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Brier score: mean squared error of probability predictions.

    Lower is better. Range [0, 1].
    Perfect = 0.0, always predicting 0.5 = 0.25, worst = 1.0.
    """
    predicted = np.clip(predicted, 0.001, 0.999)
    return float(np.mean((predicted - actual) ** 2))


def log_loss(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Log loss (cross-entropy). Lower is better.

    Heavily penalizes confident wrong predictions.
    """
    predicted = np.clip(predicted, 0.001, 0.999)
    losses = -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
    return float(np.mean(losses))


def brier_skill_score(model_brier: float, baseline_brier: float) -> float:
    """Brier Skill Score: improvement over baseline (naive base rate).

    BSS = 1 - (model_brier / baseline_brier)
    Positive = better than baseline. 1.0 = perfect. 0.0 = no skill.
    """
    if baseline_brier <= 0:
        return 0.0
    return 1.0 - (model_brier / baseline_brier)


def calibration_curve(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Compute calibration curve: predicted vs actual hit rates in bins.

    Returns list of dicts with bin boundaries, predicted avg, actual rate, count.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    result = []

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (predicted >= low) & (predicted < high)
        if i == n_bins - 1:
            mask = (predicted >= low) & (predicted <= high)

        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue

        pred_avg = float(predicted[mask].mean())
        actual_rate = float(actual[mask].mean())
        cal_error = abs(pred_avg - actual_rate)

        result.append({
            "prob_lower": round(low, 2),
            "prob_upper": round(high, 2),
            "predicted_avg": round(pred_avg, 4),
            "actual_rate": round(actual_rate, 4),
            "n_bets": n_in_bin,
            "calibration_error": round(cal_error, 4),
            "is_overconfident": pred_avg > actual_rate,
        })

    return result


def expected_calibration_error(predicted: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error — weighted average of bin calibration errors."""
    curve = calibration_curve(predicted, actual, n_bins)
    if not curve:
        return 0.0
    total_n = sum(b["n_bets"] for b in curve)
    if total_n == 0:
        return 0.0
    ece = sum(b["calibration_error"] * b["n_bets"] / total_n for b in curve)
    return round(ece, 4)


class PredictiveAnalyzer:
    """Analyze the predictive component of edge."""

    def analyze(self, records: List[GolfBetRecord]) -> EdgeComponent:
        """Compute predictive edge metrics across all bets and per market type."""
        if not records:
            return EdgeComponent(
                name="predictive", value=0.0, confidence=0.0,
                verdict="No data to analyze.",
            )

        predicted = np.array([r.predicted_prob for r in records])
        actual = np.array([r.actual_outcome for r in records])

        # Overall metrics
        overall_brier = brier_score(predicted, actual)
        overall_logloss = log_loss(predicted, actual)
        base_rate = float(actual.mean())
        baseline_brier = base_rate * (1 - base_rate) + (1 - base_rate) * base_rate
        # Baseline brier for constant predictor = base_rate
        naive_brier = brier_score(np.full_like(predicted, base_rate), actual)
        bss = brier_skill_score(overall_brier, naive_brier)
        ece = expected_calibration_error(predicted, actual)
        cal_curve = calibration_curve(predicted, actual)

        # Per market type
        by_market = self._per_market_metrics(records)

        # Statistical significance: test if model beats naive
        # Using paired t-test on squared errors
        model_errors = (predicted - actual) ** 2
        naive_errors = (base_rate - actual) ** 2
        if len(model_errors) >= 10:
            t_stat, p_val = scipy_stats.ttest_rel(naive_errors, model_errors)
            # One-sided: model better than naive
            p_value = p_val / 2 if t_stat > 0 else 1.0 - p_val / 2
        else:
            p_value = 1.0

        # Edge value in cents: average probability advantage * 100
        avg_edge = float(np.mean(predicted - actual))
        # Predictive value = how much better model is than market
        pred_value = float(np.mean(predicted[actual == 1]) - base_rate) if actual.sum() > 0 else 0.0
        edge_cents = pred_value * 100

        confidence = min(len(records) / 200.0, 1.0) * (1.0 - min(p_value, 1.0))

        # Verdict
        if bss > 0.05 and p_value < 0.05:
            verdict = f"Strong predictive edge (BSS={bss:.3f}, p={p_value:.4f})"
        elif bss > 0.02 and p_value < 0.10:
            verdict = f"Moderate predictive edge (BSS={bss:.3f}, p={p_value:.4f})"
        elif bss > 0:
            verdict = f"Weak predictive edge, not statistically significant (BSS={bss:.3f}, p={p_value:.4f})"
        else:
            verdict = f"No predictive edge detected (BSS={bss:.3f})"

        return EdgeComponent(
            name="predictive",
            value=round(edge_cents, 2),
            confidence=round(confidence, 3),
            details={
                "brier_score": round(overall_brier, 4),
                "log_loss": round(overall_logloss, 4),
                "brier_skill_score": round(bss, 4),
                "expected_calibration_error": ece,
                "calibration_curve": cal_curve,
                "base_rate": round(base_rate, 4),
                "p_value": round(p_value, 4),
                "n_bets": len(records),
                "by_market_type": by_market,
            },
            verdict=verdict,
        )

    def _per_market_metrics(self, records: List[GolfBetRecord]) -> dict:
        """Compute Brier/logloss per market type."""
        grouped = defaultdict(list)
        for r in records:
            grouped[r.market_type].append(r)

        result = {}
        for mtype, recs in grouped.items():
            pred = np.array([r.predicted_prob for r in recs])
            act = np.array([r.actual_outcome for r in recs])
            if len(pred) < 3:
                continue
            result[mtype] = {
                "n_bets": len(recs),
                "brier_score": round(brier_score(pred, act), 4),
                "log_loss": round(log_loss(pred, act), 4),
                "avg_predicted": round(float(pred.mean()), 4),
                "actual_rate": round(float(act.mean()), 4),
                "calibration_error": round(abs(float(pred.mean() - act.mean())), 4),
            }
        return result
