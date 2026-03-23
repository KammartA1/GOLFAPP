"""
edge_analysis/predictive.py
============================
Component 1: PREDICTIVE EDGE — Golf probability accuracy vs actuals.

Golf-specific: tracks Brier score for multiple market types:
  - Tournament winner, Top 5, Top 10, Top 20, Make Cut, Matchups
  - Each market type has different baseline difficulty

Computes per-market Brier score, log loss, calibration, and overall predictive attribution.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy import stats as sp_stats

from edge_analysis.schemas import GolfBetRecord, CalibrationPoint, EdgeComponentResult

log = logging.getLogger(__name__)

# Calibration bucket edges
_BUCKET_EDGES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]

# Golf market type base rates (approximate)
_MARKET_BASE_RATES = {
    "outright": 0.006,      # ~1/156 field
    "top5": 0.032,          # ~5/156
    "top10": 0.064,         # ~10/156
    "top20": 0.128,         # ~20/156
    "make_cut": 0.54,       # ~70/130 (after WDs)
    "matchup": 0.50,        # Head-to-head
}


def _brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def _log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(outcomes * np.log(p) + (1.0 - outcomes) * np.log(1.0 - p)))


def _build_calibration_curve(probs: np.ndarray, outcomes: np.ndarray) -> List[CalibrationPoint]:
    points = []
    for i in range(len(_BUCKET_EDGES) - 1):
        lo, hi = _BUCKET_EDGES[i], _BUCKET_EDGES[i + 1]
        mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n < 3:
            continue
        pred_avg = float(np.mean(probs[mask]))
        actual_rate = float(np.mean(outcomes[mask]))
        cal_err = abs(pred_avg - actual_rate)
        points.append(CalibrationPoint(
            bucket_lower=lo, bucket_upper=hi,
            predicted_avg=round(pred_avg, 4),
            actual_rate=round(actual_rate, 4),
            n_bets=n,
            calibration_error=round(cal_err, 4),
        ))
    return points


def _skill_score(model_score: float, baseline_score: float) -> float:
    if baseline_score <= 0:
        return 0.0
    return 1.0 - (model_score / baseline_score)


def compute_predictive_edge(
    bets: List[GolfBetRecord],
    total_roi: float,
) -> EdgeComponentResult:
    """Analyze predictive accuracy across golf market types.

    Tracks per-market Brier score: outrights are notoriously hard to predict
    (base rate ~0.6%), while matchups are 50/50 baselines.
    """
    settled = [b for b in bets if b.won is not None and b.predicted_prob > 0]
    if len(settled) < 15:
        return EdgeComponentResult(
            name="predictive",
            edge_pct_of_roi=0.0, absolute_value=0.0, p_value=1.0,
            is_significant=False, is_positive=False,
            sample_size=len(settled),
            verdict="Insufficient data for predictive edge analysis (need 15+ settled bets)",
        )

    # Overall metrics
    model_probs = np.array([b.predicted_prob for b in settled])
    market_probs = np.array([b.market_prob_at_bet for b in settled])
    outcomes = np.array([1.0 if b.won else 0.0 for b in settled])

    brier_model = _brier_score(model_probs, outcomes)
    brier_market = _brier_score(market_probs, outcomes)
    brier_skill = _skill_score(brier_model, brier_market)

    logloss_model = _log_loss(model_probs, outcomes)
    logloss_market = _log_loss(market_probs, outcomes)
    logloss_skill = _skill_score(logloss_model, logloss_market)

    cal_curve = _build_calibration_curve(model_probs, outcomes)
    mean_cal_error = float(np.mean([p.calibration_error for p in cal_curve])) if cal_curve else 0.0

    # Paired t-test on squared errors
    model_sq = (model_probs - outcomes) ** 2
    market_sq = (market_probs - outcomes) ** 2
    diff = market_sq - model_sq
    t_stat, p_two = (0.0, 1.0)
    if np.std(diff) > 0:
        t_stat, p_two = sp_stats.ttest_1samp(diff, 0.0)
        t_stat = float(t_stat)
        p_two = float(p_two)
    p_value = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2

    # Per-market-type breakdown
    by_market: Dict[str, Dict] = {}
    market_groups: Dict[str, List[GolfBetRecord]] = defaultdict(list)
    for b in settled:
        market_groups[b.market_type].append(b)

    for mtype, group in market_groups.items():
        m_probs = np.array([b.predicted_prob for b in group])
        mk_probs = np.array([b.market_prob_at_bet for b in group])
        m_out = np.array([1.0 if b.won else 0.0 for b in group])

        if len(group) < 5:
            by_market[mtype] = {"n": len(group), "insufficient": True}
            continue

        bm = _brier_score(m_probs, m_out)
        bmk = _brier_score(mk_probs, m_out)
        base_rate = _MARKET_BASE_RATES.get(mtype, 0.5)
        naive_brier = base_rate * (1 - base_rate) ** 2 + (1 - base_rate) * base_rate ** 2

        by_market[mtype] = {
            "n": len(group),
            "brier_model": round(bm, 6),
            "brier_market": round(bmk, 6),
            "brier_naive": round(naive_brier, 6),
            "skill_vs_market": round(_skill_score(bm, bmk), 4),
            "skill_vs_naive": round(_skill_score(bm, naive_brier), 4),
            "win_rate": round(float(np.mean(m_out)), 4),
            "avg_predicted_prob": round(float(np.mean(m_probs)), 4),
        }

    is_positive = brier_skill > 0 and logloss_skill > 0
    is_significant = p_value < 0.05
    avg_skill = (brier_skill + logloss_skill) / 2.0
    predictive_pct = max(0.0, min(1.0, avg_skill)) * 100.0

    # Verdict
    verdict_parts = []
    if is_positive and is_significant:
        verdict_parts.append(
            f"REAL predictive edge. Model Brier {brier_model:.4f} vs market {brier_market:.4f} "
            f"(skill={brier_skill:.1%}). p={p_value:.4f}."
        )
    elif is_positive:
        verdict_parts.append(
            f"Possible predictive edge but NOT significant (p={p_value:.4f}). "
            f"Model Brier {brier_model:.4f} vs market {brier_market:.4f}."
        )
    else:
        verdict_parts.append(
            f"NO predictive edge. Model Brier {brier_model:.4f} vs market {brier_market:.4f}."
        )

    # Highlight best/worst market types
    real_markets = {k: v for k, v in by_market.items() if not v.get("insufficient")}
    if real_markets:
        best = max(real_markets.items(), key=lambda x: x[1].get("skill_vs_market", 0))
        worst = min(real_markets.items(), key=lambda x: x[1].get("skill_vs_market", 0))
        verdict_parts.append(
            f"Best market: {best[0]} (skill={best[1]['skill_vs_market']:.1%}). "
            f"Worst: {worst[0]} (skill={worst[1]['skill_vs_market']:.1%})."
        )

    return EdgeComponentResult(
        name="predictive",
        edge_pct_of_roi=round(predictive_pct, 2),
        absolute_value=round(brier_skill, 4),
        p_value=round(p_value, 4),
        is_significant=is_significant,
        is_positive=is_positive,
        sample_size=len(settled),
        details={
            "brier_model": round(brier_model, 6),
            "brier_market": round(brier_market, 6),
            "brier_skill": round(brier_skill, 4),
            "logloss_model": round(logloss_model, 6),
            "logloss_market": round(logloss_market, 6),
            "logloss_skill": round(logloss_skill, 4),
            "mean_calibration_error": round(mean_cal_error, 4),
            "by_market_type": by_market,
            "calibration_curve": [
                {
                    "bucket": f"{p.bucket_lower:.0%}-{p.bucket_upper:.0%}",
                    "predicted": p.predicted_avg,
                    "actual": p.actual_rate,
                    "n": p.n_bets,
                    "error": p.calibration_error,
                }
                for p in cal_curve
            ],
        },
        verdict=" ".join(verdict_parts),
    )
