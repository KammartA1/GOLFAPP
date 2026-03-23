"""Probability perturbation test — What happens when model probabilities are wrong?

This test systematically perturbs model probabilities by various amounts
and measures the impact on system decisions and PnL. A robust system should
degrade gracefully, not collapse when probabilities are slightly off.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerturbationResult:
    """Result of a single perturbation scenario."""
    perturbation_type: str
    perturbation_magnitude: float
    n_bets: int
    original_roi: float
    perturbed_roi: float
    roi_delta: float
    original_clv: float
    perturbed_clv: float
    original_approved_pct: float
    perturbed_approved_pct: float
    bets_changed: int  # How many bets flipped approval
    max_loss_increase: float  # Worst-case loss increase


class ProbabilityPerturbationTest:
    """Test system robustness to probability estimation errors.

    Perturbation Types:
        1. Uniform bias: Add constant to all probabilities
        2. Multiplicative bias: Scale all probabilities
        3. Random noise: Add Gaussian noise
        4. Overconfidence: Push probabilities away from 0.5
        5. Underconfidence: Push probabilities toward 0.5
        6. Correlated error: Errors correlated with true probability
    """

    PERTURBATION_MAGNITUDES = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def run_all(self, bet_records: List[dict]) -> dict:
        """Run all perturbation tests on historical bet records.

        Args:
            bet_records: List of dicts with keys:
                model_prob, market_prob, odds_decimal, stake, pnl, status

        Returns:
            Full test report with all perturbation results.
        """
        if not bet_records or len(bet_records) < 10:
            return {"error": "Need at least 10 bet records", "n_records": len(bet_records)}

        results = {}

        for ptype in [
            "uniform_bias_up", "uniform_bias_down",
            "multiplicative_up", "multiplicative_down",
            "random_noise", "overconfidence", "underconfidence",
            "correlated_positive", "correlated_negative",
        ]:
            results[ptype] = self._run_perturbation_type(bet_records, ptype)

        # Summary
        worst_case = self._find_worst_case(results)
        robustness_score = self._compute_robustness_score(results)

        return {
            "n_bet_records": len(bet_records),
            "perturbation_types_tested": len(results),
            "results": results,
            "worst_case": worst_case,
            "robustness_score": round(robustness_score, 2),
            "verdict": "robust" if robustness_score >= 70 else
                       "marginal" if robustness_score >= 40 else "fragile",
        }

    def _run_perturbation_type(
        self, bet_records: List[dict], ptype: str
    ) -> List[dict]:
        """Run perturbation at all magnitudes for a specific type."""
        results = []
        for magnitude in self.PERTURBATION_MAGNITUDES:
            result = self._evaluate_perturbation(bet_records, ptype, magnitude)
            results.append(result)
        return results

    def _evaluate_perturbation(
        self,
        bet_records: List[dict],
        ptype: str,
        magnitude: float,
    ) -> dict:
        """Evaluate a single perturbation scenario."""
        probs = np.array([r["model_prob"] for r in bet_records])
        market_probs = np.array([r["market_prob"] for r in bet_records])
        odds = np.array([r["odds_decimal"] for r in bet_records])
        stakes = np.array([r["stake"] for r in bet_records])
        pnl = np.array([r["pnl"] for r in bet_records])
        outcomes = np.array([1.0 if r["status"] == "won" else 0.0 for r in bet_records])

        # Apply perturbation
        perturbed = self._perturb_probabilities(probs, ptype, magnitude)

        # Compute original metrics
        original_edges = probs - market_probs
        original_approved = original_edges > 0.02  # Minimum edge threshold
        original_roi = float(np.sum(pnl)) / max(float(np.sum(stakes)), 1)

        # Recompute with perturbed probabilities
        perturbed_edges = perturbed - market_probs
        perturbed_approved = perturbed_edges > 0.02

        # Simulate PnL with perturbed sizing
        perturbed_kelly_fractions = np.clip(
            (perturbed * odds - 1) / (odds - 1) * 0.25,  # Quarter Kelly
            0, 0.05,
        )
        perturbed_stakes = perturbed_kelly_fractions * 1000  # Assume $1000 bankroll
        perturbed_pnl = np.where(
            perturbed_approved,
            outcomes * perturbed_stakes * (odds - 1) - (1 - outcomes) * perturbed_stakes,
            0,
        )
        perturbed_total_staked = float(np.sum(perturbed_stakes[perturbed_approved]))
        perturbed_roi = float(np.sum(perturbed_pnl)) / max(perturbed_total_staked, 1)

        # CLV proxy
        original_clv = float(np.mean(original_edges[original_approved])) * 100 if np.any(original_approved) else 0.0
        perturbed_clv = float(np.mean(perturbed_edges[perturbed_approved])) * 100 if np.any(perturbed_approved) else 0.0

        bets_changed = int(np.sum(original_approved != perturbed_approved))

        return {
            "perturbation_type": ptype,
            "magnitude": magnitude,
            "n_bets": len(bet_records),
            "original_roi": round(original_roi, 4),
            "perturbed_roi": round(perturbed_roi, 4),
            "roi_delta": round(perturbed_roi - original_roi, 4),
            "original_clv": round(original_clv, 2),
            "perturbed_clv": round(perturbed_clv, 2),
            "original_approved_pct": round(float(np.mean(original_approved)), 4),
            "perturbed_approved_pct": round(float(np.mean(perturbed_approved)), 4),
            "bets_changed": bets_changed,
            "bets_changed_pct": round(bets_changed / len(bet_records), 4),
        }

    def _perturb_probabilities(
        self, probs: np.ndarray, ptype: str, magnitude: float
    ) -> np.ndarray:
        """Apply perturbation to probability array."""
        if ptype == "uniform_bias_up":
            perturbed = probs + magnitude
        elif ptype == "uniform_bias_down":
            perturbed = probs - magnitude
        elif ptype == "multiplicative_up":
            perturbed = probs * (1 + magnitude)
        elif ptype == "multiplicative_down":
            perturbed = probs * (1 - magnitude)
        elif ptype == "random_noise":
            noise = self.rng.normal(0, magnitude, size=len(probs))
            perturbed = probs + noise
        elif ptype == "overconfidence":
            # Push away from 0.5
            deviation = probs - 0.5
            perturbed = 0.5 + deviation * (1 + magnitude * 2)
        elif ptype == "underconfidence":
            # Push toward 0.5
            deviation = probs - 0.5
            perturbed = 0.5 + deviation * (1 - magnitude * 2)
        elif ptype == "correlated_positive":
            # Higher probs get more positive error
            error = magnitude * (probs - 0.5) * 2
            perturbed = probs + error
        elif ptype == "correlated_negative":
            # Higher probs get more negative error
            error = -magnitude * (probs - 0.5) * 2
            perturbed = probs + error
        else:
            perturbed = probs.copy()

        return np.clip(perturbed, 0.01, 0.99)

    def _find_worst_case(self, results: dict) -> dict:
        """Find the worst-case scenario across all perturbations."""
        worst = {"roi_delta": 0.0}
        for ptype, scenarios in results.items():
            for scenario in scenarios:
                if scenario["roi_delta"] < worst["roi_delta"]:
                    worst = scenario
        return worst

    def _compute_robustness_score(self, results: dict) -> float:
        """Compute 0-100 robustness score from perturbation results.

        Measures how much performance degrades under perturbation.
        Score of 100 = perfectly robust (no degradation).
        Score of 0 = completely fragile (collapses at smallest perturbation).
        """
        degradations = []

        for ptype, scenarios in results.items():
            for scenario in scenarios:
                magnitude = scenario["magnitude"]
                roi_delta = abs(scenario["roi_delta"])
                # Normalize: how much degradation per unit of perturbation
                sensitivity = roi_delta / max(magnitude, 0.001)
                degradations.append(sensitivity)

        if not degradations:
            return 50.0

        avg_sensitivity = float(np.mean(degradations))
        max_sensitivity = float(np.max(degradations))

        # Score: lower sensitivity → higher score
        # Sensitivity of 0.5 = losing 0.5% ROI per 1% perturbation = okay
        # Sensitivity of 5.0 = losing 5% ROI per 1% perturbation = bad
        score = 100.0 - min(100.0, avg_sensitivity * 20 + max_sensitivity * 5)
        return max(0.0, score)
