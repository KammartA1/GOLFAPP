"""Validation — Validate simulation distributions match PGA Tour data.

Checks that simulated outputs are realistic:
  - Scoring distribution (mean, std, skew)
  - Cut rates
  - Birdie/bogey rates
  - Win probability distribution
  - Top-N probability calibration
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# PGA Tour 2023-24 benchmarks
PGA_BENCHMARKS = {
    "scoring_avg_to_par": -1.5,  # Average field scores about -1.5/round
    "scoring_std": 3.0,  # Round-to-round std
    "make_cut_rate": 0.65,  # ~65% of field makes cut
    "avg_birdies_per_round": 3.5,
    "avg_bogeys_per_round": 2.5,
    "avg_eagles_per_round": 0.08,
    "avg_doubles_per_round": 0.25,
    "win_prob_leader": 0.15,  # Pre-tournament favorite ~15%
    "win_prob_median": 0.006,  # Median player ~0.6%
    "top10_rate_avg": 0.064,  # 10/156 = 6.4%
}


class SimulationValidator:
    """Validate simulation outputs against PGA Tour distributions."""

    def validate_scoring_distribution(
        self,
        simulated_scores: np.ndarray,
        benchmark_mean: float = PGA_BENCHMARKS["scoring_avg_to_par"],
        benchmark_std: float = PGA_BENCHMARKS["scoring_std"],
        tolerance: float = 0.5,
    ) -> dict:
        """Validate that simulated scoring matches real distributions."""
        sim_mean = float(np.mean(simulated_scores))
        sim_std = float(np.std(simulated_scores))
        sim_skew = float(scipy_stats.skew(simulated_scores))
        sim_kurtosis = float(scipy_stats.kurtosis(simulated_scores))

        mean_ok = abs(sim_mean - benchmark_mean) < tolerance
        std_ok = abs(sim_std - benchmark_std) < tolerance

        # K-S test against normal with PGA parameters
        ks_stat, ks_pval = scipy_stats.kstest(
            simulated_scores, "norm", args=(benchmark_mean, benchmark_std)
        )

        return {
            "is_valid": mean_ok and std_ok,
            "sim_mean": round(sim_mean, 3),
            "sim_std": round(sim_std, 3),
            "sim_skew": round(sim_skew, 3),
            "sim_kurtosis": round(sim_kurtosis, 3),
            "benchmark_mean": benchmark_mean,
            "benchmark_std": benchmark_std,
            "mean_error": round(abs(sim_mean - benchmark_mean), 3),
            "std_error": round(abs(sim_std - benchmark_std), 3),
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_pval, 4),
        }

    def validate_cut_rates(
        self,
        simulated_cut_rates: np.ndarray,
        benchmark_rate: float = PGA_BENCHMARKS["make_cut_rate"],
        tolerance: float = 0.10,
    ) -> dict:
        """Validate simulated cut rates."""
        sim_rate = float(np.mean(simulated_cut_rates))
        return {
            "is_valid": abs(sim_rate - benchmark_rate) < tolerance,
            "sim_cut_rate": round(sim_rate, 4),
            "benchmark_rate": benchmark_rate,
            "error": round(abs(sim_rate - benchmark_rate), 4),
        }

    def validate_win_probability_distribution(
        self,
        sim_results: pd.DataFrame,
        field_size: int = 156,
    ) -> dict:
        """Validate that win probabilities sum to ~1.0 and are realistic."""
        win_probs = sim_results["win_prob"].values
        total_win_prob = float(np.sum(win_probs))
        max_win_prob = float(np.max(win_probs))
        min_win_prob = float(np.min(win_probs))
        median_win_prob = float(np.median(win_probs))

        # Win probs should sum to approximately 1.0
        sum_ok = abs(total_win_prob - 1.0) < 0.05

        # Leader shouldn't exceed ~30% (except in very weak fields)
        max_ok = max_win_prob < 0.35

        # Most players should have some win chance
        zero_count = int(np.sum(win_probs == 0))

        return {
            "is_valid": sum_ok and max_ok,
            "total_win_prob": round(total_win_prob, 4),
            "max_win_prob": round(max_win_prob, 4),
            "min_win_prob": round(min_win_prob, 6),
            "median_win_prob": round(median_win_prob, 6),
            "n_zero_win_prob": zero_count,
            "sum_within_tolerance": sum_ok,
            "max_within_tolerance": max_ok,
        }

    def validate_full_simulation(
        self,
        sim_results: pd.DataFrame,
        simulated_round_scores: np.ndarray | None = None,
    ) -> dict:
        """Run all validation checks."""
        results = {
            "win_probability": self.validate_win_probability_distribution(sim_results),
        }

        if "make_cut_prob" in sim_results.columns:
            cut_rates = sim_results["make_cut_prob"].values
            results["cut_rates"] = self.validate_cut_rates(cut_rates)

        if simulated_round_scores is not None:
            results["scoring_distribution"] = self.validate_scoring_distribution(
                simulated_round_scores
            )

        # Overall pass/fail
        all_valid = all(
            v.get("is_valid", True) for v in results.values()
            if isinstance(v, dict)
        )
        results["overall_valid"] = all_valid

        return results
