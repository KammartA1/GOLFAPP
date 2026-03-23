"""Edge Attribution Engine — Counterfactual analysis and Monte Carlo significance.

Advanced attribution methods:
  - Counterfactual analysis: What if we removed each edge source?
  - Monte Carlo significance: Bootstrap confidence intervals
  - CLV attribution: Decompose CLV into source contributions
  - Market inefficiency segmentation: Which markets are exploitable?
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List

import numpy as np
from scipy import stats as scipy_stats

from edge_analysis.schemas import GolfBetRecord, EdgeReport
from edge_analysis.decomposer import GolfEdgeDecomposer

logger = logging.getLogger(__name__)


class EdgeAttributionEngine:
    """Advanced edge attribution with counterfactual and Monte Carlo methods."""

    def __init__(self, n_bootstrap: int = 5000, confidence_level: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.decomposer = GolfEdgeDecomposer()

    def run_full_attribution(self, records: List[GolfBetRecord]) -> dict:
        """Run complete attribution analysis.

        Returns dict with:
          - decomposition: Full 5-component EdgeReport
          - counterfactual: What-if analysis removing each source
          - monte_carlo: Bootstrap confidence intervals
          - clv_attribution: CLV broken down by source
          - market_segments: Per-market inefficiency analysis
        """
        if len(records) < 20:
            return {
                "error": f"Insufficient data ({len(records)} bets). Need at least 20.",
                "decomposition": None,
                "counterfactual": {},
                "monte_carlo": {},
                "clv_attribution": {},
                "market_segments": {},
            }

        logger.info("Running full attribution on %d records", len(records))

        decomposition = self.decomposer.full_decomposition(records)
        counterfactual = self._counterfactual_analysis(records)
        monte_carlo = self._monte_carlo_significance(records)
        clv_attribution = self._clv_attribution(records)
        market_segments = self._market_inefficiency_segmentation(records)

        return {
            "decomposition": decomposition,
            "counterfactual": counterfactual,
            "monte_carlo": monte_carlo,
            "clv_attribution": clv_attribution,
            "market_segments": market_segments,
        }

    def _counterfactual_analysis(self, records: List[GolfBetRecord]) -> dict:
        """What-if analysis: how would performance change without each source?

        For each data source, remove bets that used it and see how
        overall metrics change. The difference is that source's contribution.
        """
        all_sources = set()
        for r in records:
            all_sources.update(r.data_sources_available)

        if not all_sources:
            # If no source tracking, do counterfactual by market type
            return self._counterfactual_by_market(records)

        baseline_clv = float(np.mean([r.clv_cents for r in records]))
        baseline_roi = sum(r.pnl for r in records) / max(sum(r.stake for r in records), 1)

        results = {}
        for source in all_sources:
            # Remove bets that used this source
            without = [r for r in records if source not in r.data_sources_available]
            only_with = [r for r in records if source in r.data_sources_available]

            if len(without) < 5 or len(only_with) < 5:
                results[source] = {
                    "insufficient_data": True,
                    "n_bets_with": len(only_with),
                    "n_bets_without": len(without),
                }
                continue

            without_clv = float(np.mean([r.clv_cents for r in without]))
            without_staked = sum(r.stake for r in without)
            without_roi = sum(r.pnl for r in without) / max(without_staked, 1)

            with_clv = float(np.mean([r.clv_cents for r in only_with]))
            with_staked = sum(r.stake for r in only_with)
            with_roi = sum(r.pnl for r in only_with) / max(with_staked, 1)

            # Marginal contribution
            clv_contribution = baseline_clv - without_clv
            roi_contribution = baseline_roi - without_roi

            results[source] = {
                "n_bets_with": len(only_with),
                "n_bets_without": len(without),
                "baseline_clv": round(baseline_clv, 2),
                "without_clv": round(without_clv, 2),
                "with_only_clv": round(with_clv, 2),
                "marginal_clv_contribution": round(clv_contribution, 2),
                "with_roi": round(with_roi, 4),
                "without_roi": round(without_roi, 4),
                "marginal_roi_contribution": round(roi_contribution, 4),
                "is_positive_contributor": clv_contribution > 0,
            }

        # Rank sources by contribution
        ranked = sorted(
            [(s, d) for s, d in results.items() if not d.get("insufficient_data")],
            key=lambda x: x[1].get("marginal_clv_contribution", 0),
            reverse=True,
        )
        results["_ranking"] = [s for s, _ in ranked]

        return results

    def _counterfactual_by_market(self, records: List[GolfBetRecord]) -> dict:
        """Counterfactual by market type when source data unavailable."""
        from edge_analysis.schemas import MARKET_TYPES

        baseline_clv = float(np.mean([r.clv_cents for r in records]))
        results = {}

        for mtype in MARKET_TYPES:
            recs_of_type = [r for r in records if r.market_type == mtype]
            recs_without = [r for r in records if r.market_type != mtype]

            if len(recs_of_type) < 3 or len(recs_without) < 3:
                continue

            without_clv = float(np.mean([r.clv_cents for r in recs_without]))
            with_clv = float(np.mean([r.clv_cents for r in recs_of_type]))

            results[mtype] = {
                "n_bets": len(recs_of_type),
                "avg_clv": round(with_clv, 2),
                "clv_without_this_market": round(without_clv, 2),
                "marginal_contribution": round(baseline_clv - without_clv, 2),
            }

        return results

    def _monte_carlo_significance(self, records: List[GolfBetRecord]) -> dict:
        """Bootstrap confidence intervals for edge estimates.

        Resample bets with replacement N times to estimate:
          - CI for CLV
          - CI for ROI
          - Probability of positive edge
          - Required sample for significance
        """
        n = len(records)
        clvs = np.array([r.clv_cents for r in records])
        pnls = np.array([r.pnl for r in records])
        stakes = np.array([r.stake for r in records])

        rng = np.random.default_rng(42)

        bootstrap_clvs = np.zeros(self.n_bootstrap)
        bootstrap_rois = np.zeros(self.n_bootstrap)
        bootstrap_win_rates = np.zeros(self.n_bootstrap)

        outcomes = np.array([r.actual_outcome for r in records])

        for i in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            bootstrap_clvs[i] = float(np.mean(clvs[idx]))
            total_staked = float(np.sum(stakes[idx]))
            bootstrap_rois[i] = float(np.sum(pnls[idx])) / max(total_staked, 1)
            bootstrap_win_rates[i] = float(np.mean(outcomes[idx]))

        alpha = 1 - self.confidence_level

        clv_ci = (
            round(float(np.percentile(bootstrap_clvs, alpha / 2 * 100)), 2),
            round(float(np.percentile(bootstrap_clvs, (1 - alpha / 2) * 100)), 2),
        )
        roi_ci = (
            round(float(np.percentile(bootstrap_rois, alpha / 2 * 100)), 4),
            round(float(np.percentile(bootstrap_rois, (1 - alpha / 2) * 100)), 4),
        )
        wr_ci = (
            round(float(np.percentile(bootstrap_win_rates, alpha / 2 * 100)), 4),
            round(float(np.percentile(bootstrap_win_rates, (1 - alpha / 2) * 100)), 4),
        )

        prob_positive_edge = float(np.mean(bootstrap_clvs > 0))
        prob_profitable = float(np.mean(bootstrap_rois > 0))

        # Estimate required sample for significance
        observed_effect = float(np.mean(clvs))
        observed_std = float(np.std(clvs))
        if observed_effect > 0 and observed_std > 0:
            # Required n for t-test at alpha=0.05, power=0.80
            z_alpha = 1.96
            z_beta = 0.84
            required_n = int(np.ceil(((z_alpha + z_beta) * observed_std / observed_effect) ** 2))
        else:
            required_n = 999

        return {
            "n_bootstrap": self.n_bootstrap,
            "confidence_level": self.confidence_level,
            "clv_mean": round(float(np.mean(bootstrap_clvs)), 2),
            "clv_ci": clv_ci,
            "roi_mean": round(float(np.mean(bootstrap_rois)), 4),
            "roi_ci": roi_ci,
            "win_rate_ci": wr_ci,
            "prob_positive_edge": round(prob_positive_edge, 4),
            "prob_profitable": round(prob_profitable, 4),
            "required_sample_for_significance": min(required_n, 9999),
            "current_sample": n,
            "sample_sufficient": n >= required_n,
        }

    def _clv_attribution(self, records: List[GolfBetRecord]) -> dict:
        """Attribute CLV to different factors using regression decomposition."""
        if len(records) < 20:
            return {"insufficient_data": True}

        # Build feature matrix for CLV attribution
        clvs = np.array([r.clv_cents for r in records])

        # Features
        edges = np.array([r.edge for r in records])
        is_am = np.array([1.0 if r.wave == "AM" else 0.0 for r in records])
        is_weather = np.array([1.0 if r.weather_conditions != "normal" else 0.0 for r in records])

        market_dummies = {}
        from edge_analysis.schemas import MARKET_TYPES
        for mtype in MARKET_TYPES:
            market_dummies[mtype] = np.array([1.0 if r.market_type == mtype else 0.0 for r in records])

        # Simple attribution via correlation
        results = {
            "edge_clv_correlation": round(float(np.corrcoef(edges, clvs)[0, 1]), 4) if len(edges) > 2 else 0.0,
            "wave_am_avg_clv": round(float(np.mean(clvs[is_am == 1])), 2) if is_am.sum() > 0 else 0.0,
            "wave_pm_avg_clv": round(float(np.mean(clvs[is_am == 0])), 2) if (is_am == 0).sum() > 0 else 0.0,
            "weather_avg_clv": round(float(np.mean(clvs[is_weather == 1])), 2) if is_weather.sum() > 0 else 0.0,
            "normal_avg_clv": round(float(np.mean(clvs[is_weather == 0])), 2) if (is_weather == 0).sum() > 0 else 0.0,
        }

        # Per-market CLV
        for mtype in MARKET_TYPES:
            mask = market_dummies[mtype]
            if mask.sum() >= 3:
                results[f"clv_{mtype}"] = round(float(np.mean(clvs[mask == 1])), 2)
                results[f"n_{mtype}"] = int(mask.sum())

        # Regression attribution (OLS)
        try:
            X = np.column_stack([edges, is_am, is_weather])
            X = np.column_stack([X, np.ones(len(X))])  # intercept
            # OLS: beta = (X'X)^-1 X'y
            beta = np.linalg.lstsq(X, clvs, rcond=None)[0]
            results["regression_coefficients"] = {
                "edge": round(float(beta[0]), 4),
                "am_wave": round(float(beta[1]), 4),
                "weather": round(float(beta[2]), 4),
                "intercept": round(float(beta[3]), 4),
            }
            # R-squared
            y_hat = X @ beta
            ss_res = float(np.sum((clvs - y_hat) ** 2))
            ss_tot = float(np.sum((clvs - np.mean(clvs)) ** 2))
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            results["r_squared"] = round(r_squared, 4)
        except (np.linalg.LinAlgError, ValueError):
            results["regression_coefficients"] = None
            results["r_squared"] = 0.0

        return results

    def _market_inefficiency_segmentation(self, records: List[GolfBetRecord]) -> dict:
        """Segment markets by inefficiency level and sustainability."""
        grouped = defaultdict(list)
        for r in records:
            grouped[r.market_type].append(r)

        segments = {}
        for mtype, recs in grouped.items():
            if len(recs) < 10:
                segments[mtype] = {"insufficient_data": True, "n_bets": len(recs)}
                continue

            clvs = np.array([r.clv_cents for r in recs])

            # Rolling CLV to detect trends
            if len(clvs) >= 20:
                window = min(20, len(clvs) // 2)
                rolling_means = []
                for i in range(window, len(clvs) + 1):
                    rolling_means.append(float(np.mean(clvs[i - window:i])))

                # Trend: is edge increasing or decreasing?
                if len(rolling_means) >= 3:
                    x = np.arange(len(rolling_means))
                    slope, intercept, r_val, p_val, std_err = scipy_stats.linregress(x, rolling_means)
                    trend = "increasing" if slope > 0.05 else "decreasing" if slope < -0.05 else "stable"
                else:
                    slope, trend = 0.0, "unknown"
            else:
                slope, trend = 0.0, "unknown"

            # T-test for this market
            t_stat, p_val = scipy_stats.ttest_1samp(clvs, 0)
            p_value = p_val / 2 if t_stat > 0 else 1.0

            # Sharpe-like ratio for this market
            mean_clv = float(np.mean(clvs))
            std_clv = float(np.std(clvs))
            sharpe = mean_clv / std_clv if std_clv > 0 else 0.0

            # Sustainability score: combines significance, trend, and sample
            sig_score = max(0, 1.0 - p_value) * 0.4
            trend_score = (0.3 if trend == "stable" else 0.4 if trend == "increasing" else 0.1) * 0.3
            sample_score = min(len(recs) / 100.0, 1.0) * 0.3
            sustainability = sig_score + trend_score + sample_score

            segments[mtype] = {
                "n_bets": len(recs),
                "avg_clv": round(mean_clv, 2),
                "std_clv": round(std_clv, 2),
                "sharpe": round(sharpe, 3),
                "p_value": round(p_value, 4),
                "trend": trend,
                "trend_slope": round(slope, 4),
                "sustainability_score": round(sustainability, 3),
                "is_exploitable": mean_clv > 1.0 and p_value < 0.10,
                "recommendation": self._market_recommendation(mean_clv, p_value, trend),
            }

        return segments

    def _market_recommendation(self, clv: float, p_value: float, trend: str) -> str:
        """Generate recommendation for a market segment."""
        if clv > 3.0 and p_value < 0.05 and trend != "decreasing":
            return "INCREASE: Strong, significant, sustainable edge."
        if clv > 1.0 and p_value < 0.10:
            if trend == "decreasing":
                return "MONITOR: Edge exists but declining. Watch for decay."
            return "MAINTAIN: Moderate edge, continue current allocation."
        if clv > 0 and p_value < 0.20:
            return "REDUCE: Weak edge, not significant. Reduce exposure."
        if clv <= 0:
            return "STOP: No edge detected. Discontinue this market."
        return "EVALUATE: Insufficient confidence. Continue with small size."
