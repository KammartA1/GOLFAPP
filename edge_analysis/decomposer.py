"""Golf Edge Decomposer — 5-component edge decomposition.

Breaks total betting edge into:
  1. Predictive — Model accuracy (Brier, log loss, calibration)
  2. Informational — Timing advantage per data source
  3. Market — CLV per market type, market inefficiency
  4. Execution — Price quality, line shopping
  5. Structural — Field correlation, wave advantage, course type

Usage:
    decomposer = GolfEdgeDecomposer()
    report = decomposer.full_decomposition(bet_records)
    print(report.verdict)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List

import numpy as np

from edge_analysis.schemas import GolfBetRecord, EdgeReport
from edge_analysis.predictive import PredictiveAnalyzer
from edge_analysis.informational import InformationalAnalyzer
from edge_analysis.market_inefficiency import MarketInefficiencyAnalyzer
from edge_analysis.execution import ExecutionAnalyzer
from edge_analysis.structural import StructuralAnalyzer

logger = logging.getLogger(__name__)


class GolfEdgeDecomposer:
    """Decomposes golf betting edge into 5 measurable components."""

    def __init__(self):
        self.predictive = PredictiveAnalyzer()
        self.informational = InformationalAnalyzer()
        self.market = MarketInefficiencyAnalyzer()
        self.execution = ExecutionAnalyzer()
        self.structural = StructuralAnalyzer()

    def full_decomposition(self, records: List[GolfBetRecord]) -> EdgeReport:
        """Run full 5-component edge decomposition.

        Args:
            records: List of GolfBetRecord with full context.

        Returns:
            EdgeReport with all components, verdict, and recommendations.
        """
        if not records:
            return EdgeReport(
                report_date=datetime.utcnow(),
                n_bets=0,
                total_pnl=0.0,
                total_roi=0.0,
                verdict="No bet records to analyze.",
            )

        logger.info("Running full edge decomposition on %d bets", len(records))

        # Run each analyzer
        pred_component = self.predictive.analyze(records)
        info_component = self.informational.analyze(records)
        market_component = self.market.analyze(records)
        exec_component = self.execution.analyze(records)
        struct_component = self.structural.analyze(records)

        # Aggregate metrics
        total_pnl = sum(r.pnl for r in records)
        total_staked = sum(r.stake for r in records)
        total_roi = total_pnl / total_staked if total_staked > 0 else 0.0

        total_edge_cents = (
            pred_component.value +
            info_component.value +
            market_component.value +
            exec_component.value +
            struct_component.value
        )

        # Statistical significance of overall edge
        clvs = np.array([r.clv_cents for r in records])
        from scipy import stats as scipy_stats
        if len(clvs) >= 20:
            t_stat, p_val = scipy_stats.ttest_1samp(clvs, 0)
            p_value = p_val / 2 if t_stat > 0 else 1.0
        else:
            p_value = 1.0

        edge_is_real = total_edge_cents > 0 and p_value < 0.05

        # Per market type breakdown
        from collections import defaultdict
        by_market = defaultdict(list)
        for r in records:
            by_market[r.market_type].append(r)
        market_summary = {}
        for mtype, recs in by_market.items():
            market_summary[mtype] = {
                "n_bets": len(recs),
                "avg_clv": round(float(np.mean([r.clv_cents for r in recs])), 2),
                "win_rate": round(float(np.mean([r.actual_outcome for r in recs])), 4),
                "total_pnl": round(sum(r.pnl for r in recs), 2),
            }

        # Warnings and recommendations
        warnings = self._generate_warnings(
            pred_component, info_component, market_component,
            exec_component, struct_component, records,
        )
        recommendations = self._generate_recommendations(
            pred_component, info_component, market_component,
            exec_component, struct_component,
        )

        # Overall verdict
        verdict = self._overall_verdict(
            total_edge_cents, edge_is_real, p_value, records,
            pred_component, market_component,
        )

        report = EdgeReport(
            report_date=datetime.utcnow(),
            n_bets=len(records),
            total_pnl=round(total_pnl, 2),
            total_roi=round(total_roi, 4),
            predictive=pred_component,
            informational=info_component,
            market=market_component,
            execution=exec_component,
            structural=struct_component,
            total_edge_cents=round(total_edge_cents, 2),
            edge_is_real=edge_is_real,
            p_value=round(p_value, 4),
            verdict=verdict,
            warnings=warnings,
            recommendations=recommendations,
            by_market_type=market_summary,
        )

        logger.info("Edge decomposition complete: %.1fc total, real=%s, p=%.4f",
                     total_edge_cents, edge_is_real, p_value)
        return report

    def _generate_warnings(self, pred, info, market, execution, structural, records) -> list:
        """Generate warnings from component analysis."""
        warnings = []

        # Predictive warnings
        if pred.details.get("brier_skill_score", 0) < 0:
            warnings.append("Model is performing worse than naive base rate predictor")
        if pred.details.get("expected_calibration_error", 0) > 0.05:
            warnings.append(f"Model is poorly calibrated (ECE={pred.details['expected_calibration_error']:.3f})")

        # Market warnings
        if market.details.get("beat_close_rate", 0) < 0.48:
            warnings.append(f"Beat-close rate below 48% ({market.details.get('beat_close_rate', 0):.1%})")

        # Execution warnings
        exec_cost = execution.details.get("execution_cost_cents", 0)
        if exec_cost > 2.0:
            warnings.append(f"High execution cost ({exec_cost:.1f}c per bet)")

        # Structural warnings
        corr = structural.details.get("field_correlation", {})
        if corr.get("avg_bets_per_tournament", 0) > 8:
            warnings.append("Over-concentrated: too many bets per tournament")

        # Sample size warning
        if len(records) < 50:
            warnings.append(f"Small sample ({len(records)} bets) — results may not be reliable")

        return warnings

    def _generate_recommendations(self, pred, info, market, execution, structural) -> list:
        """Generate actionable recommendations."""
        recs = []

        # Lean into strongest component
        components = [pred, info, market, execution, structural]
        best = max(components, key=lambda c: c.value)
        if best.value > 1.0:
            recs.append(f"Primary edge source: {best.name} ({best.value:.1f}c). Focus here.")

        # Fix weakest component
        worst = min(components, key=lambda c: c.value)
        if worst.value < -0.5:
            recs.append(f"Edge leak: {worst.name} ({worst.value:.1f}c). Investigate and fix.")

        # Market-specific recommendations
        market_data = market.details.get("by_market_type", {})
        for mtype, data in market_data.items():
            if data.get("is_significant") and data.get("avg_clv", 0) > 3.0:
                recs.append(f"Increase allocation to {mtype} markets (CLV={data['avg_clv']:.1f}c, significant)")
            if data.get("avg_clv", 0) < -1.0 and data.get("n_bets", 0) > 20:
                recs.append(f"Reduce or stop {mtype} bets (CLV={data['avg_clv']:.1f}c)")

        # Informational recommendations
        best_source_key = info.details.get("best_source", "none")
        if best_source_key != "none":
            source_data = info.details.get("by_source", {}).get(best_source_key, {})
            if source_data.get("edge_contribution", 0) > 0.02:
                recs.append(f"Best data source: {best_source_key} — ensure always available before betting")

        return recs

    def _overall_verdict(self, total_edge, is_real, p_value, records, pred, market) -> str:
        """Generate the overall verdict string."""
        n = len(records)

        if n < 20:
            return (f"INSUFFICIENT DATA: Only {n} bets analyzed. Need 50+ for reliable decomposition. "
                    f"Current total edge: {total_edge:.1f}c (unreliable).")

        if is_real:
            dominant = "predictive" if pred.value > market.value else "market"
            return (f"EDGE EXISTS: {total_edge:.1f}c total edge (p={p_value:.4f}). "
                    f"Dominant source: {dominant}. Continue betting with current approach.")

        if total_edge > 0 and p_value < 0.10:
            return (f"EDGE PROBABLE: {total_edge:.1f}c total edge (p={p_value:.4f}). "
                    f"Not yet significant at 5% level. Continue with reduced sizing.")

        if total_edge > 0:
            return (f"EDGE UNCLEAR: {total_edge:.1f}c positive but not significant (p={p_value:.4f}). "
                    f"May be variance. Maintain position but monitor closely.")

        return (f"NO EDGE: {total_edge:.1f}c (p={p_value:.4f}). "
                f"Model is not beating the market. Suspend betting and investigate.")
