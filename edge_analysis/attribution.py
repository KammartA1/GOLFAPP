"""
edge_analysis/attribution.py
=============================
Edge Attribution Engine — answers ONE question:
"Where does profit ACTUALLY come from?"

Decomposes every dollar of profit into four buckets:
  1. PREDICTION EDGE   — profit from being right about probabilities
  2. CLV/TIMING EDGE   — profit from betting before the line moves
  3. MARKET INEFFICIENCY — profit from structural mispricing
  4. VARIANCE           — profit from pure luck (the enemy)

Golf-specific: segments by tournament, market type (outright/top5/H2H/cut),
wave advantage, course, and day of week.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AttributionResult:
    """Dollar and percentage attribution for one edge source."""
    label: str
    dollar_value: float
    pct_of_total: float
    description: str = ""


@dataclass
class CounterfactualResult:
    """Outcome of re-running bets at closing line prices."""
    actual_profit: float
    closing_line_profit: float
    timing_edge_dollars: float
    prediction_edge_dollars: float
    timing_edge_pct: float
    prediction_edge_pct: float
    n_bets_better_at_close: int
    n_bets_only_timing: int


@dataclass
class BootstrapResult:
    """Output from Monte Carlo bootstrap significance test."""
    n_simulations: int
    mean_profit: float
    std_profit: float
    ci_lower_95: float
    ci_upper_95: float
    ci_lower_99: float
    ci_upper_99: float
    p_profit_positive: float
    is_significant_95: bool
    is_significant_99: bool
    profit_distribution: List[float]
    required_sample_for_significance: int


@dataclass
class SegmentEdge:
    """Profit concentration in a specific segment."""
    segment_name: str
    segment_value: str
    n_bets: int
    total_pnl: float
    roi_pct: float
    avg_edge: float
    pct_of_total_profit: float


@dataclass
class MarketInefficiencyResult:
    """Structural mispricing analysis across segments."""
    total_inefficiency_dollars: float
    total_inefficiency_pct: float
    segments: List[SegmentEdge]
    concentrated: bool
    dominant_segment: Optional[str]


@dataclass
class AttributionReport:
    """Complete edge attribution report."""
    generated_at: datetime
    total_profit: float
    total_bets: int
    total_staked: float
    roi_pct: float

    prediction_edge: AttributionResult
    clv_timing_edge: AttributionResult
    market_inefficiency: AttributionResult
    variance_component: AttributionResult

    counterfactual: CounterfactualResult
    bootstrap: BootstrapResult
    inefficiency: MarketInefficiencyResult

    clv_pct_of_profit: float
    is_clv_bot: bool

    verdict: str
    confidence_level: str


# ---------------------------------------------------------------------------
# Bet record adapter — works with GolfBetRecord, BetRecord, or dicts
# ---------------------------------------------------------------------------

def _extract_bet_fields(bet) -> dict:
    """Extract fields needed from any bet record type."""
    if isinstance(bet, dict):
        return {
            "bet_id": bet.get("bet_id", ""),
            "timestamp": bet.get("timestamp", datetime.now()),
            "player": bet.get("player", ""),
            "tournament": bet.get("tournament", "unknown"),
            "market_type": bet.get("market_type", bet.get("stat_type", "unknown")),
            "direction": bet.get("direction", "over"),
            "wave": bet.get("wave"),
            "bet_line": bet.get("bet_line", bet.get("line", 0.0)),
            "closing_line": bet.get("closing_line", None),
            "predicted_prob": bet.get("predicted_prob", bet.get("model_prob", 0.5)),
            "market_prob_at_bet": bet.get("market_prob_at_bet", bet.get("market_prob", 0.5)),
            "won": bet.get("won", None),
            "stake": bet.get("stake", 0.0),
            "pnl": bet.get("pnl", 0.0),
            "odds_decimal": bet.get("odds_decimal", 1.909),
            "odds_american": bet.get("odds_american", -110),
            "model_projection": bet.get("model_projection", 0.0),
            "confidence_score": bet.get("confidence_score", 0.0),
            "book": bet.get("book", bet.get("sportsbook", "unknown")),
            "course_id": bet.get("course_id"),
        }

    fields = {}
    fields["bet_id"] = getattr(bet, "bet_id", "")
    fields["timestamp"] = getattr(bet, "timestamp", datetime.now())
    fields["player"] = getattr(bet, "player", "")
    fields["tournament"] = getattr(bet, "tournament", "unknown")
    fields["market_type"] = getattr(bet, "market_type", getattr(bet, "stat_type", "unknown"))
    fields["direction"] = getattr(bet, "direction", "over")
    fields["wave"] = getattr(bet, "wave", None)
    fields["bet_line"] = getattr(bet, "bet_line", getattr(bet, "line", 0.0))
    fields["closing_line"] = getattr(bet, "closing_line", None)
    fields["predicted_prob"] = getattr(bet, "predicted_prob", getattr(bet, "model_prob", 0.5))
    fields["market_prob_at_bet"] = getattr(bet, "market_prob_at_bet", getattr(bet, "market_prob", 0.5))

    won = getattr(bet, "won", None)
    if won is None:
        status = getattr(bet, "status", None)
        if status is not None:
            status_val = status.value if hasattr(status, "value") else str(status)
            won = status_val == "won"
    fields["won"] = won

    fields["stake"] = getattr(bet, "stake", 0.0)
    fields["pnl"] = getattr(bet, "pnl", 0.0)
    fields["odds_decimal"] = getattr(bet, "odds_decimal", 1.909)
    fields["odds_american"] = getattr(bet, "odds_american", -110)
    fields["model_projection"] = getattr(bet, "model_projection", 0.0)
    fields["confidence_score"] = getattr(bet, "confidence_score", 0.0)
    fields["book"] = getattr(bet, "book", getattr(bet, "sportsbook", "unknown"))
    fields["course_id"] = getattr(bet, "course_id", None)
    return fields


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class EdgeAttributionEngine:
    """
    Decomposes golf betting profit into prediction edge, CLV/timing edge,
    market inefficiency, and variance.
    """

    MIN_BETS_FOR_ANALYSIS = 30
    MIN_BETS_FOR_SIGNIFICANCE = 100

    def __init__(self, bet_history: list):
        self.raw_bets = bet_history
        self.bets = [_extract_bet_fields(b) for b in bet_history]
        self.settled = [b for b in self.bets if b["won"] is not None]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def decompose(self) -> AttributionReport:
        """Full profit decomposition into the four edge buckets."""
        now = datetime.now()
        n = len(self.settled)
        total_pnl = sum(b["pnl"] for b in self.settled)
        total_staked = sum(b["stake"] for b in self.settled)
        roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0.0

        if n < self.MIN_BETS_FOR_ANALYSIS:
            return self._insufficient_data_report(now, n, total_pnl, total_staked, roi)

        cf = self.counterfactual_analysis()
        bs = self.monte_carlo_significance()
        mi = self.market_inefficiency_analysis()

        prediction_dollars = cf.prediction_edge_dollars
        timing_dollars = cf.timing_edge_dollars
        inefficiency_dollars = mi.total_inefficiency_dollars

        # Avoid double-counting
        if abs(prediction_dollars) > 0:
            inefficiency_from_prediction = min(
                abs(inefficiency_dollars), abs(prediction_dollars)
            ) * (1 if inefficiency_dollars >= 0 else -1)
            prediction_dollars -= inefficiency_from_prediction
            inefficiency_dollars = inefficiency_from_prediction
        else:
            inefficiency_dollars = min(abs(inefficiency_dollars), abs(total_pnl)) * (
                1 if inefficiency_dollars >= 0 else -1
            )

        attributed = prediction_dollars + timing_dollars + inefficiency_dollars
        variance_dollars = total_pnl - attributed

        if not bs.is_significant_95 and total_pnl > 0:
            variance_fraction = 1.0 - bs.p_profit_positive
            variance_dollars = total_pnl * max(variance_fraction, 0.3)
            remaining = total_pnl - variance_dollars
            if abs(attributed) > 0:
                scale = remaining / attributed
                prediction_dollars *= scale
                timing_dollars *= scale
                inefficiency_dollars *= scale

        abs_total = abs(total_pnl) if abs(total_pnl) > 0 else 1.0

        prediction_result = AttributionResult(
            label="Prediction Edge",
            dollar_value=round(prediction_dollars, 2),
            pct_of_total=round(prediction_dollars / abs_total * 100, 1),
            description="Profit from being right about probabilities",
        )
        timing_result = AttributionResult(
            label="CLV/Timing Edge",
            dollar_value=round(timing_dollars, 2),
            pct_of_total=round(timing_dollars / abs_total * 100, 1),
            description="Profit from betting before the line moves",
        )
        inefficiency_result = AttributionResult(
            label="Market Inefficiency",
            dollar_value=round(inefficiency_dollars, 2),
            pct_of_total=round(inefficiency_dollars / abs_total * 100, 1),
            description="Profit from structural mispricing in specific segments",
        )
        variance_result = AttributionResult(
            label="Variance",
            dollar_value=round(variance_dollars, 2),
            pct_of_total=round(variance_dollars / abs_total * 100, 1),
            description="Profit attributable to luck (positive or negative)",
        )

        # CLV diagnostic
        clv_dollars = sum(
            (b["bet_line"] - b["closing_line"]) * b["stake"]
            for b in self.settled
            if b["closing_line"] is not None and b["closing_line"] != 0
        )
        clv_pct = (clv_dollars / abs_total * 100) if abs_total > 0 else 0.0
        is_clv_bot = abs(clv_pct) > 80

        verdict, confidence = self._compute_verdict(bs, cf, n)

        return AttributionReport(
            generated_at=now,
            total_profit=round(total_pnl, 2),
            total_bets=n,
            total_staked=round(total_staked, 2),
            roi_pct=round(roi, 2),
            prediction_edge=prediction_result,
            clv_timing_edge=timing_result,
            market_inefficiency=inefficiency_result,
            variance_component=variance_result,
            counterfactual=cf,
            bootstrap=bs,
            inefficiency=mi,
            clv_pct_of_profit=round(clv_pct, 1),
            is_clv_bot=is_clv_bot,
            verdict=verdict,
            confidence_level=confidence,
        )

    # ------------------------------------------------------------------
    # Counterfactual analysis
    # ------------------------------------------------------------------

    def counterfactual_analysis(self) -> CounterfactualResult:
        """
        Re-run all bets at closing line prices.
        Difference = timing/CLV edge. Remainder = prediction edge.
        """
        actual_profit = 0.0
        closing_profit = 0.0
        n_better_at_close = 0
        n_only_timing = 0

        for bet in self.settled:
            actual_profit += bet["pnl"]
            cl = bet["closing_line"]

            if cl is None or cl == 0:
                closing_profit += bet["pnl"]
                continue

            bet_line = bet["bet_line"]
            stake = bet["stake"]
            won = bet["won"]
            odds_dec = bet["odds_decimal"]
            direction = bet["direction"].lower()
            market = bet["market_type"].lower()

            if won:
                cf_pnl = stake * (odds_dec - 1)

                # For outright/top-N markets: line is odds, not a stat line
                # Higher closing odds = worse for us (we got value early)
                if market in ("outright", "top5", "top10", "top20", "make_cut", "h2h"):
                    # Odds-based: closing_line > bet_line means odds drifted out
                    # If closing odds > bet odds, we got better value → timing edge
                    line_advantage = cl - bet_line
                    if line_advantage <= 0:
                        closing_profit += cf_pnl
                        n_better_at_close += 1
                    else:
                        movement_ratio = abs(line_advantage) / max(abs(bet_line), 1.0)
                        if movement_ratio < 0.05:
                            closing_profit += cf_pnl
                            n_better_at_close += 1
                        else:
                            timing_fraction = min(movement_ratio * 5, 1.0)
                            closing_profit += cf_pnl * (1 - timing_fraction)
                            n_only_timing += 1
                else:
                    # Stat-line based (over/under)
                    if direction == "over":
                        line_advantage = cl - bet_line
                    elif direction == "under":
                        line_advantage = bet_line - cl
                    else:
                        line_advantage = 0

                    if line_advantage <= 0:
                        closing_profit += cf_pnl
                        n_better_at_close += 1
                    else:
                        movement_ratio = line_advantage / max(abs(bet_line), 0.5)
                        if movement_ratio < 0.05:
                            closing_profit += cf_pnl
                            n_better_at_close += 1
                        else:
                            timing_fraction = min(movement_ratio * 5, 1.0)
                            closing_profit += cf_pnl * (1 - timing_fraction)
                            n_only_timing += 1
            else:
                closing_profit += -stake

        timing_edge = actual_profit - closing_profit
        prediction_edge = closing_profit
        total = abs(actual_profit) if abs(actual_profit) > 0 else 1.0

        return CounterfactualResult(
            actual_profit=round(actual_profit, 2),
            closing_line_profit=round(closing_profit, 2),
            timing_edge_dollars=round(timing_edge, 2),
            prediction_edge_dollars=round(prediction_edge, 2),
            timing_edge_pct=round(timing_edge / total * 100, 1),
            prediction_edge_pct=round(prediction_edge / total * 100, 1),
            n_bets_better_at_close=n_better_at_close,
            n_bets_only_timing=n_only_timing,
        )

    # ------------------------------------------------------------------
    # Monte Carlo bootstrap significance
    # ------------------------------------------------------------------

    def monte_carlo_significance(self, n_sims: int = 10000) -> BootstrapResult:
        """
        Bootstrap resample bet outcomes.
        P(profit > 0) < 0.95 → profit is likely variance.
        """
        if not self.settled:
            return BootstrapResult(
                n_simulations=0, mean_profit=0, std_profit=0,
                ci_lower_95=0, ci_upper_95=0, ci_lower_99=0, ci_upper_99=0,
                p_profit_positive=0, is_significant_95=False, is_significant_99=False,
                profit_distribution=[], required_sample_for_significance=999,
            )

        pnl_array = np.array([b["pnl"] for b in self.settled])
        n = len(pnl_array)
        rng = np.random.default_rng(seed=42)

        sim_profits = np.zeros(n_sims)
        for i in range(n_sims):
            indices = rng.integers(0, n, size=n)
            sim_profits[i] = pnl_array[indices].sum()

        mean_profit = float(np.mean(sim_profits))
        std_profit = float(np.std(sim_profits))
        ci_lower_95 = float(np.percentile(sim_profits, 2.5))
        ci_upper_95 = float(np.percentile(sim_profits, 97.5))
        ci_lower_99 = float(np.percentile(sim_profits, 0.5))
        ci_upper_99 = float(np.percentile(sim_profits, 99.5))
        p_positive = float(np.mean(sim_profits > 0))

        mean_per_bet = float(np.mean(pnl_array))
        std_per_bet = float(np.std(pnl_array))
        if abs(mean_per_bet) > 0 and std_per_bet > 0:
            required_n = int(math.ceil((1.96 * std_per_bet / abs(mean_per_bet)) ** 2))
            required_n = max(required_n, 30)
        else:
            required_n = 999

        return BootstrapResult(
            n_simulations=n_sims,
            mean_profit=round(mean_profit, 2),
            std_profit=round(std_profit, 2),
            ci_lower_95=round(ci_lower_95, 2),
            ci_upper_95=round(ci_upper_95, 2),
            ci_lower_99=round(ci_lower_99, 2),
            ci_upper_99=round(ci_upper_99, 2),
            p_profit_positive=round(p_positive, 4),
            is_significant_95=ci_lower_95 > 0,
            is_significant_99=ci_lower_99 > 0,
            profit_distribution=sim_profits.tolist(),
            required_sample_for_significance=required_n,
        )

    # ------------------------------------------------------------------
    # Market inefficiency segmentation (golf-specific)
    # ------------------------------------------------------------------

    def market_inefficiency_analysis(self) -> MarketInefficiencyResult:
        """
        Segment bets by market type, tournament, wave, day of week, book.
        Find if profit concentrates in specific segments.
        """
        total_pnl = sum(b["pnl"] for b in self.settled)
        abs_total = abs(total_pnl) if abs(total_pnl) > 0 else 1.0

        segments: List[SegmentEdge] = []

        # --- Segment by market type (outright, top5, H2H, cut, etc.) ---
        by_market = self._group_by("market_type")
        for market, bets in by_market.items():
            seg = self._compute_segment("Market", market, bets, abs_total)
            if seg:
                segments.append(seg)

        # --- Segment by tournament ---
        by_tournament = self._group_by("tournament")
        for tournament, bets in by_tournament.items():
            if tournament and tournament != "unknown":
                seg = self._compute_segment("Tournament", tournament, bets, abs_total)
                if seg:
                    segments.append(seg)

        # --- Segment by wave (AM/PM tee times) ---
        wave_bets = [b for b in self.settled if b.get("wave")]
        if wave_bets:
            by_wave = defaultdict(list)
            for b in wave_bets:
                by_wave[b["wave"]].append(b)
            for wave, bets in by_wave.items():
                seg = self._compute_segment("Wave", wave, bets, abs_total)
                if seg:
                    segments.append(seg)

        # --- Segment by direction ---
        by_direction = self._group_by("direction")
        for direction, bets in by_direction.items():
            seg = self._compute_segment("Direction", direction, bets, abs_total)
            if seg:
                segments.append(seg)

        # --- Segment by day of week ---
        day_buckets = defaultdict(list)
        for b in self.settled:
            ts = b["timestamp"]
            if isinstance(ts, datetime):
                day_buckets[ts.strftime("%A")].append(b)
        for day, bets in day_buckets.items():
            seg = self._compute_segment("Day of Week", day, bets, abs_total)
            if seg:
                segments.append(seg)

        # --- Segment by book ---
        by_book = self._group_by("book")
        for book, bets in by_book.items():
            if book and book != "unknown":
                seg = self._compute_segment("Book", book, bets, abs_total)
                if seg:
                    segments.append(seg)

        # Find concentration
        profitable_segments = sorted(
            [s for s in segments if s.total_pnl > 0],
            key=lambda s: s.total_pnl, reverse=True,
        )

        concentrated = False
        dominant = None
        if profitable_segments:
            top_seg = profitable_segments[0]
            if top_seg.pct_of_total_profit > 50:
                concentrated = True
                dominant = f"{top_seg.segment_name}: {top_seg.segment_value}"

        inefficiency_dollars = 0.0
        if concentrated and profitable_segments:
            total_stake = sum(b["stake"] for b in self.settled)
            avg_roi = (total_pnl / total_stake * 100) if total_stake > 0 else 0
            for seg in profitable_segments:
                if seg.roi_pct > avg_roi * 1.5:
                    excess = seg.total_pnl * (1 - avg_roi / seg.roi_pct) if seg.roi_pct > 0 else 0
                    inefficiency_dollars += excess

        return MarketInefficiencyResult(
            total_inefficiency_dollars=round(inefficiency_dollars, 2),
            total_inefficiency_pct=round(inefficiency_dollars / abs_total * 100, 1),
            segments=segments,
            concentrated=concentrated,
            dominant_segment=dominant,
        )

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        """Generate a human-readable edge attribution report."""
        report = self.decompose()
        lines = [
            "EDGE ATTRIBUTION REPORT (GOLF)",
            "=" * 50,
            f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')}",
            f"Total Bets: {report.total_bets}",
            f"Total Staked: ${report.total_staked:,.2f}",
            f"ROI: {report.roi_pct:+.2f}%",
            "",
            f"Total Profit: ${report.total_profit:,.2f}",
            f"  - From Prediction:         ${report.prediction_edge.dollar_value:>10,.2f}  ({report.prediction_edge.pct_of_total:>6.1f}%)",
            f"  - From CLV/Timing:          ${report.clv_timing_edge.dollar_value:>10,.2f}  ({report.clv_timing_edge.pct_of_total:>6.1f}%)",
            f"  - From Market Inefficiency:  ${report.market_inefficiency.dollar_value:>10,.2f}  ({report.market_inefficiency.pct_of_total:>6.1f}%)",
            f"  - Likely Variance:           ${report.variance_component.dollar_value:>10,.2f}  ({report.variance_component.pct_of_total:>6.1f}%)",
            "",
            "STATISTICAL SIGNIFICANCE:",
            f"  - P(profit > 0 | bootstrap): {report.bootstrap.p_profit_positive:.1%}",
            f"  - 95% CI: [${report.bootstrap.ci_lower_95:,.2f}, ${report.bootstrap.ci_upper_95:,.2f}]",
            f"  - Required sample for significance: {report.bootstrap.required_sample_for_significance} bets",
            "",
            "CLV DIAGNOSTIC:",
            f"  - CLV accounts for {report.clv_pct_of_profit:.1f}% of profit",
            f"  - System type: {'CLV BOT (timing-based)' if report.is_clv_bot else 'PREDICTION ENGINE'}",
            "",
        ]

        if report.inefficiency.concentrated:
            lines.append(f"CONCENTRATION WARNING: {report.inefficiency.dominant_segment}")
            lines.append("")

        lines.append(f"VERDICT: {report.verdict}")
        lines.append(f"CONFIDENCE: {report.confidence_level}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _group_by(self, key: str) -> Dict[str, List[dict]]:
        groups = defaultdict(list)
        for b in self.settled:
            groups[b.get(key, "unknown")].append(b)
        return dict(groups)

    def _compute_segment(
        self, seg_name: str, seg_value: str,
        bets: List[dict], abs_total_pnl: float,
    ) -> Optional[SegmentEdge]:
        if len(bets) < 3:
            return None
        total_pnl = sum(b["pnl"] for b in bets)
        total_stake = sum(b["stake"] for b in bets)
        roi = (total_pnl / total_stake * 100) if total_stake > 0 else 0
        avg_edge = float(np.mean([
            b["predicted_prob"] - b["market_prob_at_bet"] for b in bets
        ]))
        pct_of_total = (total_pnl / abs_total_pnl * 100) if abs_total_pnl > 0 else 0
        return SegmentEdge(
            segment_name=seg_name,
            segment_value=str(seg_value),
            n_bets=len(bets),
            total_pnl=round(total_pnl, 2),
            roi_pct=round(roi, 2),
            avg_edge=round(avg_edge, 4),
            pct_of_total_profit=round(pct_of_total, 1),
        )

    def _compute_verdict(
        self, bs: BootstrapResult, cf: CounterfactualResult, n: int,
    ) -> Tuple[str, str]:
        if n < self.MIN_BETS_FOR_SIGNIFICANCE:
            return "INSUFFICIENT DATA", "LOW"

        if bs.is_significant_99:
            if cf.prediction_edge_pct > 30:
                return "REAL EDGE", "HIGH"
            elif cf.timing_edge_pct > 60:
                return "REAL EDGE (CLV-DRIVEN)", "HIGH"
            else:
                return "REAL EDGE", "MEDIUM"
        elif bs.is_significant_95:
            if cf.prediction_edge_pct > 20:
                return "PROBABLE EDGE", "MEDIUM"
            else:
                return "PROBABLE EDGE (NEEDS MORE DATA)", "MEDIUM"
        elif bs.p_profit_positive > 0.80:
            return "POSSIBLE EDGE (NEEDS MORE DATA)", "LOW"
        else:
            return "LIKELY VARIANCE", "LOW"

    def _insufficient_data_report(
        self, now: datetime, n: int,
        total_pnl: float, total_staked: float, roi: float,
    ) -> AttributionReport:
        empty_attr = AttributionResult(
            label="N/A", dollar_value=0, pct_of_total=0,
            description="Insufficient data",
        )
        empty_cf = CounterfactualResult(
            actual_profit=total_pnl, closing_line_profit=0,
            timing_edge_dollars=0, prediction_edge_dollars=0,
            timing_edge_pct=0, prediction_edge_pct=0,
            n_bets_better_at_close=0, n_bets_only_timing=0,
        )
        empty_bs = BootstrapResult(
            n_simulations=0, mean_profit=0, std_profit=0,
            ci_lower_95=0, ci_upper_95=0, ci_lower_99=0, ci_upper_99=0,
            p_profit_positive=0, is_significant_95=False, is_significant_99=False,
            profit_distribution=[], required_sample_for_significance=self.MIN_BETS_FOR_SIGNIFICANCE,
        )
        empty_mi = MarketInefficiencyResult(
            total_inefficiency_dollars=0, total_inefficiency_pct=0,
            segments=[], concentrated=False, dominant_segment=None,
        )
        return AttributionReport(
            generated_at=now, total_profit=round(total_pnl, 2),
            total_bets=n, total_staked=round(total_staked, 2),
            roi_pct=round(roi, 2),
            prediction_edge=empty_attr, clv_timing_edge=empty_attr,
            market_inefficiency=empty_attr, variance_component=empty_attr,
            counterfactual=empty_cf, bootstrap=empty_bs,
            inefficiency=empty_mi,
            clv_pct_of_profit=0, is_clv_bot=False,
            verdict="INSUFFICIENT DATA",
            confidence_level="LOW",
        )
