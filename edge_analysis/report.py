"""Full edge attribution report generator.

Produces a comprehensive report combining all 5 components with
human-readable summaries, visualizable data, and actionable verdicts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional

import numpy as np

from database.connection import DatabaseManager
from database.models import EdgeReport as EdgeReportModel, Bet, CLVLog
from edge_analysis.schemas import GolfBetRecord, EdgeReport
from edge_analysis.decomposer import GolfEdgeDecomposer

logger = logging.getLogger(__name__)


class EdgeReportGenerator:
    """Generate and persist edge attribution reports."""

    def __init__(self, sport: str = "golf"):
        self.sport = sport
        self.decomposer = GolfEdgeDecomposer()

    def generate_report(
        self,
        records: Optional[List[GolfBetRecord]] = None,
        lookback_days: int = 90,
    ) -> EdgeReport:
        """Generate a full edge report.

        If records not provided, loads from database.
        """
        if records is None:
            records = self._load_records(lookback_days)

        report = self.decomposer.full_decomposition(records)
        self._persist_report(report)
        return report

    def _load_records(self, lookback_days: int) -> List[GolfBetRecord]:
        """Load settled bet records from database and convert to GolfBetRecord."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)

        records = []
        with DatabaseManager.session_scope() as session:
            bets = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.status.in_(["won", "lost"]),
                    Bet.timestamp >= cutoff,
                )
                .order_by(Bet.timestamp.asc())
                .all()
            )

            for bet in bets:
                # Map bet_type to market_type
                market_type_map = {
                    "outright": "outright",
                    "h2h": "matchup",
                    "top5": "top5",
                    "top10": "top10",
                    "top20": "top20",
                    "make_cut": "make_cut",
                    "over": "make_cut",
                    "under": "make_cut",
                }
                market_type = market_type_map.get(bet.bet_type, "outright")

                closing_line = bet.closing_line if bet.closing_line else bet.line
                actual_outcome = 1.0 if bet.status == "won" else 0.0

                try:
                    features = json.loads(bet.features_snapshot) if bet.features_snapshot else {}
                except (json.JSONDecodeError, TypeError):
                    features = {}

                try:
                    record = GolfBetRecord(
                        bet_id=bet.bet_id,
                        tournament=bet.event_id or "unknown",
                        player=bet.player,
                        market_type=market_type,
                        signal_line=bet.line,
                        bet_line=bet.line,
                        closing_line=closing_line,
                        predicted_prob=bet.model_prob,
                        actual_outcome=actual_outcome,
                        weather_conditions=features.get("weather", "normal"),
                        wave=features.get("wave", "unknown"),
                        course_id=features.get("course_id", ""),
                        bet_timestamp=bet.timestamp,
                        odds_american=bet.odds_american,
                        odds_decimal=bet.odds_decimal,
                        closing_odds_american=bet.closing_odds,
                        data_sources_available=features.get("data_sources", []),
                        book=bet.book or "unknown",
                        pnl=bet.pnl,
                        stake=bet.stake,
                    )
                    records.append(record)
                except (ValueError, TypeError) as e:
                    logger.warning("Skipping bet %s: %s", bet.bet_id, e)

        logger.info("Loaded %d settled bet records for edge analysis", len(records))
        return records

    def _persist_report(self, report: EdgeReport) -> None:
        """Save report summary to database."""
        try:
            with DatabaseManager.session_scope() as session:
                db_report = EdgeReportModel(
                    sport=self.sport,
                    report_date=report.report_date,
                    clv_last_50=report.market.details.get("overall_clv_cents", 0.0),
                    clv_last_100=report.market.details.get("overall_clv_cents", 0.0),
                    clv_last_250=report.total_edge_cents,
                    clv_last_500=report.total_edge_cents,
                    clv_trend="improving" if report.total_edge_cents > 0 else "declining",
                    calibration_error=report.predictive.details.get("expected_calibration_error", 0.0),
                    calibration_buckets=json.dumps(report.predictive.details.get("calibration_curve", [])),
                    model_roi=report.total_roi,
                    expected_roi=report.total_roi,
                    edge_exists=report.edge_is_real,
                    system_state="active" if report.edge_is_real else "reduced",
                    warnings=json.dumps(report.warnings),
                    actions=json.dumps(report.recommendations),
                )
                session.add(db_report)
            logger.info("Edge report persisted for %s", report.report_date)
        except Exception:
            logger.exception("Failed to persist edge report")

    def format_text_report(self, report: EdgeReport) -> str:
        """Format report as human-readable text."""
        lines = [
            "=" * 70,
            f"GOLF EDGE ATTRIBUTION REPORT — {report.report_date.strftime('%Y-%m-%d %H:%M')}",
            "=" * 70,
            f"Bets analyzed: {report.n_bets}",
            f"Total P&L: ${report.total_pnl:,.2f} (ROI: {report.total_roi:.2%})",
            f"Total edge: {report.total_edge_cents:.1f} cents",
            f"Edge is real: {report.edge_is_real} (p={report.p_value:.4f})",
            "",
            "COMPONENT BREAKDOWN:",
            "-" * 40,
        ]

        for comp in report.components_list():
            lines.append(f"  {comp.name:15s}: {comp.value:+6.1f}c  (conf={comp.confidence:.2f})")
            lines.append(f"    {comp.verdict}")

        lines.extend([
            "",
            "MARKET TYPE BREAKDOWN:",
            "-" * 40,
        ])
        for mtype, data in report.by_market_type.items():
            lines.append(
                f"  {mtype:12s}: {data['n_bets']:3d} bets, "
                f"CLV={data['avg_clv']:+5.1f}c, "
                f"WR={data['win_rate']:.1%}, "
                f"P&L=${data['total_pnl']:+.2f}"
            )

        if report.warnings:
            lines.extend(["", "WARNINGS:", "-" * 40])
            for w in report.warnings:
                lines.append(f"  * {w}")

        if report.recommendations:
            lines.extend(["", "RECOMMENDATIONS:", "-" * 40])
            for r in report.recommendations:
                lines.append(f"  > {r}")

        lines.extend(["", "VERDICT:", "-" * 40, f"  {report.verdict}", "=" * 70])

        return "\n".join(lines)
