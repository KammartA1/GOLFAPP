"""
edge_analysis/report.py
========================
Generates the full golf edge attribution report.

Final verdict: "Which component is doing the heavy lifting — and which are illusions?"
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from edge_analysis.decomposer import EdgeDecomposer
from edge_analysis.schemas import EdgeReport

log = logging.getLogger(__name__)


class EdgeReportGenerator:
    """Generates and persists golf edge attribution reports."""

    def __init__(self, sport: str = "golf", db_path: str | None = None):
        self.sport = sport.lower()
        self._db_path = db_path
        self._decomposer = EdgeDecomposer(sport=sport, db_path=db_path)

    def generate(self) -> EdgeReport:
        return self._decomposer.run()

    def generate_and_store(self) -> EdgeReport:
        report = self.generate()
        self._store_report(report)
        return report

    def _store_report(self, report: EdgeReport) -> None:
        try:
            from database.models import EdgeReport as EdgeReportModel, Base
            from database.connection import get_session
            session = get_session()
            try:
                entry = EdgeReportModel(
                    report_type="edge_decomposition",
                    sport="GOLF",
                    report_json=self.to_json(report),
                )
                session.add(entry)
                session.commit()
                log.info("Golf edge decomposition report stored")
            except Exception as exc:
                session.rollback()
                log.warning("Could not store report: %s", exc)
            finally:
                session.close()
        except Exception as exc:
            log.warning("DB unavailable: %s", exc)

    def to_text(self, report: Optional[EdgeReport] = None) -> str:
        if report is None:
            report = self.generate()
        return report.verdict

    def to_dict(self, report: Optional[EdgeReport] = None) -> Dict[str, Any]:
        if report is None:
            report = self.generate()

        result = {
            "generated_at": report.generated_at.isoformat(),
            "sport": report.sport,
            "total_roi": report.total_roi,
            "total_bets": report.total_bets,
            "total_pnl": report.total_pnl,
            "attribution": {
                "predictive_pct": report.predictive_pct,
                "informational_pct": report.informational_pct,
                "market_pct": report.market_pct,
                "execution_pct": report.execution_pct,
                "structural_pct": report.structural_pct,
            },
            "scoring": {
                "brier_score": report.brier_score,
                "brier_baseline": report.brier_baseline,
                "log_loss": report.log_loss,
                "log_loss_baseline": report.log_loss_baseline,
            },
            "heavy_lifter": report.heavy_lifter,
            "illusions": report.illusions,
            "verdict": report.verdict,
        }

        for comp_name in ["predictive", "informational", "market_inefficiency", "execution", "structural"]:
            comp = getattr(report, comp_name, None)
            if comp:
                result[comp_name] = {
                    "edge_pct": comp.edge_pct_of_roi,
                    "absolute_value": comp.absolute_value,
                    "p_value": comp.p_value,
                    "is_significant": comp.is_significant,
                    "is_positive": comp.is_positive,
                    "sample_size": comp.sample_size,
                    "details": comp.details,
                    "verdict": comp.verdict,
                }

        if report.calibration_curve:
            result["calibration_curve"] = [
                {
                    "bucket_lower": p.bucket_lower,
                    "bucket_upper": p.bucket_upper,
                    "predicted": p.predicted_avg,
                    "actual": p.actual_rate,
                    "n": p.n_bets,
                    "error": p.calibration_error,
                }
                for p in report.calibration_curve
            ]

        return result

    def to_json(self, report: Optional[EdgeReport] = None) -> str:
        return json.dumps(self.to_dict(report), indent=2, default=str)
