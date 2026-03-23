"""Report Worker — Daily edge reports and system health monitoring.

Schedule: Daily at 6 AM ET (or after last event completes)

1. Run edge validation (CLV, calibration, drawdown, win rate)
2. Generate EdgeReport and persist to DB
3. Update system state
4. Log critical alerts

Writes to: edge_reports, system_state_log, audit_logs
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from database.connection import DatabaseManager
from database.models import (
    EdgeReport as EdgeReportModel,
    Bet, CLVLog, SystemStateLog, ModelVersion,
)
from .base_worker import BaseWorker

logger = logging.getLogger(__name__)


class ReportWorker(BaseWorker):
    WORKER_NAME = "report_worker_golf"
    SPORT = "golf"
    DEFAULT_INTERVAL_SECONDS = 86400  # 24 hours

    def execute(self) -> dict:
        """Generate daily edge report and update system state."""
        results = {
            "edge_exists": None,
            "system_state": None,
            "warnings": [],
            "metrics": {},
        }

        try:
            # Try using the full QuantEngine edge validator
            report = self._run_edge_validation()
            if report:
                results.update(report)
            else:
                # Fallback: compute metrics directly from DB
                report = self._compute_report_from_db()
                results.update(report)

            # Persist the report
            self._persist_report(results)

            # Log alerts for critical states
            self._log_alerts(results)

        except Exception:
            logger.exception("Report generation failed")
            results["error"] = "Report generation failed"

        return results

    def _run_edge_validation(self) -> dict | None:
        """Run edge validation using QuantEngine."""
        try:
            from quant_system.engine import QuantEngine
            from quant_system.core.types import Sport

            engine = QuantEngine(sport=Sport.GOLF)
            risk_state = engine.bankroll_mgr.get_risk_state()

            edge_report = engine.edge_validator.validate(
                bankroll=risk_state.bankroll,
                peak_bankroll=risk_state.peak_bankroll,
            )

            return {
                "edge_exists": edge_report.edge_exists,
                "system_state": edge_report.system_state.value,
                "warnings": edge_report.warnings,
                "actions": edge_report.actions,
                "metrics": {
                    "clv_last_50": edge_report.clv_last_50,
                    "clv_last_100": edge_report.clv_last_100,
                    "clv_last_250": edge_report.clv_last_250,
                    "clv_last_500": edge_report.clv_last_500,
                    "calibration_error": edge_report.calibration_error,
                    "model_roi": edge_report.model_roi,
                    "expected_roi": edge_report.expected_roi,
                    "bankroll": risk_state.bankroll,
                    "peak_bankroll": risk_state.peak_bankroll,
                    "drawdown_pct": risk_state.current_drawdown_pct,
                },
            }

        except Exception:
            logger.debug("QuantEngine not available — falling back to DB computation")
            return None

    def _compute_report_from_db(self) -> dict:
        """Compute edge report metrics directly from the database."""
        import numpy as np

        metrics = {}
        warnings = []

        with DatabaseManager.session_scope() as session:
            # CLV metrics
            for window in [50, 100, 250, 500]:
                clv_rows = (
                    session.query(CLVLog)
                    .filter(CLVLog.sport == "golf")
                    .order_by(CLVLog.calculated_at.desc())
                    .limit(window)
                    .all()
                )
                if clv_rows:
                    avg_clv = float(np.mean([c.clv_cents for c in clv_rows]))
                    beat_rate = sum(1 for c in clv_rows if c.beat_close) / len(clv_rows)
                    metrics[f"clv_last_{window}"] = round(avg_clv, 2)
                    metrics[f"clv_beat_rate_{window}"] = round(beat_rate, 4)
                else:
                    metrics[f"clv_last_{window}"] = 0.0

            # P&L metrics
            settled = (
                session.query(Bet)
                .filter(
                    Bet.sport == "golf",
                    Bet.status.in_(["won", "lost"]),
                )
                .all()
            )

            if settled:
                total_staked = sum(b.stake for b in settled)
                total_pnl = sum(b.pnl for b in settled)
                wins = sum(1 for b in settled if b.status == "won")

                metrics["total_bets"] = len(settled)
                metrics["win_rate"] = round(wins / len(settled), 4) if settled else 0.0
                metrics["roi"] = round(total_pnl / total_staked, 4) if total_staked > 0 else 0.0
                metrics["total_pnl"] = round(total_pnl, 2)
            else:
                metrics["total_bets"] = 0

            # Bankroll / drawdown (from recent edge report or bets)
            peak = max((b.stake for b in settled), default=0) * 20 if settled else 1000.0
            current = peak + sum(b.pnl for b in settled) if settled else 1000.0
            drawdown = (peak - current) / peak if peak > 0 and current < peak else 0.0

            metrics["bankroll"] = round(current, 2)
            metrics["peak_bankroll"] = round(peak, 2)
            metrics["drawdown_pct"] = round(drawdown, 4)

            # Warnings
            if metrics.get("clv_last_100", 0) < 0:
                warnings.append(f"CLV negative over last 100 bets: {metrics['clv_last_100']:.2f} cents")
            if drawdown > 0.20:
                warnings.append(f"Drawdown at {drawdown:.1%}")
            if metrics.get("roi", 0) < -0.05:
                warnings.append(f"ROI is {metrics['roi']:.1%}")

            # System state
            if drawdown > 0.50:
                state = "killed"
            elif drawdown > 0.35 or metrics.get("clv_last_250", 0) < -2.0:
                state = "suspended"
            elif drawdown > 0.20 or metrics.get("clv_last_100", 0) < -1.0:
                state = "reduced"
            else:
                state = "active"

            edge_exists = (
                metrics.get("clv_last_100", 0) > 0
                and drawdown < 0.35
            )

        return {
            "edge_exists": edge_exists,
            "system_state": state,
            "warnings": warnings,
            "actions": [],
            "metrics": metrics,
        }

    def _persist_report(self, report: dict):
        """Save edge report to database."""
        try:
            metrics = report.get("metrics", {})

            with DatabaseManager.session_scope() as session:
                # Check for existing report today
                today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                existing = (
                    session.query(EdgeReportModel)
                    .filter(
                        EdgeReportModel.sport == "golf",
                        EdgeReportModel.report_date >= today,
                    )
                    .first()
                )

                if existing:
                    # Update existing
                    edge_report = existing
                else:
                    edge_report = EdgeReportModel(sport="golf", report_date=datetime.utcnow())
                    session.add(edge_report)

                edge_report.clv_last_50 = metrics.get("clv_last_50", 0.0)
                edge_report.clv_last_100 = metrics.get("clv_last_100", 0.0)
                edge_report.clv_last_250 = metrics.get("clv_last_250", 0.0)
                edge_report.clv_last_500 = metrics.get("clv_last_500", 0.0)
                edge_report.calibration_error = metrics.get("calibration_error", 0.0)
                edge_report.model_roi = metrics.get("model_roi", metrics.get("roi", 0.0))
                edge_report.expected_roi = metrics.get("expected_roi", 0.0)
                edge_report.bankroll = metrics.get("bankroll")
                edge_report.peak_bankroll = metrics.get("peak_bankroll")
                edge_report.drawdown_pct = metrics.get("drawdown_pct")
                edge_report.edge_exists = report.get("edge_exists", False)
                edge_report.system_state = report.get("system_state", "active")
                edge_report.warnings = json.dumps(report.get("warnings", []))
                edge_report.actions = json.dumps(report.get("actions", []))

                # Get active model version
                active_model = (
                    session.query(ModelVersion)
                    .filter_by(sport="golf", is_active=True)
                    .first()
                )
                if active_model:
                    edge_report.model_version_id = active_model.id

        except Exception:
            logger.exception("Failed to persist edge report")

    def _log_alerts(self, report: dict):
        """Log critical alerts to audit log."""
        state = report.get("system_state", "active")
        warnings = report.get("warnings", [])

        if state in ("suspended", "killed"):
            self._log_audit(
                "critical",
                f"System state: {state.upper()} — {len(warnings)} warnings",
                {"warnings": warnings, "metrics": report.get("metrics", {})},
            )
        elif state == "reduced":
            self._log_audit(
                "warning",
                f"System state: REDUCED — {len(warnings)} warnings",
                {"warnings": warnings},
            )
        elif warnings:
            self._log_audit(
                "info",
                f"Edge report: ACTIVE with {len(warnings)} warnings",
                {"warnings": warnings},
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    worker = ReportWorker()
    worker.run_once()
