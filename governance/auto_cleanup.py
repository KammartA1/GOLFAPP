"""Auto cleanup — Automated maintenance of database tables and old records."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List

from database.connection import DatabaseManager
from database.models import (
    AuditLog, LineMovement, Signal, ModelVersion,
    FeatureLog, CalibrationLog, SystemStateLog,
)

logger = logging.getLogger(__name__)


class AutoCleanup:
    """Automated database maintenance and cleanup.

    Manages data retention policies to prevent unbounded growth while
    preserving all data needed for analysis and audit trails.
    """

    # Retention policies (days)
    RETENTION = {
        "audit_logs": 365,         # Keep 1 year of audit logs
        "line_movements": 180,     # Keep 6 months of line data
        "signals_unapproved": 90,  # Unapproved signals: 3 months
        "feature_log": 365,        # Feature importance: 1 year
        "calibration_log": 365,    # Calibration data: 1 year
        "system_state_log": 730,   # State changes: 2 years
        "old_model_versions": 365, # Inactive models: 1 year metadata kept
    }

    def __init__(self, sport: str = "golf", dry_run: bool = False):
        self.sport = sport
        self.dry_run = dry_run

    def run_full_cleanup(self) -> dict:
        """Run all cleanup tasks.

        Returns summary of what was cleaned up.
        """
        results = {}

        results["audit_logs"] = self._cleanup_audit_logs()
        results["line_movements"] = self._cleanup_line_movements()
        results["unapproved_signals"] = self._cleanup_unapproved_signals()
        results["feature_log"] = self._cleanup_feature_log()
        results["calibration_log"] = self._cleanup_calibration_log()
        results["system_state_log"] = self._cleanup_system_state_log()
        results["orphaned_models"] = self._cleanup_orphaned_models()

        total_deleted = sum(r.get("deleted", 0) for r in results.values())
        results["total_deleted"] = total_deleted
        results["dry_run"] = self.dry_run
        results["timestamp"] = datetime.utcnow().isoformat()

        # Log the cleanup
        if not self.dry_run and total_deleted > 0:
            self._log_cleanup(results)

        logger.info(
            "Cleanup %s: %d total records %s",
            "preview" if self.dry_run else "complete",
            total_deleted,
            "would be deleted" if self.dry_run else "deleted",
        )

        return results

    def get_table_sizes(self) -> dict:
        """Get current row counts for all managed tables."""
        with DatabaseManager.session_scope() as session:
            return {
                "audit_logs": session.query(AuditLog).count(),
                "line_movements": session.query(LineMovement).filter(
                    LineMovement.sport == self.sport
                ).count(),
                "signals": session.query(Signal).filter(
                    Signal.sport == self.sport
                ).count(),
                "feature_log": session.query(FeatureLog).filter(
                    FeatureLog.sport == self.sport
                ).count(),
                "calibration_log": session.query(CalibrationLog).filter(
                    CalibrationLog.sport == self.sport
                ).count(),
                "system_state_log": session.query(SystemStateLog).filter(
                    SystemStateLog.sport == self.sport
                ).count(),
                "model_versions": session.query(ModelVersion).filter(
                    ModelVersion.sport == self.sport
                ).count(),
            }

    # ── Cleanup tasks ───────────────────────────────────────────────────

    def _cleanup_audit_logs(self) -> dict:
        """Remove old audit logs beyond retention period."""
        cutoff = datetime.utcnow() - timedelta(days=self.RETENTION["audit_logs"])

        with DatabaseManager.session_scope() as session:
            query = session.query(AuditLog).filter(
                AuditLog.timestamp < cutoff,
                # Never delete critical/error logs
                AuditLog.level.in_(["info", "debug"]),
            )
            count = query.count()

            if not self.dry_run and count > 0:
                query.delete(synchronize_session=False)
                logger.info("Deleted %d old audit logs", count)

            return {"deleted": count, "cutoff": cutoff.isoformat()}

    def _cleanup_line_movements(self) -> dict:
        """Remove old line movement data beyond retention period.

        Preserves opening and closing lines (is_opening=True, is_closing=True).
        """
        cutoff = datetime.utcnow() - timedelta(days=self.RETENTION["line_movements"])

        with DatabaseManager.session_scope() as session:
            query = session.query(LineMovement).filter(
                LineMovement.sport == self.sport,
                LineMovement.captured_at < cutoff,
                LineMovement.is_opening == False,
                LineMovement.is_closing == False,
            )
            count = query.count()

            if not self.dry_run and count > 0:
                query.delete(synchronize_session=False)
                logger.info("Deleted %d old line movements (preserved open/close)", count)

            return {"deleted": count, "cutoff": cutoff.isoformat()}

    def _cleanup_unapproved_signals(self) -> dict:
        """Remove old signals that were never approved."""
        cutoff = datetime.utcnow() - timedelta(days=self.RETENTION["signals_unapproved"])

        with DatabaseManager.session_scope() as session:
            query = session.query(Signal).filter(
                Signal.sport == self.sport,
                Signal.generated_at < cutoff,
                Signal.approved == False,
                Signal.bet_id.is_(None),  # Never became a bet
            )
            count = query.count()

            if not self.dry_run and count > 0:
                query.delete(synchronize_session=False)
                logger.info("Deleted %d old unapproved signals", count)

            return {"deleted": count, "cutoff": cutoff.isoformat()}

    def _cleanup_feature_log(self) -> dict:
        """Remove old feature importance records beyond retention."""
        cutoff = datetime.utcnow() - timedelta(days=self.RETENTION["feature_log"])

        with DatabaseManager.session_scope() as session:
            query = session.query(FeatureLog).filter(
                FeatureLog.sport == self.sport,
                FeatureLog.report_date < cutoff,
            )
            count = query.count()

            if not self.dry_run and count > 0:
                query.delete(synchronize_session=False)
                logger.info("Deleted %d old feature log entries", count)

            return {"deleted": count, "cutoff": cutoff.isoformat()}

    def _cleanup_calibration_log(self) -> dict:
        """Remove old calibration records beyond retention."""
        cutoff = datetime.utcnow() - timedelta(days=self.RETENTION["calibration_log"])

        with DatabaseManager.session_scope() as session:
            query = session.query(CalibrationLog).filter(
                CalibrationLog.sport == self.sport,
                CalibrationLog.report_date < cutoff,
            )
            count = query.count()

            if not self.dry_run and count > 0:
                query.delete(synchronize_session=False)
                logger.info("Deleted %d old calibration log entries", count)

            return {"deleted": count, "cutoff": cutoff.isoformat()}

    def _cleanup_system_state_log(self) -> dict:
        """Remove old system state change logs beyond retention."""
        cutoff = datetime.utcnow() - timedelta(days=self.RETENTION["system_state_log"])

        with DatabaseManager.session_scope() as session:
            query = session.query(SystemStateLog).filter(
                SystemStateLog.sport == self.sport,
                SystemStateLog.timestamp < cutoff,
            )
            count = query.count()

            if not self.dry_run and count > 0:
                query.delete(synchronize_session=False)
                logger.info("Deleted %d old system state log entries", count)

            return {"deleted": count, "cutoff": cutoff.isoformat()}

    def _cleanup_orphaned_models(self) -> dict:
        """Clean up metadata for very old inactive model versions.

        Does NOT delete the row — just clears large JSON fields to save space.
        The model record is kept for audit trail purposes.
        """
        cutoff = datetime.utcnow() - timedelta(days=self.RETENTION["old_model_versions"])

        with DatabaseManager.session_scope() as session:
            old_models = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.sport == self.sport,
                    ModelVersion.is_active == False,
                    ModelVersion.retired_at.isnot(None),
                    ModelVersion.retired_at < cutoff,
                )
                .all()
            )

            count = 0
            for mv in old_models:
                if mv.hyperparameters and len(mv.hyperparameters) > 100:
                    if not self.dry_run:
                        mv.hyperparameters = "{}"  # Clear but keep structure
                        mv.feature_list = "[]"
                    count += 1

            if not self.dry_run and count > 0:
                logger.info("Cleaned metadata for %d old model versions", count)

            return {"deleted": count, "cutoff": cutoff.isoformat(), "note": "metadata_cleared_not_deleted"}

    def _log_cleanup(self, results: dict) -> None:
        """Log cleanup activity to audit log."""
        import json
        with DatabaseManager.session_scope() as session:
            session.add(AuditLog(
                sport=self.sport,
                category="system",
                level="info",
                message=f"Auto cleanup: {results['total_deleted']} records processed",
                details=json.dumps({
                    k: v for k, v in results.items()
                    if k not in ("total_deleted", "dry_run", "timestamp")
                }),
            ))
