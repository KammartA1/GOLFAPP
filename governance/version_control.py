"""Model version control — Track, register, and manage model versions."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from database.connection import DatabaseManager
from database.models import ModelVersion, Bet, CLVLog, AuditLog

logger = logging.getLogger(__name__)


class ModelVersionController:
    """Manage model version lifecycle: register, activate, retire, compare."""

    def __init__(self, sport: str = "golf"):
        self.sport = sport

    def register_version(
        self,
        version: str,
        hyperparameters: dict,
        feature_list: List[str],
        training_start: Optional[datetime] = None,
        training_end: Optional[datetime] = None,
        n_training_samples: int = 0,
        train_metrics: Optional[dict] = None,
        val_metrics: Optional[dict] = None,
        notes: str = "",
    ) -> int:
        """Register a new model version.

        Returns the model version ID.
        """
        with DatabaseManager.session_scope() as session:
            mv = ModelVersion(
                sport=self.sport,
                version=version,
                trained_at=datetime.utcnow(),
                training_start=training_start,
                training_end=training_end,
                n_training_samples=n_training_samples,
                hyperparameters=json.dumps(hyperparameters),
                feature_list=json.dumps(feature_list),
                is_active=False,  # Not active until explicitly promoted
                notes=notes,
            )

            if train_metrics:
                mv.train_accuracy = train_metrics.get("accuracy")
                mv.train_log_loss = train_metrics.get("log_loss")
                mv.train_brier_score = train_metrics.get("brier_score")

            if val_metrics:
                mv.val_accuracy = val_metrics.get("accuracy")
                mv.val_log_loss = val_metrics.get("log_loss")
                mv.val_brier_score = val_metrics.get("brier_score")

            session.add(mv)
            session.flush()
            model_id = mv.id

            # Audit log
            session.add(AuditLog(
                sport=self.sport,
                category="model",
                level="info",
                message=f"Registered model version {version} (id={model_id})",
                details=json.dumps({
                    "version": version,
                    "n_features": len(feature_list),
                    "n_training_samples": n_training_samples,
                }),
            ))

            logger.info("Registered model version %s (id=%d)", version, model_id)
            return model_id

    def promote_version(self, version_id: int) -> bool:
        """Promote a version to active, retiring the current active version.

        Only one version can be active per sport at a time.
        """
        with DatabaseManager.session_scope() as session:
            new_model = session.query(ModelVersion).get(version_id)
            if not new_model:
                logger.error("Model version %d not found", version_id)
                return False

            if new_model.sport != self.sport:
                logger.error("Model version %d is for sport %s, not %s",
                             version_id, new_model.sport, self.sport)
                return False

            # Retire current active version
            current_active = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.sport == self.sport,
                    ModelVersion.is_active == True,
                    ModelVersion.id != version_id,
                )
                .all()
            )
            for old in current_active:
                old.is_active = False
                old.retired_at = datetime.utcnow()
                old.replaced_by = version_id
                logger.info("Retired model version %s (id=%d)", old.version, old.id)

            # Activate new version
            new_model.is_active = True
            new_model.retired_at = None

            session.add(AuditLog(
                sport=self.sport,
                category="model",
                level="info",
                message=f"Promoted model {new_model.version} (id={version_id}) to active",
                details=json.dumps({
                    "promoted": version_id,
                    "retired": [old.id for old in current_active],
                }),
            ))

            logger.info("Promoted model version %s (id=%d) to active",
                         new_model.version, version_id)
            return True

    def get_active_version(self) -> Optional[dict]:
        """Get the current active model version."""
        with DatabaseManager.session_scope() as session:
            mv = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.sport == self.sport,
                    ModelVersion.is_active == True,
                )
                .order_by(ModelVersion.trained_at.desc())
                .first()
            )
            if not mv:
                return None

            return {
                "id": mv.id,
                "version": mv.version,
                "trained_at": mv.trained_at.isoformat() if mv.trained_at else None,
                "n_live_bets": mv.n_live_bets,
                "live_clv_avg": mv.live_clv_avg,
                "live_roi": mv.live_roi,
                "is_degraded": mv.is_degraded,
                "psi_score": mv.psi_score,
                "hyperparameters": json.loads(mv.hyperparameters) if mv.hyperparameters else {},
                "feature_list": json.loads(mv.feature_list) if mv.feature_list else [],
            }

    def get_version_history(self, limit: int = 20) -> List[dict]:
        """Get model version history ordered by creation date."""
        with DatabaseManager.session_scope() as session:
            versions = (
                session.query(ModelVersion)
                .filter(ModelVersion.sport == self.sport)
                .order_by(ModelVersion.trained_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": v.id,
                    "version": v.version,
                    "trained_at": v.trained_at.isoformat() if v.trained_at else None,
                    "is_active": v.is_active,
                    "is_degraded": v.is_degraded,
                    "n_live_bets": v.n_live_bets,
                    "live_clv_avg": v.live_clv_avg,
                    "live_roi": v.live_roi,
                    "retired_at": v.retired_at.isoformat() if v.retired_at else None,
                    "replaced_by": v.replaced_by,
                }
                for v in versions
            ]

    def compare_versions(self, version_id_a: int, version_id_b: int) -> dict:
        """Compare two model versions on live performance metrics."""
        with DatabaseManager.session_scope() as session:
            a = session.query(ModelVersion).get(version_id_a)
            b = session.query(ModelVersion).get(version_id_b)

            if not a or not b:
                return {"error": "One or both versions not found"}

            # Get bets for each version
            bets_a = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.model_version_id == version_id_a,
                    Bet.status.in_(["won", "lost"]),
                )
                .all()
            )
            bets_b = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.model_version_id == version_id_b,
                    Bet.status.in_(["won", "lost"]),
                )
                .all()
            )

            def _bet_stats(bets):
                if not bets:
                    return {"n_bets": 0}
                total_staked = sum(b.stake for b in bets)
                total_pnl = sum(b.pnl for b in bets)
                wins = sum(1 for b in bets if b.status == "won")
                return {
                    "n_bets": len(bets),
                    "total_pnl": round(total_pnl, 2),
                    "roi": round(total_pnl / max(total_staked, 1), 4),
                    "win_rate": round(wins / max(len(bets), 1), 4),
                }

            return {
                "version_a": {
                    "id": a.id,
                    "version": a.version,
                    "live_clv_avg": a.live_clv_avg,
                    "live_roi": a.live_roi,
                    "is_degraded": a.is_degraded,
                    "bet_stats": _bet_stats(bets_a),
                },
                "version_b": {
                    "id": b.id,
                    "version": b.version,
                    "live_clv_avg": b.live_clv_avg,
                    "live_roi": b.live_roi,
                    "is_degraded": b.is_degraded,
                    "bet_stats": _bet_stats(bets_b),
                },
            }

    def update_live_metrics(self, version_id: int) -> None:
        """Recompute live metrics for a model version from settled bets."""
        with DatabaseManager.session_scope() as session:
            mv = session.query(ModelVersion).get(version_id)
            if not mv:
                return

            bets = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.model_version_id == version_id,
                    Bet.status.in_(["won", "lost"]),
                )
                .all()
            )

            if not bets:
                return

            import numpy as np

            total_staked = sum(b.stake for b in bets)
            total_pnl = sum(b.pnl for b in bets)
            wins = sum(1 for b in bets if b.status == "won")

            mv.n_live_bets = len(bets)
            mv.live_roi = round(total_pnl / max(total_staked, 1), 4)
            mv.live_accuracy = round(wins / len(bets), 4)

            # Live log loss
            log_loss_sum = 0.0
            n_valid = 0
            for b in bets:
                if b.model_prob and 0 < b.model_prob < 1:
                    outcome = 1.0 if b.status == "won" else 0.0
                    prob = max(min(b.model_prob, 0.999), 0.001)
                    import math
                    log_loss_sum += -(outcome * math.log(prob) + (1 - outcome) * math.log(1 - prob))
                    n_valid += 1
            if n_valid > 0:
                mv.live_log_loss = round(log_loss_sum / n_valid, 4)

            # Live CLV average
            bet_ids = [b.bet_id for b in bets]
            clv_records = (
                session.query(CLVLog)
                .filter(CLVLog.bet_id.in_(bet_ids))
                .all()
            )
            if clv_records:
                mv.live_clv_avg = round(
                    float(np.mean([r.clv_cents for r in clv_records])), 2
                )

            logger.info(
                "Updated live metrics for model %s: %d bets, ROI=%.2f%%, CLV=%.2fc",
                mv.version, len(bets), (mv.live_roi or 0) * 100, mv.live_clv_avg or 0,
            )
