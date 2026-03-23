"""Model rollback — Safe rollback to previous model version with validation."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from database.connection import DatabaseManager
from database.models import ModelVersion, AuditLog

logger = logging.getLogger(__name__)


class ModelRollback:
    """Safely rollback to a previous model version.

    Rollback is performed when the current model shows degradation that
    cannot be fixed by retraining. The previous version is restored with
    full audit logging.
    """

    def __init__(self, sport: str = "golf"):
        self.sport = sport

    def get_rollback_candidates(self, limit: int = 5) -> list:
        """Get recent model versions that could be rolled back to.

        Only returns versions that were previously active and had
        positive live performance.
        """
        with DatabaseManager.session_scope() as session:
            candidates = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.sport == self.sport,
                    ModelVersion.is_active == False,
                    ModelVersion.retired_at.isnot(None),
                )
                .order_by(ModelVersion.retired_at.desc())
                .limit(limit)
                .all()
            )

            results = []
            for mv in candidates:
                # Assess fitness for rollback
                is_viable = True
                reasons = []

                if mv.is_degraded:
                    is_viable = False
                    reasons.append("Was flagged as degraded")

                if mv.live_clv_avg is not None and mv.live_clv_avg < 0:
                    is_viable = False
                    reasons.append(f"Had negative live CLV: {mv.live_clv_avg:.2f}c")

                if mv.live_roi is not None and mv.live_roi < -0.05:
                    is_viable = False
                    reasons.append(f"Had negative live ROI: {mv.live_roi:.1%}")

                # Age penalty
                age_days = (datetime.utcnow() - mv.trained_at).days if mv.trained_at else 999
                if age_days > 120:
                    reasons.append(f"Model is {age_days} days old — may be stale")
                    # Still viable, just warned

                results.append({
                    "id": mv.id,
                    "version": mv.version,
                    "trained_at": mv.trained_at.isoformat() if mv.trained_at else None,
                    "retired_at": mv.retired_at.isoformat() if mv.retired_at else None,
                    "live_clv_avg": mv.live_clv_avg,
                    "live_roi": mv.live_roi,
                    "n_live_bets": mv.n_live_bets,
                    "age_days": age_days,
                    "is_viable": is_viable,
                    "concerns": reasons,
                })

            return results

    def rollback_to(
        self,
        version_id: int,
        reason: str = "Manual rollback",
        force: bool = False,
    ) -> dict:
        """Rollback to a specific model version.

        Args:
            version_id: ID of the version to rollback to.
            reason: Why the rollback is happening.
            force: If True, skip viability checks.

        Returns:
            {
                'success': bool,
                'previous_version': str,
                'restored_version': str,
                'reason': str,
            }
        """
        with DatabaseManager.session_scope() as session:
            target = session.query(ModelVersion).get(version_id)
            if not target:
                return {"success": False, "reason": f"Version {version_id} not found"}

            if target.sport != self.sport:
                return {"success": False, "reason": "Version belongs to different sport"}

            # Viability check
            if not force:
                if target.is_degraded:
                    return {
                        "success": False,
                        "reason": "Target version is flagged as degraded. Use force=True to override.",
                    }
                if target.live_clv_avg is not None and target.live_clv_avg < -1.0:
                    return {
                        "success": False,
                        "reason": f"Target version had deeply negative CLV ({target.live_clv_avg:.2f}c). Use force=True.",
                    }

            # Find current active version
            current = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.sport == self.sport,
                    ModelVersion.is_active == True,
                )
                .first()
            )

            current_version_str = current.version if current else "none"

            # Retire current
            if current:
                current.is_active = False
                current.retired_at = datetime.utcnow()
                current.replaced_by = version_id

            # Restore target
            target.is_active = True
            target.retired_at = None
            target.replaced_by = None

            # Audit log
            session.add(AuditLog(
                sport=self.sport,
                category="model",
                level="warning",
                message=(
                    f"ROLLBACK: {current_version_str} -> {target.version} "
                    f"(id={version_id}). Reason: {reason}"
                ),
                details=json.dumps({
                    "rollback_from": current.id if current else None,
                    "rollback_to": version_id,
                    "reason": reason,
                    "forced": force,
                }),
            ))

            logger.warning(
                "Model rollback: %s -> %s (reason: %s)",
                current_version_str, target.version, reason,
            )

            return {
                "success": True,
                "previous_version": current_version_str,
                "restored_version": target.version,
                "restored_version_id": version_id,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def auto_rollback_if_degraded(self) -> Optional[dict]:
        """Automatically rollback if current model is degraded and a viable
        previous version exists.

        Returns rollback result if executed, None otherwise.
        """
        with DatabaseManager.session_scope() as session:
            current = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.sport == self.sport,
                    ModelVersion.is_active == True,
                )
                .first()
            )

            if not current:
                return None

            # Check if current is degraded
            needs_rollback = False
            rollback_reason = ""

            if current.is_degraded:
                needs_rollback = True
                rollback_reason = "Current model flagged as degraded"
            elif current.live_clv_avg is not None and current.live_clv_avg < -1.5:
                needs_rollback = True
                rollback_reason = f"Current model CLV deeply negative: {current.live_clv_avg:.2f}c"
            elif current.psi_score is not None and current.psi_score > 0.30:
                needs_rollback = True
                rollback_reason = f"Severe feature drift: PSI={current.psi_score:.3f}"

            if not needs_rollback:
                return None

        # Find best rollback candidate
        candidates = self.get_rollback_candidates()
        viable = [c for c in candidates if c["is_viable"]]

        if not viable:
            logger.warning("Model degraded but no viable rollback candidates found")
            return None

        # Pick best candidate (highest CLV)
        best = max(viable, key=lambda c: c.get("live_clv_avg") or 0)

        return self.rollback_to(
            best["id"],
            reason=f"Auto-rollback: {rollback_reason}",
        )
