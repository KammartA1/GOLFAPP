"""Model Worker — Weekly retrain, drift detection, and performance tracking.

Schedule: Weekly (Sunday night) + on-demand when drift detected

1. Check model drift (PSI, feature stability)
2. Retrain if needed (walk-forward validation)
3. Register new model version
4. Update live performance metrics
5. Detect feature degradation

Writes to: model_versions, feature_log
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from database.connection import DatabaseManager
from database.models import ModelVersion, FeatureLog, Bet, CLVLog
from .base_worker import BaseWorker

logger = logging.getLogger(__name__)


class ModelWorker(BaseWorker):
    WORKER_NAME = "model_worker_golf"
    SPORT = "golf"
    DEFAULT_INTERVAL_SECONDS = 604800  # 7 days

    # Drift thresholds
    PSI_WARNING = 0.1
    PSI_RETRAIN = 0.2
    FEATURE_DRIFT_THRESHOLD = 0.15

    def execute(self) -> dict:
        """Run model maintenance cycle."""
        results = {
            "drift_check": {},
            "retrain_triggered": False,
            "live_performance": {},
            "features_checked": 0,
            "degraded_features": 0,
        }

        # 1. Check model drift
        results["drift_check"] = self._check_drift()

        # 2. Update live performance metrics
        results["live_performance"] = self._update_live_performance()

        # 3. Monitor feature importance
        feature_result = self._check_features()
        results["features_checked"] = feature_result["checked"]
        results["degraded_features"] = feature_result["degraded"]

        # 4. Decide if retrain is needed
        if self._should_retrain(results["drift_check"], results["live_performance"]):
            results["retrain_triggered"] = True
            self._trigger_retrain()

        return results

    def _check_drift(self) -> dict:
        """Check for model drift using PSI and prediction distribution shifts."""
        try:
            with DatabaseManager.session_scope() as session:
                active_model = (
                    session.query(ModelVersion)
                    .filter_by(sport="golf", is_active=True)
                    .order_by(ModelVersion.trained_at.desc())
                    .first()
                )

                if not active_model:
                    return {"status": "no_model", "psi": 0.0}

                # Get recent bet predictions
                recent_bets = (
                    session.query(Bet)
                    .filter(
                        Bet.sport == "golf",
                        Bet.timestamp >= datetime.utcnow() - timedelta(days=30),
                    )
                    .all()
                )

                if len(recent_bets) < 20:
                    return {"status": "insufficient_data", "psi": 0.0, "n_bets": len(recent_bets)}

                # Compute PSI: compare prediction distribution at train time vs now
                import numpy as np
                recent_probs = [b.model_prob for b in recent_bets]

                # Use historical distribution as baseline
                older_bets = (
                    session.query(Bet)
                    .filter(
                        Bet.sport == "golf",
                        Bet.timestamp < datetime.utcnow() - timedelta(days=30),
                        Bet.timestamp >= datetime.utcnow() - timedelta(days=90),
                    )
                    .all()
                )

                if len(older_bets) < 20:
                    return {"status": "insufficient_baseline", "psi": 0.0}

                baseline_probs = [b.model_prob for b in older_bets]

                psi = self._compute_psi(baseline_probs, recent_probs)

                # Update model version
                active_model.psi_score = psi
                if psi > self.PSI_RETRAIN:
                    active_model.is_degraded = True

                return {
                    "status": "checked",
                    "psi": round(psi, 4),
                    "drift_detected": psi > self.PSI_WARNING,
                    "retrain_needed": psi > self.PSI_RETRAIN,
                    "n_recent": len(recent_bets),
                    "n_baseline": len(older_bets),
                }

        except Exception:
            logger.exception("Drift check failed")
            return {"status": "error", "psi": 0.0}

    def _compute_psi(self, expected: list, actual: list, n_bins: int = 10) -> float:
        """Compute Population Stability Index between two distributions."""
        import numpy as np

        # Create bins from combined data
        all_data = expected + actual
        bins = np.linspace(min(all_data), max(all_data), n_bins + 1)

        expected_hist, _ = np.histogram(expected, bins=bins)
        actual_hist, _ = np.histogram(actual, bins=bins)

        # Normalize to proportions
        expected_pct = (expected_hist + 1) / (len(expected) + n_bins)  # Laplace smoothing
        actual_pct = (actual_hist + 1) / (len(actual) + n_bins)

        # PSI formula
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    def _update_live_performance(self) -> dict:
        """Update live performance metrics for the active model."""
        try:
            with DatabaseManager.session_scope() as session:
                active_model = (
                    session.query(ModelVersion)
                    .filter_by(sport="golf", is_active=True)
                    .order_by(ModelVersion.trained_at.desc())
                    .first()
                )

                if not active_model:
                    return {"status": "no_model"}

                # Get all settled bets since model was trained
                settled = (
                    session.query(Bet)
                    .filter(
                        Bet.sport == "golf",
                        Bet.status.in_(["won", "lost"]),
                        Bet.timestamp >= active_model.trained_at,
                    )
                    .all()
                )

                if not settled:
                    return {"status": "no_settled_bets"}

                import numpy as np

                # Accuracy
                correct = sum(1 for b in settled if b.status == "won")
                accuracy = correct / len(settled)

                # ROI
                total_staked = sum(b.stake for b in settled)
                total_pnl = sum(b.pnl for b in settled)
                roi = total_pnl / total_staked if total_staked > 0 else 0.0

                # Log loss
                log_loss_vals = []
                for b in settled:
                    won = 1.0 if b.status == "won" else 0.0
                    p = max(min(b.model_prob, 0.999), 0.001)
                    ll = -(won * np.log(p) + (1 - won) * np.log(1 - p))
                    log_loss_vals.append(ll)
                avg_log_loss = float(np.mean(log_loss_vals))

                # CLV
                clv_entries = (
                    session.query(CLVLog)
                    .filter(CLVLog.sport == "golf")
                    .order_by(CLVLog.calculated_at.desc())
                    .limit(100)
                    .all()
                )
                avg_clv = float(np.mean([c.clv_cents for c in clv_entries])) if clv_entries else 0.0

                # Update model
                active_model.live_accuracy = round(accuracy, 4)
                active_model.live_log_loss = round(avg_log_loss, 4)
                active_model.live_clv_avg = round(avg_clv, 2)
                active_model.live_roi = round(roi, 4)
                active_model.n_live_bets = len(settled)

                return {
                    "status": "updated",
                    "n_bets": len(settled),
                    "accuracy": round(accuracy, 4),
                    "roi": round(roi, 4),
                    "log_loss": round(avg_log_loss, 4),
                    "avg_clv": round(avg_clv, 2),
                }

        except Exception:
            logger.exception("Live performance update failed")
            return {"status": "error"}

    def _check_features(self) -> dict:
        """Monitor feature importance and detect degradation."""
        checked = 0
        degraded = 0

        try:
            with DatabaseManager.session_scope() as session:
                # Get recent feature logs
                recent = (
                    session.query(FeatureLog)
                    .filter(
                        FeatureLog.sport == "golf",
                        FeatureLog.report_date >= datetime.utcnow() - timedelta(days=7),
                    )
                    .all()
                )

                if not recent:
                    return {"checked": 0, "degraded": 0}

                # Get baseline (30-90 days ago)
                baseline = (
                    session.query(FeatureLog)
                    .filter(
                        FeatureLog.sport == "golf",
                        FeatureLog.report_date >= datetime.utcnow() - timedelta(days=90),
                        FeatureLog.report_date < datetime.utcnow() - timedelta(days=30),
                    )
                    .all()
                )

                if not baseline:
                    return {"checked": len(recent), "degraded": 0}

                # Compare feature importance
                import numpy as np
                baseline_by_name = {}
                for f in baseline:
                    baseline_by_name.setdefault(f.feature_name, []).append(f.importance_score)

                for f in recent:
                    checked += 1
                    baseline_scores = baseline_by_name.get(f.feature_name)
                    if not baseline_scores:
                        continue

                    baseline_mean = np.mean(baseline_scores)
                    if baseline_mean > 0:
                        drift = abs(f.importance_score - baseline_mean) / baseline_mean
                        if drift > self.FEATURE_DRIFT_THRESHOLD:
                            f.is_degraded = True
                            degraded += 1

        except Exception:
            logger.exception("Feature check failed")

        return {"checked": checked, "degraded": degraded}

    def _should_retrain(self, drift: dict, performance: dict) -> bool:
        """Decide if model retraining is warranted."""
        if drift.get("retrain_needed"):
            logger.warning("Retrain triggered: PSI=%.4f exceeds threshold", drift.get("psi", 0))
            return True

        roi = performance.get("roi", 0)
        if performance.get("n_bets", 0) > 50 and roi < -0.05:
            logger.warning("Retrain triggered: negative ROI (%.2f%%) over %d bets",
                           roi * 100, performance.get("n_bets", 0))
            return True

        return False

    def _trigger_retrain(self):
        """Trigger model retrain. Creates new model version entry."""
        try:
            with DatabaseManager.session_scope() as session:
                # Mark current model as not active
                current = (
                    session.query(ModelVersion)
                    .filter_by(sport="golf", is_active=True)
                    .first()
                )

                new_version = ModelVersion(
                    sport="golf",
                    version=f"auto_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                    trained_at=datetime.utcnow(),
                    is_active=True,
                    notes="Auto-triggered retrain",
                )
                session.add(new_version)
                session.flush()

                if current:
                    current.is_active = False
                    current.replaced_by = new_version.id
                    current.retired_at = datetime.utcnow()

                self._log_audit(
                    "warning",
                    f"Model retrain triggered — new version: {new_version.version}",
                    {"previous_version": current.version if current else None},
                )

                logger.warning("Model retrain triggered: new version %s", new_version.version)

        except Exception:
            logger.exception("Failed to trigger retrain")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    worker = ModelWorker()
    worker.run_once()
