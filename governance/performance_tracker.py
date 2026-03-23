"""Performance tracker — Ongoing model and system performance monitoring."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from database.connection import DatabaseManager
from database.models import (
    Bet, CLVLog, ModelVersion, EdgeReport, CalibrationLog, AuditLog,
)

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track ongoing model and system performance with degradation detection."""

    # Degradation thresholds
    PSI_WARNING = 0.10
    PSI_CRITICAL = 0.25
    CLV_DEGRADATION_THRESHOLD = -0.5  # Avg CLV below this → degraded
    BRIER_DEGRADATION_THRESHOLD = 0.30

    def __init__(self, sport: str = "golf"):
        self.sport = sport

    def compute_performance_snapshot(self) -> dict:
        """Compute comprehensive performance snapshot.

        Returns:
            {
                'overall_health': str ('healthy', 'warning', 'degraded'),
                'roi': dict,
                'clv': dict,
                'calibration': dict,
                'win_rate': dict,
                'by_market_type': dict,
                'by_time_window': dict,
                'degradation_flags': list,
            }
        """
        with DatabaseManager.session_scope() as session:
            bets = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.status.in_(["won", "lost"]),
                )
                .order_by(Bet.settled_at.desc())
                .limit(500)
                .all()
            )

            clv_records = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport)
                .order_by(CLVLog.calculated_at.desc())
                .limit(500)
                .all()
            )

            if not bets:
                return {"overall_health": "no_data", "degradation_flags": []}

            now = datetime.utcnow()
            degradation_flags = []

            # === ROI ===
            roi_data = self._compute_roi_windows(bets, now)

            # === CLV ===
            clv_data = self._compute_clv_windows(clv_records)

            # === Calibration ===
            calibration_data = self._compute_calibration(bets)

            # === Win rate ===
            win_rate_data = self._compute_win_rate_windows(bets, now)

            # === By market type ===
            by_market = self._compute_by_market_type(bets)

            # === By time window ===
            by_window = {
                "last_7d": self._window_stats(bets, now, days=7),
                "last_30d": self._window_stats(bets, now, days=30),
                "last_90d": self._window_stats(bets, now, days=90),
                "all_time": self._window_stats(bets, now, days=36500),
            }

            # === Degradation detection ===
            if clv_data.get("avg_100", 0) < self.CLV_DEGRADATION_THRESHOLD:
                degradation_flags.append(
                    f"CLV degraded: {clv_data['avg_100']:.2f}c (threshold {self.CLV_DEGRADATION_THRESHOLD}c)"
                )

            if calibration_data.get("brier_score", 0) > self.BRIER_DEGRADATION_THRESHOLD:
                degradation_flags.append(
                    f"Brier score degraded: {calibration_data['brier_score']:.4f}"
                )

            # ROI declining
            if roi_data.get("roi_30d", 0) < -0.05:
                degradation_flags.append(
                    f"30-day ROI negative: {roi_data['roi_30d']:.1%}"
                )

            # Determine overall health
            if len(degradation_flags) >= 2:
                overall_health = "degraded"
            elif len(degradation_flags) >= 1:
                overall_health = "warning"
            else:
                overall_health = "healthy"

            return {
                "overall_health": overall_health,
                "roi": roi_data,
                "clv": clv_data,
                "calibration": calibration_data,
                "win_rate": win_rate_data,
                "by_market_type": by_market,
                "by_time_window": by_window,
                "degradation_flags": degradation_flags,
                "n_bets_total": len(bets),
                "timestamp": now.isoformat(),
            }

    def detect_psi_drift(self, version_id: int) -> dict:
        """Compute Population Stability Index for feature distribution drift.

        Compares the training distribution (from model version config) to the
        recent live distribution.

        PSI < 0.10 → No significant change
        0.10 <= PSI < 0.25 → Moderate change, monitor
        PSI >= 0.25 → Significant change, retrain
        """
        with DatabaseManager.session_scope() as session:
            mv = session.query(ModelVersion).get(version_id)
            if not mv or not mv.hyperparameters:
                return {"psi": None, "status": "no_data"}

            import json
            hp = json.loads(mv.hyperparameters)
            train_dist = hp.get("feature_distribution", {})

            if not train_dist:
                return {"psi": None, "status": "no_training_distribution"}

            # Get recent bets for live distribution
            bets = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.model_version_id == version_id,
                )
                .order_by(Bet.timestamp.desc())
                .limit(200)
                .all()
            )

            if len(bets) < 30:
                return {"psi": None, "status": "insufficient_live_data"}

            # Compute PSI on model_prob distribution
            train_probs = train_dist.get("model_prob_bins", None)
            if not train_probs:
                # Fall back: use first half vs second half of live bets
                all_probs = [b.model_prob for b in bets if b.model_prob]
                mid = len(all_probs) // 2
                train_probs_arr = np.array(all_probs[mid:])
                live_probs_arr = np.array(all_probs[:mid])
            else:
                train_probs_arr = np.array(train_probs)
                live_probs_arr = np.array([b.model_prob for b in bets if b.model_prob])

            psi = self._compute_psi(train_probs_arr, live_probs_arr, n_bins=10)

            # Update model version
            mv.psi_score = round(psi, 4)
            if psi >= self.PSI_CRITICAL:
                mv.is_degraded = True

            status = "stable"
            if psi >= self.PSI_CRITICAL:
                status = "critical_drift"
            elif psi >= self.PSI_WARNING:
                status = "moderate_drift"

            return {
                "psi": round(psi, 4),
                "status": status,
                "threshold_warning": self.PSI_WARNING,
                "threshold_critical": self.PSI_CRITICAL,
            }

    def generate_daily_report(self) -> dict:
        """Generate daily performance report and persist to EdgeReport table."""
        snapshot = self.compute_performance_snapshot()

        with DatabaseManager.session_scope() as session:
            today = datetime.utcnow().date()

            # Check if report already exists for today
            existing = (
                session.query(EdgeReport)
                .filter(
                    EdgeReport.sport == self.sport,
                    EdgeReport.report_date >= datetime(today.year, today.month, today.day),
                )
                .first()
            )

            clv = snapshot.get("clv", {})
            roi = snapshot.get("roi", {})
            calibration = snapshot.get("calibration", {})

            # Determine system state
            if snapshot["overall_health"] == "degraded":
                system_state = "suspended"
            elif snapshot["overall_health"] == "warning":
                system_state = "reduced"
            else:
                system_state = "active"

            edge_exists = clv.get("avg_100", 0) > 0 and clv.get("beat_rate_100", 0.5) > 0.48

            import json
            warnings_json = json.dumps(snapshot.get("degradation_flags", []))

            if existing:
                existing.clv_last_50 = clv.get("avg_50", 0)
                existing.clv_last_100 = clv.get("avg_100", 0)
                existing.clv_last_250 = clv.get("avg_250", 0)
                existing.clv_last_500 = clv.get("avg_500", 0)
                existing.calibration_error = calibration.get("ece", 0)
                existing.model_roi = roi.get("roi_all", 0)
                existing.edge_exists = edge_exists
                existing.system_state = system_state
                existing.warnings = warnings_json
            else:
                report = EdgeReport(
                    sport=self.sport,
                    report_date=datetime.utcnow(),
                    clv_last_50=clv.get("avg_50", 0),
                    clv_last_100=clv.get("avg_100", 0),
                    clv_last_250=clv.get("avg_250", 0),
                    clv_last_500=clv.get("avg_500", 0),
                    calibration_error=calibration.get("ece", 0),
                    model_roi=roi.get("roi_all", 0),
                    edge_exists=edge_exists,
                    system_state=system_state,
                    warnings=warnings_json,
                )
                session.add(report)

        return snapshot

    # ── Internal helpers ────────────────────────────────────────────────

    def _compute_roi_windows(self, bets: list, now: datetime) -> dict:
        """Compute ROI across different time windows."""
        result = {}
        for label, days in [("7d", 7), ("30d", 30), ("90d", 90), ("all", 36500)]:
            cutoff = now - timedelta(days=days)
            subset = [b for b in bets if b.settled_at and b.settled_at >= cutoff]
            if subset:
                staked = sum(b.stake for b in subset)
                pnl = sum(b.pnl for b in subset)
                result[f"roi_{label}"] = round(pnl / max(staked, 1), 4)
                result[f"pnl_{label}"] = round(pnl, 2)
                result[f"n_bets_{label}"] = len(subset)
            else:
                result[f"roi_{label}"] = 0.0
                result[f"pnl_{label}"] = 0.0
                result[f"n_bets_{label}"] = 0
        return result

    def _compute_clv_windows(self, clv_records: list) -> dict:
        """Compute CLV across rolling windows."""
        result = {}
        for window in [25, 50, 100, 250, 500]:
            subset = clv_records[:window]
            if subset:
                values = [r.clv_cents for r in subset]
                result[f"avg_{window}"] = round(float(np.mean(values)), 2)
                result[f"median_{window}"] = round(float(np.median(values)), 2)
                result[f"beat_rate_{window}"] = round(
                    float(np.mean([1.0 if r.beat_close else 0.0 for r in subset])), 4
                )
        return result

    def _compute_calibration(self, bets: list) -> dict:
        """Compute calibration metrics from settled bets."""
        valid = [(b.model_prob, 1.0 if b.status == "won" else 0.0)
                 for b in bets if b.model_prob and 0 < b.model_prob < 1]

        if len(valid) < 20:
            return {"brier_score": None, "ece": None}

        probs = np.array([v[0] for v in valid])
        outcomes = np.array([v[1] for v in valid])

        brier = float(np.mean((probs - outcomes) ** 2))

        # ECE with 10 bins
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_acc = float(np.mean(outcomes[mask]))
                bin_conf = float(np.mean(probs[mask]))
                ece += np.sum(mask) / len(probs) * abs(bin_acc - bin_conf)

        return {
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
            "n_valid": len(valid),
        }

    def _compute_win_rate_windows(self, bets: list, now: datetime) -> dict:
        """Compute win rate across time windows."""
        result = {}
        for label, days in [("7d", 7), ("30d", 30), ("90d", 90), ("all", 36500)]:
            cutoff = now - timedelta(days=days)
            subset = [b for b in bets if b.settled_at and b.settled_at >= cutoff]
            if subset:
                wins = sum(1 for b in subset if b.status == "won")
                result[f"wr_{label}"] = round(wins / len(subset), 4)
        return result

    def _compute_by_market_type(self, bets: list) -> dict:
        """Breakdown performance by bet/market type."""
        from collections import defaultdict
        by_type = defaultdict(list)
        for b in bets:
            by_type[b.bet_type or "unknown"].append(b)

        result = {}
        for mtype, type_bets in by_type.items():
            staked = sum(b.stake for b in type_bets)
            pnl = sum(b.pnl for b in type_bets)
            wins = sum(1 for b in type_bets if b.status == "won")
            result[mtype] = {
                "n_bets": len(type_bets),
                "roi": round(pnl / max(staked, 1), 4),
                "pnl": round(pnl, 2),
                "win_rate": round(wins / max(len(type_bets), 1), 4),
            }
        return result

    def _window_stats(self, bets: list, now: datetime, days: int) -> dict:
        """Compute stats for a specific time window."""
        cutoff = now - timedelta(days=days)
        subset = [b for b in bets if b.settled_at and b.settled_at >= cutoff]
        if not subset:
            return {"n_bets": 0}

        staked = sum(b.stake for b in subset)
        pnl = sum(b.pnl for b in subset)
        wins = sum(1 for b in subset if b.status == "won")

        return {
            "n_bets": len(subset),
            "pnl": round(pnl, 2),
            "roi": round(pnl / max(staked, 1), 4),
            "win_rate": round(wins / len(subset), 4),
            "avg_stake": round(staked / len(subset), 2),
        }

    @staticmethod
    def _compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        """Compute Population Stability Index between two distributions."""
        eps = 1e-4
        bin_edges = np.linspace(
            min(expected.min(), actual.min()) - eps,
            max(expected.max(), actual.max()) + eps,
            n_bins + 1,
        )

        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        expected_pct = (expected_counts + eps) / (len(expected) + n_bins * eps)
        actual_pct = (actual_counts + eps) / (len(actual) + n_bins * eps)

        psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
        return max(0.0, psi)
