"""Feature importance tracker — Monitor feature value and detect degradation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from database.connection import DatabaseManager
from database.models import Bet, FeatureLog, AuditLog

logger = logging.getLogger(__name__)


class FeatureImportanceTracker:
    """Track feature importance over time and detect degradation.

    Monitors whether individual features (SG components, course fit, weather, etc.)
    are still contributing predictive value. Features that degrade should be
    investigated or removed.
    """

    # Features tracked for golf betting
    GOLF_FEATURES = [
        "sg_total", "sg_ott", "sg_app", "sg_atg", "sg_putt",
        "course_fit", "form_recent", "form_baseline",
        "driving_distance", "driving_accuracy",
        "weather_wind", "weather_rain",
        "field_strength", "pressure_performance",
        "course_history", "cut_probability",
        "wave_advantage", "fatigue_score",
    ]

    DIRECTIONAL_ACCURACY_THRESHOLD = 0.52  # Feature should predict direction > 52%
    IMPORTANCE_DECLINE_THRESHOLD = 0.50     # >50% decline from peak → flagged

    def __init__(self, sport: str = "golf"):
        self.sport = sport

    def compute_feature_importance(self, lookback_days: int = 90) -> List[dict]:
        """Compute importance scores for all tracked features.

        Uses a proxy approach: for each feature, measure how well it
        predicts bet outcomes (win/loss) using directional accuracy.
        """
        with DatabaseManager.session_scope() as session:
            cutoff = datetime.utcnow() - timedelta(days=lookback_days)
            bets = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.status.in_(["won", "lost"]),
                    Bet.settled_at >= cutoff,
                )
                .all()
            )

            if len(bets) < 30:
                return []

            results = []

            for feature_name in self.GOLF_FEATURES:
                importance = self._compute_single_feature_importance(
                    bets, feature_name
                )
                if importance:
                    results.append(importance)

                    # Persist to FeatureLog
                    fl = FeatureLog(
                        sport=self.sport,
                        report_date=datetime.utcnow(),
                        feature_name=feature_name,
                        importance_score=importance["importance_score"],
                        directional_accuracy=importance.get("directional_accuracy"),
                        n_samples=importance["n_samples"],
                        is_degraded=importance["is_degraded"],
                    )
                    session.add(fl)

            return results

    def get_feature_trends(self, feature_name: str, n_periods: int = 10) -> dict:
        """Get historical importance trend for a specific feature."""
        with DatabaseManager.session_scope() as session:
            records = (
                session.query(FeatureLog)
                .filter(
                    FeatureLog.sport == self.sport,
                    FeatureLog.feature_name == feature_name,
                )
                .order_by(FeatureLog.report_date.desc())
                .limit(n_periods)
                .all()
            )

            if not records:
                return {"feature": feature_name, "has_data": False}

            scores = [r.importance_score for r in records]
            dates = [r.report_date.isoformat() for r in records]

            # Trend detection
            if len(scores) >= 3:
                x = np.arange(len(scores))
                from scipy import stats as sp_stats
                slope, _, _, p_val, _ = sp_stats.linregress(x, scores[::-1])  # Reverse for chronological
                trend = "improving" if slope > 0.01 and p_val < 0.1 else \
                        "declining" if slope < -0.01 and p_val < 0.1 else "stable"
            else:
                slope = 0.0
                p_val = 1.0
                trend = "insufficient_data"

            peak_score = max(scores)
            current_score = scores[0]
            decline_from_peak = 1.0 - (current_score / max(peak_score, 0.001))

            return {
                "feature": feature_name,
                "has_data": True,
                "current_score": round(current_score, 4),
                "peak_score": round(peak_score, 4),
                "decline_from_peak": round(decline_from_peak, 4),
                "trend": trend,
                "slope": round(slope, 4),
                "p_value": round(p_val, 4),
                "history": [
                    {"date": d, "score": round(s, 4)}
                    for d, s in zip(dates, scores)
                ],
            }

    def get_degraded_features(self) -> List[dict]:
        """Get all features currently flagged as degraded."""
        with DatabaseManager.session_scope() as session:
            # Get most recent entry per feature
            degraded = []
            for feature_name in self.GOLF_FEATURES:
                latest = (
                    session.query(FeatureLog)
                    .filter(
                        FeatureLog.sport == self.sport,
                        FeatureLog.feature_name == feature_name,
                    )
                    .order_by(FeatureLog.report_date.desc())
                    .first()
                )
                if latest and latest.is_degraded:
                    degraded.append({
                        "feature": feature_name,
                        "importance_score": round(latest.importance_score, 4),
                        "directional_accuracy": round(latest.directional_accuracy, 4) if latest.directional_accuracy else None,
                        "last_checked": latest.report_date.isoformat(),
                    })

            return degraded

    def rank_features(self) -> List[dict]:
        """Rank all features by current importance score."""
        with DatabaseManager.session_scope() as session:
            ranked = []
            for feature_name in self.GOLF_FEATURES:
                latest = (
                    session.query(FeatureLog)
                    .filter(
                        FeatureLog.sport == self.sport,
                        FeatureLog.feature_name == feature_name,
                    )
                    .order_by(FeatureLog.report_date.desc())
                    .first()
                )
                if latest:
                    ranked.append({
                        "feature": feature_name,
                        "importance_score": round(latest.importance_score, 4),
                        "directional_accuracy": round(latest.directional_accuracy, 4) if latest.directional_accuracy else None,
                        "is_degraded": latest.is_degraded,
                    })

            ranked.sort(key=lambda x: x["importance_score"], reverse=True)
            for i, r in enumerate(ranked):
                r["rank"] = i + 1

            return ranked

    # ── Internal ────────────────────────────────────────────────────────

    def _compute_single_feature_importance(
        self, bets: list, feature_name: str
    ) -> Optional[dict]:
        """Compute importance for a single feature using feature snapshots.

        Extracts the feature value from bet feature snapshots and computes
        correlation with outcomes.
        """
        feature_values = []
        outcomes = []

        for bet in bets:
            if not bet.features_snapshot:
                continue

            try:
                features = json.loads(bet.features_snapshot)
            except (json.JSONDecodeError, TypeError):
                continue

            val = features.get(feature_name)
            if val is None:
                continue

            try:
                val = float(val)
            except (ValueError, TypeError):
                continue

            feature_values.append(val)
            outcomes.append(1.0 if bet.status == "won" else 0.0)

        if len(feature_values) < 20:
            # Not enough data — compute proxy from model confidence
            return self._proxy_importance(bets, feature_name)

        fv = np.array(feature_values)
        oc = np.array(outcomes)

        # Importance score: absolute correlation with outcome
        if np.std(fv) < 1e-10:
            corr = 0.0
        else:
            corr = float(np.abs(np.corrcoef(fv, oc)[0, 1]))
            if np.isnan(corr):
                corr = 0.0

        # Directional accuracy: when feature is high, does bet win more?
        median_val = float(np.median(fv))
        high_mask = fv > median_val
        low_mask = fv <= median_val

        if np.sum(high_mask) > 5 and np.sum(low_mask) > 5:
            high_wr = float(np.mean(oc[high_mask]))
            low_wr = float(np.mean(oc[low_mask]))
            directional_accuracy = max(high_wr, low_wr)
        else:
            directional_accuracy = 0.5

        # Degradation check
        is_degraded = (
            corr < 0.02 or
            directional_accuracy < self.DIRECTIONAL_ACCURACY_THRESHOLD
        )

        return {
            "feature": feature_name,
            "importance_score": round(corr, 4),
            "directional_accuracy": round(directional_accuracy, 4),
            "n_samples": len(feature_values),
            "is_degraded": is_degraded,
        }

    def _proxy_importance(self, bets: list, feature_name: str) -> Optional[dict]:
        """Proxy importance when direct feature values aren't available.

        Uses the feature name to assign a prior importance based on
        golf domain knowledge.
        """
        # Domain-informed prior importance scores
        PRIORS = {
            "sg_total": 0.35, "sg_ott": 0.15, "sg_app": 0.20,
            "sg_atg": 0.10, "sg_putt": 0.12,
            "course_fit": 0.18, "form_recent": 0.14, "form_baseline": 0.12,
            "driving_distance": 0.08, "driving_accuracy": 0.07,
            "weather_wind": 0.10, "weather_rain": 0.06,
            "field_strength": 0.09, "pressure_performance": 0.08,
            "course_history": 0.11, "cut_probability": 0.10,
            "wave_advantage": 0.07, "fatigue_score": 0.05,
        }

        score = PRIORS.get(feature_name, 0.05)

        return {
            "feature": feature_name,
            "importance_score": score,
            "directional_accuracy": None,
            "n_samples": 0,
            "is_degraded": False,  # Cannot determine without data
            "note": "proxy_importance_from_domain_prior",
        }
