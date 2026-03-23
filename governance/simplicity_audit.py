"""Simplicity audit — Detect model and system over-complexity.

Complexity is the enemy of a robust quant system. This auditor checks
for signs that the system is becoming too complex to maintain or that
complexity is not justified by performance improvement.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np

from database.connection import DatabaseManager
from database.models import ModelVersion, Bet, FeatureLog, AuditLog

logger = logging.getLogger(__name__)


class SimplicityAuditor:
    """Audit system complexity and recommend simplifications.

    Key Principle: Every additional feature/model/rule must justify its
    existence with measurable improvement. If it doesn't help, remove it.
    """

    # Complexity thresholds
    MAX_FEATURES_RECOMMENDED = 25        # More features than this → audit flag
    MAX_FEATURES_HARD_LIMIT = 50         # Hard limit — definitely over-complex
    MIN_BETS_PER_FEATURE = 10            # Need at least 10 bets per feature for justification
    FEATURE_IMPORTANCE_FLOOR = 0.02      # Features below this score add noise
    CORRELATION_THRESHOLD = 0.80         # Features correlated above this are redundant

    def __init__(self, sport: str = "golf"):
        self.sport = sport

    def full_audit(self) -> dict:
        """Run complete simplicity audit.

        Returns:
            {
                'complexity_score': float (0-100, lower is simpler),
                'verdict': str ('simple', 'acceptable', 'complex', 'over_complex'),
                'feature_audit': dict,
                'model_audit': dict,
                'redundancy_audit': dict,
                'recommendations': list,
                'timestamp': str,
            }
        """
        feature_audit = self._audit_features()
        model_audit = self._audit_model_complexity()
        redundancy_audit = self._audit_redundancy()

        # Compute overall complexity score
        scores = []
        if feature_audit.get("n_features", 0) > 0:
            # Feature count score (0-40)
            n = feature_audit["n_features"]
            if n <= 15:
                feat_score = 0
            elif n <= self.MAX_FEATURES_RECOMMENDED:
                feat_score = (n - 15) * 2
            elif n <= self.MAX_FEATURES_HARD_LIMIT:
                feat_score = 20 + (n - self.MAX_FEATURES_RECOMMENDED)
            else:
                feat_score = 40
            scores.append(feat_score)

        # Unused features score (0-30)
        n_unused = feature_audit.get("n_low_importance", 0)
        scores.append(min(30, n_unused * 5))

        # Redundancy score (0-30)
        n_redundant = redundancy_audit.get("n_redundant_pairs", 0)
        scores.append(min(30, n_redundant * 10))

        complexity_score = sum(scores)

        if complexity_score <= 15:
            verdict = "simple"
        elif complexity_score <= 35:
            verdict = "acceptable"
        elif complexity_score <= 60:
            verdict = "complex"
        else:
            verdict = "over_complex"

        # Generate recommendations
        recommendations = self._generate_recommendations(
            feature_audit, model_audit, redundancy_audit
        )

        result = {
            "complexity_score": round(complexity_score, 1),
            "verdict": verdict,
            "feature_audit": feature_audit,
            "model_audit": model_audit,
            "redundancy_audit": redundancy_audit,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Log if concerning
        if verdict in ("complex", "over_complex"):
            logger.warning("Simplicity audit: %s (score=%d)", verdict, complexity_score)
            self._log_audit(result)

        return result

    def _audit_features(self) -> dict:
        """Audit feature set size and value."""
        with DatabaseManager.session_scope() as session:
            # Get active model's feature list
            model = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.sport == self.sport,
                    ModelVersion.is_active == True,
                )
                .first()
            )

            if not model or not model.feature_list:
                return {"n_features": 0, "status": "no_model"}

            features = json.loads(model.feature_list) if model.feature_list else []

            # Get latest importance scores
            importance_data = {}
            for fname in features:
                latest = (
                    session.query(FeatureLog)
                    .filter(
                        FeatureLog.sport == self.sport,
                        FeatureLog.feature_name == fname,
                    )
                    .order_by(FeatureLog.report_date.desc())
                    .first()
                )
                if latest:
                    importance_data[fname] = {
                        "importance": round(latest.importance_score, 4),
                        "is_degraded": latest.is_degraded,
                        "directional_accuracy": round(latest.directional_accuracy, 4) if latest.directional_accuracy else None,
                    }

            # Identify low-importance features
            low_importance = [
                name for name, data in importance_data.items()
                if data["importance"] < self.FEATURE_IMPORTANCE_FLOOR
            ]

            # Identify degraded features
            degraded = [
                name for name, data in importance_data.items()
                if data.get("is_degraded", False)
            ]

            # Sample size check
            n_bets = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.status.in_(["won", "lost"]),
                )
                .count()
            )
            bets_per_feature = n_bets / max(len(features), 1)

            return {
                "n_features": len(features),
                "features": features,
                "n_with_importance": len(importance_data),
                "n_low_importance": len(low_importance),
                "low_importance_features": low_importance,
                "n_degraded": len(degraded),
                "degraded_features": degraded,
                "bets_per_feature": round(bets_per_feature, 1),
                "sufficient_data": bets_per_feature >= self.MIN_BETS_PER_FEATURE,
                "importance_data": importance_data,
            }

    def _audit_model_complexity(self) -> dict:
        """Audit model hyperparameter complexity."""
        with DatabaseManager.session_scope() as session:
            model = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.sport == self.sport,
                    ModelVersion.is_active == True,
                )
                .first()
            )

            if not model or not model.hyperparameters:
                return {"status": "no_model"}

            hp = json.loads(model.hyperparameters)

            # Count complexity indicators
            n_hyperparams = len(hp)
            has_ensemble = hp.get("n_models", 1) > 1 or hp.get("ensemble", False)
            n_layers = hp.get("n_layers", hp.get("max_depth", 1))

            # Training sample ratio
            n_params_estimate = n_hyperparams * 10  # Rough estimate
            n_samples = model.n_training_samples or 0
            param_to_sample_ratio = n_params_estimate / max(n_samples, 1)

            return {
                "n_hyperparams": n_hyperparams,
                "has_ensemble": has_ensemble,
                "n_layers": n_layers,
                "n_training_samples": n_samples,
                "param_to_sample_ratio": round(param_to_sample_ratio, 4),
                "overfitting_risk": "high" if param_to_sample_ratio > 0.1 else "low",
            }

    def _audit_redundancy(self) -> dict:
        """Detect redundant features by checking pairwise correlations."""
        with DatabaseManager.session_scope() as session:
            bets = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.status.in_(["won", "lost"]),
                )
                .order_by(Bet.settled_at.desc())
                .limit(200)
                .all()
            )

            if len(bets) < 30:
                return {"n_redundant_pairs": 0, "status": "insufficient_data"}

            # Extract feature vectors from snapshots
            feature_vectors: Dict[str, List[float]] = {}
            n_valid = 0

            for bet in bets:
                if not bet.features_snapshot:
                    continue
                try:
                    features = json.loads(bet.features_snapshot)
                except (json.JSONDecodeError, TypeError):
                    continue

                for fname, fval in features.items():
                    try:
                        fval = float(fval)
                    except (ValueError, TypeError):
                        continue
                    feature_vectors.setdefault(fname, []).append(fval)
                n_valid += 1

            if n_valid < 20:
                return {"n_redundant_pairs": 0, "status": "insufficient_feature_data"}

            # Find features with enough data
            min_samples = 20
            valid_features = {
                k: np.array(v) for k, v in feature_vectors.items()
                if len(v) >= min_samples
            }

            if len(valid_features) < 2:
                return {"n_redundant_pairs": 0, "status": "insufficient_features"}

            # Compute pairwise correlations
            feature_names = sorted(valid_features.keys())
            redundant_pairs = []

            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    f1 = feature_names[i]
                    f2 = feature_names[j]

                    # Align to same length
                    min_len = min(len(valid_features[f1]), len(valid_features[f2]))
                    v1 = valid_features[f1][:min_len]
                    v2 = valid_features[f2][:min_len]

                    if np.std(v1) < 1e-10 or np.std(v2) < 1e-10:
                        continue

                    corr = float(np.abs(np.corrcoef(v1, v2)[0, 1]))
                    if np.isnan(corr):
                        continue

                    if corr >= self.CORRELATION_THRESHOLD:
                        redundant_pairs.append({
                            "feature_1": f1,
                            "feature_2": f2,
                            "correlation": round(corr, 4),
                            "recommendation": f"Consider removing one of ({f1}, {f2})",
                        })

            return {
                "n_redundant_pairs": len(redundant_pairs),
                "redundant_pairs": redundant_pairs,
                "n_features_analyzed": len(feature_names),
                "threshold": self.CORRELATION_THRESHOLD,
            }

    def _generate_recommendations(
        self,
        feature_audit: dict,
        model_audit: dict,
        redundancy_audit: dict,
    ) -> List[str]:
        """Generate actionable recommendations from audit results."""
        recs = []

        n_features = feature_audit.get("n_features", 0)
        if n_features > self.MAX_FEATURES_HARD_LIMIT:
            recs.append(
                f"CRITICAL: {n_features} features exceeds hard limit of "
                f"{self.MAX_FEATURES_HARD_LIMIT}. Aggressively prune low-value features."
            )
        elif n_features > self.MAX_FEATURES_RECOMMENDED:
            recs.append(
                f"Feature count ({n_features}) exceeds recommended limit of "
                f"{self.MAX_FEATURES_RECOMMENDED}. Consider removing lowest-importance features."
            )

        low_imp = feature_audit.get("low_importance_features", [])
        if low_imp:
            recs.append(
                f"Remove {len(low_imp)} low-importance features: {', '.join(low_imp[:5])}"
                + ("..." if len(low_imp) > 5 else "")
            )

        degraded = feature_audit.get("degraded_features", [])
        if degraded:
            recs.append(
                f"Investigate {len(degraded)} degraded features: {', '.join(degraded[:5])}"
            )

        if not feature_audit.get("sufficient_data", True):
            bpf = feature_audit.get("bets_per_feature", 0)
            recs.append(
                f"Only {bpf:.0f} bets per feature (need {self.MIN_BETS_PER_FEATURE}). "
                f"Reduce features or wait for more data."
            )

        if model_audit.get("overfitting_risk") == "high":
            recs.append("High overfitting risk — reduce model complexity or gather more training data")

        for pair in redundancy_audit.get("redundant_pairs", [])[:3]:
            recs.append(pair["recommendation"])

        if not recs:
            recs.append("System complexity is within acceptable bounds. No action needed.")

        return recs

    def _log_audit(self, result: dict) -> None:
        """Log audit to AuditLog table."""
        try:
            with DatabaseManager.session_scope() as session:
                session.add(AuditLog(
                    sport=self.sport,
                    category="governance",
                    level="warning",
                    message=(
                        f"Simplicity audit: {result['verdict']} "
                        f"(score={result['complexity_score']})"
                    ),
                    details=json.dumps({
                        "complexity_score": result["complexity_score"],
                        "verdict": result["verdict"],
                        "recommendations": result["recommendations"],
                    }),
                ))
        except Exception as e:
            logger.error("Failed to log simplicity audit: %s", e)
