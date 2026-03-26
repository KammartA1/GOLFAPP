"""Final verdict — Pulls from ALL subsystems to produce unified GO/NO-GO decision.

This is the top-level decision engine. Before any bet is placed, the full
system state is evaluated here. The verdict considers:

1. Edge monitor (CLV, trend, daily metrics)
2. Kill switch (6 independent halt conditions)
3. Capital optimizer (Kelly, risk, portfolio)
4. Data quality audit
5. Execution reality (slippage, limits, latency)
6. Market reaction (survival, edge decay)
7. Model governance (version health, feature drift)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from database.connection import DatabaseManager
from database.models import (
    Bet, CLVLog, EdgeReport, ModelVersion, CalibrationLog,
    SystemStateLog, AuditLog, FeatureLog,
)
from quant_system.core.types import SystemState

logger = logging.getLogger(__name__)


class FinalVerdict:
    """Unified GO/NO-GO decision engine.

    Pulls from every subsystem to produce a single actionable verdict
    with full transparency into the reasoning.
    """

    # Weights for each subsystem's vote (sum to 1.0)
    SUBSYSTEM_WEIGHTS = {
        "edge_monitor": 0.30,
        "kill_switch": 0.25,
        "execution_quality": 0.15,
        "data_quality": 0.10,
        "model_health": 0.10,
        "market_survival": 0.10,
    }

    def __init__(self, sport: str = "golf"):
        self.sport = sport

    def generate(self, bankroll: float = 0.0) -> dict:
        """Generate the final system verdict.

        Returns:
            {
                'verdict': str ('GO', 'CAUTION', 'STOP'),
                'system_state': str (SystemState value),
                'confidence': float (0-1),
                'overall_score': float (0-100),
                'subsystem_scores': dict,
                'blocking_issues': list of str,
                'warnings': list of str,
                'recommendations': list of str,
                'bet_sizing_multiplier': float (0.0-1.0),
                'detailed_report': dict,
                'timestamp': str,
            }
        """
        timestamp = datetime.utcnow()

        # Gather all subsystem assessments
        edge_assessment = self._assess_edge()
        kill_assessment = self._assess_kill_switch()
        execution_assessment = self._assess_execution_quality()
        data_assessment = self._assess_data_quality()
        model_assessment = self._assess_model_health()
        market_assessment = self._assess_market_survival()

        subsystem_scores = {
            "edge_monitor": edge_assessment,
            "kill_switch": kill_assessment,
            "execution_quality": execution_assessment,
            "data_quality": data_assessment,
            "model_health": model_assessment,
            "market_survival": market_assessment,
        }

        # Weighted overall score
        overall_score = sum(
            subsystem_scores[name]["score"] * weight
            for name, weight in self.SUBSYSTEM_WEIGHTS.items()
        )

        # Collect blocking issues and warnings
        blocking_issues: List[str] = []
        warnings: List[str] = []
        recommendations: List[str] = []

        for name, assessment in subsystem_scores.items():
            for issue in assessment.get("blocking", []):
                blocking_issues.append(f"[{name}] {issue}")
            for warn in assessment.get("warnings", []):
                warnings.append(f"[{name}] {warn}")
            for rec in assessment.get("recommendations", []):
                recommendations.append(f"[{name}] {rec}")

        # Kill switch is absolute — any fatal condition blocks everything
        if kill_assessment.get("any_fatal", False):
            blocking_issues.insert(0, "[KILL SWITCH] Fatal condition triggered")

        # Determine verdict
        verdict, system_state, confidence, sizing_multiplier = self._determine_verdict(
            overall_score, blocking_issues, warnings, subsystem_scores
        )

        # Build detailed report
        detailed_report = {
            "subsystems": {
                name: {
                    "score": round(a["score"], 1),
                    "status": a.get("status", "unknown"),
                    "weight": self.SUBSYSTEM_WEIGHTS[name],
                    "weighted_contribution": round(
                        a["score"] * self.SUBSYSTEM_WEIGHTS[name], 2
                    ),
                    "details": a.get("details", {}),
                }
                for name, a in subsystem_scores.items()
            },
            "score_breakdown": {
                "overall": round(overall_score, 1),
                "min_subsystem": min(a["score"] for a in subsystem_scores.values()),
                "max_subsystem": max(a["score"] for a in subsystem_scores.values()),
            },
        }

        # Log the verdict
        self._log_verdict(verdict, system_state, overall_score, blocking_issues)

        return {
            "verdict": verdict,
            "system_state": system_state,
            "confidence": round(confidence, 3),
            "overall_score": round(overall_score, 1),
            "subsystem_scores": {
                name: round(a["score"], 1) for name, a in subsystem_scores.items()
            },
            "blocking_issues": blocking_issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "bet_sizing_multiplier": round(sizing_multiplier, 3),
            "detailed_report": detailed_report,
            "timestamp": timestamp.isoformat(),
        }

    def quick_check(self) -> bool:
        """Fast GO/NO-GO without full report. Use for pre-bet gating."""
        # Check kill switch first (fastest path to NO)
        kill = self._assess_kill_switch()
        if kill.get("any_fatal", False):
            return False

        # Check edge monitor
        edge = self._assess_edge()
        if edge["score"] < 30:
            return False

        return True

    def format_text_report(self, verdict_result: dict) -> str:
        """Format verdict as human-readable text report."""
        v = verdict_result
        lines = [
            "=" * 70,
            f"  FINAL SYSTEM VERDICT: {v['verdict']}",
            f"  System State: {v['system_state']}",
            f"  Overall Score: {v['overall_score']}/100",
            f"  Confidence: {v['confidence']:.1%}",
            f"  Bet Sizing: {v['bet_sizing_multiplier']:.0%} of normal",
            "=" * 70,
            "",
            "SUBSYSTEM SCORES:",
        ]

        for name, score in v["subsystem_scores"].items():
            bar = "#" * int(score / 5) + "." * (20 - int(score / 5))
            lines.append(f"  {name:<25} [{bar}] {score:.0f}/100")

        if v["blocking_issues"]:
            lines.append("")
            lines.append("BLOCKING ISSUES:")
            for issue in v["blocking_issues"]:
                lines.append(f"  [X] {issue}")

        if v["warnings"]:
            lines.append("")
            lines.append("WARNINGS:")
            for warn in v["warnings"]:
                lines.append(f"  [!] {warn}")

        if v["recommendations"]:
            lines.append("")
            lines.append("RECOMMENDATIONS:")
            for rec in v["recommendations"]:
                lines.append(f"  -> {rec}")

        lines.append("")
        lines.append(f"Generated: {v['timestamp']}")
        lines.append("=" * 70)

        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════════
    #  SUBSYSTEM ASSESSMENTS
    # ══════════════════════════════════════════════════════════════════════

    def _assess_edge(self) -> dict:
        """Assess edge from CLV data and edge reports."""
        with DatabaseManager.session_scope() as session:
            # Get latest edge report
            report = (
                session.query(EdgeReport)
                .filter(EdgeReport.sport == self.sport)
                .order_by(EdgeReport.report_date.desc())
                .first()
            )

            # Get recent CLV
            clv_records = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport)
                .order_by(CLVLog.calculated_at.desc())
                .limit(100)
                .all()
            )

            blocking = []
            warnings = []
            recommendations = []
            details = {}

            if not clv_records or len(clv_records) < 10:
                return {
                    "score": 50.0,
                    "status": "insufficient_data",
                    "blocking": [],
                    "warnings": ["Insufficient CLV data"],
                    "recommendations": ["Wait for more settled bets"],
                    "details": {},
                }

            clv_values = np.array([r.clv_cents for r in clv_records])
            avg_clv = float(np.mean(clv_values))
            beat_rate = float(np.mean([1.0 if r.beat_close else 0.0 for r in clv_records]))

            details["avg_clv_100"] = round(avg_clv, 2)
            details["beat_rate"] = round(beat_rate, 4)
            details["n_records"] = len(clv_records)

            # Score calculation
            # CLV component (0-50 points)
            if avg_clv >= 2.0:
                clv_score = 50.0
            elif avg_clv >= 1.0:
                clv_score = 35.0 + (avg_clv - 1.0) * 15.0
            elif avg_clv >= 0:
                clv_score = 20.0 + avg_clv * 15.0
            elif avg_clv >= -1.0:
                clv_score = max(0.0, 20.0 + avg_clv * 20.0)
            else:
                clv_score = 0.0

            # Beat rate component (0-30 points)
            if beat_rate >= 0.55:
                br_score = 30.0
            elif beat_rate >= 0.50:
                br_score = 15.0 + (beat_rate - 0.50) * 300.0
            elif beat_rate >= 0.45:
                br_score = (beat_rate - 0.45) * 300.0
            else:
                br_score = 0.0

            # Trend component (0-20 points)
            trend_score = 10.0  # Neutral default
            if report:
                if report.clv_trend == "improving":
                    trend_score = 20.0
                elif report.clv_trend == "stable":
                    trend_score = 12.0
                elif report.clv_trend == "declining":
                    trend_score = 3.0
                    warnings.append(f"CLV trend is declining")

            score = clv_score + br_score + trend_score
            details["clv_score"] = round(clv_score, 1)
            details["beat_rate_score"] = round(br_score, 1)
            details["trend_score"] = round(trend_score, 1)

            if avg_clv < -0.5:
                blocking.append(f"Negative CLV: {avg_clv:.2f}c over last {len(clv_records)} bets")
            if beat_rate < 0.47:
                blocking.append(f"Low beat rate: {beat_rate:.1%}")

            if avg_clv > 0 and avg_clv < 0.5:
                warnings.append("CLV is marginally positive — monitor closely")
            if beat_rate < 0.50:
                warnings.append(f"Beat rate below 50%: {beat_rate:.1%}")

            if avg_clv > 1.5:
                recommendations.append("Edge looks strong — maintain current approach")
            elif 0 < avg_clv < 0.5:
                recommendations.append("Consider reducing bet sizes until CLV improves")

            status = "healthy" if score >= 60 else "marginal" if score >= 35 else "unhealthy"

            return {
                "score": min(100.0, score),
                "status": status,
                "blocking": blocking,
                "warnings": warnings,
                "recommendations": recommendations,
                "details": details,
            }

    def _assess_kill_switch(self) -> dict:
        """Assess kill switch conditions."""
        from services.kill_switch import KillSwitch

        ks = KillSwitch(self.sport)
        try:
            result = ks.check_all()
        except Exception as e:
            logger.error("Kill switch check failed: %s", e)
            return {
                "score": 50.0,
                "status": "error",
                "blocking": [f"Kill switch check failed: {e}"],
                "warnings": [],
                "recommendations": ["Fix kill switch errors"],
                "details": {},
                "any_fatal": False,
            }

        n_fatal = result.get("n_fatal", 0)
        n_critical = result.get("n_critical", 0)
        n_warnings = result.get("n_warnings", 0)

        # Score: start at 100, deduct for each condition
        score = 100.0
        score -= n_fatal * 40.0
        score -= n_critical * 20.0
        score -= n_warnings * 5.0
        score = max(0.0, score)

        blocking = []
        warnings = []
        if n_fatal > 0:
            for c in result["conditions"]:
                if c["triggered"] and c["severity"] == "fatal":
                    blocking.append(c["message"])
        if n_critical > 0:
            for c in result["conditions"]:
                if c["triggered"] and c["severity"] == "critical":
                    warnings.append(c["message"])

        status = "healthy" if n_fatal == 0 and n_critical == 0 else "critical" if n_fatal > 0 else "warning"

        return {
            "score": score,
            "status": status,
            "blocking": blocking,
            "warnings": warnings,
            "recommendations": [f"Investigate: {r}" for r in result.get("halt_reasons", [])],
            "details": {
                "n_fatal": n_fatal,
                "n_critical": n_critical,
                "n_warnings": n_warnings,
                "conditions": result.get("conditions", []),
            },
            "any_fatal": n_fatal > 0,
        }

    def _assess_execution_quality(self) -> dict:
        """Assess execution quality from bet data."""
        with DatabaseManager.session_scope() as session:
            bets = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.status.in_(["won", "lost"]),
                )
                .order_by(Bet.settled_at.desc())
                .limit(100)
                .all()
            )

            if len(bets) < 20:
                return {
                    "score": 60.0,
                    "status": "insufficient_data",
                    "blocking": [],
                    "warnings": ["Insufficient bet data for execution assessment"],
                    "recommendations": [],
                    "details": {},
                }

            # Check if we have CLV data to compare theoretical vs actual
            clv_records = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport)
                .order_by(CLVLog.calculated_at.desc())
                .limit(100)
                .all()
            )

            total_staked = sum(b.stake for b in bets)
            total_pnl = sum(b.pnl for b in bets)
            realized_roi = total_pnl / max(total_staked, 1)

            details = {
                "n_bets": len(bets),
                "total_staked": round(total_staked, 2),
                "total_pnl": round(total_pnl, 2),
                "realized_roi": round(realized_roi, 4),
            }

            blocking = []
            warnings = []
            recommendations = []

            # Calculate execution efficiency
            if clv_records:
                avg_clv = float(np.mean([r.clv_cents for r in clv_records]))
                theoretical_roi = avg_clv / 100.0
                if theoretical_roi > 0:
                    efficiency = realized_roi / max(theoretical_roi, 0.001)
                    details["execution_efficiency"] = round(efficiency, 4)
                    details["theoretical_roi"] = round(theoretical_roi, 4)

                    if efficiency < 0.3:
                        blocking.append(
                            f"Execution efficiency {efficiency:.0%} — most edge lost to friction"
                        )
                    elif efficiency < 0.5:
                        warnings.append(
                            f"Execution efficiency {efficiency:.0%} — significant friction"
                        )
                        recommendations.append("Review sportsbook limits and slippage")

            # Score: based on ROI and efficiency
            if realized_roi > 0.03:
                score = 90.0
            elif realized_roi > 0.01:
                score = 70.0
            elif realized_roi > 0:
                score = 55.0
            elif realized_roi > -0.02:
                score = 35.0
            else:
                score = 15.0

            status = "healthy" if score >= 60 else "marginal" if score >= 35 else "unhealthy"

            return {
                "score": score,
                "status": status,
                "blocking": blocking,
                "warnings": warnings,
                "recommendations": recommendations,
                "details": details,
            }

    def _assess_data_quality(self) -> dict:
        """Assess data freshness and completeness."""
        with DatabaseManager.session_scope() as session:
            now = datetime.utcnow()
            blocking = []
            warnings = []
            recommendations = []
            details = {}
            score = 100.0

            # Check line data freshness
            from database.models import LineMovement
            latest_line = (
                session.query(LineMovement)
                .filter(LineMovement.sport == self.sport)
                .order_by(LineMovement.captured_at.desc())
                .first()
            )
            if latest_line:
                age = (now - latest_line.captured_at).total_seconds() / 3600
                details["line_age_hours"] = round(age, 1)
                if age > 24:
                    score -= 30
                    blocking.append(f"Line data is {age:.0f}h old")
                elif age > 6:
                    score -= 10
                    warnings.append(f"Line data is {age:.1f}h old")
            else:
                score -= 20
                warnings.append("No line movement data found")

            # Check CLV capture coverage
            recent_bets = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.status.in_(["won", "lost"]),
                    Bet.settled_at >= now - timedelta(days=30),
                )
                .count()
            )
            recent_clv = (
                session.query(CLVLog)
                .filter(
                    CLVLog.sport == self.sport,
                    CLVLog.calculated_at >= now - timedelta(days=30),
                )
                .count()
            )
            if recent_bets > 0:
                clv_coverage = recent_clv / recent_bets
                details["clv_coverage_30d"] = round(clv_coverage, 2)
                if clv_coverage < 0.5:
                    score -= 20
                    warnings.append(f"CLV coverage only {clv_coverage:.0%} in last 30d")
                    recommendations.append("Improve closing line capture process")
                elif clv_coverage < 0.8:
                    score -= 5
                    warnings.append(f"CLV coverage {clv_coverage:.0%} in last 30d")

            # Check scraper health
            from database.models import ScraperStatus
            unhealthy_scrapers = (
                session.query(ScraperStatus)
                .filter(
                    ScraperStatus.sport == self.sport,
                    ScraperStatus.is_healthy == False,
                )
                .all()
            )
            if unhealthy_scrapers:
                names = [s.scraper_name for s in unhealthy_scrapers]
                score -= len(unhealthy_scrapers) * 10
                warnings.append(f"Unhealthy scrapers: {', '.join(names)}")
                recommendations.append("Fix failing scrapers")

            score = max(0.0, min(100.0, score))
            status = "healthy" if score >= 70 else "marginal" if score >= 40 else "unhealthy"

            return {
                "score": score,
                "status": status,
                "blocking": blocking,
                "warnings": warnings,
                "recommendations": recommendations,
                "details": details,
            }

    def _assess_model_health(self) -> dict:
        """Assess model version health and drift."""
        with DatabaseManager.session_scope() as session:
            model = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.sport == self.sport,
                    ModelVersion.is_active == True,
                )
                .order_by(ModelVersion.trained_at.desc())
                .first()
            )

            blocking = []
            warnings = []
            recommendations = []
            details = {}

            if not model:
                return {
                    "score": 40.0,
                    "status": "no_model",
                    "blocking": ["No active model version found"],
                    "warnings": [],
                    "recommendations": ["Register active model version"],
                    "details": {},
                }

            score = 80.0  # Start healthy

            details["version"] = model.version
            details["trained_at"] = model.trained_at.isoformat() if model.trained_at else None
            details["n_live_bets"] = model.n_live_bets

            # Model age
            if model.trained_at:
                age_days = (datetime.utcnow() - model.trained_at).days
                details["age_days"] = age_days
                if age_days > 90:
                    score -= 20
                    warnings.append(f"Model is {age_days} days old — consider retraining")
                elif age_days > 60:
                    score -= 10
                    warnings.append(f"Model is {age_days} days old")

            # Degradation flag
            if model.is_degraded:
                score -= 30
                blocking.append("Model flagged as degraded")
                recommendations.append("Retrain model immediately")

            # PSI drift
            if model.psi_score is not None:
                details["psi_score"] = round(model.psi_score, 4)
                if model.psi_score > 0.25:
                    score -= 25
                    blocking.append(f"High feature drift: PSI={model.psi_score:.3f}")
                elif model.psi_score > 0.10:
                    score -= 10
                    warnings.append(f"Moderate feature drift: PSI={model.psi_score:.3f}")

            # Live performance
            if model.live_clv_avg is not None:
                details["live_clv_avg"] = round(model.live_clv_avg, 2)
                if model.live_clv_avg < 0:
                    score -= 20
                    warnings.append(f"Live CLV negative: {model.live_clv_avg:.2f}c")

            if model.live_roi is not None:
                details["live_roi"] = round(model.live_roi, 4)
                if model.live_roi < -0.05:
                    score -= 15
                    warnings.append(f"Live ROI deeply negative: {model.live_roi:.1%}")

            # Feature importance drift
            recent_features = (
                session.query(FeatureLog)
                .filter(
                    FeatureLog.sport == self.sport,
                    FeatureLog.is_degraded == True,
                )
                .order_by(FeatureLog.report_date.desc())
                .limit(20)
                .all()
            )
            if recent_features:
                degraded_names = list(set(f.feature_name for f in recent_features))
                details["degraded_features"] = degraded_names[:5]
                if len(degraded_names) > 3:
                    score -= 15
                    warnings.append(
                        f"{len(degraded_names)} features showing drift: "
                        f"{', '.join(degraded_names[:3])}..."
                    )

            score = max(0.0, min(100.0, score))
            status = "healthy" if score >= 60 else "marginal" if score >= 35 else "unhealthy"

            return {
                "score": score,
                "status": status,
                "blocking": blocking,
                "warnings": warnings,
                "recommendations": recommendations,
                "details": details,
            }

    def _assess_market_survival(self) -> dict:
        """Assess long-term market survival outlook."""
        with DatabaseManager.session_scope() as session:
            # Get bet history for survival assessment
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

            blocking = []
            warnings = []
            recommendations = []
            details = {}

            if len(bets) < 30:
                return {
                    "score": 60.0,
                    "status": "insufficient_data",
                    "blocking": [],
                    "warnings": ["Insufficient data for survival assessment"],
                    "recommendations": [],
                    "details": {},
                }

            # Book diversity
            books_used = set(b.book for b in bets if b.book)
            details["n_books"] = len(books_used)

            if len(books_used) <= 1:
                warnings.append("Only using 1 sportsbook — high concentration risk")
                recommendations.append("Diversify across multiple books for sustainability")

            # Check stake trajectory — are limits tightening?
            stakes = [b.stake for b in bets if b.stake > 0]
            if len(stakes) >= 20:
                # Bets ordered DESC (newest first): [:half] = recent, [half:] = older
                recent_avg = float(np.mean(stakes[:len(stakes)//2]))
                older_avg = float(np.mean(stakes[len(stakes)//2:]))
                stake_ratio = recent_avg / max(older_avg, 1)
                details["stake_ratio_recent_vs_old"] = round(stake_ratio, 3)

                if stake_ratio < 0.5:
                    warnings.append(
                        f"Stakes declining: recent avg ${recent_avg:.0f} "
                        f"vs prior ${older_avg:.0f}"
                    )
                    recommendations.append("Review sportsbook limits — may be getting limited")

            # Monthly PnL trend
            if bets:
                monthly_pnl: Dict[str, float] = {}
                for b in bets:
                    if b.settled_at:
                        key = b.settled_at.strftime("%Y-%m")
                        monthly_pnl.setdefault(key, 0.0)
                        monthly_pnl[key] += b.pnl

                if len(monthly_pnl) >= 3:
                    months_sorted = sorted(monthly_pnl.keys())
                    pnl_values = [monthly_pnl[m] for m in months_sorted]
                    negative_months = sum(1 for p in pnl_values if p < 0)
                    details["n_months"] = len(pnl_values)
                    details["negative_months"] = negative_months

                    if negative_months / len(pnl_values) > 0.5:
                        warnings.append(
                            f"{negative_months}/{len(pnl_values)} months negative"
                        )

            # Score based on diversification and sustainability
            score = 70.0
            score += min(15.0, len(books_used) * 5.0)  # Book diversity bonus
            if len(warnings) == 0:
                score += 15.0  # No warnings bonus

            score -= len(blocking) * 20.0
            score -= len(warnings) * 7.0
            score = max(0.0, min(100.0, score))

            status = "healthy" if score >= 60 else "marginal" if score >= 35 else "unhealthy"

            return {
                "score": score,
                "status": status,
                "blocking": blocking,
                "warnings": warnings,
                "recommendations": recommendations,
                "details": details,
            }

    # ══════════════════════════════════════════════════════════════════════
    #  VERDICT LOGIC
    # ══════════════════════════════════════════════════════════════════════

    def _determine_verdict(
        self,
        overall_score: float,
        blocking_issues: List[str],
        warnings: List[str],
        subsystem_scores: dict,
    ) -> tuple:
        """Determine final verdict from aggregated assessments.

        Returns:
            (verdict, system_state, confidence, sizing_multiplier)
        """
        # Any blocking issue → STOP
        if blocking_issues:
            # Check severity: kill switch fatal = KILLED, else SUSPENDED
            has_fatal = subsystem_scores["kill_switch"].get("any_fatal", False)
            state = SystemState.KILLED.value if has_fatal else SystemState.SUSPENDED.value
            confidence = min(0.95, 0.5 + len(blocking_issues) * 0.1)
            return "STOP", state, confidence, 0.0

        # Low overall score → CAUTION
        if overall_score < 50:
            return "CAUTION", SystemState.REDUCED.value, 0.6, 0.25

        # Moderate score with warnings → CAUTION
        if overall_score < 65 or len(warnings) >= 4:
            # Sizing multiplier proportional to score
            multiplier = max(0.25, (overall_score - 30) / 70)
            return "CAUTION", SystemState.REDUCED.value, 0.5, round(multiplier, 2)

        # Edge monitor specifically weak → CAUTION even if overall OK
        edge_score = subsystem_scores["edge_monitor"]["score"]
        if edge_score < 40:
            return "CAUTION", SystemState.REDUCED.value, 0.55, 0.50

        # Good overall with some warnings → GO with reduced sizing
        if warnings and overall_score < 80:
            multiplier = max(0.5, 1.0 - len(warnings) * 0.1)
            return "GO", SystemState.ACTIVE.value, 0.7, round(multiplier, 2)

        # Strong overall → GO full speed
        confidence = min(0.95, overall_score / 100)
        return "GO", SystemState.ACTIVE.value, confidence, 1.0

    def _log_verdict(
        self,
        verdict: str,
        system_state: str,
        score: float,
        blocking_issues: List[str],
    ) -> None:
        """Log verdict to audit log."""
        try:
            with DatabaseManager.session_scope() as session:
                import json
                log = AuditLog(
                    sport=self.sport,
                    category="verdict",
                    level="warning" if verdict != "GO" else "info",
                    message=f"Final verdict: {verdict} (score={score:.1f}, state={system_state})",
                    details=json.dumps({
                        "verdict": verdict,
                        "system_state": system_state,
                        "score": round(score, 1),
                        "blocking_issues": blocking_issues,
                    }),
                )
                session.add(log)
        except Exception as e:
            logger.error("Failed to log verdict: %s", e)
