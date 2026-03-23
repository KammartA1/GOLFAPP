"""DataQualityReport — Composite data quality score from all audits."""

from __future__ import annotations

import logging
from datetime import datetime

from services.data_audit.timestamp_audit import TimestampAuditor
from services.data_audit.odds_audit import OddsAuditor
from services.data_audit.closing_line_audit import ClosingLineAuditor
from services.data_audit.completeness_audit import CompletenessAuditor

logger = logging.getLogger(__name__)


class DataQualityReport:
    """Generate composite data quality report from all audit modules."""

    def __init__(self, sport: str = "golf"):
        self.sport = sport
        self.timestamp_auditor = TimestampAuditor(sport)
        self.odds_auditor = OddsAuditor(sport)
        self.closing_auditor = ClosingLineAuditor(sport)
        self.completeness_auditor = CompletenessAuditor(sport)

    def generate(self) -> dict:
        """Generate full data quality report.

        Returns dict with:
          - composite_score: 0-100
          - per-audit results
          - issues and recommendations
        """
        logger.info("Generating data quality report for %s", self.sport)

        timestamp_result = self.timestamp_auditor.audit()
        odds_result = self.odds_auditor.audit()
        closing_result = self.closing_auditor.audit()
        completeness_result = self.completeness_auditor.audit()

        # Extract scores
        scores = {
            "timestamp_freshness": timestamp_result.get("overall_score", 50),
            "odds_quality": odds_result.get("score", 50),
            "closing_line_quality": closing_result.get("score", 50),
            "data_completeness": completeness_result.get("overall_score", 50),
        }

        # Weighted composite score
        weights = {
            "timestamp_freshness": 0.20,
            "odds_quality": 0.30,
            "closing_line_quality": 0.30,
            "data_completeness": 0.20,
        }
        composite = sum(scores[k] * weights[k] for k in scores)

        # Collect all issues
        all_issues = []
        for result in [timestamp_result, odds_result, closing_result]:
            if isinstance(result, dict):
                all_issues.extend(result.get("issues", []))

        # Recommendations
        recommendations = []
        if scores["timestamp_freshness"] < 70:
            recommendations.append("Data staleness detected — check scraper health")
        if scores["odds_quality"] < 70:
            recommendations.append("Odds data quality issues — review ingestion pipeline")
        if scores["closing_line_quality"] < 70:
            recommendations.append("Closing line capture needs improvement — CLV reliability affected")
        if scores["data_completeness"] < 70:
            recommendations.append("Data gaps detected — check data sources and backfill")

        # Overall status
        if composite >= 85:
            status = "healthy"
        elif composite >= 60:
            status = "degraded"
        else:
            status = "critical"

        report = {
            "report_date": datetime.utcnow().isoformat(),
            "sport": self.sport,
            "composite_score": round(composite, 1),
            "status": status,
            "component_scores": scores,
            "details": {
                "timestamp": timestamp_result,
                "odds": odds_result,
                "closing_lines": closing_result,
                "completeness": completeness_result,
            },
            "issues": all_issues,
            "recommendations": recommendations,
        }

        logger.info("Data quality report: score=%.1f, status=%s, issues=%d",
                     composite, status, len(all_issues))
        return report
