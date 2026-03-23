"""
workers/data_audit_worker.py
=============================
Background worker that runs the data quality audit system for golf.

Runs automatically after each data ingestion cycle and on a scheduled
interval. Generates a full DataQualityReport and stores the results.

Run standalone:
    python -m workers.data_audit_worker          # one-shot
    python -m workers.data_audit_worker --loop   # scheduled loop
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from workers.base import BaseWorker

log = logging.getLogger(__name__)


class DataAuditWorker(BaseWorker):
    """Runs the full data quality audit on a scheduled interval.

    Default interval: 30 minutes (runs after odds ingestion cycles).
    """

    name = "data_audit_worker"
    interval_seconds = int(os.environ.get("AUDIT_INTERVAL", "1800"))
    max_retries = 2
    retry_delay = 15.0
    description = "Runs data quality audit across all dimensions"

    def execute(self) -> dict:
        from services.data_audit.report import DataQualityReport

        self._logger.info("Starting data quality audit")

        report = DataQualityReport(sport="GOLF")
        result = report.generate_dict()

        composite_score = result.get("composite_score", 0)
        is_trustworthy = result.get("is_trustworthy", False)
        critical_count = len(result.get("critical_findings", []))

        self._logger.info(
            "Data quality audit complete: score=%.1f, trustworthy=%s, "
            "critical_findings=%d",
            composite_score, is_trustworthy, critical_count,
        )

        if not is_trustworthy:
            self._logger.warning(
                "DATA QUALITY BELOW THRESHOLD (%.1f < 80). "
                "Edge may be fabricated!", composite_score,
            )

        if critical_count > 0:
            self._logger.warning("CRITICAL FINDINGS (%d):", critical_count)
            for finding in result.get("critical_findings", []):
                self._logger.warning("  !! %s", finding)

        return {
            "items_processed": 1,
            "composite_score": composite_score,
            "is_trustworthy": is_trustworthy,
            "critical_findings": critical_count,
        }


def run_post_ingestion_audit() -> Dict[str, Any]:
    """Lightweight audit hook to run after each data ingestion cycle.

    Called from the odds worker after storing new lines.
    Catches and logs errors so it never blocks ingestion.
    """
    try:
        from services.data_audit.report import DataQualityReport

        report = DataQualityReport(sport="GOLF")
        result = report.generate_dict()

        score = result.get("composite_score", 0)
        log.info("Post-ingestion audit: score=%.1f", score)

        if not result.get("is_trustworthy", True):
            log.warning(
                "POST-INGESTION AUDIT WARNING: score=%.1f — "
                "data quality below threshold", score
            )

        return {
            "ok": True,
            "composite_score": score,
            "critical_findings": len(result.get("critical_findings", [])),
        }

    except Exception as exc:
        log.warning("Post-ingestion audit failed (non-blocking): %s", exc)
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    from database.connection import init_db
    init_db()

    worker = DataAuditWorker()
    if "--loop" in sys.argv:
        import signal as _sig
        import threading

        stop = threading.Event()

        def _handler(signum, frame):
            stop.set()

        _sig.signal(_sig.SIGINT, _handler)
        _sig.signal(_sig.SIGTERM, _handler)
        worker.run_forever(stop_event=stop)
    else:
        result = worker.run_once()
        if not result.get("items_processed"):
            sys.exit(1)
