"""Base worker class — shared infrastructure for all background workers.

Handles:
  - Worker state persistence (last_run, status, errors)
  - Scraper status tracking
  - Audit logging
  - Error handling with exponential backoff
  - Graceful shutdown
"""

from __future__ import annotations

import logging
import signal
import time
import traceback
from datetime import datetime, timedelta
from threading import Event as ThreadEvent

from database.connection import DatabaseManager
from database.models import WorkerState, ScraperStatus, AuditLog

logger = logging.getLogger(__name__)


class BaseWorker:
    """Base class for all background workers."""

    WORKER_NAME: str = "base"
    SPORT: str = "golf"
    DEFAULT_INTERVAL_SECONDS: int = 900  # 15 minutes

    def __init__(self, interval_seconds: int | None = None):
        self.interval = interval_seconds or self.DEFAULT_INTERVAL_SECONDS
        self._stop_event = ThreadEvent()
        self._run_count = 0
        self._total_duration = 0.0

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        logger.info("[%s] Shutdown signal received, stopping gracefully...", self.WORKER_NAME)
        self._stop_event.set()

    # ── Override this in subclasses ────────────────────────────────────

    def execute(self) -> dict:
        """Execute one cycle of the worker. Return a result dict.

        Must be overridden by subclasses.

        Returns:
            {"lines_processed": 42, "errors": 0, ...}
        """
        raise NotImplementedError

    # ── Run loop ──────────────────────────────────────────────────────

    def run_forever(self):
        """Main run loop. Executes worker on interval until stopped."""
        logger.info("[%s] Starting worker loop (interval=%ds)", self.WORKER_NAME, self.interval)
        self._update_worker_state("running")

        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("[%s] Unhandled error in run loop", self.WORKER_NAME)
                self._update_worker_state("error", error=traceback.format_exc())
                # Backoff on repeated failures
                backoff = min(self.interval * 2, 3600)
                logger.info("[%s] Backing off for %ds", self.WORKER_NAME, backoff)
                self._stop_event.wait(backoff)
                continue

            # Wait for next cycle
            self._stop_event.wait(self.interval)

        self._update_worker_state("idle")
        logger.info("[%s] Worker stopped", self.WORKER_NAME)

    def run_once(self) -> dict:
        """Execute a single cycle with timing and state tracking."""
        logger.info("[%s] Starting cycle #%d", self.WORKER_NAME, self._run_count + 1)
        self._update_worker_state("running")

        start = time.monotonic()
        try:
            result = self.execute()
            duration = time.monotonic() - start

            self._run_count += 1
            self._total_duration += duration
            avg_duration = self._total_duration / self._run_count

            self._update_worker_state(
                "idle",
                run_count=self._run_count,
                avg_duration=avg_duration,
            )

            self._log_audit(
                "info",
                f"Cycle #{self._run_count} completed in {duration:.1f}s",
                result,
            )

            logger.info("[%s] Cycle #%d completed in %.1fs: %s",
                        self.WORKER_NAME, self._run_count, duration, result)
            return result

        except Exception as e:
            duration = time.monotonic() - start
            error_msg = traceback.format_exc()

            self._update_worker_state("error", error=str(e))
            self._update_scraper_status(success=False, error=str(e))
            self._log_audit("error", f"Cycle failed after {duration:.1f}s: {e}", {"traceback": error_msg})

            raise

    # ── State persistence ─────────────────────────────────────────────

    def _update_worker_state(
        self,
        status: str,
        run_count: int | None = None,
        avg_duration: float | None = None,
        error: str | None = None,
    ):
        """Update worker state in database."""
        try:
            with DatabaseManager.session_scope() as session:
                state = session.query(WorkerState).filter_by(
                    worker_name=self.WORKER_NAME
                ).first()

                if state is None:
                    state = WorkerState(
                        worker_name=self.WORKER_NAME,
                        sport=self.SPORT,
                    )
                    session.add(state)

                state.status = status
                state.updated_at = datetime.utcnow()

                if status == "idle" and run_count:
                    state.last_run = datetime.utcnow()
                    state.next_scheduled = datetime.utcnow() + timedelta(seconds=self.interval)
                    state.run_count = run_count

                if avg_duration is not None:
                    state.avg_duration_seconds = avg_duration

                if error:
                    state.last_error = error

        except Exception:
            logger.exception("[%s] Failed to update worker state", self.WORKER_NAME)

    def _update_scraper_status(
        self,
        success: bool = True,
        lines_count: int = 0,
        error: str | None = None,
        scraper_name: str | None = None,
    ):
        """Update scraper status tracking."""
        name = scraper_name or self.WORKER_NAME
        try:
            with DatabaseManager.session_scope() as session:
                status = session.query(ScraperStatus).filter_by(
                    scraper_name=name
                ).first()

                if status is None:
                    status = ScraperStatus(
                        scraper_name=name,
                        sport=self.SPORT,
                    )
                    session.add(status)

                status.total_runs += 1

                if success:
                    status.last_success = datetime.utcnow()
                    status.consecutive_failures = 0
                    status.lines_last_scrape = lines_count
                    status.is_healthy = True
                else:
                    status.last_failure = datetime.utcnow()
                    status.last_error = error
                    status.consecutive_failures += 1
                    status.total_failures += 1
                    status.is_healthy = status.consecutive_failures < 5

                status.updated_at = datetime.utcnow()

        except Exception:
            logger.exception("[%s] Failed to update scraper status", self.WORKER_NAME)

    def _log_audit(self, level: str, message: str, details: dict | None = None):
        """Write to audit log."""
        try:
            import json
            with DatabaseManager.session_scope() as session:
                log = AuditLog(
                    timestamp=datetime.utcnow(),
                    sport=self.SPORT,
                    category="worker",
                    level=level,
                    message=f"[{self.WORKER_NAME}] {message}",
                    details=json.dumps(details) if details else None,
                )
                session.add(log)
        except Exception:
            logger.exception("[%s] Failed to write audit log", self.WORKER_NAME)
