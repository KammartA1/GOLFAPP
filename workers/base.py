"""
Golf Quant Engine -- Base Worker
=================================
Abstract base class for all background workers.  Provides:
  - Retry logic with configurable max_retries / retry_delay
  - Automatic logging of start / end / error to the worker_status table
  - Support for one-shot and scheduled (looping) execution modes
  - Graceful shutdown via threading Event
"""
import abc
import json
import logging
import time
import traceback
from datetime import datetime, timedelta
from threading import Event as ThreadEvent

from database.connection import get_session_factory, init_db
from database.models import WorkerStatus

log = logging.getLogger(__name__)


class BaseWorker(abc.ABC):
    """All workers inherit from this class and implement ``execute()``."""

    name: str = "base_worker"
    interval_seconds: int = 900          # default 15 minutes
    max_retries: int = 3
    retry_delay: float = 30.0            # seconds between retries
    description: str = ""

    def __init__(self):
        self._shutdown_event = ThreadEvent()
        self._logger = logging.getLogger(f"workers.{self.name}")
        self._session_factory = None

    # ------------------------------------------------------------------
    # Abstract method every subclass MUST implement
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def execute(self) -> dict:
        """Run the worker's core logic.

        Returns a dict with at least ``{"items_processed": int}``.
        Raise on failure -- BaseWorker handles retries.
        """
        ...

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def run_once(self) -> bool:
        """Execute a single run with retry logic.  Returns True on success."""
        self._ensure_db()
        self._update_status("running")
        self._logger.info("Worker [%s] starting one-shot run", self.name)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start = datetime.utcnow()
                result = self.execute()
                elapsed = (datetime.utcnow() - start).total_seconds()
                items = result.get("items_processed", 0) if isinstance(result, dict) else 0

                self._logger.info(
                    "Worker [%s] completed in %.1fs -- %d items processed",
                    self.name, elapsed, items,
                )
                self._record_success(items, result)
                return True

            except Exception as exc:
                last_error = exc
                tb = traceback.format_exc()
                self._logger.error(
                    "Worker [%s] attempt %d/%d failed: %s\n%s",
                    self.name, attempt, self.max_retries, exc, tb,
                )
                if attempt < self.max_retries:
                    self._logger.info(
                        "Retrying in %.0fs ...", self.retry_delay,
                    )
                    time.sleep(self.retry_delay)

        # All retries exhausted
        self._record_failure(str(last_error))
        return False

    def run_loop(self) -> None:
        """Run in scheduled-loop mode until ``shutdown()`` is called."""
        self._ensure_db()
        self._logger.info(
            "Worker [%s] starting loop (interval=%ds)",
            self.name, self.interval_seconds,
        )

        while not self._shutdown_event.is_set():
            self.run_once()
            self._schedule_next_run()

            # Sleep in small increments so shutdown is responsive
            waited = 0.0
            while waited < self.interval_seconds and not self._shutdown_event.is_set():
                chunk = min(5.0, self.interval_seconds - waited)
                time.sleep(chunk)
                waited += chunk

        self._update_status("idle")
        self._logger.info("Worker [%s] loop stopped", self.name)

    def shutdown(self) -> None:
        """Signal the worker to stop its loop."""
        self._shutdown_event.set()
        self._logger.info("Worker [%s] shutdown requested", self.name)

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------
    def _ensure_db(self):
        """Make sure the DB is initialised and cache a session factory."""
        if self._session_factory is None:
            init_db()
            self._session_factory = get_session_factory()

    def _get_session(self):
        return self._session_factory()

    def _update_status(self, status: str, extra: dict | None = None):
        """Upsert the worker_status row."""
        session = self._get_session()
        try:
            row = session.query(WorkerStatus).filter_by(worker_name=self.name).first()
            if row is None:
                row = WorkerStatus(worker_name=self.name)
                session.add(row)
            row.status = status
            row.last_run = datetime.utcnow()
            if extra:
                row.metadata_json = json.dumps(extra)
            session.commit()
        except Exception:
            session.rollback()
            self._logger.exception("Failed to update worker_status for %s", self.name)
        finally:
            session.close()

    def _record_success(self, items_processed: int, result: dict | None = None):
        session = self._get_session()
        try:
            row = session.query(WorkerStatus).filter_by(worker_name=self.name).first()
            if row is None:
                row = WorkerStatus(worker_name=self.name)
                session.add(row)
            now = datetime.utcnow()
            row.status = "idle"
            row.last_run = now
            row.last_success = now
            row.last_error = None
            extra = row.metadata_json
            meta = json.loads(extra) if extra else {}
            meta["total_runs"] = meta.get("total_runs", 0) + 1
            meta["total_successes"] = meta.get("total_successes", 0) + 1
            meta["items_processed"] = items_processed
            meta["last_result"] = _safe_serialize(result)
            row.metadata_json = json.dumps(meta)
            session.commit()
        except Exception:
            session.rollback()
            self._logger.exception("Failed to record success for %s", self.name)
        finally:
            session.close()

    def _record_failure(self, error_msg: str):
        session = self._get_session()
        try:
            row = session.query(WorkerStatus).filter_by(worker_name=self.name).first()
            if row is None:
                row = WorkerStatus(worker_name=self.name)
                session.add(row)
            now = datetime.utcnow()
            row.status = "error"
            row.last_run = now
            row.last_error = error_msg[:2000]
            extra = row.metadata_json
            meta = json.loads(extra) if extra else {}
            meta["total_runs"] = meta.get("total_runs", 0) + 1
            meta["total_errors"] = meta.get("total_errors", 0) + 1
            meta["last_error_at"] = now.isoformat()
            row.metadata_json = json.dumps(meta)
            session.commit()
        except Exception:
            session.rollback()
            self._logger.exception("Failed to record failure for %s", self.name)
        finally:
            session.close()

    def _schedule_next_run(self):
        session = self._get_session()
        try:
            row = session.query(WorkerStatus).filter_by(worker_name=self.name).first()
            if row:
                row.next_scheduled_run = datetime.utcnow() + timedelta(seconds=self.interval_seconds)
                session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()


def _safe_serialize(obj) -> str | None:
    """Attempt JSON-safe serialization, truncate large payloads."""
    if obj is None:
        return None
    try:
        s = json.dumps(obj, default=str)
        return s[:4000]
    except Exception:
        return str(obj)[:4000]
