"""
Golf Quant Engine -- Worker Scheduler
=======================================
Sets up APScheduler with all workers, configurable intervals, graceful
shutdown, and a health-check endpoint.

Run as:
    python -m workers.scheduler
    python -m workers.scheduler --health     # print health and exit

Environment variable overrides:
    ODDS_WORKER_INTERVAL     (default 900)
    SIGNAL_WORKER_INTERVAL   (default 1800)
    CLOSING_WORKER_INTERVAL  (default 900)
    MODEL_WORKER_INTERVAL    (default 3600)
    REPORT_WORKER_INTERVAL   (default 86400)
    SCHEDULER_HEALTH_PORT    (default 8089)
"""
import json
import logging
import os
import signal
import sys
import threading
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from database.connection import get_session_factory, init_db
from database.models import WorkerStatus

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Health-check HTTP handler
# ------------------------------------------------------------------
class HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that serves worker health info."""

    scheduler_ref = None  # set by GolfScheduler before starting

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            body = self._build_health()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body, default=str, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _build_health(self) -> dict:
        try:
            factory = get_session_factory()
            session = factory()
            try:
                rows = session.query(WorkerStatus).all()
                workers = {}
                for r in rows:
                    meta = json.loads(r.metadata_json) if r.metadata_json else {}
                    workers[r.worker_name] = {
                        "status": r.status,
                        "last_run": r.last_run.isoformat() if r.last_run else None,
                        "last_success": r.last_success.isoformat() if r.last_success else None,
                        "last_error": r.last_error,
                        "next_scheduled": r.next_scheduled_run.isoformat() if r.next_scheduled_run else None,
                        "total_runs": meta.get("total_runs", 0),
                        "total_errors": meta.get("total_errors", 0),
                    }
                return {
                    "status": "ok",
                    "timestamp": datetime.utcnow().isoformat(),
                    "workers": workers,
                }
            finally:
                session.close()
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def log_message(self, format, *args):
        """Suppress default stderr logging."""
        pass


# ------------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------------
class GolfScheduler:
    """Manages all worker jobs via APScheduler."""

    def __init__(self):
        self._shutdown_event = threading.Event()
        self._workers = []
        self._scheduler = None
        self._health_server = None
        self._health_thread = None

    def setup(self):
        """Import workers, configure APScheduler, and register jobs."""
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        from apscheduler.triggers.cron import CronTrigger

        from workers.odds_worker import OddsWorker
        from workers.signal_worker import SignalWorker
        from workers.closing_worker import ClosingWorker
        from workers.model_worker import ModelWorker
        from workers.report_worker import ReportWorker
        from workers.stats_worker import StatsWorker

        init_db()

        self._scheduler = BackgroundScheduler(
            timezone="UTC",
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 300,
            },
        )

        odds = OddsWorker()
        signal_w = SignalWorker()
        closing = ClosingWorker()
        model = ModelWorker()
        report = ReportWorker()
        stats = StatsWorker()

        self._workers = [odds, signal_w, closing, model, report, stats]

        # Odds worker: every 15 minutes (golf)
        self._scheduler.add_job(
            odds.run_once,
            IntervalTrigger(seconds=odds.interval_seconds),
            id="odds_worker",
            name="Odds Worker",
            next_run_time=datetime.utcnow() + timedelta(seconds=5),
        )

        # Signal worker: every 30 minutes
        self._scheduler.add_job(
            signal_w.run_once,
            IntervalTrigger(seconds=signal_w.interval_seconds),
            id="signal_worker",
            name="Signal Worker",
            next_run_time=datetime.utcnow() + timedelta(seconds=60),
        )

        # Closing worker: every 15 minutes
        self._scheduler.add_job(
            closing.run_once,
            IntervalTrigger(seconds=closing.interval_seconds),
            id="closing_worker",
            name="Closing Worker",
            next_run_time=datetime.utcnow() + timedelta(seconds=120),
        )

        # Model worker: every hour (checks if retrain needed)
        self._scheduler.add_job(
            model.run_once,
            IntervalTrigger(seconds=model.interval_seconds),
            id="model_worker",
            name="Model Worker",
            next_run_time=datetime.utcnow() + timedelta(seconds=180),
        )

        # Report worker: daily at 03:00 UTC (11 PM ET)
        self._scheduler.add_job(
            report.run_once,
            CronTrigger(hour=3, minute=0, timezone="UTC"),
            id="report_worker",
            name="Report Worker",
        )

        # Stats worker: daily at 06:00 UTC (2 AM ET) — ingests player stats, SG, fields
        self._scheduler.add_job(
            stats.run_once,
            CronTrigger(hour=6, minute=0, timezone="UTC"),
            id="stats_worker",
            name="Stats Worker",
        )

        log.info("Scheduler configured with %d jobs", len(self._workers))

    def start(self):
        """Start the scheduler and health-check server."""
        if self._scheduler is None:
            self.setup()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Start health-check HTTP server
        health_port = int(os.environ.get("SCHEDULER_HEALTH_PORT", 8089))
        try:
            HealthHandler.scheduler_ref = self
            self._health_server = HTTPServer(("0.0.0.0", health_port), HealthHandler)
            self._health_thread = threading.Thread(
                target=self._health_server.serve_forever,
                daemon=True,
            )
            self._health_thread.start()
            log.info("Health-check endpoint running on http://0.0.0.0:%d/health", health_port)
        except OSError as exc:
            log.warning("Could not start health server on port %d: %s", health_port, exc)

        # Start APScheduler
        self._scheduler.start()
        log.info("Scheduler started -- workers are running")

        # Block main thread until shutdown
        try:
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            pass

        self.stop()

    def stop(self):
        """Gracefully shut down the scheduler and all workers."""
        log.info("Shutting down scheduler...")

        # Shut down APScheduler
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=True)

        # Shut down workers
        for w in self._workers:
            w.shutdown()

        # Shut down health server
        if self._health_server:
            self._health_server.shutdown()

        log.info("Scheduler stopped.")

    def _handle_signal(self, signum, frame):
        log.info("Received signal %d, initiating graceful shutdown...", signum)
        self._shutdown_event.set()

    def health(self) -> dict:
        """Return health status of all workers."""
        handler = HealthHandler.__new__(HealthHandler)
        return handler._build_health()


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    # Add file handler
    from config.settings import LOG_DIR
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_DIR / "scheduler.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s -- %(message)s"))
    logging.getLogger().addHandler(file_handler)

    import argparse
    parser = argparse.ArgumentParser(description="Golf Worker Scheduler")
    parser.add_argument("--health", action="store_true", help="Print health status and exit")
    args = parser.parse_args()

    if args.health:
        init_db()
        scheduler = GolfScheduler()
        health = scheduler.health()
        print(json.dumps(health, default=str, indent=2))
        return

    scheduler = GolfScheduler()
    scheduler.start()


if __name__ == "__main__":
    main()
