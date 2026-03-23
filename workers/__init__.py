"""
Golf Quant Engine -- Workers Layer
====================================
Background workers that run independently of Streamlit.

Workers:
  - OddsWorker:    Fetches odds from PrizePicks, Odds API, DraftKings/FanDuel
  - SignalWorker:  Generates betting signals from new line movements + SG model
  - ClosingWorker: Captures closing lines and calculates CLV
  - ModelWorker:   Retrains the SG projection model on schedule or trigger
  - ReportWorker:  Generates daily/weekly edge reports

Quick start:
    # Run all workers with APScheduler
    from workers import run_all
    run_all()

    # Or run a single worker
    from workers import OddsWorker
    worker = OddsWorker()
    worker.run_once()

CLI:
    python -m workers                   # start scheduler with all workers
    python -m workers.odds_worker       # run odds worker once
    python -m workers.scheduler         # start scheduler
    python -m workers.scheduler --health  # check health
"""
from workers.base import BaseWorker
from workers.odds_worker import OddsWorker
from workers.signal_worker import SignalWorker
from workers.closing_worker import ClosingWorker
from workers.model_worker import ModelWorker
from workers.report_worker import ReportWorker
from workers.scheduler import GolfScheduler

__all__ = [
    "BaseWorker",
    "OddsWorker",
    "SignalWorker",
    "ClosingWorker",
    "ModelWorker",
    "ReportWorker",
    "GolfScheduler",
    "run_all",
]


def run_all():
    """Start all workers via the APScheduler-based scheduler.

    This is the main entry point for running the full worker suite.
    Blocks until shutdown signal is received (SIGTERM / SIGINT).
    """
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    )
    scheduler = GolfScheduler()
    scheduler.start()
