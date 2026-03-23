"""
CLI entry point for the workers package.

Usage:
    python -m workers              # start all workers via scheduler
    python -m workers --health     # print worker health status
    python -m workers --run odds   # run a single worker once
    python -m workers --run signal
    python -m workers --run closing
    python -m workers --run model
    python -m workers --run report
    python -m workers --list       # list all registered workers
"""
import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        prog="python -m workers",
        description="Golf Quant Engine -- Worker Management CLI",
    )
    parser.add_argument(
        "--health", action="store_true",
        help="Print worker health status and exit",
    )
    parser.add_argument(
        "--run", type=str, metavar="WORKER",
        help="Run a single worker once (odds, signal, closing, model, report)",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Run the specified worker in loop mode (requires --run)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all registered workers",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    )

    if args.list:
        _list_workers()
        return

    if args.health:
        _show_health()
        return

    if args.run:
        _run_single(args.run, loop=args.loop)
        return

    # Default: start scheduler with all workers
    from workers import run_all
    run_all()


def _list_workers():
    from workers import OddsWorker, SignalWorker, ClosingWorker, ModelWorker, ReportWorker

    workers = [OddsWorker, SignalWorker, ClosingWorker, ModelWorker, ReportWorker]
    print("Registered workers:")
    print("-" * 70)
    for cls in workers:
        w = cls()
        print(f"  {w.name:<20s} interval={w.interval_seconds:>6d}s  retries={w.max_retries}")
        if w.description:
            print(f"  {'':20s} {w.description}")
    print("-" * 70)


def _show_health():
    from database.connection import init_db
    from workers.scheduler import GolfScheduler

    init_db()
    scheduler = GolfScheduler()
    health = scheduler.health()
    print(json.dumps(health, default=str, indent=2))


def _run_single(worker_name: str, loop: bool = False):
    worker_map = {}
    from workers.odds_worker import OddsWorker
    from workers.signal_worker import SignalWorker
    from workers.closing_worker import ClosingWorker
    from workers.model_worker import ModelWorker
    from workers.report_worker import ReportWorker

    worker_map = {
        "odds": OddsWorker,
        "odds_worker": OddsWorker,
        "signal": SignalWorker,
        "signal_worker": SignalWorker,
        "closing": ClosingWorker,
        "closing_worker": ClosingWorker,
        "model": ModelWorker,
        "model_worker": ModelWorker,
        "report": ReportWorker,
        "report_worker": ReportWorker,
    }

    cls = worker_map.get(worker_name.lower())
    if cls is None:
        print(f"Unknown worker: {worker_name}")
        print(f"Available: {', '.join(sorted(set(worker_map.values()), key=lambda c: c.name))}")
        sys.exit(1)

    worker = cls()
    if loop:
        worker.run_loop()
    else:
        success = worker.run_once()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
