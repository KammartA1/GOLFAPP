"""
Report Service — Report reading layer
=======================================
Surfaces edge reports, calibration data, SG accuracy reports,
and system state from the database.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func, and_

from database.models import (
    EdgeReport, CalibrationSnapshot, SystemState,
    Bet, Signal, SGStat, Player,
)
from services._db import get_session as _session

log = logging.getLogger(__name__)


def get_latest_report(report_type: str = "daily") -> dict:
    """
    Return the most recent edge report of the given type.

    report_type: 'daily', 'weekly', 'calibration', 'performance', etc.

    Returns dict with report metadata and parsed report_json payload,
    or empty dict if none found.
    """
    with _session() as session:
        row = (
            session.query(EdgeReport)
            .filter(
                EdgeReport.sport == "GOLF",
                EdgeReport.report_type == report_type,
            )
            .order_by(EdgeReport.generated_at.desc())
            .first()
        )

        if not row:
            return {}

        result = row.to_dict()
        # Parse the JSON payload into a usable dict
        if row.report_json:
            try:
                result["data"] = json.loads(row.report_json)
            except (json.JSONDecodeError, TypeError):
                result["data"] = {}
        else:
            result["data"] = {}

        return result


def get_report_history(
    report_type: str,
    days: int = 30,
) -> list[dict]:
    """
    Return edge reports of a given type within the last N days,
    ordered newest first.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    with _session() as session:
        rows = (
            session.query(EdgeReport)
            .filter(
                EdgeReport.sport == "GOLF",
                EdgeReport.report_type == report_type,
                EdgeReport.generated_at >= cutoff,
            )
            .order_by(EdgeReport.generated_at.desc())
            .all()
        )

        results = []
        for r in rows:
            d = r.to_dict()
            if r.report_json:
                try:
                    d["data"] = json.loads(r.report_json)
                except (json.JSONDecodeError, TypeError):
                    d["data"] = {}
            else:
                d["data"] = {}
            results.append(d)

        return results


def get_calibration_data(sport: str = "GOLF") -> list[dict]:
    """
    Return the latest calibration snapshot for each probability bucket.

    Calibration snapshots compare predicted probability buckets against
    actual hit rates. A well-calibrated model has predicted_avg close
    to actual_rate in every bucket.
    """
    with _session() as session:
        # Get the most recent snapshot date
        latest_date = (
            session.query(func.max(CalibrationSnapshot.snapshot_date))
            .filter(CalibrationSnapshot.sport == sport)
            .scalar()
        )

        if not latest_date:
            return []

        rows = (
            session.query(CalibrationSnapshot)
            .filter(
                CalibrationSnapshot.sport == sport,
                CalibrationSnapshot.snapshot_date == latest_date,
            )
            .order_by(CalibrationSnapshot.prob_lower)
            .all()
        )

        return [r.to_dict() for r in rows]


def get_system_state(sport: str = "GOLF") -> dict:
    """
    Return the current system state (ACTIVE / REDUCED / SUSPENDED / KILLED)
    along with historical context.

    Returns dict with current state plus last 5 state changes.
    """
    with _session() as session:
        current = (
            session.query(SystemState)
            .filter(SystemState.sport == sport)
            .order_by(SystemState.changed_at.desc())
            .first()
        )

        history = (
            session.query(SystemState)
            .filter(SystemState.sport == sport)
            .order_by(SystemState.changed_at.desc())
            .limit(5)
            .all()
        )

        if not current:
            return {
                "state": "UNKNOWN",
                "reason": "No system state recorded",
                "changed_at": None,
                "history": [],
            }

        return {
            "state": current.state,
            "reason": current.reason,
            "changed_at": current.changed_at.isoformat() if current.changed_at else None,
            "clv_at_change": current.clv_at_change,
            "bankroll_at_change": current.bankroll_at_change,
            "drawdown_at_change": current.drawdown_at_change,
            "history": [h.to_dict() for h in history],
        }


def get_sg_accuracy_report() -> dict:
    """
    Compute how accurate our SG-based projections have been
    by comparing model predictions (from signals) against actual
    bet outcomes.

    Returns dict with:
      - overall_accuracy: fraction of correct directional calls
      - avg_absolute_error: mean |projection - actual|
      - by_market: per-market breakdown
      - by_confidence: per-confidence-tier breakdown
      - sample_size: number of evaluated bets
    """
    with _session() as session:
        # Get settled bets that have both projection and actual outcome
        bets = (
            session.query(Bet)
            .filter(
                Bet.sport == "GOLF",
                Bet.status.in_(["won", "lost"]),
                Bet.model_projection.isnot(None),
                Bet.actual_outcome.isnot(None),
            )
            .order_by(Bet.settled_at.desc())
            .limit(500)
            .all()
        )

    if not bets:
        return {
            "overall_accuracy": 0,
            "avg_absolute_error": 0,
            "by_market": {},
            "by_confidence": {},
            "sample_size": 0,
        }

    # Compute accuracy metrics
    correct = 0
    errors = []
    by_market: dict[str, dict] = {}
    by_confidence: dict[str, dict] = {}

    for b in bets:
        proj = b.model_projection or 0
        actual = b.actual_outcome or 0
        line = b.bet_line or 0
        direction = (b.direction or "").upper()
        market = b.market or "unknown"

        # Directional accuracy: did we correctly predict over/under?
        if direction == "OVER":
            predicted_correct = proj > line
            was_correct = actual > line
        elif direction == "UNDER":
            predicted_correct = proj < line
            was_correct = actual < line
        else:
            predicted_correct = True
            was_correct = b.status == "won"

        hit = predicted_correct == was_correct or b.status == "won"
        if hit:
            correct += 1

        abs_error = abs(proj - actual)
        errors.append(abs_error)

        # Aggregate by market
        if market not in by_market:
            by_market[market] = {"correct": 0, "total": 0, "errors": []}
        by_market[market]["total"] += 1
        if hit:
            by_market[market]["correct"] += 1
        by_market[market]["errors"].append(abs_error)

        # Aggregate by confidence score bucket
        conf = b.confidence_score or 0
        if conf >= 0.8:
            bucket = "high"
        elif conf >= 0.5:
            bucket = "medium"
        else:
            bucket = "low"

        if bucket not in by_confidence:
            by_confidence[bucket] = {"correct": 0, "total": 0, "errors": []}
        by_confidence[bucket]["total"] += 1
        if hit:
            by_confidence[bucket]["correct"] += 1
        by_confidence[bucket]["errors"].append(abs_error)

    n = len(bets)
    overall_accuracy = correct / n if n else 0
    avg_error = sum(errors) / n if n else 0

    # Summarize sub-groups
    market_summary = {}
    for mkt, data in by_market.items():
        t = data["total"]
        market_summary[mkt] = {
            "accuracy": round(data["correct"] / t, 4) if t else 0,
            "avg_error": round(sum(data["errors"]) / t, 3) if t else 0,
            "sample_size": t,
        }

    conf_summary = {}
    for bucket, data in by_confidence.items():
        t = data["total"]
        conf_summary[bucket] = {
            "accuracy": round(data["correct"] / t, 4) if t else 0,
            "avg_error": round(sum(data["errors"]) / t, 3) if t else 0,
            "sample_size": t,
        }

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "avg_absolute_error": round(avg_error, 3),
        "by_market": market_summary,
        "by_confidence": conf_summary,
        "sample_size": n,
    }
