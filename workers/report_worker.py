"""
Golf Quant Engine -- Report Worker
====================================
Generates comprehensive daily and weekly edge reports.

Report contents:
  - Tournament P&L summary
  - CLV analysis (rolling 50 / 100 / 250 / 500)
  - Calibration report (bucket accuracy)
  - Course fit accuracy (predicted vs actual)
  - SG prediction accuracy (projected vs actual SG)
  - Weather impact analysis
  - Book efficiency comparison
  - System state recommendation

Runs daily at 11 PM ET (03:00 UTC) after tournament rounds complete.

Run standalone:
    python -m workers.report_worker
"""
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workers.base import BaseWorker
from database.connection import get_session_factory, init_db
from database.models import (
    EdgeReport, Bet, LineMovement, Signal, Event,
    CalibrationSnapshot, SystemState, SGStat, ModelVersion,
)

log = logging.getLogger(__name__)


class ReportWorker(BaseWorker):
    name = "report_worker"
    interval_seconds = int(os.environ.get("REPORT_WORKER_INTERVAL", 86400))  # 24 hours
    max_retries = 2
    retry_delay = 30.0
    description = "Generates daily/weekly edge reports"

    def execute(self) -> dict:
        init_db()
        factory = get_session_factory()
        session = factory()

        reports_created = 0

        try:
            now = datetime.utcnow()

            # 1. Daily report
            daily_report = self._generate_daily_report(session, now)
            self._save_report(session, "daily", daily_report)
            reports_created += 1

            # 2. Weekly report (Mondays)
            if now.weekday() == 0:  # Monday
                weekly_report = self._generate_weekly_report(session, now)
                self._save_report(session, "weekly", weekly_report)
                reports_created += 1

            # 3. Tournament report if a tournament just completed
            tournament_report = self._generate_tournament_report(session, now)
            if tournament_report:
                self._save_report(session, "tournament", tournament_report)
                reports_created += 1

            session.commit()

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return {"items_processed": reports_created}

    # ------------------------------------------------------------------ #
    # Daily report
    # ------------------------------------------------------------------ #
    def _generate_daily_report(self, session, now: datetime) -> dict:
        """Generate comprehensive daily report."""
        yesterday = now - timedelta(days=1)

        report = {
            "report_date": now.isoformat(),
            "period": "daily",
        }

        # P&L summary
        report["pnl"] = self._compute_pnl_summary(session, yesterday, now)

        # CLV analysis
        report["clv"] = self._compute_clv_analysis(session)

        # Calibration
        report["calibration"] = self._compute_calibration(session)

        # Signal quality
        report["signals"] = self._compute_signal_quality(session, yesterday, now)

        # Book efficiency
        report["books"] = self._compute_book_efficiency(session)

        # System recommendation
        report["recommendation"] = self._compute_recommendation(report)

        # Summary text
        report["summary"] = self._format_summary(report)

        return report

    # ------------------------------------------------------------------ #
    # Weekly report
    # ------------------------------------------------------------------ #
    def _generate_weekly_report(self, session, now: datetime) -> dict:
        """Generate comprehensive weekly report."""
        week_ago = now - timedelta(days=7)

        report = {
            "report_date": now.isoformat(),
            "period": "weekly",
        }

        report["pnl"] = self._compute_pnl_summary(session, week_ago, now)
        report["clv"] = self._compute_clv_analysis(session)
        report["calibration"] = self._compute_calibration(session)
        report["signals"] = self._compute_signal_quality(session, week_ago, now)
        report["books"] = self._compute_book_efficiency(session)
        report["sg_accuracy"] = self._compute_sg_accuracy(session, week_ago, now)
        report["weather_impact"] = self._compute_weather_impact(session, week_ago, now)
        report["course_fit"] = self._compute_course_fit_accuracy(session)
        report["recommendation"] = self._compute_recommendation(report)
        report["summary"] = self._format_summary(report)

        return report

    # ------------------------------------------------------------------ #
    # Tournament report
    # ------------------------------------------------------------------ #
    def _generate_tournament_report(self, session, now: datetime) -> dict | None:
        """Generate report for a recently completed tournament."""
        recent_completed = (
            session.query(Event)
            .filter(Event.sport == "GOLF")
            .filter(Event.status == "completed")
            .order_by(Event.start_time.desc())
            .first()
        )

        if recent_completed is None:
            return None

        # Only generate if completed in the last 24 hours
        if recent_completed.start_time:
            # Tournaments last ~4 days, so end time is start + 4 days
            estimated_end = recent_completed.start_time + timedelta(days=4)
            if (now - estimated_end).total_seconds() > 86400:
                return None

        # Check if we already generated a report for this event
        existing = (
            session.query(EdgeReport)
            .filter(EdgeReport.sport == "GOLF")
            .filter(EdgeReport.report_type == "tournament")
            .filter(EdgeReport.report_json.contains(recent_completed.event_name))
            .first()
        )
        if existing:
            return None

        event_name = recent_completed.event_name
        self._logger.info("Generating tournament report for [%s]", event_name)

        # Bets for this tournament
        bets = (
            session.query(Bet)
            .filter(Bet.sport == "GOLF")
            .filter(Bet.event == event_name)
            .all()
        )

        total_stake = sum(b.stake or 0 for b in bets)
        total_pnl = sum(b.pnl or 0 for b in bets if b.status in ("won", "lost"))
        won = sum(1 for b in bets if b.status == "won")
        lost = sum(1 for b in bets if b.status == "lost")
        pending = sum(1 for b in bets if b.status == "pending")

        # CLV for tournament bets
        clv_values = [b.closing_line - (b.bet_line or 0) for b in bets
                       if b.closing_line is not None and b.bet_line is not None]
        avg_clv = float(np.mean(clv_values)) if clv_values else 0.0
        beat_close = sum(1 for v in clv_values if v > 0)
        beat_close_pct = beat_close / max(len(clv_values), 1)

        report = {
            "report_date": now.isoformat(),
            "period": "tournament",
            "event_name": event_name,
            "course_name": recent_completed.course_name or "",
            "total_bets": len(bets),
            "won": won,
            "lost": lost,
            "pending": pending,
            "total_stake": round(total_stake, 2),
            "total_pnl": round(total_pnl, 2),
            "roi_pct": round(total_pnl / max(total_stake, 1) * 100, 2),
            "win_rate": round(won / max(won + lost, 1), 3),
            "avg_clv": round(avg_clv, 3),
            "beat_close_pct": round(beat_close_pct, 3),
        }

        return report

    # ------------------------------------------------------------------ #
    # Component: P&L
    # ------------------------------------------------------------------ #
    def _compute_pnl_summary(self, session, start: datetime, end: datetime) -> dict:
        """Compute P&L summary for a given period."""
        bets = (
            session.query(Bet)
            .filter(Bet.sport == "GOLF")
            .filter(Bet.timestamp >= start)
            .filter(Bet.timestamp <= end)
            .all()
        )

        settled = [b for b in bets if b.status in ("won", "lost")]
        total_stake = sum(b.stake or 0 for b in settled)
        total_pnl = sum(b.pnl or 0 for b in settled)
        won = sum(1 for b in settled if b.status == "won")
        lost = sum(1 for b in settled if b.status == "lost")

        return {
            "total_bets": len(bets),
            "settled": len(settled),
            "won": won,
            "lost": lost,
            "pending": sum(1 for b in bets if b.status == "pending"),
            "total_stake": round(total_stake, 2),
            "total_pnl": round(total_pnl, 2),
            "roi_pct": round(total_pnl / max(total_stake, 1) * 100, 2),
            "win_rate": round(won / max(won + lost, 1), 3),
            "avg_edge": round(
                float(np.mean([b.edge_pct for b in settled if b.edge_pct is not None])) if settled else 0, 4
            ),
        }

    # ------------------------------------------------------------------ #
    # Component: CLV
    # ------------------------------------------------------------------ #
    def _compute_clv_analysis(self, session) -> dict:
        """Compute rolling CLV metrics across multiple windows."""
        result = {}

        for window in [50, 100, 250, 500]:
            bets = (
                session.query(Bet)
                .filter(Bet.sport == "GOLF")
                .filter(Bet.status.in_(["won", "lost"]))
                .filter(Bet.closing_line.isnot(None))
                .order_by(Bet.settled_at.desc())
                .limit(window)
                .all()
            )

            if len(bets) < 10:
                result[f"clv_{window}"] = {
                    "n_bets": len(bets),
                    "avg_clv": 0.0,
                    "beat_close_pct": 0.0,
                }
                continue

            clv_values = []
            for b in bets:
                bet_line = b.bet_line or b.signal_line or 0
                closing = b.closing_line or 0
                direction = (b.direction or "").upper()

                if direction == "OVER":
                    clv = closing - bet_line
                elif direction == "UNDER":
                    clv = bet_line - closing
                else:
                    clv = bet_line - closing  # outright

                clv_values.append(clv)

            avg_clv = float(np.mean(clv_values)) if clv_values else 0.0
            beat_count = sum(1 for v in clv_values if v > 0)
            beat_pct = beat_count / max(len(clv_values), 1)

            result[f"clv_{window}"] = {
                "n_bets": len(bets),
                "avg_clv": round(avg_clv, 4),
                "beat_close_pct": round(beat_pct, 3),
                "positive": avg_clv > 0,
            }

        return result

    # ------------------------------------------------------------------ #
    # Component: Calibration
    # ------------------------------------------------------------------ #
    def _compute_calibration(self, session) -> dict:
        """Compute calibration metrics from settled bets."""
        bets = (
            session.query(Bet)
            .filter(Bet.sport == "GOLF")
            .filter(Bet.status.in_(["won", "lost"]))
            .filter(Bet.predicted_prob.isnot(None))
            .order_by(Bet.settled_at.desc())
            .limit(500)
            .all()
        )

        if len(bets) < 20:
            return {"n_bets": len(bets), "mae": 0.0, "brier": 0.0, "overconfident_pct": 0.0}

        preds = [b.predicted_prob for b in bets]
        outcomes = [1.0 if b.status == "won" else 0.0 for b in bets]

        # Brier score
        brier = float(np.mean([(p - o) ** 2 for p, o in zip(preds, outcomes)]))

        # Bucket calibration
        buckets = [
            (0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
            (0.70, 0.75), (0.75, 0.80), (0.80, 0.90), (0.90, 1.00),
        ]
        errors = []
        overconf = 0
        bucket_details = []

        for lo, hi in buckets:
            indices = [i for i, p in enumerate(preds) if lo <= p < hi]
            if len(indices) < 3:
                continue
            bucket_preds = [preds[i] for i in indices]
            bucket_outs = [outcomes[i] for i in indices]
            pred_avg = float(np.mean(bucket_preds))
            actual_rate = float(np.mean(bucket_outs))
            err = abs(pred_avg - actual_rate)
            errors.append(err)
            is_overconf = pred_avg > actual_rate
            if is_overconf:
                overconf += 1

            bucket_details.append({
                "range": f"{int(lo*100)}-{int(hi*100)}%",
                "predicted": round(pred_avg, 3),
                "actual": round(actual_rate, 3),
                "error": round(err, 3),
                "n": len(indices),
                "overconfident": is_overconf,
            })

        mae = float(np.mean(errors)) if errors else 0.0
        overconf_pct = overconf / max(len(bucket_details), 1)

        # Save calibration snapshot
        for bd in bucket_details:
            snap = CalibrationSnapshot(
                sport="GOLF",
                bucket_label=bd["range"],
                prob_lower=float(bd["range"].split("-")[0].replace("%", "")) / 100,
                prob_upper=float(bd["range"].split("-")[1].replace("%", "")) / 100,
                predicted_avg=bd["predicted"],
                actual_rate=bd["actual"],
                n_bets=bd["n"],
                calibration_error=bd["error"],
            )
            session.add(snap)

        return {
            "n_bets": len(bets),
            "brier_score": round(brier, 4),
            "mae": round(mae, 4),
            "overconfident_pct": round(overconf_pct, 3),
            "buckets": bucket_details,
        }

    # ------------------------------------------------------------------ #
    # Component: Signal quality
    # ------------------------------------------------------------------ #
    def _compute_signal_quality(self, session, start: datetime, end: datetime) -> dict:
        """Evaluate quality of signals generated in the period."""
        signals = (
            session.query(Signal)
            .filter(Signal.sport == "GOLF")
            .filter(Signal.generated_at >= start)
            .filter(Signal.generated_at <= end)
            .all()
        )

        if not signals:
            return {"total": 0}

        edges = [s.edge_pct for s in signals if s.edge_pct is not None]
        high_conf = sum(1 for s in signals if (s.confidence or 0) > 0.7)
        med_conf = sum(1 for s in signals if 0.4 <= (s.confidence or 0) <= 0.7)
        low_conf = sum(1 for s in signals if (s.confidence or 0) < 0.4)

        return {
            "total": len(signals),
            "avg_edge": round(float(np.mean(edges)), 4) if edges else 0.0,
            "max_edge": round(float(max(edges)), 4) if edges else 0.0,
            "high_confidence": high_conf,
            "medium_confidence": med_conf,
            "low_confidence": low_conf,
            "over_signals": sum(1 for s in signals if (s.direction or "") == "over"),
            "under_signals": sum(1 for s in signals if (s.direction or "") == "under"),
        }

    # ------------------------------------------------------------------ #
    # Component: Book efficiency
    # ------------------------------------------------------------------ #
    def _compute_book_efficiency(self, session) -> dict:
        """Compare performance across different sportsbooks."""
        bets = (
            session.query(Bet)
            .filter(Bet.sport == "GOLF")
            .filter(Bet.status.in_(["won", "lost"]))
            .order_by(Bet.settled_at.desc())
            .limit(500)
            .all()
        )

        by_book: dict[str, dict] = {}
        for b in bets:
            book = b.market or "unknown"
            if book not in by_book:
                by_book[book] = {"won": 0, "lost": 0, "pnl": 0.0, "stake": 0.0}
            by_book[book]["pnl"] += b.pnl or 0
            by_book[book]["stake"] += b.stake or 0
            if b.status == "won":
                by_book[book]["won"] += 1
            else:
                by_book[book]["lost"] += 1

        result = {}
        for book, data in by_book.items():
            total = data["won"] + data["lost"]
            result[book] = {
                "total_bets": total,
                "win_rate": round(data["won"] / max(total, 1), 3),
                "pnl": round(data["pnl"], 2),
                "roi_pct": round(data["pnl"] / max(data["stake"], 1) * 100, 2),
            }

        return result

    # ------------------------------------------------------------------ #
    # Component: SG prediction accuracy
    # ------------------------------------------------------------------ #
    def _compute_sg_accuracy(self, session, start: datetime, end: datetime) -> dict:
        """Compare projected SG to actual results."""
        # This requires matching signals to tournament results
        # For now, compute aggregate SG stats accuracy
        sg_stats = (
            session.query(SGStat)
            .filter(SGStat.sg_total.isnot(None))
            .filter(SGStat.created_at >= start)
            .filter(SGStat.created_at <= end)
            .all()
        )

        if not sg_stats:
            return {"n_records": 0}

        totals = [s.sg_total for s in sg_stats if s.sg_total is not None]

        return {
            "n_records": len(sg_stats),
            "avg_sg_total": round(float(np.mean(totals)), 3) if totals else 0.0,
            "std_sg_total": round(float(np.std(totals)), 3) if totals else 0.0,
            "min_sg_total": round(float(min(totals)), 3) if totals else 0.0,
            "max_sg_total": round(float(max(totals)), 3) if totals else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Component: Weather impact
    # ------------------------------------------------------------------ #
    def _compute_weather_impact(self, session, start: datetime, end: datetime) -> dict:
        """Analyze weather impact on predictions and outcomes."""
        signals = (
            session.query(Signal)
            .filter(Signal.sport == "GOLF")
            .filter(Signal.generated_at >= start)
            .filter(Signal.generated_at <= end)
            .all()
        )

        # Count signals by weather-adjusted flag
        # We cannot directly query weather data from here, but we can look at
        # signal metadata if it includes weather info
        return {
            "total_signals": len(signals),
            "note": "Weather impact tracked via signal generation adjustments",
        }

    # ------------------------------------------------------------------ #
    # Component: Course fit accuracy
    # ------------------------------------------------------------------ #
    def _compute_course_fit_accuracy(self, session) -> dict:
        """Evaluate how well course fit predictions matched outcomes."""
        # Get bets that have features snapshots with course_fit data
        bets = (
            session.query(Bet)
            .filter(Bet.sport == "GOLF")
            .filter(Bet.status.in_(["won", "lost"]))
            .filter(Bet.features_snapshot_json.isnot(None))
            .order_by(Bet.settled_at.desc())
            .limit(200)
            .all()
        )

        high_fit_won = 0
        high_fit_total = 0
        low_fit_won = 0
        low_fit_total = 0

        for b in bets:
            try:
                features = json.loads(b.features_snapshot_json)
            except (json.JSONDecodeError, TypeError):
                continue

            fit_score = features.get("course_fit_score", 50)

            if fit_score > 65:
                high_fit_total += 1
                if b.status == "won":
                    high_fit_won += 1
            elif fit_score < 40:
                low_fit_total += 1
                if b.status == "won":
                    low_fit_won += 1

        return {
            "high_fit_win_rate": round(high_fit_won / max(high_fit_total, 1), 3),
            "high_fit_total": high_fit_total,
            "low_fit_win_rate": round(low_fit_won / max(low_fit_total, 1), 3),
            "low_fit_total": low_fit_total,
            "fit_edge": round(
                (high_fit_won / max(high_fit_total, 1)) - (low_fit_won / max(low_fit_total, 1)), 3
            ),
        }

    # ------------------------------------------------------------------ #
    # System recommendation
    # ------------------------------------------------------------------ #
    def _compute_recommendation(self, report: dict) -> str:
        """Generate system state recommendation based on report data."""
        pnl = report.get("pnl", {})
        clv = report.get("clv", {})
        cal = report.get("calibration", {})

        roi = pnl.get("roi_pct", 0)
        clv_100 = clv.get("clv_100", {})
        avg_clv = clv_100.get("avg_clv", 0) if clv_100 else 0
        beat_close = clv_100.get("beat_close_pct", 0) if clv_100 else 0
        brier = cal.get("brier_score", 0.25)
        mae = cal.get("mae", 0)

        # Decision logic
        if avg_clv > 0.02 and beat_close > 0.55 and brier < 0.25:
            return "aggressive"
        elif avg_clv > 0 and beat_close > 0.50 and brier < 0.28:
            return "normal"
        elif avg_clv < -0.02 or beat_close < 0.45 or brier > 0.30:
            return "conservative"
        elif avg_clv < -0.05 or beat_close < 0.40 or brier > 0.35:
            return "suspend"
        else:
            return "normal"

    # ------------------------------------------------------------------ #
    # Summary formatter
    # ------------------------------------------------------------------ #
    def _format_summary(self, report: dict) -> str:
        """Generate human-readable summary text."""
        pnl = report.get("pnl", {})
        clv = report.get("clv", {})
        cal = report.get("calibration", {})
        rec = report.get("recommendation", "normal")
        period = report.get("period", "daily")

        clv_100 = clv.get("clv_100", {})
        avg_clv = clv_100.get("avg_clv", 0) if clv_100 else 0
        beat_close = clv_100.get("beat_close_pct", 0) if clv_100 else 0

        lines = [
            f"Golf Edge Report ({period.upper()}) -- {report.get('report_date', '')}",
            f"P&L: ${pnl.get('total_pnl', 0):.2f} ({pnl.get('roi_pct', 0):.1f}% ROI) | "
            f"{pnl.get('won', 0)}W-{pnl.get('lost', 0)}L",
            f"CLV (100): avg={avg_clv:.4f} | beat_close={beat_close:.1%}",
            f"Calibration: Brier={cal.get('brier_score', 0):.4f} | MAE={cal.get('mae', 0):.4f}",
            f"Recommendation: {rec.upper()}",
        ]

        return " | ".join(lines)

    # ------------------------------------------------------------------ #
    # Save report
    # ------------------------------------------------------------------ #
    def _save_report(self, session, report_type: str, data: dict):
        """Persist the report to the edge_reports table."""
        report = EdgeReport(
            report_type=report_type,
            sport="GOLF",
            report_json=json.dumps(data, default=str),
        )
        session.add(report)
        self._logger.info("Saved %s report", report_type)


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    )
    import argparse
    parser = argparse.ArgumentParser(description="Golf Report Worker")
    parser.add_argument("--loop", action="store_true", help="Run in loop mode")
    parser.add_argument("--weekly", action="store_true", help="Force weekly report")
    args = parser.parse_args()

    worker = ReportWorker()
    if args.loop:
        worker.run_loop()
    else:
        success = worker.run_once()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
