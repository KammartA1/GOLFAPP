"""Data Service — Read-only data layer for Streamlit.

Streamlit calls these functions to get data. All functions are read-only
or write only user-initiated actions (place bet, update settings).
No session_state needed — everything comes from/goes to the database.

Usage in Streamlit:
    from services.data_service import DataService

    ds = DataService("golf")
    lines = ds.get_active_lines()
    report = ds.get_latest_edge_report()
    ds.save_user_setting("bankroll", 1000.0)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from database.connection import DatabaseManager, get_session
from database.models import (
    Bet, LineMovement, Signal, EdgeReport, ModelVersion,
    CLVLog, CalibrationLog, SystemStateLog, FeatureLog,
    Player, Event, ScraperStatus, WorkerState, AuditLog,
    UserSetting, Projection, SGStats, TournamentResult, WeatherData,
    normalize_player_name,
)

logger = logging.getLogger(__name__)


class DataService:
    """Unified data service for the Streamlit frontend."""

    def __init__(self, sport: str = "golf"):
        self.sport = sport

    # ── Lines & Odds ──────────────────────────────────────────────────

    def get_active_lines(self, source: str | None = None) -> list[dict]:
        """Get all currently active lines."""
        with DatabaseManager.session_scope() as session:
            q = (
                session.query(LineMovement)
                .filter(LineMovement.sport == self.sport, LineMovement.is_active == True)
            )
            if source:
                q = q.filter(LineMovement.source == source)
            q = q.order_by(LineMovement.captured_at.desc())

            return [
                {
                    "id": lm.id,
                    "player": lm.player,
                    "stat_type": lm.stat_type,
                    "line": lm.line,
                    "source": lm.source,
                    "odds_type": lm.odds_type,
                    "is_flash_sale": lm.is_flash_sale,
                    "captured_at": lm.captured_at.isoformat() if lm.captured_at else None,
                }
                for lm in q.all()
            ]

    def get_line_history(self, player: str, stat_type: str, days: int = 7) -> list[dict]:
        """Get line movement history for a player/stat."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        with DatabaseManager.session_scope() as session:
            rows = (
                session.query(LineMovement)
                .filter(
                    LineMovement.sport == self.sport,
                    LineMovement.player == player,
                    LineMovement.stat_type == stat_type,
                    LineMovement.captured_at >= cutoff,
                )
                .order_by(LineMovement.captured_at.asc())
                .all()
            )
            return [
                {"line": r.line, "source": r.source, "captured_at": r.captured_at.isoformat()}
                for r in rows
            ]

    # ── Signals ───────────────────────────────────────────────────────

    def get_recent_signals(self, hours: int = 24, approved_only: bool = False) -> list[dict]:
        """Get recent betting signals."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        with DatabaseManager.session_scope() as session:
            q = (
                session.query(Signal)
                .filter(Signal.sport == self.sport, Signal.generated_at >= cutoff)
            )
            if approved_only:
                q = q.filter(Signal.approved == True)
            q = q.order_by(Signal.generated_at.desc())

            return [
                {
                    "id": s.id,
                    "player": s.player,
                    "stat_type": s.stat_type,
                    "line": s.line,
                    "direction": s.direction,
                    "edge": s.edge,
                    "model_prob": s.model_prob,
                    "approved": s.approved,
                    "rejection_reason": s.rejection_reason,
                    "recommended_stake": s.recommended_stake,
                    "generated_at": s.generated_at.isoformat(),
                }
                for s in q.all()
            ]

    # ── Bets ──────────────────────────────────────────────────────────

    def get_bets(self, status: str | None = None, limit: int = 100) -> list[dict]:
        """Get bets, optionally filtered by status."""
        with DatabaseManager.session_scope() as session:
            q = session.query(Bet).filter(Bet.sport == self.sport)
            if status:
                q = q.filter(Bet.status == status)
            q = q.order_by(Bet.timestamp.desc()).limit(limit)

            return [
                {
                    "bet_id": b.bet_id,
                    "player": b.player,
                    "stat_type": b.stat_type,
                    "line": b.line,
                    "direction": b.direction,
                    "stake": b.stake,
                    "odds_american": b.odds_american,
                    "edge": b.edge,
                    "model_prob": b.model_prob,
                    "status": b.status,
                    "pnl": b.pnl,
                    "actual_result": b.actual_result,
                    "timestamp": b.timestamp.isoformat(),
                    "settled_at": b.settled_at.isoformat() if b.settled_at else None,
                }
                for b in q.all()
            ]

    def get_bet_summary(self) -> dict:
        """Get aggregate bet statistics."""
        with DatabaseManager.session_scope() as session:
            settled = (
                session.query(Bet)
                .filter(Bet.sport == self.sport, Bet.status.in_(["won", "lost"]))
                .all()
            )

            if not settled:
                return {
                    "total_bets": 0, "wins": 0, "losses": 0,
                    "win_rate": 0.0, "total_pnl": 0.0, "roi": 0.0,
                    "avg_stake": 0.0, "avg_edge": 0.0,
                }

            wins = sum(1 for b in settled if b.status == "won")
            total_staked = sum(b.stake for b in settled)
            total_pnl = sum(b.pnl for b in settled)

            pending = (
                session.query(Bet)
                .filter(Bet.sport == self.sport, Bet.status == "pending")
                .count()
            )

            return {
                "total_bets": len(settled),
                "pending": pending,
                "wins": wins,
                "losses": len(settled) - wins,
                "win_rate": round(wins / len(settled), 4),
                "total_pnl": round(total_pnl, 2),
                "roi": round(total_pnl / total_staked, 4) if total_staked > 0 else 0.0,
                "avg_stake": round(total_staked / len(settled), 2),
                "avg_edge": round(sum(b.edge for b in settled) / len(settled), 4),
                "total_staked": round(total_staked, 2),
            }

    # ── CLV ───────────────────────────────────────────────────────────

    def get_clv_summary(self) -> dict:
        """Get CLV rolling averages."""
        import numpy as np

        with DatabaseManager.session_scope() as session:
            all_clv = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport)
                .order_by(CLVLog.calculated_at.desc())
                .limit(500)
                .all()
            )

            if not all_clv:
                return {f"clv_{w}": {"avg": 0, "beat_rate": 0, "n": 0} for w in [50, 100, 250, 500]}

            result = {}
            for window in [50, 100, 250, 500]:
                subset = all_clv[:window]
                if subset:
                    result[f"clv_{window}"] = {
                        "avg": round(float(np.mean([c.clv_cents for c in subset])), 2),
                        "beat_rate": round(sum(1 for c in subset if c.beat_close) / len(subset), 4),
                        "n": len(subset),
                    }
                else:
                    result[f"clv_{window}"] = {"avg": 0, "beat_rate": 0, "n": 0}

            return result

    def get_clv_timeseries(self, limit: int = 200) -> list[dict]:
        """Get CLV time series for charting."""
        with DatabaseManager.session_scope() as session:
            rows = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport)
                .order_by(CLVLog.calculated_at.asc())
                .limit(limit)
                .all()
            )
            return [
                {"date": c.calculated_at.isoformat(), "clv_cents": c.clv_cents, "beat_close": c.beat_close}
                for c in rows
            ]

    # ── Edge Reports ──────────────────────────────────────────────────

    def get_latest_edge_report(self) -> dict | None:
        """Get the most recent edge report."""
        with DatabaseManager.session_scope() as session:
            report = (
                session.query(EdgeReport)
                .filter(EdgeReport.sport == self.sport)
                .order_by(EdgeReport.report_date.desc())
                .first()
            )

            if not report:
                return None

            return {
                "report_date": report.report_date.isoformat(),
                "clv_last_50": report.clv_last_50,
                "clv_last_100": report.clv_last_100,
                "clv_last_250": report.clv_last_250,
                "clv_last_500": report.clv_last_500,
                "calibration_error": report.calibration_error,
                "model_roi": report.model_roi,
                "expected_roi": report.expected_roi,
                "bankroll": report.bankroll,
                "peak_bankroll": report.peak_bankroll,
                "drawdown_pct": report.drawdown_pct,
                "edge_exists": report.edge_exists,
                "system_state": report.system_state,
                "warnings": json.loads(report.warnings) if report.warnings else [],
                "actions": json.loads(report.actions) if report.actions else [],
            }

    def get_edge_history(self, days: int = 30) -> list[dict]:
        """Get edge report history for trend charting."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        with DatabaseManager.session_scope() as session:
            rows = (
                session.query(EdgeReport)
                .filter(EdgeReport.sport == self.sport, EdgeReport.report_date >= cutoff)
                .order_by(EdgeReport.report_date.asc())
                .all()
            )
            return [
                {
                    "date": r.report_date.isoformat(),
                    "clv_100": r.clv_last_100,
                    "calibration_error": r.calibration_error,
                    "edge_exists": r.edge_exists,
                    "system_state": r.system_state,
                    "drawdown_pct": r.drawdown_pct,
                }
                for r in rows
            ]

    # ── System State ──────────────────────────────────────────────────

    def get_system_state(self) -> str:
        """Get current system state."""
        with DatabaseManager.session_scope() as session:
            latest = (
                session.query(SystemStateLog)
                .filter(SystemStateLog.sport == self.sport)
                .order_by(SystemStateLog.timestamp.desc())
                .first()
            )
            return latest.new_state if latest else "active"

    def get_state_history(self, limit: int = 20) -> list[dict]:
        """Get system state change history."""
        with DatabaseManager.session_scope() as session:
            rows = (
                session.query(SystemStateLog)
                .filter(SystemStateLog.sport == self.sport)
                .order_by(SystemStateLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "previous": r.previous_state,
                    "new": r.new_state,
                    "reason": r.reason,
                }
                for r in rows
            ]

    # ── Model Versions ────────────────────────────────────────────────

    def get_active_model(self) -> dict | None:
        """Get the active model version."""
        with DatabaseManager.session_scope() as session:
            model = (
                session.query(ModelVersion)
                .filter_by(sport=self.sport, is_active=True)
                .first()
            )
            if not model:
                return None
            return {
                "id": model.id,
                "version": model.version,
                "trained_at": model.trained_at.isoformat(),
                "live_accuracy": model.live_accuracy,
                "live_roi": model.live_roi,
                "live_clv_avg": model.live_clv_avg,
                "n_live_bets": model.n_live_bets,
                "psi_score": model.psi_score,
                "is_degraded": model.is_degraded,
            }

    # ── Workers & Health ──────────────────────────────────────────────

    def get_worker_statuses(self) -> list[dict]:
        """Get all worker statuses."""
        with DatabaseManager.session_scope() as session:
            workers = session.query(WorkerState).all()
            return [
                {
                    "name": w.worker_name,
                    "status": w.status,
                    "last_run": w.last_run.isoformat() if w.last_run else None,
                    "next_scheduled": w.next_scheduled.isoformat() if w.next_scheduled else None,
                    "run_count": w.run_count,
                    "avg_duration": w.avg_duration_seconds,
                    "last_error": w.last_error,
                }
                for w in workers
            ]

    def get_scraper_statuses(self) -> list[dict]:
        """Get all scraper statuses."""
        with DatabaseManager.session_scope() as session:
            scrapers = session.query(ScraperStatus).all()
            return [
                {
                    "name": s.scraper_name,
                    "is_healthy": s.is_healthy,
                    "last_success": s.last_success.isoformat() if s.last_success else None,
                    "consecutive_failures": s.consecutive_failures,
                    "total_runs": s.total_runs,
                    "lines_last_scrape": s.lines_last_scrape,
                }
                for s in scrapers
            ]

    def get_system_health(self) -> dict:
        """Comprehensive system health check."""
        from database.connection import DatabaseManager as DM
        db_health = DM.health_check()

        return {
            "database": db_health,
            "workers": self.get_worker_statuses(),
            "scrapers": self.get_scraper_statuses(),
            "system_state": self.get_system_state(),
            "active_model": self.get_active_model(),
        }

    # ── Audit Logs ────────────────────────────────────────────────────

    def get_recent_logs(self, limit: int = 50, level: str | None = None) -> list[dict]:
        """Get recent audit log entries."""
        with DatabaseManager.session_scope() as session:
            q = session.query(AuditLog)
            if level:
                q = q.filter(AuditLog.level == level)
            rows = q.order_by(AuditLog.timestamp.desc()).limit(limit).all()
            return [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "category": r.category,
                    "level": r.level,
                    "message": r.message,
                }
                for r in rows
            ]

    # ── User Settings (replaces session_state) ────────────────────────

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a user setting from database."""
        with DatabaseManager.session_scope() as session:
            setting = session.query(UserSetting).filter_by(key=key).first()
            if setting and setting.value is not None:
                try:
                    return json.loads(setting.value)
                except (json.JSONDecodeError, TypeError):
                    return setting.value
            return default

    def save_setting(self, key: str, value: Any, category: str = "general"):
        """Save a user setting to database."""
        with DatabaseManager.session_scope() as session:
            setting = session.query(UserSetting).filter_by(key=key).first()
            if setting is None:
                setting = UserSetting(key=key, category=category)
                session.add(setting)
            setting.value = json.dumps(value)
            setting.updated_at = datetime.utcnow()

    def get_all_settings(self, category: str | None = None) -> dict:
        """Get all user settings as a dict."""
        with DatabaseManager.session_scope() as session:
            q = session.query(UserSetting)
            if category:
                q = q.filter_by(category=category)
            settings = q.all()
            result = {}
            for s in settings:
                try:
                    result[s.key] = json.loads(s.value) if s.value else None
                except (json.JSONDecodeError, TypeError):
                    result[s.key] = s.value
            return result

    # ── Golf-Specific ─────────────────────────────────────────────────

    def get_players(self, active_only: bool = True) -> list[dict]:
        """Get all tracked players."""
        with DatabaseManager.session_scope() as session:
            q = session.query(Player).filter(Player.sport == self.sport)
            if active_only:
                q = q.filter(Player.active == True)
            q = q.order_by(Player.name)

            return [
                {
                    "id": p.id,
                    "name": p.name,
                    "world_rank": p.world_rank,
                    "active": p.active,
                }
                for p in q.all()
            ]

    def get_player_projections(self, event_id: str | None = None) -> list[dict]:
        """Get latest projections."""
        with DatabaseManager.session_scope() as session:
            q = session.query(Projection)
            if event_id:
                q = q.filter(Projection.event_id == event_id)
            q = q.order_by(Projection.updated_at.desc()).limit(200)

            return [
                {
                    "player_id": p.player_id,
                    "event_id": p.event_id,
                    "sg_total": p.projected_sg_total,
                    "win_prob": p.win_prob,
                    "top10_prob": p.top10_prob,
                    "make_cut_prob": p.make_cut_prob,
                    "dfs_points": p.projected_dfs_points,
                    "confidence": p.confidence,
                    "edge_pct": p.edge_pct,
                }
                for p in q.all()
            ]

    def get_current_event(self) -> dict | None:
        """Get the current/upcoming event."""
        with DatabaseManager.session_scope() as session:
            event = (
                session.query(Event)
                .filter(
                    Event.sport == self.sport,
                    Event.status.in_(["scheduled", "in_progress"]),
                )
                .order_by(Event.start_date.asc())
                .first()
            )
            if not event:
                return None
            return {
                "id": event.id,
                "event_id": event.event_id,
                "name": event.name,
                "course": event.course_name,
                "start_date": event.start_date.isoformat() if event.start_date else None,
                "status": event.status,
                "purse": event.purse,
                "field_size": event.field_size,
            }
