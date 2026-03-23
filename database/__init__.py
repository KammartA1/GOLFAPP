"""Unified database package — single source of truth for all data.

Usage:
    from database.connection import DatabaseManager, get_session
    from database.models import Bet, Player, LineMovement, Signal
    from database.migrations import MigrationManager

    # Initialize on startup
    MigrationManager.run_pending()

    # Use context-managed sessions
    with DatabaseManager.session_scope() as session:
        bets = session.query(Bet).filter_by(sport="golf").all()
"""

from .connection import DatabaseManager, get_engine, get_session, reset_engine
from .models import (
    Base,
    Player, Event, Bet, LineMovement, ModelVersion,
    EdgeReport, Signal, CLVLog, CalibrationLog,
    SystemStateLog, FeatureLog, ScraperStatus, AuditLog,
    SGStats, TournamentResult, Projection, WeatherData, DFSLineup,
    WorkerState, UserSetting,
    normalize_player_name,
)
from .migrations import MigrationManager

__all__ = [
    "DatabaseManager", "get_engine", "get_session", "reset_engine",
    "Base", "MigrationManager",
    "Player", "Event", "Bet", "LineMovement", "ModelVersion",
    "EdgeReport", "Signal", "CLVLog", "CalibrationLog",
    "SystemStateLog", "FeatureLog", "ScraperStatus", "AuditLog",
    "SGStats", "TournamentResult", "Projection", "WeatherData", "DFSLineup",
    "WorkerState", "UserSetting",
    "normalize_player_name",
]
