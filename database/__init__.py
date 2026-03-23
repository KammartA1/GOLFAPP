"""
Golf Quant Engine — Database Package
=====================================
Re-exports from the new SQLAlchemy layer **and** the legacy ``db_manager``
module so that existing code continues to work.

Usage examples::

    from database import get_engine, get_session, init_db, Bet, Player
    from database import get_conn  # legacy sqlite3 helper still available
"""

# ── New SQLAlchemy layer ─────────────────────────────────────────────────
from database.connection import (
    get_engine,
    get_session,
    get_session_factory,
    init_db,
    health_check,
    reset_engine,
)

from database.models import (
    Base,
    Player,
    Event,
    Bet,
    LineMovement,
    ModelVersion,
    EdgeReport,
    Signal,
    UserSetting,
    WorkerStatus,
    CalibrationSnapshot,
    SystemState,
    SGStat,
    TournamentField,
    ALL_MODELS,
)

from database.migrations import (
    auto_migrate,
    rollback,
    get_current_version,
    set_version,
    get_migration_status,
    register_migration,
)

# ── Legacy raw-sqlite helpers (db_manager.py) ────────────────────────────
from database.db_manager import (
    get_conn,
    normalize_name,
    deactivate_old_pp_lines,
    insert_pp_line,
    get_active_pp_lines,
    get_pp_last_scraped,
    insert_odds_line,
    upsert_odds_consensus,
    get_odds_consensus,
    get_odds_last_scraped,
    insert_weather,
    get_current_weather,
    get_weather_forecast,
    get_weather_last_fetched,
    upsert_tournament,
    get_current_tournament,
    upsert_live_score,
    get_leaderboard,
    insert_bet,
    settle_bet,
    get_bets,
)

__all__ = [
    # connection
    "get_engine",
    "get_session",
    "get_session_factory",
    "init_db",
    "health_check",
    "reset_engine",
    # models
    "Base",
    "Player",
    "Event",
    "Bet",
    "LineMovement",
    "ModelVersion",
    "EdgeReport",
    "Signal",
    "UserSetting",
    "WorkerStatus",
    "CalibrationSnapshot",
    "SystemState",
    "SGStat",
    "TournamentField",
    "ALL_MODELS",
    # migrations
    "auto_migrate",
    "rollback",
    "get_current_version",
    "set_version",
    "get_migration_status",
    "register_migration",
    # legacy
    "get_conn",
    "normalize_name",
    "deactivate_old_pp_lines",
    "insert_pp_line",
    "get_active_pp_lines",
    "get_pp_last_scraped",
    "insert_odds_line",
    "upsert_odds_consensus",
    "get_odds_consensus",
    "get_odds_last_scraped",
    "insert_weather",
    "get_current_weather",
    "get_weather_forecast",
    "get_weather_last_fetched",
    "upsert_tournament",
    "get_current_tournament",
    "upsert_live_score",
    "get_leaderboard",
    "insert_bet",
    "settle_bet",
    "get_bets",
]
