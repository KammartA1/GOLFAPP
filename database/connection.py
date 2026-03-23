"""Unified database connection manager.

Single source of truth for ALL database connections across the app.
SQLite with WAL mode now, upgradeable to PostgreSQL by changing the URL.

Usage:
    from database.connection import get_engine, get_session, DatabaseManager

    # Quick session
    with DatabaseManager.session_scope() as session:
        session.query(Bet).all()

    # Or manual
    session = get_session()
    try:
        ...
    finally:
        session.close()
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_engine = None
_SessionFactory = None


def _default_db_path() -> str:
    """Resolve the default database path."""
    return os.environ.get(
        "QUANT_DB_PATH",
        os.path.join(os.path.dirname(__file__), "..", "data", "quant_system.db"),
    )


def get_db_url(db_path: str | None = None) -> str:
    """Build a database URL. Supports SQLite and PostgreSQL.

    Set QUANT_DATABASE_URL env var for PostgreSQL:
        export QUANT_DATABASE_URL=postgresql://user:pass@host:5432/dbname
    """
    env_url = os.environ.get("QUANT_DATABASE_URL")
    if env_url:
        return env_url

    if db_path is None:
        db_path = _default_db_path()
    db_path = os.path.abspath(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return f"sqlite:///{db_path}"


def get_engine(db_path: str | None = None):
    """Get or create the singleton SQLAlchemy engine."""
    global _engine
    if _engine is not None:
        return _engine

    with _lock:
        if _engine is not None:
            return _engine

        url = get_db_url(db_path)
        is_sqlite = url.startswith("sqlite")

        kwargs = {"echo": False}
        if is_sqlite:
            kwargs["connect_args"] = {"check_same_thread": False}
            # Use StaticPool for in-memory DBs (testing)
            if ":memory:" in url:
                kwargs["poolclass"] = StaticPool

        _engine = create_engine(url, **kwargs)

        if is_sqlite:
            # Enable WAL mode and foreign keys on every connection
            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=5000")
                cursor.close()

        # Create all tables
        from .models import Base
        Base.metadata.create_all(_engine)

        logger.info("Database engine initialized: %s", url.split("@")[-1] if "@" in url else url)

    return _engine


def get_session(db_path: str | None = None) -> Session:
    """Get a new SQLAlchemy session."""
    global _SessionFactory
    if _SessionFactory is None:
        with _lock:
            if _SessionFactory is None:
                engine = get_engine(db_path)
                _SessionFactory = sessionmaker(bind=engine)
    return _SessionFactory()


def reset_engine():
    """Reset the engine and session factory. Used in testing."""
    global _engine, _SessionFactory
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _SessionFactory = None


class DatabaseManager:
    """High-level database manager with context-managed sessions."""

    @staticmethod
    @contextmanager
    def session_scope():
        """Provide a transactional scope around a series of operations.

        Usage:
            with DatabaseManager.session_scope() as session:
                session.add(something)
                # auto-commits on success, auto-rollbacks on exception
        """
        session = get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def execute_raw(sql: str, params: dict | None = None):
        """Execute raw SQL for migrations or one-off queries."""
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            conn.commit()
            return result

    @staticmethod
    def health_check() -> dict:
        """Quick database health check."""
        try:
            engine = get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"status": "healthy", "url": str(engine.url).split("@")[-1]}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
