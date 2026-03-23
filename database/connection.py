"""
Golf Quant Engine — Database Connection Manager
================================================
SQLite-backed with SQLAlchemy 2.0+ ORM.  Designed for seamless upgrade to
PostgreSQL by swapping the DATABASE_URL env var.

Features:
  - Singleton engine / session factory
  - WAL journal mode for SQLite
  - Foreign keys enforced
  - Connection pooling via SQLAlchemy
  - Health-check helper
"""

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_DB_NAME = "golf_quant.db"

_engine: Engine | None = None
_session_factory: sessionmaker | None = None
_lock = Lock()


def _resolve_url() -> str:
    """Return the database URL from env or build a default SQLite path."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
    db_path = os.environ.get("GOLF_DB_PATH", str(_DEFAULT_DB_DIR / _DEFAULT_DB_NAME))
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


# ---------------------------------------------------------------------------
# SQLite-specific PRAGMA hooks
# ---------------------------------------------------------------------------
def _set_sqlite_pragmas(dbapi_conn, connection_record):
    """Enable WAL mode and foreign keys for every new raw DBAPI connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_engine() -> Engine:
    """Return the singleton SQLAlchemy engine, creating it on first call."""
    global _engine
    if _engine is not None:
        return _engine

    with _lock:
        if _engine is not None:
            return _engine

        url = _resolve_url()
        is_sqlite = url.startswith("sqlite")

        kwargs: dict = {}
        if is_sqlite:
            kwargs.update(
                pool_pre_ping=True,
                connect_args={"check_same_thread": False},
            )
        else:
            kwargs.update(
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,
            )

        engine = create_engine(url, echo=False, **kwargs)

        if is_sqlite:
            event.listen(engine, "connect", _set_sqlite_pragmas)

        _engine = engine
        log.info("Database engine created: %s", url.split("?")[0])
        return _engine


def get_session_factory() -> sessionmaker:
    """Return the singleton session factory."""
    global _session_factory
    if _session_factory is not None:
        return _session_factory

    with _lock:
        if _session_factory is not None:
            return _session_factory
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
        return _session_factory


def get_session() -> Generator[Session, None, None]:
    """Context-manager that yields a scoped session and auto-commits / rolls
    back.  Usage::

        with get_session() as session:
            session.add(obj)
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create every table defined in *models.py* (idempotent)."""
    from database.models import Base  # local import to break circular ref

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    log.info("All tables created / verified.")


def health_check() -> dict:
    """Quick connectivity + version check.  Returns a dict with status info."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            if engine.url.drivername.startswith("sqlite"):
                row = conn.execute(text("SELECT sqlite_version()")).fetchone()
                version = row[0] if row else "unknown"
                wal_row = conn.execute(text("PRAGMA journal_mode")).fetchone()
                journal = wal_row[0] if wal_row else "unknown"
            else:
                row = conn.execute(text("SELECT version()")).fetchone()
                version = row[0] if row else "unknown"
                journal = "n/a"

            return {
                "status": "ok",
                "driver": engine.url.drivername,
                "version": version,
                "journal_mode": journal,
            }
    except Exception as exc:
        log.error("Health-check failed: %s", exc)
        return {"status": "error", "error": str(exc)}


def reset_engine() -> None:
    """Dispose of the current engine (useful for testing)."""
    global _engine, _session_factory
    with _lock:
        if _engine is not None:
            _engine.dispose()
            _engine = None
        _session_factory = None
