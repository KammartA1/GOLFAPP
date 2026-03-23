"""Schema versioning and migrations.

Lightweight migration system that tracks schema versions and applies
incremental changes. For production PostgreSQL, swap to Alembic.

Usage:
    from database.migrations import MigrationManager
    MigrationManager.run_pending()
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, Text, text, inspect

from .connection import get_engine, DatabaseManager
from .models import Base

logger = logging.getLogger(__name__)


# ── Schema Version Table ──────────────────────────────────────────────

class SchemaVersion(Base):
    """Tracks applied migrations."""
    __tablename__ = "schema_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(Integer, nullable=False, unique=True)
    description = Column(String(256), nullable=False)
    applied_at = Column(DateTime, default=datetime.utcnow)
    sql_applied = Column(Text, nullable=True)


# ── Migration Definitions ─────────────────────────────────────────────

MIGRATIONS = [
    {
        "version": 1,
        "description": "Initial schema — all tables created by SQLAlchemy create_all",
        "sql": None,  # Handled by Base.metadata.create_all
    },
    {
        "version": 2,
        "description": "Add composite indexes for common query patterns",
        "sql": [
            "CREATE INDEX IF NOT EXISTS ix_bet_settled_sport ON bets(sport, settled_at) WHERE settled_at IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS ix_line_recent ON line_movements(sport, source, captured_at DESC)",
        ],
    },
    {
        "version": 3,
        "description": "Add worker heartbeat column",
        "sql": [
            "ALTER TABLE worker_state ADD COLUMN heartbeat_at DATETIME",
        ],
    },
]


class MigrationManager:
    """Manages database schema migrations."""

    @staticmethod
    def get_current_version() -> int:
        """Get the current schema version."""
        engine = get_engine()
        inspector = inspect(engine)

        if "schema_versions" not in inspector.get_table_names():
            return 0

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT MAX(version) FROM schema_versions")
            )
            row = result.fetchone()
            return row[0] if row and row[0] else 0

    @staticmethod
    def run_pending():
        """Apply all pending migrations."""
        # Ensure all tables exist first
        engine = get_engine()
        Base.metadata.create_all(engine)

        current = MigrationManager.get_current_version()
        pending = [m for m in MIGRATIONS if m["version"] > current]

        if not pending:
            logger.debug("No pending migrations (current version: %d)", current)
            return

        logger.info("Applying %d pending migrations (current: v%d → v%d)",
                     len(pending), current, pending[-1]["version"])

        for migration in pending:
            MigrationManager._apply(migration)

    @staticmethod
    def _apply(migration: dict):
        """Apply a single migration."""
        version = migration["version"]
        description = migration["description"]
        sql_statements = migration.get("sql")

        engine = get_engine()
        sql_log = []

        try:
            if sql_statements:
                with engine.connect() as conn:
                    for stmt in sql_statements:
                        try:
                            conn.execute(text(stmt))
                            sql_log.append(stmt)
                        except Exception as e:
                            # SQLite doesn't support IF NOT EXISTS for ALTER TABLE
                            if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                                logger.debug("Skipping already-applied: %s", stmt[:80])
                                sql_log.append(f"SKIPPED: {stmt}")
                            else:
                                raise
                    conn.commit()

            # Record migration
            with DatabaseManager.session_scope() as session:
                record = SchemaVersion(
                    version=version,
                    description=description,
                    applied_at=datetime.utcnow(),
                    sql_applied="\n".join(sql_log) if sql_log else "create_all",
                )
                session.add(record)

            logger.info("Migration v%d applied: %s", version, description)

        except Exception:
            logger.exception("Migration v%d FAILED: %s", version, description)
            raise

    @staticmethod
    def verify_schema() -> dict:
        """Verify that all expected tables exist."""
        engine = get_engine()
        inspector = inspect(engine)
        existing = set(inspector.get_table_names())

        expected = {t.name for t in Base.metadata.sorted_tables}
        missing = expected - existing
        extra = existing - expected - {"sqlite_sequence"}  # SQLite internal

        return {
            "version": MigrationManager.get_current_version(),
            "expected_tables": len(expected),
            "existing_tables": len(existing),
            "missing": sorted(missing),
            "extra": sorted(extra),
            "healthy": len(missing) == 0,
        }
