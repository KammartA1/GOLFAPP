"""
Golf Quant Engine — Schema Migrations
======================================
Lightweight migration framework built on SQLAlchemy.

Each migration is a versioned pair of ``up()`` / ``down()`` callables stored
in a global registry.  ``auto_migrate()`` applies every pending migration in
order.

The current schema version is persisted in a ``schema_versions`` table.
"""

import logging
from datetime import datetime
from typing import Callable

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    Table,
    MetaData,
    text,
    inspect,
)
from sqlalchemy.engine import Engine

from database.connection import get_engine

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema-version tracking table (raw metadata — not part of ORM Base)
# ---------------------------------------------------------------------------
_meta = MetaData()

schema_versions_table = Table(
    "schema_versions",
    _meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("version", Integer, nullable=False, unique=True),
    Column("description", String(500), nullable=True),
    Column("applied_at", DateTime, default=datetime.utcnow),
    Column("rollback_sql", Text, nullable=True),
)


def _ensure_version_table(engine: Engine) -> None:
    """Create the schema_versions table if it doesn't exist."""
    _meta.create_all(bind=engine, tables=[schema_versions_table])


# ---------------------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------------------
_migrations: dict[int, dict] = {}


def register_migration(
    version: int,
    description: str,
    up: Callable[[Engine], None],
    down: Callable[[Engine], None],
) -> None:
    """Register a migration with its version, description, and up/down callables."""
    if version in _migrations:
        raise ValueError(f"Migration version {version} is already registered.")
    _migrations[version] = {
        "description": description,
        "up": up,
        "down": down,
    }


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------
def get_current_version(engine: Engine | None = None) -> int:
    """Return the highest applied migration version, or 0 if none."""
    engine = engine or get_engine()
    _ensure_version_table(engine)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT MAX(version) AS v FROM schema_versions")
        ).fetchone()
        return row[0] if row and row[0] is not None else 0


def set_version(engine: Engine, version: int, description: str = "") -> None:
    """Record that a migration version has been applied."""
    with engine.begin() as conn:
        conn.execute(
            schema_versions_table.insert().values(
                version=version,
                description=description,
                applied_at=datetime.utcnow(),
            )
        )


def _remove_version(engine: Engine, version: int) -> None:
    """Remove a version record (used during rollback)."""
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM schema_versions WHERE version = :v"),
            {"v": version},
        )


# ---------------------------------------------------------------------------
# Core migration runner
# ---------------------------------------------------------------------------
def auto_migrate(engine: Engine | None = None) -> list[int]:
    """Apply all pending migrations in ascending order.

    Returns the list of version numbers that were applied.
    """
    engine = engine or get_engine()
    _ensure_version_table(engine)
    current = get_current_version(engine)

    pending = sorted(v for v in _migrations if v > current)
    if not pending:
        log.info("Schema is up-to-date at version %d.", current)
        return []

    applied: list[int] = []
    for ver in pending:
        mig = _migrations[ver]
        log.info("Applying migration v%d: %s", ver, mig["description"])
        try:
            mig["up"](engine)
            set_version(engine, ver, mig["description"])
            applied.append(ver)
            log.info("Migration v%d applied successfully.", ver)
        except Exception:
            log.exception("Migration v%d FAILED — stopping.", ver)
            raise

    return applied


def rollback(target_version: int = 0, engine: Engine | None = None) -> list[int]:
    """Roll back migrations from current version down to (and including)
    target_version + 1.

    Returns the list of version numbers that were rolled back.
    """
    engine = engine or get_engine()
    _ensure_version_table(engine)
    current = get_current_version(engine)

    to_rollback = sorted(
        (v for v in _migrations if target_version < v <= current),
        reverse=True,
    )
    if not to_rollback:
        log.info("Nothing to roll back (current=%d, target=%d).", current, target_version)
        return []

    rolled: list[int] = []
    for ver in to_rollback:
        mig = _migrations[ver]
        log.info("Rolling back migration v%d: %s", ver, mig["description"])
        try:
            mig["down"](engine)
            _remove_version(engine, ver)
            rolled.append(ver)
            log.info("Rollback v%d completed.", ver)
        except Exception:
            log.exception("Rollback v%d FAILED — stopping.", ver)
            raise

    return rolled


def get_migration_status(engine: Engine | None = None) -> dict:
    """Return a summary of migration state."""
    engine = engine or get_engine()
    _ensure_version_table(engine)
    current = get_current_version(engine)
    latest = max(_migrations.keys()) if _migrations else 0
    pending = sorted(v for v in _migrations if v > current)
    return {
        "current_version": current,
        "latest_available": latest,
        "pending_versions": pending,
        "is_up_to_date": current >= latest,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MIGRATION V1 — Initial schema (creates all ORM tables)
# ═══════════════════════════════════════════════════════════════════════════
def _v1_up(engine: Engine) -> None:
    """Create every table defined in models.py via metadata.create_all."""
    from database.models import Base
    Base.metadata.create_all(bind=engine)
    log.info("V1 UP: All ORM tables created.")


def _v1_down(engine: Engine) -> None:
    """Drop every ORM-managed table (destructive!)."""
    from database.models import Base
    Base.metadata.drop_all(bind=engine)
    log.info("V1 DOWN: All ORM tables dropped.")


register_migration(
    version=1,
    description="Initial schema — create all 13 ORM tables",
    up=_v1_up,
    down=_v1_down,
)


# ═══════════════════════════════════════════════════════════════════════════
# MIGRATION V2 — Add composite indexes for common dashboard queries
# ═══════════════════════════════════════════════════════════════════════════
def _v2_up(engine: Engine) -> None:
    """Add supplemental indexes for dashboard performance."""
    ddl_statements = [
        "CREATE INDEX IF NOT EXISTS ix_bets_sport_settled ON bets (sport, settled_at)",
        "CREATE INDEX IF NOT EXISTS ix_bets_status_pnl ON bets (status, pnl)",
        "CREATE INDEX IF NOT EXISTS ix_signals_edge ON signals (edge_pct)",
        "CREATE INDEX IF NOT EXISTS ix_sg_stats_sg_total ON sg_stats (sg_total)",
        "CREATE INDEX IF NOT EXISTS ix_lm_player_ts ON line_movements (player, timestamp)",
    ]
    with engine.begin() as conn:
        for stmt in ddl_statements:
            conn.execute(text(stmt))
    log.info("V2 UP: Dashboard indexes created.")


def _v2_down(engine: Engine) -> None:
    """Remove the supplemental indexes."""
    drop_statements = [
        "DROP INDEX IF EXISTS ix_bets_sport_settled",
        "DROP INDEX IF EXISTS ix_bets_status_pnl",
        "DROP INDEX IF EXISTS ix_signals_edge",
        "DROP INDEX IF EXISTS ix_sg_stats_sg_total",
        "DROP INDEX IF EXISTS ix_lm_player_ts",
    ]
    with engine.begin() as conn:
        for stmt in drop_statements:
            conn.execute(text(stmt))
    log.info("V2 DOWN: Dashboard indexes dropped.")


register_migration(
    version=2,
    description="Add composite indexes for dashboard query performance",
    up=_v2_up,
    down=_v2_down,
)
