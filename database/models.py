"""
Golf Quant Engine — SQLAlchemy ORM Models
==========================================
All table definitions for the quantitative golf betting platform.
Uses SQLAlchemy 2.0+ Mapped-column style.
"""

import json
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Shared declarative base for every model."""
    pass


# ---------------------------------------------------------------------------
# Helper mixin
# ---------------------------------------------------------------------------
class TimestampMixin:
    """Adds created_at / updated_at columns to any model."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class DictMixin:
    """Generic to_dict() that serialises every mapped column."""

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for col in self.__table__.columns:
            val = getattr(self, col.name)
            if isinstance(val, datetime):
                val = val.isoformat()
            result[col.name] = val
        return result


# ═══════════════════════════════════════════════════════════════════════════
# 1. Players
# ═══════════════════════════════════════════════════════════════════════════
class Player(Base, TimestampMixin, DictMixin):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    team: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, doc="Tour name for golf")
    sport: Mapped[str] = mapped_column(String(50), nullable=False, default="GOLF")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_updated: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # relationships
    sg_stats: Mapped[list["SGStat"]] = relationship("SGStat", back_populates="player", lazy="select")
    tournament_fields: Mapped[list["TournamentField"]] = relationship(
        "TournamentField", back_populates="player", lazy="select"
    )

    __table_args__ = (
        Index("ix_players_sport_active", "sport", "active"),
        Index("ix_players_name_sport", "name", "sport"),
    )

    def __repr__(self) -> str:
        return f"<Player id={self.id} name={self.name!r} sport={self.sport}>"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Events
# ═══════════════════════════════════════════════════════════════════════════
class Event(Base, TimestampMixin, DictMixin):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sport: Mapped[str] = mapped_column(String(50), nullable=False, default="GOLF")
    event_name: Mapped[str] = mapped_column(String(300), nullable=False)
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="scheduled",
        doc="scheduled | live | completed | cancelled",
    )
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    season: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    venue: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    course_name: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    purse: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # relationships
    tournament_fields: Mapped[list["TournamentField"]] = relationship(
        "TournamentField", back_populates="event", lazy="select"
    )
    bets: Mapped[list["Bet"]] = relationship("Bet", back_populates="tournament", lazy="select")
    sg_stats: Mapped[list["SGStat"]] = relationship("SGStat", back_populates="tournament", lazy="select")

    __table_args__ = (
        Index("ix_events_sport_status", "sport", "status"),
        Index("ix_events_start_time", "start_time"),
    )

    def __repr__(self) -> str:
        return f"<Event id={self.id} name={self.event_name!r} status={self.status}>"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Bets
# ═══════════════════════════════════════════════════════════════════════════
class Bet(Base, TimestampMixin, DictMixin):
    __tablename__ = "bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sport: Mapped[str] = mapped_column(String(50), nullable=False, default="GOLF")
    event: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    market: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    player: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    signal_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bet_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    closing_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_outcome: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stake: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, default=func.now())
    model_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    direction: Mapped[Optional[str]] = mapped_column(String(10), nullable=True, doc="OVER / UNDER / WIN / PLACE")
    odds_american: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    odds_decimal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_projection: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_std: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending",
        doc="pending | won | lost | push | void",
    )
    settled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    features_snapshot_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tournament_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("events.id"), nullable=True
    )

    # relationships
    tournament: Mapped[Optional["Event"]] = relationship("Event", back_populates="bets")

    __table_args__ = (
        Index("ix_bets_sport_status", "sport", "status"),
        Index("ix_bets_player", "player"),
        Index("ix_bets_timestamp", "timestamp"),
        Index("ix_bets_tournament_id", "tournament_id"),
        Index("ix_bets_model_version", "model_version"),
    )

    def __repr__(self) -> str:
        return (
            f"<Bet id={self.id} player={self.player!r} market={self.market!r} "
            f"status={self.status}>"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Line Movements
# ═══════════════════════════════════════════════════════════════════════════
class LineMovement(Base, DictMixin):
    __tablename__ = "line_movements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sport: Mapped[str] = mapped_column(String(50), nullable=False, default="GOLF")
    event: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    market: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    book: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    player: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, default=func.now())
    is_opening: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_closing: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("ix_lm_player_market", "player", "market"),
        Index("ix_lm_timestamp", "timestamp"),
        Index("ix_lm_book", "book"),
        Index("ix_lm_event", "event"),
    )

    def __repr__(self) -> str:
        return (
            f"<LineMovement id={self.id} player={self.player!r} "
            f"line={self.line} book={self.book!r}>"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Model Versions
# ═══════════════════════════════════════════════════════════════════════════
class ModelVersion(Base, DictMixin):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    parameters_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    training_data_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    performance_metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sport: Mapped[str] = mapped_column(String(50), nullable=False, default="GOLF")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("ix_mv_sport_active", "sport", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<ModelVersion id={self.id} version={self.version!r} active={self.is_active}>"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Edge Reports
# ═══════════════════════════════════════════════════════════════════════════
class EdgeReport(Base, DictMixin):
    __tablename__ = "edge_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_type: Mapped[str] = mapped_column(String(100), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    report_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sport: Mapped[str] = mapped_column(String(50), nullable=False, default="GOLF")

    __table_args__ = (
        Index("ix_er_sport_type", "sport", "report_type"),
        Index("ix_er_generated_at", "generated_at"),
    )

    def __repr__(self) -> str:
        return f"<EdgeReport id={self.id} type={self.report_type!r}>"


# ═══════════════════════════════════════════════════════════════════════════
# 7. Signals
# ═══════════════════════════════════════════════════════════════════════════
class Signal(Base, DictMixin):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sport: Mapped[str] = mapped_column(String(50), nullable=False, default="GOLF")
    event: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    market: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    player: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    signal_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    direction: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    edge_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    kelly_stake: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("ix_sig_player_market", "player", "market"),
        Index("ix_sig_generated_at", "generated_at"),
        Index("ix_sig_sport_event", "sport", "event"),
    )

    def __repr__(self) -> str:
        return (
            f"<Signal id={self.id} player={self.player!r} "
            f"edge={self.edge_pct} dir={self.direction}>"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 8. User Settings
# ═══════════════════════════════════════════════════════════════════════════
class UserSetting(Base, DictMixin):
    __tablename__ = "user_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, default="default")
    setting_key: Mapped[str] = mapped_column(String(200), nullable=False)
    setting_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("user_id", "setting_key", name="uq_user_setting"),
        Index("ix_us_user_key", "user_id", "setting_key"),
    )

    def __repr__(self) -> str:
        return f"<UserSetting user={self.user_id!r} key={self.setting_key!r}>"


# ═══════════════════════════════════════════════════════════════════════════
# 9. Worker Status
# ═══════════════════════════════════════════════════════════════════════════
class WorkerStatus(Base, DictMixin):
    __tablename__ = "worker_status"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    worker_name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    last_run: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_success: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    next_scheduled_run: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="idle")
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_ws_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<WorkerStatus name={self.worker_name!r} status={self.status}>"


# ═══════════════════════════════════════════════════════════════════════════
# 10. Calibration Snapshots
# ═══════════════════════════════════════════════════════════════════════════
class CalibrationSnapshot(Base, DictMixin):
    __tablename__ = "calibration_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sport: Mapped[str] = mapped_column(String(50), nullable=False, default="GOLF")
    bucket_label: Mapped[str] = mapped_column(String(50), nullable=False)
    prob_lower: Mapped[float] = mapped_column(Float, nullable=False)
    prob_upper: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_avg: Mapped[float] = mapped_column(Float, nullable=False)
    actual_rate: Mapped[float] = mapped_column(Float, nullable=False)
    n_bets: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    calibration_error: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    snapshot_date: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_cs_sport_date", "sport", "snapshot_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<CalibrationSnapshot id={self.id} bucket={self.bucket_label!r} "
            f"n={self.n_bets} err={self.calibration_error:.4f}>"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 11. System State
# ═══════════════════════════════════════════════════════════════════════════
class SystemState(Base, DictMixin):
    __tablename__ = "system_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sport: Mapped[str] = mapped_column(String(50), nullable=False, default="GOLF")
    state: Mapped[str] = mapped_column(
        String(20), nullable=False, default="ACTIVE",
        doc="ACTIVE | REDUCED | SUSPENDED | KILLED",
    )
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    changed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    clv_at_change: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bankroll_at_change: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    drawdown_at_change: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("ix_ss_sport_changed", "sport", "changed_at"),
    )

    def __repr__(self) -> str:
        return f"<SystemState id={self.id} sport={self.sport} state={self.state}>"


# ═══════════════════════════════════════════════════════════════════════════
# 12. Strokes Gained Stats
# ═══════════════════════════════════════════════════════════════════════════
class SGStat(Base, DictMixin):
    __tablename__ = "sg_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False)
    tournament_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("events.id"), nullable=True
    )
    sg_total: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sg_ott: Mapped[Optional[float]] = mapped_column(Float, nullable=True, doc="Off the tee")
    sg_app: Mapped[Optional[float]] = mapped_column(Float, nullable=True, doc="Approach")
    sg_atg: Mapped[Optional[float]] = mapped_column(Float, nullable=True, doc="Around the green")
    sg_putt: Mapped[Optional[float]] = mapped_column(Float, nullable=True, doc="Putting")
    rounds_played: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    season: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    # relationships
    player: Mapped["Player"] = relationship("Player", back_populates="sg_stats")
    tournament: Mapped[Optional["Event"]] = relationship("Event", back_populates="sg_stats")

    __table_args__ = (
        Index("ix_sg_player_season", "player_id", "season"),
        Index("ix_sg_tournament", "tournament_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<SGStat id={self.id} player_id={self.player_id} "
            f"total={self.sg_total} season={self.season}>"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 13. Tournament Field
# ═══════════════════════════════════════════════════════════════════════════
class WeatherData(Base, DictMixin):
    __tablename__ = "weather_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tournament_name: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    course_name: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    forecast_time: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    temp_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_speed_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_gust_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_direction_deg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precipitation_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cloud_cover_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    fetched_at: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    __table_args__ = (
        Index("ix_wd_course_forecast", "course_name", "forecast_time"),
        Index("ix_wd_fetched", "fetched_at"),
    )

    def __repr__(self) -> str:
        return f"<WeatherData id={self.id} course={self.course_name!r}>"


class TournamentField(Base, TimestampMixin, DictMixin):
    __tablename__ = "tournament_field"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tournament_id: Mapped[int] = mapped_column(Integer, ForeignKey("events.id"), nullable=False)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="confirmed")
    tee_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # relationships
    event: Mapped["Event"] = relationship("Event", back_populates="tournament_fields")
    player: Mapped["Player"] = relationship("Player", back_populates="tournament_fields")

    __table_args__ = (
        UniqueConstraint("tournament_id", "player_id", name="uq_field_entry"),
        Index("ix_tf_tournament", "tournament_id"),
        Index("ix_tf_player", "player_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<TournamentField id={self.id} tournament_id={self.tournament_id} "
            f"player_id={self.player_id} status={self.status}>"
        )


# ---------------------------------------------------------------------------
# Convenience: list every model class for programmatic iteration
# ---------------------------------------------------------------------------
ALL_MODELS = [
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
    WeatherData,
    TournamentField,
]
