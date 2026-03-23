"""Unified database models — Single source of truth for ALL tables.

Consolidates three previous database systems:
  - database/db_manager.py (raw sqlite3) → REPLACED
  - data/storage/database.py (SQLAlchemy ORM) → REPLACED
  - quant_system/db/schema.py (SQLAlchemy ORM) → KEPT for backward compat, mirrored here

Section 9 required tables: bets, line_movements, model_versions, edge_reports,
signals, players, events — plus all sport-specific and quant tables.
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, Text,
    Index, ForeignKey, UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ═══════════════════════════════════════════════════════════════════════════
# 1. PLAYERS — Unified player registry
# ═══════════════════════════════════════════════════════════════════════════

class Player(Base):
    """Every player ever tracked. Single registry across all sources."""
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False, index=True)
    normalized_name = Column(String(128), nullable=False, unique=True, index=True)
    sport = Column(String(10), nullable=False, index=True)  # "golf"
    # Golf-specific IDs
    pga_player_id = Column(String(32), nullable=True)
    datagolf_id = Column(String(32), nullable=True)
    espn_id = Column(String(32), nullable=True)
    # Rankings
    world_rank = Column(Integer, nullable=True)
    owgr_rank = Column(Integer, nullable=True)
    datagolf_rank = Column(Integer, nullable=True)
    fedex_rank = Column(Integer, nullable=True)
    # Metadata
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    sg_stats = relationship("SGStats", back_populates="player")
    tournament_results = relationship("TournamentResult", back_populates="player")
    projections = relationship("Projection", back_populates="player")

    __table_args__ = (
        Index("ix_player_sport_name", "sport", "normalized_name"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. EVENTS — Tournaments (golf) / Games (NBA)
# ═══════════════════════════════════════════════════════════════════════════

class Event(Base):
    """Tournament or game event."""
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(10), nullable=False, index=True)
    event_id = Column(String(64), nullable=False, unique=True, index=True)
    name = Column(String(256), nullable=False)
    # Golf: course info
    course_name = Column(String(256), nullable=True)
    course_par = Column(Integer, nullable=True)
    course_yardage = Column(Integer, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    # Dates
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    # Tournament details
    purse = Column(Float, nullable=True)
    field_size = Column(Integer, nullable=True)
    tour = Column(String(32), nullable=True)  # "PGA", "LIV", "DP World"
    status = Column(String(32), default="scheduled")  # scheduled/in_progress/completed/cancelled
    season = Column(Integer, nullable=True)
    # Metadata
    raw_data = Column(Text, nullable=True)  # JSON blob for extra fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    tournament_results = relationship("TournamentResult", back_populates="event")
    projections = relationship("Projection", back_populates="event")
    weather = relationship("WeatherData", back_populates="event")

    __table_args__ = (
        Index("ix_event_sport_status", "sport", "status"),
        Index("ix_event_dates", "start_date", "end_date"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. BETS — Every bet ever placed (immutable after creation, except settlement)
# ═══════════════════════════════════════════════════════════════════════════

class Bet(Base):
    """Unified bet log. Replaces bet_log, bet_tracker, and bets tables."""
    __tablename__ = "bets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bet_id = Column(String(64), unique=True, nullable=False, index=True)
    sport = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    # What
    player = Column(String(128), nullable=False, index=True)
    event_id = Column(String(64), nullable=True)  # FK to events.event_id
    bet_type = Column(String(32), nullable=False)  # over/under/outright/h2h/top5/etc
    stat_type = Column(String(64), nullable=False)  # points/birdies/fantasy_score/etc
    line = Column(Float, nullable=False)
    direction = Column(String(10), nullable=False)
    # Pricing
    model_prob = Column(Float, nullable=False)
    market_prob = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)
    odds_american = Column(Integer, nullable=False)
    odds_decimal = Column(Float, nullable=False)
    # Sizing
    stake = Column(Float, nullable=False)
    kelly_fraction = Column(Float, nullable=False)
    # Model context
    model_projection = Column(Float, nullable=False)
    model_std = Column(Float, nullable=False)
    confidence_score = Column(Float, default=0.0)
    engine_agreement = Column(Float, default=0.0)
    model_version_id = Column(Integer, nullable=True)  # FK to model_versions.id
    # Settlement
    status = Column(String(16), default="pending", index=True)  # pending/won/lost/push/void
    actual_result = Column(Float, nullable=True)
    closing_line = Column(Float, nullable=True)
    closing_odds = Column(Integer, nullable=True)
    settled_at = Column(DateTime, nullable=True)
    pnl = Column(Float, default=0.0)
    # Book info
    book = Column(String(32), nullable=True)  # prizepicks/draftkings/fanduel/etc
    # Metadata
    features_snapshot = Column(Text, default="")  # JSON blob
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_bet_sport_status", "sport", "status"),
        Index("ix_bet_sport_timestamp", "sport", "timestamp"),
        Index("ix_bet_player_stat", "player", "stat_type"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. LINE_MOVEMENTS — All line/odds data from every source
# ═══════════════════════════════════════════════════════════════════════════

class LineMovement(Base):
    """Time-series of line movements. Consolidates line_snapshots + prizepicks_lines + odds."""
    __tablename__ = "line_movements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(10), nullable=False)
    player = Column(String(128), nullable=False)
    stat_type = Column(String(64), nullable=False)
    source = Column(String(32), nullable=False)  # prizepicks/draftkings/fanduel/pinnacle/etc
    event_id = Column(String(64), nullable=True)
    # Line data
    line = Column(Float, nullable=False)
    odds_american = Column(Integer, nullable=True)
    odds_decimal = Column(Float, nullable=True)
    over_prob_implied = Column(Float, nullable=True)
    under_prob_implied = Column(Float, nullable=True)
    # PrizePicks specific
    odds_type = Column(String(16), nullable=True)  # standard/demon/goblin
    is_flash_sale = Column(Boolean, default=False)
    discount_pct = Column(Float, nullable=True)
    # Timing
    captured_at = Column(DateTime, nullable=False, index=True)
    is_opening = Column(Boolean, default=False)
    is_closing = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    # Consensus
    consensus_prob = Column(Float, nullable=True)  # Aggregated probability across books
    # Metadata
    raw_data = Column(Text, nullable=True)  # JSON for extra fields

    __table_args__ = (
        Index("ix_line_player_stat", "player", "stat_type", "captured_at"),
        Index("ix_line_sport_source", "sport", "source", "captured_at"),
        Index("ix_line_active", "sport", "is_active"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. MODEL_VERSIONS — Track every model retrain
# ═══════════════════════════════════════════════════════════════════════════

class ModelVersion(Base):
    """Model version registry. Tracks retrains, performance, and drift."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(10), nullable=False, index=True)
    version = Column(String(32), nullable=False)
    trained_at = Column(DateTime, nullable=False)
    # Training data
    training_start = Column(DateTime, nullable=True)
    training_end = Column(DateTime, nullable=True)
    n_training_samples = Column(Integer, nullable=True)
    # Performance metrics at train time
    train_accuracy = Column(Float, nullable=True)
    train_log_loss = Column(Float, nullable=True)
    train_brier_score = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    val_log_loss = Column(Float, nullable=True)
    val_brier_score = Column(Float, nullable=True)
    # Live performance (updated as bets settle)
    live_accuracy = Column(Float, nullable=True)
    live_log_loss = Column(Float, nullable=True)
    live_clv_avg = Column(Float, nullable=True)
    live_roi = Column(Float, nullable=True)
    n_live_bets = Column(Integer, default=0)
    # Drift detection
    psi_score = Column(Float, nullable=True)  # Population Stability Index
    feature_drift_score = Column(Float, nullable=True)
    is_degraded = Column(Boolean, default=False)
    # Status
    is_active = Column(Boolean, default=True)
    replaced_by = Column(Integer, nullable=True)  # id of replacement model
    retired_at = Column(DateTime, nullable=True)
    # Config
    hyperparameters = Column(Text, nullable=True)  # JSON
    feature_list = Column(Text, nullable=True)  # JSON array of feature names
    notes = Column(Text, default="")

    __table_args__ = (
        Index("ix_model_sport_active", "sport", "is_active"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6. EDGE_REPORTS — Daily edge validation snapshots
# ═══════════════════════════════════════════════════════════════════════════

class EdgeReport(Base):
    """Daily edge validation report. One row per sport per day."""
    __tablename__ = "edge_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(10), nullable=False)
    report_date = Column(DateTime, nullable=False)
    # CLV metrics
    clv_last_50 = Column(Float, default=0.0)
    clv_last_100 = Column(Float, default=0.0)
    clv_last_250 = Column(Float, default=0.0)
    clv_last_500 = Column(Float, default=0.0)
    clv_trend = Column(String(16), nullable=True)  # improving/stable/declining
    # Calibration
    calibration_error = Column(Float, default=0.0)
    calibration_buckets = Column(Text, nullable=True)  # JSON
    overconfidence_ratio = Column(Float, nullable=True)
    # Performance
    model_roi = Column(Float, default=0.0)
    expected_roi = Column(Float, default=0.0)
    actual_win_rate = Column(Float, nullable=True)
    expected_win_rate = Column(Float, nullable=True)
    # Bankroll
    bankroll = Column(Float, nullable=True)
    peak_bankroll = Column(Float, nullable=True)
    drawdown_pct = Column(Float, nullable=True)
    # Verdict
    edge_exists = Column(Boolean, nullable=False)
    system_state = Column(String(16), nullable=False)  # active/reduced/suspended/killed
    previous_state = Column(String(16), nullable=True)
    # Details
    warnings = Column(Text, nullable=True)  # JSON array
    actions = Column(Text, nullable=True)  # JSON array
    model_version_id = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_edge_sport_date", "sport", "report_date"),
        UniqueConstraint("sport", "report_date", name="uq_edge_sport_date"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7. SIGNALS — Generated betting signals (pre-bet evaluation)
# ═══════════════════════════════════════════════════════════════════════════

class Signal(Base):
    """Betting signal generated by the model. May or may not become a bet."""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(10), nullable=False, index=True)
    generated_at = Column(DateTime, nullable=False, index=True)
    event_id = Column(String(64), nullable=True)
    # Signal details
    player = Column(String(128), nullable=False)
    stat_type = Column(String(64), nullable=False)
    line = Column(Float, nullable=False)
    direction = Column(String(10), nullable=False)
    source = Column(String(32), nullable=False)  # prizepicks/draftkings/etc
    # Model output
    model_prob = Column(Float, nullable=False)
    market_prob = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)
    model_projection = Column(Float, nullable=True)
    model_std = Column(Float, nullable=True)
    confidence_score = Column(Float, default=0.0)
    engine_agreement = Column(Float, default=0.0)
    # Evaluation
    approved = Column(Boolean, nullable=True)  # NULL = not evaluated yet
    rejection_reason = Column(String(256), nullable=True)
    recommended_stake = Column(Float, nullable=True)
    # Linkage
    bet_id = Column(String(64), nullable=True)  # If signal became a bet
    model_version_id = Column(Integer, nullable=True)
    # Sharp money context
    sharp_agrees = Column(Boolean, nullable=True)
    sharp_confidence = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_signal_sport_date", "sport", "generated_at"),
        Index("ix_signal_player", "player", "stat_type"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 8. CLV_LOG — Closing Line Value per settled bet
# ═══════════════════════════════════════════════════════════════════════════

class CLVLog(Base):
    """CLV measurement for every settled bet."""
    __tablename__ = "clv_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bet_id = Column(String(64), nullable=False, index=True)
    sport = Column(String(10), nullable=False)
    opening_line = Column(Float, nullable=False)
    bet_line = Column(Float, nullable=False)
    closing_line = Column(Float, nullable=False)
    line_movement = Column(Float, nullable=False)
    clv_raw = Column(Float, nullable=False)  # closing_prob - bet_prob
    clv_cents = Column(Float, nullable=False)  # CLV in cents/dollar
    beat_close = Column(Boolean, nullable=False)
    calculated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_clv_sport", "sport", "calculated_at"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 9. CALIBRATION_LOG — Periodic calibration snapshots
# ═══════════════════════════════════════════════════════════════════════════

class CalibrationLog(Base):
    """Periodic calibration snapshots."""
    __tablename__ = "calibration_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(10), nullable=False)
    report_date = Column(DateTime, nullable=False)
    bucket_label = Column(String(16), nullable=False)
    prob_lower = Column(Float, nullable=False)
    prob_upper = Column(Float, nullable=False)
    predicted_avg = Column(Float, nullable=False)
    actual_rate = Column(Float, nullable=False)
    n_bets = Column(Integer, nullable=False)
    calibration_error = Column(Float, nullable=False)
    is_overconfident = Column(Boolean, nullable=False)

    __table_args__ = (
        Index("ix_cal_sport_date", "sport", "report_date"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 10. SYSTEM_STATE_LOG — Audit trail of state changes
# ═══════════════════════════════════════════════════════════════════════════

class SystemStateLog(Base):
    """Audit trail of system state changes."""
    __tablename__ = "system_state_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    previous_state = Column(String(16), nullable=False)
    new_state = Column(String(16), nullable=False)
    reason = Column(Text, nullable=False)
    clv_at_change = Column(Float, nullable=True)
    bankroll_at_change = Column(Float, nullable=True)
    drawdown_at_change = Column(Float, nullable=True)


# ═══════════════════════════════════════════════════════════════════════════
# 11. FEATURE_LOG — Feature importance drift tracking
# ═══════════════════════════════════════════════════════════════════════════

class FeatureLog(Base):
    """Track feature importance over time for drift detection."""
    __tablename__ = "feature_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(10), nullable=False)
    report_date = Column(DateTime, nullable=False)
    feature_name = Column(String(64), nullable=False)
    importance_score = Column(Float, nullable=False)
    directional_accuracy = Column(Float, nullable=True)
    n_samples = Column(Integer, nullable=False)
    is_degraded = Column(Boolean, default=False)

    __table_args__ = (
        Index("ix_feature_sport_date", "sport", "report_date"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 12. SCRAPER_STATUS — Monitor all scrapers
# ═══════════════════════════════════════════════════════════════════════════

class ScraperStatus(Base):
    """Per-scraper health monitoring."""
    __tablename__ = "scraper_status"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scraper_name = Column(String(64), nullable=False, unique=True, index=True)
    sport = Column(String(10), nullable=False)
    last_success = Column(DateTime, nullable=True)
    last_failure = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    consecutive_failures = Column(Integer, default=0)
    total_runs = Column(Integer, default=0)
    total_failures = Column(Integer, default=0)
    lines_last_scrape = Column(Integer, default=0)
    avg_duration_seconds = Column(Float, nullable=True)
    is_healthy = Column(Boolean, default=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════════════
# 13. AUDIT_LOGS — System-wide event logging
# ═══════════════════════════════════════════════════════════════════════════

class AuditLog(Base):
    """Free-form event logging for all system operations."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    sport = Column(String(10), nullable=True)
    category = Column(String(32), nullable=False, index=True)  # scraper/model/edge/bet/system
    level = Column(String(16), default="info")  # info/warning/error/critical
    message = Column(Text, nullable=False)
    details = Column(Text, nullable=True)  # JSON blob


# ═══════════════════════════════════════════════════════════════════════════
# GOLF-SPECIFIC TABLES
# ═══════════════════════════════════════════════════════════════════════════

class SGStats(Base):
    """Strokes Gained statistics per player per tournament."""
    __tablename__ = "sg_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False, index=True)
    event_id = Column(String(64), nullable=True)
    season = Column(Integer, nullable=True)
    # Strokes Gained breakdown
    sg_total = Column(Float, nullable=True)
    sg_ott = Column(Float, nullable=True)  # Off the tee
    sg_app = Column(Float, nullable=True)  # Approach
    sg_atg = Column(Float, nullable=True)  # Around the green
    sg_putt = Column(Float, nullable=True)  # Putting
    sg_t2g = Column(Float, nullable=True)  # Tee to green
    # Traditional stats
    driving_distance = Column(Float, nullable=True)
    driving_accuracy = Column(Float, nullable=True)
    gir_pct = Column(Float, nullable=True)
    scrambling_pct = Column(Float, nullable=True)
    proximity_to_hole = Column(Float, nullable=True)
    birdie_avg = Column(Float, nullable=True)
    bogey_avg = Column(Float, nullable=True)
    scoring_avg = Column(Float, nullable=True)
    # Source
    source = Column(String(32), default="pga_tour")
    captured_at = Column(DateTime, default=datetime.utcnow)

    player = relationship("Player", back_populates="sg_stats")

    __table_args__ = (
        Index("ix_sg_player_event", "player_id", "event_id"),
    )


class TournamentResult(Base):
    """Player results per tournament."""
    __tablename__ = "tournament_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False, index=True)
    event_id = Column(String(64), ForeignKey("events.event_id"), nullable=False, index=True)
    # Results
    finish_position = Column(Integer, nullable=True)
    finish_position_str = Column(String(8), nullable=True)  # "T5", "CUT", etc
    total_score = Column(Integer, nullable=True)
    total_to_par = Column(Integer, nullable=True)
    round_scores = Column(Text, nullable=True)  # JSON array [68, 72, 69, 71]
    made_cut = Column(Boolean, nullable=True)
    # Earnings
    earnings = Column(Float, default=0.0)
    fedex_points = Column(Float, default=0.0)
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    player = relationship("Player", back_populates="tournament_results")
    event = relationship("Event", back_populates="tournament_results")

    __table_args__ = (
        UniqueConstraint("player_id", "event_id", name="uq_result_player_event"),
    )


class Projection(Base):
    """Course-fitted projections per player per event."""
    __tablename__ = "projections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False, index=True)
    event_id = Column(String(64), ForeignKey("events.event_id"), nullable=False, index=True)
    model_version_id = Column(Integer, nullable=True)
    # SG projections (course-fitted)
    projected_sg_total = Column(Float, nullable=True)
    projected_sg_ott = Column(Float, nullable=True)
    projected_sg_app = Column(Float, nullable=True)
    projected_sg_atg = Column(Float, nullable=True)
    projected_sg_putt = Column(Float, nullable=True)
    # Outcome probabilities
    win_prob = Column(Float, nullable=True)
    top5_prob = Column(Float, nullable=True)
    top10_prob = Column(Float, nullable=True)
    top20_prob = Column(Float, nullable=True)
    make_cut_prob = Column(Float, nullable=True)
    # DFS
    projected_dfs_points = Column(Float, nullable=True)
    dfs_ownership = Column(Float, nullable=True)
    dfs_value = Column(Float, nullable=True)
    # Context
    form_trend = Column(Float, nullable=True)  # -1 to +1
    course_fit_score = Column(Float, nullable=True)
    weather_adjustment = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    edge_pct = Column(Float, nullable=True)
    # Raw data
    raw_data = Column(Text, nullable=True)  # JSON blob
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    player = relationship("Player", back_populates="projections")
    event = relationship("Event", back_populates="projections")

    __table_args__ = (
        Index("ix_proj_player_event", "player_id", "event_id"),
    )


class WeatherData(Base):
    """Course weather forecasts per event."""
    __tablename__ = "weather_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), ForeignKey("events.event_id"), nullable=False, index=True)
    forecast_date = Column(DateTime, nullable=False)
    round_num = Column(Integer, nullable=True)
    # Weather
    temp_high = Column(Float, nullable=True)
    temp_low = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)
    wind_gust = Column(Float, nullable=True)
    wind_direction = Column(String(8), nullable=True)
    humidity = Column(Float, nullable=True)
    precip_chance = Column(Float, nullable=True)
    precip_amount = Column(Float, nullable=True)
    conditions = Column(String(64), nullable=True)  # sunny/cloudy/rain/etc
    # Difficulty adjustment
    scoring_difficulty = Column(Float, nullable=True)  # Model's weather impact score
    # Metadata
    source = Column(String(32), default="openweather")
    captured_at = Column(DateTime, default=datetime.utcnow)

    event = relationship("Event", back_populates="weather")


class DFSLineup(Base):
    """Saved DFS lineups with performance tracking."""
    __tablename__ = "dfs_lineups"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), nullable=True)
    platform = Column(String(32), nullable=False)  # draftkings/fanduel
    contest_type = Column(String(32), nullable=True)  # gpp/cash/etc
    # Lineup
    players_json = Column(Text, nullable=False)  # JSON array of {player, salary, projected}
    total_salary = Column(Integer, nullable=True)
    salary_remaining = Column(Integer, nullable=True)
    projected_points = Column(Float, nullable=True)
    # Results
    actual_points = Column(Float, nullable=True)
    finish_position = Column(Integer, nullable=True)
    entry_fee = Column(Float, nullable=True)
    winnings = Column(Float, nullable=True)
    roi = Column(Float, nullable=True)
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════════════
# WORKER STATE — Track background worker status
# ═══════════════════════════════════════════════════════════════════════════

class WorkerState(Base):
    """Track background worker execution state. Workers update this on every run."""
    __tablename__ = "worker_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    worker_name = Column(String(64), nullable=False, unique=True, index=True)
    sport = Column(String(10), nullable=True)
    last_run = Column(DateTime, nullable=True)
    next_scheduled = Column(DateTime, nullable=True)
    status = Column(String(16), default="idle")  # idle/running/error/disabled
    last_error = Column(Text, nullable=True)
    run_count = Column(Integer, default=0)
    avg_duration_seconds = Column(Float, nullable=True)
    config_json = Column(Text, nullable=True)  # JSON worker config
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════════════
# USER SETTINGS — Persist user preferences (replaces session_state)
# ═══════════════════════════════════════════════════════════════════════════

class UserSetting(Base):
    """Key-value store for user settings. Replaces st.session_state for persistence."""
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(128), nullable=False, unique=True, index=True)
    value = Column(Text, nullable=True)  # JSON-encoded value
    category = Column(String(32), default="general")  # general/display/risk/notifications
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def decoded_value(self):
        if self.value is None:
            return None
        try:
            return json.loads(self.value)
        except (json.JSONDecodeError, TypeError):
            return self.value

    @decoded_value.setter
    def decoded_value(self, val):
        self.value = json.dumps(val)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Name normalization
# ═══════════════════════════════════════════════════════════════════════════

def normalize_player_name(name: str) -> str:
    """Standardize player names across sources.

    'Scottie Scheffler' == 'SCOTTIE SCHEFFLER' == 'scheffler, scottie'
    """
    import re
    name = name.strip().lower()
    # Handle "Last, First" format
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        name = f"{parts[1]} {parts[0]}" if len(parts) == 2 else name
    # Remove suffixes
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", name, flags=re.IGNORECASE)
    # Normalize whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name
