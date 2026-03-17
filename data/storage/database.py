"""
Golf Quant Engine — Database Layer
SQLite via SQLAlchemy. Stores: players, tournaments, SG stats, projections, bets, audit logs.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    Boolean, DateTime, Text, ForeignKey, Index, event
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

from config.settings import DB_PATH

log = logging.getLogger(__name__)
Base = declarative_base()

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

class Player(Base):
    __tablename__ = "players"
    id              = Column(Integer, primary_key=True)
    pga_player_id   = Column(String, unique=True, index=True)
    name            = Column(String, nullable=False, index=True)
    country         = Column(String)
    world_rank      = Column(Integer)
    owgr_rank       = Column(Integer)
    datagolf_rank   = Column(Integer)   # future DataGolf integration
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    sg_stats        = relationship("SGStats", back_populates="player")
    tournament_results = relationship("TournamentResult", back_populates="player")
    bets            = relationship("Bet", back_populates="player")


class Tournament(Base):
    __tablename__ = "tournaments"
    id              = Column(Integer, primary_key=True)
    pga_event_id    = Column(String, unique=True, index=True)
    name            = Column(String, nullable=False)
    course_name     = Column(String)
    start_date      = Column(DateTime)
    end_date        = Column(DateTime)
    purse           = Column(Float)
    field_size      = Column(Integer)
    tour            = Column(String, default="PGA")     # PGA, DP World, LIV, etc.
    created_at      = Column(DateTime, default=datetime.utcnow)

    results         = relationship("TournamentResult", back_populates="tournament")


class SGStats(Base):
    """
    Strokes Gained stats per player per tournament.
    One row = one player's performance in one tournament.
    """
    __tablename__ = "sg_stats"
    __table_args__ = (
        Index("ix_sg_player_tournament", "player_id", "tournament_id"),
    )
    id              = Column(Integer, primary_key=True)
    player_id       = Column(Integer, ForeignKey("players.id"), nullable=False)
    tournament_id   = Column(Integer, ForeignKey("tournaments.id"))
    tournament_name = Column(String)    # denormalized for quick queries
    course_name     = Column(String)
    event_date      = Column(DateTime, index=True)
    season          = Column(Integer)

    # Core SG Categories
    sg_total        = Column(Float)
    sg_ott          = Column(Float)     # off the tee
    sg_app          = Column(Float)     # approach
    sg_atg          = Column(Float)     # around the green
    sg_putt         = Column(Float)     # putting
    sg_t2g          = Column(Float)     # tee to green (ott + app + atg)

    # Supplementary stats
    driving_distance = Column(Float)
    driving_accuracy = Column(Float)
    gir             = Column(Float)     # greens in regulation %
    scrambling      = Column(Float)
    proximity_175_200 = Column(Float)   # approach proximity 175-200 yards
    proximity_150_175 = Column(Float)
    birdie_avg      = Column(Float)
    bogey_avoid     = Column(Float)
    score_avg       = Column(Float)
    finish_position = Column(Integer)
    made_cut        = Column(Boolean)

    # DataGolf supplementary (populated when API key available)
    dg_sg_app_adjusted = Column(Float)  # course-adjusted approach SG
    dg_course_fit_score = Column(Float)

    player          = relationship("Player", back_populates="sg_stats")


class TournamentResult(Base):
    __tablename__ = "tournament_results"
    id              = Column(Integer, primary_key=True)
    player_id       = Column(Integer, ForeignKey("players.id"))
    tournament_id   = Column(Integer, ForeignKey("tournaments.id"))
    finish_position = Column(Integer)
    finish_str      = Column(String)    # "T3", "MC", "WD", "DQ"
    score_total     = Column(Integer)
    score_r1        = Column(Integer)
    score_r2        = Column(Integer)
    score_r3        = Column(Integer)
    score_r4        = Column(Integer)
    earnings        = Column(Float)
    fedex_points    = Column(Float)
    made_cut        = Column(Boolean)
    created_at      = Column(DateTime, default=datetime.utcnow)

    player          = relationship("Player", back_populates="tournament_results")
    tournament      = relationship("Tournament", back_populates="results")


class Projection(Base):
    """Model projections for a specific player in a specific tournament."""
    __tablename__ = "projections"
    id              = Column(Integer, primary_key=True)
    player_id       = Column(Integer, ForeignKey("players.id"))
    tournament_id   = Column(Integer, ForeignKey("tournaments.id"))
    generated_at    = Column(DateTime, default=datetime.utcnow)
    model_version   = Column(String, default="1.0")

    # Projected SG values (course-adjusted)
    proj_sg_total   = Column(Float)
    proj_sg_ott     = Column(Float)
    proj_sg_app     = Column(Float)
    proj_sg_atg     = Column(Float)
    proj_sg_putt    = Column(Float)

    # Course fit score (0-100, higher = better fit)
    course_fit_score = Column(Float)

    # Win probabilities
    win_prob        = Column(Float)
    top5_prob       = Column(Float)
    top10_prob      = Column(Float)
    top20_prob      = Column(Float)
    make_cut_prob   = Column(Float)

    # DFS metrics
    dk_salary       = Column(Integer)
    fd_salary       = Column(Integer)
    dk_proj_pts     = Column(Float)
    fd_proj_pts     = Column(Float)
    dk_value        = Column(Float)     # proj_pts / (salary / 1000)
    fd_value        = Column(Float)
    proj_ownership  = Column(Float)     # projected public ownership %
    leverage_score  = Column(Float)     # our proj ownership vs public

    # Form signal
    form_trend      = Column(String)    # "improving", "declining", "stable"
    last4_avg_sg    = Column(Float)
    last12_avg_sg   = Column(Float)

    # Weather adjustment
    weather_adj     = Column(Float)     # bonus/penalty from tee time + conditions
    tee_time_adv    = Column(Float)     # advantage of their draw

    # Raw JSON of full projection data
    raw_data        = Column(Text)

    def set_raw(self, data: dict):
        self.raw_data = json.dumps(data)

    def get_raw(self) -> dict:
        return json.loads(self.raw_data) if self.raw_data else {}


class Bet(Base):
    """Every bet placed — core of the audit system."""
    __tablename__ = "bets"
    id              = Column(Integer, primary_key=True)
    placed_at       = Column(DateTime, default=datetime.utcnow)
    tournament_id   = Column(Integer, ForeignKey("tournaments.id"))
    player_id       = Column(Integer, ForeignKey("players.id"))

    # Bet details
    bet_type        = Column(String)    # "outright", "h2h", "top5", "top10", "top20", "make_cut", "dfs_gpp", "dfs_cash"
    book            = Column(String)    # "DraftKings", "FanDuel", "Pinnacle", etc.
    market          = Column(String)    # specific market description
    odds_american   = Column(Integer)
    odds_decimal    = Column(Float)
    stake           = Column(Float)     # dollars wagered
    bankroll_at_time = Column(Float)    # bankroll when bet placed
    pct_bankroll    = Column(Float)     # stake / bankroll

    # Model edge
    model_win_prob  = Column(Float)
    implied_prob    = Column(Float)
    edge_pct        = Column(Float)     # model_prob - implied_prob
    kelly_fraction  = Column(Float)
    kelly_rec_stake = Column(Float)

    # Result
    settled         = Column(Boolean, default=False)
    won             = Column(Boolean)
    profit_loss     = Column(Float)
    settlement_date = Column(DateTime)
    notes           = Column(Text)

    player          = relationship("Player", back_populates="bets")


class DFSLineup(Base):
    """Saved DFS lineups."""
    __tablename__ = "dfs_lineups"
    id              = Column(Integer, primary_key=True)
    tournament_id   = Column(Integer, ForeignKey("tournaments.id"))
    created_at      = Column(DateTime, default=datetime.utcnow)
    platform        = Column(String)    # "DraftKings", "FanDuel"
    contest_type    = Column(String)    # "GPP", "Cash"
    players_json    = Column(Text)      # JSON list of player names
    total_salary    = Column(Integer)
    proj_total_pts  = Column(Float)
    total_ownership = Column(Float)
    actual_pts      = Column(Float)     # filled in after tournament
    cash_result     = Column(Boolean)
    profit_loss     = Column(Float)

    def get_players(self) -> list:
        return json.loads(self.players_json) if self.players_json else []


class AuditLog(Base):
    """Free-form audit events — model updates, data refreshes, anomalies."""
    __tablename__ = "audit_logs"
    id          = Column(Integer, primary_key=True)
    timestamp   = Column(DateTime, default=datetime.utcnow)
    event_type  = Column(String, index=True)    # "data_refresh", "projection_run", "bet_placed", "edge_alert"
    description = Column(Text)
    data_json   = Column(Text)


# ─────────────────────────────────────────────
# DATABASE ENGINE
# ─────────────────────────────────────────────

_engine = None
_SessionLocal = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            f"sqlite:///{DB_PATH}",
            connect_args={"check_same_thread": False},
            echo=False
        )
        # Enable WAL mode for better concurrent access
        @event.listens_for(_engine, "connect")
        def set_wal(dbapi_conn, connection_record):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(_engine)
        log.info(f"Database initialized at {DB_PATH}")
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=True, autocommit=False)
    return _SessionLocal()


def init_db():
    """Initialize database and create all tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    log.info("All tables created.")
    return engine
