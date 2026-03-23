"""SQLAlchemy models for CLV tracking tables."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, Text, Index
from database.models import Base


class OddsSnapshot(Base):
    """Point-in-time odds snapshot from any source."""
    __tablename__ = "odds_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), nullable=False, index=True)
    player = Column(String(128), nullable=False)
    market_type = Column(String(32), nullable=False)  # outright/matchup/top5/etc
    source = Column(String(32), nullable=False)  # pinnacle/draftkings/betmgm/etc
    odds_american = Column(Integer, nullable=True)
    odds_decimal = Column(Float, nullable=True)
    implied_prob = Column(Float, nullable=True)
    line = Column(Float, nullable=True)  # For over/under markets
    captured_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    is_opening = Column(Boolean, default=False)
    is_closing = Column(Boolean, default=False)
    raw_data = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_odds_snap_player_market", "player", "market_type", "source", "captured_at"),
        Index("ix_odds_snap_event", "event_id", "market_type"),
    )


class BetPriceSnapshot(Base):
    """Snapshot of odds at the exact moment a bet was placed."""
    __tablename__ = "bet_price_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bet_id = Column(String(64), nullable=False, unique=True, index=True)
    event_id = Column(String(64), nullable=True)
    player = Column(String(128), nullable=False)
    market_type = Column(String(32), nullable=False)
    # Bet-time prices
    bet_odds_american = Column(Integer, nullable=False)
    bet_odds_decimal = Column(Float, nullable=False)
    bet_implied_prob = Column(Float, nullable=False)
    bet_line = Column(Float, nullable=True)
    # Best available at bet time
    best_odds_decimal = Column(Float, nullable=True)
    best_source = Column(String(32), nullable=True)
    # Closing prices
    closing_odds_american = Column(Integer, nullable=True)
    closing_odds_decimal = Column(Float, nullable=True)
    closing_implied_prob = Column(Float, nullable=True)
    closing_line = Column(Float, nullable=True)
    # CLV
    clv_cents = Column(Float, nullable=True)
    beat_close = Column(Boolean, nullable=True)
    # Timestamps
    bet_placed_at = Column(DateTime, nullable=False)
    closing_captured_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_bet_price_event", "event_id", "market_type"),
    )


class ClosingLineRecord(Base):
    """Closing line for each player/market at tournament start."""
    __tablename__ = "closing_lines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), nullable=False, index=True)
    player = Column(String(128), nullable=False)
    market_type = Column(String(32), nullable=False)
    source = Column(String(32), nullable=False)
    closing_odds_american = Column(Integer, nullable=True)
    closing_odds_decimal = Column(Float, nullable=True)
    closing_implied_prob = Column(Float, nullable=True)
    closing_line = Column(Float, nullable=True)
    first_tee_time = Column(DateTime, nullable=True)
    captured_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_closing_player_market", "player", "market_type", "event_id"),
    )
