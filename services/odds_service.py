"""
Odds Service — Odds and line data layer
========================================
All odds reads/writes go through here. Streamlit never queries odds directly.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import and_, func, text

from database.models import LineMovement, Event
from services._db import get_session as _session

log = logging.getLogger(__name__)


def get_current_lines(sport: str = "GOLF") -> list[dict]:
    """
    Return all current (most recent per player+market+book) lines.

    Lines are considered current if recorded within the last 48 hours.
    """
    cutoff = datetime.utcnow() - timedelta(hours=48)

    with _session() as session:
        # Subquery for max timestamp per player+market+book
        sub = (
            session.query(
                LineMovement.player,
                LineMovement.market,
                LineMovement.book,
                func.max(LineMovement.timestamp).label("latest"),
            )
            .filter(
                LineMovement.sport == sport,
                LineMovement.timestamp >= cutoff,
            )
            .group_by(LineMovement.player, LineMovement.market, LineMovement.book)
            .subquery()
        )

        rows = (
            session.query(LineMovement)
            .join(
                sub,
                and_(
                    LineMovement.player == sub.c.player,
                    LineMovement.market == sub.c.market,
                    LineMovement.book == sub.c.book,
                    LineMovement.timestamp == sub.c.latest,
                ),
            )
            .order_by(LineMovement.player, LineMovement.market)
            .all()
        )
        return [r.to_dict() for r in rows]


def get_lines_for_player(
    player_name: str,
    market: Optional[str] = None,
) -> list[dict]:
    """
    Get all lines for a specific player, optionally filtered by market.
    Returns latest per book.
    """
    cutoff = datetime.utcnow() - timedelta(hours=48)

    with _session() as session:
        q = (
            session.query(LineMovement)
            .filter(
                LineMovement.player == player_name,
                LineMovement.timestamp >= cutoff,
            )
        )
        if market:
            q = q.filter(LineMovement.market == market)

        rows = q.order_by(LineMovement.timestamp.desc()).all()

        # Deduplicate: keep latest per book+market
        seen = set()
        results = []
        for r in rows:
            key = (r.book, r.market)
            if key not in seen:
                seen.add(key)
                results.append(r.to_dict())
        return results


def get_line_history(
    player: str,
    market: str,
    book: Optional[str] = None,
    hours: int = 48,
) -> list[dict]:
    """
    Return chronological line history for a player+market, optionally filtered
    to a specific book.
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    with _session() as session:
        q = (
            session.query(LineMovement)
            .filter(
                LineMovement.player == player,
                LineMovement.market == market,
                LineMovement.timestamp >= cutoff,
            )
        )
        if book:
            q = q.filter(LineMovement.book == book)

        rows = q.order_by(LineMovement.timestamp.asc()).all()
        return [r.to_dict() for r in rows]


def get_best_available(player: str, market: str) -> dict:
    """
    Find the best available line for a player+market across all books.

    "Best" = highest line value (for overs) or lowest line value (for unders).
    Returns a single dict with the best line data, or empty dict.
    """
    cutoff = datetime.utcnow() - timedelta(hours=24)

    with _session() as session:
        rows = (
            session.query(LineMovement)
            .filter(
                LineMovement.player == player,
                LineMovement.market == market,
                LineMovement.timestamp >= cutoff,
            )
            .order_by(LineMovement.timestamp.desc())
            .all()
        )

        if not rows:
            return {}

        # Deduplicate by book (latest per book)
        by_book = {}
        for r in rows:
            if r.book not in by_book:
                by_book[r.book] = r

        if not by_book:
            return {}

        # Best line: highest line value (best for under bettors)
        # and lowest line value (best for over bettors).
        # Return both.
        entries = list(by_book.values())
        highest = max(entries, key=lambda r: r.line or 0)
        lowest = min(entries, key=lambda r: r.line or 0)

        return {
            "player": player,
            "market": market,
            "books_available": len(by_book),
            "best_over": lowest.to_dict(),   # lowest line is best for OVER
            "best_under": highest.to_dict(), # highest line is best for UNDER
            "spread": (highest.line or 0) - (lowest.line or 0),
        }


def detect_sharp_movements(
    minutes: int = 60,
    threshold: float = 0.5,
) -> list[dict]:
    """
    Detect line movements that indicate sharp action within the given window.

    A sharp movement is defined as a line moving >= threshold in the window.

    Returns list of dicts with player, market, old_line, new_line, move_size, book.
    """
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)

    with _session() as session:
        # Get lines that have multiple readings in the window
        recent = (
            session.query(LineMovement)
            .filter(LineMovement.timestamp >= cutoff)
            .order_by(LineMovement.player, LineMovement.market, LineMovement.book, LineMovement.timestamp)
            .all()
        )

    # Group by player+market+book, compare first vs last
    groups: dict[tuple, list] = {}
    for r in recent:
        key = (r.player, r.market, r.book)
        groups.setdefault(key, []).append(r)

    movements = []
    for (player, market, book), entries in groups.items():
        if len(entries) < 2:
            continue
        first = entries[0]
        last = entries[-1]
        if first.line is None or last.line is None:
            continue
        move = abs(last.line - first.line)
        if move >= threshold:
            movements.append({
                "player": player,
                "market": market,
                "book": book,
                "old_line": first.line,
                "new_line": last.line,
                "move_size": round(move, 2),
                "direction": "UP" if last.line > first.line else "DOWN",
                "first_seen": first.timestamp.isoformat() if first.timestamp else None,
                "last_seen": last.timestamp.isoformat() if last.timestamp else None,
                "n_updates": len(entries),
            })

    movements.sort(key=lambda x: x["move_size"], reverse=True)
    return movements


def store_lines(lines: list[dict]) -> None:
    """
    Bulk-insert line data into line_movements table.

    Each dict should have: player, market, book, line, odds (optional),
    event (optional), sport (optional, defaults to GOLF).
    """
    if not lines:
        return

    with _session() as session:
        for entry in lines:
            lm = LineMovement(
                sport=entry.get("sport", "GOLF"),
                event=entry.get("event", ""),
                market=entry.get("market", ""),
                book=entry.get("book", ""),
                player=entry.get("player", entry.get("player_name", "")),
                line=entry.get("line", entry.get("line_value")),
                odds=entry.get("odds"),
                timestamp=datetime.utcnow(),
                is_opening=entry.get("is_opening", False),
                is_closing=entry.get("is_closing", False),
            )
            session.add(lm)

    log.info("Stored %d lines", len(lines))


def get_prizepicks_lines() -> list[dict]:
    """
    Return current PrizePicks lines from line_movements table
    (book = 'PrizePicks').
    """
    cutoff = datetime.utcnow() - timedelta(hours=24)

    with _session() as session:
        rows = (
            session.query(LineMovement)
            .filter(
                LineMovement.book == "PrizePicks",
                LineMovement.sport == "GOLF",
                LineMovement.timestamp >= cutoff,
            )
            .order_by(LineMovement.timestamp.desc())
            .all()
        )

        # Deduplicate: latest per player+market
        seen = set()
        results = []
        for r in rows:
            key = (r.player, r.market)
            if key not in seen:
                seen.add(key)
                results.append(r.to_dict())
        return results


def get_tournament_odds() -> list[dict]:
    """
    Return outright winner odds from line_movements table.
    Aggregates across books and returns consensus probabilities.
    """
    cutoff = datetime.utcnow() - timedelta(hours=48)

    with _session() as session:
        rows = (
            session.query(LineMovement)
            .filter(
                LineMovement.sport == "GOLF",
                LineMovement.market.in_(["outrights", "outright", "winner", "to_win"]),
                LineMovement.timestamp >= cutoff,
            )
            .order_by(LineMovement.player, LineMovement.book)
            .all()
        )

    if not rows:
        return []

    # Deduplicate per player+book (latest)
    latest_by_pb: dict[tuple, LineMovement] = {}
    for r in rows:
        key = (r.player, r.book)
        existing = latest_by_pb.get(key)
        if not existing or (r.timestamp and existing.timestamp and r.timestamp > existing.timestamp):
            latest_by_pb[key] = r

    # Aggregate by player
    by_player: dict[str, list] = {}
    for (player, _), lm in latest_by_pb.items():
        by_player.setdefault(player, []).append(lm)

    consensus = []
    for player, entries in by_player.items():
        probs = []
        for e in entries:
            if e.odds and e.odds > 0:
                # odds stored as decimal
                probs.append(1.0 / e.odds)
            elif e.line and e.line > 0:
                probs.append(e.line)

        if probs:
            avg_prob = sum(probs) / len(probs)
        else:
            avg_prob = 0

        consensus.append({
            "player_name": player,
            "consensus_prob": round(avg_prob, 6),
            "num_books": len(entries),
            "min_prob": round(min(probs), 6) if probs else 0,
            "max_prob": round(max(probs), 6) if probs else 0,
        })

    consensus.sort(key=lambda x: x["consensus_prob"], reverse=True)
    return consensus
