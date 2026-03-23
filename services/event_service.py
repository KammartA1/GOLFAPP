"""
Event Service — Tournament / event management layer
=====================================================
All tournament CRUD and field management goes through here.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func

from database.models import Event, TournamentField, Player
from services._db import get_session as _session

log = logging.getLogger(__name__)


def get_current_tournament() -> dict:
    """
    Return the most relevant current tournament.

    Priority:
      1. Events with status 'live'
      2. Events with status 'scheduled' and start_time within +-7 days
      3. Most recently created event

    Returns event dict or empty dict.
    """
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    week_ahead = now + timedelta(days=7)

    with _session() as session:
        # 1. Live events
        live = (
            session.query(Event)
            .filter(Event.sport == "GOLF", Event.status == "live")
            .order_by(Event.start_time.desc())
            .first()
        )
        if live:
            return _enrich_event(session, live)

        # 2. Scheduled events within the next week
        upcoming = (
            session.query(Event)
            .filter(
                Event.sport == "GOLF",
                Event.status == "scheduled",
                Event.start_time >= week_ago,
                Event.start_time <= week_ahead,
            )
            .order_by(Event.start_time.asc())
            .first()
        )
        if upcoming:
            return _enrich_event(session, upcoming)

        # 3. Fallback: most recent
        latest = (
            session.query(Event)
            .filter(Event.sport == "GOLF")
            .order_by(Event.created_at.desc())
            .first()
        )
        if latest:
            return _enrich_event(session, latest)

    return {}


def get_upcoming_tournaments(weeks: int = 4) -> list[dict]:
    """
    Return scheduled events within the next N weeks, sorted by start time.
    """
    now = datetime.utcnow()
    future = now + timedelta(weeks=weeks)

    with _session() as session:
        rows = (
            session.query(Event)
            .filter(
                Event.sport == "GOLF",
                Event.status.in_(["scheduled", "live"]),
                Event.start_time >= now,
                Event.start_time <= future,
            )
            .order_by(Event.start_time.asc())
            .all()
        )
        return [_enrich_event(session, e) for e in rows]


def get_or_create_event(
    event_name: str,
    start_time: datetime,
    course: Optional[str] = None,
    venue: Optional[str] = None,
) -> dict:
    """
    Look up an event by name (case-insensitive). Create if not found.

    Returns the event record as a dict.
    """
    normalized = event_name.strip()
    if not normalized:
        return {"error": "Empty event name"}

    with _session() as session:
        existing = (
            session.query(Event)
            .filter(
                func.lower(Event.event_name) == normalized.lower(),
                Event.sport == "GOLF",
            )
            .first()
        )

        if existing:
            # Update fields if provided
            if course and not existing.course_name:
                existing.course_name = course
            if venue and not existing.venue:
                existing.venue = venue
            if start_time and not existing.start_time:
                existing.start_time = start_time
            return existing.to_dict()

        # Create
        event = Event(
            sport="GOLF",
            event_name=normalized,
            start_time=start_time,
            course_name=course or "",
            venue=venue or "",
            status="scheduled",
        )
        session.add(event)
        session.flush()
        result = event.to_dict()

    log.info("Created event: %s (start=%s)", normalized, start_time)
    return result


def update_event_status(event_id: int, status: str) -> None:
    """
    Update an event's status.

    Valid statuses: scheduled, live, completed, cancelled.
    """
    valid = {"scheduled", "live", "completed", "cancelled"}
    if status not in valid:
        log.warning("Invalid status '%s'; must be one of %s", status, valid)
        return

    with _session() as session:
        event = session.get(Event, event_id)
        if not event:
            log.warning("Event %d not found", event_id)
            return

        old_status = event.status
        event.status = status
        log.info("Event %d status: %s -> %s", event_id, old_status, status)


def get_tournament_results(tournament_id: int) -> list[dict]:
    """
    Return final results for a completed tournament.

    Pulls from the tournament_field table, enriched with player SG stats
    for that event. If the event has associated SGStat records with
    matching tournament_id those are included.
    """
    from database.models import SGStat

    with _session() as session:
        event = session.get(Event, tournament_id)
        if not event:
            return []

        entries = (
            session.query(TournamentField, Player)
            .join(Player, TournamentField.player_id == Player.id)
            .filter(TournamentField.tournament_id == tournament_id)
            .order_by(Player.name)
            .all()
        )

        results = []
        for tf, player in entries:
            sg = (
                session.query(SGStat)
                .filter(
                    SGStat.player_id == player.id,
                    SGStat.tournament_id == tournament_id,
                )
                .first()
            )

            entry = {
                "player_id": player.id,
                "player_name": player.name,
                "status": tf.status,
                "event_name": event.event_name,
                "event_status": event.status,
            }

            if sg:
                entry.update({
                    "sg_total": sg.sg_total,
                    "sg_ott": sg.sg_ott,
                    "sg_app": sg.sg_app,
                    "sg_atg": sg.sg_atg,
                    "sg_putt": sg.sg_putt,
                    "rounds_played": sg.rounds_played,
                })

            # Parse any metadata
            if tf.metadata_json:
                try:
                    meta = json.loads(tf.metadata_json)
                    entry["position"] = meta.get("position")
                    entry["total_score"] = meta.get("total_score")
                except (json.JSONDecodeError, TypeError):
                    pass

            results.append(entry)

    return results


# ── internal helpers ──────────────────────────────────────────────

def _enrich_event(session, event: Event) -> dict:
    """Convert Event ORM object to dict with field count."""
    d = event.to_dict()

    field_count = (
        session.query(func.count(TournamentField.id))
        .filter(TournamentField.tournament_id == event.id)
        .scalar()
    )
    d["field_count"] = field_count or 0
    return d
