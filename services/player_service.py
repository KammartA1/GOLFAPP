"""
Player Service — Player data layer
====================================
Manages player records, SG stats, and tournament field lookups.
"""
import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import or_, func

from database.models import Player, SGStat, TournamentField, Event
from services._db import get_session as _session

log = logging.getLogger(__name__)


def get_or_create_player(name: str, tour: str = "PGA") -> dict:
    """
    Look up a player by name (case-insensitive). Create if not found.

    The 'tour' value is stored in the ``team`` column (e.g. "PGA", "LIV",
    "DP World Tour").

    Returns the player record as a dict.
    """
    normalized = name.strip()
    if not normalized:
        return {"error": "Empty player name"}

    with _session() as session:
        player = (
            session.query(Player)
            .filter(
                func.lower(Player.name) == normalized.lower(),
                Player.sport == "GOLF",
            )
            .first()
        )

        if player:
            return player.to_dict()

        # Create
        player = Player(
            name=normalized,
            team=tour,
            sport="GOLF",
            active=True,
            last_updated=datetime.utcnow(),
        )
        session.add(player)
        session.flush()
        result = player.to_dict()

    log.info("Created player: %s (tour=%s)", normalized, tour)
    return result


def search_players(query: str) -> list[dict]:
    """
    Search players by partial name match (case-insensitive).

    Returns up to 50 matching players sorted alphabetically.
    """
    if not query or len(query) < 2:
        return []

    pattern = f"%{query.strip()}%"

    with _session() as session:
        rows = (
            session.query(Player)
            .filter(
                Player.sport == "GOLF",
                Player.name.ilike(pattern),
            )
            .order_by(Player.name)
            .limit(50)
            .all()
        )
        return [r.to_dict() for r in rows]


def get_player_sg_stats(
    player_name: str,
    n_events: int = 10,
) -> dict:
    """
    Return aggregated SG stats for a player from the last N events.

    Returns dict with per-category averages and list of per-event stats.
    """
    with _session() as session:
        player = (
            session.query(Player)
            .filter(
                func.lower(Player.name) == player_name.strip().lower(),
                Player.sport == "GOLF",
            )
            .first()
        )
        if not player:
            return {
                "player_name": player_name,
                "found": False,
                "events": [],
                "averages": {},
            }

        sg_rows = (
            session.query(SGStat)
            .filter(SGStat.player_id == player.id)
            .order_by(SGStat.created_at.desc())
            .limit(n_events)
            .all()
        )

        events = [r.to_dict() for r in sg_rows]

    if not events:
        return {
            "player_name": player_name,
            "found": True,
            "player_id": player.id,
            "events": [],
            "averages": {
                "sg_total": 0, "sg_ott": 0, "sg_app": 0,
                "sg_atg": 0, "sg_putt": 0,
            },
        }

    # Compute averages
    cats = ["sg_total", "sg_ott", "sg_app", "sg_atg", "sg_putt"]
    averages = {}
    for cat in cats:
        vals = [e.get(cat) for e in events if e.get(cat) is not None]
        averages[cat] = round(sum(vals) / len(vals), 3) if vals else 0

    return {
        "player_name": player_name,
        "found": True,
        "player_id": player.id,
        "n_events": len(events),
        "events": events,
        "averages": averages,
    }


def get_tournament_field(tournament_id: int) -> list[dict]:
    """
    Return the full field for a tournament, including player details
    and latest SG averages.
    """
    with _session() as session:
        entries = (
            session.query(TournamentField, Player)
            .join(Player, TournamentField.player_id == Player.id)
            .filter(TournamentField.tournament_id == tournament_id)
            .order_by(Player.name)
            .all()
        )

        results = []
        for tf, player in entries:
            # Fetch latest SG for this player
            sg = (
                session.query(SGStat)
                .filter(SGStat.player_id == player.id)
                .order_by(SGStat.created_at.desc())
                .first()
            )

            entry = {
                "field_id": tf.id,
                "player_id": player.id,
                "player_name": player.name,
                "tour": player.team,
                "status": tf.status,
                "tee_time": tf.tee_time.isoformat() if tf.tee_time else None,
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
            else:
                entry.update({
                    "sg_total": None, "sg_ott": None, "sg_app": None,
                    "sg_atg": None, "sg_putt": None, "rounds_played": None,
                })

            results.append(entry)

    return results


def update_player_sg(player_id: int, sg_data: dict) -> None:
    """
    Insert a new SG stat record for a player.

    sg_data keys:
        sg_total, sg_ott, sg_app, sg_atg, sg_putt,
        rounds_played (optional), season (optional),
        source (optional), tournament_id (optional).
    """
    with _session() as session:
        player = session.get(Player, player_id)
        if not player:
            log.warning("Player %d not found for SG update", player_id)
            return

        stat = SGStat(
            player_id=player_id,
            tournament_id=sg_data.get("tournament_id"),
            sg_total=sg_data.get("sg_total"),
            sg_ott=sg_data.get("sg_ott"),
            sg_app=sg_data.get("sg_app"),
            sg_atg=sg_data.get("sg_atg"),
            sg_putt=sg_data.get("sg_putt"),
            rounds_played=sg_data.get("rounds_played"),
            season=sg_data.get("season"),
            source=sg_data.get("source", "manual"),
        )
        session.add(stat)

        player.last_updated = datetime.utcnow()

    log.info(
        "Updated SG for player %d: total=%.3f",
        player_id,
        sg_data.get("sg_total", 0) or 0,
    )
