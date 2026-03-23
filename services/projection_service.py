"""
Projection Service — Model computation layer
=============================================
Wraps the SG projection pipeline and probability calculator.
All computation reads/writes database, not session state.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from database.models import (
    Event, Player, Signal, SGStat, TournamentField,
)
from models.strokes_gained import SGModel
from models.probability_calculator import (
    analyze_line, project_stat, calc_probability, kelly_stake,
)
from services._db import get_session as _session

log = logging.getLogger(__name__)


def _weather_impact_factor(wind_speed_mph: float, precipitation_mm: float) -> dict:
    """Compute weather adjustment dict from raw values."""
    try:
        from scrapers.weather_scraper import weather_impact_factor
        return weather_impact_factor(wind_speed_mph, precipitation_mm)
    except ImportError:
        pass

    # Inline fallback
    is_significant = wind_speed_mph > 15 or precipitation_mm > 1.0
    variance_mult = 1.0
    projection_mult = 1.0
    description = "Normal conditions"

    if wind_speed_mph > 25:
        variance_mult = 1.25
        projection_mult = 0.97
        description = f"Strong wind ({wind_speed_mph:.0f} mph)"
    elif wind_speed_mph > 15:
        variance_mult = 1.10
        projection_mult = 0.99
        description = f"Moderate wind ({wind_speed_mph:.0f} mph)"

    if precipitation_mm > 2.0:
        variance_mult *= 1.10
        projection_mult *= 0.98
        description += f", rain ({precipitation_mm:.1f}mm)"

    return {
        "is_significant": is_significant,
        "variance_mult": variance_mult,
        "projection_mult": projection_mult,
        "description": description,
    }


def run_projection(
    player_name: str,
    tournament_id: int,
    market: str,
    line: float,
    settings_dict: dict,
) -> dict:
    """
    Run the full SG projection pipeline for a single player/line.

    Parameters
    ----------
    player_name : str
    tournament_id : int
        Event ID in the events table.
    market : str
        Stat type, e.g. "birdies", "fantasy_score".
    line : float
        The line value to evaluate against.
    settings_dict : dict
        Must include bankroll, kelly_fraction, min_edge.
        May include weather (dict with wind_speed_mph, precipitation_mm).

    Returns
    -------
    dict with projection, probability, EV, Kelly stake, etc.
    """
    bankroll = float(settings_dict.get("bankroll", 1000))
    kelly_frac = float(settings_dict.get("kelly_fraction", 0.25))

    # Build player SG profile from database
    player_sg = _load_player_sg(player_name)

    # Weather adjustment
    weather_adj = None
    weather_raw = settings_dict.get("weather")
    if weather_raw and isinstance(weather_raw, dict):
        weather_adj = _weather_impact_factor(
            weather_raw.get("wind_speed_mph", 0),
            weather_raw.get("precipitation_mm", 0),
        )

    # Consensus probability (from odds, if available)
    consensus_prob = settings_dict.get("consensus_prob")

    # Run the analysis
    result = analyze_line(
        player_name=player_name,
        stat_type=market,
        line_value=line,
        player_sg=player_sg,
        weather_adj=weather_adj,
        consensus_prob=consensus_prob,
        bankroll=bankroll,
        kelly_frac=kelly_frac,
    )

    # Persist the signal to DB
    _store_signal(result, tournament_id)

    return result


def run_tournament_projections(
    tournament_id: int,
    settings: dict,
) -> list[dict]:
    """
    Project all players in a tournament field.

    Returns list of projection dicts, one per player.
    """
    with _session() as session:
        event = session.get(Event, tournament_id)
        if not event:
            log.warning("Event %d not found", tournament_id)
            return []

        course_name = event.course_name or ""

        field_entries = (
            session.query(TournamentField)
            .filter(TournamentField.tournament_id == tournament_id)
            .all()
        )

        if not field_entries:
            log.info("No field entries for event %d", tournament_id)
            return []

        player_ids = [fe.player_id for fe in field_entries]
        players = (
            session.query(Player)
            .filter(Player.id.in_(player_ids))
            .all()
        )
        player_map = {p.id: p.name for p in players}

    # Build history per player for SGModel
    sg_model = SGModel()
    field_history = {}

    for pid, pname in player_map.items():
        history = _load_player_history(pid)
        if history:
            field_history[pname] = history

    if not field_history:
        log.info("No SG history for any player in field")
        return []

    df = sg_model.project_field(field_history, course_name)
    results = df.to_dict(orient="records")

    # Store signals for each player
    for row in results:
        signal_data = {
            "player_name": row.get("name", ""),
            "best_edge": row.get("proj_sg_total", 0) * 0.01,  # rough edge proxy
            "projection": row.get("proj_sg_total", 0),
            "confidence": row.get("data_quality", "low"),
        }
        _store_signal(signal_data, tournament_id)

    return results


def get_cached_projection(
    player: str,
    tournament_id: int,
) -> Optional[dict]:
    """
    Check the signals table for a recent projection (within last 6 hours).

    Returns the signal dict or None.
    """
    cutoff = datetime.utcnow() - timedelta(hours=6)

    with _session() as session:
        sig = (
            session.query(Signal)
            .filter(
                Signal.player == player,
                Signal.sport == "GOLF",
                Signal.generated_at >= cutoff,
            )
            .order_by(Signal.generated_at.desc())
            .first()
        )
        if sig:
            return sig.to_dict()

    return None


# ── internal helpers ──────────────────────────────────────────────

def _load_player_sg(player_name: str) -> dict:
    """Load latest SG stats for a player from DB, returning a flat dict."""
    with _session() as session:
        player = (
            session.query(Player)
            .filter(Player.name == player_name, Player.sport == "GOLF")
            .first()
        )
        if not player:
            return {"sg_total": 0, "sg_ott": 0, "sg_app": 0, "sg_atg": 0, "sg_putt": 0}

        stats = (
            session.query(SGStat)
            .filter(SGStat.player_id == player.id)
            .order_by(SGStat.created_at.desc())
            .first()
        )
        if not stats:
            return {"sg_total": 0, "sg_ott": 0, "sg_app": 0, "sg_atg": 0, "sg_putt": 0}

        return {
            "sg_total": stats.sg_total or 0,
            "sg_ott": stats.sg_ott or 0,
            "sg_app": stats.sg_app or 0,
            "sg_atg": stats.sg_atg or 0,
            "sg_putt": stats.sg_putt or 0,
        }


def _load_player_history(player_id: int) -> list[dict]:
    """Load SG history for a player (for recency weighting)."""
    with _session() as session:
        rows = (
            session.query(SGStat)
            .filter(SGStat.player_id == player_id)
            .order_by(SGStat.created_at.asc())
            .all()
        )
        history = []
        for r in rows:
            history.append({
                "event_date": r.created_at,
                "sg_total": r.sg_total or 0,
                "sg_ott": r.sg_ott or 0,
                "sg_app": r.sg_app or 0,
                "sg_atg": r.sg_atg or 0,
                "sg_putt": r.sg_putt or 0,
            })
        return history


def _store_signal(result: dict, tournament_id: int) -> None:
    """Persist a projection result to the signals table."""
    try:
        with _session() as session:
            # Look up event name
            event = session.get(Event, tournament_id) if tournament_id else None
            event_name = event.event_name if event else ""

            edge = result.get("best_edge", result.get("edge_pct", 0)) or 0
            direction = result.get("best_side", result.get("direction", ""))
            confidence = result.get("confidence", result.get("best_prob", 0))
            if isinstance(confidence, str):
                conf_val = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.5, "NO_BET": 0.2}.get(confidence, 0.5)
            else:
                conf_val = float(confidence) if confidence else 0.5

            sig = Signal(
                sport="GOLF",
                event=event_name,
                market=result.get("stat_type", result.get("market", "")),
                player=result.get("player_name", result.get("name", "")),
                signal_value=result.get("projection", result.get("proj_sg_total", 0)),
                confidence=conf_val,
                direction=direction,
                edge_pct=float(edge) if edge else 0,
                kelly_stake=result.get("kelly_stake", 0),
            )
            session.add(sig)
    except Exception as exc:
        log.error("Failed to store signal: %s", exc)
