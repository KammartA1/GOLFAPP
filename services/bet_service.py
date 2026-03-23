"""
Bet Service — Bet management layer
====================================
Place, settle, query, and analyse bets. All P&L and CLV computation lives here.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import and_, func, case

from database.models import Bet, Event, LineMovement
from services._db import get_session as _session

log = logging.getLogger(__name__)


def place_bet(bet_data: dict) -> int:
    """
    Record a new bet in the database.

    bet_data keys:
        player, market, event, direction (OVER/UNDER/WIN),
        bet_line, predicted_prob, stake, odds_american, odds_decimal,
        model_projection, model_std, confidence_score,
        tournament_id (optional), model_version (optional),
        signal_line (optional), notes (optional),
        features_snapshot (optional, dict serialised to JSON).

    Returns the new bet ID.
    """
    with _session() as session:
        bet = Bet(
            sport=bet_data.get("sport", "GOLF"),
            event=bet_data.get("event", ""),
            market=bet_data.get("market", ""),
            player=bet_data.get("player", ""),
            direction=bet_data.get("direction", ""),
            signal_line=bet_data.get("signal_line"),
            bet_line=bet_data.get("bet_line"),
            predicted_prob=bet_data.get("predicted_prob"),
            stake=bet_data.get("stake", 0),
            odds_american=bet_data.get("odds_american"),
            odds_decimal=bet_data.get("odds_decimal"),
            model_projection=bet_data.get("model_projection"),
            model_std=bet_data.get("model_std"),
            confidence_score=bet_data.get("confidence_score"),
            tournament_id=bet_data.get("tournament_id"),
            model_version=bet_data.get("model_version"),
            status="pending",
            notes=bet_data.get("notes"),
            timestamp=datetime.utcnow(),
        )

        features = bet_data.get("features_snapshot")
        if features and isinstance(features, dict):
            bet.features_snapshot_json = json.dumps(features)

        session.add(bet)
        session.flush()
        bet_id = bet.id

    log.info("Placed bet #%d: %s %s %s", bet_id, bet.player, bet.market, bet.direction)
    return bet_id


def settle_bet(bet_id: int, actual_result: float) -> dict:
    """
    Settle a pending bet given the actual stat result.

    Determines win/loss/push based on direction and bet_line,
    calculates P&L, and updates the record.

    Returns the settled bet as a dict.
    """
    with _session() as session:
        bet = session.get(Bet, bet_id)
        if not bet:
            log.warning("Bet #%d not found", bet_id)
            return {"error": f"Bet {bet_id} not found"}

        if bet.status != "pending":
            log.warning("Bet #%d already settled (%s)", bet_id, bet.status)
            return {"error": f"Bet {bet_id} already settled as {bet.status}"}

        bet.actual_outcome = actual_result
        bet_line = bet.bet_line or 0
        direction = (bet.direction or "").upper()
        stake = bet.stake or 0
        odds_dec = bet.odds_decimal or 1.87

        # Determine outcome
        if actual_result == bet_line:
            bet.status = "push"
            bet.profit = 0.0
            bet.pnl = 0.0
        elif direction in ("OVER", "WIN"):
            won = actual_result > bet_line
            bet.status = "won" if won else "lost"
            bet.profit = round(stake * (odds_dec - 1), 2) if won else round(-stake, 2)
            bet.pnl = bet.profit
        elif direction == "UNDER":
            won = actual_result < bet_line
            bet.status = "won" if won else "lost"
            bet.profit = round(stake * (odds_dec - 1), 2) if won else round(-stake, 2)
            bet.pnl = bet.profit
        else:
            # Outright / place — simple win/loss on actual_result > 0
            won = actual_result > 0
            bet.status = "won" if won else "lost"
            bet.profit = round(stake * (odds_dec - 1), 2) if won else round(-stake, 2)
            bet.pnl = bet.profit

        # Record closing line if available
        closing = _get_closing_line(session, bet.player, bet.market)
        if closing is not None:
            bet.closing_line = closing

        bet.settled_at = datetime.utcnow()
        result = bet.to_dict()

    log.info("Settled bet #%d: %s (P&L: $%.2f)", bet_id, bet.status, bet.profit or 0)
    return result


def get_pending_bets() -> list[dict]:
    """Return all bets with status='pending'."""
    with _session() as session:
        rows = (
            session.query(Bet)
            .filter(Bet.status == "pending", Bet.sport == "GOLF")
            .order_by(Bet.timestamp.desc())
            .all()
        )
        return [r.to_dict() for r in rows]


def get_settled_bets(days: int = 30) -> list[dict]:
    """Return settled bets within the last N days."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    with _session() as session:
        rows = (
            session.query(Bet)
            .filter(
                Bet.sport == "GOLF",
                Bet.status.in_(["won", "lost", "push"]),
                Bet.settled_at >= cutoff,
            )
            .order_by(Bet.settled_at.desc())
            .all()
        )
        return [r.to_dict() for r in rows]


def get_bet_history(filters: dict) -> list[dict]:
    """
    Query bet history with flexible filters.

    Supported filter keys:
        player, market, status, tournament_id,
        model_version, min_edge, days, limit.
    """
    with _session() as session:
        q = session.query(Bet).filter(Bet.sport == "GOLF")

        if filters.get("player"):
            q = q.filter(Bet.player == filters["player"])
        if filters.get("market"):
            q = q.filter(Bet.market == filters["market"])
        if filters.get("status"):
            q = q.filter(Bet.status == filters["status"])
        if filters.get("tournament_id"):
            q = q.filter(Bet.tournament_id == filters["tournament_id"])
        if filters.get("model_version"):
            q = q.filter(Bet.model_version == filters["model_version"])
        if filters.get("min_edge"):
            # edge = predicted_prob - implied_prob; approximate via confidence_score
            q = q.filter(Bet.confidence_score >= filters["min_edge"])
        if filters.get("days"):
            cutoff = datetime.utcnow() - timedelta(days=int(filters["days"]))
            q = q.filter(Bet.timestamp >= cutoff)

        limit = int(filters.get("limit", 500))
        rows = q.order_by(Bet.timestamp.desc()).limit(limit).all()
        return [r.to_dict() for r in rows]


def get_pnl_summary(period: str = "daily") -> dict:
    """
    Aggregate P&L summary.

    period: 'daily', 'weekly', 'monthly', 'all'

    Returns dict with total_pnl, total_staked, roi, win_rate,
    bets_settled, bets_won, bets_lost, avg_stake, best_bet, worst_bet.
    """
    if period == "daily":
        cutoff = datetime.utcnow() - timedelta(days=1)
    elif period == "weekly":
        cutoff = datetime.utcnow() - timedelta(weeks=1)
    elif period == "monthly":
        cutoff = datetime.utcnow() - timedelta(days=30)
    else:
        cutoff = datetime(2000, 1, 1)

    with _session() as session:
        settled = (
            session.query(Bet)
            .filter(
                Bet.sport == "GOLF",
                Bet.status.in_(["won", "lost", "push"]),
                Bet.settled_at >= cutoff,
            )
            .all()
        )

    if not settled:
        return {
            "period": period,
            "total_pnl": 0,
            "total_staked": 0,
            "roi": 0,
            "win_rate": 0,
            "bets_settled": 0,
            "bets_won": 0,
            "bets_lost": 0,
            "bets_pushed": 0,
            "avg_stake": 0,
            "best_bet": 0,
            "worst_bet": 0,
        }

    profits = [(b.pnl or b.profit or 0) for b in settled]
    stakes = [(b.stake or 0) for b in settled]
    total_pnl = sum(profits)
    total_staked = sum(stakes)
    won = sum(1 for b in settled if b.status == "won")
    lost = sum(1 for b in settled if b.status == "lost")
    pushed = sum(1 for b in settled if b.status == "push")

    return {
        "period": period,
        "total_pnl": round(total_pnl, 2),
        "total_staked": round(total_staked, 2),
        "roi": round(total_pnl / total_staked, 4) if total_staked > 0 else 0,
        "win_rate": round(won / len(settled), 4) if settled else 0,
        "bets_settled": len(settled),
        "bets_won": won,
        "bets_lost": lost,
        "bets_pushed": pushed,
        "avg_stake": round(total_staked / len(settled), 2) if settled else 0,
        "best_bet": round(max(profits), 2) if profits else 0,
        "worst_bet": round(min(profits), 2) if profits else 0,
    }


def get_clv_summary(window: int = 100) -> dict:
    """
    Compute Closing Line Value summary over the last N settled bets.

    CLV = (closing_line - bet_line) for UNDER or (bet_line - closing_line) for OVER.
    Positive CLV means we consistently beat the closing line, which is the
    strongest predictor of long-term profitability.

    Returns dict with avg_clv, median_clv, pct_positive, n_bets, clv_by_market.
    """
    with _session() as session:
        bets = (
            session.query(Bet)
            .filter(
                Bet.sport == "GOLF",
                Bet.status.in_(["won", "lost", "push"]),
                Bet.closing_line.isnot(None),
                Bet.bet_line.isnot(None),
            )
            .order_by(Bet.settled_at.desc())
            .limit(window)
            .all()
        )

    if not bets:
        return {
            "avg_clv": 0,
            "median_clv": 0,
            "pct_positive": 0,
            "n_bets": 0,
            "clv_by_market": {},
        }

    clvs = []
    by_market: dict[str, list] = {}

    for b in bets:
        cl = b.closing_line or 0
        bl = b.bet_line or 0
        direction = (b.direction or "").upper()

        if direction == "OVER":
            clv = bl - cl  # lower closing line = we got better value
        elif direction == "UNDER":
            clv = cl - bl  # higher closing line = we got better value
        else:
            # Outright: compare implied probs via odds
            clv = cl - bl  # simplified

        clvs.append(clv)
        by_market.setdefault(b.market or "unknown", []).append(clv)

    import statistics
    avg_clv = sum(clvs) / len(clvs)
    median_clv = statistics.median(clvs)
    pct_positive = sum(1 for c in clvs if c > 0) / len(clvs)

    market_summary = {}
    for mkt, vals in by_market.items():
        market_summary[mkt] = {
            "avg_clv": round(sum(vals) / len(vals), 4),
            "n_bets": len(vals),
            "pct_positive": round(sum(1 for v in vals if v > 0) / len(vals), 4),
        }

    return {
        "avg_clv": round(avg_clv, 4),
        "median_clv": round(median_clv, 4),
        "pct_positive": round(pct_positive, 4),
        "n_bets": len(clvs),
        "clv_by_market": market_summary,
    }


# ── internal helpers ──────────────────────────────────────────────

def _get_closing_line(session, player: str, market: str) -> Optional[float]:
    """
    Fetch the most recent closing line for a player+market.
    Looks for lines marked is_closing=True, or falls back to the most
    recent line entry.
    """
    # Try explicit closing line
    closing = (
        session.query(LineMovement)
        .filter(
            LineMovement.player == player,
            LineMovement.market == market,
            LineMovement.is_closing == True,
        )
        .order_by(LineMovement.timestamp.desc())
        .first()
    )
    if closing and closing.line is not None:
        return closing.line

    # Fallback: most recent line
    latest = (
        session.query(LineMovement)
        .filter(
            LineMovement.player == player,
            LineMovement.market == market,
        )
        .order_by(LineMovement.timestamp.desc())
        .first()
    )
    if latest and latest.line is not None:
        return latest.line

    return None
