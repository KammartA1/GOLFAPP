"""Closing Worker — Capture closing lines, settle bets, compute CLV.

Runs on event completion:
  Golf: After each round completes + after tournament ends

1. Capture closing lines (last line before event start)
2. Settle pending bets with actual results
3. Compute CLV for each settled bet
4. Update bankroll and P&L

Writes to: bets (settlement), clv_log, line_movements (closing flags)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from database.connection import DatabaseManager
from database.models import Bet, LineMovement, CLVLog
from .base_worker import BaseWorker

logger = logging.getLogger(__name__)


class ClosingWorker(BaseWorker):
    WORKER_NAME = "closing_worker_golf"
    SPORT = "golf"
    DEFAULT_INTERVAL_SECONDS = 3600  # 1 hour

    def execute(self) -> dict:
        """Run closing line capture and settlement cycle."""
        results = {
            "closing_lines_captured": 0,
            "bets_settled": 0,
            "clv_computed": 0,
            "errors": [],
        }

        # 1. Mark closing lines for events about to start
        results["closing_lines_captured"] = self._capture_closing_lines()

        # 2. Settle any bets where we have actual results
        settle_result = self._settle_pending_bets()
        results["bets_settled"] = settle_result["settled"]
        results["errors"].extend(settle_result.get("errors", []))

        # 3. Compute CLV for settled bets that have closing lines
        results["clv_computed"] = self._compute_clv()

        self._update_scraper_status(
            success=len(results["errors"]) == 0,
            lines_count=results["bets_settled"],
        )

        return results

    def _capture_closing_lines(self) -> int:
        """Mark the latest line for each player/stat as 'closing' before event starts."""
        captured = 0
        try:
            with DatabaseManager.session_scope() as session:
                # Find active lines that haven't been marked as closing yet
                # Lines from the last 2 hours are candidates for closing
                cutoff = datetime.utcnow() - timedelta(hours=2)

                # Get distinct player/stat combos with active lines
                from sqlalchemy import func, and_
                active_combos = (
                    session.query(
                        LineMovement.player,
                        LineMovement.stat_type,
                        LineMovement.source,
                        func.max(LineMovement.captured_at).label("latest"),
                    )
                    .filter(
                        LineMovement.sport == "golf",
                        LineMovement.is_active == True,
                    )
                    .group_by(
                        LineMovement.player,
                        LineMovement.stat_type,
                        LineMovement.source,
                    )
                    .all()
                )

                for combo in active_combos:
                    # Find pending bets for this player/stat
                    pending_bet = (
                        session.query(Bet)
                        .filter(
                            Bet.sport == "golf",
                            Bet.player == combo.player,
                            Bet.stat_type == combo.stat_type,
                            Bet.status == "pending",
                        )
                        .first()
                    )

                    if pending_bet:
                        # Mark the latest line as closing
                        latest_line = (
                            session.query(LineMovement)
                            .filter(
                                LineMovement.player == combo.player,
                                LineMovement.stat_type == combo.stat_type,
                                LineMovement.source == combo.source,
                                LineMovement.sport == "golf",
                            )
                            .order_by(LineMovement.captured_at.desc())
                            .first()
                        )

                        if latest_line and not latest_line.is_closing:
                            latest_line.is_closing = True
                            captured += 1

                            # Update the bet's closing line
                            pending_bet.closing_line = latest_line.line
                            if latest_line.odds_american:
                                pending_bet.closing_odds = latest_line.odds_american

        except Exception:
            logger.exception("Failed to capture closing lines")

        if captured > 0:
            logger.info("Captured %d closing lines", captured)
        return captured

    def _settle_pending_bets(self) -> dict:
        """Settle bets where actual results are available.

        For golf: checks tournament results / stat actuals.
        Results must be provided externally (via API or manual entry).
        """
        result = {"settled": 0, "errors": []}

        try:
            with DatabaseManager.session_scope() as session:
                pending_bets = (
                    session.query(Bet)
                    .filter(
                        Bet.sport == "golf",
                        Bet.status == "pending",
                    )
                    .all()
                )

                if not pending_bets:
                    return result

                for bet in pending_bets:
                    actual = self._get_actual_result(bet.player, bet.stat_type, bet.event_id, session)

                    if actual is None:
                        continue  # Result not available yet

                    try:
                        # Determine outcome
                        if bet.direction == "over":
                            won = actual > bet.line
                        elif bet.direction == "under":
                            won = actual < bet.line
                        else:
                            won = actual > bet.line

                        if actual == bet.line:
                            bet.status = "push"
                            bet.pnl = 0.0
                        elif won:
                            bet.status = "won"
                            bet.pnl = bet.stake * (bet.odds_decimal - 1.0)
                        else:
                            bet.status = "lost"
                            bet.pnl = -bet.stake

                        bet.actual_result = actual
                        bet.settled_at = datetime.utcnow()
                        result["settled"] += 1

                        logger.info("Settled bet %s: %s %s %s @ %.1f → actual=%.1f → %s (P&L=$%.2f)",
                                    bet.bet_id, bet.player, bet.direction,
                                    bet.stat_type, bet.line, actual, bet.status, bet.pnl)

                    except Exception as e:
                        result["errors"].append(f"settle {bet.bet_id}: {e}")
                        logger.exception("Failed to settle bet %s", bet.bet_id)

        except Exception as e:
            result["errors"].append(str(e))
            logger.exception("Settlement cycle failed")

        return result

    def _get_actual_result(
        self, player: str, stat_type: str, event_id: str | None, session
    ) -> float | None:
        """Look up actual result for a player/stat.

        Checks tournament_results and any external data sources.
        Returns None if not yet available.
        """
        try:
            from database.models import TournamentResult, Player as PlayerModel

            if not event_id:
                return None

            player_record = (
                session.query(PlayerModel)
                .filter(PlayerModel.name == player)
                .first()
            )
            if not player_record:
                return None

            tr = (
                session.query(TournamentResult)
                .filter_by(player_id=player_record.id, event_id=event_id)
                .first()
            )
            if not tr:
                return None

            # Map stat_type to actual result
            stat_map = {
                "fantasy_score": None,  # Would need DFS scoring
                "finish_position": tr.finish_position,
                "total_score": tr.total_score,
                "total_to_par": tr.total_to_par,
            }

            return stat_map.get(stat_type.lower())

        except Exception:
            return None

    def _compute_clv(self) -> int:
        """Compute CLV for settled bets that have closing lines."""
        computed = 0
        try:
            with DatabaseManager.session_scope() as session:
                # Find settled bets with closing lines but no CLV entry
                from sqlalchemy import and_
                settled = (
                    session.query(Bet)
                    .filter(
                        Bet.sport == "golf",
                        Bet.status.in_(["won", "lost"]),
                        Bet.closing_line.isnot(None),
                    )
                    .all()
                )

                for bet in settled:
                    # Check if CLV already computed
                    existing = (
                        session.query(CLVLog)
                        .filter_by(bet_id=bet.bet_id)
                        .first()
                    )
                    if existing:
                        continue

                    # Compute CLV
                    opening_line = self._get_opening_line(
                        bet.player, bet.stat_type, session
                    )
                    if opening_line is None:
                        opening_line = bet.line  # Fallback

                    closing_line = bet.closing_line
                    line_movement = closing_line - opening_line

                    # CLV in probability space
                    # If we bet OVER and line moved UP → we got +CLV
                    # If we bet UNDER and line moved DOWN → we got +CLV
                    if bet.direction == "over":
                        clv_raw = (closing_line - bet.line) / max(abs(bet.line), 1.0)
                    else:
                        clv_raw = (bet.line - closing_line) / max(abs(bet.line), 1.0)

                    # CLV in cents per dollar (standard metric)
                    clv_cents = clv_raw * 100

                    beat_close = clv_raw > 0

                    clv_entry = CLVLog(
                        bet_id=bet.bet_id,
                        sport="golf",
                        opening_line=opening_line,
                        bet_line=bet.line,
                        closing_line=closing_line,
                        line_movement=line_movement,
                        clv_raw=clv_raw,
                        clv_cents=clv_cents,
                        beat_close=beat_close,
                        calculated_at=datetime.utcnow(),
                    )
                    session.add(clv_entry)
                    computed += 1

        except Exception:
            logger.exception("CLV computation failed")

        if computed > 0:
            logger.info("Computed CLV for %d bets", computed)
        return computed

    def _get_opening_line(self, player: str, stat_type: str, session) -> float | None:
        """Get the opening line for a player/stat."""
        opening = (
            session.query(LineMovement)
            .filter(
                LineMovement.player == player,
                LineMovement.stat_type == stat_type,
                LineMovement.sport == "golf",
                LineMovement.is_opening == True,
            )
            .order_by(LineMovement.captured_at.desc())
            .first()
        )
        if opening:
            return opening.line

        # Fallback: earliest line
        earliest = (
            session.query(LineMovement)
            .filter(
                LineMovement.player == player,
                LineMovement.stat_type == stat_type,
                LineMovement.sport == "golf",
            )
            .order_by(LineMovement.captured_at.asc())
            .first()
        )
        return earliest.line if earliest else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    worker = ClosingWorker()
    worker.run_forever()
