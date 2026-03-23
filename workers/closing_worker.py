"""
Golf Quant Engine -- Closing Line Worker
==========================================
Captures closing lines at event start / round start for CLV calculation.

Golf-specific logic:
  - PGA Tour events have 4 rounds over 4 days (Thu--Sun typically)
  - Each round has tee times spread over ~4 hours
  - "Closing" = the last line snapshot before the first tee time of each round
  - Must handle Thursday (R1), Friday (R2), Saturday (R3), Sunday (R4) separately

Every 15 minutes, this worker:
  1. Checks events with rounds starting within the next 30 minutes
  2. Fetches final lines from all books
  3. Marks line_movements as is_closing=True
  4. Updates any open bets with closing_line
  5. Calculates CLV for bets that now have closing lines

Run standalone:
    python -m workers.closing_worker
"""
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workers.base import BaseWorker
from database.connection import get_session_factory, init_db
from database.models import LineMovement, Bet, Event

log = logging.getLogger(__name__)

# Standard PGA Tour round start times (Eastern)
# Thursday R1: ~7:00 AM ET first tee
# These are approximate -- actual tee times vary
ROUND_START_OFFSETS = {
    1: 0,   # Thursday
    2: 1,   # Friday
    3: 2,   # Saturday
    4: 3,   # Sunday
}


class ClosingWorker(BaseWorker):
    name = "closing_worker"
    interval_seconds = int(os.environ.get("CLOSING_WORKER_INTERVAL", 900))  # 15 min
    max_retries = 2
    retry_delay = 10.0
    description = "Captures closing lines and calculates CLV"

    def execute(self) -> dict:
        init_db()
        factory = get_session_factory()
        session = factory()

        lines_marked = 0
        bets_updated = 0
        clv_calculated = 0

        try:
            now = datetime.utcnow()
            window_start = now
            window_end = now + timedelta(minutes=30)

            # 1. Find live/scheduled events where a round might be starting soon
            active_events = (
                session.query(Event)
                .filter(Event.sport == "GOLF")
                .filter(Event.status.in_(["scheduled", "live"]))
                .all()
            )

            for event in active_events:
                meta = json.loads(event.metadata_json) if event.metadata_json else {}
                current_round = meta.get("current_round", 0)

                # Check if we should capture closing lines
                # We capture closing lines when the event is about to go live
                # or when a new round is about to start
                should_capture = self._should_capture_closing(
                    event, current_round, window_start, window_end
                )

                if not should_capture:
                    continue

                self._logger.info(
                    "Capturing closing lines for [%s] (round %d)",
                    event.event_name, current_round or 1,
                )

                # 2. Get the most recent line for each player+market+book combination
                marked = self._mark_closing_lines(session, event)
                lines_marked += marked

                # 3. Update open bets with closing lines
                updated, clv_count = self._update_bets_with_closing(session, event)
                bets_updated += updated
                clv_calculated += clv_count

            session.commit()

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return {
            "items_processed": lines_marked + bets_updated,
            "lines_marked_closing": lines_marked,
            "bets_updated": bets_updated,
            "clv_calculated": clv_calculated,
        }

    # ------------------------------------------------------------------ #
    # Determine if closing lines should be captured
    # ------------------------------------------------------------------ #
    def _should_capture_closing(
        self, event: Event, current_round: int, window_start: datetime, window_end: datetime
    ) -> bool:
        """Determine if we should capture closing lines for this event."""
        if not event.start_time:
            # No start time known -- check if event status just went live
            return event.status == "live"

        start = event.start_time

        # For each round, calculate the approximate round start time
        for round_num in range(1, 5):
            round_start = start + timedelta(days=ROUND_START_OFFSETS.get(round_num, round_num - 1))

            # Set round start to 7 AM on that day (typical first tee)
            round_start = round_start.replace(hour=11, minute=0, second=0)  # 11 UTC = 7 AM ET

            # Capture closing lines in the 30-minute window before round starts
            capture_window_start = round_start - timedelta(minutes=30)
            capture_window_end = round_start

            # Check if current time falls in the capture window
            if capture_window_start <= window_start <= capture_window_end:
                return True
            if capture_window_start <= window_end <= capture_window_end:
                return True

        # Also capture if event just transitioned to "live"
        meta = json.loads(event.metadata_json) if event.metadata_json else {}
        prev_status = meta.get("prev_status")
        if event.status == "live" and prev_status in ("scheduled", None):
            return True

        return False

    # ------------------------------------------------------------------ #
    # Mark closing lines
    # ------------------------------------------------------------------ #
    def _mark_closing_lines(self, session, event: Event) -> int:
        """Find the most recent line per player+market+book and mark as closing."""
        # Get all unique player+market+book combos for this event
        from sqlalchemy import func, and_

        subquery = (
            session.query(
                LineMovement.player,
                LineMovement.market,
                LineMovement.book,
                func.max(LineMovement.timestamp).label("max_ts"),
            )
            .filter(LineMovement.event == event.event_name)
            .filter(LineMovement.sport == "GOLF")
            .filter(LineMovement.is_closing == False)
            .group_by(LineMovement.player, LineMovement.market, LineMovement.book)
            .subquery()
        )

        # Mark those rows as closing
        latest_lines = (
            session.query(LineMovement)
            .join(
                subquery,
                and_(
                    LineMovement.player == subquery.c.player,
                    LineMovement.market == subquery.c.market,
                    LineMovement.book == subquery.c.book,
                    LineMovement.timestamp == subquery.c.max_ts,
                ),
            )
            .filter(LineMovement.event == event.event_name)
            .filter(LineMovement.sport == "GOLF")
            .all()
        )

        count = 0
        for lm in latest_lines:
            if not lm.is_closing:
                lm.is_closing = True
                count += 1

        if count:
            self._logger.info(
                "Marked %d lines as closing for [%s]", count, event.event_name,
            )

        return count

    # ------------------------------------------------------------------ #
    # Update bets with closing lines and calculate CLV
    # ------------------------------------------------------------------ #
    def _update_bets_with_closing(self, session, event: Event) -> tuple[int, int]:
        """Update pending bets with closing lines and compute CLV.

        Returns (bets_updated, clv_calculated).
        """
        # Find pending bets for this event
        pending_bets = (
            session.query(Bet)
            .filter(Bet.sport == "GOLF")
            .filter(Bet.event == event.event_name)
            .filter(Bet.status == "pending")
            .filter(Bet.closing_line.is_(None))
            .all()
        )

        if not pending_bets:
            return 0, 0

        bets_updated = 0
        clv_calculated = 0

        for bet in pending_bets:
            # Find the closing line for this bet's player + market
            closing_lm = (
                session.query(LineMovement)
                .filter(LineMovement.event == event.event_name)
                .filter(LineMovement.sport == "GOLF")
                .filter(LineMovement.player == bet.player)
                .filter(LineMovement.market == bet.market)
                .filter(LineMovement.is_closing == True)
                .order_by(LineMovement.timestamp.desc())
                .first()
            )

            if closing_lm is None:
                continue

            closing_line = closing_lm.line
            if closing_line is None:
                continue

            bet.closing_line = closing_line
            bets_updated += 1

            # Calculate CLV
            clv = self._calculate_clv(bet, closing_line)
            if clv is not None:
                bet.pnl = clv.get("clv_raw")  # Will be overwritten on settlement
                clv_calculated += 1
                self._logger.debug(
                    "CLV for %s %s %s: raw=%.3f",
                    bet.player, bet.direction, bet.market, clv["clv_raw"],
                )

        if bets_updated:
            self._logger.info(
                "Updated %d bets with closing lines, calculated %d CLV values for [%s]",
                bets_updated, clv_calculated, event.event_name,
            )

        return bets_updated, clv_calculated

    def _calculate_clv(self, bet: Bet, closing_line: float) -> dict | None:
        """Calculate closing line value for a bet.

        CLV > 0 means we got a better number than the market's final assessment.
        """
        bet_line = bet.bet_line or bet.signal_line
        if bet_line is None:
            return None

        direction = (bet.direction or "").upper()

        if direction == "OVER":
            # For over bets: CLV > 0 if closing line moved UP (market agreed with us)
            clv_raw = closing_line - bet_line
        elif direction == "UNDER":
            # For under bets: CLV > 0 if closing line moved DOWN
            clv_raw = bet_line - closing_line
        elif direction in ("BACK", "WIN"):
            # For outright/winner: CLV > 0 if closing odds shortened
            # (closing line < bet line means price decreased = market agreed)
            clv_raw = bet_line - closing_line
        else:
            # Default: treat as over-like
            clv_raw = closing_line - bet_line

        # Convert to approximate cents
        # Each 0.5 point of line movement ~ 2-3% edge shift
        clv_cents = clv_raw * 4.0  # rough approximation

        return {
            "clv_raw": round(clv_raw, 3),
            "clv_cents": round(clv_cents, 2),
            "beat_close": clv_raw > 0,
        }


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    )
    import argparse
    parser = argparse.ArgumentParser(description="Golf Closing Line Worker")
    parser.add_argument("--loop", action="store_true", help="Run in loop mode")
    args = parser.parse_args()

    worker = ClosingWorker()
    if args.loop:
        worker.run_loop()
    else:
        success = worker.run_once()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
