"""
Golf Quant Engine -- Signal Worker
====================================
Generates betting signals when new odds data is available.

For each active tournament with fresh line_movements:
  1. Looks up player SG stats
  2. Computes projection via the SGModel
  3. Applies course fit and weather adjustments
  4. Calculates edge vs current market line
  5. Writes Signal records for downstream consumption

Runs every 30 minutes during tournament weeks.

Run standalone:
    python -m workers.signal_worker
"""
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workers.base import BaseWorker
from database.connection import get_session_factory, init_db
from database.models import (
    LineMovement, Signal, Event, Player, SGStat, WorkerStatus,
)

log = logging.getLogger(__name__)

# Golf-specific stat types where we can generate over/under signals
PROJECTABLE_MARKETS = {
    "birdies", "birdies_or_better", "strokes_total", "bogey_free_holes",
    "pars_or_better", "fantasy_score", "holes_under_par",
    "greens_in_regulation", "fairways_hit", "eagles",
}

# Map PrizePicks stat types to SG-derived projection methods
STAT_SG_MAP = {
    "birdies": "birdie_proj",
    "birdies_or_better": "birdie_proj",
    "strokes_total": "strokes_proj",
    "fantasy_score": "fantasy_proj",
    "bogey_free_holes": "bogey_free_proj",
    "pars_or_better": "pars_proj",
    "holes_under_par": "holes_under_par_proj",
}


class SignalWorker(BaseWorker):
    name = "signal_worker"
    interval_seconds = int(os.environ.get("SIGNAL_WORKER_INTERVAL", 1800))  # 30 min
    max_retries = 2
    retry_delay = 15.0
    description = "Generates signals from new line movements and SG projections"

    def execute(self) -> dict:
        init_db()
        factory = get_session_factory()
        session = factory()

        signals_created = 0
        events_processed = 0

        try:
            # 1. Find the last time this worker ran successfully
            last_run = self._get_last_success_time(session)

            # 2. Find active events with new line movements since last run
            active_events = (
                session.query(Event)
                .filter(Event.sport == "GOLF")
                .filter(Event.status.in_(["scheduled", "live"]))
                .all()
            )

            if not active_events:
                self._logger.info("No active golf events found")
                return {"items_processed": 0, "events_checked": 0}

            for event in active_events:
                # Check if there are new line movements for this event
                new_lines_query = (
                    session.query(LineMovement)
                    .filter(LineMovement.event == event.event_name)
                    .filter(LineMovement.sport == "GOLF")
                )
                if last_run:
                    new_lines_query = new_lines_query.filter(
                        LineMovement.timestamp >= last_run
                    )

                new_lines = new_lines_query.all()
                if not new_lines:
                    continue

                self._logger.info(
                    "Event [%s]: %d new lines since %s",
                    event.event_name, len(new_lines),
                    last_run.isoformat() if last_run else "beginning",
                )

                # 3. Get unique players from new lines
                players_in_lines = set()
                lines_by_player: dict[str, list] = {}
                for lm in new_lines:
                    if lm.player:
                        players_in_lines.add(lm.player)
                        lines_by_player.setdefault(lm.player, []).append(lm)

                # 4. Get weather data for course
                weather_impact = self._get_weather_impact(session, event)

                # 5. Get course info from event metadata
                course_name = event.course_name or event.venue or ""

                # 6. Generate signals for each player
                for player_name in players_in_lines:
                    player_signals = self._generate_player_signals(
                        session=session,
                        event=event,
                        player_name=player_name,
                        player_lines=lines_by_player.get(player_name, []),
                        course_name=course_name,
                        weather_impact=weather_impact,
                    )
                    signals_created += player_signals

                events_processed += 1

            session.commit()
            self._logger.info(
                "Signal generation complete: %d signals across %d events",
                signals_created, events_processed,
            )

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return {
            "items_processed": signals_created,
            "events_processed": events_processed,
        }

    # ------------------------------------------------------------------ #
    # Signal generation per player
    # ------------------------------------------------------------------ #
    def _generate_player_signals(
        self,
        session,
        event: Event,
        player_name: str,
        player_lines: list,
        course_name: str,
        weather_impact: dict,
    ) -> int:
        """Generate signals for a single player.  Returns count of signals created."""
        # Look up player's SG stats
        player_row = (
            session.query(Player)
            .filter_by(name=player_name, sport="GOLF")
            .first()
        )
        if player_row is None:
            return 0

        sg_stats = (
            session.query(SGStat)
            .filter_by(player_id=player_row.id)
            .order_by(SGStat.created_at.desc())
            .limit(50)
            .all()
        )

        # Build projection from SG history
        sg_projection = self._compute_sg_projection(sg_stats, course_name)
        if sg_projection is None:
            return 0

        # Weather adjustment
        weather_adj = 0.0
        if weather_impact.get("is_significant"):
            weather_adj = (weather_impact.get("projection_mult", 1.0) - 1.0) * sg_projection["sg_total"]

        adjusted_sg_total = sg_projection["sg_total"] + weather_adj

        count = 0
        for lm in player_lines:
            market = lm.market or ""

            # For outright / winner markets, signal is based on win probability
            if market in ("outrights", "outright_winner", "h2h"):
                signal = self._outright_signal(
                    session, event, player_name, lm, adjusted_sg_total, sg_projection
                )
            elif market.lower() in PROJECTABLE_MARKETS:
                signal = self._prop_signal(
                    lm, adjusted_sg_total, sg_projection, weather_impact
                )
            else:
                continue

            if signal is None:
                continue

            # Only store signals with meaningful edge
            edge = signal.get("edge_pct", 0)
            if abs(edge) < 0.02:  # less than 2% edge -- not actionable
                continue

            sig_row = Signal(
                sport="GOLF",
                event=event.event_name,
                market=market,
                player=player_name,
                signal_value=signal.get("model_projection", 0.0),
                confidence=signal.get("confidence", 0.0),
                direction=signal.get("direction", ""),
                edge_pct=edge,
                model_version=signal.get("model_version", "8.0"),
                generated_at=datetime.utcnow(),
            )
            session.add(sig_row)
            count += 1

        return count

    # ------------------------------------------------------------------ #
    # SG projection
    # ------------------------------------------------------------------ #
    def _compute_sg_projection(self, sg_stats: list, course_name: str) -> dict | None:
        """Compute a projection from the player's SG stat history."""
        if not sg_stats:
            return None

        # Build history list for the SGModel
        history = []
        for s in sg_stats:
            history.append({
                "sg_total": s.sg_total or 0.0,
                "sg_ott": s.sg_ott or 0.0,
                "sg_app": s.sg_app or 0.0,
                "sg_atg": s.sg_atg or 0.0,
                "sg_putt": s.sg_putt or 0.0,
                "event_date": s.created_at.isoformat() if s.created_at else "2024-01-01",
            })

        if not history:
            return None

        try:
            from models.strokes_gained import SGModel
            model = SGModel()
            projection = model.project_player(history, course_name)
            return {
                "sg_total": projection.get("proj_sg_total", 0.0),
                "sg_ott": projection.get("proj_sg_ott", 0.0),
                "sg_app": projection.get("proj_sg_app", 0.0),
                "sg_atg": projection.get("proj_sg_atg", 0.0),
                "sg_putt": projection.get("proj_sg_putt", 0.0),
                "course_fit": projection.get("course_fit_score", 50.0),
                "variance": projection.get("player_variance", 2.75),
                "form_trend": projection.get("form_trend", "stable"),
                "data_quality": projection.get("data_quality", "low"),
                "events_played": projection.get("events_played", 0),
            }
        except Exception as exc:
            self._logger.warning("SG projection failed: %s", exc)
            # Fallback: simple average of recent stats
            recent = sg_stats[:12]
            totals = [s.sg_total for s in recent if s.sg_total is not None]
            if not totals:
                return None
            avg = float(np.mean(totals))
            return {
                "sg_total": avg,
                "sg_ott": 0.0,
                "sg_app": 0.0,
                "sg_atg": 0.0,
                "sg_putt": 0.0,
                "course_fit": 50.0,
                "variance": 2.75,
                "form_trend": "unknown",
                "data_quality": "low",
                "events_played": len(recent),
            }

    # ------------------------------------------------------------------ #
    # Outright / winner signal
    # ------------------------------------------------------------------ #
    def _outright_signal(
        self, session, event, player_name, lm, adj_sg_total, sg_proj
    ) -> dict | None:
        """Generate signal for outright/winner markets."""
        odds_decimal = lm.odds if lm.odds and lm.odds > 1 else 2.0
        implied_prob = 1.0 / odds_decimal if odds_decimal > 0 else 0.0

        # Estimate win probability from SG projection
        # Get field SG values
        field_sgs = (
            session.query(SGStat.sg_total)
            .join(Player, SGStat.player_id == Player.id)
            .filter(Player.sport == "GOLF", Player.active == True)
            .filter(SGStat.sg_total.isnot(None))
            .order_by(SGStat.created_at.desc())
            .limit(156)
            .all()
        )
        field_values = [row[0] for row in field_sgs if row[0] is not None]

        if not field_values:
            return None

        # Simplified win probability calculation
        field_mean = float(np.mean(field_values))
        field_std = max(float(np.std(field_values)), 0.5)
        tournament_std = 2.75 * np.sqrt(4) / 2  # 4 rounds, correlated

        combined_std = np.sqrt(field_std ** 2 + tournament_std ** 2)
        z = (adj_sg_total - field_mean) / combined_std if combined_std > 0 else 0

        from scipy import stats as sp_stats
        raw_win_p = 1 - sp_stats.norm.cdf(0, loc=z, scale=1) if z > 0 else 0.005
        n_players = max(len(field_values), 50)
        model_win_prob = min(raw_win_p / n_players * 1.5, 0.55)
        model_win_prob = max(0.001, model_win_prob)

        edge = model_win_prob - implied_prob

        if edge > 0:
            direction = "back"
            confidence = min(edge / 0.10, 1.0)
        else:
            direction = "fade"
            confidence = min(abs(edge) / 0.10, 1.0)

        return {
            "model_projection": model_win_prob,
            "edge_pct": round(edge, 4),
            "direction": direction,
            "confidence": round(confidence, 3),
            "model_version": "8.0",
        }

    # ------------------------------------------------------------------ #
    # Prop signal (over/under)
    # ------------------------------------------------------------------ #
    def _prop_signal(self, lm, adj_sg_total, sg_proj, weather_impact) -> dict | None:
        """Generate signal for prop markets (birdies, strokes, etc.)."""
        market = (lm.market or "").lower()
        line_value = lm.line if lm.line else 0

        if not line_value or line_value == 0:
            return None

        # Convert SG to stat-specific projection
        projection = self._sg_to_stat_projection(market, adj_sg_total, sg_proj)
        if projection is None:
            return None

        model_proj = projection["mean"]
        model_std = projection["std"]

        # Apply weather adjustment to variance
        variance_mult = weather_impact.get("variance_mult", 1.0)
        model_std *= np.sqrt(variance_mult)

        # Calculate probability of over
        if model_std > 0:
            from scipy import stats as sp_stats
            z = (line_value - model_proj) / model_std
            prob_over = 1 - sp_stats.norm.cdf(z)
        else:
            prob_over = 0.5

        # PrizePicks implied is ~52.4% for both sides
        implied_over = 0.524
        implied_under = 0.524

        edge_over = prob_over - implied_over
        edge_under = (1 - prob_over) - implied_under

        if abs(edge_over) >= abs(edge_under):
            direction = "over"
            edge = edge_over
            confidence = min(abs(edge) / 0.08, 1.0)
        else:
            direction = "under"
            edge = edge_under
            confidence = min(abs(edge) / 0.08, 1.0)

        # Adjust confidence based on data quality
        quality_mult = {"high": 1.0, "medium": 0.85, "low": 0.7, "very_low": 0.5, "none": 0.3}
        confidence *= quality_mult.get(sg_proj.get("data_quality", "low"), 0.7)

        return {
            "model_projection": round(model_proj, 2),
            "edge_pct": round(edge, 4),
            "direction": direction,
            "confidence": round(min(confidence, 1.0), 3),
            "model_version": "8.0",
        }

    # ------------------------------------------------------------------ #
    # SG -> stat projection mapping
    # ------------------------------------------------------------------ #
    def _sg_to_stat_projection(self, market: str, sg_total: float, sg_proj: dict) -> dict | None:
        """Convert SG projection to a specific stat projection.

        Uses empirical relationships between strokes gained and counting stats.
        All relationships are derived from PGA Tour statistical correlations.
        """
        sg_ott = sg_proj.get("sg_ott", 0)
        sg_app = sg_proj.get("sg_app", 0)
        sg_putt = sg_proj.get("sg_putt", 0)
        sg_atg = sg_proj.get("sg_atg", 0)

        if market in ("birdies", "birdies_or_better"):
            # Tour avg ~3.5 birdies per round; +1 SG total ~ +0.7 birdies
            # Putting and approach are strongest predictors
            base = 3.5
            adj = sg_total * 0.7 + sg_putt * 0.15 + sg_app * 0.1
            return {"mean": round(base + adj, 2), "std": 1.4}

        elif market in ("strokes_total", "strokes"):
            # Tour avg = par (72); +1 SG total = -1 stroke from par
            base = 72.0
            adj = -sg_total * 1.0
            return {"mean": round(base + adj, 2), "std": 3.0}

        elif market == "fantasy_score":
            # DK scoring: base ~37 pts; +1 SG ~ +9.2 pts
            base = 37.0
            adj = sg_total * 9.2
            return {"mean": round(base + adj, 2), "std": 12.0}

        elif market in ("bogey_free_holes",):
            # Tour avg ~13 bogey-free holes per 18; better SG -> more clean holes
            base = 13.0
            adj = sg_total * 0.8 + sg_putt * 0.2 + sg_atg * 0.15
            return {"mean": round(base + adj, 2), "std": 2.5}

        elif market in ("pars_or_better",):
            # Tour avg ~14.5 pars or better per round
            base = 14.5
            adj = sg_total * 0.5 + sg_app * 0.2
            return {"mean": round(base + adj, 2), "std": 1.8}

        elif market in ("holes_under_par",):
            # Tour avg ~3.5 holes under par per round
            base = 3.5
            adj = sg_total * 0.65 + sg_putt * 0.15
            return {"mean": round(base + adj, 2), "std": 1.5}

        elif market in ("greens_in_regulation", "gir"):
            # Tour avg ~66% GIR = ~12/18; approach SG is the main driver
            base = 12.0
            adj = sg_app * 1.2 + sg_ott * 0.4
            return {"mean": round(base + adj, 2), "std": 2.0}

        elif market in ("fairways_hit",):
            # Tour avg ~60% = ~8.4/14; OTT is the driver
            base = 8.4
            adj = sg_ott * 0.8
            return {"mean": round(base + adj, 2), "std": 2.0}

        elif market in ("eagles",):
            # Rare event: ~0.15 per round on average
            base = 0.15
            adj = sg_total * 0.05 + sg_ott * 0.03
            return {"mean": round(max(base + adj, 0.01), 3), "std": 0.4}

        return None

    # ------------------------------------------------------------------ #
    # Weather helper
    # ------------------------------------------------------------------ #
    def _get_weather_impact(self, session, event: Event) -> dict:
        """Fetch weather impact for the event's course."""
        try:
            from scrapers.weather_scraper import weather_impact_factor
            from database.db_manager import get_current_weather

            meta = json.loads(event.metadata_json) if event.metadata_json else {}
            course_name = event.course_name or event.venue or ""

            weather = get_current_weather(event.event_name)
            if not weather:
                return {"variance_mult": 1.0, "projection_mult": 1.0, "is_significant": False}

            wind = weather.get("wind_speed_mph", 0)
            precip = weather.get("precipitation_mm", 0)
            return weather_impact_factor(wind, precip)

        except Exception as exc:
            self._logger.debug("Weather lookup failed: %s", exc)
            return {"variance_mult": 1.0, "projection_mult": 1.0, "is_significant": False}

    # ------------------------------------------------------------------ #
    # Helper
    # ------------------------------------------------------------------ #
    def _get_last_success_time(self, session) -> datetime | None:
        row = session.query(WorkerStatus).filter_by(worker_name=self.name).first()
        if row and row.last_success:
            return row.last_success
        # Default: look back 2 hours
        return datetime.utcnow() - timedelta(hours=2)


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    )
    import argparse
    parser = argparse.ArgumentParser(description="Golf Signal Worker")
    parser.add_argument("--loop", action="store_true", help="Run in loop mode")
    args = parser.parse_args()

    worker = SignalWorker()
    if args.loop:
        worker.run_loop()
    else:
        success = worker.run_once()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
