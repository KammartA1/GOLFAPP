"""Signal Worker — Evaluate lines and generate betting signals.

Runs after odds_worker scrapes new lines. For each active line:
1. Run model projection
2. Compute edge (model_prob - market_prob)
3. Check system state + circuit breakers
4. Generate Signal (approved/rejected)
5. If approved, optionally auto-place bet

Writes to: signals table
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from database.connection import DatabaseManager
from database.models import LineMovement, Signal, Bet, ModelVersion
from .base_worker import BaseWorker

logger = logging.getLogger(__name__)


class SignalWorker(BaseWorker):
    WORKER_NAME = "signal_worker_golf"
    SPORT = "golf"
    DEFAULT_INTERVAL_SECONDS = 960  # 16 min (offset from odds worker)

    def __init__(self, interval_seconds: int | None = None, auto_bet: bool = False):
        super().__init__(interval_seconds)
        self.auto_bet = auto_bet
        self._engine = None

    def _get_engine(self):
        """Lazy-load the QuantEngine."""
        if self._engine is None:
            try:
                from quant_system.engine import QuantEngine
                from quant_system.core.types import Sport
                self._engine = QuantEngine(sport=Sport.GOLF)
            except ImportError:
                logger.error("QuantEngine not available — signals will be model-only")
        return self._engine

    def execute(self) -> dict:
        """Evaluate all active lines and generate signals."""
        results = {"lines_evaluated": 0, "signals_generated": 0, "signals_approved": 0, "bets_placed": 0}

        # Get active lines from last scrape cycle (last 30 min)
        cutoff = datetime.utcnow() - timedelta(minutes=30)

        with DatabaseManager.session_scope() as session:
            active_lines = (
                session.query(LineMovement)
                .filter(
                    LineMovement.sport == "golf",
                    LineMovement.is_active == True,
                    LineMovement.captured_at >= cutoff,
                )
                .all()
            )

            if not active_lines:
                logger.info("No active lines to evaluate")
                return results

            # Get active model version
            active_model = (
                session.query(ModelVersion)
                .filter_by(sport="golf", is_active=True)
                .order_by(ModelVersion.trained_at.desc())
                .first()
            )
            model_version_id = active_model.id if active_model else None

            engine = self._get_engine()

            for line in active_lines:
                results["lines_evaluated"] += 1

                try:
                    signal_result = self._evaluate_line(line, engine, model_version_id, session)
                    if signal_result:
                        results["signals_generated"] += 1
                        if signal_result.approved:
                            results["signals_approved"] += 1

                except Exception:
                    logger.exception("Failed to evaluate line: %s %s %s",
                                     line.player, line.stat_type, line.line)

        return results

    def _evaluate_line(
        self,
        line: LineMovement,
        engine,
        model_version_id: int | None,
        session,
    ) -> Signal | None:
        """Evaluate a single line and create a Signal."""
        # Skip lines with no stat type
        if not line.stat_type or line.stat_type == "unknown":
            return None

        # Check if signal already exists for this line
        existing = (
            session.query(Signal)
            .filter(
                Signal.player == line.player,
                Signal.stat_type == line.stat_type,
                Signal.line == line.line,
                Signal.source == line.source,
                Signal.generated_at >= datetime.utcnow() - timedelta(hours=2),
            )
            .first()
        )
        if existing:
            return None  # Already evaluated recently

        # Compute market implied probability
        market_prob = 0.5  # Default for PrizePicks (no juice lines)
        if line.over_prob_implied:
            market_prob = line.over_prob_implied

        # Get model projection
        model_prob, model_projection, model_std = self._get_model_output(
            line.player, line.stat_type, line.line
        )

        if model_prob is None:
            return None  # No model output available

        edge = model_prob - market_prob
        direction = "over" if model_prob > 0.5 else "under"

        # Create signal
        signal = Signal(
            sport="golf",
            generated_at=datetime.utcnow(),
            event_id=line.event_id,
            player=line.player,
            stat_type=line.stat_type,
            line=line.line,
            direction=direction,
            source=line.source,
            model_prob=model_prob,
            market_prob=market_prob,
            edge=edge,
            model_projection=model_projection,
            model_std=model_std,
            model_version_id=model_version_id,
        )

        # Run through QuantEngine for approval
        if engine and abs(edge) >= 0.03:  # Only evaluate if edge >= 3%
            try:
                from quant_system.core.types import BetType
                decision = engine.evaluate_bet(
                    player=line.player,
                    bet_type=BetType.OVER if direction == "over" else BetType.UNDER,
                    stat_type=line.stat_type,
                    line=line.line,
                    direction=direction,
                    model_prob=model_prob,
                    model_projection=model_projection or 0.0,
                    model_std=model_std or 1.0,
                    confidence_score=signal.confidence_score,
                    engine_agreement=signal.engine_agreement,
                )

                signal.approved = decision["approved"]
                signal.rejection_reason = decision.get("rejection_reason", "")
                signal.recommended_stake = decision.get("stake", 0.0)

                sharp_signal = decision.get("sharp_signal", {})
                signal.sharp_agrees = sharp_signal.get("agrees", None)
                signal.sharp_confidence = sharp_signal.get("confidence_multiplier", None)

            except Exception:
                logger.exception("QuantEngine evaluation failed for %s", line.player)
                signal.approved = None  # Unevaluated
        else:
            # Below edge threshold
            signal.approved = False
            signal.rejection_reason = f"Edge {edge:.3f} below 3% threshold"

        session.add(signal)
        session.flush()

        return signal

    def _get_model_output(
        self, player: str, stat_type: str, line: float
    ) -> tuple[float | None, float | None, float | None]:
        """Get model probability, projection, and std for a player/stat.

        Integrates with existing projection system.
        """
        try:
            from database.models import Projection, Player as PlayerModel
            from database.connection import get_session

            session = get_session()
            try:
                # Find latest projection for this player
                player_record = (
                    session.query(PlayerModel)
                    .filter(PlayerModel.normalized_name == player.strip().lower())
                    .first()
                )

                if not player_record:
                    return None, None, None

                proj = (
                    session.query(Projection)
                    .filter_by(player_id=player_record.id)
                    .order_by(Projection.updated_at.desc())
                    .first()
                )

                if not proj:
                    return None, None, None

                # Map stat_type to projection field
                projection_value = self._map_stat_to_projection(proj, stat_type)
                if projection_value is None:
                    return None, None, None

                # Simple probability estimate: P(actual > line) using normal approx
                import scipy.stats as stats
                std_est = abs(projection_value) * 0.25 + 0.5  # Rough std estimate
                prob_over = 1.0 - stats.norm.cdf(line, loc=projection_value, scale=std_est)

                return prob_over, projection_value, std_est

            finally:
                session.close()

        except Exception:
            logger.debug("Model output unavailable for %s/%s", player, stat_type)
            return None, None, None

    def _map_stat_to_projection(self, proj, stat_type: str) -> float | None:
        """Map a stat_type string to a projection field."""
        stat_map = {
            "birdies": None,  # Derived from scoring
            "bogeys": None,
            "pars": None,
            "fantasy_score": getattr(proj, "projected_dfs_points", None),
            "sg_total": getattr(proj, "projected_sg_total", None),
        }
        return stat_map.get(stat_type.lower())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    worker = SignalWorker()
    worker.run_forever()
