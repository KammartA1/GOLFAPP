"""Kill switch — Automated system halt with 6 independent kill conditions.

Each condition is evaluated independently. ANY single condition triggering
will halt the system. This is the last line of defense.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from database.connection import DatabaseManager
from database.models import (
    Bet, CLVLog, EdgeReport, SystemStateLog, ModelVersion, CalibrationLog,
)
from quant_system.core.types import SystemState

logger = logging.getLogger(__name__)


class KillConditionResult:
    """Result of evaluating a single kill condition."""

    __slots__ = ("name", "triggered", "severity", "message", "value", "threshold")

    def __init__(
        self,
        name: str,
        triggered: bool,
        severity: str,
        message: str,
        value: float = 0.0,
        threshold: float = 0.0,
    ):
        self.name = name
        self.triggered = triggered
        self.severity = severity          # "warning", "critical", "fatal"
        self.message = message
        self.value = value
        self.threshold = threshold

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "triggered": self.triggered,
            "severity": self.severity,
            "message": self.message,
            "value": round(self.value, 4),
            "threshold": round(self.threshold, 4),
        }


class KillSwitch:
    """Automated system halt with 6 independent kill conditions.

    Kill Conditions:
        1. CLV Death           — CLV consistently negative (edge gone)
        2. Model Worse Than Market — Model predictions worse than naive market
        3. Edge Decay          — Statistically significant declining trend
        4. Execution Destroys Edge — Slippage/limits eat all theoretical edge
        5. Max Drawdown        — Bankroll drawdown exceeds catastrophic threshold
        6. Variance Blowup     — Results far outside expected distribution

    Philosophy: It is better to stop betting and investigate than to continue
    losing money. False positives (stopping when edge exists) are much less
    costly than false negatives (continuing when edge is dead).
    """

    # ── Thresholds ──────────────────────────────────────────────────────
    CLV_DEATH_WINDOW = 100          # Evaluate over last N bets
    CLV_DEATH_AVG_THRESHOLD = -1.0  # Avg CLV below this → dead
    CLV_DEATH_BEAT_RATE = 0.42      # Beat-rate below this → dead
    CLV_DEATH_PCT_NEGATIVE = 0.65   # >65% negative CLV → dead

    MODEL_BRIER_THRESHOLD = 0.28    # Worse than this Brier score → fatal
    MODEL_VS_MARKET_THRESHOLD = 0.0 # If model Brier > market Brier → fatal

    EDGE_DECAY_MIN_BETS = 100       # Need this many bets for decay detection
    EDGE_DECAY_SLOPE_THRESHOLD = -0.02  # Slope steeper than this → fatal

    EXECUTION_MIN_BETS = 50
    EXECUTION_FRICTION_THRESHOLD = 1.0  # If avg friction > avg CLV → fatal

    MAX_DRAWDOWN_WARNING = 0.15     # 15% drawdown → warning
    MAX_DRAWDOWN_CRITICAL = 0.25    # 25% drawdown → critical
    MAX_DRAWDOWN_FATAL = 0.35       # 35% drawdown → kill

    VARIANCE_Z_THRESHOLD = 3.0      # Results >3 std devs from expected → fatal
    VARIANCE_MIN_BETS = 30

    def __init__(self, sport: str = "golf"):
        self.sport = sport

    # ══════════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════════════

    def check_all(self) -> dict:
        """Evaluate ALL 6 kill conditions.

        Returns:
            {
                'system_active': bool,
                'recommended_state': str,
                'halt_reasons': list of str,
                'conditions': list of KillConditionResult dicts,
                'timestamp': str,
            }
        """
        conditions: List[KillConditionResult] = [
            self._check_clv_death(),
            self._check_model_worse_than_market(),
            self._check_edge_decay(),
            self._check_execution_destroys_edge(),
            self._check_max_drawdown(),
            self._check_variance_blowup(),
        ]

        fatal = [c for c in conditions if c.triggered and c.severity == "fatal"]
        critical = [c for c in conditions if c.triggered and c.severity == "critical"]
        warnings = [c for c in conditions if c.triggered and c.severity == "warning"]

        # Determine recommended state
        if fatal:
            recommended_state = SystemState.KILLED.value
        elif critical:
            recommended_state = SystemState.SUSPENDED.value
        elif warnings:
            recommended_state = SystemState.REDUCED.value
        else:
            recommended_state = SystemState.ACTIVE.value

        halt_reasons = [c.message for c in conditions if c.triggered]

        system_active = recommended_state == SystemState.ACTIVE.value

        result = {
            "system_active": system_active,
            "recommended_state": recommended_state,
            "halt_reasons": halt_reasons,
            "n_fatal": len(fatal),
            "n_critical": len(critical),
            "n_warnings": len(warnings),
            "conditions": [c.to_dict() for c in conditions],
            "timestamp": datetime.utcnow().isoformat(),
        }

        if not system_active:
            logger.warning(
                "Kill switch recommends %s: %s",
                recommended_state,
                "; ".join(halt_reasons),
            )

        return result

    def is_system_active(self) -> bool:
        """Quick check: should the system keep betting?"""
        result = self.check_all()
        return result["system_active"]

    def get_halt_reason(self) -> Optional[str]:
        """Get the primary halt reason, or None if system is active."""
        result = self.check_all()
        if result["system_active"]:
            return None
        return "; ".join(result["halt_reasons"])

    def log_state_change(
        self,
        previous_state: str,
        new_state: str,
        reason: str,
        bankroll: float = 0.0,
    ) -> None:
        """Persist state change to audit log."""
        with DatabaseManager.session_scope() as session:
            # Get current CLV for context
            clv_record = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport)
                .order_by(CLVLog.calculated_at.desc())
                .first()
            )
            clv_val = clv_record.clv_cents if clv_record else None

            # Get current drawdown
            peak_report = (
                session.query(EdgeReport)
                .filter(EdgeReport.sport == self.sport)
                .order_by(EdgeReport.report_date.desc())
                .first()
            )
            drawdown = peak_report.drawdown_pct if peak_report else None

            log_entry = SystemStateLog(
                sport=self.sport,
                timestamp=datetime.utcnow(),
                previous_state=previous_state,
                new_state=new_state,
                reason=reason,
                clv_at_change=clv_val,
                bankroll_at_change=bankroll,
                drawdown_at_change=drawdown,
            )
            session.add(log_entry)

    # ══════════════════════════════════════════════════════════════════════
    #  CONDITION 1: CLV Death
    # ══════════════════════════════════════════════════════════════════════

    def _check_clv_death(self) -> KillConditionResult:
        """CLV consistently negative = edge is gone.

        This is the gold standard indicator. If we cannot beat the closing
        line, we have no edge, period.
        """
        with DatabaseManager.session_scope() as session:
            records = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport)
                .order_by(CLVLog.calculated_at.desc())
                .limit(self.CLV_DEATH_WINDOW)
                .all()
            )

            if len(records) < 30:
                return KillConditionResult(
                    name="clv_death",
                    triggered=False,
                    severity="warning",
                    message="Insufficient CLV data for evaluation",
                )

            clv_values = np.array([r.clv_cents for r in records])
            avg_clv = float(np.mean(clv_values))
            beat_rate = float(np.mean([1.0 if r.beat_close else 0.0 for r in records]))
            pct_negative = float(np.mean(clv_values < 0))

            # Fatal: avg CLV deeply negative AND poor beat rate
            if avg_clv < self.CLV_DEATH_AVG_THRESHOLD and beat_rate < self.CLV_DEATH_BEAT_RATE:
                return KillConditionResult(
                    name="clv_death",
                    triggered=True,
                    severity="fatal",
                    message=(
                        f"CLV death: avg={avg_clv:.2f}c, beat_rate={beat_rate:.1%}, "
                        f"{pct_negative:.0%} negative over last {len(records)} bets"
                    ),
                    value=avg_clv,
                    threshold=self.CLV_DEATH_AVG_THRESHOLD,
                )

            # Critical: avg CLV negative AND high pct negative
            if avg_clv < 0 and pct_negative > self.CLV_DEATH_PCT_NEGATIVE:
                return KillConditionResult(
                    name="clv_death",
                    triggered=True,
                    severity="critical",
                    message=(
                        f"CLV critical: avg={avg_clv:.2f}c, "
                        f"{pct_negative:.0%} negative over last {len(records)} bets"
                    ),
                    value=avg_clv,
                    threshold=0.0,
                )

            # Warning: avg CLV barely positive
            if avg_clv < 0.3:
                return KillConditionResult(
                    name="clv_death",
                    triggered=True,
                    severity="warning",
                    message=f"CLV marginal: avg={avg_clv:.2f}c (threshold 0.3c)",
                    value=avg_clv,
                    threshold=0.3,
                )

            return KillConditionResult(
                name="clv_death",
                triggered=False,
                severity="warning",
                message=f"CLV healthy: avg={avg_clv:.2f}c, beat_rate={beat_rate:.1%}",
                value=avg_clv,
                threshold=self.CLV_DEATH_AVG_THRESHOLD,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  CONDITION 2: Model Worse Than Market
    # ══════════════════════════════════════════════════════════════════════

    def _check_model_worse_than_market(self) -> KillConditionResult:
        """Model predictions are worse than naive market-implied probabilities.

        Uses Brier score comparison. If our model cannot outperform the
        market's implied probabilities, we have no informational edge.
        """
        with DatabaseManager.session_scope() as session:
            bets = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.status.in_(["won", "lost"]),
                )
                .order_by(Bet.settled_at.desc())
                .limit(200)
                .all()
            )

            if len(bets) < 50:
                return KillConditionResult(
                    name="model_worse_than_market",
                    triggered=False,
                    severity="warning",
                    message="Insufficient settled bets for model evaluation",
                )

            model_brier_sum = 0.0
            market_brier_sum = 0.0
            n = 0

            for bet in bets:
                outcome = 1.0 if bet.status == "won" else 0.0
                model_prob = bet.model_prob
                market_prob = bet.market_prob

                if model_prob is None or market_prob is None:
                    continue
                if not (0 < model_prob < 1) or not (0 < market_prob < 1):
                    continue

                model_brier_sum += (model_prob - outcome) ** 2
                market_brier_sum += (market_prob - outcome) ** 2
                n += 1

            if n < 30:
                return KillConditionResult(
                    name="model_worse_than_market",
                    triggered=False,
                    severity="warning",
                    message="Insufficient valid probabilities for Brier comparison",
                )

            model_brier = model_brier_sum / n
            market_brier = market_brier_sum / n
            brier_diff = model_brier - market_brier  # Negative = model is better

            # Fatal: model is strictly worse than market
            if brier_diff > 0.01 and model_brier > self.MODEL_BRIER_THRESHOLD:
                return KillConditionResult(
                    name="model_worse_than_market",
                    triggered=True,
                    severity="fatal",
                    message=(
                        f"Model worse than market: model_brier={model_brier:.4f}, "
                        f"market_brier={market_brier:.4f}, diff={brier_diff:+.4f}"
                    ),
                    value=brier_diff,
                    threshold=self.MODEL_VS_MARKET_THRESHOLD,
                )

            # Critical: model barely better or equal
            if brier_diff > -0.005:
                return KillConditionResult(
                    name="model_worse_than_market",
                    triggered=True,
                    severity="critical",
                    message=(
                        f"Model barely outperforms market: diff={brier_diff:+.4f} "
                        f"(model={model_brier:.4f}, market={market_brier:.4f})"
                    ),
                    value=brier_diff,
                    threshold=-0.005,
                )

            # Warning: model advantage is thin
            if brier_diff > -0.02:
                return KillConditionResult(
                    name="model_worse_than_market",
                    triggered=True,
                    severity="warning",
                    message=(
                        f"Model advantage thin: diff={brier_diff:+.4f} "
                        f"(model={model_brier:.4f}, market={market_brier:.4f})"
                    ),
                    value=brier_diff,
                    threshold=-0.02,
                )

            return KillConditionResult(
                name="model_worse_than_market",
                triggered=False,
                severity="warning",
                message=(
                    f"Model outperforms market: diff={brier_diff:+.4f} "
                    f"(model={model_brier:.4f}, market={market_brier:.4f})"
                ),
                value=brier_diff,
                threshold=self.MODEL_VS_MARKET_THRESHOLD,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  CONDITION 3: Edge Decay
    # ══════════════════════════════════════════════════════════════════════

    def _check_edge_decay(self) -> KillConditionResult:
        """Statistically significant declining trend in CLV.

        Uses linear regression on CLV time series. If slope is significantly
        negative, edge is decaying and will reach zero.
        """
        with DatabaseManager.session_scope() as session:
            records = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport)
                .order_by(CLVLog.calculated_at.asc())
                .limit(500)
                .all()
            )

            if len(records) < self.EDGE_DECAY_MIN_BETS:
                return KillConditionResult(
                    name="edge_decay",
                    triggered=False,
                    severity="warning",
                    message=f"Insufficient data ({len(records)}/{self.EDGE_DECAY_MIN_BETS}) for decay detection",
                )

            clv_values = np.array([r.clv_cents for r in records])
            x = np.arange(len(clv_values))

            from scipy import stats as sp_stats
            slope, intercept, r_val, p_val, std_err = sp_stats.linregress(x, clv_values)

            # Projected CLV at current point
            current_projected = slope * len(clv_values) + intercept

            # Bets until CLV reaches zero (if declining)
            if slope < 0 and current_projected > 0:
                bets_to_zero = int(-current_projected / slope)
            else:
                bets_to_zero = -1  # Not approaching zero

            is_significant = p_val < 0.05

            # Fatal: steep negative slope that is statistically significant
            if slope < self.EDGE_DECAY_SLOPE_THRESHOLD and is_significant:
                return KillConditionResult(
                    name="edge_decay",
                    triggered=True,
                    severity="fatal",
                    message=(
                        f"Edge decay fatal: slope={slope:.4f}/bet (p={p_val:.4f}), "
                        f"projected zero in {bets_to_zero} bets"
                    ),
                    value=slope,
                    threshold=self.EDGE_DECAY_SLOPE_THRESHOLD,
                )

            # Critical: moderate negative slope, significant
            if slope < -0.01 and is_significant:
                return KillConditionResult(
                    name="edge_decay",
                    triggered=True,
                    severity="critical",
                    message=(
                        f"Edge decay critical: slope={slope:.4f}/bet (p={p_val:.4f}), "
                        f"R²={r_val**2:.3f}"
                    ),
                    value=slope,
                    threshold=-0.01,
                )

            # Warning: any significant negative slope
            if slope < 0 and is_significant:
                return KillConditionResult(
                    name="edge_decay",
                    triggered=True,
                    severity="warning",
                    message=(
                        f"Edge decay warning: slope={slope:.4f}/bet (p={p_val:.4f})"
                    ),
                    value=slope,
                    threshold=0.0,
                )

            return KillConditionResult(
                name="edge_decay",
                triggered=False,
                severity="warning",
                message=f"No edge decay detected: slope={slope:.4f} (p={p_val:.4f})",
                value=slope,
                threshold=self.EDGE_DECAY_SLOPE_THRESHOLD,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  CONDITION 4: Execution Destroys Edge
    # ══════════════════════════════════════════════════════════════════════

    def _check_execution_destroys_edge(self) -> KillConditionResult:
        """Execution friction (slippage + limits) exceeds theoretical edge.

        Compares the theoretical edge (from CLV) to the realized edge
        (actual PnL). If the gap is too large, execution costs are
        destroying whatever edge the model has.
        """
        with DatabaseManager.session_scope() as session:
            bets = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.status.in_(["won", "lost"]),
                )
                .order_by(Bet.settled_at.desc())
                .limit(200)
                .all()
            )

            clv_records = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport)
                .order_by(CLVLog.calculated_at.desc())
                .limit(200)
                .all()
            )

            if len(bets) < self.EXECUTION_MIN_BETS or len(clv_records) < self.EXECUTION_MIN_BETS:
                return KillConditionResult(
                    name="execution_destroys_edge",
                    triggered=False,
                    severity="warning",
                    message="Insufficient data for execution analysis",
                )

            # Theoretical edge from CLV
            avg_clv = float(np.mean([r.clv_cents for r in clv_records]))

            # Realized edge from actual PnL
            total_staked = sum(b.stake for b in bets if b.stake > 0)
            total_pnl = sum(b.pnl for b in bets)
            realized_roi = total_pnl / max(total_staked, 1)
            realized_edge_cents = realized_roi * 100  # Convert to cents per dollar

            # Execution friction = theoretical - realized
            friction = avg_clv - realized_edge_cents

            # Fatal: friction exceeds theoretical edge (net negative after execution)
            if avg_clv > 0 and friction > avg_clv * self.EXECUTION_FRICTION_THRESHOLD:
                return KillConditionResult(
                    name="execution_destroys_edge",
                    triggered=True,
                    severity="fatal",
                    message=(
                        f"Execution destroys edge: theoretical={avg_clv:.2f}c, "
                        f"realized={realized_edge_cents:.2f}c, friction={friction:.2f}c"
                    ),
                    value=friction,
                    threshold=avg_clv,
                )

            # Critical: friction is more than 70% of edge
            if avg_clv > 0 and friction > avg_clv * 0.70:
                return KillConditionResult(
                    name="execution_destroys_edge",
                    triggered=True,
                    severity="critical",
                    message=(
                        f"Execution friction high: {friction:.2f}c of {avg_clv:.2f}c "
                        f"theoretical edge ({friction/max(avg_clv,0.01)*100:.0f}%)"
                    ),
                    value=friction,
                    threshold=avg_clv * 0.70,
                )

            # Warning: friction is more than 40% of edge
            if avg_clv > 0 and friction > avg_clv * 0.40:
                return KillConditionResult(
                    name="execution_destroys_edge",
                    triggered=True,
                    severity="warning",
                    message=(
                        f"Execution friction moderate: {friction:.2f}c of {avg_clv:.2f}c "
                        f"({friction/max(avg_clv,0.01)*100:.0f}%)"
                    ),
                    value=friction,
                    threshold=avg_clv * 0.40,
                )

            return KillConditionResult(
                name="execution_destroys_edge",
                triggered=False,
                severity="warning",
                message=(
                    f"Execution OK: theoretical={avg_clv:.2f}c, "
                    f"realized={realized_edge_cents:.2f}c, friction={friction:.2f}c"
                ),
                value=friction,
                threshold=avg_clv * self.EXECUTION_FRICTION_THRESHOLD if avg_clv > 0 else 0.0,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  CONDITION 5: Max Drawdown
    # ══════════════════════════════════════════════════════════════════════

    def _check_max_drawdown(self) -> KillConditionResult:
        """Bankroll drawdown exceeds catastrophic threshold.

        Tracks peak-to-trough drawdown from settled bet PnL.
        """
        with DatabaseManager.session_scope() as session:
            bets = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.status.in_(["won", "lost"]),
                )
                .order_by(Bet.settled_at.asc())
                .all()
            )

            if len(bets) < 20:
                return KillConditionResult(
                    name="max_drawdown",
                    triggered=False,
                    severity="warning",
                    message="Insufficient bet history for drawdown calculation",
                )

            # Build cumulative PnL curve
            pnl_series = np.array([b.pnl for b in bets])
            cumulative = np.cumsum(pnl_series)

            # Estimate starting bankroll from total staked
            total_staked = sum(b.stake for b in bets)
            avg_stake = total_staked / len(bets)
            estimated_bankroll = avg_stake * 20  # Assume ~5% avg bet size

            # Use edge report bankroll if available
            report = (
                session.query(EdgeReport)
                .filter(EdgeReport.sport == self.sport)
                .order_by(EdgeReport.report_date.desc())
                .first()
            )
            if report and report.peak_bankroll and report.peak_bankroll > 0:
                peak_bankroll = report.peak_bankroll
            else:
                peak_bankroll = estimated_bankroll + float(np.max(cumulative))

            # Calculate max drawdown
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative
            max_drawdown_dollars = float(np.max(drawdowns))
            max_drawdown_pct = max_drawdown_dollars / max(peak_bankroll, 1)

            # Current drawdown
            current_drawdown = float(running_max[-1] - cumulative[-1])
            current_drawdown_pct = current_drawdown / max(peak_bankroll, 1)

            # Fatal: catastrophic drawdown
            if current_drawdown_pct >= self.MAX_DRAWDOWN_FATAL:
                return KillConditionResult(
                    name="max_drawdown",
                    triggered=True,
                    severity="fatal",
                    message=(
                        f"Catastrophic drawdown: {current_drawdown_pct:.1%} "
                        f"(${current_drawdown:.0f}). Max ever: {max_drawdown_pct:.1%}"
                    ),
                    value=current_drawdown_pct,
                    threshold=self.MAX_DRAWDOWN_FATAL,
                )

            # Critical: severe drawdown
            if current_drawdown_pct >= self.MAX_DRAWDOWN_CRITICAL:
                return KillConditionResult(
                    name="max_drawdown",
                    triggered=True,
                    severity="critical",
                    message=(
                        f"Severe drawdown: {current_drawdown_pct:.1%} "
                        f"(${current_drawdown:.0f})"
                    ),
                    value=current_drawdown_pct,
                    threshold=self.MAX_DRAWDOWN_CRITICAL,
                )

            # Warning: moderate drawdown
            if current_drawdown_pct >= self.MAX_DRAWDOWN_WARNING:
                return KillConditionResult(
                    name="max_drawdown",
                    triggered=True,
                    severity="warning",
                    message=(
                        f"Moderate drawdown: {current_drawdown_pct:.1%} "
                        f"(${current_drawdown:.0f})"
                    ),
                    value=current_drawdown_pct,
                    threshold=self.MAX_DRAWDOWN_WARNING,
                )

            return KillConditionResult(
                name="max_drawdown",
                triggered=False,
                severity="warning",
                message=(
                    f"Drawdown OK: current={current_drawdown_pct:.1%}, "
                    f"max_ever={max_drawdown_pct:.1%}"
                ),
                value=current_drawdown_pct,
                threshold=self.MAX_DRAWDOWN_WARNING,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  CONDITION 6: Variance Blowup
    # ══════════════════════════════════════════════════════════════════════

    def _check_variance_blowup(self) -> KillConditionResult:
        """Results far outside expected distribution.

        If actual results are >3 standard deviations worse than expected
        (given model probabilities), something is seriously wrong — either
        the model is miscalibrated or the market has changed.
        """
        with DatabaseManager.session_scope() as session:
            bets = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.status.in_(["won", "lost"]),
                )
                .order_by(Bet.settled_at.desc())
                .limit(100)
                .all()
            )

            if len(bets) < self.VARIANCE_MIN_BETS:
                return KillConditionResult(
                    name="variance_blowup",
                    triggered=False,
                    severity="warning",
                    message="Insufficient data for variance analysis",
                )

            # Expected wins from model probabilities
            outcomes = np.array([1.0 if b.status == "won" else 0.0 for b in bets])
            model_probs = np.array([b.model_prob for b in bets])

            # Filter valid probs
            valid_mask = (model_probs > 0.01) & (model_probs < 0.99)
            if np.sum(valid_mask) < self.VARIANCE_MIN_BETS:
                return KillConditionResult(
                    name="variance_blowup",
                    triggered=False,
                    severity="warning",
                    message="Insufficient valid probabilities for variance analysis",
                )

            outcomes = outcomes[valid_mask]
            model_probs = model_probs[valid_mask]
            n = len(outcomes)

            expected_wins = float(np.sum(model_probs))
            actual_wins = float(np.sum(outcomes))

            # Variance of sum of Bernoulli RVs
            variance = float(np.sum(model_probs * (1 - model_probs)))
            std_dev = np.sqrt(variance)

            if std_dev < 0.01:
                return KillConditionResult(
                    name="variance_blowup",
                    triggered=False,
                    severity="warning",
                    message="Standard deviation too small for z-score calculation",
                )

            # Z-score: how many std devs below expected?
            z_score = (actual_wins - expected_wins) / std_dev

            # Also check PnL z-score
            pnl_values = np.array([b.pnl for b in bets])
            pnl_values = pnl_values[valid_mask]
            expected_pnl_per_bet = np.mean(model_probs * (np.array([b.odds_decimal for b in bets])[valid_mask] - 1) - (1 - model_probs))
            actual_pnl_per_bet = float(np.mean(pnl_values / np.array([max(b.stake, 1) for b in bets])[valid_mask]))

            # Fatal: extreme negative z-score on wins
            if z_score < -self.VARIANCE_Z_THRESHOLD:
                return KillConditionResult(
                    name="variance_blowup",
                    triggered=True,
                    severity="fatal",
                    message=(
                        f"Variance blowup: z={z_score:.2f} "
                        f"(expected {expected_wins:.1f} wins, got {actual_wins:.0f} "
                        f"in {n} bets, {self.VARIANCE_Z_THRESHOLD}σ threshold)"
                    ),
                    value=z_score,
                    threshold=-self.VARIANCE_Z_THRESHOLD,
                )

            # Critical: moderately bad
            if z_score < -2.5:
                return KillConditionResult(
                    name="variance_blowup",
                    triggered=True,
                    severity="critical",
                    message=(
                        f"Variance elevated: z={z_score:.2f} "
                        f"(expected {expected_wins:.1f} wins, got {actual_wins:.0f})"
                    ),
                    value=z_score,
                    threshold=-2.5,
                )

            # Warning: slightly below expected
            if z_score < -2.0:
                return KillConditionResult(
                    name="variance_blowup",
                    triggered=True,
                    severity="warning",
                    message=(
                        f"Variance warning: z={z_score:.2f} "
                        f"(expected {expected_wins:.1f} wins, got {actual_wins:.0f})"
                    ),
                    value=z_score,
                    threshold=-2.0,
                )

            return KillConditionResult(
                name="variance_blowup",
                triggered=False,
                severity="warning",
                message=(
                    f"Variance normal: z={z_score:.2f} "
                    f"(expected {expected_wins:.1f} wins, got {actual_wins:.0f})"
                ),
                value=z_score,
                threshold=-self.VARIANCE_Z_THRESHOLD,
            )
