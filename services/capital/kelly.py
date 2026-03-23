"""Kelly criterion — Full, fractional, and uncertainty-adjusted Kelly sizing."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class KellyCriterion:
    """Full/fractional/uncertainty-adjusted Kelly criterion for bet sizing."""

    def __init__(self, default_fraction: float = 0.25, max_kelly_pct: float = 0.05):
        """
        Args:
            default_fraction: Fractional Kelly multiplier (0.25 = quarter Kelly).
            max_kelly_pct: Maximum % of bankroll per bet.
        """
        self.default_fraction = default_fraction
        self.max_kelly_pct = max_kelly_pct

    def full_kelly(self, win_prob: float, odds_decimal: float) -> float:
        """Full Kelly fraction: f* = (bp - q) / b

        where b = decimal_odds - 1, p = win probability, q = 1 - p.
        """
        if odds_decimal <= 1.0 or win_prob <= 0 or win_prob >= 1:
            return 0.0
        b = odds_decimal - 1.0
        p = win_prob
        q = 1.0 - p
        kelly = (b * p - q) / b
        return max(0.0, kelly)

    def fractional_kelly(
        self,
        win_prob: float,
        odds_decimal: float,
        fraction: float | None = None,
    ) -> float:
        """Fractional Kelly = full_kelly * fraction.

        Standard practice: use 1/4 Kelly to reduce variance.
        """
        f = fraction or self.default_fraction
        return self.full_kelly(win_prob, odds_decimal) * f

    def uncertainty_adjusted_kelly(
        self,
        win_prob: float,
        odds_decimal: float,
        prob_uncertainty: float = 0.05,
        fraction: float | None = None,
    ) -> float:
        """Kelly adjusted for uncertainty in probability estimate.

        When we're uncertain about our probability estimate, we should
        bet less. This uses the lower end of the confidence interval.

        Args:
            win_prob: Estimated win probability.
            odds_decimal: Decimal odds.
            prob_uncertainty: Standard error of probability estimate.
            fraction: Kelly fraction to apply.
        """
        # Use lower bound of probability estimate (conservative)
        adjusted_prob = max(0.01, win_prob - prob_uncertainty)
        return self.fractional_kelly(adjusted_prob, odds_decimal, fraction)

    def optimal_stake(
        self,
        win_prob: float,
        odds_decimal: float,
        bankroll: float,
        prob_uncertainty: float = 0.05,
        clv_avg_cents: float = 0.0,
        calibration_mae: float = 0.0,
        fraction: float | None = None,
    ) -> dict:
        """Compute optimal stake with all adjustments.

        Returns:
            {
                'full_kelly': float,
                'fractional_kelly': float,
                'uncertainty_kelly': float,
                'stake_dollars': float,
                'stake_pct': float,
                'edge': float,
                'expected_value': float,
                'blocked': bool,
                'block_reason': str,
            }
        """
        f = fraction or self.default_fraction

        full_k = self.full_kelly(win_prob, odds_decimal)
        frac_k = full_k * f
        unc_k = self.uncertainty_adjusted_kelly(win_prob, odds_decimal, prob_uncertainty, f)

        # Additional adjustments
        adjusted_k = unc_k

        # CLV adjustment: if recent CLV is negative, reduce sizing
        if clv_avg_cents < -1.0:
            clv_penalty = max(0.3, 1.0 + clv_avg_cents / 10.0)
            adjusted_k *= clv_penalty

        # Calibration adjustment: poor calibration -> reduce sizing
        if calibration_mae > 0.05:
            cal_penalty = max(0.5, 1.0 - (calibration_mae - 0.05) * 5)
            adjusted_k *= cal_penalty

        # Apply maximum cap
        adjusted_k = min(adjusted_k, self.max_kelly_pct)

        stake = bankroll * adjusted_k
        edge = win_prob - (1.0 / odds_decimal if odds_decimal > 1 else 0.5)
        ev = stake * edge

        # Block conditions
        blocked = False
        block_reason = ""
        if edge <= 0:
            blocked = True
            block_reason = "No edge (negative or zero)"
            stake = 0.0
        elif adjusted_k <= 0.001:
            blocked = True
            block_reason = "Kelly fraction too small after adjustments"
            stake = 0.0
        elif stake < 1.0:
            blocked = True
            block_reason = "Stake below $1 minimum"
            stake = 0.0

        return {
            "full_kelly": round(full_k, 6),
            "fractional_kelly": round(frac_k, 6),
            "uncertainty_kelly": round(unc_k, 6),
            "adjusted_kelly": round(adjusted_k, 6),
            "stake_dollars": round(max(0, stake), 2),
            "stake_pct": round(adjusted_k * 100, 3),
            "edge": round(edge, 4),
            "expected_value": round(ev, 2),
            "blocked": blocked,
            "block_reason": block_reason,
        }
