"""Risk-adjusted metrics — Sharpe, Sortino, max drawdown, Calmar, VaR, CVaR."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class RiskMetrics:
    """Compute risk-adjusted return metrics for the betting portfolio."""

    def compute_all(
        self,
        returns: List[float],
        risk_free_rate: float = 0.0,
        confidence_level: float = 0.95,
    ) -> dict:
        """Compute all risk metrics from a returns series.

        Args:
            returns: List of period returns (e.g., daily or weekly PnL / bankroll).
            risk_free_rate: Risk-free rate per period (usually 0 for betting).
            confidence_level: Confidence level for VaR/CVaR.

        Returns:
            Dict with all risk metrics.
        """
        if len(returns) < 5:
            return {"error": "Insufficient data", "n_periods": len(returns)}

        arr = np.array(returns, dtype=float)
        n = len(arr)

        # Basic stats
        mean_return = float(np.mean(arr))
        std_return = float(np.std(arr, ddof=1))
        total_return = float(np.sum(arr))
        cumulative = np.cumsum(arr)

        # Sharpe Ratio
        excess = arr - risk_free_rate
        sharpe = float(np.mean(excess) / max(np.std(excess, ddof=1), 0.0001))

        # Sortino Ratio (only penalizes downside)
        downside = arr[arr < risk_free_rate]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0001
        sortino = float(np.mean(excess) / max(downside_std, 0.0001))

        # Max Drawdown
        peak = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - peak
        max_drawdown = float(np.min(drawdowns))
        max_drawdown_pct = abs(max_drawdown) / max(float(np.max(peak)), 0.01)

        # Calmar Ratio (annualized return / max drawdown)
        annualization = np.sqrt(252)  # Assuming daily periods
        annualized_return = mean_return * 252
        calmar = annualized_return / max(abs(max_drawdown), 0.0001) if max_drawdown != 0 else 0.0

        # VaR (Value at Risk)
        alpha = 1 - confidence_level
        var = float(np.percentile(arr, alpha * 100))

        # CVaR (Conditional VaR / Expected Shortfall)
        cvar = float(np.mean(arr[arr <= var])) if np.any(arr <= var) else var

        # Win/Loss analysis
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        win_rate = float(len(wins) / max(n, 1))
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        profit_factor = abs(float(np.sum(wins)) / min(float(np.sum(losses)), -0.01)) if len(losses) > 0 else float('inf')

        # Recovery analysis
        if max_drawdown < 0:
            dd_end_idx = int(np.argmin(drawdowns))
            recovery_idx = None
            for i in range(dd_end_idx, n):
                if cumulative[i] >= peak[dd_end_idx]:
                    recovery_idx = i
                    break
            recovery_periods = recovery_idx - dd_end_idx if recovery_idx else None
        else:
            recovery_periods = 0

        return {
            "n_periods": n,
            "total_return": round(total_return, 4),
            "mean_return": round(mean_return, 6),
            "std_return": round(std_return, 6),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown": round(max_drawdown, 4),
            "max_drawdown_pct": round(max_drawdown_pct, 4),
            "calmar_ratio": round(calmar, 3),
            "var": round(var, 4),
            "cvar": round(cvar, 4),
            "confidence_level": confidence_level,
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 6),
            "avg_loss": round(avg_loss, 6),
            "profit_factor": round(min(profit_factor, 999), 3),
            "recovery_periods": recovery_periods,
            "annualized_sharpe": round(sharpe * annualization, 3),
        }
