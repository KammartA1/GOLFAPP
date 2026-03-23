"""
edge_analysis/structural.py
==============================
Component 5: STRUCTURAL EDGE — Golf portfolio construction analysis.

Golf-specific structural factors:
  - Field-level diversification: spreading bets across field positions
  - Wave advantage correlation: AM/PM wave bets may be correlated
  - Tournament-to-tournament independence
  - Outright portfolio construction (top-heavy vs spread)
  - Kelly criterion adherence in high-variance golf markets
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as sp_stats

from edge_analysis.schemas import GolfBetRecord, EdgeComponentResult

log = logging.getLogger(__name__)


def _compute_daily_returns(bets: List[GolfBetRecord]) -> np.ndarray:
    daily: Dict[str, float] = defaultdict(float)
    for b in bets:
        if b.pnl is not None and b.pnl != 0:
            day_key = b.timestamp.strftime("%Y-%m-%d")
            daily[day_key] += b.pnl
    if not daily:
        return np.array([])
    return np.array([v for _, v in sorted(daily.items())])


def _compute_bet_correlation(bets: List[GolfBetRecord]) -> Tuple[float, float]:
    """Estimate correlation between bet outcomes using tournament-level clustering."""
    # Group by tournament (golf bets within same event are more likely correlated)
    tourn_bets: Dict[str, List[GolfBetRecord]] = defaultdict(list)
    for b in bets:
        if b.won is not None:
            tourn_bets[b.tournament].append(b)

    if len(tourn_bets) < 5:
        return 0.0, 1.0

    actual_variances = []
    expected_variances = []

    for tourn, tbets in tourn_bets.items():
        if len(tbets) < 2:
            continue
        outcomes = np.array([1.0 if b.won else 0.0 for b in tbets])
        n = len(outcomes)
        p = float(np.mean(outcomes))

        expected_var = n * p * (1.0 - p) if 0 < p < 1 else 0.0
        actual_var = float(np.var(outcomes) * n)

        if expected_var > 0:
            actual_variances.append(actual_var)
            expected_variances.append(expected_var)

    if not actual_variances:
        return 0.0, 1.0

    avg_actual = float(np.mean(actual_variances))
    avg_expected = float(np.mean(expected_variances))
    variance_ratio = avg_actual / avg_expected if avg_expected > 0 else 1.0

    avg_n = float(np.mean([len(v) for v in tourn_bets.values() if len(v) >= 2]))
    rho = (variance_ratio - 1.0) / (avg_n - 1.0) if avg_n > 1 else 0.0

    return float(np.clip(rho, -1.0, 1.0)), variance_ratio


def _wave_correlation_analysis(bets: List[GolfBetRecord]) -> Dict:
    """Analyze correlation between bets on same wave."""
    wave_bets = [b for b in bets if b.wave is not None and b.won is not None]
    if len(wave_bets) < 10:
        return {"sufficient_data": False}

    # Group by tournament + wave
    wave_groups: Dict[str, List[GolfBetRecord]] = defaultdict(list)
    for b in wave_bets:
        key = f"{b.tournament}_{b.wave}"
        wave_groups[key].append(b)

    # Check if same-wave bets are more correlated than cross-wave
    same_wave_outcomes = []
    for key, group in wave_groups.items():
        if len(group) >= 2:
            outcomes = [1.0 if b.won else 0.0 for b in group]
            same_wave_outcomes.extend(outcomes)

    am_results = [1.0 if b.won else 0.0 for b in wave_bets if b.wave == "AM"]
    pm_results = [1.0 if b.won else 0.0 for b in wave_bets if b.wave == "PM"]

    return {
        "sufficient_data": True,
        "n_wave_bets": len(wave_bets),
        "am_win_rate": round(float(np.mean(am_results)), 4) if am_results else None,
        "pm_win_rate": round(float(np.mean(pm_results)), 4) if pm_results else None,
        "n_am": len(am_results),
        "n_pm": len(pm_results),
        "same_wave_clustering": len(wave_groups),
    }


def _kelly_analysis(bets: List[GolfBetRecord]) -> Dict:
    settled = [b for b in bets if b.won is not None and b.kelly_fraction > 0]
    if len(settled) < 10:
        return {"sufficient_data": False}

    kelly_fractions = np.array([b.kelly_fraction for b in settled])
    avg_kelly = float(np.mean(kelly_fractions))
    max_kelly = float(np.max(kelly_fractions))

    # Growth rate
    returns = []
    for b in settled:
        if b.won:
            r = b.odds_decimal - 1.0
        else:
            r = -1.0
        growth = np.log(max(1e-10, 1.0 + b.kelly_fraction * r))
        returns.append(growth)
    growth_rate = float(np.mean(returns)) if returns else 0.0

    # Full Kelly comparison
    full_kelly_returns = []
    for b in settled:
        edge = b.predicted_prob - b.market_prob_at_bet
        if b.odds_decimal > 1:
            f_star = max(0, edge / (b.odds_decimal - 1.0))
        else:
            f_star = 0
        r = (b.odds_decimal - 1.0) if b.won else -1.0
        growth = np.log(max(1e-10, 1.0 + f_star * r))
        full_kelly_returns.append(growth)
    full_kelly_growth = float(np.mean(full_kelly_returns)) if full_kelly_returns else 0.0

    return {
        "sufficient_data": True,
        "avg_kelly_fraction": round(avg_kelly, 4),
        "max_kelly_fraction": round(max_kelly, 4),
        "actual_growth_rate": round(growth_rate, 6),
        "full_kelly_growth_rate": round(full_kelly_growth, 6),
        "kelly_efficiency": round(growth_rate / full_kelly_growth, 4) if full_kelly_growth > 0 else 0.0,
    }


def _diversification_analysis(bets: List[GolfBetRecord]) -> Dict:
    if not bets:
        return {}

    market_counts: Dict[str, int] = defaultdict(int)
    player_counts: Dict[str, int] = defaultdict(int)
    tournament_counts: Dict[str, int] = defaultdict(int)
    for b in bets:
        market_counts[b.market_type] += 1
        player_counts[b.player] += 1
        tournament_counts[b.tournament] += 1

    total = len(bets)
    hhi_market = sum((c / total) ** 2 for c in market_counts.values())
    hhi_player = sum((c / total) ** 2 for c in player_counts.values())
    hhi_tournament = sum((c / total) ** 2 for c in tournament_counts.values())

    top_market = max(market_counts.items(), key=lambda x: x[1])
    top_player = max(player_counts.items(), key=lambda x: x[1])

    return {
        "n_unique_markets": len(market_counts),
        "n_unique_players": len(player_counts),
        "n_unique_tournaments": len(tournament_counts),
        "hhi_market": round(hhi_market, 4),
        "hhi_player": round(hhi_player, 4),
        "hhi_tournament": round(hhi_tournament, 4),
        "top_market": {"name": top_market[0], "pct": round(top_market[1] / total, 4)},
        "top_player": {"name": top_player[0], "pct": round(top_player[1] / total, 4)},
        "market_distribution": dict(sorted(market_counts.items(), key=lambda x: -x[1])),
    }


def compute_structural_edge(
    bets: List[GolfBetRecord],
    total_roi: float,
) -> EdgeComponentResult:
    """Analyze golf structural (portfolio/bankroll) edge.

    Golf-specific structural concerns:
    1. Field-level diversification across players
    2. Wave correlation (AM/PM bets within same tournament)
    3. Tournament independence (week-to-week)
    4. Outright portfolio construction (top-heavy concentration)
    5. Kelly sizing in high-variance markets (outrights can be +5000)
    """
    settled = [b for b in bets if b.won is not None]
    if len(settled) < 15:
        return EdgeComponentResult(
            name="structural",
            edge_pct_of_roi=0.0, absolute_value=0.0, p_value=1.0,
            is_significant=False, is_positive=False,
            sample_size=len(settled),
            verdict="Insufficient data for structural edge analysis (need 15+ settled bets)",
        )

    # Correlation
    rho, variance_ratio = _compute_bet_correlation(settled)

    # Kelly
    kelly = _kelly_analysis(settled)

    # Wave correlation
    wave = _wave_correlation_analysis(settled)

    # Diversification
    diversification = _diversification_analysis(settled)

    # Daily returns
    daily_returns = _compute_daily_returns(settled)
    sharpe = 0.0
    max_drawdown = 0.0
    if len(daily_returns) >= 10:
        if np.std(daily_returns) > 0:
            sharpe = float(np.mean(daily_returns)) / float(np.std(daily_returns)) * np.sqrt(52)
        cumulative = np.cumsum(daily_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Significance — bootstrap variance ratio test
    p_value = 0.5
    if len(daily_returns) >= 15:
        n_boot = 1000
        boot_ratios = []
        orig_var = float(np.var(daily_returns))
        if orig_var > 0:
            for _ in range(n_boot):
                idx = np.random.choice(len(daily_returns), size=len(daily_returns), replace=True)
                boot_var = float(np.var(daily_returns[idx]))
                boot_ratios.append(boot_var / orig_var)
            p_value = float(np.mean(np.array(boot_ratios) > variance_ratio))

    is_positive = variance_ratio < 1.2 and (
        not kelly.get("sufficient_data") or kelly.get("kelly_efficiency", 0) > 0.3
    )
    is_significant = p_value < 0.05 or (
        kelly.get("sufficient_data") and kelly.get("kelly_efficiency", 0) > 0.5
    )

    # Attribution
    structural_pct = 0.0
    if is_positive:
        if kelly.get("sufficient_data"):
            structural_pct = min(20.0, kelly.get("kelly_efficiency", 0) * 25.0)
        else:
            structural_pct = 10.0 if variance_ratio < 1.1 else 5.0
        hhi = diversification.get("hhi_tournament", 1.0)
        if hhi < 0.15:
            structural_pct += 3.0
    else:
        if variance_ratio > 1.5:
            structural_pct = -10.0
        elif variance_ratio > 1.2:
            structural_pct = -5.0

    # Verdict
    verdict_parts = []
    if abs(rho) < 0.05:
        verdict_parts.append(
            f"Bet independence: GOOD (rho={rho:.3f}, variance ratio={variance_ratio:.2f})."
        )
    elif rho > 0.05:
        verdict_parts.append(
            f"WARNING: Bets positively correlated (rho={rho:.3f}). "
            f"Tournament-level clustering — diversify across weeks."
        )
    else:
        verdict_parts.append(
            f"Natural hedge detected (rho={rho:.3f}). Good portfolio construction."
        )

    if kelly.get("sufficient_data"):
        eff = kelly.get("kelly_efficiency", 0)
        if eff > 0.7:
            verdict_parts.append(f"Kelly sizing: EXCELLENT (efficiency={eff:.0%}).")
        elif eff > 0.4:
            verdict_parts.append(f"Kelly sizing: ACCEPTABLE (efficiency={eff:.0%}).")
        else:
            verdict_parts.append(f"Kelly sizing: POOR (efficiency={eff:.0%}). Review outright sizing.")

    if wave.get("sufficient_data"):
        verdict_parts.append(
            f"Wave analysis: AM={wave.get('am_win_rate', 0):.1%} win rate, "
            f"PM={wave.get('pm_win_rate', 0):.1%}. "
            f"Same-wave groups: {wave.get('same_wave_clustering', 0)}."
        )

    hhi_t = diversification.get("hhi_tournament", 1.0)
    if hhi_t < 0.1:
        verdict_parts.append("Tournament diversification: EXCELLENT.")
    elif hhi_t < 0.25:
        verdict_parts.append("Tournament diversification: ACCEPTABLE.")
    else:
        verdict_parts.append("Tournament diversification: POOR — too concentrated.")

    return EdgeComponentResult(
        name="structural",
        edge_pct_of_roi=round(structural_pct, 2),
        absolute_value=round(variance_ratio, 4),
        p_value=round(p_value, 4),
        is_significant=is_significant,
        is_positive=is_positive,
        sample_size=len(settled),
        details={
            "correlation_rho": round(rho, 4),
            "variance_ratio": round(variance_ratio, 4),
            "annualized_sharpe": round(sharpe, 3),
            "max_drawdown": round(max_drawdown, 2),
            "kelly": kelly,
            "wave_analysis": wave,
            "diversification": diversification,
        },
        verdict=" ".join(verdict_parts),
    )
