"""
edge_analysis/informational.py
================================
Component 2: INFORMATIONAL EDGE — Golf-specific timing analysis.

Golf informational edge sources:
  - Weather data timing: did we get forecast before the market priced it in?
  - Injury/withdrawal timing: did we know about WDs before odds adjusted?
  - Course history weighting: do we use historical data the market underweights?
  - Wave advantage: did we identify AM/PM wave edge before line adjustments?

Key metric: time_delta = signal_generation_time - line_movement_time
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from scipy import stats as sp_stats

from edge_analysis.schemas import GolfBetRecord, EdgeComponentResult

log = logging.getLogger(__name__)


def _compute_signal_lead_time(bet: GolfBetRecord) -> Optional[float]:
    """Signal lead time in minutes. Positive = we were early."""
    if bet.signal_generated_at is None or bet.line_moved_at is None:
        return None
    return (bet.line_moved_at - bet.signal_generated_at).total_seconds() / 60.0


def _compute_weather_lead_time(bet: GolfBetRecord) -> Optional[float]:
    """How early did we have weather data vs line movement?"""
    if bet.weather_fetched_at is None or bet.line_moved_at is None:
        return None
    return (bet.line_moved_at - bet.weather_fetched_at).total_seconds() / 60.0


def _compute_withdrawal_lead_time(bet: GolfBetRecord) -> Optional[float]:
    """Did we know about injury/withdrawal before line adjusted?"""
    if bet.injury_withdrawal_at is None or bet.line_moved_at is None:
        return None
    return (bet.line_moved_at - bet.injury_withdrawal_at).total_seconds() / 60.0


def compute_informational_edge(
    bets: List[GolfBetRecord],
    total_roi: float,
) -> EdgeComponentResult:
    """Analyze golf informational timing edge.

    Golf-specific informational advantages:
    1. Weather data: AM/PM wind forecasts affect wave advantage
    2. Injury/WD timing: field changes affect outright odds dramatically
    3. Course history data: heavy-tailed course fits the market underweights
    4. Wave assignments: first-day wave info before market adjusts

    Returns EdgeComponentResult with timing analysis.
    """
    # Compute all timing sources
    signal_leads = []
    weather_leads = []
    wd_leads = []
    inferred_timing = []

    for bet in bets:
        sl = _compute_signal_lead_time(bet)
        if sl is not None:
            signal_leads.append(sl)
        wl = _compute_weather_lead_time(bet)
        if wl is not None:
            weather_leads.append(wl)
        wdl = _compute_withdrawal_lead_time(bet)
        if wdl is not None:
            wd_leads.append(wdl)

        # Inferred timing from line movements
        if bet.won is not None and bet.closing_line != 0:
            if bet.direction.upper() in ("OVER", "WIN", "PLACE"):
                post_bet_value = bet.closing_line - bet.bet_line
            else:
                post_bet_value = bet.bet_line - bet.closing_line
            inferred_timing.append(post_bet_value)

    # Use best available timing source
    has_direct = len(signal_leads) >= 10
    has_weather = len(weather_leads) >= 5
    has_wd = len(wd_leads) >= 3
    has_inferred = len(inferred_timing) >= 15

    if not (has_direct or has_weather or has_inferred):
        return EdgeComponentResult(
            name="informational",
            edge_pct_of_roi=0.0, absolute_value=0.0, p_value=1.0,
            is_significant=False, is_positive=False,
            sample_size=max(len(signal_leads), len(inferred_timing)),
            verdict="Insufficient timing data for informational edge analysis",
        )

    # Primary: direct signal timing
    if has_direct:
        primary_arr = np.array(signal_leads)
        timing_source = "direct signal timing"
    else:
        primary_arr = np.array(inferred_timing)
        timing_source = "inferred line movement"

    median_lead = float(np.median(primary_arr))
    mean_lead = float(np.mean(primary_arr))
    pct_ahead = float(np.mean(primary_arr > 0))

    t_stat, p_two = sp_stats.ttest_1samp(primary_arr, 0.0)
    p_value = float(p_two / 2 if t_stat > 0 else 1.0 - p_two / 2)

    is_positive = median_lead > 0
    is_significant = p_value < 0.05

    # Weather timing sub-analysis
    weather_summary = None
    if has_weather:
        w_arr = np.array(weather_leads)
        weather_summary = {
            "median_lead_minutes": round(float(np.median(w_arr)), 2),
            "pct_ahead": round(float(np.mean(w_arr > 0)), 4),
            "n_samples": len(weather_leads),
        }

    # WD timing sub-analysis
    wd_summary = None
    if has_wd:
        wd_arr = np.array(wd_leads)
        wd_summary = {
            "median_lead_minutes": round(float(np.median(wd_arr)), 2),
            "pct_ahead": round(float(np.mean(wd_arr > 0)), 4),
            "n_samples": len(wd_leads),
        }

    # Wave advantage timing
    wave_bets = [b for b in bets if b.wave is not None and b.won is not None]
    wave_summary = None
    if len(wave_bets) >= 10:
        am_wins = [1.0 if b.won else 0.0 for b in wave_bets if b.wave == "AM"]
        pm_wins = [1.0 if b.won else 0.0 for b in wave_bets if b.wave == "PM"]
        if am_wins and pm_wins:
            wave_summary = {
                "am_win_rate": round(float(np.mean(am_wins)), 4) if am_wins else None,
                "pm_win_rate": round(float(np.mean(pm_wins)), 4) if pm_wins else None,
                "n_am": len(am_wins),
                "n_pm": len(pm_wins),
                "wave_edge": round(float(np.mean(am_wins)) - float(np.mean(pm_wins)), 4) if am_wins and pm_wins else 0.0,
            }

    # Attribution
    if is_positive and is_significant:
        info_pct = min(25.0, pct_ahead * 35.0)
    elif is_positive:
        info_pct = min(12.0, pct_ahead * 20.0)
    else:
        info_pct = 0.0

    # Bonus for weather/WD timing
    if weather_summary and weather_summary["pct_ahead"] > 0.6:
        info_pct += 3.0
    if wd_summary and wd_summary["pct_ahead"] > 0.7:
        info_pct += 5.0

    # Verdict
    verdict_parts = []
    if is_positive and is_significant:
        verdict_parts.append(
            f"REAL informational edge via {timing_source}. "
            f"Median lead: {median_lead:+.1f} min. {pct_ahead:.0%} ahead. p={p_value:.4f}."
        )
    elif is_positive:
        verdict_parts.append(
            f"Possible informational edge but NOT significant (p={p_value:.4f}). "
            f"Median lead: {median_lead:+.1f} min."
        )
    else:
        verdict_parts.append(
            f"NO informational edge. Signals fire AFTER line moves. "
            f"Median lead: {median_lead:+.1f} min."
        )
    if weather_summary:
        verdict_parts.append(
            f"Weather timing: {weather_summary['pct_ahead']:.0%} ahead of market "
            f"(median {weather_summary['median_lead_minutes']:+.0f} min)."
        )
    if wd_summary:
        verdict_parts.append(
            f"Injury/WD timing: {wd_summary['pct_ahead']:.0%} ahead "
            f"(median {wd_summary['median_lead_minutes']:+.0f} min)."
        )
    if wave_summary:
        verdict_parts.append(
            f"Wave edge: AM={wave_summary['am_win_rate']:.1%}, PM={wave_summary['pm_win_rate']:.1%} "
            f"(diff={wave_summary['wave_edge']:+.1%})."
        )

    return EdgeComponentResult(
        name="informational",
        edge_pct_of_roi=round(info_pct, 2),
        absolute_value=round(median_lead, 2),
        p_value=round(p_value, 4),
        is_significant=is_significant,
        is_positive=is_positive,
        sample_size=len(primary_arr),
        details={
            "timing_source": timing_source,
            "median_lead_minutes": round(median_lead, 2),
            "mean_lead_minutes": round(mean_lead, 2),
            "pct_ahead_of_market": round(pct_ahead, 4),
            "t_stat": round(float(t_stat), 3),
            "n_direct_timing": len(signal_leads),
            "n_inferred_timing": len(inferred_timing),
            "weather_timing": weather_summary,
            "withdrawal_timing": wd_summary,
            "wave_analysis": wave_summary,
        },
        verdict=" ".join(verdict_parts),
    )
