"""
Golf Quant Engine — PrizePicks Lab Tab
========================================
Reads PrizePicks lines from DB via odds_service.
Runs analysis via projection_service.
Displays edge filtering and best bets.
No scraping here -- workers handle that.
"""

import pandas as pd
import numpy as np
import streamlit as st

from streamlit_app.config import (
    DEFAULT_USER_ID,
    CONFIDENCE_COLORS,
    CONFIDENCE_SORT_ORDER,
)
from streamlit_app.state import load_user_settings, get_ui_state
from streamlit_app.components.charts import render_edge_distribution

from services.odds_service import get_prizepicks_lines, get_tournament_odds
from services.event_service import get_current_tournament
from services.weather_service import get_current_weather, get_weather_adjustment_factor
from services.projection_service import run_projection
from services.player_service import get_player_sg_stats


def render() -> None:
    """Render the PrizePicks Lab tab. All data from DB."""
    st.markdown("# \U0001f3af PrizePicks Lab")

    # Load settings from DB
    settings = load_user_settings(DEFAULT_USER_ID)
    bankroll = float(settings.get("bankroll", 1000))
    kelly_frac = float(settings.get("kelly_fraction", 0.25))
    min_edge = float(settings.get("min_edge", 0.03))

    # Load tournament context from DB
    tournament = get_current_tournament()
    tournament_id = tournament.get("id", 0) if tournament else 0
    t_name = tournament.get("event_name", "No Tournament") if tournament else "No Tournament"

    # Load PrizePicks lines from DB
    pp_lines = get_prizepicks_lines()

    # Load tournament odds from DB for consensus
    odds_consensus = get_tournament_odds()
    consensus_lookup = {}
    for c in odds_consensus:
        key = (c.get("player_name", "") or "").strip().lower()
        if key:
            consensus_lookup[key] = c.get("consensus_prob", 0)

    # Load weather from DB
    weather_data = None
    course_name = tournament.get("course_name", "") if tournament else ""
    if course_name:
        weather_raw = get_current_weather(course_name)
        if weather_raw and not weather_raw.get("error"):
            weather_data = weather_raw

    # Status bar
    status_cols = st.columns(4)
    status_cols[0].metric("PP Lines", len(pp_lines))
    status_cols[1].metric("Odds Players", len(odds_consensus))
    status_cols[2].metric("Tournament", t_name[:25] if t_name else "None")

    if weather_data:
        adj = get_weather_adjustment_factor(weather_data)
        if adj > 0.1:
            status_cols[3].metric("Weather", f"\u26a0\ufe0f +{adj:.2f} strokes")
        else:
            status_cols[3].metric("Weather", "\u2705 Normal")
    else:
        status_cols[3].metric("Weather", "\u2014 No data")

    if not pp_lines:
        st.info(
            "\U0001f4ed **No PrizePicks lines in database.**\n\n"
            "Lines are populated by background workers. "
            "Ensure the odds_worker is running and has scraped PrizePicks data."
        )
        return

    # Run projections for each line
    analyses = _analyze_all_lines(
        pp_lines=pp_lines,
        tournament_id=tournament_id,
        consensus_lookup=consensus_lookup,
        weather_data=weather_data,
        settings=settings,
    )

    # Filter by minimum edge
    bettable = [
        a for a in analyses
        if a.get("best_edge", 0) >= min_edge
        and a.get("confidence") != "NO_BET"
        and not a.get("error")
    ]
    bettable.sort(key=lambda x: x.get("best_edge", 0), reverse=True)

    # Best Bets Board
    _render_best_bets(bettable)

    # Full Analysis Table
    _render_analysis_table(analyses, min_edge)

    # Session Summary
    _render_session_summary(analyses, bettable)

    # Edge distribution chart
    with st.expander("\U0001f4ca Edge Distribution"):
        render_edge_distribution(analyses)


def _analyze_all_lines(
    pp_lines: list[dict],
    tournament_id: int,
    consensus_lookup: dict,
    weather_data: dict | None,
    settings: dict,
) -> list[dict]:
    """
    Run projection for each PrizePicks line. Uses projection_service
    which reads SG data from DB.
    """
    analyses = []

    for line in pp_lines:
        player = line.get("player", line.get("player_name", "Unknown"))
        market = line.get("market", line.get("stat_type", "unknown"))
        line_val = line.get("line", line.get("line_value", 0))

        if line_val is None:
            line_val = 0

        # Build settings dict for projection_service
        proj_settings = {
            "bankroll": float(settings.get("bankroll", 1000)),
            "kelly_fraction": float(settings.get("kelly_fraction", 0.25)),
            "min_edge": float(settings.get("min_edge", 0.03)),
        }

        # Add weather if available
        if weather_data and not weather_data.get("error"):
            proj_settings["weather"] = weather_data

        # Add consensus probability
        player_lower = player.strip().lower()
        consensus = consensus_lookup.get(player_lower, None)
        if consensus:
            proj_settings["consensus_prob"] = consensus

        try:
            result = run_projection(
                player_name=player,
                tournament_id=tournament_id,
                market=market,
                line=float(line_val),
                settings_dict=proj_settings,
            )
            result["stat_display"] = market
            analyses.append(result)
        except Exception as exc:
            analyses.append({
                "player_name": player,
                "stat_type": market,
                "stat_display": market,
                "line_value": line_val,
                "error": str(exc),
                "best_edge": 0,
                "confidence": "NO_BET",
            })

    return analyses


def _render_best_bets(bettable: list[dict]) -> None:
    """Render the Best Bets board from pre-analyzed data."""
    st.markdown("## \U0001f525 Best Bets")

    top_bets = bettable[:8]

    if not top_bets:
        st.info("No bets meet the minimum edge threshold. Try lowering the edge filter in settings.")
        return

    for i in range(0, len(top_bets), 4):
        cols = st.columns(min(4, len(top_bets) - i))
        for j, col in enumerate(cols):
            if i + j >= len(top_bets):
                break
            bet = top_bets[i + j]
            conf_icon = CONFIDENCE_COLORS.get(bet.get("confidence", ""), "\u26aa")
            mkt_icon = "\u2705" if bet.get("market_alignment") is True else (
                "\u26a0\ufe0f" if bet.get("market_alignment") is False else "\u2014"
            )

            player_name = bet.get("player_name", "Unknown")
            stat_display = bet.get("stat_display", bet.get("stat_type", ""))
            best_side = bet.get("best_side", "")
            line_value = bet.get("line_value", 0)
            projection = bet.get("projection", 0)
            std_dev = bet.get("std_dev", 0)
            best_edge = bet.get("best_edge", 0)
            best_prob = bet.get("best_prob", 0)
            confidence = bet.get("confidence", "")
            kelly_stake = bet.get("kelly_stake", 0)
            kelly_pct = bet.get("kelly_pct", 0)

            with col:
                st.markdown(f"""
                **{player_name}**
                {stat_display}

                **{best_side}** {line_value}
                Proj: {projection:.1f} (s={std_dev:.1f})

                Edge: **{best_edge*100:+.1f}%** | P: {best_prob*100:.0f}%
                {conf_icon} {confidence} | Mkt: {mkt_icon}
                Kelly: ${kelly_stake:.0f} ({kelly_pct:.1f}%)
                """)

                warnings = bet.get("warnings", [])
                if warnings:
                    for w in warnings:
                        st.warning(w)


def _render_analysis_table(analyses: list[dict], min_edge: float) -> None:
    """Render the full analysis table with filters."""
    st.markdown("## \U0001f4cb All Lines Analysis")

    # Filters
    filter_cols = st.columns(4)

    stat_types = sorted(set(
        a.get("stat_display", a.get("stat_type", ""))
        for a in analyses
        if not a.get("error")
    ))
    stat_filter = filter_cols[0].selectbox(
        "Stat Type", ["All"] + stat_types, key="pp_stat_filter_widget"
    )
    conf_filter = filter_cols[1].selectbox(
        "Confidence", ["All", "HIGH", "MEDIUM", "LOW"], key="pp_conf_filter_widget"
    )
    side_filter = filter_cols[2].selectbox(
        "Side", ["All", "OVER", "UNDER"], key="pp_side_filter_widget"
    )
    sort_by = filter_cols[3].selectbox(
        "Sort by", ["Edge %", "Confidence", "Player"], key="pp_sort_widget"
    )

    # Apply filters
    filtered = [a for a in analyses if not a.get("error")]

    if stat_filter != "All":
        filtered = [
            a for a in filtered
            if a.get("stat_display") == stat_filter or a.get("stat_type") == stat_filter
        ]
    if conf_filter != "All":
        filtered = [a for a in filtered if a.get("confidence") == conf_filter]
    if side_filter != "All":
        filtered = [a for a in filtered if a.get("best_side") == side_filter]

    # Sort
    if sort_by == "Edge %":
        filtered.sort(key=lambda x: x.get("best_edge", 0), reverse=True)
    elif sort_by == "Confidence":
        filtered.sort(key=lambda x: (
            CONFIDENCE_SORT_ORDER.get(x.get("confidence", "NO_BET"), 9),
            -x.get("best_edge", 0),
        ))
    else:
        filtered.sort(key=lambda x: x.get("player_name", ""))

    if not filtered:
        st.info("No lines match your filters.")
        return

    # Build DataFrame
    rows = []
    for a in filtered:
        rows.append({
            "Player": a.get("player_name", ""),
            "Stat": a.get("stat_display", a.get("stat_type", "")),
            "Line": a.get("line_value", 0),
            "Proj": f"{a.get('projection', 0):.1f}",
            "s": f"{a.get('std_dev', 0):.1f}",
            "Over%": f"{a.get('prob_over', 0)*100:.0f}%",
            "Under%": f"{a.get('prob_under', 0)*100:.0f}%",
            "Side": a.get("best_side", ""),
            "Edge": f"{a.get('best_edge', 0)*100:+.1f}%",
            "Conf": a.get("confidence", ""),
            "Mkt": "\u2705" if a.get("market_alignment") is True else (
                "\u26a0\ufe0f" if a.get("market_alignment") is False else "\u2014"
            ),
            "Kelly $": f"${a.get('kelly_stake', 0):.0f}",
            "Z": f"{a.get('z_score', 0):.2f}",
        })

    df = pd.DataFrame(rows)

    def _color_edge(val):
        try:
            v = float(str(val).replace("%", "").replace("+", ""))
            if v >= 8:
                return "color: #00FF88; font-weight: bold"
            if v >= 5:
                return "color: #00AAFF"
            if v >= 3:
                return "color: #FFB800"
            return "color: #FF4444"
        except (ValueError, TypeError):
            return ""

    def _color_conf(val):
        if val == "HIGH":
            return "color: #00FF88; font-weight: bold"
        if val == "MEDIUM":
            return "color: #FFB800"
        if val == "LOW":
            return "color: #FF8800"
        return "color: #FF4444"

    styled = df.style.map(_color_edge, subset=["Edge"]).map(_color_conf, subset=["Conf"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=600)

    # Expandable detail view
    with st.expander("\U0001f4ca Detailed Analysis (click to expand)"):
        for a in filtered[:20]:
            st.markdown(f"""
            **{a.get('player_name', '')}** -- {a.get('stat_display', '')} {a.get('best_side', '')} {a.get('line_value', 0)}
            - Projection: {a.get('projection', 0):.2f} +/- {a.get('std_dev', 0):.2f} | Z-score: {a.get('z_score', 0):.3f}
            - P(Over): {a.get('prob_over', 0)*100:.1f}% | P(Under): {a.get('prob_under', 0)*100:.1f}%
            - Edge: {a.get('best_edge', 0)*100:+.2f}% | Confidence: {a.get('confidence', '')}
            - Market Consensus: {a.get('consensus_prob', 'N/A')}
            - Weather: {a.get('weather_impact', 'None')}
            - Kelly: ${a.get('kelly_stake', 0):.2f} ({a.get('kelly_pct', 0):.1f}%) | EV: ${a.get('ev_dollars', 0):.2f}
            {'- Warning: ' + ', '.join(a.get('warnings', [])) if a.get('warnings') else ''}
            ---
            """)


def _render_session_summary(analyses: list[dict], bettable: list[dict]) -> None:
    """Render session summary metrics."""
    st.markdown("## \U0001f4c8 Session Summary")

    valid_analyses = [a for a in analyses if not a.get("error")]

    sum_cols = st.columns(4)
    sum_cols[0].metric("Total Lines", len(valid_analyses))
    sum_cols[1].metric("Bettable", len(bettable))

    high = len([a for a in bettable if a.get("confidence") == "HIGH"])
    sum_cols[2].metric("HIGH Confidence", high)

    if bettable:
        avg_edge = np.mean([a.get("best_edge", 0) for a in bettable]) * 100
        sum_cols[3].metric("Avg Edge", f"{avg_edge:+.1f}%")
    else:
        sum_cols[3].metric("Avg Edge", "N/A")
