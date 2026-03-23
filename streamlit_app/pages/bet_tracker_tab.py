"""
Golf Quant Engine — Bet Tracker Tab
=====================================
All bet data lives in the database via bet_service.
- bet_service.get_bet_history() for logged bets
- bet_service.place_bet() for new bets
- bet_service.settle_bet() when user enters results
- bet_service.get_pnl_summary() for P&L display
"""

import pandas as pd
import streamlit as st

from streamlit_app.config import DEFAULT_USER_ID
from streamlit_app.state import load_user_settings
from streamlit_app.components.charts import render_pnl_curve, render_roi_by_market

from services.bet_service import (
    place_bet,
    settle_bet,
    get_pending_bets,
    get_settled_bets,
    get_bet_history,
    get_pnl_summary,
)
from services.event_service import get_current_tournament


def render() -> None:
    """Render the Bet Tracker tab. All data from DB."""
    st.markdown("# \U0001f4b0 Bet Tracker")

    # P&L Summary section
    _render_pnl_summary()

    st.divider()

    # Pending bets section
    _render_pending_bets()

    st.divider()

    # Place new bet form
    _render_place_bet_form()

    st.divider()

    # Full bet history
    _render_bet_history()


def _render_pnl_summary() -> None:
    """Render P&L summary metrics and chart from bet_service."""
    st.markdown("### P&L Summary")

    # Period selector
    period = st.selectbox(
        "Period",
        ["all", "monthly", "weekly", "daily"],
        index=0,
        key="pnl_period_select",
    )

    summary = get_pnl_summary(period)

    col1, col2, col3, col4 = st.columns(4)

    total_pnl = summary.get("total_pnl", 0)
    col1.metric(
        "Total P&L",
        f"${total_pnl:+.2f}",
        delta=f"${total_pnl:+.2f}",
        delta_color="normal" if total_pnl >= 0 else "inverse",
    )
    col2.metric("Bets Settled", summary.get("bets_settled", 0))

    win_rate = summary.get("win_rate", 0)
    col3.metric("Win Rate", f"{win_rate*100:.0f}%")

    roi = summary.get("roi", 0)
    col4.metric(
        "ROI",
        f"{roi*100:+.1f}%",
        delta_color="normal" if roi >= 0 else "inverse",
    )

    # Additional metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Won", summary.get("bets_won", 0))
    col2.metric("Lost", summary.get("bets_lost", 0))
    col3.metric("Best Bet", f"${summary.get('best_bet', 0):+.2f}")
    col4.metric("Worst Bet", f"${summary.get('worst_bet', 0):+.2f}")

    # P&L Curve
    settled = get_settled_bets(days=90)
    if settled:
        # Reverse to chronological order for the chart
        settled_chrono = list(reversed(settled))
        with st.expander("\U0001f4c8 P&L Curve", expanded=True):
            render_pnl_curve(settled_chrono)

        with st.expander("\U0001f4ca ROI by Market"):
            render_roi_by_market(settled)


def _render_pending_bets() -> None:
    """Render pending bets with settle buttons."""
    st.markdown("### Pending Bets")

    pending = get_pending_bets()

    if not pending:
        st.info("No pending bets.")
        return

    for bet in pending:
        bet_id = bet.get("id", 0)
        player = bet.get("player", "Unknown")
        market = bet.get("market", "")
        direction = bet.get("direction", "")
        bet_line = bet.get("bet_line", 0)
        stake = bet.get("stake", 0)

        col1, col2, col3 = st.columns([3, 1, 2])
        col1.write(
            f"**{player}** {market} {direction} {bet_line} | "
            f"Stake: ${stake:.2f}"
        )
        col2.write(f"#{bet_id}")

        # Settle form
        with col3:
            settle_key = f"settle_result_{bet_id}"
            actual = st.number_input(
                "Actual",
                value=0.0,
                step=0.5,
                key=settle_key,
                label_visibility="collapsed",
            )
            if st.button(f"Settle #{bet_id}", key=f"settle_btn_{bet_id}"):
                result = settle_bet(bet_id, actual)
                if result.get("error"):
                    st.error(result["error"])
                else:
                    status = result.get("status", "")
                    profit = result.get("profit", 0)
                    st.success(f"Settled: {status} (P&L: ${profit:+.2f})")
                    st.rerun()


def _render_place_bet_form() -> None:
    """Render a form to manually place a new bet. Writes to DB via bet_service."""
    st.markdown("### Place New Bet")

    tournament = get_current_tournament()
    event_name = tournament.get("event_name", "") if tournament else ""
    tournament_id = tournament.get("id") if tournament else None

    with st.form("place_bet_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)

        player = col1.text_input("Player Name", key="form_player")
        market = col2.text_input("Market / Stat Type", key="form_market")
        direction = col3.selectbox(
            "Direction", ["OVER", "UNDER", "WIN", "PLACE"],
            key="form_direction",
        )

        col1, col2, col3 = st.columns(3)

        bet_line = col1.number_input("Line", value=0.0, step=0.5, key="form_line")
        stake = col2.number_input("Stake ($)", value=0.0, step=5.0, min_value=0.0, key="form_stake")
        odds_decimal = col3.number_input(
            "Odds (decimal)", value=1.87, step=0.05, min_value=1.01, key="form_odds"
        )

        col1, col2 = st.columns(2)
        model_proj = col1.number_input(
            "Model Projection (optional)", value=0.0, step=0.5, key="form_proj"
        )
        confidence = col2.slider(
            "Confidence Score (0-1)", 0.0, 1.0, 0.5, step=0.05, key="form_conf"
        )

        notes = st.text_area("Notes (optional)", key="form_notes", height=60)

        submitted = st.form_submit_button("Place Bet", use_container_width=True)

        if submitted:
            if not player.strip():
                st.error("Player name is required.")
            elif stake <= 0:
                st.error("Stake must be greater than zero.")
            else:
                bet_data = {
                    "sport": "GOLF",
                    "player": player.strip(),
                    "market": market.strip(),
                    "event": event_name,
                    "direction": direction,
                    "bet_line": bet_line,
                    "stake": stake,
                    "odds_decimal": odds_decimal,
                    "odds_american": _decimal_to_american(odds_decimal),
                    "model_projection": model_proj if model_proj != 0 else None,
                    "confidence_score": confidence,
                    "tournament_id": tournament_id,
                    "notes": notes.strip() if notes else None,
                }

                bet_id = place_bet(bet_data)
                st.success(f"Bet #{bet_id} placed: {player} {market} {direction} {bet_line} (${stake:.2f})")
                st.rerun()


def _render_bet_history() -> None:
    """Render full bet history table from DB."""
    st.markdown("### Bet History")

    col1, col2 = st.columns(2)
    days = col1.selectbox(
        "Show last",
        [7, 14, 30, 60, 90, 365],
        index=2,
        key="history_days_select",
    )
    status_filter = col2.selectbox(
        "Status",
        ["all", "pending", "won", "lost", "push"],
        key="history_status_select",
    )

    filters = {"days": days}
    if status_filter != "all":
        filters["status"] = status_filter

    history = get_bet_history(filters)

    if not history:
        st.info("No bets found matching your filters.")
        return

    # Build display DataFrame
    display_rows = []
    for b in history:
        display_rows.append({
            "ID": b.get("id", ""),
            "Player": b.get("player", ""),
            "Market": b.get("market", ""),
            "Dir": b.get("direction", ""),
            "Line": b.get("bet_line", ""),
            "Stake": f"${b.get('stake', 0):.2f}" if b.get("stake") else "",
            "Odds": b.get("odds_decimal", ""),
            "Status": b.get("status", ""),
            "P&L": f"${(b.get('pnl') or b.get('profit') or 0):+.2f}",
            "Placed": str(b.get("timestamp", ""))[:16],
            "Settled": str(b.get("settled_at", ""))[:16] if b.get("settled_at") else "",
        })

    df = pd.DataFrame(display_rows)

    # Color status
    def _color_status(val):
        if val == "won":
            return "color: #00FF88; font-weight: bold"
        if val == "lost":
            return "color: #FF4444"
        if val == "pending":
            return "color: #FFB800"
        if val == "push":
            return "color: #888888"
        return ""

    styled = df.style.map(_color_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=400)

    st.caption(f"Showing {len(history)} bets.")


def _decimal_to_american(dec: float) -> int:
    """Convert decimal odds to American odds."""
    if dec >= 2.0:
        return int((dec - 1) * 100)
    elif dec > 1.0:
        return int(-100 / (dec - 1))
    return 0
