"""
Golf Quant Engine — Live Leaderboard Tab
==========================================
Reads live scores from the legacy db_manager leaderboard table
and tournament info from event_service.
No scraping, no live fetching -- workers populate the DB.
"""

import pandas as pd
import streamlit as st

from services.event_service import get_current_tournament
from services.odds_service import get_prizepicks_lines
from database.db_manager import get_leaderboard, normalize_name


def render() -> None:
    """Render the Live Leaderboard tab. All data from DB."""
    st.markdown("# \U0001f4ca Live Leaderboard")

    # Get current tournament from DB
    tournament = get_current_tournament()

    if tournament and tournament.get("event_name"):
        st.caption(
            f"Tournament: **{tournament.get('event_name', '')}** | "
            f"Status: {tournament.get('status', 'unknown')}"
        )
        tournament_id = tournament.get("id")
    else:
        st.caption("No tournament context available.")
        tournament_id = None

    # Refresh button just reruns to re-read from DB
    if st.button("\U0001f504 Refresh Leaderboard"):
        st.rerun()

    # Read leaderboard from DB
    if tournament_id:
        leaderboard = get_leaderboard(tournament_id)
    else:
        leaderboard = get_leaderboard()

    if not leaderboard:
        st.info(
            "No leaderboard data available in the database. "
            "Leaderboard data is populated by background workers during live tournaments."
        )
        return

    # Get PP players to highlight
    pp_lines = get_prizepicks_lines()
    pp_players = set()
    for line in pp_lines:
        name = line.get("player", line.get("player_name", ""))
        if name:
            pp_players.add(normalize_name(name))

    # Build display table
    rows = []
    for entry in leaderboard[:100]:
        player_name = entry.get("player_name", entry.get("name", ""))
        has_pp = normalize_name(player_name) in pp_players

        rows.append({
            "Pos": entry.get("position", ""),
            "Player": player_name,
            "Score": entry.get("total_score", 0),
            "Today": entry.get("today_score", 0),
            "Thru": entry.get("thru", ""),
            "R1": entry.get("round1", ""),
            "R2": entry.get("round2", ""),
            "R3": entry.get("round3", ""),
            "R4": entry.get("round4", ""),
            "PP": "\U0001f3af" if has_pp else "",
        })

    if rows:
        df = pd.DataFrame(rows)

        # Color the score column
        def _color_score(val):
            try:
                v = int(val)
                if v < 0:
                    return "color: #00FF88"
                if v > 0:
                    return "color: #FF4444"
                return "color: #FFB800"
            except (ValueError, TypeError):
                return ""

        styled = df.style.map(_color_score, subset=["Score", "Today"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=700)

        st.caption(
            f"Showing {len(rows)} players. "
            f"\U0001f3af = Has PrizePicks lines."
        )
    else:
        st.info("No leaderboard data. Tournament may not have started yet.")
