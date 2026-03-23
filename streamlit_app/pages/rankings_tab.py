"""
Golf Quant Engine — Power Rankings Tab
========================================
Reads odds consensus from odds_service.get_tournament_odds().
Displays power rankings from DB. No computation.
"""

import pandas as pd
import streamlit as st

from services.odds_service import get_tournament_odds
from services.player_service import get_player_sg_stats
from streamlit_app.components.charts import render_sg_comparison_chart


def render() -> None:
    """Render the Power Rankings tab. All data from DB."""
    st.markdown("# \U0001f3c6 Power Rankings")

    consensus_list = get_tournament_odds()

    if not consensus_list:
        st.info(
            "No odds data available in the database. "
            "Tournament odds are populated by background workers via the Odds API."
        )
        return

    st.markdown(f"*Based on {len(consensus_list)} players from sportsbook consensus*")

    # Build rankings table
    rows = []
    for i, c in enumerate(consensus_list[:50], 1):
        prob = c.get("consensus_prob", 0)
        if prob > 0.01:
            implied_odds = f"+{int(1/prob * 100 - 100)}"
        else:
            implied_odds = "+99999"

        rows.append({
            "Rank": i,
            "Player": c.get("player_name", "Unknown"),
            "Win %": f"{prob*100:.1f}%",
            "Implied Odds": implied_odds,
            "Books": c.get("num_books", 0),
            "Min": f"{c.get('min_prob', 0)*100:.1f}%",
            "Max": f"{c.get('max_prob', 0)*100:.1f}%",
        })

    df = pd.DataFrame(rows)

    # Style the win probability column
    def _color_prob(val):
        try:
            v = float(str(val).replace("%", ""))
            if v >= 10:
                return "color: #00FF88; font-weight: bold"
            if v >= 5:
                return "color: #00AAFF"
            if v >= 2:
                return "color: #FFB800"
            return ""
        except (ValueError, TypeError):
            return ""

    styled = df.style.map(_color_prob, subset=["Win %"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=700)

    # SG Comparison for top players
    with st.expander("\U0001f4ca SG Comparison (Top 10)"):
        top_players = consensus_list[:10]
        sg_profiles = []
        for p in top_players:
            name = p.get("player_name", "")
            if name:
                sg = get_player_sg_stats(name, n_events=5)
                if sg.get("found") and sg.get("averages"):
                    profile = {"player_name": name}
                    profile.update(sg["averages"])
                    sg_profiles.append(profile)

        if sg_profiles:
            render_sg_comparison_chart(sg_profiles)
        else:
            st.info("No SG data available for top-ranked players.")
