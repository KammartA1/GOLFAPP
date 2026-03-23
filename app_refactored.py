"""
Golf Quant Engine — Refactored Entry Point
============================================
Pure frontend Streamlit app. Reads/writes from database only.
No computation in Streamlit. If Streamlit crashes, zero data is lost.

Usage:
    streamlit run app_refactored.py

Architecture:
    - All persistent state lives in SQLite/PostgreSQL via services layer
    - st.session_state is ONLY used for transient UI state (filters, toggles)
    - Background workers handle scraping and computation
    - This app is a thin read/write layer over the database
"""

import sys
import logging
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# ---- Page config (must be first Streamlit call) ----
from streamlit_app.config import PAGE_CONFIG, TABS, PREMIUM_CSS
st.set_page_config(**PAGE_CONFIG)

# ---- Initialize database ----
from database.connection import init_db
from database.migrations import auto_migrate

init_db()
auto_migrate()

# ---- Initialize UI state ----
from streamlit_app.state import init_ui_state
init_ui_state()

# ---- Apply theme ----
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# ---- Render sidebar ----
from streamlit_app.components.sidebar import render as render_sidebar
render_sidebar()

# ---- Render tabs ----
from streamlit_app.pages import (
    prizepicks_tab,
    rankings_tab,
    leaderboard_tab,
    bet_tracker_tab,
    system_tab,
    quant_tab,
)

tab_pp, tab_rank, tab_lb, tab_bets, tab_sys, tab_quant = st.tabs(TABS)

with tab_pp:
    prizepicks_tab.render()

with tab_rank:
    rankings_tab.render()

with tab_lb:
    leaderboard_tab.render()

with tab_bets:
    bet_tracker_tab.render()

with tab_sys:
    system_tab.render()

with tab_quant:
    quant_tab.render()
