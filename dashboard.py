"""
Golf Quant Engine v2.0 — Main Streamlit Dashboard
===================================================
Thin frontend entry point. ALL persistent state lives in the database.
ALL computation is performed in the services layer or background workers.
If Streamlit crashes and restarts, ZERO data is lost.

Architecture:
    Streamlit (this file)
        └── streamlit_app/pages/*.py  (tab renderers, read DB via services)
        └── streamlit_app/components/ (charts, sidebar)
        └── services/*               (all computation + DB read/write)
        └── database/*               (connection, ORM models, migrations)
        └── workers/*                (background ingestion, signals, retraining)

Usage:
    streamlit run dashboard.py
"""

import sys
import os
import logging
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

# ── Page config (MUST be the first Streamlit call) ──────────────────────────
from streamlit_app.config import PAGE_CONFIG, TABS, PREMIUM_CSS

st.set_page_config(**PAGE_CONFIG)

# ── Initialize database ─────────────────────────────────────────────────────
from database.connection import init_db
from database.migrations import auto_migrate

init_db()
auto_migrate()

# ── Initialize transient UI state ───────────────────────────────────────────
from streamlit_app.state import init_ui_state

init_ui_state()

# ── Apply premium CSS theme ─────────────────────────────────────────────────
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# ── Render sidebar (tournament info, weather, settings — all from DB) ───────
from streamlit_app.components.sidebar import render as render_sidebar

render_sidebar()

# ── Render tabs ─────────────────────────────────────────────────────────────
from streamlit_app.pages import (
    prizepicks_tab,
    rankings_tab,
    leaderboard_tab,
    bet_tracker_tab,
    system_tab,
    quant_tab,
    clv_tab,
    data_quality_tab,
    edge_sources_tab,
    edge_decomposition_tab,
    attribution_tab,
)

tab_pp, tab_rank, tab_lb, tab_bets, tab_sys, tab_quant, tab_clv, tab_dq, tab_edge, tab_decomp, tab_attr = st.tabs(TABS)

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

with tab_clv:
    clv_tab.render()

with tab_dq:
    data_quality_tab.render()

with tab_edge:
    edge_sources_tab.render()

with tab_decomp:
    edge_decomposition_tab.render()

with tab_attr:
    attribution_tab.render()
