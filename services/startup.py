"""Application startup — Initialize database, run migrations, sync state.

Call this ONCE at the top of dashboard.py:
    from services.startup import initialize_app
    initialize_app()
"""

from __future__ import annotations

import logging
import streamlit as st

logger = logging.getLogger(__name__)


def initialize_app():
    """Initialize the application on first load.

    - Creates database tables
    - Runs pending migrations
    - Loads persistent settings into session_state
    - Sets up the DataService singleton
    """
    if st.session_state.get("_app_initialized"):
        return

    try:
        from database.connection import get_engine
        from database.migrations import MigrationManager

        # Initialize DB engine (creates tables)
        get_engine()

        # Run any pending migrations
        MigrationManager.run_pending()

        # Load persistent settings from DB into session_state
        _sync_settings_from_db()

        st.session_state["_app_initialized"] = True
        logger.info("Application initialized successfully")

    except Exception as e:
        logger.exception("Application initialization failed")
        # Don't crash the app — just log the error
        st.session_state["_app_init_error"] = str(e)
        st.session_state["_app_initialized"] = True


def _sync_settings_from_db():
    """Load persistent user settings from database into session_state."""
    try:
        from services.data_service import DataService
        ds = DataService("golf")

        # Bankroll
        saved_bankroll = ds.get_setting("bankroll")
        if saved_bankroll is not None and "bankroll_total" not in st.session_state:
            st.session_state["bankroll_total"] = saved_bankroll

        # Bankroll history
        saved_history = ds.get_setting("bankroll_history")
        if saved_history and "bankroll_history" not in st.session_state:
            st.session_state["bankroll_history"] = saved_history

        # Logged bets
        saved_bets = ds.get_setting("logged_bets")
        if saved_bets and "logged_bets" not in st.session_state:
            st.session_state["logged_bets"] = saved_bets

    except Exception:
        logger.debug("Failed to sync settings from DB — will use defaults")


def save_state_to_db():
    """Persist current session_state to database. Call on user actions."""
    try:
        from services.data_service import DataService
        ds = DataService("golf")

        if "bankroll_total" in st.session_state:
            ds.save_setting("bankroll", st.session_state["bankroll_total"], "risk")

        if "bankroll_history" in st.session_state:
            ds.save_setting("bankroll_history", st.session_state["bankroll_history"], "risk")

        if "logged_bets" in st.session_state:
            ds.save_setting("logged_bets", st.session_state["logged_bets"], "bets")

    except Exception:
        logger.debug("Failed to save state to DB")


def get_data_service() -> "DataService":
    """Get or create the DataService singleton."""
    if "_data_service" not in st.session_state:
        from services.data_service import DataService
        st.session_state["_data_service"] = DataService("golf")
    return st.session_state["_data_service"]
