"""
Golf Quant Engine — Session State Manager
==========================================
Thin wrapper around st.session_state for UI-TRANSIENT state only.

CRITICAL RULES:
  - session_state is ONLY for: current form values being edited,
    UI toggle states, tab selection, ephemeral display state.
  - All PERSISTENT data goes to/from database via services.
  - If Streamlit crashes, nothing of value is lost because all
    persistent state lives in the database.
"""

import streamlit as st

from streamlit_app.config import DEFAULT_USER_ID
from services.settings_service import get_all_settings, set_setting, get_setting


def init_ui_state() -> None:
    """
    Initialize transient UI state defaults on first run.
    Called once at app startup. Only sets keys that do not already exist
    in session_state.
    """
    _defaults = {
        # UI filter states (transient, not persisted)
        "pp_stat_filter": "All",
        "pp_conf_filter": "All",
        "pp_side_filter": "All",
        "pp_sort_by": "Edge %",
        # Bet tracker form state
        "bet_form_player": "",
        "bet_form_market": "",
        "bet_form_direction": "OVER",
        "bet_form_line": 0.0,
        "bet_form_stake": 0.0,
        "bet_form_odds": 1.87,
        # Bet tracker period filter
        "pnl_period": "all",
        "bet_history_days": 30,
        # System tab toggles
        "show_debug_json": False,
    }

    for key, default in _defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get_ui_state(key: str, default=None):
    """
    Read a transient UI state value from session_state.
    Returns default if key does not exist.
    """
    return st.session_state.get(key, default)


def set_ui_state(key: str, value) -> None:
    """
    Write a transient UI state value to session_state.
    This data is ephemeral and will be lost on Streamlit restart.
    """
    st.session_state[key] = value


def load_user_settings(user_id: str = DEFAULT_USER_ID) -> dict:
    """
    Load all user settings from the database via settings_service.
    Returns a dict with bankroll, kelly_fraction, min_edge, etc.
    Every expected key is present (merged with defaults).
    """
    return get_all_settings(user_id)


def save_user_setting(key: str, value, user_id: str = DEFAULT_USER_ID) -> None:
    """
    Write a single user setting to the database via settings_service.
    Survives Streamlit crashes.
    """
    set_setting(user_id, key, value)


def save_user_settings(settings: dict, user_id: str = DEFAULT_USER_ID) -> None:
    """
    Write multiple user settings to the database via settings_service.
    """
    for key, value in settings.items():
        set_setting(user_id, key, value)


def get_user_setting(key: str, default=None, user_id: str = DEFAULT_USER_ID):
    """
    Read a single user setting from the database.
    Falls back to default if not found.
    """
    return get_setting(user_id, key, default)
