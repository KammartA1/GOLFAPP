"""
Golf Quant Engine — Sidebar Component
=======================================
Renders the sidebar with tournament info, weather, system status,
and user settings. All data comes from database via services.
All settings changes write to database immediately.
"""

import streamlit as st

from streamlit_app.config import (
    DEFAULT_USER_ID,
    STATUS_DISPLAY,
    SYSTEM_STATE_COLORS,
)
from streamlit_app.state import load_user_settings, save_user_setting

from services.event_service import get_current_tournament
from services.weather_service import get_current_weather, get_weather_adjustment_factor
from services.odds_service import get_prizepicks_lines, get_tournament_odds
from services.report_service import get_system_state
from database.connection import health_check


def render() -> None:
    """Render the full sidebar. Reads all data from DB via services."""
    with st.sidebar:
        st.markdown("## \u26f3 Golf Quant Engine")

        _render_tournament_info()
        st.divider()
        _render_weather()
        st.divider()
        _render_system_status()
        st.divider()
        _render_settings()
        st.divider()
        _render_refresh_buttons()


def _render_tournament_info() -> None:
    """Display current tournament info from event_service."""
    tournament = get_current_tournament()

    if tournament and not tournament.get("error") and tournament.get("event_name"):
        name = tournament.get("event_name", tournament.get("name", "Unknown"))
        st.success(f"**{name}**")

        course = tournament.get("course_name", "")
        if course:
            st.caption(f"\U0001f4cd {course}")

        status = tournament.get("status", "")
        status_display = STATUS_DISPLAY.get(status, status)
        st.caption(f"Status: {status_display}")

        field_count = tournament.get("field_count", 0)
        venue = tournament.get("venue", "")
        info_parts = []
        if field_count:
            info_parts.append(f"Field: {field_count} players")
        if venue:
            info_parts.append(f"Venue: {venue}")
        if info_parts:
            st.caption(" | ".join(info_parts))
    else:
        st.warning("No tournament detected")
        if tournament and tournament.get("error"):
            st.caption(f"Error: {tournament['error']}")


def _render_weather() -> None:
    """Display weather from weather_service using DB-cached data."""
    tournament = get_current_tournament()
    course_name = ""
    if tournament:
        course_name = tournament.get("course_name", "")

    if not course_name:
        return

    weather = get_current_weather(course_name)

    if not weather or weather.get("error"):
        st.caption("\U0001f327\ufe0f Weather: No data available")
        return

    st.markdown("### \U0001f324\ufe0f Course Weather")

    temp_f = weather.get("temp_f", 0)
    humidity = weather.get("humidity_pct", 0)
    wind_mph = weather.get("wind_speed_mph", 0)
    wind_gust = weather.get("wind_gust_mph", 0)
    cloud_cover = weather.get("cloud_cover_pct", 0)
    description = weather.get("description", "")

    col1, col2 = st.columns(2)
    col1.metric("Temp", f"{temp_f:.0f}\u00b0F")
    col2.metric("Humidity", f"{humidity}%")

    col1, col2 = st.columns(2)
    wind_dir_deg = weather.get("wind_direction_deg", 0)
    wind_dir_str = _wind_direction(wind_dir_deg)
    col1.metric("Wind", f"{wind_mph:.0f} mph {wind_dir_str}")
    if wind_gust:
        col2.metric("Gusts", f"{wind_gust:.0f} mph")
    else:
        col2.metric("Clouds", f"{cloud_cover}%")

    if description:
        st.caption(f"\u2601\ufe0f {description.title()}")

    # Weather scoring adjustment
    adjustment = get_weather_adjustment_factor(weather)
    if adjustment > 0.1:
        st.warning(f"\u26a0\ufe0f Weather impact: +{adjustment:.2f} strokes")


def _render_system_status() -> None:
    """Display system status indicators from DB."""
    st.markdown("### \U0001f4e1 System Status")

    # Database health
    db_health = health_check()
    if db_health.get("status") == "ok":
        st.markdown(f"\U0001f7e2 **Database**: OK ({db_health.get('driver', 'sqlite')})")
    else:
        st.markdown(f"\U0001f534 **Database**: {db_health.get('error', 'Error')}")

    # PrizePicks lines from DB
    pp_lines = get_prizepicks_lines()
    pp_count = len(pp_lines)
    if pp_count > 0:
        st.markdown(f"\U0001f7e2 **PrizePicks**: {pp_count} lines in DB")
    else:
        st.markdown("\U0001f7e1 **PrizePicks**: No lines in DB")

    # Tournament odds from DB
    odds = get_tournament_odds()
    odds_count = len(odds)
    if odds_count > 0:
        st.markdown(f"\U0001f7e2 **Odds**: {odds_count} players")
    else:
        st.markdown("\U0001f7e1 **Odds**: No data in DB")

    # System state
    sys_state = get_system_state()
    state = sys_state.get("state", "UNKNOWN")
    icon = SYSTEM_STATE_COLORS.get(state, "\u26aa")
    st.markdown(f"{icon} **System**: {state}")


def _render_settings() -> None:
    """
    Render user settings controls. Reads from DB, writes changes to DB.
    Uses st.session_state only for the widget keys (required by Streamlit).
    """
    st.markdown("### \u2699\ufe0f Settings")

    # Load current settings from DB
    settings = load_user_settings(DEFAULT_USER_ID)

    current_bankroll = float(settings.get("bankroll", 1000))
    current_min_edge = float(settings.get("min_edge", 0.03))
    current_kelly = float(settings.get("kelly_fraction", 0.25))

    # Bankroll input
    new_bankroll = st.number_input(
        "Bankroll ($)",
        min_value=10.0,
        value=current_bankroll,
        step=100.0,
        key="sidebar_bankroll_input",
    )
    if new_bankroll != current_bankroll:
        save_user_setting("bankroll", new_bankroll, DEFAULT_USER_ID)

    # Min edge slider
    edge_pct = int(current_min_edge * 100)
    new_edge_pct = st.slider(
        "Min Edge %",
        min_value=1,
        max_value=20,
        value=edge_pct,
        key="sidebar_edge_input",
    )
    new_min_edge = new_edge_pct / 100.0
    if new_min_edge != current_min_edge:
        save_user_setting("min_edge", new_min_edge, DEFAULT_USER_ID)

    # Kelly fraction
    kelly_options = [0.10, 0.15, 0.20, 0.25, 0.33, 0.50]
    # Find closest match in options
    closest_kelly = min(kelly_options, key=lambda x: abs(x - current_kelly))
    new_kelly = st.select_slider(
        "Kelly Fraction",
        options=kelly_options,
        value=closest_kelly,
        key="sidebar_kelly_input",
    )
    if new_kelly != current_kelly:
        save_user_setting("kelly_fraction", new_kelly, DEFAULT_USER_ID)


def _render_refresh_buttons() -> None:
    """Render refresh buttons. Clearing cache forces re-read from DB."""
    col1, col2 = st.columns(2)
    if col1.button("\U0001f504 Refresh All", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    if col2.button("\U0001f504 Reload", use_container_width=True):
        st.rerun()


def _wind_direction(degrees: float) -> str:
    """Convert wind direction in degrees to compass string."""
    if degrees is None or degrees == 0:
        return ""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                   "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(degrees / 22.5) % 16
    return directions[idx]
