"""
Golf Quant Engine — System Status Tab
========================================
Worker status from database worker_status table.
API health checks from database.connection.health_check().
Report data from report_service.
"""

import json
import streamlit as st
from datetime import datetime

from database.connection import health_check
from database.migrations import get_migration_status
from database.models import WorkerStatus

from services._db import get_session as _session
from services.report_service import get_latest_report, get_sg_accuracy_report
from services.event_service import get_current_tournament
from services.odds_service import get_prizepicks_lines, get_tournament_odds
from services.weather_service import get_current_weather


def render() -> None:
    """Render the System Status tab. All data from DB."""
    st.markdown("# \U0001f527 System Status & Debug")

    col1, col2 = st.columns(2)

    with col1:
        _render_database_health()
        _render_worker_status()
        _render_tournament_context()

    with col2:
        _render_data_status()
        _render_sg_accuracy()
        _render_migration_status()

    # Debug JSON (collapsible)
    with st.expander("\U0001f41b Debug Information"):
        _render_debug_info()


def _render_database_health() -> None:
    """Display database health check results."""
    st.markdown("### Database Health")

    health = health_check()
    status = health.get("status", "unknown")

    if status == "ok":
        st.success("Database is healthy")
        st.json({
            "driver": health.get("driver", "unknown"),
            "version": health.get("version", "unknown"),
            "journal_mode": health.get("journal_mode", "unknown"),
        })
    else:
        st.error(f"Database error: {health.get('error', 'Unknown')}")


def _render_worker_status() -> None:
    """Display status of all background workers from the worker_status table."""
    st.markdown("### Worker Status")

    try:
        with _session() as session:
            workers = (
                session.query(WorkerStatus)
                .order_by(WorkerStatus.worker_name)
                .all()
            )
            worker_list = [w.to_dict() for w in workers]
    except Exception as exc:
        st.warning(f"Could not query worker status: {exc}")
        worker_list = []

    if not worker_list:
        st.info("No workers registered. Workers register themselves when they run.")
        return

    for w in worker_list:
        name = w.get("worker_name", "unknown")
        status = w.get("status", "unknown")
        last_run = w.get("last_run", "Never")
        last_success = w.get("last_success", "Never")
        last_error = w.get("last_error", "")
        next_run = w.get("next_scheduled_run", "")

        # Status icon
        if status == "running":
            icon = "\U0001f7e2"
        elif status == "idle":
            icon = "\U0001f7e1"
        elif status == "error":
            icon = "\U0001f534"
        else:
            icon = "\u26aa"

        with st.container():
            st.markdown(f"{icon} **{name}** ({status})")
            detail_cols = st.columns(3)
            detail_cols[0].caption(f"Last run: {_format_dt(last_run)}")
            detail_cols[1].caption(f"Last success: {_format_dt(last_success)}")
            if next_run:
                detail_cols[2].caption(f"Next: {_format_dt(next_run)}")

            if last_error:
                st.caption(f"Last error: {last_error[:200]}")


def _render_tournament_context() -> None:
    """Display current tournament context from event_service."""
    st.markdown("### Tournament Context")

    tournament = get_current_tournament()

    if tournament and tournament.get("event_name"):
        st.json({
            "id": tournament.get("id"),
            "name": tournament.get("event_name"),
            "course": tournament.get("course_name"),
            "venue": tournament.get("venue"),
            "status": tournament.get("status"),
            "start_time": str(tournament.get("start_time", "")),
            "field_count": tournament.get("field_count", 0),
        })
    else:
        st.warning("No tournament context in database")


def _render_data_status() -> None:
    """Display counts of data in various tables."""
    st.markdown("### Data Status")

    pp_lines = get_prizepicks_lines()
    odds = get_tournament_odds()

    # Weather
    tournament = get_current_tournament()
    course = tournament.get("course_name", "") if tournament else ""
    weather = get_current_weather(course) if course else {}
    has_weather = bool(weather and not weather.get("error"))

    st.json({
        "prizepicks_lines": len(pp_lines),
        "tournament_odds_players": len(odds),
        "weather_available": has_weather,
    })

    # Latest reports
    daily_report = get_latest_report("daily")
    weekly_report = get_latest_report("weekly")

    st.markdown("### Latest Reports")
    if daily_report:
        st.caption(
            f"Daily: generated {daily_report.get('generated_at', 'unknown')}"
        )
    else:
        st.caption("Daily: No reports generated yet")

    if weekly_report:
        st.caption(
            f"Weekly: generated {weekly_report.get('generated_at', 'unknown')}"
        )
    else:
        st.caption("Weekly: No reports generated yet")


def _render_sg_accuracy() -> None:
    """Display SG accuracy report from report_service."""
    st.markdown("### Model Accuracy")

    accuracy = get_sg_accuracy_report()

    if accuracy.get("sample_size", 0) == 0:
        st.info("No accuracy data yet. Need settled bets with model projections.")
        return

    cols = st.columns(3)
    cols[0].metric(
        "Overall Accuracy",
        f"{accuracy.get('overall_accuracy', 0)*100:.1f}%",
    )
    cols[1].metric(
        "Avg Abs Error",
        f"{accuracy.get('avg_absolute_error', 0):.3f}",
    )
    cols[2].metric("Sample Size", accuracy.get("sample_size", 0))

    # By confidence tier
    by_conf = accuracy.get("by_confidence", {})
    if by_conf:
        st.markdown("**Accuracy by Confidence Tier:**")
        for tier, data in sorted(by_conf.items()):
            acc = data.get("accuracy", 0)
            n = data.get("sample_size", 0)
            st.caption(f"  {tier}: {acc*100:.1f}% (n={n})")


def _render_migration_status() -> None:
    """Display database migration status."""
    st.markdown("### Schema Migrations")

    try:
        mig_status = get_migration_status()
        if mig_status.get("is_up_to_date"):
            st.success(f"Schema up to date (v{mig_status.get('current_version', 0)})")
        else:
            pending = mig_status.get("pending_versions", [])
            st.warning(
                f"Schema at v{mig_status.get('current_version', 0)}, "
                f"{len(pending)} migrations pending: {pending}"
            )
    except Exception as exc:
        st.warning(f"Could not check migration status: {exc}")


def _render_debug_info() -> None:
    """Display debug information for troubleshooting."""
    import sys
    from pathlib import Path

    st.json({
        "python_version": sys.version,
        "app_root": str(Path(__file__).resolve().parent.parent.parent),
        "timestamp": datetime.utcnow().isoformat(),
    })

    # Raw data dumps
    st.markdown("#### Raw PrizePicks Lines (first 5)")
    pp = get_prizepicks_lines()
    for line in pp[:5]:
        st.json(line)

    st.markdown("#### Raw Tournament Odds (first 5)")
    odds = get_tournament_odds()
    for o in odds[:5]:
        st.json(o)


def _format_dt(val) -> str:
    """Format a datetime value for display."""
    if val is None:
        return "Never"
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d %H:%M")
    s = str(val)
    if s and s != "None":
        return s[:16]
    return "Never"
