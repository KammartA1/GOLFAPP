"""
Golf Quant Engine — Reusable Chart Components
===============================================
All chart functions take data as input and return plotly figures
or render Streamlit elements. No computation, no DB access.
"""

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def render_sg_comparison_chart(players: list[dict]) -> None:
    """
    Render a grouped bar chart comparing SG categories across players.

    Each dict in players should have:
        player_name, sg_total, sg_ott, sg_app, sg_atg, sg_putt
    """
    if not players or not HAS_PLOTLY:
        st.info("No SG data to chart.")
        return

    categories = ["sg_ott", "sg_app", "sg_atg", "sg_putt"]
    labels = ["Off the Tee", "Approach", "Around Green", "Putting"]

    fig = go.Figure()

    for p in players[:10]:
        name = p.get("player_name", "Unknown")
        vals = [p.get(cat, 0) or 0 for cat in categories]
        fig.add_trace(go.Bar(name=name, x=labels, y=vals))

    fig.update_layout(
        barmode="group",
        title="Strokes Gained Comparison",
        yaxis_title="Strokes Gained",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_pnl_curve(bets: list[dict]) -> None:
    """
    Render a cumulative P&L line chart over time.

    Each dict should have: settled_at (or timestamp), pnl (or profit).
    Bets should be in chronological order (oldest first).
    """
    if not bets or not HAS_PLOTLY:
        st.info("No settled bets to chart.")
        return

    cumulative = 0.0
    dates = []
    values = []

    for b in bets:
        dt = b.get("settled_at") or b.get("timestamp") or ""
        pnl = b.get("pnl") or b.get("profit") or 0

        cumulative += float(pnl)
        dates.append(str(dt)[:10] if dt else "")
        values.append(cumulative)

    if not dates:
        st.info("No P&L data to display.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode="lines+markers",
        line=dict(color="#00FF88" if cumulative >= 0 else "#FF4444", width=2),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(0,255,136,0.1)" if cumulative >= 0 else "rgba(255,68,68,0.1)",
    ))

    fig.update_layout(
        title=f"Cumulative P&L: ${cumulative:+.2f}",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L ($)",
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    st.plotly_chart(fig, use_container_width=True)


def render_calibration_chart(calibration_data: list[dict]) -> None:
    """
    Render a calibration chart: predicted probability vs actual hit rate.

    Each dict should have: predicted_avg, actual_rate, n_bets, bucket_label.
    A well-calibrated model follows the 45-degree line.
    """
    if not calibration_data or not HAS_PLOTLY:
        st.info("No calibration data available.")
        return

    predicted = [d.get("predicted_avg", 0) for d in calibration_data]
    actual = [d.get("actual_rate", 0) for d in calibration_data]
    labels = [d.get("bucket_label", "") for d in calibration_data]
    sizes = [max(8, min(30, (d.get("n_bets", 1) or 1))) for d in calibration_data]

    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="gray", dash="dash", width=1),
        name="Perfect",
        showlegend=True,
    ))

    # Actual calibration points
    fig.add_trace(go.Scatter(
        x=predicted,
        y=actual,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=actual,
            colorscale="RdYlGn",
            showscale=False,
        ),
        text=labels,
        textposition="top center",
        textfont=dict(size=9),
        name="Model",
    ))

    fig.update_layout(
        title="Model Calibration",
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Hit Rate",
        template="plotly_dark",
        height=400,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        margin=dict(l=40, r=20, t=50, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_edge_distribution(analyses: list[dict]) -> None:
    """
    Render a histogram of edge percentages across all analyzed lines.

    Each dict should have: best_edge (float, e.g. 0.05 for 5%).
    """
    if not analyses or not HAS_PLOTLY:
        return

    edges = [a.get("best_edge", 0) * 100 for a in analyses if a.get("best_edge")]

    if not edges:
        return

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=edges,
        nbinsx=20,
        marker_color="#00AAFF",
        opacity=0.75,
    ))

    fig.update_layout(
        title="Edge Distribution (%)",
        xaxis_title="Edge %",
        yaxis_title="Count",
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_roi_by_market(bet_history: list[dict]) -> None:
    """
    Render a bar chart of ROI by market type.

    Each dict should have: market, stake, pnl (or profit).
    """
    if not bet_history or not HAS_PLOTLY:
        return

    by_market: dict[str, dict] = {}
    for b in bet_history:
        market = b.get("market") or "unknown"
        stake = float(b.get("stake") or 0)
        pnl = float(b.get("pnl") or b.get("profit") or 0)

        if market not in by_market:
            by_market[market] = {"total_stake": 0, "total_pnl": 0, "count": 0}
        by_market[market]["total_stake"] += stake
        by_market[market]["total_pnl"] += pnl
        by_market[market]["count"] += 1

    if not by_market:
        return

    markets = []
    rois = []
    colors = []

    for mkt, data in sorted(by_market.items()):
        roi = (data["total_pnl"] / data["total_stake"] * 100) if data["total_stake"] > 0 else 0
        markets.append(f"{mkt} ({data['count']})")
        rois.append(round(roi, 1))
        colors.append("#00FF88" if roi >= 0 else "#FF4444")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=markets,
        y=rois,
        marker_color=colors,
    ))

    fig.update_layout(
        title="ROI by Market",
        yaxis_title="ROI %",
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    st.plotly_chart(fig, use_container_width=True)


def render_clv_chart(clv_data: dict) -> None:
    """
    Render a CLV summary display with key metrics.

    clv_data should have: avg_clv, median_clv, pct_positive, n_bets, clv_by_market.
    """
    if not clv_data or clv_data.get("n_bets", 0) == 0:
        st.info("No CLV data available yet. Bets need closing lines to compute CLV.")
        return

    cols = st.columns(4)
    avg_clv = clv_data.get("avg_clv", 0)
    cols[0].metric(
        "Avg CLV",
        f"{avg_clv:+.4f}",
        delta="Positive" if avg_clv > 0 else "Negative",
        delta_color="normal" if avg_clv > 0 else "inverse",
    )
    cols[1].metric("Median CLV", f"{clv_data.get('median_clv', 0):+.4f}")
    cols[2].metric("% Positive", f"{clv_data.get('pct_positive', 0) * 100:.1f}%")
    cols[3].metric("Sample Size", clv_data.get("n_bets", 0))

    # Per-market CLV breakdown
    by_market = clv_data.get("clv_by_market", {})
    if by_market and HAS_PLOTLY:
        markets = []
        clvs = []
        colors = []

        for mkt, vals in sorted(by_market.items()):
            avg = vals.get("avg_clv", 0)
            markets.append(f"{mkt} (n={vals.get('n_bets', 0)})")
            clvs.append(avg)
            colors.append("#00FF88" if avg > 0 else "#FF4444")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=markets, y=clvs, marker_color=colors))
        fig.update_layout(
            title="CLV by Market",
            yaxis_title="Avg CLV",
            template="plotly_dark",
            height=250,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
