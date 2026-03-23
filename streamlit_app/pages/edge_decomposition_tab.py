"""
streamlit_app/pages/edge_decomposition_tab.py
===============================================
EDGE DECOMPOSITION — Atomic-level golf performance attribution.

5 components analyzed:
  1. Predictive — Brier per market (outright/top5/top10/top20/cut/matchup)
  2. Informational — Weather timing, injury/WD timing, wave advantage
  3. Market Inefficiency — CLV per market type (outrights notoriously inefficient)
  4. Execution — Line shopping effectiveness
  5. Structural — Field diversification, wave correlation, Kelly

Final verdict: "Which component is doing the heavy lifting — and which are illusions?"
"""

from __future__ import annotations

import logging

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

log = logging.getLogger(__name__)

_GREEN = "#10B981"
_RED = "#EF4444"
_YELLOW = "#F59E0B"
_MUTED = "#6B7280"
_BG = "rgba(0,0,0,0)"


def _get_report():
    from edge_analysis.report import EdgeReportGenerator
    gen = EdgeReportGenerator(sport="golf")
    return gen.generate(), gen


def render():
    st.markdown("### Edge Decomposition — Atomic Level")
    st.caption("Which component is doing the heavy lifting — and which are illusions?")

    tab_overview, tab_predictive, tab_market, tab_execution, tab_structural, tab_verdict = st.tabs([
        "Overview", "Predictive", "CLV / Market", "Execution", "Structural", "Verdict",
    ])

    try:
        report, gen = _get_report()
    except Exception as exc:
        st.error(f"Edge decomposition unavailable: {exc}")
        _render_empty_state()
        return

    if report.total_bets < 10:
        _render_empty_state()
        return

    with tab_overview:
        _render_overview(report)
    with tab_predictive:
        _render_predictive(report)
    with tab_market:
        _render_market(report)
    with tab_execution:
        _render_execution(report)
    with tab_structural:
        _render_structural(report)
    with tab_verdict:
        _render_verdict(report)


def _render_overview(report):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total ROI", f"{report.total_roi:+.2f}%")
    with col2:
        st.metric("Total Bets", f"{report.total_bets}")
    with col3:
        st.metric("Total P&L", f"${report.total_pnl:+,.2f}")
    with col4:
        st.metric("Heavy Lifter", report.heavy_lifter)

    st.divider()

    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        st.markdown("##### Edge Attribution")
        labels = ["Predictive", "Informational", "Market Inefficiency", "Execution", "Structural"]
        values = [
            abs(report.predictive_pct), abs(report.informational_pct),
            abs(report.market_pct), abs(report.execution_pct), abs(report.structural_pct),
        ]
        colors_list = []
        for comp_name in ["predictive", "informational", "market_inefficiency", "execution", "structural"]:
            comp = getattr(report, comp_name, None)
            if comp and comp.is_positive and comp.is_significant:
                colors_list.append(_GREEN)
            elif comp and comp.is_positive:
                colors_list.append(_YELLOW)
            else:
                colors_list.append(_RED)

        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=0.55,
            marker=dict(colors=colors_list),
            textinfo="label+percent",
            textfont=dict(size=11),
        )])
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=_BG, plot_bgcolor=_BG,
            height=380, margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
            annotations=[dict(
                text=f"<b>{report.total_roi:+.1f}%</b><br>ROI",
                x=0.5, y=0.5, font_size=16, showarrow=False,
                font=dict(color=_GREEN if report.total_roi > 0 else _RED),
            )],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("##### Component Status")
        rows = []
        for comp_name, label in [
            ("predictive", "Predictive"), ("informational", "Informational"),
            ("market_inefficiency", "Market Inefficiency"),
            ("execution", "Execution"), ("structural", "Structural"),
        ]:
            comp = getattr(report, comp_name, None)
            if comp:
                status = "REAL" if comp.is_positive and comp.is_significant else (
                    "POSSIBLE" if comp.is_positive else "ILLUSION"
                )
                rows.append({
                    "Component": label,
                    "Share": f"{comp.edge_pct_of_roi:+.1f}%",
                    "Status": status,
                    "P-Value": f"{comp.p_value:.4f}",
                    "N": comp.sample_size,
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if report.illusions:
            st.warning(f"Illusions: **{', '.join(report.illusions)}**")
        else:
            st.success("All components show genuine contribution")

    # Bar chart
    st.markdown("##### Attribution Breakdown (signed)")
    names = ["Predictive", "Informational", "Market\nInefficiency", "Execution", "Structural"]
    vals = [report.predictive_pct, report.informational_pct, report.market_pct,
            report.execution_pct, report.structural_pct]
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=names, y=vals,
        marker_color=[_GREEN if v > 0 else _RED for v in vals],
        text=[f"{v:+.1f}%" for v in vals], textposition="outside",
    ))
    fig_bar.add_hline(y=0, line_dash="dash", line_color=_MUTED)
    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        yaxis_title="% of Total Edge", height=320,
        margin=dict(l=40, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


def _render_predictive(report):
    comp = report.predictive
    if not comp:
        st.info("Predictive edge analysis not available")
        return

    st.markdown(f"**Verdict:** {comp.verdict}")

    d = comp.details
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Brier (Model)", f"{d.get('brier_model', 0):.4f}")
    with col2:
        st.metric("Brier (Market)", f"{d.get('brier_market', 0):.4f}")
    with col3:
        st.metric("Brier Skill", f"{d.get('brier_skill', 0):.1%}")
    with col4:
        st.metric("P-Value", f"{comp.p_value:.4f}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Log Loss (Model)", f"{d.get('logloss_model', 0):.4f}")
    with col6:
        st.metric("Log Loss (Market)", f"{d.get('logloss_market', 0):.4f}")
    with col7:
        st.metric("Log Loss Skill", f"{d.get('logloss_skill', 0):.1%}")
    with col8:
        st.metric("Mean Cal Error", f"{d.get('mean_calibration_error', 0):.1%}")

    st.divider()

    # Per market type Brier breakdown
    by_market = d.get("by_market_type", {})
    if by_market:
        st.markdown("##### Brier Score by Market Type")
        mkt_rows = []
        for mtype, metrics in sorted(by_market.items()):
            if metrics.get("insufficient"):
                mkt_rows.append({
                    "Market": mtype, "N": metrics["n"], "Brier Model": "N/A",
                    "Brier Market": "N/A", "Skill vs Market": "N/A", "Skill vs Naive": "N/A",
                })
            else:
                mkt_rows.append({
                    "Market": mtype,
                    "N": metrics["n"],
                    "Brier Model": f"{metrics['brier_model']:.4f}",
                    "Brier Market": f"{metrics['brier_market']:.4f}",
                    "Skill vs Market": f"{metrics['skill_vs_market']:.1%}",
                    "Skill vs Naive": f"{metrics['skill_vs_naive']:.1%}",
                })
        st.dataframe(pd.DataFrame(mkt_rows), use_container_width=True, hide_index=True)

    # Calibration curve
    cal_data = d.get("calibration_curve", [])
    if cal_data:
        st.markdown("##### Calibration Curve")
        predicted = [p["predicted"] for p in cal_data]
        actual = [p["actual"] for p in cal_data]
        n_bets = [p["n"] for p in cal_data]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
            line=dict(color=_MUTED, dash="dash", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=predicted, y=actual, mode="lines+markers", name="Model",
            line=dict(color=_GREEN, width=2),
            marker=dict(size=[max(4, min(15, n / 3)) for n in n_bets]),
            text=[f"n={n}" for n in n_bets],
            hovertemplate="Predicted: %{x:.1%}<br>Actual: %{y:.1%}<br>%{text}",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=_BG, plot_bgcolor=_BG,
            xaxis_title="Predicted Probability", yaxis_title="Actual Win Rate",
            height=400, margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_market(report):
    comp = report.market_inefficiency
    if not comp:
        st.info("Market inefficiency analysis not available")
        return

    st.markdown(f"**Verdict:** {comp.verdict}")

    d = comp.details
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg CLV", f"{d.get('avg_clv', 0):+.4f}")
    with col2:
        st.metric("Beat Close Rate", f"{d.get('beat_close_rate', 0):.1%}")
    with col3:
        st.metric("Median CLV", f"{d.get('median_clv', 0):+.4f}")
    with col4:
        st.metric("P-Value", f"{comp.p_value:.4f}")

    st.divider()

    # CLV distribution
    clv_dist = d.get("clv_distribution", [])
    if clv_dist:
        st.markdown("##### CLV Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=clv_dist, nbinsx=40, marker_color=_GREEN, opacity=0.7))
        fig.add_vline(x=0, line_dash="dash", line_color=_RED, line_width=2)
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=_BG, plot_bgcolor=_BG,
            xaxis_title="CLV", yaxis_title="Count", height=350,
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # CLV by market type
    clv_by = d.get("clv_by_market", {})
    if clv_by:
        st.markdown("##### CLV by Market Type")
        rows = []
        for seg, m in sorted(clv_by.items(), key=lambda x: x[1]["avg_clv"], reverse=True):
            rows.append({
                "Market": seg,
                "Avg CLV": f"{m['avg_clv']:+.4f}",
                "Beat Close": f"{m['beat_close_pct']:.1%}",
                "N": m["n_bets"],
                "Significant": m.get("is_significant", False),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Bar chart
        markets = [r["Market"] for r in rows]
        clv_vals = [clv_by[m]["avg_clv"] for m in markets]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=markets, y=clv_vals,
            marker_color=[_GREEN if v > 0 else _RED for v in clv_vals],
        ))
        fig.add_hline(y=0, line_dash="dash", line_color=_MUTED)
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=_BG, plot_bgcolor=_BG,
            yaxis_title="Avg CLV", height=300, margin=dict(l=40, r=20, t=20, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_execution(report):
    comp = report.execution
    if not comp:
        st.info("Execution analysis not available")
        return

    st.markdown(f"**Verdict:** {comp.verdict}")

    d = comp.details
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Slippage", f"{d.get('avg_slippage', 0):+.4f}")
    with col2:
        st.metric("Price Improved %", f"{d.get('pct_price_improved', 0):.1%}")
    with col3:
        st.metric("Capture Rate", f"{d.get('avg_capture_rate', 0):.1%}")
    with col4:
        st.metric("Total Drag", f"{d.get('total_drag', 0):+.4f}")

    cost_by = d.get("cost_by_market", {})
    if cost_by:
        st.markdown("##### Execution Cost by Market")
        rows = []
        for seg, m in sorted(cost_by.items(), key=lambda x: x[1]["avg_slippage"]):
            rows.append({
                "Market": seg,
                "Avg Slippage": f"{m['avg_slippage']:+.4f}",
                "Price Improved %": f"{m['pct_improved']:.1%}",
                "N": m["n_bets"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_structural(report):
    comp = report.structural
    if not comp:
        st.info("Structural analysis not available")
        return

    st.markdown(f"**Verdict:** {comp.verdict}")

    d = comp.details
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Correlation (rho)", f"{d.get('correlation_rho', 0):.4f}")
    with col2:
        st.metric("Variance Ratio", f"{d.get('variance_ratio', 1):.3f}")
    with col3:
        st.metric("Sharpe Ratio", f"{d.get('annualized_sharpe', 0):.3f}")
    with col4:
        st.metric("Max Drawdown", f"${d.get('max_drawdown', 0):.2f}")

    st.divider()

    kelly = d.get("kelly", {})
    if kelly.get("sufficient_data"):
        st.markdown("##### Kelly Criterion")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Kelly", f"{kelly.get('avg_kelly_fraction', 0):.4f}")
        with col2:
            st.metric("Kelly Efficiency", f"{kelly.get('kelly_efficiency', 0):.1%}")
        with col3:
            st.metric("Growth Rate", f"{kelly.get('actual_growth_rate', 0):.6f}")

    wave = d.get("wave_analysis", {})
    if wave.get("sufficient_data"):
        st.markdown("##### Wave Correlation Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AM Win Rate", f"{wave.get('am_win_rate', 0):.1%}")
        with col2:
            st.metric("PM Win Rate", f"{wave.get('pm_win_rate', 0):.1%}")
        with col3:
            st.metric("Wave Bets", wave.get("n_wave_bets", 0))

    div = d.get("diversification", {})
    if div:
        st.markdown("##### Portfolio Diversification")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Markets", div.get("n_unique_markets", 0))
        with col2:
            st.metric("Players", div.get("n_unique_players", 0))
        with col3:
            st.metric("Tournaments", div.get("n_unique_tournaments", 0))
        with col4:
            hhi = div.get("hhi_tournament", 1.0)
            label = "Excellent" if hhi < 0.1 else ("OK" if hhi < 0.25 else "Poor")
            st.metric("Tournament HHI", f"{hhi:.4f} ({label})")


def _render_verdict(report):
    st.markdown("##### FINAL VERDICT")

    if report.market_inefficiency and report.market_inefficiency.is_positive and report.market_inefficiency.is_significant:
        color, banner = _GREEN, "GENUINE EDGE DETECTED"
    elif report.total_roi > 0:
        color, banner = _YELLOW, "PROFITABLE BUT UNCONFIRMED"
    else:
        color, banner = _RED, "NO CONFIRMED EDGE"

    st.markdown(f"""
    <div style='text-align: center; padding: 20px; border: 2px solid {color};
    border-radius: 12px; margin: 10px 0;'>
        <div style='font-size: 1.6rem; font-weight: bold; color: {color};'>{banner}</div>
        <div style='font-size: 0.85rem; color: {_MUTED}; margin-top: 8px;'>
        Heavy Lifter: <span style="color: {_GREEN}">{report.heavy_lifter}</span>
        &nbsp;|&nbsp;
        Illusions: <span style="color: {_RED}">{', '.join(report.illusions) if report.illusions else 'None'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.code(report.verdict, language="text")

    st.divider()
    st.markdown("##### Component Verdicts")
    for comp_name, label in [
        ("predictive", "1. PREDICTIVE EDGE"),
        ("informational", "2. INFORMATIONAL EDGE"),
        ("market_inefficiency", "3. MARKET INEFFICIENCY"),
        ("execution", "4. EXECUTION EDGE"),
        ("structural", "5. STRUCTURAL EDGE"),
    ]:
        comp = getattr(report, comp_name, None)
        if comp:
            status = "REAL" if comp.is_positive and comp.is_significant else (
                "POSSIBLE" if comp.is_positive else "ILLUSION"
            )
            icon_color = _GREEN if status == "REAL" else (_YELLOW if status == "POSSIBLE" else _RED)
            st.markdown(f"""
            <div style='margin: 8px 0; padding: 10px 15px; border-left: 3px solid {icon_color};
            background: rgba(255,255,255,0.02); border-radius: 0 8px 8px 0;'>
                <div style='font-size: 0.85rem; color: {icon_color}; font-weight: 600;'>
                {label} — {status} ({comp.edge_pct_of_roi:+.1f}%, p={comp.p_value:.4f})
                </div>
                <div style='font-size: 0.75rem; color: #9CA3AF; margin-top: 4px;'>
                {comp.verdict}
                </div>
            </div>
            """, unsafe_allow_html=True)


def _render_empty_state():
    st.info(
        "The edge decomposition system needs at least 10 settled bets to produce analysis.\n\n"
        "**5 Components Analyzed:**\n"
        "1. Predictive — Brier score per market type\n"
        "2. Informational — Weather/injury/wave timing\n"
        "3. Market Inefficiency — CLV per market\n"
        "4. Execution — Line shopping effectiveness\n"
        "5. Structural — Field diversification, wave correlation\n\n"
        "Once enough bets settle, this will show which components carry real weight."
    )
