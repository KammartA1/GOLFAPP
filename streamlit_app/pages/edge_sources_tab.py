"""
streamlit_app/pages/edge_sources_tab.py
========================================
Edge Sources dashboard — 12 independent edge signals ranked, validated,
and checked for pairwise independence.

Shows:
  - All sources ranked by standalone Sharpe / composite score
  - Independence heatmap (pairwise correlation matrix)
  - Per-source validation metrics and mechanism details
  - Source health dashboard (accepted / rejected / flagged)
"""

from __future__ import annotations

import logging

import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

log = logging.getLogger(__name__)


def _get_registry():
    from edge_analysis.source_registry import EdgeSourceRegistry
    return EdgeSourceRegistry()


def _generate_demo_observations(n: int = 200) -> list[dict]:
    """
    Generate synthetic player-tournament observations for validation
    and independence analysis when no live data is available.
    """
    rng = np.random.RandomState(42)
    courses = [
        "Augusta National", "Pebble Beach", "TPC Sawgrass", "Torrey Pines",
        "Bay Hill", "Harbour Town", "Muirfield Village", "Royal Troon",
        "Pinehurst No. 2", "TPC Scottsdale",
    ]
    observations = []
    for i in range(n):
        sg_ott = rng.normal(0.0, 0.8)
        sg_app = rng.normal(0.0, 0.9)
        sg_atg = rng.normal(0.0, 0.7)
        sg_putt = rng.normal(0.0, 1.0)
        sg_total = sg_ott + sg_app + sg_atg + sg_putt
        wind_sg = rng.normal(0.0, 0.3)
        rain_sg = rng.normal(0.0, 0.2)
        cold_sg = rng.normal(0.0, 0.2)
        hot_sg = rng.normal(0.0, 0.15)
        course = courses[rng.randint(len(courses))]
        wind_mph = rng.uniform(3, 25)
        temp_f = rng.uniform(45, 100)
        field_strength = rng.uniform(0.2, 0.95)
        driving_distance = rng.normal(299, 12)

        # Generate recent results for form
        n_recent = rng.randint(4, 15)
        recent_results = []
        for j in range(n_recent):
            recent_results.append({
                "sg_total": sg_total + rng.normal(0, 1.2),
                "finish": max(1, int(rng.exponential(30))),
                "field_strength": rng.uniform(0.2, 0.9),
                "events_ago": j,
            })

        # Generate course history
        n_course = rng.randint(0, 8)
        course_history = []
        for j in range(n_course):
            course_history.append({
                "sg_total": sg_total + rng.normal(0.1, 1.5),
                "finish": max(1, int(rng.exponential(25))),
                "year": 2026 - rng.randint(0, 6),
                "rounds": rng.choice([2, 4, 4, 4]),
                "field_strength": rng.uniform(0.3, 0.9),
            })

        player = {
            "sg_ott": sg_ott,
            "sg_app": sg_app,
            "sg_atg": sg_atg,
            "sg_putt": sg_putt,
            "sg_total": sg_total,
            "baseline_sg": sg_total * 0.9 + rng.normal(0, 0.3),
            "wind_sg": wind_sg,
            "rain_sg": rain_sg,
            "cold_sg": cold_sg,
            "hot_sg": hot_sg,
            "driving_distance": driving_distance,
            "driving_accuracy": rng.uniform(0.50, 0.75),
            "form_factor": rng.uniform(0.85, 1.15),
            "owgr": max(1, int(rng.exponential(80))),
            "wave": rng.choice(["AM", "PM"]),
            "am_sg": rng.normal(0.0, 0.2),
            "pm_sg": rng.normal(0.0, 0.2),
            "recent_results": recent_results,
            "course_history": course_history,
            "major_sg": sg_total + rng.normal(0, 0.5),
            "regular_sg": sg_total,
            "major_starts": rng.randint(0, 30),
            "made_cuts_last_10": rng.randint(4, 10),
            "total_starts_last_10": 10,
            "consecutive_starts": rng.choice([1, 1, 2, 2, 3, 4]),
            "miles_last_30_days": rng.uniform(500, 10000),
            "age": rng.randint(22, 48),
            "fitness_tier": rng.choice(["elite", "good", "average"]),
            "had_week_off": rng.choice([True, False]),
            "timezone_changes": rng.randint(0, 5),
            "avg_field_strength": rng.uniform(0.3, 0.8),
            "trajectory": rng.choice(["high", "mid", "low"]),
            "spin_rate": rng.choice(["high", "medium", "low"]),
        }

        tournament_context = {
            "course": course,
            "wind_mph": wind_mph,
            "rain_prob": rng.uniform(0, 0.6),
            "temperature_f": temp_f,
            "humidity_pct": rng.uniform(20, 95),
            "field_strength": field_strength,
            "event_type": rng.choice(["major", "signature", "regular", "regular", "regular"]),
            "round_number": rng.randint(1, 5),
            "course_yardage": rng.randint(6900, 7600),
            "elevation_ft": rng.choice([0, 50, 330, 480, 900, 1500, 5280]),
            "course_condition": rng.choice(["parkland", "links", "desert", "coastal"]),
            "field_avg_owgr": rng.uniform(40, 150),
        }

        # Outcome: correlated with SG + noise
        finish = max(1, int(70 - sg_total * 8 + rng.normal(0, 15)))

        observations.append({
            "player": player,
            "tournament_context": tournament_context,
            "actual_finish": min(finish, 80),
            "made_cut": finish <= 65,
        })

    return observations


def render():
    """Render the Edge Sources tab."""
    st.markdown("""
    <h2 style='font-family: Inter, sans-serif; color: #00FFB2; margin-bottom: 0;
    font-size: 1.3rem; letter-spacing: 0.05em;'>
    EDGE SOURCES &mdash; SIGNAL ANALYSIS
    </h2>
    <p style='font-family: JetBrains Mono, monospace; color: #6B7B8D; font-size: 0.65rem;
    margin-top: 0.2rem;'>
    12 independent edge signals &bull; validated &bull; ranked &bull; independence-checked
    </p>
    """, unsafe_allow_html=True)

    tab_ranking, tab_heatmap, tab_validation, tab_health = st.tabs([
        "SOURCE RANKINGS", "INDEPENDENCE HEATMAP", "VALIDATION DETAIL", "HEALTH DASHBOARD",
    ])

    with tab_ranking:
        _render_rankings()

    with tab_heatmap:
        _render_heatmap()

    with tab_validation:
        _render_validation()

    with tab_health:
        _render_health()


# ── SOURCE RANKINGS ──────────────────────────────────────────────────────

def _render_rankings():
    """All sources ranked by composite score."""
    registry = _get_registry()
    registry.load_all()

    with st.spinner("Generating observations and validating sources..."):
        obs = _generate_demo_observations(300)
        registry.validate_all(obs)
        registry.compute_independence(obs)
        rankings = registry.rank_sources()

    if not rankings:
        st.info("No edge sources available")
        return

    # Summary metrics
    summary = registry.summary()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("TOTAL SOURCES", summary["total_sources"])
    with col2:
        st.metric("ACCEPTED", summary["accepted"])
    with col3:
        st.metric("REJECTED", summary["rejected"])
    with col4:
        ind_score = summary.get("independence_score")
        if ind_score is not None:
            st.metric("INDEPENDENCE", f"{ind_score:.0%}")
        else:
            st.metric("INDEPENDENCE", "N/A")

    st.markdown("---")
    st.markdown("##### Source Rankings (by Composite Score)")

    rows = []
    for r in rankings:
        status_icon = "\u2705" if r["accepted"] else "\u274c"
        decay = r.get("decay_risk", "")
        if "LOW" in decay:
            decay_label = "LOW"
        elif "MEDIUM" in decay:
            decay_label = "MED"
        else:
            decay_label = "HIGH"

        rows.append({
            "": status_icon,
            "Source": r["name"],
            "Category": r["category"],
            "Composite": f"{r['composite_score']:.4f}",
            "Sharpe": f"{r['sharpe']:+.4f}",
            "P-Value": f"{r['p_value']:.4f}",
            "Samples": r["sample_size"],
            "VIF": f"{r['vif']:.1f}",
            "Spread": f"{r['quintile_spread']:+.4f}",
            "Decay Risk": decay_label,
            "Status": r["status"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Composite score bar chart
    fig = go.Figure()
    names = [r["name"] for r in rankings]
    scores = [r["composite_score"] for r in rankings]
    colors = ["#00FFB2" if r["accepted"] else "#FF3358" for r in rankings]

    fig.add_trace(go.Bar(
        x=names,
        y=scores,
        marker_color=colors,
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
        textfont=dict(size=10, color="#A0AEC0"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="Composite Score",
        height=380,
        margin=dict(l=40, r=20, t=30, b=100),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sharpe ratio comparison
    st.markdown("##### Sharpe Ratio Comparison")
    fig2 = go.Figure()
    sharpes = [r["sharpe"] for r in rankings]
    sharpe_colors = ["#00FFB2" if s > 0 else "#FF3358" for s in sharpes]
    fig2.add_trace(go.Bar(
        x=names,
        y=sharpes,
        marker_color=sharpe_colors,
        text=[f"{s:+.3f}" for s in sharpes],
        textposition="outside",
        textfont=dict(size=10, color="#A0AEC0"),
    ))
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="Sharpe Ratio",
        height=350,
        margin=dict(l=40, r=20, t=30, b=100),
        xaxis_tickangle=-45,
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="#6B7B8D", opacity=0.5)
    st.plotly_chart(fig2, use_container_width=True)


# ── INDEPENDENCE HEATMAP ─────────────────────────────────────────────────

def _render_heatmap():
    """Pairwise correlation heatmap across all edge sources."""
    registry = _get_registry()
    registry.load_all()

    with st.spinner("Computing independence matrix..."):
        obs = _generate_demo_observations(300)
        registry.validate_all(obs)
        indep = registry.compute_independence(obs)

    if indep is None:
        st.info("Independence analysis not available")
        return

    corr_mat = indep["correlation_matrix"]
    names = indep["source_names"]

    # Heatmap
    st.markdown("##### Pairwise Spearman Correlation Matrix")

    fig = go.Figure(data=go.Heatmap(
        z=corr_mat,
        x=names,
        y=names,
        colorscale=[
            [0.0, "#1a237e"],
            [0.25, "#283593"],
            [0.4, "#0D1B2A"],
            [0.5, "#1B2838"],
            [0.6, "#0D1B2A"],
            [0.75, "#b71c1c"],
            [1.0, "#ff1744"],
        ],
        zmin=-1,
        zmax=1,
        text=np.round(corr_mat, 3),
        texttemplate="%{text}",
        textfont=dict(size=9, color="#E0E0E0"),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.4f}<extra></extra>",
        colorbar=dict(
            title="Spearman r",
            titlefont=dict(color="#A0AEC0"),
            tickfont=dict(color="#A0AEC0"),
        ),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=550,
        margin=dict(l=120, r=20, t=30, b=120),
        xaxis_tickangle=-45,
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Independence score
    score = indep.get("independence_score", 0)
    if score >= 0.85:
        color = "#00FFB2"
        label = "EXCELLENT"
    elif score >= 0.70:
        color = "#FFB800"
        label = "ACCEPTABLE"
    else:
        color = "#FF3358"
        label = "POOR"

    st.markdown(f"""
    <div style='text-align: center; padding: 15px; border: 2px solid {color};
    border-radius: 12px; margin: 10px 0;'>
        <div style='font-size: 2rem; font-weight: bold; color: {color};
        font-family: Inter, sans-serif;'>{score:.0%}</div>
        <div style='font-size: 0.75rem; color: {color};
        font-family: JetBrains Mono, monospace;'>INDEPENDENCE SCORE &mdash; {label}</div>
    </div>
    """, unsafe_allow_html=True)

    # Flagged pairs
    flagged = indep.get("flagged_pairs", [])
    if flagged:
        st.markdown("##### Flagged Non-Independent Pairs")
        flag_rows = []
        for f in flagged:
            flag_rows.append({
                "Source A": f["source_a"],
                "Source B": f["source_b"],
                "|r|": f"{f['abs_correlation']:.4f}",
                "Correlation": f"{f['correlation']:+.4f}",
                "P-Value": f"{f['p_value']:.6f}",
                "Reason": f["reason"],
                "Action": f["recommendation"],
            })
        st.dataframe(pd.DataFrame(flag_rows), use_container_width=True, hide_index=True)
    else:
        st.success("No non-independent pairs detected (|r| < 0.40 threshold)")

    # VIF table
    vif_results = indep.get("vif_results", [])
    if vif_results:
        st.markdown("##### Variance Inflation Factors")
        vif_rows = []
        for v in vif_results:
            status_color = {
                "OK": "#00FFB2", "MILD": "#FFB800",
                "MODERATE": "#FF8C00", "SEVERE": "#FF3358",
            }.get(v["status"], "#6B7B8D")
            vif_rows.append({
                "Source": v["source"],
                "VIF": f"{v['vif']:.2f}",
                "R-Squared": f"{v.get('r_squared', 0):.4f}",
                "Status": v["status"],
            })
        st.dataframe(pd.DataFrame(vif_rows), use_container_width=True, hide_index=True)


# ── VALIDATION DETAIL ────────────────────────────────────────────────────

def _render_validation():
    """Per-source validation metrics and mechanism explanations."""
    registry = _get_registry()
    registry.load_all()

    with st.spinner("Validating all sources..."):
        obs = _generate_demo_observations(300)
        val_results = registry.validate_all(obs)
        registry.compute_independence(obs)
        rankings = registry.rank_sources()

    sources = registry.get_sources()
    source_names = [getattr(s, "name", "?") for s in sources]

    selected = st.selectbox("Select Source", source_names)
    if not selected:
        return

    src = registry.get_source_by_name(selected)
    detail = registry.get_source_detail(selected)

    if not src or not detail:
        st.info("Source not found")
        return

    # Header
    accepted = detail.get("accepted", False)
    status_badge = (
        '<span style="color: #00FFB2; font-weight: bold;">ACCEPTED</span>'
        if accepted else
        '<span style="color: #FF3358; font-weight: bold;">REJECTED</span>'
    )
    st.markdown(f"""
    <div style='padding: 12px; border: 1px solid {"#00FFB2" if accepted else "#FF3358"};
    border-radius: 8px; margin-bottom: 15px;'>
        <span style='font-size: 1.1rem; font-weight: 600; color: #E0E0E0;
        font-family: Inter, sans-serif;'>{selected}</span>
        &nbsp;&nbsp;{status_badge}
        &nbsp;&nbsp;<span style='color: #6B7B8D; font-size: 0.75rem;
        font-family: JetBrains Mono, monospace;'>
        category: {detail.get("category", "?")} &bull; v{getattr(src, "version", "1.0")}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("SHARPE", f"{detail['sharpe']:+.4f}")
    with col2:
        p = detail["p_value"]
        st.metric("P-VALUE", f"{p:.4f}", delta="Sig" if p < 0.05 else "Not sig")
    with col3:
        st.metric("SAMPLES", detail["sample_size"])
    with col4:
        st.metric("VIF", f"{detail['vif']:.1f}")
    with col5:
        st.metric("Q-SPREAD", f"{detail['quintile_spread']:+.4f}")

    # Rejection reasons
    if detail.get("rejection_reasons"):
        st.warning("**Rejection reasons:** " + " | ".join(detail["rejection_reasons"]))

    st.markdown("---")

    # Mechanism
    st.markdown("##### Edge Mechanism")
    st.markdown(f"""
    <div style='padding: 12px; background: rgba(0,255,178,0.04);
    border-left: 3px solid #00FFB2; border-radius: 4px;
    font-family: JetBrains Mono, monospace; font-size: 0.72rem;
    color: #A0AEC0; line-height: 1.6;'>
    {detail.get("mechanism", "Not available")}
    </div>
    """, unsafe_allow_html=True)

    # Decay risk
    st.markdown("##### Decay Risk Assessment")
    decay = detail.get("decay_risk", "")
    if "LOW" in decay:
        decay_color = "#00FFB2"
    elif "MEDIUM" in decay:
        decay_color = "#FFB800"
    else:
        decay_color = "#FF3358"

    st.markdown(f"""
    <div style='padding: 12px; background: rgba(0,0,0,0.2);
    border-left: 3px solid {decay_color}; border-radius: 4px;
    font-family: JetBrains Mono, monospace; font-size: 0.72rem;
    color: #A0AEC0; line-height: 1.6;'>
    {decay}
    </div>
    """, unsafe_allow_html=True)

    # Signal distribution
    st.markdown("##### Signal Distribution (demo data)")
    signals = []
    for o in obs[:200]:
        try:
            sig = src.get_signal(o["player"], o["tournament_context"])
            signals.append(sig)
        except Exception:
            pass

    if signals:
        sig_arr = np.array(signals)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=sig_arr,
            nbinsx=40,
            marker_color="#00FFB2",
            opacity=0.7,
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="#FF3358", opacity=0.6)
        fig.add_vline(x=float(np.mean(sig_arr)), line_dash="dot",
                      line_color="#FFB800", opacity=0.8,
                      annotation_text=f"mean={np.mean(sig_arr):.3f}")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Signal Value",
            yaxis_title="Count",
            height=300,
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Stats
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Mean", f"{np.mean(sig_arr):.4f}")
        with col_b:
            st.metric("Std Dev", f"{np.std(sig_arr):.4f}")
        with col_c:
            st.metric("Min", f"{np.min(sig_arr):.4f}")
        with col_d:
            st.metric("Max", f"{np.max(sig_arr):.4f}")


# ── HEALTH DASHBOARD ─────────────────────────────────────────────────────

def _render_health():
    """Source health dashboard — accepted, rejected, flagged pairs."""
    registry = _get_registry()
    registry.load_all()

    with st.spinner("Running full health check..."):
        obs = _generate_demo_observations(300)
        registry.validate_all(obs)
        registry.compute_independence(obs)
        rankings = registry.rank_sources()

    accepted = registry.get_accepted_sources()
    rejected = registry.get_rejected_sources()
    indep = registry.get_independence_report()

    # Overall system health
    n_total = len(rankings)
    n_accepted = len(accepted)
    n_flagged = len(indep.get("flagged_pairs", [])) if indep else 0
    ind_score = indep.get("independence_score", 0) if indep else 0

    # Health score: weighted combination
    acceptance_rate = n_accepted / max(n_total, 1)
    health = (acceptance_rate * 0.4 + ind_score * 0.3 + (1.0 - n_flagged / max(n_total, 1)) * 0.3)
    health = max(0.0, min(1.0, health))

    if health >= 0.80:
        health_color = "#00FFB2"
        health_label = "HEALTHY"
    elif health >= 0.60:
        health_color = "#FFB800"
        health_label = "DEGRADED"
    else:
        health_color = "#FF3358"
        health_label = "CRITICAL"

    st.markdown(f"""
    <div style='text-align: center; padding: 20px; border: 2px solid {health_color};
    border-radius: 12px; margin: 10px 0;'>
        <div style='font-size: 2.5rem; font-weight: bold; color: {health_color};
        font-family: Inter, sans-serif;'>{health:.0%}</div>
        <div style='font-size: 0.8rem; color: {health_color};
        font-family: JetBrains Mono, monospace;'>SYSTEM HEALTH &mdash; {health_label}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Acceptance Rate", f"{acceptance_rate:.0%}")
    with col2:
        st.metric("Independence", f"{ind_score:.0%}")
    with col3:
        st.metric("Flagged Pairs", n_flagged)
    with col4:
        avg_sharpe = float(np.mean([r["sharpe"] for r in rankings])) if rankings else 0
        st.metric("Avg Sharpe", f"{avg_sharpe:+.3f}")

    st.markdown("---")

    # Accepted sources
    st.markdown("##### Accepted Sources")
    if accepted:
        for r in accepted:
            with st.expander(f"{r['name']} (composite: {r['composite_score']:.4f})"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Sharpe", f"{r['sharpe']:+.4f}")
                with c2:
                    st.metric("P-Value", f"{r['p_value']:.4f}")
                with c3:
                    st.metric("VIF", f"{r['vif']:.1f}")
                st.caption(f"**Category:** {r['category']} | **Samples:** {r['sample_size']}")
    else:
        st.warning("No sources passed validation gates")

    # Rejected sources
    st.markdown("##### Rejected Sources")
    if rejected:
        for r in rejected:
            with st.expander(f"{r['name']} -- REJECTED"):
                st.error("**Rejection reasons:** " + " | ".join(r.get("rejection_reasons", ["Unknown"])))
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Sharpe", f"{r['sharpe']:+.4f}")
                with c2:
                    st.metric("P-Value", f"{r['p_value']:.4f}")
                with c3:
                    st.metric("VIF", f"{r['vif']:.1f}")
    else:
        st.success("All sources passed validation")

    # Category breakdown
    st.markdown("##### Sources by Category")
    categories = {}
    for r in rankings:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"count": 0, "accepted": 0, "avg_sharpe": []}
        categories[cat]["count"] += 1
        if r["accepted"]:
            categories[cat]["accepted"] += 1
        categories[cat]["avg_sharpe"].append(r["sharpe"])

    cat_rows = []
    for cat, info in sorted(categories.items()):
        cat_rows.append({
            "Category": cat.title(),
            "Sources": info["count"],
            "Accepted": info["accepted"],
            "Avg Sharpe": f"{np.mean(info['avg_sharpe']):+.4f}",
        })
    st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

    # Decay risk summary
    st.markdown("##### Decay Risk Distribution")
    decay_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for r in rankings:
        decay = r.get("decay_risk", "")
        if "LOW" in decay:
            decay_counts["LOW"] += 1
        elif "MEDIUM" in decay:
            decay_counts["MEDIUM"] += 1
        else:
            decay_counts["HIGH"] += 1

    fig = go.Figure(data=[go.Pie(
        labels=list(decay_counts.keys()),
        values=list(decay_counts.values()),
        marker=dict(colors=["#00FFB2", "#FFB800", "#FF3358"]),
        hole=0.4,
        textinfo="label+value",
        textfont=dict(size=12, color="#E0E0E0"),
    )])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
