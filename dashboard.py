# ============================================================
# GOLF QUANT ENGINE v2.0 — Professional-Grade Golf Analytics
# Research-validated SG model · Monte Carlo simulation
# Bayesian shrinkage · Course fit · Weather · Kelly sizing
# Claude AI integration · PrizePicks edge analyzer
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from scipy.special import expit as sigmoid
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json
import math
import warnings
import hashlib
import traceback

warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Golf Quant Engine v2.0",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PREMIUM CSS — Apple-style dark theme
# ============================================================
st.markdown("""
<style>
/* ── Font imports ───────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

/* ── Global resets ──────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.stApp {
    background: #0a0f1a;
    color: #e2e8f0;
}

/* ── Hide Streamlit chrome ──────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1220 0%, #0a0f1a 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 0.875rem;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
}

/* ── Dashboard header ───────────────────────────────────── */
.dash-header {
    background: linear-gradient(135deg, #0c1425 0%, #111b2e 50%, #0c1425 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 18px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}
.dash-header-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.025em;
    line-height: 1.2;
}
.dash-header-sub {
    font-size: 0.8rem;
    color: #64748b;
    font-weight: 400;
    letter-spacing: 0.02em;
    margin-top: 2px;
}
.dash-header-badge {
    display: inline-block;
    background: linear-gradient(135deg, #34d399 0%, #22d3ee 100%);
    color: #0a0f1a;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-left: 10px;
}

/* ── Frosted glass cards ────────────────────────────────── */
.glass-card {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(255,255,255,0.12);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* ── Metric cards ───────────────────────────────────────── */
.metric-card {
    background: rgba(15, 23, 42, 0.5);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 20px 22px;
    text-align: left;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.metric-card.green::before { background: linear-gradient(90deg, #34d399, #22d3ee); }
.metric-card.blue::before { background: linear-gradient(90deg, #60a5fa, #818cf8); }
.metric-card.amber::before { background: linear-gradient(90deg, #fbbf24, #f97316); }
.metric-card.red::before { background: linear-gradient(90deg, #f87171, #fb923c); }
.metric-card.purple::before { background: linear-gradient(90deg, #a78bfa, #c084fc); }

.metric-card:hover {
    border-color: rgba(255,255,255,0.12);
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.25);
}
.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    line-height: 1.1;
    font-family: 'JetBrains Mono', monospace;
}
.metric-delta {
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 6px;
    display: flex;
    align-items: center;
    gap: 4px;
}
.metric-delta.positive { color: #34d399; }
.metric-delta.negative { color: #f87171; }
.metric-delta.neutral { color: #64748b; }

/* ── Section headers ────────────────────────────────────── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 32px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: -0.01em;
}
.section-header .sh-icon {
    font-size: 1.1rem;
}
.section-header .sh-badge {
    font-size: 0.6rem;
    font-weight: 700;
    background: rgba(96, 165, 250, 0.15);
    color: #60a5fa;
    padding: 2px 8px;
    border-radius: 10px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Tabs ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.4);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(255,255,255,0.04);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 8px 20px;
    border: none;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #94a3b8;
    background: rgba(255,255,255,0.04);
}
.stTabs [aria-selected="true"] {
    background: rgba(255,255,255,0.08) !important;
    color: #f1f5f9 !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ── Buttons ────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, rgba(52,211,153,0.15) 0%, rgba(96,165,250,0.15) 100%);
    border: 1px solid rgba(52,211,153,0.3);
    color: #34d399;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 8px 24px;
    transition: all 0.3s ease;
    letter-spacing: 0.01em;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(52,211,153,0.25) 0%, rgba(96,165,250,0.25) 100%);
    border-color: rgba(52,211,153,0.5);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(52,211,153,0.15);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── Inputs & selects ───────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: rgba(96,165,250,0.4) !important;
    box-shadow: 0 0 0 3px rgba(96,165,250,0.1) !important;
}
.stTextInput label, .stNumberInput label,
.stSelectbox label, .stMultiSelect label,
.stSlider label {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
}

/* ── Sliders ────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #34d399 !important;
    border: 2px solid #0a0f1a !important;
    width: 18px !important;
    height: 18px !important;
}
.stSlider [data-baseweb="slider"] div[data-testid="stTickBarMin"],
.stSlider [data-baseweb="slider"] div[data-testid="stTickBarMax"] {
    color: #64748b !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Dataframes ─────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stDataFrame"] table {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
}
[data-testid="stDataFrame"] th {
    background: rgba(15, 23, 42, 0.8) !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stDataFrame"] td {
    color: #e2e8f0 !important;
    background: rgba(15, 23, 42, 0.4) !important;
    border-bottom: 1px solid rgba(255,255,255,0.03) !important;
}

/* ── Expanders ──────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
}

/* ── Confidence badge ───────────────────────────────────── */
.conf-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 20px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.conf-badge.high {
    background: rgba(52,211,153,0.15);
    color: #34d399;
    border: 1px solid rgba(52,211,153,0.25);
}
.conf-badge.med {
    background: rgba(251,191,36,0.15);
    color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.25);
}
.conf-badge.low {
    background: rgba(248,113,113,0.15);
    color: #f87171;
    border: 1px solid rgba(248,113,113,0.25);
}

/* ── Probability bar ────────────────────────────────────── */
.prob-bar-wrap {
    background: rgba(255,255,255,0.04);
    border-radius: 6px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    position: relative;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.5s ease;
}
.prob-bar-fill.green { background: linear-gradient(90deg, #34d399, #22d3ee); }
.prob-bar-fill.blue { background: linear-gradient(90deg, #60a5fa, #818cf8); }
.prob-bar-fill.amber { background: linear-gradient(90deg, #fbbf24, #f97316); }
.prob-bar-fill.red { background: linear-gradient(90deg, #f87171, #fb923c); }

/* ── Edge badge ─────────────────────────────────────────── */
.edge-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 12px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
}
.edge-badge.positive {
    background: rgba(52,211,153,0.12);
    color: #34d399;
    border: 1px solid rgba(52,211,153,0.2);
}
.edge-badge.negative {
    background: rgba(248,113,113,0.12);
    color: #f87171;
    border: 1px solid rgba(248,113,113,0.2);
}
.edge-badge.neutral {
    background: rgba(100,116,139,0.12);
    color: #94a3b8;
    border: 1px solid rgba(100,116,139,0.2);
}

/* ── Trend badge ────────────────────────────────────────── */
.trend-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'Inter', sans-serif;
}
.trend-badge.up {
    background: rgba(52,211,153,0.1);
    color: #34d399;
}
.trend-badge.down {
    background: rgba(248,113,113,0.1);
    color: #f87171;
}
.trend-badge.flat {
    background: rgba(100,116,139,0.1);
    color: #94a3b8;
}

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

/* ── Misc ───────────────────────────────────────────────── */
.stMarkdown a { color: #60a5fa; text-decoration: none; }
.stMarkdown a:hover { color: #93bbfd; text-decoration: underline; }
hr { border-color: rgba(255,255,255,0.06) !important; }

.mono { font-family: 'JetBrains Mono', monospace; }
.text-muted { color: #64748b; }
.text-green { color: #34d399; }
.text-blue { color: #60a5fa; }
.text-amber { color: #fbbf24; }
.text-red { color: #f87171; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS — Reusable UI components
# ============================================================

def metric_card(label: str, value: str, delta: str = "", delta_dir: str = "neutral",
                color: str = "green") -> str:
    """Return HTML for a styled metric card.

    Args:
        label: Upper label text
        value: Main display value
        delta: Small delta text below value
        delta_dir: 'positive' | 'negative' | 'neutral'
        color: 'green' | 'blue' | 'amber' | 'red' | 'purple'
    """
    delta_html = ""
    if delta:
        arrow = ""
        if delta_dir == "positive":
            arrow = "&#9650; "
        elif delta_dir == "negative":
            arrow = "&#9660; "
        delta_html = f'<div class="metric-delta {delta_dir}">{arrow}{delta}</div>'

    return f"""
    <div class="metric-card {color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def section_header(title: str, icon: str = "", badge: str = "") -> str:
    """Return HTML for a section header with optional icon and badge."""
    icon_html = f'<span class="sh-icon">{icon}</span>' if icon else ""
    badge_html = f'<span class="sh-badge">{badge}</span>' if badge else ""
    return f'<div class="section-header">{icon_html}{title}{badge_html}</div>'


def conf_badge(level: str) -> str:
    """Return HTML for a confidence badge: 'high', 'med', or 'low'."""
    labels = {"high": "HIGH CONF", "med": "MED CONF", "low": "LOW CONF"}
    label = labels.get(level, level.upper())
    css_class = level if level in ("high", "med", "low") else "med"
    return f'<span class="conf-badge {css_class}">{label}</span>'


def prob_bar_html(probability: float, color: str = "green", label: str = "") -> str:
    """Return HTML for a horizontal probability bar.

    Args:
        probability: Value between 0 and 1.
        color: 'green' | 'blue' | 'amber' | 'red'
        label: Optional text label to the left.
    """
    pct = max(0.0, min(1.0, probability)) * 100
    label_html = f'<span style="font-size:0.75rem;color:#94a3b8;margin-right:8px;">{label}</span>' if label else ""
    pct_text = f'<span class="mono" style="font-size:0.75rem;color:#e2e8f0;margin-left:8px;">{pct:.1f}%</span>'
    return f"""
    <div style="display:flex;align-items:center;margin:4px 0;">
        {label_html}
        <div class="prob-bar-wrap" style="flex:1;">
            <div class="prob-bar-fill {color}" style="width:{pct}%;"></div>
        </div>
        {pct_text}
    </div>
    """


def edge_badge(edge: float) -> str:
    """Return HTML for an edge badge showing +/- percentage edge.

    Args:
        edge: Edge value as a decimal (e.g., 0.05 for 5% edge).
    """
    pct = edge * 100
    if pct > 1.0:
        css_class = "positive"
        sign = "+"
    elif pct < -1.0:
        css_class = "negative"
        sign = ""
    else:
        css_class = "neutral"
        sign = "+" if pct >= 0 else ""
    return f'<span class="edge-badge {css_class}">{sign}{pct:.1f}%</span>'


def trend_badge(slope: float, label: str = "") -> str:
    """Return HTML for a trend direction badge.

    Args:
        slope: Positive = trending up, negative = trending down.
        label: Optional override label text.
    """
    if slope > 0.02:
        css_class = "up"
        arrow = "&#9650;"
        default_label = "Trending Up"
    elif slope < -0.02:
        css_class = "down"
        arrow = "&#9660;"
        default_label = "Trending Down"
    else:
        css_class = "flat"
        arrow = "&#9654;"
        default_label = "Stable"
    text = label if label else default_label
    return f'<span class="trend-badge {css_class}">{arrow} {text}</span>'


# ============================================================
# QUANT ENGINE CONSTANTS
# ============================================================

# ── Strokes Gained category weights (research-validated) ───
# Sum to 1.0 — relative importance for total SG prediction
SG_WEIGHTS = {
    "sg_ott":   0.18,   # Off the Tee
    "sg_app":   0.30,   # Approach
    "sg_arg":   0.17,   # Around the Green
    "sg_putt":  0.35,   # Putting
}

# ── Form windows with exponential recency weighting ────────
# L = last N events; weight = how much to weight that window
FORM_WINDOWS = {
    "L4":  {"events": 4,  "weight": 0.40},
    "L8":  {"events": 8,  "weight": 0.30},
    "L12": {"events": 12, "weight": 0.20},
    "L24": {"events": 24, "weight": 0.10},
}

# ── Regression-to-mean factors per SG category ─────────────
# Lower = more regression (noisier stat). Based on Broadie research.
# Putting regresses heavily (high variance), approach is more stable.
REGRESSION_FACTORS = {
    "sg_putt":  0.55,   # Putting: high noise, regress ~45%
    "sg_app":   0.20,   # Approach: most stable, regress ~20%
    "sg_arg":   0.35,   # Around green: moderate noise
    "sg_ott":   0.25,   # Off the tee: fairly stable
    "sg_total": 0.22,   # Total: composite stability
}

# ── Positional priors for Bayesian shrinkage ────────────────
# Tour average (prior mean) and prior precision (strength of prior)
POSITIONAL_PRIORS = {
    "sg_ott":   {"mu": 0.00, "sigma": 0.55, "n_prior": 12},
    "sg_app":   {"mu": 0.00, "sigma": 0.70, "n_prior": 10},
    "sg_arg":   {"mu": 0.00, "sigma": 0.45, "n_prior": 15},
    "sg_putt":  {"mu": 0.00, "sigma": 0.65, "n_prior": 20},
    "sg_total": {"mu": 0.00, "sigma": 1.20, "n_prior": 8},
}

# ── Tour baseline stats for PrizePicks conversion ──────────
TOUR_BASELINES = {
    "scoring_avg":       70.5,
    "birdies_per_round": 3.5,
    "bogeys_per_round":  2.8,
    "gir_pct":           65.0,
    "fairways_pct":      60.0,
    "putts_per_round":   29.0,
    "scrambling_pct":    58.0,
    "dd_avg":            295.0,
    "fantasy_score":     45.0,
    "bogey_free_pct":    22.0,
}

# ── PrizePicks payout structures ────────────────────────────
PP_PAYOUTS = {
    "power_play": {
        2: {"all_correct": 3.0},
        3: {"all_correct": 5.0},
        4: {"all_correct": 10.0},
        5: {"all_correct": 20.0},
        6: {"all_correct": 40.0},
    },
    "flex_play": {
        3: {"3_correct": 2.25, "2_correct": 1.25},
        4: {"4_correct": 5.0,  "3_correct": 1.5},
        5: {"5_correct": 10.0, "4_correct": 2.0, "3_correct": 0.4},
        6: {"6_correct": 25.0, "5_correct": 2.0, "4_correct": 0.4},
    },
}


# ============================================================
# COURSE PROFILES DATABASE
# ============================================================
COURSE_PROFILES = {
    "Augusta National": {
        "sg_weights": {"sg_ott": 0.20, "sg_app": 0.35, "sg_arg": 0.20, "sg_putt": 0.25},
        "distance_bonus": 0.08, "accuracy_penalty": -0.02,
        "bermuda_greens": False, "wind_sensitivity": 0.3,
        "elevation": 350, "par": 72,
        "notes": "Premium approach course, slope/undulation on greens favor elite iron players",
    },
    "TPC Sawgrass": {
        "sg_weights": {"sg_ott": 0.15, "sg_app": 0.30, "sg_arg": 0.20, "sg_putt": 0.35},
        "distance_bonus": 0.00, "accuracy_penalty": -0.06,
        "bermuda_greens": True, "wind_sensitivity": 0.5,
        "elevation": 15, "par": 72,
        "notes": "Accuracy-premium layout, water on 6 holes, bermuda greens",
    },
    "Pebble Beach": {
        "sg_weights": {"sg_ott": 0.15, "sg_app": 0.35, "sg_arg": 0.20, "sg_putt": 0.30},
        "distance_bonus": -0.02, "accuracy_penalty": -0.04,
        "bermuda_greens": False, "wind_sensitivity": 0.7,
        "elevation": 80, "par": 72,
        "notes": "Coastal links-style, wind dominant factor, small greens reward precision",
    },
    "Torrey Pines South": {
        "sg_weights": {"sg_ott": 0.22, "sg_app": 0.30, "sg_arg": 0.15, "sg_putt": 0.33},
        "distance_bonus": 0.06, "accuracy_penalty": -0.03,
        "bermuda_greens": False, "wind_sensitivity": 0.4,
        "elevation": 120, "par": 72,
        "notes": "Long US Open venue, kikuyu rough punishes misses, coastal wind",
    },
    "TPC Scottsdale": {
        "sg_weights": {"sg_ott": 0.17, "sg_app": 0.28, "sg_arg": 0.18, "sg_putt": 0.37},
        "distance_bonus": 0.04, "accuracy_penalty": -0.01,
        "bermuda_greens": True, "wind_sensitivity": 0.2,
        "elevation": 1500, "par": 71,
        "notes": "Birdie-fest, overseeded bermuda, altitude adds ~5% distance, putting-premium",
    },
    "Bay Hill": {
        "sg_weights": {"sg_ott": 0.20, "sg_app": 0.28, "sg_arg": 0.17, "sg_putt": 0.35},
        "distance_bonus": 0.05, "accuracy_penalty": -0.04,
        "bermuda_greens": True, "wind_sensitivity": 0.5,
        "elevation": 100, "par": 72,
        "notes": "Arnold Palmer venue, heavy wind exposure, bermuda greens, ball-strikers course",
    },
    "Riviera CC": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.33, "sg_arg": 0.17, "sg_putt": 0.32},
        "distance_bonus": 0.02, "accuracy_penalty": -0.03,
        "bermuda_greens": False, "wind_sensitivity": 0.3,
        "elevation": 200, "par": 71,
        "notes": "Classic Hogan course, kikuyu lies test approach versatility, small greens",
    },
    "Harbour Town": {
        "sg_weights": {"sg_ott": 0.12, "sg_app": 0.30, "sg_arg": 0.22, "sg_putt": 0.36},
        "distance_bonus": -0.05, "accuracy_penalty": -0.07,
        "bermuda_greens": True, "wind_sensitivity": 0.6,
        "elevation": 5, "par": 71,
        "notes": "Short, tight, windy. Accuracy over distance, small greens, Pete Dye design",
    },
    "Muirfield Village": {
        "sg_weights": {"sg_ott": 0.20, "sg_app": 0.30, "sg_arg": 0.18, "sg_putt": 0.32},
        "distance_bonus": 0.04, "accuracy_penalty": -0.03,
        "bermuda_greens": False, "wind_sensitivity": 0.3,
        "elevation": 900, "par": 72,
        "notes": "Jack's place, well-rounded test, bentgrass greens, strategic bunkering",
    },
    "TPC River Highlands": {
        "sg_weights": {"sg_ott": 0.14, "sg_app": 0.28, "sg_arg": 0.20, "sg_putt": 0.38},
        "distance_bonus": -0.02, "accuracy_penalty": -0.02,
        "bermuda_greens": False, "wind_sensitivity": 0.3,
        "elevation": 50, "par": 70,
        "notes": "Short birdie-fest, putting is king, wedge approaches dominate",
    },
    "Quail Hollow": {
        "sg_weights": {"sg_ott": 0.22, "sg_app": 0.28, "sg_arg": 0.17, "sg_putt": 0.33},
        "distance_bonus": 0.06, "accuracy_penalty": -0.04,
        "bermuda_greens": True, "wind_sensitivity": 0.3,
        "elevation": 700, "par": 72,
        "notes": "Power course, long par 4s/5s, bermuda greens, Green Mile finish",
    },
    "East Lake": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.30, "sg_arg": 0.18, "sg_putt": 0.34},
        "distance_bonus": 0.03, "accuracy_penalty": -0.03,
        "bermuda_greens": True, "wind_sensitivity": 0.3,
        "elevation": 1000, "par": 72,
        "notes": "Tour Championship venue, bermuda overseeded, balanced test",
    },
    "Kapalua Plantation": {
        "sg_weights": {"sg_ott": 0.25, "sg_app": 0.25, "sg_arg": 0.15, "sg_putt": 0.35},
        "distance_bonus": 0.10, "accuracy_penalty": 0.00,
        "bermuda_greens": True, "wind_sensitivity": 0.7,
        "elevation": 500, "par": 73,
        "notes": "Wide open bomber's paradise, extreme elevation and wind, large greens",
    },
    "Colonial CC": {
        "sg_weights": {"sg_ott": 0.13, "sg_app": 0.32, "sg_arg": 0.20, "sg_putt": 0.35},
        "distance_bonus": -0.04, "accuracy_penalty": -0.06,
        "bermuda_greens": True, "wind_sensitivity": 0.4,
        "elevation": 650, "par": 70,
        "notes": "Hogan's Alley — precision, small bermuda greens, tight fairways",
    },
    "TPC Twin Cities": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.28, "sg_arg": 0.18, "sg_putt": 0.36},
        "distance_bonus": 0.03, "accuracy_penalty": -0.02,
        "bermuda_greens": False, "wind_sensitivity": 0.3,
        "elevation": 850, "par": 71,
        "notes": "3M Open venue, bentgrass, balanced but birdie-friendly",
    },
    "St Andrews Old Course": {
        "sg_weights": {"sg_ott": 0.15, "sg_app": 0.25, "sg_arg": 0.25, "sg_putt": 0.35},
        "distance_bonus": 0.02, "accuracy_penalty": -0.02,
        "bermuda_greens": False, "wind_sensitivity": 0.9,
        "elevation": 20, "par": 72,
        "notes": "Home of Golf, wind dominates, massive double greens, deep pot bunkers",
    },
    "Royal Liverpool": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.28, "sg_arg": 0.22, "sg_putt": 0.32},
        "distance_bonus": 0.02, "accuracy_penalty": -0.05,
        "bermuda_greens": False, "wind_sensitivity": 0.85,
        "elevation": 15, "par": 72,
        "notes": "Open Championship links, wind and pot bunkers, firm conditions",
    },
    "Valhalla": {
        "sg_weights": {"sg_ott": 0.22, "sg_app": 0.28, "sg_arg": 0.17, "sg_putt": 0.33},
        "distance_bonus": 0.07, "accuracy_penalty": -0.03,
        "bermuda_greens": False, "wind_sensitivity": 0.25,
        "elevation": 500, "par": 72,
        "notes": "PGA Championship venue, long and soft, favors power players",
    },
    "Pinehurst No. 2": {
        "sg_weights": {"sg_ott": 0.17, "sg_app": 0.28, "sg_arg": 0.25, "sg_putt": 0.30},
        "distance_bonus": 0.01, "accuracy_penalty": -0.04,
        "bermuda_greens": True, "wind_sensitivity": 0.35,
        "elevation": 500, "par": 72,
        "notes": "Donald Ross masterpiece, crowned greens make ARG critical, wiregrass",
    },
    "Oakmont CC": {
        "sg_weights": {"sg_ott": 0.20, "sg_app": 0.32, "sg_arg": 0.18, "sg_putt": 0.30},
        "distance_bonus": 0.04, "accuracy_penalty": -0.06,
        "bermuda_greens": False, "wind_sensitivity": 0.3,
        "elevation": 1100, "par": 70,
        "notes": "Hardest US Open setup, church pew bunkers, lightning-fast greens",
    },
    "Bethpage Black": {
        "sg_weights": {"sg_ott": 0.23, "sg_app": 0.30, "sg_arg": 0.15, "sg_putt": 0.32},
        "distance_bonus": 0.07, "accuracy_penalty": -0.05,
        "bermuda_greens": False, "wind_sensitivity": 0.35,
        "elevation": 200, "par": 70,
        "notes": "Brute-force US Open venue, extreme length, thick rough, pure ball-striker test",
    },
    "Southern Hills": {
        "sg_weights": {"sg_ott": 0.19, "sg_app": 0.30, "sg_arg": 0.18, "sg_putt": 0.33},
        "distance_bonus": 0.04, "accuracy_penalty": -0.04,
        "bermuda_greens": True, "wind_sensitivity": 0.45,
        "elevation": 700, "par": 70,
        "notes": "PGA Championship venue, bermuda, Oklahoma wind, premium ball-striking",
    },
    "Shinnecock Hills": {
        "sg_weights": {"sg_ott": 0.17, "sg_app": 0.30, "sg_arg": 0.22, "sg_putt": 0.31},
        "distance_bonus": 0.02, "accuracy_penalty": -0.05,
        "bermuda_greens": False, "wind_sensitivity": 0.75,
        "elevation": 50, "par": 70,
        "notes": "Links-influenced US Open venue, coastal wind, fescue, firm conditions",
    },
}


# ============================================================
# CORE QUANT FUNCTIONS
# ============================================================

def bayesian_shrink(observed: float, n_obs: int, category: str = "sg_total") -> float:
    """Efron-Morris style Bayesian shrinkage estimator.

    Shrinks the observed SG average toward the population mean (0) based on
    how many observations we have vs. the prior precision.

    Formula:
        shrunk = prior_mu + (n_obs / (n_obs + n_prior)) * (observed - prior_mu)

    With small n_obs, result stays close to prior (0).
    With large n_obs, result converges to observed.

    Args:
        observed: Raw observed SG average.
        n_obs: Number of rounds/events observed.
        category: SG category key for prior lookup.

    Returns:
        Bayesian-shrunk SG estimate.
    """
    prior = POSITIONAL_PRIORS.get(category, POSITIONAL_PRIORS["sg_total"])
    mu_prior = prior["mu"]
    n_prior = prior["n_prior"]

    # Shrinkage weight: how much to trust the data vs prior
    data_weight = n_obs / (n_obs + n_prior)
    shrunk = mu_prior + data_weight * (observed - mu_prior)
    return round(shrunk, 4)


def recency_weighted_sg(history_df: pd.DataFrame) -> dict:
    """Compute recency-weighted SG averages across L4/L8/L12/L24 windows.

    Expects history_df with columns: sg_ott, sg_app, sg_arg, sg_putt, sg_total
    sorted by date descending (most recent first).

    Returns dict with weighted average for each SG category.
    """
    sg_cols = ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_total"]

    # Ensure we have the columns we need; fill missing with 0
    for col in sg_cols:
        if col not in history_df.columns:
            history_df[col] = 0.0

    result = {col: 0.0 for col in sg_cols}
    total_weight = 0.0

    for window_name, spec in FORM_WINDOWS.items():
        n = spec["events"]
        w = spec["weight"]
        window_data = history_df.head(n)

        if len(window_data) == 0:
            continue

        # Within the window, apply exponential decay by position
        n_rows = len(window_data)
        decay_weights = np.array([0.9 ** i for i in range(n_rows)])
        decay_weights = decay_weights / decay_weights.sum()

        for col in sg_cols:
            values = window_data[col].fillna(0.0).values[:n_rows]
            weighted_avg = np.dot(values, decay_weights)
            result[col] += w * weighted_avg

        total_weight += w

    # Normalize by total weight used (handles case where not all windows have data)
    if total_weight > 0:
        for col in sg_cols:
            result[col] = round(result[col] / total_weight, 4)

    return result


def regress_to_mean(raw_sg: float, n_events: int, category: str = "sg_total") -> float:
    """Regress a raw SG value toward the mean based on sample size.

    Uses category-specific regression factors. More events = less regression.
    The regression factor controls the rate of convergence.

    Formula:
        reliability = n_events / (n_events + k)
        where k = (1 / factor - 1) * base_n

    Args:
        raw_sg: Raw SG average.
        n_events: Number of events observed.
        category: SG category for regression factor lookup.

    Returns:
        Regressed SG estimate closer to 0 for small samples.
    """
    factor = REGRESSION_FACTORS.get(category, 0.25)
    # k is the "half-life" in events: how many events to reach 50% reliability
    base_n = 20  # At 20 events, reliability equals the factor
    k = base_n * (1.0 / factor - 1.0)
    k = max(k, 1.0)

    reliability = n_events / (n_events + k)
    regressed = reliability * raw_sg
    return round(regressed, 4)


def apply_course_fit(sg_dict: dict, course_name: str) -> dict:
    """Reweight a player's SG profile by course-specific demands.

    Applies the course's custom SG weights and distance/accuracy modifiers.

    Args:
        sg_dict: Dict with keys sg_ott, sg_app, sg_arg, sg_putt (and optionally sg_total).
        course_name: Key into COURSE_PROFILES.

    Returns:
        Dict with 'course_adj_total', per-category contributions, and fit metadata.
    """
    profile = COURSE_PROFILES.get(course_name)
    if profile is None:
        # Fall back to default weights
        total = sum(sg_dict.get(k, 0.0) * v for k, v in SG_WEIGHTS.items())
        return {
            "course_adj_total": round(total, 4),
            "course_found": False,
            "contributions": {k: round(sg_dict.get(k, 0.0) * v, 4) for k, v in SG_WEIGHTS.items()},
        }

    cw = profile["sg_weights"]
    contributions = {}
    total = 0.0

    for cat in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]:
        val = sg_dict.get(cat, 0.0)
        weight = cw.get(cat, SG_WEIGHTS.get(cat, 0.25))
        # Normalize weights to sum to 1
        contrib = val * weight
        contributions[cat] = round(contrib, 4)
        total += contrib

    # Apply distance bonus (positive SG OTT distance advantage)
    dist_bonus = profile.get("distance_bonus", 0.0)
    ott_val = sg_dict.get("sg_ott", 0.0)
    dist_adj = dist_bonus * max(ott_val, 0.0)  # only helps if OTT is positive
    total += dist_adj

    # Apply accuracy penalty (negative if player is inaccurate)
    acc_penalty = profile.get("accuracy_penalty", 0.0)
    # Accuracy penalty scales with negative OTT (wild off the tee)
    if ott_val < 0:
        acc_adj = acc_penalty * abs(ott_val)
        total += acc_adj
    else:
        acc_adj = 0.0

    return {
        "course_adj_total": round(total, 4),
        "course_found": True,
        "course_name": course_name,
        "contributions": contributions,
        "distance_adj": round(dist_adj, 4),
        "accuracy_adj": round(acc_adj, 4),
        "bermuda_greens": profile.get("bermuda_greens", False),
        "wind_sensitivity": profile.get("wind_sensitivity", 0.3),
        "elevation": profile.get("elevation", 0),
    }


def compute_form_trend(df: pd.DataFrame) -> dict:
    """Compute form trend from recent SG history using OLS slope regression.

    Also computes a convergence score: how consistent recent performance is.

    Args:
        df: DataFrame with sg_total column, sorted by date descending.

    Returns:
        Dict with slope, r_squared, convergence, trend_label, n_events.
    """
    col = "sg_total"
    if col not in df.columns or len(df) < 3:
        return {
            "slope": 0.0,
            "r_squared": 0.0,
            "convergence": 0.0,
            "trend_label": "Insufficient Data",
            "n_events": len(df) if col in df.columns else 0,
        }

    values = df[col].dropna().values
    n = min(len(values), 24)
    values = values[:n][::-1]  # chronological order for regression

    if len(values) < 3:
        return {
            "slope": 0.0, "r_squared": 0.0, "convergence": 0.0,
            "trend_label": "Insufficient Data", "n_events": len(values),
        }

    x = np.arange(len(values), dtype=float)
    slope_result = sp_stats.linregress(x, values)
    slope = slope_result.slope
    r_sq = slope_result.rvalue ** 2

    # Convergence: inverse of coefficient of variation (consistency)
    mean_val = np.mean(values)
    std_val = np.std(values)
    if std_val > 0 and abs(mean_val) > 0.01:
        cv = std_val / abs(mean_val)
        convergence = max(0.0, min(1.0, 1.0 - cv))
    else:
        convergence = 0.5

    # Label the trend
    if slope > 0.05:
        trend_label = "Strong Uptrend"
    elif slope > 0.02:
        trend_label = "Mild Uptrend"
    elif slope < -0.05:
        trend_label = "Strong Downtrend"
    elif slope < -0.02:
        trend_label = "Mild Downtrend"
    else:
        trend_label = "Stable"

    return {
        "slope": round(slope, 4),
        "r_squared": round(r_sq, 4),
        "convergence": round(convergence, 4),
        "trend_label": trend_label,
        "n_events": n,
    }


def monte_carlo_win_prob(player_sg: float, field_sg_list: list,
                         n_sims: int = 10000, player_sigma: float = 1.1) -> dict:
    """Full Monte Carlo simulation for tournament outcome probabilities.

    Simulates n_sims tournaments. Each player's 4-round total is drawn from
    N(sg * 4, sigma * sqrt(4)). Player wins if their total is the best.

    Args:
        player_sg: Player's projected per-round SG total.
        field_sg_list: List of SG totals for all other players in the field.
        n_sims: Number of simulations (default 10,000).
        player_sigma: Per-round standard deviation for the player.

    Returns:
        Dict with win_prob, top5_prob, top10_prob, top20_prob, made_cut_prob,
        avg_finish, median_finish.
    """
    rng = np.random.default_rng(seed=42)
    n_opponents = len(field_sg_list)
    field_sg = np.array(field_sg_list, dtype=float)
    n_rounds = 4

    # Player's 4-round total across all sims
    player_totals = rng.normal(
        loc=player_sg * n_rounds,
        scale=player_sigma * math.sqrt(n_rounds),
        size=n_sims,
    )

    # Opponents' 4-round totals: shape (n_sims, n_opponents)
    # Each opponent has their own mean but shared sigma assumption
    opponent_sigmas = np.full(n_opponents, 1.15)  # field slightly higher variance
    opponent_means = field_sg * n_rounds

    opponent_totals = rng.normal(
        loc=opponent_means[np.newaxis, :],
        scale=(opponent_sigmas * math.sqrt(n_rounds))[np.newaxis, :],
        size=(n_sims, n_opponents),
    )

    # Rank: higher SG total is better, so rank descending
    # For each sim, count how many opponents beat the player
    beaten_by = np.sum(opponent_totals > player_totals[:, np.newaxis], axis=1)
    finishes = beaten_by + 1  # 1-indexed finish position

    win_prob = float(np.mean(finishes == 1))
    top5_prob = float(np.mean(finishes <= 5))
    top10_prob = float(np.mean(finishes <= 10))
    top20_prob = float(np.mean(finishes <= 20))
    avg_finish = float(np.mean(finishes))
    median_finish = float(np.median(finishes))

    # Made cut: approximate as top 65 + ties (roughly top 50% for 156-player field)
    cut_line = max(65, int(n_opponents * 0.45))
    made_cut_prob = float(np.mean(finishes <= cut_line))

    return {
        "win_prob": round(win_prob, 5),
        "top5_prob": round(top5_prob, 4),
        "top10_prob": round(top10_prob, 4),
        "top20_prob": round(top20_prob, 4),
        "made_cut_prob": round(made_cut_prob, 4),
        "avg_finish": round(avg_finish, 1),
        "median_finish": round(median_finish, 1),
        "n_sims": n_sims,
        "field_size": n_opponents + 1,
    }


def sg_to_top_n_prob(sg: float, field_sg: list, n: int,
                     n_sims: int = 10000) -> float:
    """Monte Carlo probability of finishing in the top N.

    Args:
        sg: Player's projected SG total per round.
        field_sg: List of opponent SG totals per round.
        n: Top N threshold (e.g., 5, 10, 20).
        n_sims: Number of simulations.

    Returns:
        Probability of finishing top N.
    """
    rng = np.random.default_rng(seed=hash(f"{sg}_{n}") % (2**31))
    n_rounds = 4
    sigma = 1.1

    player_totals = rng.normal(sg * n_rounds, sigma * math.sqrt(n_rounds), n_sims)

    field_arr = np.array(field_sg, dtype=float)
    opponent_totals = rng.normal(
        loc=(field_arr * n_rounds)[np.newaxis, :],
        scale=1.15 * math.sqrt(n_rounds),
        size=(n_sims, len(field_arr)),
    )

    beaten_by = np.sum(opponent_totals > player_totals[:, np.newaxis], axis=1)
    finishes = beaten_by + 1
    return round(float(np.mean(finishes <= n)), 4)


def sg_to_make_cut_prob(sg: float) -> float:
    """Logistic regression calibration: SG total -> made cut probability.

    Calibrated on historical PGA Tour data.
    At SG=0 (tour average), cut probability is ~65%.
    At SG=+2.0, cut probability is ~92%.
    At SG=-1.0, cut probability is ~38%.

    Args:
        sg: Player's projected SG total per round.

    Returns:
        Probability of making the cut (0 to 1).
    """
    # Logistic coefficients calibrated to historical data
    # logit(p) = beta0 + beta1 * sg
    beta0 = 0.62   # intercept: logit(0.65) ≈ 0.62
    beta1 = 1.35   # slope: each SG stroke ≈ 1.35 logit units
    logit_p = beta0 + beta1 * sg
    prob = sigmoid(logit_p)
    return round(float(prob), 4)


def field_strength_adjustment(field_sg_values: list) -> dict:
    """Analyze field strength and compute adjustment factors.

    Compares the field's average SG to tour average (0) and computes
    adjustments for probability calculations.

    Args:
        field_sg_values: List of SG totals for all players in the field.

    Returns:
        Dict with field_avg, field_std, strength_label, adjustment_factor.
    """
    if not field_sg_values:
        return {
            "field_avg": 0.0, "field_std": 1.0,
            "strength_label": "Unknown", "adjustment_factor": 1.0,
            "n_players": 0,
        }

    arr = np.array(field_sg_values, dtype=float)
    field_avg = float(np.mean(arr))
    field_std = float(np.std(arr))
    n_players = len(arr)

    # Count elite players (SG > 1.0)
    n_elite = int(np.sum(arr > 1.0))
    pct_elite = n_elite / n_players if n_players > 0 else 0

    # Strength classification
    if field_avg > 0.3 or pct_elite > 0.15:
        strength_label = "Elite"
        adjustment_factor = 0.88  # harder to place well
    elif field_avg > 0.1 or pct_elite > 0.08:
        strength_label = "Strong"
        adjustment_factor = 0.94
    elif field_avg > -0.1:
        strength_label = "Average"
        adjustment_factor = 1.0
    elif field_avg > -0.3:
        strength_label = "Weak"
        adjustment_factor = 1.08
    else:
        strength_label = "Very Weak"
        adjustment_factor = 1.15

    return {
        "field_avg": round(field_avg, 3),
        "field_std": round(field_std, 3),
        "strength_label": strength_label,
        "adjustment_factor": round(adjustment_factor, 3),
        "n_players": n_players,
        "n_elite": n_elite,
        "pct_elite": round(pct_elite, 3),
    }


def compute_projection_ci(mu: float, sigma: float,
                           confidence: float = 0.80) -> tuple:
    """Compute confidence interval for a projection.

    Args:
        mu: Projected mean.
        sigma: Standard deviation of projection.
        confidence: Confidence level (default 80%).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    alpha = 1.0 - confidence
    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)
    lower = mu - z * sigma
    upper = mu + z * sigma
    return (round(lower, 3), round(upper, 3))


def kelly_fraction_calc(win_prob: float, decimal_odds: float,
                        fractional: float = 0.25) -> float:
    """Compute Kelly Criterion fraction for optimal bet sizing.

    Uses fractional Kelly (default quarter-Kelly) for risk management.

    Formula:
        full_kelly = (p * (odds - 1) - (1 - p)) / (odds - 1)
        fractional_kelly = full_kelly * fractional

    Args:
        win_prob: Estimated probability of winning the bet.
        decimal_odds: Decimal odds (e.g., 3.0 means 2:1 payout).
        fractional: Kelly fraction (0.25 = quarter Kelly).

    Returns:
        Recommended fraction of bankroll to wager (0 if negative edge).
    """
    if decimal_odds <= 1.0 or win_prob <= 0.0 or win_prob >= 1.0:
        return 0.0

    b = decimal_odds - 1.0  # net payout per unit wagered
    p = win_prob
    q = 1.0 - p

    full_kelly = (p * b - q) / b
    if full_kelly <= 0:
        return 0.0

    frac_kelly = full_kelly * fractional
    # Cap at 10% of bankroll as safety
    capped = min(frac_kelly, 0.10)
    return round(capped, 4)


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds.

    Args:
        odds: American odds (e.g., +500, -110).

    Returns:
        Decimal odds (e.g., 6.0, 1.909).
    """
    if odds > 0:
        return round(1.0 + odds / 100.0, 4)
    elif odds < 0:
        return round(1.0 + 100.0 / abs(odds), 4)
    else:
        return 1.0


def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability (no-vig).

    Args:
        odds: American odds (e.g., +500, -110).

    Returns:
        Implied probability (0 to 1).
    """
    if odds > 0:
        prob = 100.0 / (odds + 100.0)
    elif odds < 0:
        prob = abs(odds) / (abs(odds) + 100.0)
    else:
        prob = 0.5
    return round(prob, 4)


# ============================================================
# CLAUDE AI INTEGRATION
# ============================================================

def _get_anthropic_key() -> str:
    """Retrieve the Anthropic API key from environment or Streamlit secrets.

    Checks in order:
      1. st.secrets["ANTHROPIC_API_KEY"]
      2. os.environ["ANTHROPIC_API_KEY"]
      3. .env file in project root

    Returns:
        API key string, or empty string if not found.
    """
    # Try streamlit secrets first
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if key:
            return key
    except Exception:
        pass

    # Try environment variable
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key

    # Try .env file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip("\"'")
                    if val:
                        return val
        except Exception:
            pass

    return ""


def _anthropic_client():
    """Create and return an Anthropic client instance.

    Returns:
        anthropic.Anthropic client, or None if unavailable.
    """
    key = _get_anthropic_key()
    if not key:
        return None

    try:
        import anthropic
        return anthropic.Anthropic(api_key=key)
    except ImportError:
        st.warning("anthropic package not installed. Run: pip install anthropic")
        return None
    except Exception as e:
        st.warning(f"Failed to initialize Anthropic client: {e}")
        return None


def ai_edge_analysis(player: str, sg_data: dict, course: str,
                     odds: int = 0, field_strength: str = "Average",
                     weather_note: str = "", recent_form: str = "") -> str:
    """Use Claude claude-haiku-4-5 for fast edge analysis on a single player.

    Args:
        player: Player name.
        sg_data: Dict of SG category values and projections.
        course: Course name for context.
        odds: American odds (0 if unavailable).
        field_strength: Field strength label.
        weather_note: Optional weather context.
        recent_form: Optional recent form summary.

    Returns:
        AI-generated edge analysis text, or fallback message.
    """
    client = _anthropic_client()
    if client is None:
        return "[AI analysis unavailable — no API key configured]"

    sg_summary = "\n".join(f"  {k}: {v:+.2f}" if isinstance(v, (int, float))
                           else f"  {k}: {v}" for k, v in sg_data.items())
    odds_str = f"American odds: {odds:+d}" if odds != 0 else "No odds available"

    prompt = f"""You are a professional golf analytics assistant. Provide a concise edge analysis (3-5 sentences) for this player's tournament outlook.

Player: {player}
Course: {course}
Field Strength: {field_strength}
{odds_str}
Weather: {weather_note if weather_note else 'Standard conditions'}
Recent Form: {recent_form if recent_form else 'Not specified'}

Strokes Gained Profile:
{sg_summary}

Focus on: course fit, current form trajectory, key statistical edges or weaknesses, and whether the odds offer value. Be specific and quantitative. Do not use generic statements."""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"[AI analysis error: {str(e)[:100]}]"


def ai_slate_briefing(edges_json: str) -> str:
    """Use Claude claude-sonnet-4-6 for comprehensive slate briefing.

    Args:
        edges_json: JSON string of all player edges and projections for the slate.

    Returns:
        AI-generated slate briefing text, or fallback message.
    """
    client = _anthropic_client()
    if client is None:
        return "[AI slate briefing unavailable — no API key configured]"

    prompt = f"""You are an elite golf betting analyst. Provide a comprehensive slate briefing (8-12 sentences) covering:

1. Top 3 value plays with specific reasoning
2. Key course-fit angles for this week
3. Players to avoid (poor course fit or negative form trends)
4. Recommended betting strategy (outright, top 10/20, matchups, PrizePicks)

Player Edge Data:
{edges_json}

Be specific, data-driven, and actionable. Reference SG numbers and probabilities. Format with clear sections."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"[AI slate briefing error: {str(e)[:100]}]"


# ============================================================
# PRIZEPICKS STAT PROJECTION SYSTEM
# ============================================================

# Standard deviations for each PrizePicks stat type (per round)
STAT_STD = {
    "fantasy_score":  12.0,
    "birdies":        1.5,
    "bogey_free":     0.42,   # binary-ish, so std of Bernoulli
    "strokes":        2.8,
    "gir":            2.5,    # greens in regulation count (out of 18)
    "fairways":       2.2,    # fairways hit count (out of 14)
    "putts":          2.0,
}

# Sensitivity mapping: how each SG category influences each PP stat
# Values represent the change in the stat per +1.0 SG in that category
SG_TO_STAT_SENSITIVITY = {
    "fantasy_score": {
        "sg_ott": 3.0, "sg_app": 5.5, "sg_arg": 3.5, "sg_putt": 6.0,
    },
    "birdies": {
        "sg_ott": 0.3, "sg_app": 0.6, "sg_arg": 0.4, "sg_putt": 0.8,
    },
    "bogey_free": {
        "sg_ott": 0.04, "sg_app": 0.06, "sg_arg": 0.05, "sg_putt": 0.03,
    },
    "strokes": {
        "sg_ott": -0.9, "sg_app": -1.1, "sg_arg": -0.8, "sg_putt": -1.0,
    },
    "gir": {
        "sg_ott": 0.4, "sg_app": 1.8, "sg_arg": 0.1, "sg_putt": 0.0,
    },
    "fairways": {
        "sg_ott": 1.5, "sg_app": 0.0, "sg_arg": 0.0, "sg_putt": 0.0,
    },
    "putts": {
        "sg_ott": 0.0, "sg_app": 0.0, "sg_arg": -0.2, "sg_putt": -1.5,
    },
}


def project_pp_stat(stat_type: str, player_proj: dict) -> tuple:
    """Project a PrizePicks stat line for a player.

    Uses SG-to-stat sensitivity mappings and tour baselines to project
    the expected stat value and uncertainty.

    Args:
        stat_type: One of 'fantasy_score', 'birdies', 'bogey_free', 'strokes',
                   'gir', 'fairways', 'putts'.
        player_proj: Dict with keys sg_ott, sg_app, sg_arg, sg_putt.

    Returns:
        Tuple of (projected_value, projected_std).
    """
    # Map stat_type to baseline key
    baseline_map = {
        "fantasy_score": "fantasy_score",
        "birdies": "birdies_per_round",
        "bogey_free": "bogey_free_pct",
        "strokes": "scoring_avg",
        "gir": "gir_pct",
        "fairways": "fairways_pct",
        "putts": "putts_per_round",
    }
    baseline_key = baseline_map.get(stat_type, stat_type)
    baseline_val = TOUR_BASELINES.get(baseline_key, 0.0)

    # Convert percentage baselines to count-based for GIR and fairways
    if stat_type == "gir":
        baseline_val = baseline_val / 100.0 * 18.0  # GIR count out of 18
    elif stat_type == "fairways":
        baseline_val = baseline_val / 100.0 * 14.0  # Fairways out of 14
    elif stat_type == "bogey_free":
        baseline_val = baseline_val / 100.0  # probability

    # Get sensitivities for this stat
    sens = SG_TO_STAT_SENSITIVITY.get(stat_type, {})

    # Compute adjustment from SG profile
    adjustment = 0.0
    for sg_cat in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]:
        sg_val = player_proj.get(sg_cat, 0.0)
        sensitivity = sens.get(sg_cat, 0.0)
        adjustment += sg_val * sensitivity

    projected = baseline_val + adjustment
    std = STAT_STD.get(stat_type, 2.0)

    # Clamp projections to reasonable ranges
    if stat_type == "birdies":
        projected = max(0.5, projected)
    elif stat_type == "bogey_free":
        projected = max(0.01, min(0.95, projected))
    elif stat_type == "strokes":
        projected = max(64.0, min(78.0, projected))
    elif stat_type == "gir":
        projected = max(4.0, min(18.0, projected))
    elif stat_type == "fairways":
        projected = max(3.0, min(14.0, projected))
    elif stat_type == "putts":
        projected = max(24.0, min(34.0, projected))
    elif stat_type == "fantasy_score":
        projected = max(10.0, projected)

    return (round(projected, 2), round(std, 2))


def prob_over(proj: float, line: float, std: float) -> float:
    """Probability that the actual stat exceeds the PrizePicks line.

    Uses normal CDF: P(X > line) = 1 - Phi((line - proj) / std).

    Args:
        proj: Projected value.
        line: PrizePicks line.
        std: Standard deviation.

    Returns:
        Probability (0 to 1).
    """
    if std <= 0:
        return 1.0 if proj > line else 0.0
    z = (line - proj) / std
    return round(float(1.0 - sp_stats.norm.cdf(z)), 4)


def prob_under(proj: float, line: float, std: float) -> float:
    """Probability that the actual stat falls under the PrizePicks line.

    Uses normal CDF: P(X < line) = Phi((line - proj) / std).

    Args:
        proj: Projected value.
        line: PrizePicks line.
        std: Standard deviation.

    Returns:
        Probability (0 to 1).
    """
    if std <= 0:
        return 1.0 if proj < line else 0.0
    z = (line - proj) / std
    return round(float(sp_stats.norm.cdf(z)), 4)


def pp_combo_ev(pick_probs: list, play_type: str = "power_play") -> dict:
    """Calculate expected value for a PrizePicks combo.

    Args:
        pick_probs: List of individual pick probabilities (each 0-1).
        play_type: 'power_play' or 'flex_play'.

    Returns:
        Dict with ev_per_dollar, combo_prob, payout_multiplier, edge,
        kelly_fraction, and detailed breakdowns for flex.
    """
    n_picks = len(pick_probs)
    payouts = PP_PAYOUTS.get(play_type, {}).get(n_picks, {})

    if not payouts:
        return {
            "ev_per_dollar": -1.0,
            "combo_prob": 0.0,
            "payout_multiplier": 0.0,
            "edge": -1.0,
            "kelly_fraction": 0.0,
            "error": f"No payout structure for {play_type} with {n_picks} picks",
        }

    if play_type == "power_play":
        # All must be correct
        combo_prob = 1.0
        for p in pick_probs:
            combo_prob *= p

        payout_mult = payouts.get("all_correct", 1.0)
        ev = combo_prob * payout_mult - 1.0  # net EV per $1

        kelly = kelly_fraction_calc(combo_prob, payout_mult, fractional=0.25)

        return {
            "ev_per_dollar": round(ev, 4),
            "combo_prob": round(combo_prob, 4),
            "payout_multiplier": payout_mult,
            "edge": round(ev, 4),
            "kelly_fraction": kelly,
            "play_type": "Power Play",
            "n_picks": n_picks,
        }

    else:
        # Flex play: multiple payout tiers
        # Calculate probability of getting exactly k correct out of n
        probs_array = np.array(pick_probs)
        n = len(probs_array)

        # Dynamic programming for exact-k-correct probabilities
        # dp[i][j] = prob of exactly j correct in first i picks
        dp = np.zeros((n + 1, n + 1))
        dp[0][0] = 1.0
        for i in range(n):
            p = probs_array[i]
            for j in range(i + 1):
                dp[i + 1][j + 1] += dp[i][j] * p        # correct
                dp[i + 1][j] += dp[i][j] * (1.0 - p)   # incorrect

        # Calculate EV from each tier
        total_ev = 0.0
        tier_details = {}
        for tier_name, mult in payouts.items():
            k = int(tier_name.split("_")[0])
            prob_k = float(dp[n][k])
            tier_ev = prob_k * mult
            total_ev += tier_ev
            tier_details[tier_name] = {
                "prob": round(prob_k, 4),
                "payout": mult,
                "ev_contribution": round(tier_ev, 4),
            }

        net_ev = total_ev - 1.0

        # Kelly on the best tier
        best_tier_prob = max((d["prob"] for d in tier_details.values()), default=0)
        best_tier_payout = max((d["payout"] for d in tier_details.values()), default=1)
        kelly = kelly_fraction_calc(best_tier_prob, best_tier_payout, fractional=0.15)

        return {
            "ev_per_dollar": round(net_ev, 4),
            "combo_prob_all": round(float(dp[n][n]), 4),
            "total_return": round(total_ev, 4),
            "edge": round(net_ev, 4),
            "kelly_fraction": kelly,
            "play_type": "Flex Play",
            "n_picks": n_picks,
            "tier_details": tier_details,
        }


# ============================================================
# PART 2 — Dashboard UI: Sidebar + 6 Tabs
# ============================================================


# ── Sample data generator for demo mode ─────────────────────
def _generate_sample_players(n: int = 30, course: str = "Augusta National") -> pd.DataFrame:
    """Generate realistic sample player data for demo/testing."""
    names = [
        "Scottie Scheffler", "Rory McIlroy", "Jon Rahm", "Viktor Hovland",
        "Xander Schauffele", "Patrick Cantlay", "Collin Morikawa", "Ludvig Aberg",
        "Wyndham Clark", "Max Homa", "Matt Fitzpatrick", "Sam Burns",
        "Tony Finau", "Sahith Theegala", "Tommy Fleetwood", "Shane Lowry",
        "Brian Harman", "Russell Henley", "Sungjae Im", "Tom Kim",
        "Cameron Young", "Justin Thomas", "Jordan Spieth", "Hideki Matsuyama",
        "Corey Conners", "Cameron Smith", "Dustin Johnson", "Brooks Koepka",
        "Will Zalatoris", "Keegan Bradley",
    ][:n]
    rng = np.random.RandomState(42)
    rows = []
    for i, name in enumerate(names):
        base = 1.5 - i * 0.08 + rng.normal(0, 0.3)
        sg_ott = round(rng.normal(0.4, 0.6), 2)
        sg_app = round(rng.normal(0.3, 0.7), 2)
        sg_arg = round(rng.normal(0.1, 0.5), 2)
        sg_putt = round(rng.normal(0.2, 0.8), 2)
        sg_total = round(sg_ott + sg_app + sg_arg + sg_putt, 2)
        events = rng.randint(8, 28)
        rows.append({
            "player": name,
            "sg_total": sg_total,
            "sg_ott": sg_ott,
            "sg_app": sg_app,
            "sg_arg": sg_arg,
            "sg_putt": sg_putt,
            "events": events,
            "odds": int(rng.choice([600, 800, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000])),
            "world_rank": i + 1,
        })
    return pd.DataFrame(rows)


def _enrich_player_row(row: dict, course: str) -> dict:
    """Add computed projections to a player row."""
    sg_dict = {k: row[k] for k in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]}

    # Apply course fit
    fitted = apply_course_fit(sg_dict, course)
    row["sg_fitted"] = round(fitted["sg_fitted_total"], 3)
    row["course_delta"] = round(fitted["sg_fitted_total"] - row["sg_total"], 3)

    # Bayesian shrinkage
    row["sg_shrunk"] = round(bayesian_shrink(row["sg_fitted"], row["events"]), 3)

    # Regression to mean
    row["sg_regressed"] = round(regress_to_mean(row["sg_shrunk"], row["events"]), 3)

    return row


def _build_projection_table(df: pd.DataFrame, course: str) -> pd.DataFrame:
    """Build full projection table with all computed columns."""
    enriched = [_enrich_player_row(row.to_dict(), course) for _, row in df.iterrows()]
    proj_df = pd.DataFrame(enriched)

    # Win / Top-5 / Top-10 / Top-20 / Make Cut probabilities
    field_sg = proj_df["sg_regressed"].tolist()
    win_probs, t5_probs, t10_probs, t20_probs, cut_probs = [], [], [], [], []

    for _, r in proj_df.iterrows():
        wp = monte_carlo_win_prob(r["sg_regressed"], field_sg, n_sims=5000)
        win_probs.append(wp["win_prob"])
        t5_probs.append(sg_to_top_n_prob(r["sg_regressed"], field_sg, 5, n_sims=3000))
        t10_probs.append(sg_to_top_n_prob(r["sg_regressed"], field_sg, 10, n_sims=3000))
        t20_probs.append(sg_to_top_n_prob(r["sg_regressed"], field_sg, 20, n_sims=3000))
        cut_probs.append(sg_to_make_cut_prob(r["sg_regressed"]))

    proj_df["win_prob"] = win_probs
    proj_df["top5_prob"] = t5_probs
    proj_df["top10_prob"] = t10_probs
    proj_df["top20_prob"] = t20_probs
    proj_df["cut_prob"] = cut_probs

    # Implied probabilities and edges from odds
    proj_df["implied_prob"] = proj_df["odds"].apply(american_to_implied_prob)
    proj_df["edge"] = proj_df["win_prob"] - proj_df["implied_prob"]
    proj_df["decimal_odds"] = proj_df["odds"].apply(american_to_decimal)
    proj_df["kelly"] = proj_df.apply(
        lambda r: kelly_fraction_calc(r["win_prob"], r["decimal_odds"]), axis=1
    )

    # Confidence interval
    for _, r in proj_df.iterrows():
        ci = compute_projection_ci(r["sg_regressed"], 0.8, n_events=r["events"])
        proj_df.loc[proj_df["player"] == r["player"], "ci_low"] = ci["ci_lower"]
        proj_df.loc[proj_df["player"] == r["player"], "ci_high"] = ci["ci_upper"]

    proj_df = proj_df.sort_values("sg_regressed", ascending=False).reset_index(drop=True)
    proj_df.index = proj_df.index + 1  # 1-based rank
    return proj_df


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar() -> dict:
    """Render sidebar controls and return settings dict."""
    with st.sidebar:
        st.markdown(section_header("Golf Quant Engine", "&#9971;", "v2.0"), unsafe_allow_html=True)
        st.markdown("---")

        # Course selector
        course = st.selectbox(
            "Tournament Course",
            options=sorted(COURSE_PROFILES.keys()),
            index=sorted(COURSE_PROFILES.keys()).index("Augusta National")
            if "Augusta National" in COURSE_PROFILES else 0,
        )

        # Course info
        cp = COURSE_PROFILES.get(course, {})
        if cp:
            st.markdown(f"""
            <div class="glass-card" style="padding:12px;margin:8px 0;">
                <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">Course Profile</div>
                <div style="font-size:0.85rem;color:#e2e8f0;margin-top:6px;">{cp.get('notes', '')}</div>
                <div style="display:flex;gap:12px;margin-top:8px;">
                    <span class="mono" style="font-size:0.75rem;color:#60a5fa;">Par {cp.get('par', 72)}</span>
                    <span class="mono" style="font-size:0.75rem;color:#fbbf24;">Wind: {cp.get('wind_sensitivity', 0):.1f}</span>
                    <span class="mono" style="font-size:0.75rem;color:#4ade80;">Elev: {cp.get('elevation', 0)}ft</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # SG Weight visualization
        st.markdown("**SG Weight Distribution**")
        weights = cp.get("sg_weights", SG_WEIGHTS)
        weight_labels = {"sg_ott": "Off Tee", "sg_app": "Approach", "sg_arg": "Around Green", "sg_putt": "Putting"}
        for cat, w in weights.items():
            label = weight_labels.get(cat, cat)
            bar_pct = w * 100 / 0.40  # normalize to max ~40%
            st.markdown(f"""
            <div style="margin:3px 0;">
                <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#94a3b8;">
                    <span>{label}</span><span class="mono">{w:.0%}</span>
                </div>
                <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:6px;overflow:hidden;">
                    <div style="width:{min(bar_pct, 100):.0f}%;height:100%;background:linear-gradient(90deg,#4ade80,#22d3ee);border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Simulation settings
        st.markdown("**Simulation Settings**")
        n_sims = st.slider("Monte Carlo Sims", 1000, 20000, 5000, 1000)
        bankroll = st.number_input("Bankroll ($)", 100, 100000, 1000, 100)
        kelly_mult = st.slider("Kelly Fraction", 0.1, 1.0, 0.25, 0.05)

        st.markdown("---")

        # Data mode
        data_mode = st.radio("Data Source", ["Demo Data", "Upload CSV"], index=0)
        upload = None
        if data_mode == "Upload CSV":
            upload = st.file_uploader("Upload player SG data", type=["csv"])
            st.markdown("""
            <div style="font-size:0.7rem;color:#64748b;margin-top:4px;">
            CSV must include: player, sg_ott, sg_app, sg_arg, sg_putt, events, odds
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="text-align:center;font-size:0.65rem;color:#475569;padding:8px 0;">
            Golf Quant Engine v2.0<br>
            Research-validated SG model<br>
            &#169; 2026
        </div>
        """, unsafe_allow_html=True)

    return {
        "course": course,
        "n_sims": n_sims,
        "bankroll": bankroll,
        "kelly_mult": kelly_mult,
        "data_mode": data_mode,
        "upload": upload,
    }


# ============================================================
# TAB 1 — POWER RANKINGS
# ============================================================
def tab_power_rankings(proj_df: pd.DataFrame, settings: dict):
    """Render the Power Rankings tab."""
    st.markdown(section_header("Power Rankings", "&#127942;", f"{len(proj_df)} Players"), unsafe_allow_html=True)

    # Top-row metrics
    top = proj_df.iloc[0] if len(proj_df) > 0 else None
    if top is not None:
        cols = st.columns(4)
        with cols[0]:
            st.markdown(metric_card("#1 Ranked", top["player"], f"SG: {top['sg_regressed']:.2f}", "positive", "green"), unsafe_allow_html=True)
        with cols[1]:
            avg_sg = proj_df["sg_regressed"].mean()
            st.markdown(metric_card("Field Avg SG", f"{avg_sg:.2f}", f"{len(proj_df)} players", "neutral", "blue"), unsafe_allow_html=True)
        with cols[2]:
            best_edge = proj_df["edge"].max()
            best_edge_player = proj_df.loc[proj_df["edge"].idxmax(), "player"]
            st.markdown(metric_card("Best Edge", f"{best_edge*100:.1f}%", best_edge_player, "positive", "amber"), unsafe_allow_html=True)
        with cols[3]:
            fs_adj = field_strength_adjustment(proj_df["sg_regressed"].tolist())
            st.markdown(metric_card("Field Strength", fs_adj["label"], f"Factor: {fs_adj['adjustment']:.3f}", "neutral", "purple"), unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # Rankings table
    display_cols = ["player", "sg_regressed", "sg_fitted", "course_delta", "win_prob",
                    "top5_prob", "top10_prob", "cut_prob", "edge", "kelly", "odds"]

    display_df = proj_df[display_cols].copy()
    display_df.columns = ["Player", "SG (adj)", "SG (fit)", "Course Δ", "Win%",
                          "Top 5%", "Top 10%", "Cut%", "Edge", "Kelly", "Odds"]

    # Format percentages
    for col in ["Win%", "Top 5%", "Top 10%", "Cut%"]:
        display_df[col] = (display_df[col] * 100).round(1)
    display_df["Edge"] = (display_df["Edge"] * 100).round(1)

    st.dataframe(
        display_df.style
        .format({"SG (adj)": "{:.2f}", "SG (fit)": "{:.2f}", "Course Δ": "{:+.2f}",
                 "Win%": "{:.1f}%", "Top 5%": "{:.1f}%", "Top 10%": "{:.1f}%",
                 "Cut%": "{:.1f}%", "Edge": "{:+.1f}%", "Kelly": "{:.2f}",
                 "Odds": "+{:.0f}"})
        .background_gradient(subset=["SG (adj)"], cmap="Greens", vmin=-1, vmax=3)
        .background_gradient(subset=["Edge"], cmap="RdYlGn", vmin=-10, vmax=10),
        use_container_width=True,
        height=600,
    )


# ============================================================
# TAB 2 — PLAYER DEEP DIVE
# ============================================================
def tab_player_deep_dive(proj_df: pd.DataFrame, settings: dict):
    """Render the Player Deep Dive tab."""
    st.markdown(section_header("Player Deep Dive", "&#128269;", "Detailed Analysis"), unsafe_allow_html=True)

    player = st.selectbox("Select Player", proj_df["player"].tolist(), index=0)
    p = proj_df[proj_df["player"] == player].iloc[0]

    # Header metrics
    cols = st.columns(5)
    with cols[0]:
        st.markdown(metric_card("SG Total", f"{p['sg_regressed']:.2f}", f"Raw: {p['sg_total']:.2f}", "positive" if p["sg_regressed"] > 0 else "negative", "green"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_card("Win Prob", f"{p['win_prob']*100:.1f}%", f"Implied: {p['implied_prob']*100:.1f}%", "positive" if p["edge"] > 0 else "negative", "blue"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_card("Edge", f"{p['edge']*100:+.1f}%", f"Kelly: {p['kelly']:.2f}", "positive" if p["edge"] > 0 else "negative", "amber"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(metric_card("Course Fit", f"{p['course_delta']:+.2f}", settings["course"], "positive" if p["course_delta"] > 0 else "negative", "green"), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(metric_card("Make Cut", f"{p['cut_prob']*100:.0f}%", f"Top 20: {p['top20_prob']*100:.1f}%", "positive" if p["cut_prob"] > 0.7 else "neutral", "blue"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # SG Radar Chart
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Strokes Gained Profile**")
        categories = ["Off Tee", "Approach", "Around Green", "Putting"]
        sg_vals = [p["sg_ott"], p["sg_app"], p["sg_arg"], p["sg_putt"]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=sg_vals + [sg_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(74, 222, 128, 0.15)",
            line=dict(color="#4ade80", width=2),
            name=player,
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[-2, 2], gridcolor="rgba(255,255,255,0.1)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="Inter"),
            showlegend=False,
            height=350,
            margin=dict(l=40, r=40, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Probability Breakdown**")
        prob_data = [
            ("Win", p["win_prob"], "green"),
            ("Top 5", p["top5_prob"], "green"),
            ("Top 10", p["top10_prob"], "blue"),
            ("Top 20", p["top20_prob"], "blue"),
            ("Make Cut", p["cut_prob"], "amber"),
        ]
        for label, prob, color in prob_data:
            st.markdown(prob_bar_html(prob, color, label), unsafe_allow_html=True)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown("**Confidence Interval**")
        ci_low = p.get("ci_low", p["sg_regressed"] - 0.5)
        ci_high = p.get("ci_high", p["sg_regressed"] + 0.5)
        st.markdown(f"""
        <div class="glass-card" style="padding:12px;">
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#94a3b8;">
                <span>Low: <span class="mono text-red">{ci_low:.2f}</span></span>
                <span>Proj: <span class="mono text-green">{p['sg_regressed']:.2f}</span></span>
                <span>High: <span class="mono text-blue">{ci_high:.2f}</span></span>
            </div>
            <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:8px;margin-top:8px;position:relative;">
                <div style="position:absolute;left:{max(0, (ci_low+2)/4*100):.0f}%;right:{max(0, 100-(ci_high+2)/4*100):.0f}%;height:100%;background:linear-gradient(90deg,#f87171,#4ade80,#60a5fa);border-radius:4px;opacity:0.6;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # AI Analysis
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    if st.button("Generate AI Analysis", key="ai_deep_dive"):
        with st.spinner("Claude is analyzing..."):
            sg_data = {k: p[k] for k in ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_total"]}
            try:
                analysis = ai_edge_analysis(player, sg_data, settings["course"], p["odds"])
                st.markdown(f"""
                <div class="glass-card" style="padding:16px;">
                    <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Claude AI Analysis</div>
                    <div style="font-size:0.85rem;color:#e2e8f0;line-height:1.6;">{analysis}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"AI analysis unavailable: {e}")


# ============================================================
# TAB 3 — COURSE FIT MATRIX
# ============================================================
def tab_course_fit(proj_df: pd.DataFrame, settings: dict):
    """Render the Course Fit Matrix tab."""
    st.markdown(section_header("Course Fit Matrix", "&#127959;", settings["course"]), unsafe_allow_html=True)

    course = settings["course"]
    cp = COURSE_PROFILES.get(course, {})
    weights = cp.get("sg_weights", SG_WEIGHTS)

    # Course profile overview
    cols = st.columns(4)
    with cols[0]:
        dominant = max(weights, key=weights.get)
        label_map = {"sg_ott": "Off Tee", "sg_app": "Approach", "sg_arg": "Around Green", "sg_putt": "Putting"}
        st.markdown(metric_card("Key Skill", label_map.get(dominant, dominant), f"Weight: {weights[dominant]:.0%}", "positive", "green"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_card("Distance Bonus", f"{cp.get('distance_bonus', 0):+.0%}", "Favors long hitters" if cp.get("distance_bonus", 0) > 0 else "Neutral", "positive" if cp.get("distance_bonus", 0) > 0 else "neutral", "blue"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_card("Wind Factor", f"{cp.get('wind_sensitivity', 0):.1f}", "High" if cp.get("wind_sensitivity", 0) > 0.5 else "Moderate", "negative" if cp.get("wind_sensitivity", 0) > 0.5 else "neutral", "amber"), unsafe_allow_html=True)
    with cols[3]:
        grass = "Bermuda" if cp.get("bermuda_greens", False) else "Bent/Poa"
        st.markdown(metric_card("Green Type", grass, f"Elev: {cp.get('elevation', 0)}ft", "neutral", "green"), unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # Course fit heatmap
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**SG Weight Comparison vs Default**")
        default_w = SG_WEIGHTS
        comp_data = []
        for cat in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]:
            comp_data.append({
                "Category": label_map.get(cat, cat),
                "Course Weight": weights.get(cat, 0),
                "Default Weight": default_w.get(cat, 0),
                "Delta": weights.get(cat, 0) - default_w.get(cat, 0),
            })
        comp_df = pd.DataFrame(comp_data)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comp_df["Category"], y=comp_df["Course Weight"],
            name=course, marker_color="#4ade80",
        ))
        fig.add_trace(go.Bar(
            x=comp_df["Category"], y=comp_df["Default Weight"],
            name="Tour Default", marker_color="rgba(255,255,255,0.2)",
        ))
        fig.update_layout(
            barmode="group", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="Inter"),
            legend=dict(orientation="h", y=1.15),
            margin=dict(l=40, r=20, t=10, b=40),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickformat=".0%"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Top Course Fits**")
        top_fits = proj_df.nlargest(10, "course_delta")[["player", "course_delta", "sg_fitted"]].reset_index(drop=True)
        for _, r in top_fits.iterrows():
            color = "#4ade80" if r["course_delta"] > 0 else "#f87171"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
                <span style="font-size:0.8rem;color:#e2e8f0;">{r['player']}</span>
                <span class="mono" style="font-size:0.8rem;color:{color};">{r['course_delta']:+.2f}</span>
            </div>
            """, unsafe_allow_html=True)

    # Multi-course comparison
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("**Multi-Course Fit Comparison (Top 10 Players)**")
    top_players = proj_df.nlargest(10, "sg_regressed")["player"].tolist()
    courses_to_compare = sorted(COURSE_PROFILES.keys())[:8]

    heat_data = []
    for pl in top_players:
        row_data = {"Player": pl}
        p_row = proj_df[proj_df["player"] == pl].iloc[0]
        sg_dict = {k: p_row[k] for k in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]}
        for c in courses_to_compare:
            fit = apply_course_fit(sg_dict, c)
            row_data[c] = round(fit["sg_fitted_total"], 2)
        heat_data.append(row_data)

    heat_df = pd.DataFrame(heat_data).set_index("Player")
    fig = px.imshow(
        heat_df.values,
        x=heat_df.columns.tolist(),
        y=heat_df.index.tolist(),
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Inter", size=11),
        margin=dict(l=120, r=20, t=10, b=80),
        xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 4 — BETTING EDGE FINDER
# ============================================================
def tab_betting_edge(proj_df: pd.DataFrame, settings: dict):
    """Render the Betting Edge Finder tab."""
    st.markdown(section_header("Betting Edge Finder", "&#128176;", "Value Plays"), unsafe_allow_html=True)

    bankroll = settings["bankroll"]
    kelly_mult = settings["kelly_mult"]

    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        min_edge = st.slider("Minimum Edge %", 0.0, 15.0, 2.0, 0.5)
    with col2:
        bet_type = st.selectbox("Bet Market", ["Outright Winner", "Top 5", "Top 10", "Top 20", "Make Cut"])
    with col3:
        max_odds = st.slider("Max Odds", 500, 30000, 15000, 500)

    # Map bet type to probability column
    prob_col_map = {
        "Outright Winner": "win_prob",
        "Top 5": "top5_prob",
        "Top 10": "top10_prob",
        "Top 20": "top20_prob",
        "Make Cut": "cut_prob",
    }
    prob_col = prob_col_map[bet_type]

    # Filter for edges
    edge_df = proj_df.copy()
    edge_df["model_prob"] = edge_df[prob_col]
    edge_df["market_implied"] = edge_df["implied_prob"]

    if bet_type != "Outright Winner":
        # Approximate implied probs for non-outright markets
        multiplier_map = {"Top 5": 5, "Top 10": 10, "Top 20": 20, "Make Cut": 0.65}
        mult = multiplier_map.get(bet_type, 1)
        if bet_type == "Make Cut":
            edge_df["market_implied"] = mult  # flat ~65% for make cut
        else:
            edge_df["market_implied"] = (edge_df["implied_prob"] * mult).clip(0, 0.95)

    edge_df["bet_edge"] = edge_df["model_prob"] - edge_df["market_implied"]
    edge_df = edge_df[edge_df["bet_edge"] >= min_edge / 100]
    edge_df = edge_df[edge_df["odds"] <= max_odds]
    edge_df = edge_df.sort_values("bet_edge", ascending=False)

    # Summary metrics
    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card("Edges Found", str(len(edge_df)), f"Min: {min_edge:.1f}%", "positive" if len(edge_df) > 0 else "neutral", "green"), unsafe_allow_html=True)
    with cols[1]:
        avg_edge = edge_df["bet_edge"].mean() * 100 if len(edge_df) > 0 else 0
        st.markdown(metric_card("Avg Edge", f"{avg_edge:.1f}%", bet_type, "positive", "blue"), unsafe_allow_html=True)
    with cols[2]:
        total_kelly = edge_df["kelly"].sum() * kelly_mult if len(edge_df) > 0 else 0
        st.markdown(metric_card("Total Kelly", f"{total_kelly:.1%}", f"of ${bankroll:,.0f}", "neutral", "amber"), unsafe_allow_html=True)
    with cols[3]:
        total_wager = total_kelly * bankroll
        st.markdown(metric_card("Total Wager", f"${total_wager:,.0f}", f"{len(edge_df)} bets", "neutral", "green"), unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    if len(edge_df) == 0:
        st.info("No edges found with current filters. Try lowering the minimum edge or increasing max odds.")
        return

    # Edge table
    bet_display = edge_df[["player", "model_prob", "market_implied", "bet_edge", "odds", "kelly"]].copy()
    bet_display["wager"] = (bet_display["kelly"] * kelly_mult * bankroll).round(0)
    bet_display.columns = ["Player", "Model Prob", "Market Implied", "Edge", "Odds", "Kelly", "Wager ($)"]

    st.dataframe(
        bet_display.style
        .format({"Model Prob": "{:.1%}", "Market Implied": "{:.1%}",
                 "Edge": "{:+.1%}", "Kelly": "{:.3f}",
                 "Wager ($)": "${:,.0f}", "Odds": "+{:.0f}"})
        .background_gradient(subset=["Edge"], cmap="Greens", vmin=0, vmax=0.15),
        use_container_width=True,
        height=400,
    )

    # Edge distribution chart
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=edge_df["player"].tolist()[:15],
        y=(edge_df["bet_edge"] * 100).tolist()[:15],
        marker_color=["#4ade80" if e > 0.05 else "#fbbf24" if e > 0.02 else "#60a5fa"
                       for e in edge_df["bet_edge"].tolist()[:15]],
    ))
    fig.update_layout(
        title=dict(text=f"Edge Distribution — {bet_type}", font=dict(size=14, color="#e2e8f0")),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Inter"),
        yaxis=dict(title="Edge %", gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(l=40, r=20, t=40, b=80),
        xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 5 — PRIZEPICKS LAB
# ============================================================
def tab_prizepicks(proj_df: pd.DataFrame, settings: dict):
    """Render the PrizePicks Lab tab."""
    st.markdown(section_header("PrizePicks Lab", "&#127183;", "Prop Analysis"), unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        pp_player = st.selectbox("Player", proj_df["player"].tolist(), key="pp_player")
    with col2:
        stat_type = st.selectbox("Stat Type", list(STAT_STD.keys()), index=0)
    with col3:
        line = st.number_input("PrizePicks Line", min_value=0.0, value=25.0, step=0.5)

    p = proj_df[proj_df["player"] == pp_player].iloc[0]
    sg_proj = {k: p[k] for k in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]}

    proj_val, proj_std = project_pp_stat(stat_type, sg_proj)

    # Probabilities
    p_over = prob_over(proj_val, line, proj_std)
    p_under = prob_under(proj_val, line, proj_std)

    # Summary cards
    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card("Projection", f"{proj_val:.1f}", f"σ: {proj_std:.1f}", "neutral", "green"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_card("Line", f"{line:.1f}", stat_type.replace("_", " ").title(), "neutral", "blue"), unsafe_allow_html=True)
    with cols[2]:
        delta = proj_val - line
        st.markdown(metric_card("OVER", f"{p_over*100:.1f}%", f"Δ: {delta:+.1f}", "positive" if p_over > 0.55 else "neutral", "green"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(metric_card("UNDER", f"{p_under*100:.1f}%", f"Δ: {-delta:+.1f}", "positive" if p_under > 0.55 else "neutral", "red"), unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # Distribution visualization
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Projection Distribution**")
        x_range = np.linspace(proj_val - 3.5 * proj_std, proj_val + 3.5 * proj_std, 200)
        pdf_vals = sp_stats.norm.pdf(x_range, proj_val, proj_std)

        fig = go.Figure()
        # Under region
        x_under = x_range[x_range <= line]
        pdf_under = sp_stats.norm.pdf(x_under, proj_val, proj_std)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_under, [line, x_under[0]]]),
            y=np.concatenate([pdf_under, [0, 0]]),
            fill="toself", fillcolor="rgba(248, 113, 113, 0.3)",
            line=dict(width=0), name="Under",
        ))
        # Over region
        x_over = x_range[x_range >= line]
        pdf_over = sp_stats.norm.pdf(x_over, proj_val, proj_std)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_over, [x_over[-1], line]]),
            y=np.concatenate([pdf_over, [0, 0]]),
            fill="toself", fillcolor="rgba(74, 222, 128, 0.3)",
            line=dict(width=0), name="Over",
        ))
        # Full PDF
        fig.add_trace(go.Scatter(
            x=x_range, y=pdf_vals,
            line=dict(color="#e2e8f0", width=2), name="PDF",
        ))
        # Line marker
        fig.add_vline(x=line, line_dash="dash", line_color="#fbbf24", annotation_text=f"Line: {line}")
        fig.add_vline(x=proj_val, line_dash="dot", line_color="#4ade80", annotation_text=f"Proj: {proj_val:.1f}")

        fig.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="Inter"),
            showlegend=False,
            margin=dict(l=40, r=20, t=10, b=40),
            yaxis=dict(visible=False),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Edge Summary**")
        fair_over = 0.50
        edge_over = p_over - fair_over
        edge_under = p_under - fair_over
        st.markdown(f"""
        <div class="glass-card" style="padding:14px;">
            <div style="margin-bottom:12px;">
                <span style="font-size:0.75rem;color:#94a3b8;">OVER Edge</span><br/>
                {edge_badge(edge_over)}
            </div>
            <div style="margin-bottom:12px;">
                <span style="font-size:0.75rem;color:#94a3b8;">UNDER Edge</span><br/>
                {edge_badge(edge_under)}
            </div>
            <div style="margin-bottom:12px;">
                <span style="font-size:0.75rem;color:#94a3b8;">Recommendation</span><br/>
                <span style="font-size:0.9rem;font-weight:600;color:{'#4ade80' if p_over > 0.57 else '#f87171' if p_under > 0.57 else '#94a3b8'};">
                    {'OVER ✓' if p_over > 0.57 else 'UNDER ✓' if p_under > 0.57 else 'PASS — No Edge'}
                </span>
            </div>
            <div>
                <span style="font-size:0.75rem;color:#94a3b8;">Confidence</span><br/>
                {conf_badge('high' if abs(delta) > 2 * proj_std else 'med' if abs(delta) > proj_std else 'low')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Flex play builder
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("**Flex Play Builder**")
    st.markdown("<div style='font-size:0.8rem;color:#94a3b8;margin-bottom:8px;'>Add 2-6 picks to calculate combo EV</div>", unsafe_allow_html=True)

    n_picks = st.slider("Number of Picks", 2, 6, 3, key="flex_picks")
    pick_probs = []
    pick_cols = st.columns(n_picks)
    for i in range(n_picks):
        with pick_cols[i]:
            prob_input = st.number_input(f"Pick {i+1} Win%", 40.0, 95.0, 58.0, 1.0, key=f"flex_p{i}")
            pick_probs.append(prob_input / 100.0)

    if st.button("Calculate Flex EV", key="calc_flex"):
        pp_result = pp_combo_ev(pick_probs, "power_play")
        flex_result = pp_combo_ev(pick_probs, "flex_play")

        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.markdown(f"""
            <div class="glass-card" style="padding:14px;">
                <div style="font-size:0.85rem;font-weight:600;color:#4ade80;margin-bottom:8px;">Power Play ({n_picks} picks)</div>
                <div style="font-size:0.75rem;color:#94a3b8;">Win Prob: <span class="mono text-green">{pp_result['win_prob']*100:.1f}%</span></div>
                <div style="font-size:0.75rem;color:#94a3b8;">Payout: <span class="mono text-blue">{pp_result['total_return']:.1f}x</span></div>
                <div style="font-size:0.75rem;color:#94a3b8;">EV: {edge_badge(pp_result['edge'])}</div>
            </div>
            """, unsafe_allow_html=True)
        with rcol2:
            st.markdown(f"""
            <div class="glass-card" style="padding:14px;">
                <div style="font-size:0.85rem;font-weight:600;color:#60a5fa;margin-bottom:8px;">Flex Play ({n_picks} picks)</div>
                <div style="font-size:0.75rem;color:#94a3b8;">Best EV: <span class="mono text-green">{flex_result['edge']*100:+.1f}%</span></div>
                <div style="font-size:0.75rem;color:#94a3b8;">Payout: <span class="mono text-blue">{flex_result['total_return']:.1f}x</span></div>
                <div style="font-size:0.75rem;color:#94a3b8;">Play Type: <span class="mono">{flex_result['play_type']}</span></div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# TAB 6 — MONTE CARLO SIM
# ============================================================
def tab_monte_carlo(proj_df: pd.DataFrame, settings: dict):
    """Render the Monte Carlo Simulation tab."""
    st.markdown(section_header("Monte Carlo Simulator", "&#127922;", f"{settings['n_sims']:,} sims"), unsafe_allow_html=True)

    sim_player = st.selectbox("Select Player", proj_df["player"].tolist(), key="mc_player")
    p = proj_df[proj_df["player"] == sim_player].iloc[0]
    field_sg = proj_df["sg_regressed"].tolist()

    if st.button("Run Simulation", key="run_mc"):
        with st.spinner(f"Running {settings['n_sims']:,} Monte Carlo simulations..."):
            mc = monte_carlo_win_prob(
                p["sg_regressed"], field_sg,
                n_sims=settings["n_sims"], volatility=0.8,
            )

        # Results
        cols = st.columns(4)
        with cols[0]:
            st.markdown(metric_card("Win Prob", f"{mc['win_prob']*100:.2f}%", f"{settings['n_sims']:,} sims", "positive", "green"), unsafe_allow_html=True)
        with cols[1]:
            st.markdown(metric_card("Avg Finish", f"{mc['avg_finish']:.1f}", "position", "neutral", "blue"), unsafe_allow_html=True)
        with cols[2]:
            st.markdown(metric_card("Median Finish", f"{mc['median_finish']:.0f}", "position", "neutral", "amber"), unsafe_allow_html=True)
        with cols[3]:
            top10_pct = mc["top_n_probs"].get(10, 0)
            st.markdown(metric_card("Top 10", f"{top10_pct*100:.1f}%", "probability", "positive" if top10_pct > 0.3 else "neutral", "green"), unsafe_allow_html=True)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # Finish distribution histogram
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Finish Position Distribution**")
            finishes = mc["finish_distribution"]
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=finishes,
                nbinsx=min(len(field_sg), 50),
                marker_color="#4ade80",
                opacity=0.7,
            ))
            fig.add_vline(x=mc["avg_finish"], line_dash="dash", line_color="#fbbf24",
                         annotation_text=f"Avg: {mc['avg_finish']:.1f}")
            fig.update_layout(
                height=350,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", family="Inter"),
                xaxis=dict(title="Finish Position", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.06)"),
                margin=dict(l=50, r=20, t=10, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Top-N Probabilities**")
            for n_val, prob in sorted(mc["top_n_probs"].items()):
                color = "green" if prob > 0.25 else "blue" if prob > 0.1 else "amber"
                st.markdown(prob_bar_html(prob, color, f"Top {n_val}"), unsafe_allow_html=True)

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            st.markdown("**Simulation Parameters**")
            st.markdown(f"""
            <div class="glass-card" style="padding:10px;font-size:0.75rem;color:#94a3b8;">
                <div>Input SG: <span class="mono text-green">{p['sg_regressed']:.2f}</span></div>
                <div>Field Size: <span class="mono text-blue">{len(field_sg)}</span></div>
                <div>Volatility: <span class="mono">0.80</span></div>
                <div>Simulations: <span class="mono text-amber">{settings['n_sims']:,}</span></div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Click 'Run Simulation' to execute Monte Carlo analysis.")


# ============================================================
# TAB BONUS — AI SLATE BRIEFING
# ============================================================
def tab_ai_briefing(proj_df: pd.DataFrame, settings: dict):
    """Render the AI Slate Briefing tab."""
    st.markdown(section_header("AI Slate Briefing", "&#129302;", "Claude-Powered"), unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="padding:14px;margin-bottom:16px;">
        <div style="font-size:0.85rem;color:#e2e8f0;">
            Generate a comprehensive AI-powered slate briefing using Claude Sonnet.
            The briefing analyzes all player edges, course fits, and betting opportunities.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top edges summary
    top_edges = proj_df.nlargest(5, "edge")
    st.markdown("**Top 5 Edges (Input to AI)**")
    for _, r in top_edges.iterrows():
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
            <span style="font-size:0.85rem;color:#e2e8f0;">{r['player']}</span>
            <div style="display:flex;gap:12px;align-items:center;">
                <span class="mono" style="font-size:0.8rem;color:#4ade80;">SG {r['sg_regressed']:.2f}</span>
                {edge_badge(r['edge'])}
                <span class="mono" style="font-size:0.8rem;color:#94a3b8;">+{r['odds']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    if st.button("Generate AI Briefing", key="ai_briefing"):
        with st.spinner("Claude Sonnet is generating your slate briefing..."):
            # Prepare edges JSON
            edges_list = []
            for _, r in proj_df.nlargest(15, "edge").iterrows():
                edges_list.append({
                    "player": r["player"],
                    "sg_regressed": round(r["sg_regressed"], 2),
                    "win_prob": round(r["win_prob"], 4),
                    "implied_prob": round(r["implied_prob"], 4),
                    "edge": round(r["edge"], 4),
                    "odds": int(r["odds"]),
                    "course_delta": round(r["course_delta"], 2),
                    "kelly": round(r["kelly"], 3),
                })
            edges_json = json.dumps({
                "course": settings["course"],
                "field_size": len(proj_df),
                "edges": edges_list,
            })

            try:
                briefing = ai_slate_briefing(edges_json)
                st.markdown(f"""
                <div class="glass-card" style="padding:20px;">
                    <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
                        &#129302; Claude AI Slate Briefing — {settings['course']}
                    </div>
                    <div style="font-size:0.85rem;color:#e2e8f0;line-height:1.7;white-space:pre-wrap;">{briefing}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"AI briefing unavailable: {e}")
    else:
        st.markdown("""
        <div style="text-align:center;padding:40px;color:#64748b;">
            Click "Generate AI Briefing" to get Claude's analysis of the current slate.
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    """Main entry point for the Golf Quant Engine dashboard."""
    settings = render_sidebar()

    # Load data
    if settings["data_mode"] == "Upload CSV" and settings["upload"] is not None:
        try:
            raw_df = pd.read_csv(settings["upload"])
            required = {"player", "sg_ott", "sg_app", "sg_arg", "sg_putt", "events", "odds"}
            if not required.issubset(set(raw_df.columns)):
                st.error(f"CSV missing columns: {required - set(raw_df.columns)}")
                return
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return
    else:
        raw_df = _generate_sample_players(30, settings["course"])

    # Build projections (cached)
    cache_key = hashlib.md5(
        f"{raw_df.to_json()}{settings['course']}{settings['n_sims']}".encode()
    ).hexdigest()

    if "proj_cache_key" not in st.session_state or st.session_state["proj_cache_key"] != cache_key:
        with st.spinner("Computing projections..."):
            proj_df = _build_projection_table(raw_df, settings["course"])
            st.session_state["proj_df"] = proj_df
            st.session_state["proj_cache_key"] = cache_key
    else:
        proj_df = st.session_state["proj_df"]

    # Tab layout
    tabs = st.tabs([
        "&#127942; Power Rankings",
        "&#128269; Player Deep Dive",
        "&#127959; Course Fit",
        "&#128176; Betting Edge",
        "&#127183; PrizePicks Lab",
        "&#127922; Monte Carlo",
        "&#129302; AI Briefing",
    ])

    with tabs[0]:
        tab_power_rankings(proj_df, settings)
    with tabs[1]:
        tab_player_deep_dive(proj_df, settings)
    with tabs[2]:
        tab_course_fit(proj_df, settings)
    with tabs[3]:
        tab_betting_edge(proj_df, settings)
    with tabs[4]:
        tab_prizepicks(proj_df, settings)
    with tabs[5]:
        tab_monte_carlo(proj_df, settings)
    with tabs[6]:
        tab_ai_briefing(proj_df, settings)


if __name__ == "__main__":
    main()
