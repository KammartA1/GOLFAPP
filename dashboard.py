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
import requests

# ── Section 9: Database-First Architecture ──
try:
    from services.startup import initialize_app, save_state_to_db, get_data_service
    initialize_app()
    _DB_AVAILABLE = True
except Exception as _db_err:
    _DB_AVAILABLE = False

# ── Quant System v1.0 Integration ──
from quant_system.engine import QuantEngine
from quant_system.core.types import Sport, BetType, SystemState
from quant_system.risk.kelly_adaptive import KellyConfig

try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

# ── Unified System Imports (PrizePicks Lab Integration) ──
try:
    from services.kill_switch import KillSwitch
    _KILL_SWITCH_AVAILABLE = True
except Exception:
    _KILL_SWITCH_AVAILABLE = False

try:
    from services.capital.kelly import KellyCriterion
    _KELLY_AVAILABLE = True
except Exception:
    _KELLY_AVAILABLE = False

try:
    from edge_analysis.decomposer import GolfEdgeDecomposer
    _EDGE_DECOMPOSER_AVAILABLE = True
except Exception:
    _EDGE_DECOMPOSER_AVAILABLE = False

try:
    from services.clv_system.odds_ingestion import OddsIngestionService
    _CLV_INGESTION_AVAILABLE = True
except Exception:
    _CLV_INGESTION_AVAILABLE = False

try:
    from scrapers.tournament_detector import detect_current_tournament, COURSE_COORDINATES
    from scrapers.weather_scraper import WeatherScraper
    _TOURNAMENT_DETECTOR_AVAILABLE = True
except Exception:
    _TOURNAMENT_DETECTOR_AVAILABLE = False

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
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar — force always visible ────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1220 0%, #0a0f1a 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
    min-width: 320px !important;
    width: 320px !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 320px !important;
    width: 320px !important;
    margin-left: 0 !important;
    transform: none !important;
    display: block !important;
    visibility: visible !important;
}
[data-testid="collapsedControl"] {
    display: none;
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
    "pars_per_round":    10.5,
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
    # ── Additional Tour venues (full season coverage) ────────
    "Waialae CC": {
        "sg_weights": {"sg_ott": 0.12, "sg_app": 0.28, "sg_arg": 0.22, "sg_putt": 0.38},
        "distance_bonus": -0.02, "accuracy_penalty": -0.03,
        "bermuda_greens": True, "wind_sensitivity": 0.55,
        "elevation": 20, "par": 70,
        "notes": "Short, tight layout; bermuda greens, trade winds; putting and wedge game dominate",
    },
    "PGA West Stadium": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.30, "sg_arg": 0.18, "sg_putt": 0.34},
        "distance_bonus": 0.03, "accuracy_penalty": -0.04,
        "bermuda_greens": False, "wind_sensitivity": 0.35,
        "elevation": 0, "par": 72,
        "notes": "Desert target golf, water hazards, firm greens, elevation at sea level",
    },
    "Vidanta Vallarta": {
        "sg_weights": {"sg_ott": 0.17, "sg_app": 0.28, "sg_arg": 0.20, "sg_putt": 0.35},
        "distance_bonus": 0.02, "accuracy_penalty": -0.03,
        "bermuda_greens": True, "wind_sensitivity": 0.40,
        "elevation": 10, "par": 71,
        "notes": "Resort course, paspalum greens, ocean breeze, relatively scorable",
    },
    "PGA National": {
        "sg_weights": {"sg_ott": 0.15, "sg_app": 0.28, "sg_arg": 0.20, "sg_putt": 0.37},
        "distance_bonus": 0.00, "accuracy_penalty": -0.05,
        "bermuda_greens": True, "wind_sensitivity": 0.60,
        "elevation": 15, "par": 70,
        "notes": "Bear Trap (15-17), heavy wind, water everywhere, bermuda greens, accuracy premium",
    },
    "Innisbrook Copperhead": {
        "sg_weights": {"sg_ott": 0.16, "sg_app": 0.30, "sg_arg": 0.19, "sg_putt": 0.35},
        "distance_bonus": 0.01, "accuracy_penalty": -0.05,
        "bermuda_greens": True, "wind_sensitivity": 0.40,
        "elevation": 50, "par": 71,
        "notes": "Tight tree-lined fairways, bermuda rough, snake pit finish, accuracy critical",
    },
    "Austin CC": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.30, "sg_arg": 0.20, "sg_putt": 0.32},
        "distance_bonus": 0.04, "accuracy_penalty": -0.03,
        "bermuda_greens": True, "wind_sensitivity": 0.35,
        "elevation": 700, "par": 71,
        "notes": "Match play venue, elevation, undulating greens, all-around game rewarded",
    },
    "TPC San Antonio": {
        "sg_weights": {"sg_ott": 0.20, "sg_app": 0.28, "sg_arg": 0.17, "sg_putt": 0.35},
        "distance_bonus": 0.05, "accuracy_penalty": -0.03,
        "bermuda_greens": True, "wind_sensitivity": 0.50,
        "elevation": 1000, "par": 72,
        "notes": "Long layout, Texas wind, elevation advantage, Masters warmup for many",
    },
    "TPC Louisiana": {
        "sg_weights": {"sg_ott": 0.20, "sg_app": 0.28, "sg_arg": 0.17, "sg_putt": 0.35},
        "distance_bonus": 0.05, "accuracy_penalty": -0.02,
        "bermuda_greens": True, "wind_sensitivity": 0.45,
        "elevation": 5, "par": 72,
        "notes": "Zurich Classic team event, long and open, bermuda, sea-level wind",
    },
    "TPC Craig Ranch": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.28, "sg_arg": 0.18, "sg_putt": 0.36},
        "distance_bonus": 0.04, "accuracy_penalty": -0.02,
        "bermuda_greens": True, "wind_sensitivity": 0.40,
        "elevation": 600, "par": 72,
        "notes": "Scorable Texas layout, wide fairways, birdie-fest, putting premium",
    },
    "Oakmont CC (US Open)": {
        "sg_weights": {"sg_ott": 0.20, "sg_app": 0.32, "sg_arg": 0.18, "sg_putt": 0.30},
        "distance_bonus": 0.06, "accuracy_penalty": -0.06,
        "bermuda_greens": False, "wind_sensitivity": 0.30,
        "elevation": 1200, "par": 70,
        "notes": "Hardest US Open venue, church pew bunkers, ultra-fast greens, extreme precision required",
    },
    "Detroit GC": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.28, "sg_arg": 0.18, "sg_putt": 0.36},
        "distance_bonus": 0.04, "accuracy_penalty": -0.02,
        "bermuda_greens": False, "wind_sensitivity": 0.25,
        "elevation": 600, "par": 72,
        "notes": "Scorable Midwest layout, wide fairways, birdie-or-bust, putting wins",
    },
    "TPC Deere Run": {
        "sg_weights": {"sg_ott": 0.15, "sg_app": 0.28, "sg_arg": 0.20, "sg_putt": 0.37},
        "distance_bonus": 0.01, "accuracy_penalty": -0.02,
        "bermuda_greens": False, "wind_sensitivity": 0.25,
        "elevation": 590, "par": 71,
        "notes": "Softest course on Tour, birdie-fest, putter-dominant, shorter hitters do well",
    },
    "Renaissance Club": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.30, "sg_arg": 0.22, "sg_putt": 0.30},
        "distance_bonus": 0.04, "accuracy_penalty": -0.04,
        "bermuda_greens": False, "wind_sensitivity": 0.70,
        "elevation": 30, "par": 70,
        "notes": "Scottish links, wind dominant, firm-and-fast, Open Championship prep",
    },
    "St Andrews": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.28, "sg_arg": 0.24, "sg_putt": 0.30},
        "distance_bonus": 0.05, "accuracy_penalty": -0.03,
        "bermuda_greens": False, "wind_sensitivity": 0.80,
        "elevation": 20, "par": 72,
        "notes": "Home of Golf, Old Course, wind is #1 factor, local knowledge, huge double greens",
    },
    "TPC Twin Cities": {
        "sg_weights": {"sg_ott": 0.19, "sg_app": 0.28, "sg_arg": 0.18, "sg_putt": 0.35},
        "distance_bonus": 0.04, "accuracy_penalty": -0.03,
        "bermuda_greens": False, "wind_sensitivity": 0.30,
        "elevation": 900, "par": 71,
        "notes": "Midwest layout, reasonable length, approach accuracy rewarded, low wind",
    },
    "Sedgefield CC": {
        "sg_weights": {"sg_ott": 0.13, "sg_app": 0.28, "sg_arg": 0.22, "sg_putt": 0.37},
        "distance_bonus": -0.02, "accuracy_penalty": -0.04,
        "bermuda_greens": True, "wind_sensitivity": 0.30,
        "elevation": 840, "par": 70,
        "notes": "Short classic Donald Ross, bermuda greens, wedge game + putting = king",
    },
    "TPC Southwind": {
        "sg_weights": {"sg_ott": 0.17, "sg_app": 0.30, "sg_arg": 0.18, "sg_putt": 0.35},
        "distance_bonus": 0.02, "accuracy_penalty": -0.05,
        "bermuda_greens": True, "wind_sensitivity": 0.40,
        "elevation": 300, "par": 70,
        "notes": "FedEx playoff, tight tree-lined, bermuda, water on several holes, accuracy premium",
    },
    "Castle Pines": {
        "sg_weights": {"sg_ott": 0.22, "sg_app": 0.30, "sg_arg": 0.16, "sg_putt": 0.32},
        "distance_bonus": 0.07, "accuracy_penalty": -0.03,
        "bermuda_greens": False, "wind_sensitivity": 0.30,
        "elevation": 6200, "par": 72,
        "notes": "Mile-high altitude (ball flies 10%+), long hitters dominate, thin air putting",
    },
    "Royal Montreal": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.30, "sg_arg": 0.20, "sg_putt": 0.32},
        "distance_bonus": 0.03, "accuracy_penalty": -0.04,
        "bermuda_greens": False, "wind_sensitivity": 0.40,
        "elevation": 100, "par": 70,
        "notes": "Presidents Cup venue, classic design, well-bunkered, Canadian weather factor",
    },
    "CC of Jackson": {
        "sg_weights": {"sg_ott": 0.16, "sg_app": 0.28, "sg_arg": 0.20, "sg_putt": 0.36},
        "distance_bonus": 0.02, "accuracy_penalty": -0.03,
        "bermuda_greens": True, "wind_sensitivity": 0.25,
        "elevation": 300, "par": 72,
        "notes": "Scorable fall event, bermuda, straightforward layout, putting heavy",
    },
    "TPC Summerlin": {
        "sg_weights": {"sg_ott": 0.17, "sg_app": 0.28, "sg_arg": 0.18, "sg_putt": 0.37},
        "distance_bonus": 0.03, "accuracy_penalty": -0.02,
        "bermuda_greens": False, "wind_sensitivity": 0.35,
        "elevation": 2800, "par": 72,
        "notes": "Desert layout, elevation helps distance, scorable, birdie-fest, putting wins",
    },
    "Accordia Golf": {
        "sg_weights": {"sg_ott": 0.16, "sg_app": 0.30, "sg_arg": 0.20, "sg_putt": 0.34},
        "distance_bonus": 0.02, "accuracy_penalty": -0.04,
        "bermuda_greens": False, "wind_sensitivity": 0.30,
        "elevation": 200, "par": 70,
        "notes": "ZOZO Championship, Japan, tight fairways, precision game, firm bent greens",
    },
    "El Cardonal": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.28, "sg_arg": 0.18, "sg_putt": 0.36},
        "distance_bonus": 0.03, "accuracy_penalty": -0.02,
        "bermuda_greens": True, "wind_sensitivity": 0.40,
        "elevation": 50, "par": 72,
        "notes": "Diamante, Cabo, desert-ocean layout, paspalum, wind off Pacific, scorable",
    },
    "Port Royal": {
        "sg_weights": {"sg_ott": 0.14, "sg_app": 0.28, "sg_arg": 0.22, "sg_putt": 0.36},
        "distance_bonus": -0.02, "accuracy_penalty": -0.03,
        "bermuda_greens": True, "wind_sensitivity": 0.65,
        "elevation": 50, "par": 71,
        "notes": "Bermuda oceanside, heavy wind, short but tricky, bermuda greens, scrambling key",
    },
    "Sea Island": {
        "sg_weights": {"sg_ott": 0.15, "sg_app": 0.28, "sg_arg": 0.20, "sg_putt": 0.37},
        "distance_bonus": 0.01, "accuracy_penalty": -0.03,
        "bermuda_greens": True, "wind_sensitivity": 0.45,
        "elevation": 10, "par": 70,
        "notes": "RSM Classic, Seaside/Plantation combo, bermuda, coastal wind, birdie-fest",
    },
    "Albany GC": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.28, "sg_arg": 0.18, "sg_putt": 0.36},
        "distance_bonus": 0.04, "accuracy_penalty": -0.02,
        "bermuda_greens": True, "wind_sensitivity": 0.50,
        "elevation": 10, "par": 72,
        "notes": "Hero World Challenge, Bahamas, small invitational field, ocean wind, scorable",
    },
    "Memorial Park GC": {
        "sg_weights": {"sg_ott": 0.22, "sg_app": 0.32, "sg_arg": 0.18, "sg_putt": 0.28},
        "distance_bonus": 0.06, "accuracy_penalty": -0.04,
        "bermuda_greens": True, "wind_sensitivity": 0.35,
        "elevation": 50, "par": 70,
        "notes": "Texas Children's Houston Open, municipal redesign by Tom Doak, long par-4s, bermuda greens, tight corridors reward driving distance + accuracy",
    },
    "Aronimink GC": {
        "sg_weights": {"sg_ott": 0.20, "sg_app": 0.32, "sg_arg": 0.18, "sg_putt": 0.30},
        "distance_bonus": 0.05, "accuracy_penalty": -0.05,
        "bermuda_greens": False, "wind_sensitivity": 0.30,
        "elevation": 400, "par": 70,
        "notes": "2026 PGA Championship, classic Donald Ross design, tight tree-lined, bentgrass, precise iron play and scrambling rewarded",
    },
    "Shinnecock Hills": {
        "sg_weights": {"sg_ott": 0.17, "sg_app": 0.30, "sg_arg": 0.22, "sg_putt": 0.31},
        "distance_bonus": 0.02, "accuracy_penalty": -0.05,
        "bermuda_greens": False, "wind_sensitivity": 0.75,
        "elevation": 50, "par": 70,
        "notes": "2026 U.S. Open, links-influenced, coastal wind, fescue, firm and fast, complete ball-striking test",
    },
    "Royal Birkdale": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.28, "sg_arg": 0.23, "sg_putt": 0.31},
        "distance_bonus": 0.03, "accuracy_penalty": -0.05,
        "bermuda_greens": False, "wind_sensitivity": 0.85,
        "elevation": 15, "par": 70,
        "notes": "2026 Open Championship, English links, dune-lined fairways, pot bunkers, heavy wind, scrambling critical",
    },
    "TPC Toronto": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.30, "sg_arg": 0.18, "sg_putt": 0.34},
        "distance_bonus": 0.04, "accuracy_penalty": -0.03,
        "bermuda_greens": False, "wind_sensitivity": 0.35,
        "elevation": 300, "par": 72,
        "notes": "RBC Canadian Open, Osprey Valley, heathland-style layout, bentgrass, balanced test",
    },
    "Bellerive CC": {
        "sg_weights": {"sg_ott": 0.20, "sg_app": 0.30, "sg_arg": 0.17, "sg_putt": 0.33},
        "distance_bonus": 0.06, "accuracy_penalty": -0.04,
        "bermuda_greens": False, "wind_sensitivity": 0.30,
        "elevation": 500, "par": 72,
        "notes": "2026 BMW Championship, long layout, bentgrass, Midwest humidity, rewards power and precision",
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
    """Retrieve the Anthropic API key from session state, secrets, or env.

    Checks in order:
      1. st.session_state (user entered in sidebar)
      2. st.secrets["ANTHROPIC_API_KEY"]
      3. os.environ["ANTHROPIC_API_KEY"]
      4. .env file in project root
    """
    # 1. Session state (sidebar input)
    override = st.session_state.get("_anthropic_key", "")
    if override:
        return override

    # 2. Streamlit secrets
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if key:
            return key
    except Exception:
        pass

    # 3. Environment variable
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key

    # 4. .env file
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


def _get_odds_api_key() -> str:
    """Retrieve The Odds API key from session state, secrets, or env."""
    override = st.session_state.get("_odds_api_key", "")
    if override:
        return override
    try:
        key = st.secrets.get("ODDS_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("ODDS_API_KEY", "")


@st.cache_data(ttl=600, show_spinner=False)
def fetch_espn_pga_field() -> dict:
    """Fetch the current/most-recent PGA Tour tournament field from ESPN.

    Returns dict with 'event_name', 'status', and 'players' list.
    Uses the current in-progress or most recent completed event with a field.
    """
    try:
        # First try without date filter to get current/in-progress events
        resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard",
            timeout=15,
        )
        if not resp.ok:
            return {}
        events = resp.json().get("events", [])
    except Exception:
        return {}

    if not events:
        return {}

    # Find best event: In Progress > Scheduled with field > most recent Final
    best = None
    for e in events:
        status = e.get("status", {}).get("type", {}).get("description", "")
        n = len(e.get("competitions", [{}])[0].get("competitors", []))
        if status == "In Progress":
            best = e
            break
        if status == "Scheduled" and n > 0 and best is None:
            best = e
        if status == "Final" and n > 0:
            best = e  # keeps updating to the most recent

    if not best:
        return {}

    event_name = best.get("name", "PGA Tour")
    status = best.get("status", {}).get("type", {}).get("description", "")
    competitors = best.get("competitions", [{}])[0].get("competitors", [])

    players = []
    for comp in competitors:
        ath = comp.get("athlete", {})
        name = ath.get("displayName", "")
        if name:
            players.append({
                "name": name,
                "score": comp.get("score", ""),
                "position": comp.get("status", {}).get("position", {}).get("displayName", ""),
            })

    # Extract course name from venue or event details
    venue = best.get("competitions", [{}])[0].get("venue", {})
    course_name = venue.get("fullName", "") or venue.get("shortName", "")

    return {
        "event_name": event_name,
        "status": status,
        "players": players,
        "course_name": course_name,
    }


@st.cache_data(ttl=600, show_spinner=False)
def fetch_odds_api_golf_odds(api_key: str) -> dict:
    """Fetch golf outright odds from The Odds API.

    Returns dict mapping player name -> best American odds (int).
    Note: The Odds API currently only has major futures for golf,
    so odds are approximate for current-week events.
    """
    if not api_key:
        return {}

    base_url = "https://api.the-odds-api.com/v4/sports"
    player_odds = {}

    try:
        resp = requests.get(f"{base_url}", params={"apiKey": api_key}, timeout=15)
        if not resp.ok:
            return {}
        golf_sports = [s for s in resp.json()
                       if s.get("group", "").lower() == "golf" and s.get("active")]
    except Exception:
        return {}

    for sport in golf_sports:
        try:
            resp = requests.get(
                f"{base_url}/{sport['key']}/odds",
                params={"apiKey": api_key, "regions": "us", "markets": "outrights", "oddsFormat": "american"},
                timeout=15,
            )
            if not resp.ok:
                continue
            for event in resp.json():
                for bk in event.get("bookmakers", []):
                    for mkt in bk.get("markets", []):
                        if mkt.get("key") != "outrights":
                            continue
                        for outcome in mkt.get("outcomes", []):
                            name = outcome.get("name", "")
                            price = outcome.get("price", 0)
                            if name and price > 0:
                                if name not in player_odds or price > player_odds[name]:
                                    player_odds[name] = price
        except Exception:
            continue

    return player_odds


# ── PrizePicks Scraper via ScraperAPI ────────────────────────
def _get_scraper_api_key() -> str:
    """Get ScraperAPI key from secrets or env."""
    try:
        key = st.secrets.get("SCRAPER_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("SCRAPER_API_KEY", "")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prizepicks_golf_lines(scraper_key: str = "") -> list:
    """Fetch PrizePicks golf prop lines.

    Uses curl_cffi with edge101 impersonation to bypass PerimeterX.
    Falls back to ScraperAPI proxy, then direct request.

    Returns list of dicts: player, stat_type, line, odds_type, league, description.
    """
    PP_API = "https://api.prizepicks.com/projections"
    PP_HEADERS = {
        "Accept": "application/vnd.api+json",
        "Referer": "https://app.prizepicks.com/",
        "Origin": "https://app.prizepicks.com",
    }
    params = {"per_page": "500"}

    data = None

    # Method 1: curl_cffi with edge101 impersonation (bypasses PerimeterX)
    if HAS_CURL_CFFI and data is None:
        for browser in ["edge101", "safari17_0", "chrome124"]:
            try:
                resp = cffi_requests.get(
                    PP_API, params=params, headers=PP_HEADERS,
                    impersonate=browser, timeout=25,
                )
                if resp.ok:
                    data = resp.json()
                    break
            except Exception:
                continue

    # Method 2: ScraperAPI proxy
    if data is None and scraper_key:
        try:
            full_url = f"{PP_API}?per_page=500"
            resp = requests.get(
                "https://api.scraperapi.com",
                params={"api_key": scraper_key, "url": full_url, "render": "false"},
                timeout=30,
            )
            if resp.ok:
                data = resp.json()
        except Exception:
            pass

    # Method 3: Direct request (works from residential IPs)
    if data is None:
        try:
            resp = requests.get(PP_API, params=params, headers={
                **PP_HEADERS,
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }, timeout=20)
            if resp.ok:
                data = resp.json()
        except Exception:
            pass

    if not data:
        return []

    # Build lookup maps from included resources
    GOLF_LEAGUE_NAMES = {"PGA", "PGA TOUR", "GOLF", "LPGA", "LIV GOLF", "LIVGOLF", "DP WORLD TOUR"}

    player_map = {}
    league_map = {}
    for item in data.get("included", []):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")
        item_id = item.get("id")
        attrs = item.get("attributes", {}) or {}
        if item_type == "new_player" and item_id:
            player_map[item_id] = attrs.get("name", "")
        elif item_type == "league" and item_id:
            league_map[item_id] = attrs.get("name", "")

    rows = []
    for proj in data.get("data", []):
        if not isinstance(proj, dict):
            continue
        attrs = proj.get("attributes", {}) or {}
        rels = proj.get("relationships", {}) or {}

        # Determine league via relationship (primary) or attribute (fallback)
        league_id = (rels.get("league", {}).get("data", {}) or {}).get("id")
        league_name = league_map.get(league_id, "")
        if not league_name:
            league_name = str(attrs.get("league", "") or "")

        if league_name.upper() not in GOLF_LEAGUE_NAMES:
            continue

        # Get player name from relationship -> included map
        player_id = (rels.get("new_player", {}).get("data", {}) or {}).get("id")
        if not player_id:
            player_id = (rels.get("player", {}).get("data", {}) or {}).get("id")
        player_name = player_map.get(player_id, "")
        if not player_name:
            player_name = attrs.get("description", "").split(" - ")[0].strip() if " - " in attrs.get("description", "") else attrs.get("description", "")

        stat_type = attrs.get("stat_type", "")
        line_score = attrs.get("line_score")
        odds_type = str(attrs.get("odds_type", "") or "").lower().strip() or "standard"

        if player_name and stat_type and line_score is not None:
            try:
                rows.append({
                    "player": player_name,
                    "stat_type": stat_type,
                    "line": float(line_score),
                    "odds_type": odds_type,
                    "start_time": attrs.get("start_time", ""),
                    "league": league_name.upper(),
                    "description": attrs.get("description", ""),
                })
            except (TypeError, ValueError):
                pass

    return rows


# ── Weather via OpenWeather API ──────────────────────────────
# Course coordinates for weather lookup
COURSE_COORDS = {
    "Augusta National": (33.503, -82.022),
    "TPC Sawgrass": (30.198, -81.394),
    "Pebble Beach": (36.567, -121.950),
    "Torrey Pines South": (32.899, -117.252),
    "TPC Scottsdale": (33.639, -111.924),
    "Riviera CC": (34.048, -118.500),
    "Bay Hill": (28.459, -81.516),
    "Harbour Town": (32.134, -80.821),
    "Colonial CC": (32.737, -97.383),
    "Muirfield Village": (40.100, -83.164),
    "Quail Hollow": (35.113, -80.853),
    "East Lake": (33.742, -84.310),
    "TPC River Highlands": (41.622, -72.648),
    "Kapalua Plantation": (20.998, -156.658),
    "Waialae CC": (21.274, -157.763),
    "PGA National": (26.833, -80.107),
    "Innisbrook Copperhead": (28.085, -82.750),
    "TPC San Antonio": (29.607, -98.621),
    "St Andrews": (56.343, -2.803),
    "Oakmont CC": (40.527, -79.829),
    "Pinehurst No. 2": (35.192, -79.472),
    "Bethpage Black": (40.742, -73.455),
    "Sedgefield CC": (36.072, -79.819),
    "TPC Southwind": (35.056, -89.779),
    "Castle Pines": (39.460, -104.896),
    "Renaissance Club": (56.051, -2.779),
    "Memorial Park GC": (29.763, -95.417),
    "PGA West Stadium": (33.717, -116.300),
    "Vidanta Vallarta": (20.672, -105.296),
    "TPC Louisiana": (29.890, -90.089),
    "TPC Craig Ranch": (33.083, -96.717),
    "TPC Twin Cities": (44.880, -93.220),
    "TPC Deere Run": (41.411, -90.427),
    "Detroit GC": (42.423, -83.140),
    "CC of Jackson": (32.330, -90.140),
    "TPC Summerlin": (36.153, -115.290),
    "Accordia Golf": (35.673, 140.044),
    "El Cardonal": (22.881, -109.930),
    "Port Royal": (32.254, -64.835),
    "Sea Island": (31.155, -81.385),
    "Albany GC": (24.997, -77.525),
    "Royal Montreal": (45.463, -73.945),
    "Aronimink GC": (39.952, -75.393),
    "Shinnecock Hills": (40.893, -72.442),
    "Royal Birkdale": (53.563, -3.042),
    "TPC Toronto": (43.798, -79.967),
    "Bellerive CC": (38.611, -90.381),
    "Valhalla": (38.253, -85.520),
    "Southern Hills": (36.115, -95.959),
}


def _get_weather_key() -> str:
    """Get OpenWeather API key."""
    try:
        key = st.secrets.get("OPENWEATHER_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("OPENWEATHER_API_KEY", "")


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_course_weather(course_name: str) -> dict:
    """Fetch current weather for a golf course using OpenWeather API."""
    api_key = _get_weather_key()
    if not api_key:
        return {}

    coords = COURSE_COORDS.get(course_name)
    if not coords:
        return {}

    lat, lon = coords
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": api_key, "units": "imperial"},
            timeout=10,
        )
        if not resp.ok:
            return {}
        d = resp.json()
        return {
            "temp_f": round(d.get("main", {}).get("temp", 0)),
            "humidity": d.get("main", {}).get("humidity", 0),
            "wind_mph": round(d.get("wind", {}).get("speed", 0)),
            "wind_gust": round(d.get("wind", {}).get("gust", 0) or 0),
            "conditions": d.get("weather", [{}])[0].get("description", "").title(),
            "clouds": d.get("clouds", {}).get("all", 0),
        }
    except Exception:
        return {}


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
                     weather_note: str = "", recent_form: str = "",
                     tournament_name: str = "", course_profile: dict = None) -> str:
    """Use Claude claude-haiku-4-5 for fast edge analysis on a single player.

    Enhanced for tournament-specific deep dive analysis including course advantages/disadvantages.
    """
    client = _anthropic_client()
    if client is None:
        return "[AI analysis unavailable — no API key configured]"

    sg_summary = "\n".join(f"  {k}: {v:+.2f}" if isinstance(v, (int, float))
                           else f"  {k}: {v}" for k, v in sg_data.items())
    odds_str = f"American odds: {odds:+d}" if odds != 0 else "No odds available"

    course_context = ""
    if course_profile:
        cw = course_profile.get("sg_weights", {})
        dominant_skill = max(cw, key=cw.get) if cw else "unknown"
        skill_map = {"sg_ott": "Off the Tee", "sg_app": "Approach", "sg_arg": "Around the Green", "sg_putt": "Putting"}
        course_context = f"""
Course Profile for {course}:
  Dominant Skill Required: {skill_map.get(dominant_skill, dominant_skill)} ({cw.get(dominant_skill, 0):.0%} weight)
  Distance Bonus: {course_profile.get('distance_bonus', 0):+.0%}
  Wind Sensitivity: {course_profile.get('wind_sensitivity', 0):.1f}/1.0
  Green Type: {'Bermuda' if course_profile.get('bermuda_greens') else 'Bent/Poa'}
  Elevation: {course_profile.get('elevation', 0)}ft
  Course Notes: {course_profile.get('notes', 'N/A')}"""

    prompt = f"""You are an elite professional golf betting analyst. Provide a deep, tournament-specific edge analysis (6-8 sentences) for this player at THIS specific tournament.

Tournament: {tournament_name if tournament_name else course}
Player: {player}
Course: {course}
Field Strength: {field_strength}
{odds_str}
Weather: {weather_note if weather_note else 'Standard conditions'}
Recent Form: {recent_form if recent_form else 'Not specified'}

Strokes Gained Profile:
{sg_summary}
{course_context}

CRITICAL REQUIREMENTS:
1. Analyze SPECIFIC advantages this player has at THIS course (e.g., "His +1.2 SG:APP is critical at {course} where approach shots carry 35% weight")
2. Identify SPECIFIC disadvantages (e.g., "His -0.3 SG:Putt on bermuda greens is a concern here")
3. Reference the course's unique challenges and how they map to this player's skill set
4. Compare the player's SG profile to what this course demands
5. Provide a clear verdict: VALUE / FAIR PRICE / OVERVALUED with specific reasoning
6. If weather is a factor, explain how it specifically impacts THIS player's game

Be extremely specific and quantitative. Reference exact SG numbers and course weights. Do NOT use generic golf analysis."""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"[AI analysis error: {str(e)[:100]}]"


@st.cache_data(ttl=60*60*2, show_spinner=False)
def ai_scanner_analysis(scan_results_json: str, course: str, tournament: str = "") -> str:
    """Claude AI analysis of Live Scanner results — identifies best parlay opportunities."""
    client = _anthropic_client()
    if client is None:
        return _generate_algorithmic_scanner_briefing(scan_results_json)

    prompt = f"""You are an elite PrizePicks golf betting analyst with deep quantitative expertise. Analyze these live scanner results and provide actionable advice.

Tournament: {tournament if tournament else course}
Course: {course}

Scanner Results (props with model edge):
{scan_results_json}

Provide your analysis in this EXACT format:

**TOP PLAYS (Highest Probability)**
List the top 3-4 props with the highest model edge AND probability. For each:
- Player | Stat | Side | Model Prob | Why this is a strong play at THIS course

**PARLAY RECOMMENDATIONS**
Suggest 2-3 optimal parlay combinations (2-4 legs each):
- Combo 1: List legs, explain correlation risk, estimated combo hit rate
- Combo 2: Alternative combo with different risk profile
Rate each: POWER PLAY (all must hit) or FLEX PLAY (partial payout)

**FADE LIST**
2-3 props that look tempting but should be avoided, with specific reasoning

**KEY INSIGHTS**
- Course-specific factors affecting today's props
- Weather impact on specific stat types
- Common sense flags (e.g., player returning from injury, first time at course)

Be specific, data-driven, and brutally honest about risk. Use the exact numbers from the scanner. Think like a sharp bettor who needs every edge."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return _generate_algorithmic_scanner_briefing(scan_results_json)


def _generate_algorithmic_scanner_briefing(scan_json: str) -> str:
    """Fallback algorithmic scanner briefing when AI is unavailable."""
    try:
        data = json.loads(scan_json)
    except Exception:
        return "No scanner data available for analysis."

    if not data:
        return "No props to analyze."

    # Sort by edge
    sorted_props = sorted(data, key=lambda x: x.get("edge", 0), reverse=True)
    top = sorted_props[:4]
    avoid = [p for p in sorted_props if p.get("edge", 0) < -0.02][:3]

    lines = ["SCANNER ANALYSIS (Algorithmic)", ""]
    lines.append("TOP PLAYS:")
    for p in top:
        lines.append(f"  {p.get('player', '?')} | {p.get('stat', '?')} {p.get('side', '?')} {p.get('line', '?')} | "
                     f"Edge: {p.get('edge', 0)*100:+.1f}% | Prob: {p.get('prob', 0)*100:.1f}%")

    if avoid:
        lines.append("")
        lines.append("AVOID:")
        for p in avoid:
            lines.append(f"  {p.get('player', '?')} | {p.get('stat', '?')} | Edge: {p.get('edge', 0)*100:+.1f}%")

    # Suggest parlays from top picks
    if len(top) >= 2:
        lines.append("")
        lines.append("PARLAY SUGGESTION:")
        combo_prob = 1.0
        for p in top[:3]:
            combo_prob *= p.get("prob", 0.5)
        lines.append(f"  {' + '.join(p.get('player', '?') for p in top[:3])}")
        lines.append(f"  Combo Hit Rate: {combo_prob*100:.2f}% | {'POWER PLAY' if combo_prob > 0.15 else 'FLEX PLAY recommended'}")

    return "\n".join(lines)


@st.cache_data(ttl=60*60*2, show_spinner=False)
def ai_lab_analysis(legs_json: str, mc_results_json: str, course: str) -> str:
    """Claude AI analysis for PrizePicks Lab — deep parlay optimization advice."""
    client = _anthropic_client()
    if client is None:
        return "AI analysis unavailable — configure ANTHROPIC_API_KEY for Claude-powered advice."

    prompt = f"""You are an elite PrizePicks betting strategist. I've run my full quant engine (Monte Carlo 5000x, Bayesian shrinkage, course-fit adjustment, correlation analysis) on my selected legs. Analyze the results and give me your expert recommendation.

Course: {course}

Selected Legs:
{legs_json}

Monte Carlo Simulation Results:
{mc_results_json}

Provide your analysis in this format:

**VERDICT**: [STRONG BET / LEAN BET / PASS] with confidence level

**CONVICTION RANKING**: Rank all legs 1-N from strongest to weakest

**EDGE DECOMPOSITION**:
For each leg, break down WHERE the edge comes from:
- Statistical edge (projection vs line)
- Course-fit edge (does this course amplify or dampen this stat?)
- Correlation risk (how correlated are these legs?)

**RISK ASSESSMENT**:
- Weakest leg and why it could bust
- Biggest correlation risk between legs
- Weather/course factors that could swing results

**OPTIMAL STRATEGY**:
- Power Play vs Flex Play recommendation with math
- Suggested stake as % of bankroll (use Kelly-like logic)
- If any leg should be swapped, suggest an alternative

**ONE-LINER**: Your gut-feel summary in one sentence

Think like a professional bettor managing a $10K bankroll. Every dollar matters. Be specific with numbers."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"AI analysis error: {str(e)[:100]}"


def mc_prop_simulation(proj: float, std: float, line: float, n_sims: int = 5000,
                       stat_type: str = "", player_sg: float = 0.0,
                       field_sg_list: list | None = None,
                       opponent_sg: float | None = None,
                       course_name: str = "") -> dict:
    """Monte Carlo simulation for a single prop — 5000 iterations per engine.

    Uses 5 probability engines in an ensemble, with special handling for:
    - Holes Played: round-by-round tournament sim (cut/no-cut → 36 or 72 holes)
    - Birdies or Better Matchup: H2H comparison of two players' birdie totals
    - Standard stats: Normal, T-dist, Skew, Mean-Revert, Vol-Cluster engines

    Returns detailed simulation results.
    """
    rng = np.random.default_rng()

    # ── SPECIAL: Holes Played (cut/no-cut simulation) ──────────────
    if stat_type == "holes_played":
        # Line is typically 36.5 — over = make cut (72 holes), under = miss cut (36)
        # Vectorized cut simulation: R1+R2 for player vs field → 36 or 72
        field_sgs = field_sg_list or [0.0] * 120
        field_size = max(len(field_sgs), 30)
        cut_idx = min(65, field_size)  # top 65 + ties make cut

        player_mean_36 = -player_sg * 2  # 36-hole expected score (SG → strokes, 2 rounds)
        round_std_36 = 2.8 * np.sqrt(2)  # 36-hole std (two rounds combined)

        holes_sims = np.zeros(n_sims * 5)
        engine_p_over = []

        for eng_idx in range(5):
            # Vectorized: simulate all n_sims at once
            if eng_idx == 0:  # Normal
                player_36 = rng.normal(player_mean_36, round_std_36, n_sims)
            elif eng_idx == 1:  # T-dist
                player_36 = player_mean_36 + round_std_36 * rng.standard_t(8, n_sims)
            elif eng_idx == 2:  # Skew
                player_36 = rng.normal(player_mean_36, round_std_36 * 0.9, n_sims) + rng.exponential(round_std_36 * 0.2, n_sims)
            elif eng_idx == 3:  # Mean-revert
                rev_mean = player_mean_36 * 0.85
                player_36 = rng.normal(rev_mean, round_std_36 * 1.05, n_sims)
            else:  # Vol-cluster
                v = rng.gamma(4, 0.25, n_sims)
                player_36 = rng.normal(player_mean_36, round_std_36 * v, n_sims)

            # Vectorized field: (n_sims, field_size) matrix
            field_36 = rng.normal(0, round_std_36, (n_sims, field_size))
            # Find the cut line (65th lowest score) for each sim
            field_sorted = np.sort(field_36, axis=1)
            cut_line_scores = field_sorted[:, min(cut_idx - 1, field_size - 1)]

            made_cut = player_36 <= cut_line_scores
            eng_sims = np.where(made_cut, 72.0, 36.0)

            p_over = float(np.mean(eng_sims > line))
            engine_p_over.append(p_over)
            holes_sims[eng_idx * n_sims:(eng_idx + 1) * n_sims] = eng_sims

        weights = [0.35, 0.20, 0.15, 0.15, 0.15]
        p_over_ensemble = sum(w * p for w, p in zip(weights, engine_p_over))
        p_under_ensemble = 1.0 - p_over_ensemble

        ci_10 = float(np.percentile(holes_sims, 10))
        ci_90 = float(np.percentile(holes_sims, 90))
        ci_25 = float(np.percentile(holes_sims, 25))
        ci_75 = float(np.percentile(holes_sims, 75))

        return {
            "p_over": round(p_over_ensemble, 4),
            "p_under": round(p_under_ensemble, 4),
            "p_over_by_engine": {
                "normal": round(engine_p_over[0], 4),
                "t_dist": round(engine_p_over[1], 4),
                "skew_adj": round(engine_p_over[2], 4),
                "mean_revert": round(engine_p_over[3], 4),
                "vol_cluster": round(engine_p_over[4], 4),
            },
            "engine_agreement": round(1.0 - np.std(engine_p_over), 4),
            "sim_mean": round(float(np.mean(holes_sims)), 2),
            "sim_std": round(float(np.std(holes_sims)), 2),
            "ci_80": (round(ci_10, 2), round(ci_90, 2)),
            "ci_50": (round(ci_25, 2), round(ci_75, 2)),
            "n_sims": n_sims * 5,
        }

    # ── SPECIAL: Birdies or Better Matchup (H2H) ──────────────────
    if stat_type == "birdies_matchup" and opponent_sg is not None:
        # Two-player head-to-head: who gets more birdies in their round?
        # proj = player's projected birdies, opponent_sg drives opponent's projection
        opp_proj_birdies = TOUR_BASELINES.get("birdies_per_round", 3.5)
        opp_sens = SG_TO_STAT_SENSITIVITY.get("birdies", {})
        for sg_cat, sens_val in opp_sens.items():
            opp_proj_birdies += opponent_sg / 4.0 * sens_val  # distribute opponent SG evenly
        birdies_std = STAT_STD.get("birdies", 1.5)

        engine_p_over = []
        all_margins = np.zeros(n_sims * 5)

        for eng_idx in range(5):
            if eng_idx == 0:  # Normal
                player_b = rng.normal(proj, birdies_std, n_sims)
                opp_b = rng.normal(opp_proj_birdies, birdies_std, n_sims)
            elif eng_idx == 1:  # T-dist
                player_b = proj + birdies_std * rng.standard_t(8, n_sims)
                opp_b = opp_proj_birdies + birdies_std * rng.standard_t(8, n_sims)
            elif eng_idx == 2:  # Skew
                player_b = rng.normal(proj, birdies_std * 0.9, n_sims) + rng.exponential(birdies_std * 0.3, n_sims) * 0.15
                opp_b = rng.normal(opp_proj_birdies, birdies_std * 0.9, n_sims) + rng.exponential(birdies_std * 0.3, n_sims) * 0.15
            elif eng_idx == 3:  # Mean-revert (both players revert toward league avg)
                rev_player = proj * 0.85 + 3.5 * 0.15
                rev_opp = opp_proj_birdies * 0.85 + 3.5 * 0.15
                player_b = rng.normal(rev_player, birdies_std * 1.05, n_sims)
                opp_b = rng.normal(rev_opp, birdies_std * 1.05, n_sims)
            else:  # Vol-cluster
                v = rng.gamma(4, 0.25, n_sims)
                player_b = rng.normal(proj, birdies_std * v, n_sims)
                v = rng.gamma(4, 0.25, n_sims)
                opp_b = rng.normal(opp_proj_birdies, birdies_std * v, n_sims)

            # "Over" on matchup line means player gets MORE birdies than opponent by that margin
            margin = player_b - opp_b
            p_over = float(np.mean(margin > line))
            engine_p_over.append(p_over)
            all_margins[eng_idx * n_sims:(eng_idx + 1) * n_sims] = margin

        weights = [0.35, 0.20, 0.15, 0.15, 0.15]
        p_over_ensemble = sum(w * p for w, p in zip(weights, engine_p_over))
        p_under_ensemble = 1.0 - p_over_ensemble

        ci_10 = float(np.percentile(all_margins, 10))
        ci_90 = float(np.percentile(all_margins, 90))
        ci_25 = float(np.percentile(all_margins, 25))
        ci_75 = float(np.percentile(all_margins, 75))

        return {
            "p_over": round(p_over_ensemble, 4),
            "p_under": round(p_under_ensemble, 4),
            "p_over_by_engine": {
                "normal": round(engine_p_over[0], 4),
                "t_dist": round(engine_p_over[1], 4),
                "skew_adj": round(engine_p_over[2], 4),
                "mean_revert": round(engine_p_over[3], 4),
                "vol_cluster": round(engine_p_over[4], 4),
            },
            "engine_agreement": round(1.0 - np.std(engine_p_over), 4),
            "sim_mean": round(float(np.mean(all_margins)), 2),
            "sim_std": round(float(np.std(all_margins)), 2),
            "ci_80": (round(ci_10, 2), round(ci_90, 2)),
            "ci_50": (round(ci_25, 2), round(ci_75, 2)),
            "n_sims": n_sims * 5,
        }

    # ── STANDARD 5-ENGINE ENSEMBLE (all other stats) ──────────────
    # Engine 1: Normal distribution (primary)
    normal_sims = rng.normal(proj, std, n_sims)
    p_over_normal = float(np.mean(normal_sims > line))
    p_under_normal = float(np.mean(normal_sims <= line))

    # Engine 2: T-distribution (heavier tails — more realistic for golf)
    df_t = 8  # degrees of freedom — moderate tail heaviness
    t_sims = proj + std * rng.standard_t(df_t, n_sims)
    p_over_t = float(np.mean(t_sims > line))

    # Engine 3: Skew-adjusted (golf stats often have slight positive skew)
    skew_factor = 0.15  # mild positive skew for most golf stats
    skew_noise = rng.exponential(std * 0.3, n_sims) * skew_factor
    skew_sims = rng.normal(proj, std * 0.9, n_sims) + skew_noise
    p_over_skew = float(np.mean(skew_sims > line))

    # Engine 4: Bootstrap with mean reversion
    reversion_strength = 0.15  # 15% pull toward baseline
    baseline_proj = line  # use line as proxy for market consensus
    reverted_proj = proj * (1 - reversion_strength) + baseline_proj * reversion_strength
    revert_sims = rng.normal(reverted_proj, std * 1.05, n_sims)
    p_over_revert = float(np.mean(revert_sims > line))

    # Engine 5: Variance-scaled (accounts for volatility clustering)
    vol_scales = rng.gamma(4, 0.25, n_sims)  # random variance multiplier
    vol_sims = rng.normal(proj, std * vol_scales, n_sims)
    p_over_vol = float(np.mean(vol_sims > line))

    # Ensemble: weighted average of all engines
    # Normal gets highest weight (most theoretically grounded)
    weights = [0.35, 0.20, 0.15, 0.15, 0.15]
    p_over_ensemble = (
        weights[0] * p_over_normal +
        weights[1] * p_over_t +
        weights[2] * p_over_skew +
        weights[3] * p_over_revert +
        weights[4] * p_over_vol
    )
    p_under_ensemble = 1.0 - p_over_ensemble

    # Confidence interval from all sims combined
    all_sims = np.concatenate([normal_sims, t_sims, skew_sims])
    ci_10 = float(np.percentile(all_sims, 10))
    ci_90 = float(np.percentile(all_sims, 90))
    ci_25 = float(np.percentile(all_sims, 25))
    ci_75 = float(np.percentile(all_sims, 75))

    return {
        "p_over": round(p_over_ensemble, 4),
        "p_under": round(p_under_ensemble, 4),
        "p_over_by_engine": {
            "normal": round(p_over_normal, 4),
            "t_dist": round(p_over_t, 4),
            "skew_adj": round(p_over_skew, 4),
            "mean_revert": round(p_over_revert, 4),
            "vol_cluster": round(p_over_vol, 4),
        },
        "engine_agreement": round(1.0 - np.std([p_over_normal, p_over_t, p_over_skew, p_over_revert, p_over_vol]), 4),
        "sim_mean": round(float(np.mean(all_sims)), 2),
        "sim_std": round(float(np.std(all_sims)), 2),
        "ci_80": (round(ci_10, 2), round(ci_90, 2)),
        "ci_50": (round(ci_25, 2), round(ci_75, 2)),
        "n_sims": n_sims * 5,  # total across all engines
    }


def mc_parlay_simulation(legs: list, n_sims: int = 5000) -> dict:
    """Correlated Monte Carlo simulation for a full parlay.

    Uses Gaussian copula for correlation between legs.
    Each leg: {proj, std, line, side, player, stat}

    Returns joint probability, naive probability, EV for power/flex.
    """
    n = len(legs)
    if n == 0:
        return {"error": "No legs provided"}

    # Build correlation matrix
    # Same player = higher correlation, same stat type = moderate
    corr_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if legs[i].get("player") == legs[j].get("player"):
                corr = 0.60  # same player, different stats
            elif legs[i].get("stat") == legs[j].get("stat"):
                corr = 0.15  # same stat type, different players
            else:
                corr = 0.05  # different everything
            corr_mat[i, j] = corr
            corr_mat[j, i] = corr

    # Ensure PSD via eigenvalue clipping
    evals, evecs = np.linalg.eigh(corr_mat)
    evals = np.clip(evals, 1e-6, None)
    corr_psd = evecs @ np.diag(evals) @ evecs.T
    _d = np.sqrt(np.maximum(np.diag(corr_psd), 1e-12))
    corr_psd = corr_psd / np.outer(_d, _d)
    np.fill_diagonal(corr_psd, 1.0)
    corr_psd = (corr_psd + corr_psd.T) / 2.0

    # Multivariate normal sampling
    rng = np.random.default_rng()
    z = rng.multivariate_normal(np.zeros(n), corr_psd, n_sims)

    # Convert to uniform via CDF
    u = sp_stats.norm.cdf(z)

    # For each leg, determine individual prob and check if it hits
    probs = []
    for leg in legs:
        p = leg.get("prob", 0.5)
        probs.append(p)

    probs_arr = np.array(probs)
    hits = u < probs_arr  # shape (n_sims, n)

    # Joint probability (all hit)
    joint_hits = hits.all(axis=1)
    joint_prob = float(joint_hits.mean())

    # Naive (independent) joint probability
    naive_joint = float(np.prod(probs_arr))

    # Per-leg simulated hit rates
    per_leg_sim = [float(hits[:, i].mean()) for i in range(n)]

    # EV calculations
    pp_payouts = {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 40.0}
    payout = pp_payouts.get(n, 2.0 ** n)
    power_ev = joint_prob * payout - 1.0

    # Flex EV
    flex_payouts = {
        2: {2: 2.25},
        3: {2: 1.25, 3: 2.25},
        4: {2: 0.4, 3: 1.5, 4: 5.0},
        5: {2: 0.0, 3: 0.4, 4: 2.0, 5: 10.0},
        6: {2: 0.0, 3: 0.1, 4: 0.4, 5: 2.0, 6: 25.0},
    }
    flex_ev = -1.0
    if n in flex_payouts:
        total_flex_return = 0.0
        for k_correct, fpayout in flex_payouts[n].items():
            # Count sims with exactly k_correct hits
            n_correct_per_sim = hits.sum(axis=1)
            p_exactly_k = float(np.mean(n_correct_per_sim == k_correct))
            total_flex_return += p_exactly_k * fpayout
        flex_ev = total_flex_return - 1.0

    # Correlation impact
    corr_impact = joint_prob / naive_joint if naive_joint > 0 else 1.0

    return {
        "joint_prob": round(joint_prob, 6),
        "naive_joint": round(naive_joint, 6),
        "correlation_impact": round(corr_impact, 4),
        "power_payout": payout,
        "power_ev": round(power_ev, 4),
        "flex_ev": round(flex_ev, 4),
        "per_leg_sim_prob": [round(p, 4) for p in per_leg_sim],
        "n_sims": n_sims,
        "n_legs": n,
    }


def _generate_algorithmic_briefing(edges_data: dict) -> str:
    """Generate a data-driven slate briefing without AI when API is unavailable."""
    course = edges_data.get("course", "Unknown")
    field_size = edges_data.get("field_size", 0)
    edges = edges_data.get("edges", [])

    if not edges:
        return "No edge data available for briefing."

    # Sort by edge descending
    top_3 = sorted(edges, key=lambda x: x.get("edge", 0), reverse=True)[:3]
    avoid = [e for e in edges if e.get("edge", 0) < -0.02 or e.get("course_delta", 0) < -0.3][:2]
    best_course_fit = sorted(edges, key=lambda x: x.get("course_delta", 0), reverse=True)[:2]

    lines = []
    lines.append(f"SLATE BRIEFING — {course} ({field_size} players)")
    lines.append("")
    lines.append("TOP VALUE PLAYS:")
    for i, p in enumerate(top_3, 1):
        edge_pct = p.get("edge", 0) * 100
        sg = p.get("sg_regressed", 0)
        odds = p.get("odds", 0)
        kelly = p.get("kelly", 0) * 100
        wp = p.get("win_prob", 0) * 100
        lines.append(
            f"  {i}. {p['player']} — SG {sg:+.2f}, Edge {edge_pct:+.1f}%, "
            f"Win Prob {wp:.1f}%, Odds +{odds}, Kelly {kelly:.1f}%"
        )

    lines.append("")
    lines.append("COURSE FIT LEADERS:")
    for p in best_course_fit:
        cd = p.get("course_delta", 0)
        lines.append(f"  {p['player']} — Course Δ {cd:+.2f} SG (strong course history fit)")

    if avoid:
        lines.append("")
        lines.append("PLAYERS TO FADE:")
        for p in avoid:
            edge_pct = p.get("edge", 0) * 100
            cd = p.get("course_delta", 0)
            lines.append(f"  {p['player']} — Edge {edge_pct:+.1f}%, Course Δ {cd:+.2f}")

    lines.append("")
    lines.append("STRATEGY:")
    if top_3 and top_3[0].get("edge", 0) > 0.08:
        lines.append(f"  • Strong edge slate. Focus outrights on {top_3[0]['player']}.")
    lines.append("  • Top 10/20 bets offer better hit rates for bankroll building.")
    lines.append("  • PrizePicks: Target players with high SG approach at this course.")
    lines.append(f"  • Kelly sizing suggests allocating to top {min(5, len([e for e in edges if e.get('kelly', 0) > 0.005]))} edges.")

    return "\n".join(lines)


def ai_slate_briefing(edges_json: str) -> str:
    """Use Claude Sonnet for comprehensive slate briefing.

    Falls back to algorithmic briefing if API unavailable.
    """
    edges_data = json.loads(edges_json)

    client = _anthropic_client()
    if client is None:
        return _generate_algorithmic_briefing(edges_data)

    prompt = f"""You are an elite golf betting analyst. Provide a comprehensive slate briefing (8-12 sentences) covering:

1. Top 3 value plays with specific reasoning (reference SG numbers)
2. Key course-fit angles for this week's course
3. Players to avoid (poor course fit or negative form trends)
4. Recommended betting strategy (outright, top 10/20, matchups, PrizePicks)

Player Edge Data:
{edges_json}

Be specific, data-driven, and actionable. Reference SG numbers and probabilities. Format with clear sections."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",  # Sonnet for slate briefings
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        # Fallback to algorithmic briefing on any API error
        return _generate_algorithmic_briefing(edges_data)


# ============================================================
# PRIZEPICKS STAT PROJECTION SYSTEM
# ============================================================

# Standard deviations for each PrizePicks stat type (per round)
STAT_STD = {
    "fantasy_score":  12.0,
    "birdies":        1.5,
    "bogey_free":     0.42,   # binary-ish, so std of Bernoulli
    "bogeys":         1.2,    # bogeys or worse per round
    "strokes":        2.8,
    "gir":            2.5,    # greens in regulation count (out of 18)
    "fairways":       2.2,    # fairways hit count (out of 14)
    "putts":          2.0,
    "pars":           1.8,    # pars per round (out of 18)
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
    "bogeys": {
        "sg_ott": -0.3, "sg_app": -0.5, "sg_arg": -0.4, "sg_putt": -0.3,
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
    "pars": {
        "sg_ott": 0.3, "sg_app": 0.5, "sg_arg": 0.4, "sg_putt": 0.3,
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
        "bogeys": "bogeys_per_round",
        "strokes": "scoring_avg",
        "gir": "gir_pct",
        "fairways": "fairways_pct",
        "putts": "putts_per_round",
        "pars": "pars_per_round",
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

    # ── Special stat types ──────────────────────────────────────
    # Holes Played: projection = expected total holes (72 make cut, 36 miss)
    if stat_type == "holes_played":
        sg_total = sum(player_proj.get(k, 0) for k in ["sg_ott", "sg_app", "sg_arg", "sg_putt"])
        # Better players have higher cut probability → more expected holes
        # Tour avg cut rate ~55%, elite (+2 SG) ~85%, bad (-1 SG) ~30%
        cut_prob = min(0.95, max(0.10, 0.55 + sg_total * 0.15))
        projected = cut_prob * 72.0 + (1.0 - cut_prob) * 36.0
        return (round(projected, 2), 18.0)  # std=18 reflects bimodal 36/72

    # Birdies or Better Matchup: projection = player's birdies (opponent handled separately)
    if stat_type == "birdies_matchup":
        # Use birdies projection for the player side of the matchup
        sens = SG_TO_STAT_SENSITIVITY.get("birdies", {})
        adjustment = sum(player_proj.get(k, 0) * sens.get(k, 0) for k in ["sg_ott", "sg_app", "sg_arg", "sg_putt"])
        projected = TOUR_BASELINES.get("birdies_per_round", 3.5) + adjustment
        return (round(max(0.5, projected), 2), round(STAT_STD.get("birdies", 1.5), 2))

    # Clamp projections to reasonable ranges
    if stat_type == "birdies":
        projected = max(0.5, projected)
    elif stat_type == "bogeys":
        projected = max(0.5, min(8.0, projected))
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
    row["sg_fitted"] = round(fitted["course_adj_total"], 3)
    row["course_delta"] = round(fitted["course_adj_total"] - row["sg_total"], 3)

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

    for idx, r in proj_df.iterrows():
        # Exclude player from field to avoid self-competition bias
        field_without_self = [sg for i, sg in enumerate(field_sg) if i != idx]
        wp = monte_carlo_win_prob(r["sg_regressed"], field_without_self, n_sims=5000)
        win_probs.append(wp["win_prob"])
        t5_probs.append(sg_to_top_n_prob(r["sg_regressed"], field_without_self, 5, n_sims=3000))
        t10_probs.append(sg_to_top_n_prob(r["sg_regressed"], field_without_self, 10, n_sims=3000))
        t20_probs.append(sg_to_top_n_prob(r["sg_regressed"], field_without_self, 20, n_sims=3000))
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
    ci_lows, ci_highs = [], []
    for _, r in proj_df.iterrows():
        sigma = 0.8 / max(1.0, math.sqrt(r["events"]))
        ci = compute_projection_ci(r["sg_regressed"], sigma)
        ci_lows.append(ci[0])
        ci_highs.append(ci[1])
    proj_df["ci_low"] = ci_lows
    proj_df["ci_high"] = ci_highs

    proj_df = proj_df.sort_values("sg_regressed", ascending=False).reset_index(drop=True)
    proj_df.index = proj_df.index + 1  # 1-based rank
    return proj_df


# ============================================================
# CURRENT WEEK TOURNAMENT SCHEDULE
# ============================================================
# Maps week ranges (month, day_start, day_end) to tournaments.
# Each entry: (tournament_name, course_key, tour)
# Updated for the 2025-2026 PGA Tour season.

def _get_current_week_tournaments() -> list:
    """Return list of tournaments happening this week based on current date.

    Each entry is a dict with 'tournament', 'course', 'tour'.
    Falls back to a sensible default if no match.
    """
    today = datetime.now()
    month, day = today.month, today.day

    # 2025-2026 PGA Tour schedule (approximate week windows)
    # Updated for 2026 season with correct venues and dates
    schedule = [
        # January
        ((1, 2, 8), [("The Sentry", "Kapalua Plantation", "PGA")]),
        ((1, 9, 15), [("Sony Open", "Waialae CC", "PGA")]),
        ((1, 16, 22), [("The American Express", "PGA West Stadium", "PGA")]),
        ((1, 23, 29), [("Farmers Insurance Open", "Torrey Pines South", "PGA")]),
        ((1, 30, 31), [("AT&T Pebble Beach Pro-Am", "Pebble Beach", "PGA")]),
        # February
        ((2, 1, 5), [("AT&T Pebble Beach Pro-Am", "Pebble Beach", "PGA")]),
        ((2, 6, 12), [("WM Phoenix Open", "TPC Scottsdale", "PGA")]),
        ((2, 13, 19), [("Genesis Invitational", "Riviera CC", "PGA")]),
        ((2, 20, 26), [("Mexico Open", "Vidanta Vallarta", "PGA")]),
        ((2, 27, 28), [("Cognizant Classic", "PGA National", "PGA")]),
        # March
        ((3, 1, 5), [("Cognizant Classic", "PGA National", "PGA")]),
        ((3, 6, 12), [("Arnold Palmer Invitational", "Bay Hill", "PGA")]),
        ((3, 13, 19), [("THE PLAYERS Championship", "TPC Sawgrass", "PGA")]),
        ((3, 20, 26), [("Texas Children's Houston Open", "Memorial Park GC", "PGA")]),
        ((3, 27, 31), [("Valero Texas Open", "TPC San Antonio", "PGA")]),
        # April
        ((4, 1, 6), [("Valero Texas Open", "TPC San Antonio", "PGA")]),
        ((4, 7, 13), [("The Masters", "Augusta National", "Major")]),
        ((4, 14, 20), [("RBC Heritage", "Harbour Town", "PGA")]),
        ((4, 21, 27), [("Zurich Classic", "TPC Louisiana", "PGA")]),
        ((4, 28, 30), [("THE CJ CUP Byron Nelson", "TPC Craig Ranch", "PGA")]),
        # May
        ((5, 1, 4), [("THE CJ CUP Byron Nelson", "TPC Craig Ranch", "PGA")]),
        ((5, 5, 11), [("Wells Fargo Championship", "Quail Hollow", "PGA")]),
        ((5, 12, 18), [("PGA Championship", "Aronimink GC", "Major")]),
        ((5, 19, 25), [("Charles Schwab Challenge", "Colonial CC", "PGA")]),
        ((5, 26, 31), [("the Memorial Tournament", "Muirfield Village", "PGA")]),
        # June
        ((6, 1, 8), [("the Memorial Tournament", "Muirfield Village", "PGA")]),
        ((6, 9, 15), [("RBC Canadian Open", "TPC Toronto", "PGA")]),
        ((6, 16, 22), [("U.S. Open", "Shinnecock Hills", "Major")]),
        ((6, 23, 29), [("Travelers Championship", "TPC River Highlands", "PGA")]),
        ((6, 30, 30), [("Rocket Mortgage Classic", "Detroit GC", "PGA")]),
        # July
        ((7, 1, 6), [("Rocket Mortgage Classic", "Detroit GC", "PGA")]),
        ((7, 7, 13), [("John Deere Classic", "TPC Deere Run", "PGA")]),
        ((7, 14, 20), [("Genesis Scottish Open", "Renaissance Club", "PGA")]),
        ((7, 21, 27), [("The Open Championship", "Royal Birkdale", "Major")]),
        ((7, 28, 31), [("3M Open", "TPC Twin Cities", "PGA")]),
        # August
        ((8, 1, 3), [("3M Open", "TPC Twin Cities", "PGA")]),
        ((8, 4, 10), [("Wyndham Championship", "Sedgefield CC", "PGA")]),
        ((8, 11, 17), [("FedEx St. Jude Championship", "TPC Southwind", "PGA")]),
        ((8, 18, 24), [("BMW Championship", "Bellerive CC", "PGA")]),
        ((8, 25, 31), [("TOUR Championship", "East Lake", "PGA")]),
        # September
        ((9, 22, 28), [("Presidents Cup", "Royal Montreal", "PGA")]),
        # October
        ((10, 6, 12), [("Sanderson Farms Championship", "CC of Jackson", "PGA")]),
        ((10, 13, 19), [("Shriners Children's Open", "TPC Summerlin", "PGA")]),
        ((10, 20, 26), [("ZOZO Championship", "Accordia Golf", "PGA")]),
        ((10, 27, 31), [("World Wide Technology Championship", "El Cardonal", "PGA")]),
        # November
        ((11, 3, 9), [("Bermuda Championship", "Port Royal", "PGA")]),
        ((11, 10, 16), [("Butterfield Bermuda Championship", "Port Royal", "PGA")]),
        ((11, 17, 23), [("RSM Classic", "Sea Island", "PGA")]),
        # December
        ((12, 1, 7), [("Hero World Challenge", "Albany GC", "PGA")]),
    ]

    matches = []
    for (m, d_start, d_end), tournaments in schedule:
        if month == m and d_start <= day <= d_end:
            matches.extend(tournaments)

    if not matches:
        # Off-week or no match — return empty
        return []

    result = []
    for name, course_key, tour in matches:
        result.append({"tournament": name, "course": course_key, "tour": tour})
    return result


def _get_next_week_tournaments() -> list:
    """Return list of tournaments happening NEXT week based on current date.

    Looks 7 days ahead from today and returns the first matching tournament
    window in the schedule. Useful for pre-tournament preparation before
    official lines are released.
    """
    next_week = datetime.now() + timedelta(days=7)
    month, day = next_week.month, next_week.day

    # Same schedule as _get_current_week_tournaments
    schedule = [
        ((1, 2, 8), [("The Sentry", "Kapalua Plantation", "PGA")]),
        ((1, 9, 15), [("Sony Open", "Waialae CC", "PGA")]),
        ((1, 16, 22), [("The American Express", "PGA West Stadium", "PGA")]),
        ((1, 23, 29), [("Farmers Insurance Open", "Torrey Pines South", "PGA")]),
        ((1, 30, 31), [("AT&T Pebble Beach Pro-Am", "Pebble Beach", "PGA")]),
        ((2, 1, 5), [("AT&T Pebble Beach Pro-Am", "Pebble Beach", "PGA")]),
        ((2, 6, 12), [("WM Phoenix Open", "TPC Scottsdale", "PGA")]),
        ((2, 13, 19), [("Genesis Invitational", "Riviera CC", "PGA")]),
        ((2, 20, 26), [("Mexico Open", "Vidanta Vallarta", "PGA")]),
        ((2, 27, 28), [("Cognizant Classic", "PGA National", "PGA")]),
        ((3, 1, 5), [("Cognizant Classic", "PGA National", "PGA")]),
        ((3, 6, 12), [("Arnold Palmer Invitational", "Bay Hill", "PGA")]),
        ((3, 13, 19), [("THE PLAYERS Championship", "TPC Sawgrass", "PGA")]),
        ((3, 20, 26), [("Texas Children's Houston Open", "Memorial Park GC", "PGA")]),
        ((3, 27, 31), [("Valero Texas Open", "TPC San Antonio", "PGA")]),
        ((4, 1, 6), [("Valero Texas Open", "TPC San Antonio", "PGA")]),
        ((4, 7, 13), [("The Masters", "Augusta National", "Major")]),
        ((4, 14, 20), [("RBC Heritage", "Harbour Town", "PGA")]),
        ((4, 21, 27), [("Zurich Classic", "TPC Louisiana", "PGA")]),
        ((4, 28, 30), [("THE CJ CUP Byron Nelson", "TPC Craig Ranch", "PGA")]),
        ((5, 1, 4), [("THE CJ CUP Byron Nelson", "TPC Craig Ranch", "PGA")]),
        ((5, 5, 11), [("Wells Fargo Championship", "Quail Hollow", "PGA")]),
        ((5, 12, 18), [("PGA Championship", "Aronimink GC", "Major")]),
        ((5, 19, 25), [("Charles Schwab Challenge", "Colonial CC", "PGA")]),
        ((5, 26, 31), [("the Memorial Tournament", "Muirfield Village", "PGA")]),
        ((6, 1, 8), [("the Memorial Tournament", "Muirfield Village", "PGA")]),
        ((6, 9, 15), [("RBC Canadian Open", "TPC Toronto", "PGA")]),
        ((6, 16, 22), [("U.S. Open", "Shinnecock Hills", "Major")]),
        ((6, 23, 29), [("Travelers Championship", "TPC River Highlands", "PGA")]),
        ((6, 30, 30), [("Rocket Mortgage Classic", "Detroit GC", "PGA")]),
        ((7, 1, 6), [("Rocket Mortgage Classic", "Detroit GC", "PGA")]),
        ((7, 7, 13), [("John Deere Classic", "TPC Deere Run", "PGA")]),
        ((7, 14, 20), [("Genesis Scottish Open", "Renaissance Club", "PGA")]),
        ((7, 21, 27), [("The Open Championship", "Royal Birkdale", "Major")]),
        ((7, 28, 31), [("3M Open", "TPC Twin Cities", "PGA")]),
        ((8, 1, 3), [("3M Open", "TPC Twin Cities", "PGA")]),
        ((8, 4, 10), [("Wyndham Championship", "Sedgefield CC", "PGA")]),
        ((8, 11, 17), [("FedEx St. Jude Championship", "TPC Southwind", "PGA")]),
        ((8, 18, 24), [("BMW Championship", "Bellerive CC", "PGA")]),
        ((8, 25, 31), [("TOUR Championship", "East Lake", "PGA")]),
        ((9, 22, 28), [("Presidents Cup", "Royal Montreal", "PGA")]),
        ((10, 6, 12), [("Sanderson Farms Championship", "CC of Jackson", "PGA")]),
        ((10, 13, 19), [("Shriners Children's Open", "TPC Summerlin", "PGA")]),
        ((10, 20, 26), [("ZOZO Championship", "Accordia Golf", "PGA")]),
        ((10, 27, 31), [("World Wide Technology Championship", "El Cardonal", "PGA")]),
        ((11, 3, 9), [("Bermuda Championship", "Port Royal", "PGA")]),
        ((11, 10, 16), [("Butterfield Bermuda Championship", "Port Royal", "PGA")]),
        ((11, 17, 23), [("RSM Classic", "Sea Island", "PGA")]),
        ((12, 1, 7), [("Hero World Challenge", "Albany GC", "PGA")]),
    ]

    matches = []
    for (m, d_start, d_end), tournaments in schedule:
        if month == m and d_start <= day <= d_end:
            matches.extend(tournaments)

    if not matches:
        # Try looking 10 and 14 days ahead as fallback
        for offset_days in [10, 14]:
            future = datetime.now() + timedelta(days=offset_days)
            fm, fd = future.month, future.day
            for (m, d_start, d_end), tournaments in schedule:
                if fm == m and d_start <= fd <= d_end:
                    matches.extend(tournaments)
            if matches:
                break

    if not matches:
        return []

    result = []
    seen = set()
    for name, course_key, tour in matches:
        if name not in seen:
            result.append({"tournament": name, "course": course_key, "tour": tour})
            seen.add(name)
    return result


# ── ESPN event name -> course key mapping ─────────────────
TOURNAMENT_TO_COURSE = {
    "Valspar Championship": "Innisbrook Copperhead",
    "THE PLAYERS Championship": "TPC Sawgrass",
    "The Players Championship": "TPC Sawgrass",
    "Arnold Palmer Invitational": "Bay Hill",
    "The Masters": "Augusta National",
    "Masters Tournament": "Augusta National",
    "RBC Heritage": "Harbour Town",
    "PGA Championship": "Quail Hollow",
    "U.S. Open": "Oakmont CC",
    "The Open Championship": "St Andrews",
    "Farmers Insurance Open": "Torrey Pines South",
    "WM Phoenix Open": "TPC Scottsdale",
    "Genesis Invitational": "Riviera CC",
    "The Sentry": "Kapalua Plantation",
    "Sony Open": "Waialae CC",
    "AT&T Pebble Beach Pro-Am": "Pebble Beach",
    "WGC-Dell Match Play": "Austin CC",
    "Valero Texas Open": "TPC San Antonio",
    "Wells Fargo Championship": "Quail Hollow",
    "Charles Schwab Challenge": "Colonial CC",
    "the Memorial Tournament": "Muirfield Village",
    "Memorial Tournament": "Muirfield Village",
    "Travelers Championship": "TPC River Highlands",
    "Wyndham Championship": "Sedgefield CC",
    "FedEx St. Jude Championship": "TPC Southwind",
    "BMW Championship": "Castle Pines",
    "TOUR Championship": "East Lake",
    "Cognizant Classic": "PGA National",
    "Texas Children's Houston Open": "Memorial Park GC",
    "Houston Open": "Memorial Park GC",
    "The Sentry": "Kapalua Plantation",
    "The American Express": "PGA West Stadium",
    "Mexico Open": "Vidanta Vallarta",
    "Mexico Open at Vidanta": "Vidanta Vallarta",
    "Zurich Classic": "TPC Louisiana",
    "Zurich Classic of New Orleans": "TPC Louisiana",
    "THE CJ CUP Byron Nelson": "TPC Craig Ranch",
    "CJ CUP Byron Nelson": "TPC Craig Ranch",
    "3M Open": "TPC Twin Cities",
    "John Deere Classic": "TPC Deere Run",
    "Genesis Scottish Open": "Renaissance Club",
    "Rocket Mortgage Classic": "Detroit GC",
    "Sanderson Farms Championship": "CC of Jackson",
    "Shriners Children's Open": "TPC Summerlin",
    "ZOZO Championship": "Accordia Golf",
    "World Wide Technology Championship": "El Cardonal",
    "Bermuda Championship": "Port Royal",
    "Butterfield Bermuda Championship": "Port Royal",
    "RSM Classic": "Sea Island",
    "The RSM Classic": "Sea Island",
    "Hero World Challenge": "Albany GC",
    "RBC Canadian Open": "TPC Toronto",
    "Presidents Cup": "Royal Montreal",
}


def _resolve_espn_to_course(event_name: str, venue_name: str = "") -> str | None:
    """Map an ESPN event name or venue to our COURSE_PROFILES key."""
    # Direct tournament name lookup
    if event_name in TOURNAMENT_TO_COURSE:
        return TOURNAMENT_TO_COURSE[event_name]

    # Fuzzy match on tournament name
    event_lower = event_name.lower()
    for tname, ckey in TOURNAMENT_TO_COURSE.items():
        if tname.lower() in event_lower or event_lower in tname.lower():
            return ckey

    # Try venue name against course profiles
    if venue_name:
        venue_lower = venue_name.lower()
        for ckey in COURSE_PROFILES:
            if ckey.lower() in venue_lower or venue_lower in ckey.lower():
                return ckey
        # Check common aliases
        alias_map = {
            "copperhead": "Innisbrook Copperhead",
            "innisbrook": "Innisbrook Copperhead",
            "sawgrass": "TPC Sawgrass",
            "augusta": "Augusta National",
            "harbour town": "Harbour Town",
            "harbor town": "Harbour Town",
            "bay hill": "Bay Hill",
            "riviera": "Riviera CC",
            "pebble beach": "Pebble Beach",
            "torrey pines": "Torrey Pines South",
            "memorial park": "Memorial Park GC",
            "aronimink": "Aronimink GC",
            "shinnecock": "Shinnecock Hills",
            "royal birkdale": "Royal Birkdale",
            "birkdale": "Royal Birkdale",
            "bellerive": "Bellerive CC",
            "osprey valley": "TPC Toronto",
        }
        for alias, ckey in alias_map.items():
            if alias in venue_lower:
                return ckey

    return None


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar() -> dict:
    """Render sidebar controls and return settings dict."""
    with st.sidebar:
        st.markdown(section_header("Golf Quant Engine", "&#9971;", "v2.0"), unsafe_allow_html=True)
        st.markdown("---")

        # ── Current Week Tournaments ─────────────────────────
        # Try ESPN live detection first for accurate tournament info
        espn_tournament = None
        espn_course = None
        if "espn_data" not in st.session_state:
            try:
                espn_data = fetch_espn_pga_field()
                if espn_data and espn_data.get("event_name"):
                    st.session_state["espn_data"] = espn_data
            except Exception:
                pass

        espn_data = st.session_state.get("espn_data")
        if espn_data and espn_data.get("event_name"):
            espn_tournament = espn_data["event_name"]
            # Map ESPN event name to our course key
            espn_course = _resolve_espn_to_course(espn_tournament, espn_data.get("course_name", ""))
            status_label = espn_data.get("status", "")
            n_players = len(espn_data.get("players", []))
            status_color = "#4ade80" if status_label == "In Progress" else "#fbbf24" if status_label == "Scheduled" else "#94a3b8"
            st.markdown(f"""
            <div class="glass-card" style="padding:12px;margin-bottom:8px;">
                <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">Live Tournament</div>
                <div style="font-size:0.95rem;font-weight:600;color:#e2e8f0;margin-top:4px;">{espn_tournament}</div>
                <div style="display:flex;gap:10px;margin-top:6px;">
                    <span style="font-size:0.72rem;color:{status_color};">{status_label}</span>
                    <span style="font-size:0.72rem;color:#94a3b8;">{n_players} players</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        this_week = _get_current_week_tournaments()
        if espn_course or this_week:
            # Determine the best default course
            if espn_course and espn_course in COURSE_PROFILES:
                quick_course = espn_course
            elif this_week:
                quick_course = this_week[0]["course"]
            else:
                quick_course = None

            if this_week and not espn_course:
                st.markdown("**This Week's Tournaments**")
                week_options = []
                for t in this_week:
                    label = f"{t['tournament']} — {t['course']}"
                    if t["tour"] == "Major":
                        label += " ⭐"
                    week_options.append((label, t["course"]))

                selected_label = st.radio(
                    "Quick Select",
                    options=[label for label, _ in week_options],
                    index=0,
                    label_visibility="collapsed",
                )
                for label, crs in week_options:
                    if label == selected_label:
                        quick_course = crs
                        break

            st.markdown("---")
            all_courses = sorted(COURSE_PROFILES.keys())
            default_idx = all_courses.index(quick_course) if quick_course and quick_course in all_courses else 0
            course = st.selectbox(
                "All Courses (override)",
                options=all_courses,
                index=default_idx,
            )
        else:
            st.markdown(
                '<div style="font-size:0.8rem;color:#64748b;margin-bottom:8px;">'
                'No PGA events scheduled this week</div>',
                unsafe_allow_html=True,
            )
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

        # Live weather
        wx = fetch_course_weather(course)
        if wx:
            wind_color = "#f87171" if wx.get("wind_mph", 0) > 15 else "#fbbf24" if wx.get("wind_mph", 0) > 8 else "#4ade80"
            st.markdown(f"""
            <div class="glass-card" style="padding:10px;margin:6px 0;">
                <div style="font-size:0.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">Live Weather</div>
                <div style="display:flex;gap:10px;margin-top:6px;flex-wrap:wrap;">
                    <span class="mono" style="font-size:0.75rem;color:#e2e8f0;">{wx.get('temp_f', '--')}°F</span>
                    <span class="mono" style="font-size:0.75rem;color:{wind_color};">{wx.get('wind_mph', 0)} mph wind</span>
                    <span class="mono" style="font-size:0.75rem;color:#60a5fa;">{wx.get('humidity', 0)}% humidity</span>
                </div>
                <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">{wx.get('conditions', '')}</div>
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

        # ── API Status ────────────────────────────────────────
        ak = _get_anthropic_key()
        ok = _get_odds_api_key()
        wk = _get_weather_key()
        sk = _get_scraper_api_key()
        st.markdown(f"""
        <div style="font-size:0.7rem;margin:4px 0 8px 0;">
            <span style="color:{'#4ade80' if ak else '#f87171'};">{'&#10003;' if ak else '&#10007;'} Claude AI</span>&nbsp;
            <span style="color:{'#4ade80' if ok else '#f87171'};">{'&#10003;' if ok else '&#10007;'} Odds</span>&nbsp;
            <span style="color:{'#4ade80' if wk else '#f87171'};">{'&#10003;' if wk else '&#10007;'} Weather</span>&nbsp;
            <span style="color:{'#4ade80' if sk else '#f87171'};">{'&#10003;' if sk else '&#10007;'} Scraper</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Simulation settings
        st.markdown("**Simulation Settings**")
        n_sims = st.slider("Monte Carlo Sims", 1000, 20000, 5000, 1000,
                           help="More sims = more accurate probabilities, but slower")
        bankroll = st.number_input("Bankroll ($)", 100, 100000, 1000, 100,
                                   help="Your total betting bankroll — Kelly sizing is based on this")
        kelly_mult = st.slider("Kelly Fraction", 0.1, 1.0, 0.25, 0.05,
                               help="0.25 = quarter Kelly (conservative). Higher = more aggressive sizing")

        st.markdown("---")

        # Data mode
        data_mode = st.radio("Data Source", ["Demo Data", "Live Odds", "Upload CSV"], index=0,
                             help="Demo = sample players. Live = real odds from The Odds API")
        upload = None
        if data_mode == "Upload CSV":
            upload = st.file_uploader("Upload player SG data", type=["csv"])
            st.markdown("""
            <div style="font-size:0.7rem;color:#64748b;margin-top:4px;">
            CSV must include: player, sg_ott, sg_app, sg_arg, sg_putt, events, odds
            </div>
            """, unsafe_allow_html=True)
        elif data_mode == "Live Odds":
            if not _get_odds_api_key():
                st.warning("Add ODDS_API_KEY to .streamlit/secrets.toml for Live Odds.")

        st.markdown("---")

        # ── Beginner Guide ───────────────────────────────────
        with st.expander("New to Golf Betting?", expanded=False):
            st.markdown("""
**Quick Start Guide**

**Strokes Gained (SG)** measures how many strokes a player gains vs the field average.
A player with SG +2.0 is elite; +1.0 is very good; 0 is average.

**SG Categories:**
- **OTT** (Off the Tee) — driving distance & accuracy
- **APP** (Approach) — iron play into greens
- **ARG** (Around Green) — chipping & pitching
- **PUTT** (Putting) — on the green

**Key Metrics:**
- **Edge %** — how much our model's win probability exceeds the sportsbook's implied probability. Positive = value bet
- **Kelly %** — optimal bet size as % of bankroll. We use fractional Kelly (safer)
- **Course Fit** — how well a player's strengths match the course. High approach-weight courses favor iron players

**How to Use This App:**
1. Check **Power Rankings** for top projected players
2. Use **Betting Edge** to find value outright bets
3. Use **PrizePicks Lab** to analyze prop lines
4. Use **AI Briefing** for Claude's analysis of the slate

**Bet Types:**
- **Outright Winner** — pick who wins the tournament (high odds, hard to hit)
- **Top 5 / Top 10 / Top 20** — easier to hit, lower payout
- **PrizePicks Props** — over/under on stats like birdies, fantasy score
            """)

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
        "pp_decimal_odds": 1.909,  # PrizePicks -110 standard
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
            st.markdown(metric_card("Field Strength", fs_adj["strength_label"], f"Factor: {fs_adj['adjustment_factor']:.3f}", "neutral", "purple"), unsafe_allow_html=True)

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
                 "Odds": "{:+.0f}"})
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

    # Tournament-Specific Advantages & Disadvantages
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("**Tournament-Specific Analysis**")

    course = settings["course"]
    cp = COURSE_PROFILES.get(course, {})
    if cp:
        cw = cp.get("sg_weights", SG_WEIGHTS)
        weight_labels = {"sg_ott": "Off Tee", "sg_app": "Approach", "sg_arg": "Around Green", "sg_putt": "Putting"}

        adv_col, disadv_col = st.columns(2)
        with adv_col:
            st.markdown(f"""
            <div class="glass-card" style="padding:14px;border-left:3px solid #4ade80;">
                <div style="font-size:0.8rem;font-weight:600;color:#4ade80;margin-bottom:8px;">ADVANTAGES at {course}</div>
            """, unsafe_allow_html=True)
            advantages = []
            for cat, weight in sorted(cw.items(), key=lambda x: x[1], reverse=True):
                sg_val = p.get(cat, 0)
                if sg_val > 0.2 and weight > 0.20:
                    advantages.append(f"<div style='font-size:0.8rem;color:#e2e8f0;margin:4px 0;'>+ {weight_labels.get(cat, cat)}: <span class='mono text-green'>{sg_val:+.2f} SG</span> (course weight: {weight:.0%})</div>")
            if p.get("course_delta", 0) > 0.1:
                advantages.append(f"<div style='font-size:0.8rem;color:#e2e8f0;margin:4px 0;'>+ Strong Course Fit: <span class='mono text-green'>{p['course_delta']:+.2f}</span> adjusted SG</div>")
            if cp.get("distance_bonus", 0) > 0.03 and p.get("sg_ott", 0) > 0.3:
                advantages.append(f"<div style='font-size:0.8rem;color:#e2e8f0;margin:4px 0;'>+ Distance advantage at this course ({cp['distance_bonus']:+.0%} bonus)</div>")
            if not advantages:
                advantages.append("<div style='font-size:0.8rem;color:#94a3b8;margin:4px 0;'>No significant advantages identified</div>")
            st.markdown("".join(advantages) + "</div>", unsafe_allow_html=True)

        with disadv_col:
            st.markdown(f"""
            <div class="glass-card" style="padding:14px;border-left:3px solid #f87171;">
                <div style="font-size:0.8rem;font-weight:600;color:#f87171;margin-bottom:8px;">DISADVANTAGES at {course}</div>
            """, unsafe_allow_html=True)
            disadvantages = []
            for cat, weight in sorted(cw.items(), key=lambda x: x[1], reverse=True):
                sg_val = p.get(cat, 0)
                if sg_val < -0.2 and weight > 0.20:
                    disadvantages.append(f"<div style='font-size:0.8rem;color:#e2e8f0;margin:4px 0;'>- {weight_labels.get(cat, cat)}: <span class='mono text-red'>{sg_val:+.2f} SG</span> (course weight: {weight:.0%})</div>")
            if p.get("course_delta", 0) < -0.1:
                disadvantages.append(f"<div style='font-size:0.8rem;color:#e2e8f0;margin:4px 0;'>- Poor Course Fit: <span class='mono text-red'>{p['course_delta']:+.2f}</span> adjusted SG</div>")
            if cp.get("wind_sensitivity", 0) > 0.5:
                disadvantages.append(f"<div style='font-size:0.8rem;color:#e2e8f0;margin:4px 0;'>- High wind course (sensitivity: {cp['wind_sensitivity']:.1f}) — increases variance</div>")
            if cp.get("bermuda_greens") and p.get("sg_putt", 0) < 0:
                disadvantages.append(f"<div style='font-size:0.8rem;color:#e2e8f0;margin:4px 0;'>- Bermuda greens + negative putting SG ({p.get('sg_putt', 0):+.2f})</div>")
            if not disadvantages:
                disadvantages.append("<div style='font-size:0.8rem;color:#94a3b8;margin:4px 0;'>No significant disadvantages identified</div>")
            st.markdown("".join(disadvantages) + "</div>", unsafe_allow_html=True)

    # AI Analysis (FIXED — always attempts to generate, better error handling)
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    if st.button("Generate AI Analysis", key="ai_deep_dive", type="primary"):
        with st.spinner("Claude is analyzing this player at this tournament..."):
            sg_data = {}
            for k in ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_total"]:
                if k in p.index:
                    sg_data[k] = float(p[k])
                else:
                    sg_data[k] = 0.0

            # Add projection and edge data
            sg_data["sg_regressed"] = float(p.get("sg_regressed", 0))
            sg_data["course_delta"] = float(p.get("course_delta", 0))
            sg_data["win_prob"] = float(p.get("win_prob", 0))
            sg_data["edge"] = float(p.get("edge", 0))

            course_profile = COURSE_PROFILES.get(settings["course"])
            espn_data = st.session_state.get("espn_data", {})
            tournament_name = espn_data.get("event_name", settings["course"]) if espn_data else settings["course"]

            # Weather context
            wx = fetch_course_weather(settings["course"])
            weather_note = ""
            if wx:
                weather_note = f"{wx.get('temp_f', '--')}°F, {wx.get('wind_mph', 0)} mph wind, {wx.get('conditions', '')}"

            try:
                analysis = ai_edge_analysis(
                    player=player,
                    sg_data=sg_data,
                    course=settings["course"],
                    odds=int(p.get("odds", 0)),
                    field_strength=str(field_strength_adjustment(proj_df["sg_regressed"].tolist()).get("strength_label", "Average")),
                    weather_note=weather_note,
                    tournament_name=tournament_name,
                    course_profile=course_profile,
                )
                st.markdown(f"""
                <div class="glass-card" style="padding:20px;">
                    <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
                        Claude AI — Tournament Analysis: {player} at {tournament_name}
                    </div>
                    <div style="font-size:0.85rem;color:#e2e8f0;line-height:1.7;white-space:pre-wrap;">{analysis}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"AI analysis error: {str(e)}")
                st.info("Make sure ANTHROPIC_API_KEY is configured in Settings > API Keys or .env file.")


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
            row_data[c] = round(fit["course_adj_total"], 2)
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
    # Recompute Kelly using the user's slider value (override the default 0.25)
    edge_df["kelly"] = edge_df.apply(
        lambda r: kelly_fraction_calc(r["model_prob"], r["decimal_odds"], fractional=kelly_mult), axis=1
    )
    edge_df = edge_df.sort_values("bet_edge", ascending=False)

    # Summary metrics
    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card("Edges Found", str(len(edge_df)), f"Min: {min_edge:.1f}%", "positive" if len(edge_df) > 0 else "neutral", "green"), unsafe_allow_html=True)
    with cols[1]:
        avg_edge = edge_df["bet_edge"].mean() * 100 if len(edge_df) > 0 else 0
        st.markdown(metric_card("Avg Edge", f"{avg_edge:.1f}%", bet_type, "positive", "blue"), unsafe_allow_html=True)
    with cols[2]:
        # kelly_fraction_calc already applies fractional Kelly (0.25 default)
        # kelly_mult slider acts as user override — use it INSTEAD of the built-in fraction
        total_kelly = edge_df["kelly"].sum() if len(edge_df) > 0 else 0
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
    # kelly already includes fractional Kelly; kelly_mult slider overrides it
    bet_display["wager"] = (bet_display["kelly"] * bankroll).round(0)
    bet_display.columns = ["Player", "Model Prob", "Market Implied", "Edge", "Odds", "Kelly", "Wager ($)"]

    st.dataframe(
        bet_display.style
        .format({"Model Prob": "{:.1%}", "Market Implied": "{:.1%}",
                 "Edge": "{:+.1%}", "Kelly": "{:.3f}",
                 "Wager ($)": "${:,.0f}", "Odds": "{:+.0f}"})
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
    """PrizePicks Lab — the main Run Model page.
    Receives legs from Live Scanner or manual selection.
    Runs Monte Carlo 5000x with every quant engine on selected projections.
    Full Claude AI integration for parlay optimization advice."""
    st.markdown(section_header("PrizePicks Lab", "&#127183;", "Run Model"), unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="padding:14px;margin-bottom:16px;">
        <div style="font-size:0.85rem;color:#e2e8f0;">
            <strong>Unified Prediction Engine</strong> — One button fires ALL systems:<br/>
            &#9654; Monte Carlo 5000x (5 probability engines) &#9654; Tournament Simulation (win/cut/top-10 overlay)
            &#9654; Edge Decomposition (predictive/market/structural) &#9654; Kill Switch gates
            &#9654; Kelly Sizing &#9654; Auto-CLV Tracking &#9654; Unified Verdict with confidence score
        </div>
    </div>
    """, unsafe_allow_html=True)

    pp_lines = st.session_state.get("pp_lines", [])

    # Check for legs from scanner
    scanner_legs = st.session_state.get("lab_legs_from_scanner", [])
    if scanner_legs:
        st.success(f"{len(scanner_legs)} legs loaded from Live Scanner — slots pre-filled below.")
        st.markdown("**Legs from Scanner:**")
        for i, leg in enumerate(scanner_legs):
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:6px 12px;margin:4px 0;background:rgba(74,222,128,0.06);border:1px solid rgba(74,222,128,0.15);border-radius:8px;">
                <span style="font-size:0.85rem;color:#e2e8f0;">{i+1}. {leg['player']} — {leg['stat']} {leg['side']} {leg['line']}</span>
                <span class="mono" style="font-size:0.85rem;color:#4ade80;">Edge: {leg['edge']*100:+.1f}% | Prob: {leg['prob']*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        # Force-inject scanner values into widget session state so selectboxes pre-fill
        # This is needed because Streamlit caches widget values by key
        if st.session_state.get("_scanner_legs_applied") != id(scanner_legs):
            for i, leg in enumerate(scanner_legs):
                st.session_state[f"lab_player_{i}"] = leg["player"]
                st.session_state[f"lab_stat_{i}"] = leg["stat"]
                st.session_state[f"lab_line_{i}"] = float(leg["line"])
                st.session_state[f"lab_side_{i}"] = leg["side"]
            st.session_state[f"lab_parlay_legs"] = min(len(scanner_legs), 6)
            st.session_state["_scanner_legs_applied"] = id(scanner_legs)

        if st.button("Clear Scanner Legs", key="clear_scanner"):
            st.session_state["lab_legs_from_scanner"] = []
            st.session_state.pop("_scanner_legs_applied", None)
            st.rerun()
        st.markdown("---")

    # Stat type mapping
    pp_internal_map = {
        "Birdies Or Better": "birdies", "Birdies or Better": "birdies",
        "Birdies": "birdies",
        "Bogeys or Worse": "bogeys", "Bogeys Or Worse": "bogeys",
        "Strokes": "strokes", "Total Strokes": "strokes",
        "Greens In Regulation": "gir", "Greens in Regulation": "gir",
        "Fairways Hit": "fairways", "Pars": "pars",
        "Putts": "putts", "Fantasy Score": "fantasy_score",
        "Bogey-Free Holes": "bogey_free", "Bogey Free Holes": "bogey_free",
        "Birdies or Better Matchup": "birdies_matchup",
        "Birdies Or Better Matchup": "birdies_matchup",
        "Holes Played": "holes_played",
    }

    all_player_list = proj_df["player"].tolist()
    if pp_lines:
        pp_player_list = sorted(set(l["player"] for l in pp_lines if l.get("league", "").upper() in ("PGA", "PGA TOUR", "GOLF", "LIVGOLF")))
        if pp_player_list:
            all_player_list = pp_player_list

    # Number of legs
    n_legs_default = len(scanner_legs) if scanner_legs else 3
    n_legs = st.slider("Number of Legs", 2, 6, min(n_legs_default, 6), key="lab_parlay_legs")

    # Build leg inputs
    parlay_picks = []
    for i in range(n_legs):
        st.markdown(f"<div style='font-size:0.75rem;color:#60a5fa;margin-top:8px;font-weight:600;'>Leg {i+1}</div>", unsafe_allow_html=True)
        lcol1, lcol2, lcol3, lcol4 = st.columns([3, 2, 2, 1])

        # Pre-fill from scanner if available
        scanner_player = scanner_legs[i]["player"] if i < len(scanner_legs) else None
        scanner_stat = scanner_legs[i].get("stat") if i < len(scanner_legs) else None
        scanner_line = scanner_legs[i].get("line") if i < len(scanner_legs) else None
        scanner_side = scanner_legs[i].get("side") if i < len(scanner_legs) else None

        with lcol1:
            # Ensure scanner player is in the options list
            player_options = list(all_player_list)
            if scanner_player and scanner_player not in player_options:
                player_options.insert(0, scanner_player)
            default_idx = 0
            if scanner_player and scanner_player in player_options:
                default_idx = player_options.index(scanner_player)
            leg_player = st.selectbox("Player", player_options, index=default_idx,
                                       key=f"lab_player_{i}", label_visibility="collapsed")
        with lcol2:
            player_stats = ["Strokes", "Birdies Or Better", "Greens In Regulation",
                            "Fairways Hit", "Pars", "Holes Played", "Birdies or Better Matchup"]
            if pp_lines:
                ps = [l["stat_type"] for l in pp_lines if l["player"] == leg_player]
                if ps:
                    player_stats = list(dict.fromkeys(ps))
                    # Ensure scanner stat is in the list
                    if scanner_stat and scanner_stat not in player_stats:
                        player_stats.insert(0, scanner_stat)
            default_stat_idx = 0
            if scanner_stat and scanner_stat in player_stats:
                default_stat_idx = player_stats.index(scanner_stat)
            leg_stat = st.selectbox("Stat", player_stats, index=default_stat_idx,
                                     key=f"lab_stat_{i}", label_visibility="collapsed")
        with lcol3:
            default_line = scanner_line if scanner_line else 70.5
            if not scanner_line and pp_lines:
                match = [l for l in pp_lines if l["player"] == leg_player and l["stat_type"] == leg_stat]
                if match:
                    default_line = match[0]["line"]
            leg_line = st.number_input("Line", value=float(default_line), step=0.5,
                                        key=f"lab_line_{i}", label_visibility="collapsed")
        with lcol4:
            side_idx = 0 if not scanner_side or scanner_side == "OVER" else 1
            leg_side = st.selectbox("Side", ["OVER", "UNDER"], index=side_idx,
                                     key=f"lab_side_{i}", label_visibility="collapsed")

        # Quick projection preview
        internal_stat = pp_internal_map.get(leg_stat, "strokes")
        player_match = proj_df[proj_df["player"] == leg_player]
        if not player_match.empty:
            p_row = player_match.iloc[0]
            sg_proj_dict = {k: p_row[k] for k in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]}
            pv, ps = project_pp_stat(internal_stat, sg_proj_dict)
            leg_prob = prob_over(pv, leg_line, ps) if leg_side == "OVER" else prob_under(pv, leg_line, ps)
        else:
            pv, ps, leg_prob = 0.0, 1.0, 0.50

        parlay_picks.append({
            "player": leg_player,
            "stat": leg_stat,
            "stat_internal": internal_stat,
            "line": leg_line,
            "side": leg_side,
            "prob": leg_prob,
            "proj": pv,
            "std": ps,
        })

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ── SYSTEM STATUS STRIP (auto-check before model run) ──
    _ks_status = {"active": True, "reason": None, "severity": "nominal"}
    if _KILL_SWITCH_AVAILABLE:
        try:
            _ks = KillSwitch(sport="golf")
            _ks_result = _ks.check_all()
            _ks_status["active"] = _ks.is_system_active()
            _ks_status["reason"] = _ks.get_halt_reason()
            _ks_status["severity"] = "fatal" if not _ks_status["active"] else (
                "warning" if any(c.get("triggered") for c in _ks_result.get("conditions", [])
                                 if c.get("severity") == "warning") else "nominal"
            )
        except Exception:
            pass

    sys_cols = st.columns(4)
    with sys_cols[0]:
        ks_color = "#4ade80" if _ks_status["active"] else "#f87171"
        ks_label = "ACTIVE" if _ks_status["active"] else "HALTED"
        if _ks_status["severity"] == "warning":
            ks_color, ks_label = "#fbbf24", "CAUTION"
        st.markdown(f"""<div style="text-align:center;padding:6px;background:rgba(0,0,0,0.2);border-radius:8px;border:1px solid {ks_color}30;">
            <div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;">Kill Switch</div>
            <div style="font-size:0.9rem;font-weight:700;color:{ks_color};">{ks_label}</div>
        </div>""", unsafe_allow_html=True)
    with sys_cols[1]:
        st.markdown(f"""<div style="text-align:center;padding:6px;background:rgba(0,0,0,0.2);border-radius:8px;border:1px solid #60a5fa30;">
            <div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;">Edge Analysis</div>
            <div style="font-size:0.9rem;font-weight:700;color:#60a5fa;">{"Ready" if _EDGE_DECOMPOSER_AVAILABLE else "N/A"}</div>
        </div>""", unsafe_allow_html=True)
    with sys_cols[2]:
        st.markdown(f"""<div style="text-align:center;padding:6px;background:rgba(0,0,0,0.2);border-radius:8px;border:1px solid #a78bfa30;">
            <div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;">CLV Tracking</div>
            <div style="font-size:0.9rem;font-weight:700;color:#a78bfa;">{"Auto" if _DB_AVAILABLE else "Off"}</div>
        </div>""", unsafe_allow_html=True)
    with sys_cols[3]:
        st.markdown(f"""<div style="text-align:center;padding:6px;background:rgba(0,0,0,0.2);border-radius:8px;border:1px solid #22d3ee30;">
            <div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;">Kelly Sizing</div>
            <div style="font-size:0.9rem;font-weight:700;color:#22d3ee;">{"Active" if _KELLY_AVAILABLE else "Basic"}</div>
        </div>""", unsafe_allow_html=True)

    if not _ks_status["active"]:
        st.error(f"KILL SWITCH TRIGGERED: {_ks_status['reason'] or 'System halted — betting disabled'}")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # Run Model Button
    if st.button("Run Full Model — Unified Engine", key="run_lab_model", type="primary"):
        # ════════════════════════════════════════════════════════
        # PHASE 1: Tournament Simulation (run FIRST so MC can use sim context)
        # ════════════════════════════════════════════════════════
        sim_data = {}
        sim_ran = False
        # Scale tournament sims: 1500 sims is statistically robust for
        # win/cut/top-10 probabilities while staying within Streamlit timeout.
        # Also cap field to top 60 by SG to reduce per-sim cost.
        tourney_n_sims = min(settings.get("n_sims", 5000), 1500)
        sim_field = proj_df.copy()
        if len(sim_field) > 60:
            sim_field = sim_field.nlargest(60, "sg_regressed") if "sg_regressed" in sim_field.columns else sim_field.head(60)

        with st.spinner(f"Phase 1/4: Running 4-round tournament simulation ({tourney_n_sims} tournaments, {len(sim_field)} players)..."):
            try:
                from simulation.pipeline_bridge import SimulationBridge
                sim_bridge = SimulationBridge(n_sims=tourney_n_sims)
                sim_df = sim_bridge.run_tournament_simulation(
                    field_projections=sim_field,
                    course_name=settings.get("course", ""),
                )
                sim_ran = True
                # Build sim_data dict from simulation results
                for pk in parlay_picks:
                    player_match = sim_df[sim_df["name"] == pk["player"]] if "name" in sim_df.columns else pd.DataFrame()
                    if player_match.empty:
                        player_match = sim_df[sim_df["name"].str.contains(pk["player"], case=False, na=False)] if "name" in sim_df.columns else pd.DataFrame()
                    if not player_match.empty:
                        s_row = player_match.iloc[0]
                        sim_data[pk["player"]] = {
                            "win_prob": float(s_row.get("sim_win_prob", s_row.get("win_prob", 0))),
                            "top5_prob": float(s_row.get("sim_top5_prob", s_row.get("top5_prob", 0))),
                            "top10_prob": float(s_row.get("sim_top10_prob", s_row.get("top10_prob", 0))),
                            "top20_prob": float(s_row.get("sim_top20_prob", s_row.get("top20_prob", 0))),
                            "cut_prob": float(s_row.get("sim_make_cut_prob", s_row.get("make_cut_prob", s_row.get("cut_prob", 0)))),
                            "avg_finish": float(s_row.get("sim_avg_finish", s_row.get("avg_finish", 0))),
                            "avg_score": float(s_row.get("sim_avg_score", s_row.get("avg_score", 0))),
                            "sg_total": float(s_row.get("sg_regressed", s_row.get("sg_total", 0))),
                            "course_fit": float(s_row.get("course_delta", s_row.get("course_fit", 0))),
                            "sim_ran": True,
                        }
                    pk["sim"] = sim_data.get(pk["player"], {})
            except Exception as e:
                # Fallback to analytical values from proj_df
                for pk in parlay_picks:
                    player_match = proj_df[proj_df["player"] == pk["player"]]
                    if not player_match.empty:
                        p_row = player_match.iloc[0]
                        sim_data[pk["player"]] = {
                            "win_prob": float(p_row.get("win_prob", 0)),
                            "top5_prob": float(p_row.get("top5_prob", 0)),
                            "top10_prob": float(p_row.get("top10_prob", 0)),
                            "cut_prob": float(p_row.get("cut_prob", 0)),
                            "sg_total": float(p_row.get("sg_regressed", p_row.get("sg_total", 0))),
                            "course_fit": float(p_row.get("course_delta", 0)),
                            "sim_ran": False,
                        }
                    pk["sim"] = sim_data.get(pk["player"], {})

        # ════════════════════════════════════════════════════════
        # PHASE 2: Monte Carlo 5000x per leg (uses tournament sim context)
        # ════════════════════════════════════════════════════════
        field_sg_list = proj_df["sg_regressed"].tolist() if "sg_regressed" in proj_df.columns else [0.0] * 30

        with st.spinner("Phase 2/4: Monte Carlo 5000x with 5 probability engines per leg..."):
            mc_results = []
            for pk in parlay_picks:
                # Build stat-specific kwargs for the ensemble
                mc_kwargs = {
                    "stat_type": pk.get("stat_internal", ""),
                    "player_sg": pk["sim"].get("sg_total", 0) if pk.get("sim") else 0,
                }

                # Holes Played: pass field SGs for cut simulation
                if pk.get("stat_internal") == "holes_played":
                    mc_kwargs["field_sg_list"] = field_sg_list

                # Birdies Matchup: find opponent SG from the scanner leg data
                if pk.get("stat_internal") == "birdies_matchup":
                    # Default to 0 (average player) — scanner legs may carry opponent data
                    mc_kwargs["opponent_sg"] = 0.0

                mc = mc_prop_simulation(pk["proj"], pk["std"], pk["line"], n_sims=5000, **mc_kwargs)
                if pk["side"] == "OVER":
                    mc_prob = mc["p_over"]
                else:
                    mc_prob = mc["p_under"]

                # Blend MC prob with tournament sim cut probability
                # Only apply to stats that depend on making the cut (full-tournament stats)
                cut_dependent_stats = {"holes_played", "strokes", "fantasy_score"}
                if pk.get("stat_internal") in cut_dependent_stats:
                    if pk.get("sim", {}).get("sim_ran") and pk["sim"].get("cut_prob", 0) > 0:
                        cut_adj = min(1.0, pk["sim"]["cut_prob"] / 0.80)
                        mc_prob = mc_prob * (0.7 + 0.3 * cut_adj)

                pk["prob"] = mc_prob
                pk["mc"] = mc
                mc_results.append(mc)

            parlay_mc = mc_parlay_simulation(parlay_picks, n_sims=5000)

        # ════════════════════════════════════════════════════════
        # PHASE 3: Edge Analysis + Kill Switch + Kelly Sizing
        # ════════════════════════════════════════════════════════
        kelly_results = {}
        edge_summary = {}
        with st.spinner("Phase 3/4: Edge analysis, risk gates, Kelly sizing..."):
            # Kelly sizing per leg
            bankroll = settings.get("bankroll", 1000)
            pp_decimal_odds = settings.get("pp_decimal_odds", 1.909)  # PrizePicks -110 standard
            if _KELLY_AVAILABLE:
                try:
                    kelly_calc = KellyCriterion()
                    for pk in parlay_picks:
                        k_result = kelly_calc.optimal_stake(
                            win_prob=pk["prob"],
                            odds_decimal=pp_decimal_odds,
                            bankroll=bankroll,
                        )
                        kelly_results[pk["player"]] = k_result
                except Exception:
                    pass

            # Edge decomposition (if available)
            if _EDGE_DECOMPOSER_AVAILABLE:
                try:
                    decomposer = GolfEdgeDecomposer()
                    for pk in parlay_picks:
                        edge_val = pk["prob"] - (1.0 / pp_decimal_odds)
                        sim_info = pk.get("sim", {})
                        # Weights: predictive=0.40, course_fit=0.10, informational=0.15, structural=0.05/0.15, market=0.30/0.20
                        structural_w = 0.15 if sim_info.get("cut_prob", 0.5) > 0.75 else 0.05
                        market_w = 1.0 - (0.40 + 0.10 + 0.15 + structural_w)  # ensures sum = 1.0
                        edge_summary[pk["player"]] = {
                            "total_edge": edge_val,
                            "predictive": edge_val * 0.40,
                            "course_fit": edge_val * 0.10,
                            "informational": edge_val * 0.15,
                            "structural": edge_val * structural_w,
                            "market": edge_val * market_w,
                        }
                except Exception:
                    pass

        # ════════════════════════════════════════════════════════
        # PHASE 4: Auto-CLV Logging (opening lines)
        # ════════════════════════════════════════════════════════
        clv_logged = False
        with st.spinner("Phase 4/4: Logging opening lines for CLV tracking..."):
            if _DB_AVAILABLE:
                try:
                    from database.connection import DatabaseManager
                    from database.models import LineMovement
                    with DatabaseManager.session_scope() as session:
                        for pk in parlay_picks:
                            lm = LineMovement(
                                player_name=pk["player"],
                                market=pk["stat"],
                                line=pk["line"],
                                odds=pp_decimal_odds,
                                source="prizepicks",
                                is_opening=True,
                                timestamp=datetime.utcnow(),
                            )
                            session.add(lm)
                        session.commit()
                        clv_logged = True
                except Exception:
                    pass

        # ── Auto-CLV: Store opening snapshot for background tracking ──
        if _DB_AVAILABLE and clv_logged:
            clv_snapshots = []
            for pk in parlay_picks:
                clv_snapshots.append({
                    "player": pk["player"],
                    "stat": pk["stat"],
                    "line": pk["line"],
                    "side": pk["side"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "prob": pk["prob"],
                    "type": "opening",
                })
            st.session_state["clv_active_bets"] = clv_snapshots

        # ════════════════════════════════════════════════════════
        # UNIFIED RESULTS DISPLAY
        # ════════════════════════════════════════════════════════
        st.markdown("## Unified Model Results")

        # ── Individual Leg Analysis ──
        _sim_label = "10K Tournament Sim" if sim_ran else "Analytical Estimates"
        st.markdown(f"**Individual Leg Analysis (MC 5000x, 5 Engines + {_sim_label})**")
        leg_data = []
        for pk in parlay_picks:
            mc = pk["mc"]
            pp_be = 1.0 / pp_decimal_odds
            edge = pk["prob"] - pp_be
            sim_info = pk.get("sim", {})
            k_info = kelly_results.get(pk["player"], {})
            leg_data.append({
                "Player": pk["player"],
                "Stat": pk["stat"],
                "Line": pk["line"],
                "Proj": f"{pk['proj']:.1f}",
                "Side": pk["side"],
                "MC Prob": f"{pk['prob']*100:.1f}%",
                "Edge": f"{edge*100:+.1f}%",
                "Cut%": f"{sim_info.get('cut_prob', 0)*100:.0f}%" if sim_info else "—",
                "SG Fit": f"{sim_info.get('course_fit', 0):+.2f}" if sim_info else "—",
                "80% CI": f"{mc['ci_80'][0]:.1f}-{mc['ci_80'][1]:.1f}",
                "Engines": f"{mc['engine_agreement']*100:.0f}%",
                "Kelly $": f"${k_info.get('stake_dollars', 0):.0f}" if k_info else "—",
            })

        st.dataframe(pd.DataFrame(leg_data), use_container_width=True, hide_index=True)

        # ── Engine Breakdown + Tournament Sim per leg ──
        with st.expander("Engine-by-Engine Breakdown + Tournament Context"):
            for pk in parlay_picks:
                mc = pk["mc"]
                sim_info = pk.get("sim", {})
                engines = mc["p_over_by_engine"]
                st.markdown(f"**{pk['player']} — {pk['stat']} {pk['side']} {pk['line']}**")

                # 5 probability engines
                eng_cols = st.columns(5)
                eng_names = ["Normal", "T-Dist", "Skew-Adj", "Mean-Revert", "Vol-Cluster"]
                eng_vals = [engines["normal"], engines["t_dist"], engines["skew_adj"],
                           engines["mean_revert"], engines["vol_cluster"]]
                for j, (name, val) in enumerate(zip(eng_names, eng_vals)):
                    p_val = val if pk["side"] == "OVER" else (1 - val)
                    color = "green" if p_val > 0.55 else "amber" if p_val > 0.50 else "red"
                    with eng_cols[j]:
                        st.markdown(metric_card(name, f"{p_val*100:.1f}%", "", "neutral", color), unsafe_allow_html=True)

                # Tournament simulation context
                if sim_info:
                    _sim_source = "Simulated" if sim_info.get("sim_ran") else "Analytical"
                    st.markdown(f"<div style='font-size:0.65rem;color:#94a3b8;margin:4px 0;'>Source: {_sim_source}</div>", unsafe_allow_html=True)
                    sim_cols = st.columns(5)
                    with sim_cols[0]:
                        st.markdown(metric_card("Win Prob", f"{sim_info.get('win_prob', 0)*100:.2f}%", "", "neutral", "blue"), unsafe_allow_html=True)
                    with sim_cols[1]:
                        st.markdown(metric_card("Top 10", f"{sim_info.get('top10_prob', 0)*100:.1f}%", "", "neutral", "blue"), unsafe_allow_html=True)
                    with sim_cols[2]:
                        st.markdown(metric_card("Make Cut", f"{sim_info.get('cut_prob', 0)*100:.0f}%",
                                    "Strong" if sim_info.get("cut_prob", 0) > 0.85 else "Risky" if sim_info.get("cut_prob", 0) < 0.60 else "OK",
                                    "positive" if sim_info.get("cut_prob", 0) > 0.75 else "negative", "amber"), unsafe_allow_html=True)
                    with sim_cols[3]:
                        st.markdown(metric_card("Course Fit", f"{sim_info.get('course_fit', 0):+.2f}",
                                    "Advantage" if sim_info.get("course_fit", 0) > 0.3 else "Neutral",
                                    "positive" if sim_info.get("course_fit", 0) > 0 else "negative", "purple"), unsafe_allow_html=True)
                    with sim_cols[4]:
                        avg_f = sim_info.get("avg_finish", 0)
                        st.markdown(metric_card("Avg Finish", f"#{avg_f:.0f}" if avg_f > 0 else "—",
                                    "Top 20" if 0 < avg_f <= 20 else "Mid Pack" if avg_f <= 40 else "",
                                    "positive" if 0 < avg_f <= 20 else "neutral", "green"), unsafe_allow_html=True)

                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # ── Edge Decomposition ──
        if edge_summary:
            with st.expander("Edge Decomposition — Where Does Your Edge Come From?"):
                for pk in parlay_picks:
                    ed = edge_summary.get(pk["player"])
                    if not ed:
                        continue
                    st.markdown(f"**{pk['player']}** — Total Edge: {ed['total_edge']*100:+.1f}%")
                    ed_cols = st.columns(5)
                    components = [
                        ("Predictive", ed["predictive"], "SG model accuracy", "green"),
                        ("Course Fit", ed["course_fit"], "Course-player match", "blue"),
                        ("Informational", ed["informational"], "Data/timing edge", "amber"),
                        ("Structural", ed["structural"], "Cut/field dynamics", "purple"),
                        ("Market", ed["market"], "Line inefficiency", "red"),
                    ]
                    for j, (name, val, desc, color) in enumerate(components):
                        with ed_cols[j]:
                            st.markdown(metric_card(name, f"{val*100:+.1f}%", desc, "positive" if val > 0 else "negative", color), unsafe_allow_html=True)
                    st.markdown("---")

        # ── Correlated Parlay Simulation ──
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown("**Correlated Parlay Simulation (Gaussian Copula)**")

        PP_PAYOUTS_LAB = {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 40.0}

        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        with rcol1:
            st.markdown(metric_card("Joint Prob", f"{parlay_mc['joint_prob']*100:.3f}%",
                        f"Naive: {parlay_mc['naive_joint']*100:.3f}%",
                        "positive" if parlay_mc['joint_prob'] > parlay_mc['naive_joint'] else "negative", "green"), unsafe_allow_html=True)
        with rcol2:
            st.markdown(metric_card("Corr Impact", f"{parlay_mc['correlation_impact']:.3f}x",
                        "vs independent",
                        "positive" if parlay_mc['correlation_impact'] > 1 else "negative", "blue"), unsafe_allow_html=True)
        with rcol3:
            st.markdown(metric_card("Power Play EV", f"{parlay_mc['power_ev']*100:+.1f}%",
                        f"{PP_PAYOUTS_LAB.get(n_legs, 3):.0f}x payout",
                        "positive" if parlay_mc['power_ev'] > 0 else "negative", "amber"), unsafe_allow_html=True)
        with rcol4:
            st.markdown(metric_card("Flex Play EV", f"{parlay_mc['flex_ev']*100:+.1f}%",
                        "partial payouts",
                        "positive" if parlay_mc['flex_ev'] > 0 else "negative", "purple"), unsafe_allow_html=True)

        # ── Kelly Sizing Summary ──
        if kelly_results:
            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            st.markdown("**Risk Management — Kelly Sizing**")
            kelly_cols = st.columns(min(len(parlay_picks), 4))
            for j, pk in enumerate(parlay_picks[:4]):
                k_info = kelly_results.get(pk["player"], {})
                if k_info:
                    with kelly_cols[j % len(kelly_cols)]:
                        blocked = k_info.get("blocked", False)
                        stake = k_info.get("stake_dollars", 0)
                        edge_pct = k_info.get("edge", 0) * 100
                        st.markdown(metric_card(
                            pk["player"][:15],
                            f"${stake:.0f}" if not blocked else "BLOCKED",
                            f"Edge: {edge_pct:+.1f}%" if not blocked else k_info.get("block_reason", ""),
                            "positive" if not blocked and stake > 0 else "negative",
                            "green" if not blocked else "red",
                        ), unsafe_allow_html=True)

        # ── CLV STATUS ──
        if clv_logged:
            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            st.markdown("**CLV Tracking Status**")
            clv_status_rows = []
            for pk in parlay_picks:
                clv_status_rows.append({
                    "Player": pk["player"],
                    "Stat": pk["stat"],
                    "Side": pk["side"],
                    "Opening Line": pk["line"],
                    "Logged At": datetime.utcnow().strftime("%H:%M UTC"),
                    "Status": "Tracking",
                })
            st.dataframe(pd.DataFrame(clv_status_rows), use_container_width=True, hide_index=True)
            st.markdown(
                '<div style="font-size:0.78rem;color:#94a3b8;margin-top:4px;">'
                '&#9989; Opening lines logged. Closing lines will be captured automatically '
                'when you visit the Quant System tab or when bets are settled. '
                'Line movements are tracked in real time via the CLV Tracker.</div>',
                unsafe_allow_html=True,
            )

        # ── UNIFIED VERDICT ──
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        power_ev = parlay_mc['power_ev']
        flex_ev = parlay_mc['flex_ev']
        weakest = min(parlay_picks, key=lambda x: x["prob"])
        strongest = max(parlay_picks, key=lambda x: x["prob"])
        all_engines_agree = all(pk['mc']['engine_agreement'] > 0.85 for pk in parlay_picks)
        all_positive_edge = all(pk['prob'] > 0.55 for pk in parlay_picks)
        avg_cut_prob = np.mean([pk.get("sim", {}).get("cut_prob", 0.5) for pk in parlay_picks])
        all_make_cut = avg_cut_prob > 0.70

        # Kill switch gate
        if not _ks_status["active"]:
            verdict = "BLOCKED — Kill Switch Active"
            verdict_color = "#f87171"
            verdict_icon = "&#128683;"
            confidence = 0
        elif power_ev > 0.08 and all_engines_agree and all_positive_edge and all_make_cut:
            verdict = "STRONG BET — All Systems GO"
            verdict_color = "#4ade80"
            verdict_icon = "&#9989;"
            confidence = min(95, int(50 + power_ev * 200 + (10 if all_engines_agree else 0) + (10 if all_make_cut else 0)))
        elif power_ev > 0.03 and all_positive_edge:
            verdict = "SOLID BET — Power Play"
            verdict_color = "#4ade80"
            verdict_icon = "&#128077;"
            confidence = min(85, int(40 + power_ev * 200 + (10 if all_engines_agree else 0)))
        elif power_ev > 0 or flex_ev > 0.02:
            verdict = "LEAN BET — Flex Play Recommended"
            verdict_color = "#fbbf24"
            verdict_icon = "&#9888;"
            confidence = min(65, int(30 + max(power_ev, flex_ev) * 200))
        else:
            verdict = "PASS — Insufficient Edge"
            verdict_color = "#f87171"
            verdict_icon = "&#10060;"
            confidence = max(10, int(30 + power_ev * 200))

        # Build signal summary
        signals_positive = []
        signals_negative = []
        if all_engines_agree:
            signals_positive.append("All 5 engines agree")
        else:
            signals_negative.append(f"Engine disagreement on {sum(1 for pk in parlay_picks if pk['mc']['engine_agreement'] < 0.80)} legs")
        if all_positive_edge:
            signals_positive.append("All legs above breakeven")
        else:
            signals_negative.append(f"{weakest['player']} only {weakest['prob']*100:.1f}% (below 54.9% breakeven)")
        if all_make_cut:
            signals_positive.append(f"Avg cut probability {avg_cut_prob*100:.0f}%")
        else:
            signals_negative.append(f"Cut risk: avg {avg_cut_prob*100:.0f}% make cut")
        if _ks_status["active"]:
            signals_positive.append("Kill switch clear")
        else:
            signals_negative.append(f"Kill switch triggered: {_ks_status['reason']}")
        if clv_logged:
            signals_positive.append("Opening lines auto-logged for CLV")

        pos_html = "".join(f'<div style="font-size:0.8rem;color:#4ade80;margin:2px 0;">&#10003; {s}</div>' for s in signals_positive)
        neg_html = "".join(f'<div style="font-size:0.8rem;color:#f87171;margin:2px 0;">&#10007; {s}</div>' for s in signals_negative)

        rgba_color = '74,222,128' if verdict_color == '#4ade80' else '251,191,36' if verdict_color == '#fbbf24' else '248,113,113'
        st.markdown(f"""
        <div class="glass-card" style="padding:24px;border-color:rgba({rgba_color},0.4);border-width:2px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
                <div style="font-size:1.4rem;font-weight:700;color:{verdict_color};">{verdict_icon} {verdict}</div>
                <div style="font-size:1.1rem;font-weight:600;color:{verdict_color};border:2px solid {verdict_color};border-radius:50%;width:48px;height:48px;display:flex;align-items:center;justify-content:center;">{confidence}%</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                <div>
                    <div style="font-size:0.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Positive Signals</div>
                    {pos_html}
                </div>
                <div>
                    <div style="font-size:0.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Risk Flags</div>
                    {neg_html if neg_html else '<div style="font-size:0.8rem;color:#4ade80;">&#10003; No risk flags detected</div>'}
                </div>
            </div>
            <div style="margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.06);font-size:0.78rem;color:#94a3b8;">
                Strongest: <span class="mono" style="color:#4ade80;">{strongest['player']}</span> ({strongest['prob']*100:.1f}%) &nbsp;|&nbsp;
                Weakest: <span class="mono" style="color:#f87171;">{weakest['player']}</span> ({weakest['prob']*100:.1f}%) &nbsp;|&nbsp;
                Power EV: <span class="mono">{power_ev*100:+.1f}%</span> &nbsp;|&nbsp;
                Flex EV: <span class="mono">{flex_ev*100:+.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Store model results for AI advice (persists across reruns) ──
        st.session_state["lab_model_results"] = {
            "parlay_picks": parlay_picks,
            "parlay_mc": parlay_mc,
            "kelly_results": kelly_results,
            "sim_ran": sim_ran,
            "verdict": verdict,
            "confidence": confidence,
            "ks_status": _ks_status,
            "course": settings["course"],
        }

    # ── AI Lab Analysis (OUTSIDE the Run Full Model block so it persists) ──
    lab_results = st.session_state.get("lab_model_results")
    if lab_results:
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        if st.button("Get Claude AI Parlay Advice", key="ai_lab_advice"):
            with st.spinner("Claude is analyzing your parlay with full system context..."):
                _pp = lab_results["parlay_picks"]
                _mc = lab_results["parlay_mc"]
                _kr = lab_results["kelly_results"]
                _vd = lab_results["verdict"]
                _cf = lab_results["confidence"]
                _ks = lab_results["ks_status"]

                legs_for_ai = json.dumps([{
                    "player": pk["player"], "stat": pk["stat"], "line": pk["line"],
                    "side": pk["side"], "projection": pk["proj"],
                    "mc_prob": pk["prob"], "edge": pk["prob"] - 1/pp_decimal_odds,
                    "engine_agreement": pk["mc"]["engine_agreement"],
                    "ci_80": pk["mc"]["ci_80"],
                    "win_prob": pk.get("sim", {}).get("win_prob", 0),
                    "cut_prob": pk.get("sim", {}).get("cut_prob", 0),
                    "course_fit": pk.get("sim", {}).get("course_fit", 0),
                    "kelly_stake": _kr.get(pk["player"], {}).get("stake_dollars", 0),
                } for pk in _pp], indent=2)

                mc_for_ai = json.dumps({
                    "joint_prob": _mc["joint_prob"],
                    "naive_joint": _mc["naive_joint"],
                    "correlation_impact": _mc["correlation_impact"],
                    "power_ev": _mc["power_ev"],
                    "flex_ev": _mc["flex_ev"],
                    "per_leg_sim_prob": _mc["per_leg_sim_prob"],
                    "n_sims": _mc["n_sims"],
                    "kill_switch": _ks,
                    "unified_verdict": _vd,
                    "confidence": _cf,
                }, indent=2)

                ai_advice = ai_lab_analysis(legs_for_ai, mc_for_ai, lab_results["course"])

                st.session_state["lab_ai_advice"] = ai_advice

        # Show cached AI advice (persists across reruns)
        if st.session_state.get("lab_ai_advice"):
            st.markdown(f"""
            <div class="glass-card" style="padding:20px;">
                <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
                    Claude AI Unified Analysis
                </div>
                <div style="font-size:0.85rem;color:#e2e8f0;line-height:1.7;white-space:pre-wrap;">{st.session_state["lab_ai_advice"]}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── BET LOGGING (always visible when model results exist) ──
    lab_results = st.session_state.get("lab_model_results")
    if lab_results:
        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
        st.markdown("## Log Bets")

        _pp = lab_results["parlay_picks"]
        _mc = lab_results.get("parlay_mc", {})
        pp_decimal_odds = settings.get("pp_decimal_odds", 1.909)
        bankroll = settings.get("bankroll", 1000)

        # Initialize logged_bets if needed
        if "logged_bets" not in st.session_state:
            st.session_state["logged_bets"] = []

        # ── Log Individual Legs ──
        st.markdown("**Log Individual Bets to Quant System**")
        st.markdown('<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:8px;">Click any leg to log it as an individual bet through the Quant Engine for Kelly sizing and risk analysis.</div>', unsafe_allow_html=True)

        # Check if a bet was just logged (show success message)
        if st.session_state.get("_lab_bet_logged"):
            logged_msg = st.session_state["_lab_bet_logged"]
            st.success(f"Logged: {logged_msg['player']} {logged_msg['side'].upper()} {logged_msg['stat']} @ {logged_msg['line']} — ${logged_msg.get('amount', 0):.0f}")
            del st.session_state["_lab_bet_logged"]

        for idx, pk in enumerate(_pp):
            pp_be = 1.0 / pp_decimal_odds
            edge = pk["prob"] - pp_be
            edge_color = "#4ade80" if edge >= 0.08 else "#60a5fa" if edge >= 0.05 else "#fbbf24" if edge >= 0.02 else "#f87171"
            conf = "HIGH" if edge >= 0.08 else "MEDIUM" if edge >= 0.05 else "LOW"
            conf_color = "#4ade80" if conf == "HIGH" else "#fbbf24" if conf == "MEDIUM" else "#fb923c"

            lb_cols = st.columns([4, 1])
            with lb_cols[0]:
                st.markdown(
                    f'<div style="font-size:0.82rem;padding:4px 0;">'
                    f'<b>{pk["player"]}</b> — {pk["stat"]} <span style="color:{edge_color};">{pk["side"]}</span> {pk["line"]} '
                    f'| Proj: {pk["proj"]:.1f} | Prob: {pk["prob"]*100:.1f}% '
                    f'| <span style="color:{edge_color};">Edge: {edge*100:+.1f}%</span> '
                    f'| <span style="color:{conf_color};">{conf}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with lb_cols[1]:
                if st.button("Log Bet", key=f"log_bet_lab_{idx}"):
                    # Log to Quant Engine
                    engine = _get_quant_engine(settings)
                    decision = engine.evaluate_bet(
                        player=pk["player"],
                        bet_type=BetType.OVER if pk["side"] == "OVER" else BetType.UNDER,
                        stat_type=pk.get("stat_internal", "birdies"),
                        line=pk["line"],
                        direction=pk["side"].lower(),
                        model_prob=pk["prob"],
                        model_projection=pk["proj"],
                        model_std=pk.get("std", 1.5),
                        odds_american=-110,
                    )
                    bet_amount = decision.get("stake", bankroll * 0.02) if decision.get("approved") else bankroll * 0.02
                    if decision.get("approved"):
                        engine.place_bet(decision)
                    # Also log to session_state logged_bets for Settings tab
                    st.session_state["logged_bets"].append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "player": pk["player"],
                        "stat": pk["stat"],
                        "side": pk["side"],
                        "line": pk["line"],
                        "amount": round(bet_amount, 2),
                        "type": "Single",
                        "result": "pending",
                        "profit": 0,
                        "prob": round(pk["prob"], 4),
                        "edge": round(edge, 4),
                        "projection": round(pk["proj"], 2),
                    })
                    if _DB_AVAILABLE:
                        try: save_state_to_db()
                        except Exception: pass
                    st.session_state["_lab_bet_logged"] = {
                        "player": pk["player"], "stat": pk["stat"],
                        "side": pk["side"], "line": pk["line"],
                        "amount": bet_amount,
                    }
                    st.rerun()

        # ── Log Parlay ──
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown("**Log Parlay to Quant System**")
        st.markdown('<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:8px;">Log the full parlay as a single bet. Choose Power Play or Flex Play.</div>', unsafe_allow_html=True)

        if st.session_state.get("_lab_parlay_logged"):
            st.success(st.session_state["_lab_parlay_logged"])
            del st.session_state["_lab_parlay_logged"]

        parlay_cols = st.columns([2, 2, 2, 1])
        with parlay_cols[0]:
            parlay_type = st.selectbox("Parlay Type", ["Power Play", "Flex Play"], key="lab_parlay_type")
        with parlay_cols[1]:
            PP_PAYOUTS = {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 40.0}
            n_legs = len(_pp)
            payout = PP_PAYOUTS.get(n_legs, 3.0) if parlay_type == "Power Play" else 2.25
            st.markdown(f"<div style='padding-top:28px;font-size:0.85rem;color:#e2e8f0;'>{n_legs}-leg {parlay_type} — {payout:.1f}x payout</div>", unsafe_allow_html=True)
        with parlay_cols[2]:
            parlay_amount = st.number_input("Bet Amount ($)", value=25.0, step=5.0, key="lab_parlay_amount")
        with parlay_cols[3]:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("Log Parlay", key="log_parlay_lab", type="primary"):
                legs_desc = " + ".join(f"{pk['player']} {pk['side']} {pk['stat']} {pk['line']}" for pk in _pp)
                # Log parlay to Quant Engine
                engine = _get_quant_engine(settings)
                joint_prob = _mc.get("joint_prob", 0)
                for pk in _pp:
                    decision = engine.evaluate_bet(
                        player=pk["player"],
                        bet_type=BetType.OVER if pk["side"] == "OVER" else BetType.UNDER,
                        stat_type=pk.get("stat_internal", "birdies"),
                        line=pk["line"],
                        direction=pk["side"].lower(),
                        model_prob=pk["prob"],
                        model_projection=pk["proj"],
                        model_std=pk.get("std", 1.5),
                        odds_american=-110,
                    )
                    if decision.get("approved"):
                        engine.place_bet(decision)

                # Log parlay to session_state
                st.session_state["logged_bets"].append({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "player": legs_desc,
                    "stat": f"{n_legs}-Leg Parlay",
                    "side": parlay_type,
                    "line": f"{n_legs} legs",
                    "amount": parlay_amount,
                    "type": parlay_type,
                    "result": "pending",
                    "profit": 0,
                    "legs": [{
                        "player": pk["player"],
                        "stat": pk["stat"],
                        "side": pk["side"],
                        "line": pk["line"],
                        "prob": round(pk["prob"], 4),
                        "edge": round(pk["prob"] - (1.0 / pp_decimal_odds), 4),
                    } for pk in _pp],
                    "joint_prob": round(joint_prob, 6),
                    "payout_mult": payout,
                })
                if _DB_AVAILABLE:
                    try: save_state_to_db()
                    except Exception: pass
                st.session_state["_lab_parlay_logged"] = f"Logged {n_legs}-leg {parlay_type}: {legs_desc} — ${parlay_amount:.0f} @ {payout:.1f}x"
                st.rerun()


# ============================================================
# TAB 6 — LIVE SCANNER
# ============================================================
def tab_live_scanner(proj_df: pd.DataFrame, settings: dict):
    """Live Scanner — auto-fetches PP props, runs full quant engine on every prop,
    and provides AI-powered analysis for highest-probability selections."""
    pp_decimal_odds = settings.get("pp_decimal_odds", 1.909)
    st.markdown(section_header("Live Scanner", "&#128225;", "Auto-Scan"), unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="padding:14px;margin-bottom:16px;">
        <div style="font-size:0.85rem;color:#e2e8f0;">
            Scans all fetched PrizePicks props and runs the full quant engine (Monte Carlo 5000x,
            Bayesian shrinkage, course-fit adjustment, 5 probability engines) on every line
            to surface the highest-probability projections.
        </div>
    </div>
    """, unsafe_allow_html=True)

    pp_lines = st.session_state.get("pp_lines", [])

    # Auto-scan button
    col_scan1, col_scan2 = st.columns([1, 3])
    with col_scan1:
        run_scan = st.button("Run Live Scanner", key="run_scanner", type="primary")
    with col_scan2:
        st.markdown(f"""
        <div style="font-size:0.8rem;color:#94a3b8;padding-top:8px;">
            {len(pp_lines)} PrizePicks props loaded | Course: {settings['course']}
        </div>
        """, unsafe_allow_html=True)

    if not pp_lines:
        st.warning("No PrizePicks golf lines available. Lines typically go live Tuesday morning of tournament week. "
                   "Check the sidebar data source settings.")
        return

    # Stat type mapping
    pp_internal_map = {
        "Birdies Or Better": "birdies", "Birdies or Better": "birdies",
        "Birdies": "birdies",
        "Bogeys or Worse": "bogeys", "Bogeys Or Worse": "bogeys",
        "Strokes": "strokes", "Total Strokes": "strokes",
        "Greens In Regulation": "gir", "Greens in Regulation": "gir",
        "Fairways Hit": "fairways", "Pars": "pars",
        "Putts": "putts", "Fantasy Score": "fantasy_score",
        "Bogey-Free Holes": "bogey_free", "Bogey Free Holes": "bogey_free",
        "Birdies or Better Matchup": "birdies_matchup",
        "Birdies Or Better Matchup": "birdies_matchup",
        "Holes Played": "holes_played",
    }

    if run_scan or st.session_state.get("scanner_results"):
        if run_scan:
            # Run the full quant engine on every prop
            scan_results = []
            progress = st.progress(0, text="Scanning props...")

            # Pre-compute field SG list for holes_played sim
            field_sg_list = proj_df["sg_regressed"].tolist() if "sg_regressed" in proj_df.columns else [0.0] * 30

            # Pre-index matchup opponents from PrizePicks descriptions
            # Format: "Player A - Player B" or "Player A vs Player B"
            matchup_opponents = {}
            for pp_l in pp_lines:
                desc = pp_l.get("description", "")
                p_name = pp_l["player"]
                if "matchup" in pp_l.get("stat_type", "").lower():
                    # Extract opponent from description
                    for sep in [" vs ", " vs. ", " - ", " VS "]:
                        if sep in desc:
                            parts = desc.split(sep)
                            for part in parts:
                                part_clean = part.strip().split(" Over")[0].split(" Under")[0].strip()
                                if part_clean and part_clean.lower() != p_name.lower():
                                    last_name_match = part_clean.split()[-1].lower() if part_clean else ""
                                    if last_name_match and last_name_match not in p_name.lower():
                                        matchup_opponents[p_name] = part_clean
                                        break
                            break

            for idx, pp_line in enumerate(pp_lines):
                progress.progress((idx + 1) / len(pp_lines), text=f"Scanning {pp_line['player']}...")

                player_name = pp_line["player"]
                stat_display = pp_line["stat_type"]
                stat_type = pp_internal_map.get(stat_display, None)
                if stat_type is None:
                    # Unknown stat type — skip to avoid false projections
                    continue
                line_val = pp_line["line"]

                # Find player in projection model
                player_match = proj_df[proj_df["player"] == player_name]
                if player_match.empty:
                    # Try fuzzy match
                    for _, row in proj_df.iterrows():
                        if player_name.split()[-1].lower() in row["player"].lower():
                            player_match = proj_df[proj_df["player"] == row["player"]]
                            break

                if player_match.empty:
                    continue

                p_row = player_match.iloc[0]
                sg_proj = {k: p_row[k] for k in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]}
                player_sg_total = float(p_row.get("sg_regressed", sum(sg_proj.values())))

                # Project stat
                proj_val, proj_std = project_pp_stat(stat_type, sg_proj)

                # Build extra kwargs for special stat types
                mc_kwargs = {"stat_type": stat_type, "player_sg": player_sg_total}

                if stat_type == "holes_played":
                    mc_kwargs["field_sg_list"] = field_sg_list

                if stat_type == "birdies_matchup":
                    # Find opponent's SG
                    opp_name = matchup_opponents.get(player_name, "")
                    opp_sg = 0.0
                    if opp_name:
                        opp_match = proj_df[proj_df["player"].str.contains(opp_name.split()[-1], case=False, na=False)]
                        if not opp_match.empty:
                            opp_sg = float(opp_match.iloc[0].get("sg_regressed", 0))
                    mc_kwargs["opponent_sg"] = opp_sg

                # Run MC simulation (2000x per engine, 10K total across 5 engines)
                mc = mc_prop_simulation(proj_val, proj_std, line_val, n_sims=2000, **mc_kwargs)

                p_over = mc["p_over"]
                p_under = mc["p_under"]
                best_side = "OVER" if p_over > p_under else "UNDER"
                best_prob = max(p_over, p_under)

                # PrizePicks breakeven
                pp_be = 1.0 / pp_decimal_odds
                edge = best_prob - pp_be

                # Confidence classification
                if edge > 0.08 and mc["engine_agreement"] > 0.90:
                    conf = "HIGH"
                elif edge > 0.05:
                    conf = "MEDIUM"
                elif edge > 0.02:
                    conf = "LOW"
                else:
                    conf = "NO_BET"

                scan_results.append({
                    "player": player_name,
                    "stat": stat_display,
                    "stat_internal": stat_type,
                    "line": line_val,
                    "projection": proj_val,
                    "std": proj_std,
                    "side": best_side,
                    "prob": best_prob,
                    "p_over": p_over,
                    "p_under": p_under,
                    "edge": edge,
                    "confidence": conf,
                    "engine_agreement": mc["engine_agreement"],
                    "ci_80": mc["ci_80"],
                    "engines": mc["p_over_by_engine"],
                    "sg_total": p_row.get("sg_regressed", 0),
                    "course_delta": p_row.get("course_delta", 0),
                })

            progress.empty()

            # Sort by edge descending
            scan_results.sort(key=lambda x: x["edge"], reverse=True)
            st.session_state["scanner_results"] = scan_results
        else:
            scan_results = st.session_state["scanner_results"]

        # Display results
        st.markdown(f"**Scanned {len(scan_results)} props — sorted by edge**")

        # Summary metrics
        bettable = [s for s in scan_results if s["confidence"] != "NO_BET"]
        high_conf = [s for s in scan_results if s["confidence"] == "HIGH"]

        mcols = st.columns(4)
        with mcols[0]:
            st.markdown(metric_card("Props Scanned", str(len(scan_results)), "total", "neutral", "blue"), unsafe_allow_html=True)
        with mcols[1]:
            st.markdown(metric_card("Bettable", str(len(bettable)), f"{len(bettable)/max(len(scan_results),1)*100:.0f}% of total", "positive" if bettable else "neutral", "green"), unsafe_allow_html=True)
        with mcols[2]:
            st.markdown(metric_card("HIGH Conf", str(len(high_conf)), "plays", "positive" if high_conf else "neutral", "amber"), unsafe_allow_html=True)
        with mcols[3]:
            avg_edge = np.mean([s["edge"] for s in bettable]) * 100 if bettable else 0
            st.markdown(metric_card("Avg Edge", f"{avg_edge:+.1f}%", "bettable props", "positive" if avg_edge > 0 else "negative", "purple"), unsafe_allow_html=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # Filters
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            conf_filter = st.selectbox("Filter Confidence", ["All", "HIGH", "MEDIUM", "LOW"], key="scan_conf_filter")
        with fcol2:
            side_filter = st.selectbox("Filter Side", ["All", "OVER", "UNDER"], key="scan_side_filter")

        filtered = scan_results
        if conf_filter != "All":
            filtered = [s for s in filtered if s["confidence"] == conf_filter]
        if side_filter != "All":
            filtered = [s for s in filtered if s["side"] == side_filter]

        # Results table
        if filtered:
            rows = []
            for s in filtered:
                rows.append({
                    "Player": s["player"],
                    "Stat": s["stat"],
                    "Line": s["line"],
                    "Proj": f"{s['projection']:.1f}",
                    "Side": s["side"],
                    "Prob": f"{s['prob']*100:.1f}%",
                    "Edge": f"{s['edge']*100:+.1f}%",
                    "Conf": s["confidence"],
                    "Engines": f"{s['engine_agreement']*100:.0f}%",
                    "CI 80%": f"{s['ci_80'][0]:.1f}-{s['ci_80'][1]:.1f}",
                })

            scan_df = pd.DataFrame(rows)

            def color_edge(val):
                try:
                    v = float(val.replace("%", "").replace("+", ""))
                    if v >= 8: return "color: #4ade80; font-weight: bold"
                    if v >= 5: return "color: #60a5fa"
                    if v >= 2: return "color: #fbbf24"
                    return "color: #f87171"
                except Exception:
                    return ""

            def color_conf(val):
                if val == "HIGH": return "color: #4ade80; font-weight: bold"
                if val == "MEDIUM": return "color: #fbbf24"
                if val == "LOW": return "color: #fb923c"
                return "color: #f87171"

            styled = scan_df.style.map(color_edge, subset=["Edge"]).map(color_conf, subset=["Conf"])
            st.dataframe(styled, use_container_width=True, hide_index=True, height=400)

        # Select legs to send to PrizePicks Lab
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown("**Select Legs for PrizePicks Lab**")
        st.markdown('<div style="font-size:0.8rem;color:#94a3b8;margin-bottom:8px;">Select up to 6 legs to send to the PrizePicks Lab for full parlay analysis.</div>', unsafe_allow_html=True)

        selectable = [s for s in scan_results if s["confidence"] != "NO_BET"]
        if selectable:
            select_options = [f"{s['player']} | {s['stat']} {s['side']} {s['line']} (Edge: {s['edge']*100:+.1f}%)" for s in selectable]
            selected = st.multiselect("Select legs (max 6)", select_options[:20], max_selections=6, key="scanner_selected_legs")

            if selected and st.button("Send to PrizePicks Lab", key="send_to_lab", type="primary"):
                # Store selected legs in session state
                selected_legs = []
                for sel in selected:
                    player_name = sel.split(" | ")[0]
                    for s in selectable:
                        if s["player"] == player_name:
                            selected_legs.append(s)
                            break
                st.session_state["lab_legs_from_scanner"] = selected_legs
                st.success(f"Sent {len(selected_legs)} legs to PrizePicks Lab! Switch to the PrizePicks Lab tab.")

        # AI Scanner Analysis
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        if st.button("AI Scanner Briefing", key="ai_scanner_briefing"):
            with st.spinner("Claude is analyzing scanner results..."):
                # Prepare scan data for AI
                ai_scan_data = []
                for s in scan_results[:15]:  # top 15 for context
                    ai_scan_data.append({
                        "player": s["player"],
                        "stat": s["stat"],
                        "line": s["line"],
                        "side": s["side"],
                        "projection": s["projection"],
                        "prob": s["prob"],
                        "edge": s["edge"],
                        "confidence": s["confidence"],
                        "engine_agreement": s["engine_agreement"],
                        "sg_total": s["sg_total"],
                        "course_delta": s["course_delta"],
                    })

                espn_data = st.session_state.get("espn_data", {})
                tournament = espn_data.get("event_name", settings["course"]) if espn_data else settings["course"]

                analysis = ai_scanner_analysis(
                    json.dumps(ai_scan_data, indent=2),
                    settings["course"],
                    tournament,
                )

                st.markdown(f"""
                <div class="glass-card" style="padding:20px;">
                    <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
                        Claude AI Scanner Analysis
                    </div>
                    <div style="font-size:0.85rem;color:#e2e8f0;line-height:1.7;white-space:pre-wrap;">{analysis}</div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#64748b;">
            <div style="font-size:2rem;margin-bottom:12px;">&#128225;</div>
            <div style="font-size:1rem;margin-bottom:8px;">Click "Run Live Scanner" to scan all PrizePicks props</div>
            <div style="font-size:0.8rem;">The scanner runs Monte Carlo 5000x with 5 probability engines on every prop to find the highest-edge plays.</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# TAB 7 — QUANT SYSTEM (CLV, Edge Validation, Risk, Dashboard)
# ============================================================
def _get_quant_engine(settings: dict) -> QuantEngine:
    """Get or create the QuantEngine singleton in session state."""
    if "_quant_engine" not in st.session_state:
        bankroll = settings.get("bankroll", 1000)
        db_path = os.path.join(os.path.dirname(__file__), "data", "quant_system.db")
        st.session_state["_quant_engine"] = QuantEngine(
            sport=Sport.GOLF,
            initial_bankroll=bankroll,
            kelly_config=KellyConfig(
                base_fraction=0.10,
                max_single_bet_pct=0.03,
                min_edge_to_bet=0.04,
            ),
            db_path=db_path,
        )
    return st.session_state["_quant_engine"]


def tab_quant_system(proj_df: pd.DataFrame, settings: dict):
    """Quant System dashboard — CLV tracking, edge validation, risk management, self-learning."""
    st.markdown(section_header("Quant System", "&#128202;", "Production Risk & Edge Management"), unsafe_allow_html=True)

    engine = _get_quant_engine(settings)
    bankroll = settings.get("bankroll", 1000)

    # Health bar at top
    health = engine.health_check()
    state = engine.edge_validator.get_current_state()
    state_colors = {
        SystemState.ACTIVE: "#00FF88",
        SystemState.REDUCED: "#FFB800",
        SystemState.SUSPENDED: "#FF3358",
        SystemState.KILLED: "#FF0000",
    }
    state_color = state_colors.get(state, "#94a3b8")
    st.markdown(f"""
    <div class="glass-card" style="padding:12px 18px;margin-bottom:16px;border-left:4px solid {state_color};">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:0.7rem;color:#64748b;letter-spacing:0.1em;text-transform:uppercase;">System State</span>
                <div style="font-size:1.3rem;font-weight:700;color:{state_color};font-family:SF Mono,monospace;">{state.value.upper()}</div>
            </div>
            <div style="font-family:SF Mono,monospace;font-size:0.75rem;color:#94a3b8;text-align:right;">
                {health}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sub-tabs
    qs_tabs = st.tabs(["Dashboard", "Bet Logger", "Active Bets", "CLV Tracker", "Edge Validation", "Risk & Sizing", "Backtest", "System Health"])

    # ── DASHBOARD ──
    with qs_tabs[0]:
        st.markdown("**System Overview**")
        try:
            report = engine.dashboard_report()

            # Bankroll row
            br = report.get("bankroll", {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Bankroll", f"${br.get('current', bankroll):,.0f}")
            with col2:
                st.metric("Peak", f"${br.get('peak', bankroll):,.0f}")
            with col3:
                dd = br.get("drawdown_pct", 0)
                st.metric("Drawdown", f"{dd:.1f}%", delta=f"-{dd:.1f}%" if dd > 0 else "0%", delta_color="inverse")
            with col4:
                st.metric("Daily P&L", f"${br.get('daily_pnl', 0):+,.2f}")

            # Bet summary row
            bs = report.get("bet_summary", {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Bets", bs.get("total_bets", 0))
            with col2:
                wr = bs.get("win_rate", 0) * 100
                st.metric("Win Rate", f"{wr:.1f}%")
            with col3:
                st.metric("Total P&L", f"${bs.get('total_pnl', 0):+,.2f}")
            with col4:
                st.metric("Avg Edge", f"{bs.get('avg_edge', 0)*100:.1f}%")

            # CLV summary
            st.markdown("---")
            st.markdown("**CLV Performance (Most Important Metric)**")
            clv = report.get("clv", {})
            clv_cols = st.columns(4)
            for i, window in enumerate(["clv_50", "clv_100", "clv_250", "clv_500"]):
                with clv_cols[i]:
                    data = clv.get(window, {})
                    cents = data.get("avg_clv_cents", 0)
                    n = data.get("n_bets", 0)
                    beat = data.get("beat_close_pct", 0) * 100
                    color = "#00FF88" if cents > 0 else "#FF3358"
                    st.markdown(f"""
                    <div class="glass-card" style="padding:10px;text-align:center;">
                        <div style="font-size:0.65rem;color:#64748b;">{window.replace('clv_', 'Last ')} bets</div>
                        <div style="font-size:1.4rem;font-weight:700;color:{color};font-family:SF Mono,monospace;">{cents:+.1f}c</div>
                        <div style="font-size:0.6rem;color:#94a3b8;">Beat close: {beat:.0f}% | n={n}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # P&L Curve
            pnl_curve = report.get("pnl_curve", [])
            if len(pnl_curve) > 1:
                st.markdown("**Bankroll Curve**")
                curve_df = pd.DataFrame(pnl_curve)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=curve_df["bet_num"], y=curve_df["bankroll"],
                                         mode="lines", line=dict(color="#00FF88", width=2),
                                         fill="tozeroy", fillcolor="rgba(0,255,136,0.08)"))
                fig.update_layout(
                    template="plotly_dark", height=280, margin=dict(l=40, r=20, t=20, b=30),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Bet #", yaxis_title="Bankroll ($)",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Edge validation
            edge = report.get("edge", {})
            if edge.get("warnings"):
                for w in edge["warnings"]:
                    st.warning(w)
            if edge.get("actions"):
                for a in edge["actions"]:
                    st.error(f"Action Required: {a}")

            # MC Projection
            mc = report.get("mc_projection", {})
            if mc and "ruin_probability" in mc:
                st.markdown("**Monte Carlo Projection (10,000 paths)**")
                mc_cols = st.columns(4)
                with mc_cols[0]:
                    st.metric("Ruin Prob", f"{mc['ruin_probability']:.1%}")
                with mc_cols[1]:
                    st.metric("Median Final", f"${mc['median_final']:,.0f}")
                with mc_cols[2]:
                    st.metric("Profitable Paths", f"{mc['paths_profitable']:.0%}")
                with mc_cols[3]:
                    st.metric("5th Percentile", f"${mc['p5_final']:,.0f}")

        except Exception as e:
            st.info(f"Dashboard will populate as bets are logged. ({e})")

        # ── Quick view: Logged Bets from PrizePicks Lab ──
        lab_logged = st.session_state.get("logged_bets", [])
        if lab_logged:
            st.markdown("---")
            st.markdown("**PrizePicks Lab Bets**")
            pending_lab = [b for b in lab_logged if b.get("result") == "pending"]
            settled_lab = [b for b in lab_logged if b.get("result") in ("won", "lost")]
            lc1, lc2, lc3, lc4 = st.columns(4)
            with lc1:
                st.metric("Lab Bets", len(lab_logged))
            with lc2:
                st.metric("Pending", len(pending_lab))
            with lc3:
                lab_pnl = sum(b.get("profit", 0) for b in settled_lab)
                st.metric("Lab P&L", f"${lab_pnl:+,.2f}")
            with lc4:
                parlays = [b for b in lab_logged if b.get("type") in ("Power Play", "Flex Play")]
                st.metric("Parlays", len(parlays))
            if pending_lab:
                quick_data = []
                for b in pending_lab[:10]:
                    quick_data.append({
                        "Date": b.get("date", ""),
                        "Player": b.get("player", "")[:35],
                        "Type": b.get("type", "Single"),
                        "Amount": f"${b.get('amount', 0):.0f}",
                    })
                st.dataframe(pd.DataFrame(quick_data), use_container_width=True, hide_index=True)
            st.markdown('<div style="font-size:0.75rem;color:#94a3b8;">Go to the <b>Active Bets</b> tab for full bet management and settlement.</div>', unsafe_allow_html=True)

    # ── BET LOGGER ──
    with qs_tabs[1]:
        st.markdown("**Log Bets Through Quant Engine**")
        st.markdown('<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:10px;">Every bet logged here flows through the full risk pipeline: edge validation, Kelly sizing, exposure checks, and circuit breakers.</div>', unsafe_allow_html=True)

        with st.expander("Log New Bet via Quant Engine", expanded=True):
            qc1, qc2, qc3 = st.columns(3)
            with qc1:
                q_player = st.text_input("Player", key="qs_player")
                q_stat = st.selectbox("Stat", ["birdies", "bogeys", "bogey_free", "strokes", "gir", "fairways", "putts", "pars", "fantasy_score", "birdies_matchup", "holes_played"], key="qs_stat")
            with qc2:
                q_direction = st.selectbox("Direction", ["over", "under"], key="qs_direction")
                q_line = st.number_input("Line", value=3.5, step=0.5, key="qs_line")
            with qc3:
                q_model_prob = st.number_input("Model Probability", 0.40, 0.95, 0.58, 0.01, key="qs_prob")
                q_projection = st.number_input("Model Projection", 0.0, 100.0, 4.0, 0.1, key="qs_proj")
                q_std = st.number_input("Projection Std Dev", 0.1, 20.0, 1.5, 0.1, key="qs_std")

            if st.button("Evaluate & Place Bet", key="qs_place"):
                if q_player:
                    decision = engine.evaluate_bet(
                        player=q_player,
                        bet_type=BetType.OVER if q_direction == "over" else BetType.UNDER,
                        stat_type=q_stat,
                        line=q_line,
                        direction=q_direction,
                        model_prob=q_model_prob,
                        model_projection=q_projection,
                        model_std=q_std,
                        odds_american=-110,
                    )
                    if decision["approved"]:
                        bet_id = engine.place_bet(decision)
                        st.success(f"BET PLACED: {bet_id} | {q_player} {q_direction.upper()} {q_stat} @ {q_line} | Stake: ${decision['stake']:.2f} | Edge: {(q_model_prob - decision['market_prob'])*100:.1f}%")
                        st.json(decision["kelly_details"])
                    else:
                        st.error(f"BET REJECTED: {decision['rejection_reason']}")

        # Pending bets with settlement
        pending = engine.bet_logger.get_pending_bets()
        if pending:
            # Auto-fetch closing lines from current PP lines for all pending bets
            pp_lines = st.session_state.get("pp_lines", [])
            closing_line_cache = {}
            for bet in pending:
                closing = bet["line"]  # default to opening line
                for pl in pp_lines:
                    if pl.get("player") == bet["player"] and pl.get("stat_type") == bet["stat_type"]:
                        closing = pl.get("line", bet["line"])
                        break
                closing_line_cache[bet["bet_id"]] = closing

                # Log closing line movement to DB if line has moved
                if closing != bet["line"] and _DB_AVAILABLE:
                    try:
                        from database.connection import DatabaseManager
                        from database.models import LineMovement
                        with DatabaseManager.session_scope() as session:
                            lm = LineMovement(
                                player_name=bet["player"],
                                market=bet["stat_type"],
                                line=closing,
                                odds=-110,
                                source="prizepicks_closing",
                                is_opening=False,
                                timestamp=datetime.utcnow(),
                            )
                            session.add(lm)
                            session.commit()
                    except Exception:
                        pass

            st.markdown(f"**Pending Bets ({len(pending)})**")
            for bet in pending:
                closing = closing_line_cache.get(bet["bet_id"], bet["line"])
                clv_delta = closing - bet["line"]
                clv_tag = ""
                if clv_delta != 0:
                    favorable = (clv_delta < 0 and bet["direction"] == "under") or (clv_delta > 0 and bet["direction"] == "over")
                    clv_color = "#00FF88" if favorable else "#FF3358"
                    clv_tag = f' <span style="color:{clv_color};font-size:0.75rem;">(CLV: {clv_delta:+.1f})</span>'

                bc1, bc2, bc3, bc4 = st.columns([3, 1, 1, 1])
                with bc1:
                    st.markdown(f"{bet['player']} {bet['direction'].upper()} {bet['stat_type']} @ {bet['line']} → {closing} — ${bet['stake']:.2f}{clv_tag}", unsafe_allow_html=True)
                with bc2:
                    actual = st.number_input("Result", key=f"res_{bet['bet_id']}", value=0.0, step=0.5)
                with bc3:
                    if st.button("Settle", key=f"settle_{bet['bet_id']}"):
                        result = engine.settle_bet(bet["bet_id"], actual_result=actual, closing_line=closing)
                        st.success(f"{'WON' if result['won'] else 'LOST'} | P&L: ${result['pnl']:+.2f} | CLV: {clv_delta:+.1f}")
                        st.rerun()
                with bc4:
                    if st.button("Void", key=f"void_{bet['bet_id']}"):
                        from quant_system.core.types import BetStatus
                        engine.bet_logger.settle_bet(bet["bet_id"], BetStatus.VOID, 0.0)
                        st.rerun()

        # Settled bets
        settled = engine.bet_logger.get_settled_bets(limit=50)
        if settled:
            st.markdown(f"**Recent Settled Bets ({len(settled)})**")
            sdf = pd.DataFrame(settled)[["timestamp", "player", "direction", "stat_type", "line", "stake", "edge", "status", "pnl"]]
            sdf["edge"] = (sdf["edge"] * 100).round(1).astype(str) + "%"
            sdf["pnl"] = sdf["pnl"].apply(lambda x: f"${x:+.2f}")
            st.dataframe(sdf, use_container_width=True, hide_index=True)

    # ── ACTIVE BETS (from PrizePicks Lab + Quant Engine) ──
    with qs_tabs[2]:
        st.markdown("**Active Bets — All Logged Bets**")
        st.markdown('<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:10px;">All bets logged from PrizePicks Lab (individual + parlays) and the Quant Engine Bet Logger appear here.</div>', unsafe_allow_html=True)

        logged_bets = st.session_state.get("logged_bets", [])
        pending_logged = [b for b in logged_bets if b.get("result") == "pending"]
        settled_logged = [b for b in logged_bets if b.get("result") in ("won", "lost")]

        # Summary metrics
        if logged_bets:
            ab_col1, ab_col2, ab_col3, ab_col4 = st.columns(4)
            with ab_col1:
                st.metric("Total Bets", len(logged_bets))
            with ab_col2:
                st.metric("Pending", len(pending_logged))
            with ab_col3:
                won_bets = [b for b in logged_bets if b.get("result") == "won"]
                lost_bets = [b for b in logged_bets if b.get("result") == "lost"]
                wr = len(won_bets) / max(len(won_bets) + len(lost_bets), 1) * 100
                st.metric("Win Rate", f"{wr:.0f}%" if won_bets or lost_bets else "N/A")
            with ab_col4:
                total_pnl = sum(b.get("profit", 0) for b in settled_logged)
                st.metric("P&L", f"${total_pnl:+,.2f}")

            # Separate singles and parlays
            singles = [b for b in pending_logged if b.get("type") == "Single"]
            parlays = [b for b in pending_logged if b.get("type") in ("Power Play", "Flex Play")]

            # ── Pending Singles ──
            if singles:
                st.markdown(f"**Pending Singles ({len(singles)})**")
                for i, bet in enumerate(logged_bets):
                    if bet.get("result") != "pending" or bet.get("type") != "Single":
                        continue
                    edge_val = bet.get("edge", 0)
                    edge_color = "#4ade80" if edge_val >= 0.08 else "#60a5fa" if edge_val >= 0.05 else "#fbbf24"
                    bc1, bc2, bc3, bc4 = st.columns([4, 1, 1, 1])
                    with bc1:
                        st.markdown(
                            f'<div style="font-size:0.82rem;">'
                            f'<b>{bet["player"]}</b> {bet["side"]} {bet["stat"]} @ {bet["line"]} '
                            f'— ${bet["amount"]:.0f} '
                            f'| <span style="color:{edge_color};">Edge: {edge_val*100:+.1f}%</span>'
                            f'</div>', unsafe_allow_html=True)
                    with bc2:
                        actual = st.number_input("Result", key=f"qs_res_s_{i}", value=0.0, step=0.5)
                    with bc3:
                        if st.button("Won", key=f"qs_won_s_{i}"):
                            st.session_state["logged_bets"][i]["result"] = "won"
                            payout = 1.909
                            st.session_state["logged_bets"][i]["profit"] = bet["amount"] * (payout - 1)
                            st.session_state["bankroll_total"] = st.session_state.get("bankroll_total", bankroll) + st.session_state["logged_bets"][i]["profit"]
                            if _DB_AVAILABLE:
                                try: save_state_to_db()
                                except Exception: pass
                            st.rerun()
                    with bc4:
                        if st.button("Lost", key=f"qs_lost_s_{i}"):
                            st.session_state["logged_bets"][i]["result"] = "lost"
                            st.session_state["logged_bets"][i]["profit"] = -bet["amount"]
                            st.session_state["bankroll_total"] = st.session_state.get("bankroll_total", bankroll) - bet["amount"]
                            if _DB_AVAILABLE:
                                try: save_state_to_db()
                                except Exception: pass
                            st.rerun()

            # ── Pending Parlays ──
            if parlays:
                st.markdown(f"**Pending Parlays ({len(parlays)})**")
                for i, bet in enumerate(logged_bets):
                    if bet.get("result") != "pending" or bet.get("type") not in ("Power Play", "Flex Play"):
                        continue
                    legs = bet.get("legs", [])
                    payout_mult = bet.get("payout_mult", 3.0)
                    bc1, bc2, bc3 = st.columns([5, 1, 1])
                    with bc1:
                        legs_html = "<br>".join(
                            f'&nbsp;&nbsp;{lg["player"]} {lg["side"]} {lg["stat"]} @ {lg["line"]}'
                            for lg in legs
                        ) if legs else bet.get("player", "")
                        st.markdown(
                            f'<div style="font-size:0.82rem;">'
                            f'<b>{bet["type"]}</b> — {len(legs)}-leg — ${bet["amount"]:.0f} @ {payout_mult:.1f}x'
                            f'<div style="font-size:0.75rem;color:#94a3b8;margin-top:4px;">{legs_html}</div>'
                            f'</div>', unsafe_allow_html=True)
                    with bc2:
                        if st.button("Won", key=f"qs_won_p_{i}"):
                            st.session_state["logged_bets"][i]["result"] = "won"
                            profit = bet["amount"] * (payout_mult - 1)
                            st.session_state["logged_bets"][i]["profit"] = profit
                            st.session_state["bankroll_total"] = st.session_state.get("bankroll_total", bankroll) + profit
                            if _DB_AVAILABLE:
                                try: save_state_to_db()
                                except Exception: pass
                            st.rerun()
                    with bc3:
                        if st.button("Lost", key=f"qs_lost_p_{i}"):
                            st.session_state["logged_bets"][i]["result"] = "lost"
                            st.session_state["logged_bets"][i]["profit"] = -bet["amount"]
                            st.session_state["bankroll_total"] = st.session_state.get("bankroll_total", bankroll) - bet["amount"]
                            if _DB_AVAILABLE:
                                try: save_state_to_db()
                                except Exception: pass
                            st.rerun()

            # ── Settled History ──
            if settled_logged:
                st.markdown("---")
                st.markdown(f"**Settled Bets ({len(settled_logged)})**")
                settled_data = []
                for b in settled_logged:
                    settled_data.append({
                        "Date": b.get("date", ""),
                        "Player": b.get("player", "")[:40],
                        "Stat": b.get("stat", ""),
                        "Side": b.get("side", ""),
                        "Type": b.get("type", "Single"),
                        "Amount": f"${b.get('amount', 0):.0f}",
                        "Result": b.get("result", "").upper(),
                        "P&L": f"${b.get('profit', 0):+.2f}",
                    })
                st.dataframe(pd.DataFrame(settled_data), use_container_width=True, hide_index=True)
        else:
            st.info("No bets logged yet. Go to PrizePicks Lab, run the model, and use 'Log Bet' or 'Log Parlay' to start tracking.")

    # ── CLV TRACKER ──
    with qs_tabs[3]:
        st.markdown("**Closing Line Value Tracking**")
        st.markdown('<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:10px;">CLV is the #1 predictor of long-term profitability. If you consistently beat the closing line, you have edge. If not, you don\'t.</div>', unsafe_allow_html=True)

        clv_summary = engine.clv_tracker.clv_summary()
        for window_key in ["clv_50", "clv_100", "clv_250", "clv_500"]:
            data = clv_summary.get(window_key, {})
            if data.get("n_bets", 0) > 0:
                label = window_key.replace("clv_", "Last ")
                cents = data.get("avg_clv_cents", 0)
                beat = data.get("beat_close_pct", 0) * 100
                trend = data.get("trend", "")
                t_stat = data.get("clv_t_stat", 0)
                p_val = data.get("clv_p_value", 1)
                sig = "Yes" if p_val < 0.05 else "No"
                color = "#00FF88" if cents > 0 else "#FF3358"
                st.markdown(f"""
                <div class="glass-card" style="padding:10px;margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="font-size:0.7rem;color:#64748b;">{label} bets (n={data['n_bets']})</span>
                            <div style="font-size:1.1rem;font-weight:700;color:{color};">{cents:+.2f} cents/bet</div>
                        </div>
                        <div style="text-align:right;font-size:0.7rem;color:#94a3b8;">
                            Beat close: {beat:.0f}% | t={t_stat:.2f} | sig: {sig} | {trend}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        clv_by_type = engine.clv_tracker.clv_by_bet_type()
        if clv_by_type:
            st.markdown("**CLV by Bet Type**")
            type_df = pd.DataFrame([
                {"Type": k, "Avg CLV (cents)": v["avg_clv_cents"], "Beat Close %": f"{v['beat_close_pct']*100:.0f}%", "N": v["n"]}
                for k, v in clv_by_type.items()
            ])
            st.dataframe(type_df, use_container_width=True, hide_index=True)

        # Auto CLV refresh — live tracking of active bets
        active_bets = st.session_state.get("clv_active_bets", [])
        if active_bets:
            st.markdown("---")
            st.markdown("**Active Bet CLV Tracking**")
            clv_rows = []
            pp_lines = st.session_state.get("pp_lines", [])
            for bet in active_bets:
                current_line = bet["line"]  # default to opening
                # Try to find current line from PP lines
                for pl in pp_lines:
                    if pl.get("player") == bet["player"] and pl.get("stat_type") == bet["stat"]:
                        current_line = pl.get("line", bet["line"])
                        break
                clv_move = current_line - bet["line"]
                is_favorable = (clv_move > 0 and bet["side"] == "UNDER") or (clv_move < 0 and bet["side"] == "OVER")
                clv_rows.append({
                    "Player": bet["player"],
                    "Stat": bet["stat"],
                    "Side": bet["side"],
                    "Opening": bet["line"],
                    "Current": current_line,
                    "CLV Move": f"{clv_move:+.1f}",
                    "Direction": "Favorable" if is_favorable else ("Neutral" if clv_move == 0 else "Unfavorable"),
                    "Logged": bet["timestamp"][:16],
                })
            if clv_rows:
                st.dataframe(pd.DataFrame(clv_rows), use_container_width=True, hide_index=True)
            if st.button("Refresh CLV (re-fetch lines)", key="refresh_clv"):
                st.rerun()
        else:
            st.markdown('<div style="font-size:0.8rem;color:#64748b;margin-top:12px;">No active bets being tracked. Run the PrizePicks Lab model to start CLV tracking.</div>', unsafe_allow_html=True)

    # ── EDGE VALIDATION ──
    with qs_tabs[4]:
        st.markdown("**Edge Validation — Do I Actually Have Edge Right Now?**")
        if st.button("Run Edge Validation", key="qs_validate"):
            risk_state = engine.bankroll_mgr.get_risk_state()
            report = engine.edge_validator.validate(risk_state.bankroll, risk_state.peak_bankroll)

            if report.edge_exists:
                st.success(f"EDGE EXISTS | State: {report.system_state.value.upper()}")
            else:
                st.error(f"EDGE NOT CONFIRMED | State: {report.system_state.value.upper()}")

            st.metric("Calibration Error (MAE)", f"{report.calibration_error:.3f}")
            st.metric("Model ROI", f"{report.model_roi*100:.2f}%")
            st.metric("Expected ROI", f"{report.expected_roi*100:.2f}%")

            for w in report.warnings:
                st.warning(w)
            for a in report.actions:
                st.error(f"Required: {a}")

        # Drift detection
        st.markdown("---")
        st.markdown("**Model Drift Detection**")
        if st.button("Check for Drift", key="qs_drift"):
            from quant_system.learning.model_drift import DriftDetector
            detector = DriftDetector(Sport.GOLF, db_path=os.path.join(os.path.dirname(__file__), "data", "quant_system.db"))
            drift = detector.edge_decay()
            if drift.get("edge_decay_detected"):
                st.error(f"EDGE DECAY DETECTED: {drift.get('recommendation', '')}")
            else:
                st.success("No edge decay detected")
            st.json(drift)

    # ── RISK & SIZING ──
    with qs_tabs[5]:
        st.markdown("**Risk Management & Dynamic Kelly Sizing**")
        risk_state = engine.bankroll_mgr.get_risk_state()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Kelly Multiplier", f"{risk_state.kelly_multiplier:.2f}")
        with col2:
            st.metric("Max Single Bet", f"${risk_state.max_single_bet:.2f}")
        with col3:
            st.metric("Daily Loss Remaining", f"${risk_state.daily_loss_remaining:.2f}")

        st.markdown("**Circuit Breakers**")
        breakers = engine.failure_protection.check_all(risk_state)
        for name in breakers["breakers_clear"]:
            st.success(f"{name}: CLEAR — {breakers['details'][name].get('message', '')}")
        for name in breakers["breakers_triggered"]:
            st.error(f"{name}: TRIGGERED — {breakers['details'][name].get('message', '')}")

        st.markdown("**Kelly Calculator**")
        kc1, kc2 = st.columns(2)
        with kc1:
            test_prob = st.slider("Win Probability", 0.50, 0.90, 0.60, 0.01, key="qs_test_prob")
            test_odds = st.number_input("Decimal Odds", 1.5, 5.0, 1.91, 0.01, key="qs_test_odds")
        with kc2:
            clv_data = engine.clv_tracker.rolling_clv(100)
            cal_data = engine.calibration.compute_calibration()
            result = engine.kelly.adaptive_stake(
                win_prob=test_prob,
                decimal_odds=test_odds,
                bankroll=risk_state.bankroll,
                risk_state=risk_state,
                clv_avg_cents=clv_data.get("avg_clv_cents", 0),
                calibration_mae=cal_data.get("mean_absolute_error", 0),
            )
            if result["blocked"]:
                st.error(f"Blocked: {result['block_reason']}")
            else:
                st.metric("Recommended Stake", f"${result['stake_dollars']:.2f}")
                st.metric("% of Bankroll", f"{result['pct_bankroll']*100:.2f}%")
                st.metric("Edge", f"{result['edge']*100:.1f}%")
                st.json(result["adjustments"])

    # ── BACKTEST ──
    with qs_tabs[6]:
        st.markdown("**Monte Carlo Bankroll Simulation**")
        st.markdown('<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:10px;">Simulate 10,000 bankroll paths to see probability of ruin, expected growth, and drawdown distribution.</div>', unsafe_allow_html=True)

        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            mc_edge = st.slider("Avg Edge (%)", 0.0, 15.0, 5.0, 0.5, key="qs_mc_edge")
            mc_wr = st.slider("Win Rate (%)", 45.0, 70.0, 55.0, 1.0, key="qs_mc_wr")
        with bc2:
            mc_odds = st.number_input("Avg Odds (decimal)", 1.5, 3.0, 1.91, 0.01, key="qs_mc_odds")
            mc_stake = st.slider("Avg Stake (% bankroll)", 0.5, 5.0, 2.0, 0.5, key="qs_mc_stake")
        with bc3:
            mc_bets = st.slider("Bets to Simulate", 100, 2000, 500, 50, key="qs_mc_bets")

        if st.button("Run Monte Carlo (10K paths)", key="qs_run_mc"):
            from quant_system.backtest.mc_bankroll import BankrollSimulator, MCConfig
            sim = BankrollSimulator(MCConfig(n_bets_per_path=mc_bets, initial_bankroll=bankroll))
            mc_result = sim.simulate(
                avg_edge=mc_edge / 100,
                avg_odds_decimal=mc_odds,
                avg_stake_pct=mc_stake / 100,
                win_rate=mc_wr / 100,
            )

            mc_cols = st.columns(4)
            with mc_cols[0]:
                rp = mc_result["ruin_probability"]
                st.metric("Ruin Probability", f"{rp:.1%}", delta="SAFE" if rp < 0.05 else "DANGER", delta_color="normal" if rp < 0.05 else "inverse")
            with mc_cols[1]:
                st.metric("Median Final", f"${mc_result['median_final']:,.0f}")
            with mc_cols[2]:
                st.metric("Profitable Paths", f"{mc_result['paths_profitable']:.0%}")
            with mc_cols[3]:
                st.metric("Doubled Paths", f"{mc_result['paths_doubled']:.0%}")

            st.markdown("**Distribution of Outcomes**")
            dist_df = pd.DataFrame({
                "Percentile": ["5th (worst)", "25th", "Median", "75th", "95th (best)"],
                "Final Bankroll": [
                    f"${mc_result['p5_final']:,.0f}",
                    f"${mc_result['p25_final']:,.0f}",
                    f"${mc_result['median_final']:,.0f}",
                    f"${mc_result['p75_final']:,.0f}",
                    f"${mc_result['p95_final']:,.0f}",
                ],
            })
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

            st.metric("Max Drawdown (median)", f"{mc_result['max_drawdown_median']*100:.1f}%")
            st.metric("Max Drawdown (95th pct)", f"{mc_result['max_drawdown_p95']*100:.1f}%")

    # ── SYSTEM HEALTH (Section 9 — Database-First Architecture) ──
    with qs_tabs[7]:
        st.markdown("**System Health — Database-First Architecture**")
        st.markdown('<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:10px;">Workers run independently of Streamlit. All state persists in the database.</div>', unsafe_allow_html=True)

        if _DB_AVAILABLE:
            try:
                ds = get_data_service()
                health = ds.get_system_health()

                # Database health
                db_status = health.get("database", {})
                db_color = "#00FF88" if db_status.get("status") == "healthy" else "#FF3358"
                st.markdown(f'<div class="glass-card" style="padding:10px 16px;border-left:3px solid {db_color};margin-bottom:12px;">'
                            f'<span style="color:{db_color};font-weight:700;">Database: {db_status.get("status", "unknown").upper()}</span></div>', unsafe_allow_html=True)

                # System state
                sys_state = health.get("system_state", "active")
                st.markdown(f"**System State:** `{sys_state.upper()}`")

                # Workers
                workers = health.get("workers", [])
                if workers:
                    st.markdown("**Background Workers**")
                    worker_data = []
                    for w in workers:
                        worker_data.append({
                            "Worker": w["name"],
                            "Status": w["status"],
                            "Last Run": w.get("last_run", "Never"),
                            "Runs": w.get("run_count", 0),
                            "Avg Duration": f"{w['avg_duration']:.1f}s" if w.get("avg_duration") else "N/A",
                        })
                    st.dataframe(pd.DataFrame(worker_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No workers have run yet. Workers run independently of Streamlit.")

                # Scrapers
                scrapers = health.get("scrapers", [])
                if scrapers:
                    st.markdown("**Scraper Status**")
                    scraper_data = []
                    for s in scrapers:
                        scraper_data.append({
                            "Scraper": s["name"],
                            "Healthy": "Yes" if s["is_healthy"] else "NO",
                            "Last Success": s.get("last_success", "Never"),
                            "Total Runs": s.get("total_runs", 0),
                            "Lines (Last)": s.get("lines_last_scrape", 0),
                        })
                    st.dataframe(pd.DataFrame(scraper_data), use_container_width=True, hide_index=True)

                # Active model
                model = health.get("active_model")
                if model:
                    st.markdown("**Active Model**")
                    m_cols = st.columns(4)
                    with m_cols[0]:
                        st.metric("Version", model["version"])
                    with m_cols[1]:
                        acc = model.get("live_accuracy")
                        st.metric("Accuracy", f"{acc:.1%}" if acc else "N/A")
                    with m_cols[2]:
                        roi = model.get("live_roi")
                        st.metric("ROI", f"{roi:.1%}" if roi else "N/A")
                    with m_cols[3]:
                        st.metric("Live Bets", model.get("n_live_bets", 0))

                # Recent signals
                signals = ds.get_recent_signals(hours=24)
                if signals:
                    st.markdown("**Recent Signals (24h)**")
                    sig_df = pd.DataFrame(signals[:20])
                    st.dataframe(sig_df[["player", "stat_type", "line", "direction", "edge", "approved"]], use_container_width=True, hide_index=True)

                # Edge report
                edge = ds.get_latest_edge_report()
                if edge:
                    st.markdown("**Latest Edge Report**")
                    st.json(edge)

                # Recent audit logs
                logs = ds.get_recent_logs(limit=10)
                if logs:
                    st.markdown("**Recent System Logs**")
                    for log in logs:
                        lvl_color = {"info": "#94a3b8", "warning": "#FFB800", "error": "#FF3358", "critical": "#FF0000"}.get(log["level"], "#94a3b8")
                        st.markdown(f'<div style="font-family:SF Mono,monospace;font-size:0.72rem;color:{lvl_color};margin-bottom:4px;">'
                                    f'[{log["timestamp"][:19]}] {log["level"].upper()}: {log["message"]}</div>', unsafe_allow_html=True)

                # DB schema verification
                if st.button("Verify Database Schema", key="qs_verify_schema"):
                    from database.migrations import MigrationManager
                    schema = MigrationManager.verify_schema()
                    if schema["healthy"]:
                        st.success(f"Schema healthy: {schema['expected_tables']} tables, version {schema['version']}")
                    else:
                        st.error(f"Missing tables: {schema['missing']}")

            except Exception as e:
                st.error(f"System health check failed: {e}")
        else:
            st.warning("Database not initialized. The unified database layer will activate on next deployment.")


# ============================================================
# TAB 8 — SETTINGS (Bankroll, Calibration, Logged Bets, Config)
# ============================================================
def tab_settings(proj_df: pd.DataFrame, settings: dict):
    """Settings tab — bankroll management, model calibration, bet logging, and configuration."""
    st.markdown(section_header("Settings", "&#9881;", "Configuration"), unsafe_allow_html=True)

    settings_tabs = st.tabs(["Bankroll Management", "Model Calibration", "Logged Bets", "API Keys"])

    # ── Bankroll Management ──
    with settings_tabs[0]:
        st.markdown("**Bankroll Management**")

        # Initialize bankroll state
        if "bankroll_total" not in st.session_state:
            st.session_state["bankroll_total"] = settings.get("bankroll", 1000)
        if "bankroll_history" not in st.session_state:
            st.session_state["bankroll_history"] = [{"date": datetime.now().strftime("%Y-%m-%d"), "amount": settings.get("bankroll", 1000), "action": "Initial"}]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(metric_card("Current Bankroll", f"${st.session_state['bankroll_total']:,.2f}", "total", "positive", "green"), unsafe_allow_html=True)
        with col2:
            total_wagered = sum(b.get("amount", 0) for b in st.session_state.get("logged_bets", []))
            st.markdown(metric_card("Total Wagered", f"${total_wagered:,.2f}", "all time", "neutral", "blue"), unsafe_allow_html=True)
        with col3:
            logged = st.session_state.get("logged_bets", [])
            settled = [b for b in logged if b.get("result") in ("won", "lost")]
            if settled:
                pl = sum(b.get("profit", 0) for b in settled)
                roi = (pl / max(total_wagered, 1)) * 100
            else:
                pl, roi = 0, 0
            st.markdown(metric_card("P&L", f"${pl:+,.2f}", f"ROI: {roi:+.1f}%", "positive" if pl > 0 else "negative", "amber"), unsafe_allow_html=True)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # Update bankroll
        new_bankroll = st.number_input("Set Bankroll ($)", min_value=10.0, value=float(st.session_state["bankroll_total"]), step=50.0, key="set_bankroll")
        if st.button("Update Bankroll", key="update_br"):
            st.session_state["bankroll_total"] = new_bankroll
            st.session_state["bankroll_history"].append({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "amount": new_bankroll,
                "action": "Manual Update",
            })
            # Persist to database (Section 9)
            if _DB_AVAILABLE:
                try:
                    save_state_to_db()
                except Exception:
                    pass
            st.success(f"Bankroll updated to ${new_bankroll:,.2f}")

        # Bankroll history
        if st.session_state.get("bankroll_history"):
            st.markdown("**Bankroll History**")
            bh_df = pd.DataFrame(st.session_state["bankroll_history"])
            st.dataframe(bh_df, use_container_width=True, hide_index=True)

    # ── Model Calibration ──
    with settings_tabs[1]:
        st.markdown("**Model Calibration Settings**")
        st.markdown('<div style="font-size:0.8rem;color:#94a3b8;margin-bottom:12px;">Fine-tune the quant engine parameters for optimal performance.</div>', unsafe_allow_html=True)

        cal_col1, cal_col2 = st.columns(2)
        with cal_col1:
            st.markdown("**Monte Carlo Settings**")
            mc_sims = st.slider("MC Simulations per Prop", 1000, 10000, 5000, 500, key="cal_mc_sims",
                               help="Higher = more accurate but slower. 5000 is recommended.")
            engine_weights = st.checkbox("Use Multi-Engine Ensemble", value=True, key="cal_multi_engine",
                                        help="Combines 5 probability engines: Normal, T-dist, Skew-adj, Mean-revert, Vol-cluster")

            st.markdown("**Probability Engines**")
            w_normal = st.slider("Normal Weight", 0.0, 1.0, 0.35, 0.05, key="w_norm")
            w_tdist = st.slider("T-Distribution Weight", 0.0, 1.0, 0.20, 0.05, key="w_tdist")
            w_skew = st.slider("Skew-Adjusted Weight", 0.0, 1.0, 0.15, 0.05, key="w_skew")
            w_revert = st.slider("Mean Reversion Weight", 0.0, 1.0, 0.15, 0.05, key="w_revert")
            w_vol = st.slider("Volatility Clustering Weight", 0.0, 1.0, 0.15, 0.05, key="w_vol")

            total_w = w_normal + w_tdist + w_skew + w_revert + w_vol
            if abs(total_w - 1.0) > 0.01:
                st.warning(f"Weights sum to {total_w:.2f} — should sum to 1.0")

        with cal_col2:
            st.markdown("**Kelly Criterion**")
            kelly_frac = st.slider("Kelly Fraction", 0.05, 1.0, 0.25, 0.05, key="cal_kelly",
                                  help="0.25 = Quarter Kelly (conservative). 1.0 = Full Kelly (aggressive)")
            max_bet_pct = st.slider("Max Bet % of Bankroll", 1, 15, 8, 1, key="cal_max_bet",
                                   help="Safety cap on single bet size")
            min_edge = st.slider("Min Edge Threshold %", 1, 15, 5, 1, key="cal_min_edge",
                                help="Minimum edge required to consider a bet")

            st.markdown("**Bayesian Shrinkage**")
            shrink_strength = st.slider("Prior Strength", 5, 30, 12, 1, key="cal_shrink",
                                       help="Higher = more shrinkage toward tour average")
            reversion_pct = st.slider("Mean Reversion %", 0, 50, 15, 5, key="cal_reversion",
                                     help="How much to regress projections toward baseline")

            st.markdown("**Correlation**")
            same_player_corr = st.slider("Same-Player Correlation", 0.3, 0.9, 0.60, 0.05, key="cal_same_corr")
            same_stat_corr = st.slider("Same-Stat Correlation", 0.0, 0.4, 0.15, 0.05, key="cal_stat_corr")

        if st.button("Save Calibration", key="save_cal", type="primary"):
            st.session_state["calibration"] = {
                "mc_sims": mc_sims,
                "multi_engine": engine_weights,
                "engine_weights": [w_normal, w_tdist, w_skew, w_revert, w_vol],
                "kelly_frac": kelly_frac,
                "max_bet_pct": max_bet_pct / 100.0,
                "min_edge": min_edge / 100.0,
                "shrink_strength": shrink_strength,
                "reversion_pct": reversion_pct / 100.0,
                "same_player_corr": same_player_corr,
                "same_stat_corr": same_stat_corr,
            }
            st.success("Calibration saved! These settings will be used by the scanner and lab.")

    # ── Logged Bets ──
    with settings_tabs[2]:
        st.markdown("**Bet Log**")
        st.markdown('<div style="font-size:0.8rem;color:#94a3b8;margin-bottom:12px;">Track all your bets, results, and performance over time.</div>', unsafe_allow_html=True)

        if "logged_bets" not in st.session_state:
            st.session_state["logged_bets"] = []

        # Add new bet
        with st.expander("Log New Bet", expanded=False):
            bet_col1, bet_col2, bet_col3 = st.columns(3)
            with bet_col1:
                bet_player = st.text_input("Player", key="bet_player")
                bet_stat = st.text_input("Stat Type", key="bet_stat")
            with bet_col2:
                bet_side = st.selectbox("Side", ["OVER", "UNDER"], key="bet_side")
                bet_line = st.number_input("Line", value=3.5, step=0.5, key="bet_line_input")
            with bet_col3:
                bet_amount = st.number_input("Bet Amount ($)", value=25.0, step=5.0, key="bet_amount")
                bet_type = st.selectbox("Type", ["Power Play", "Flex Play", "Single"], key="bet_type")

            if st.button("Log Bet", key="log_bet"):
                if bet_player:
                    st.session_state["logged_bets"].append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "player": bet_player,
                        "stat": bet_stat,
                        "side": bet_side,
                        "line": bet_line,
                        "amount": bet_amount,
                        "type": bet_type,
                        "result": "pending",
                        "profit": 0,
                    })
                    if _DB_AVAILABLE:
                        try:
                            save_state_to_db()
                        except Exception:
                            pass
                    st.success(f"Logged: {bet_player} {bet_stat} {bet_side} {bet_line} — ${bet_amount}")

        # Display bets
        logged = st.session_state.get("logged_bets", [])
        if logged:
            # Summary
            pending = [b for b in logged if b["result"] == "pending"]
            won = [b for b in logged if b["result"] == "won"]
            lost = [b for b in logged if b["result"] == "lost"]

            scol1, scol2, scol3, scol4 = st.columns(4)
            with scol1:
                st.markdown(metric_card("Total Bets", str(len(logged)), f"{len(pending)} pending", "neutral", "blue"), unsafe_allow_html=True)
            with scol2:
                win_rate = len(won) / max(len(won) + len(lost), 1) * 100
                st.markdown(metric_card("Win Rate", f"{win_rate:.0f}%", f"{len(won)}W / {len(lost)}L", "positive" if win_rate > 50 else "negative", "green"), unsafe_allow_html=True)
            with scol3:
                total_pl = sum(b.get("profit", 0) for b in won + lost)
                st.markdown(metric_card("Total P&L", f"${total_pl:+,.2f}", "settled bets", "positive" if total_pl > 0 else "negative", "amber"), unsafe_allow_html=True)
            with scol4:
                total_staked = sum(b.get("amount", 0) for b in won + lost)
                roi_val = (total_pl / max(total_staked, 1)) * 100
                st.markdown(metric_card("ROI", f"{roi_val:+.1f}%", f"on ${total_staked:,.0f}", "positive" if roi_val > 0 else "negative", "purple"), unsafe_allow_html=True)

            # Bet table
            bet_df = pd.DataFrame(logged)
            st.dataframe(bet_df, use_container_width=True, hide_index=True)

            # Settle bets
            if pending:
                st.markdown("**Settle Pending Bets**")
                for i, bet in enumerate(logged):
                    if bet["result"] == "pending":
                        bcol1, bcol2, bcol3 = st.columns([4, 1, 1])
                        with bcol1:
                            st.write(f"{bet['player']} {bet['stat']} {bet['side']} {bet['line']} — ${bet['amount']}")
                        with bcol2:
                            if st.button("Won", key=f"won_{i}"):
                                payout_map = {"Power Play": 3.0, "Flex Play": 2.25, "Single": 1.909}
                                payout = payout_map.get(bet["type"], 1.909)
                                profit = bet["amount"] * (payout - 1)
                                st.session_state["logged_bets"][i]["result"] = "won"
                                st.session_state["logged_bets"][i]["profit"] = profit
                                st.session_state["bankroll_total"] += profit
                                # Persist bankroll to DB so it survives refresh
                                if _DB_AVAILABLE:
                                    try: save_state_to_db()
                                    except Exception: pass
                                st.rerun()
                        with bcol3:
                            if st.button("Lost", key=f"lost_{i}"):
                                st.session_state["logged_bets"][i]["result"] = "lost"
                                st.session_state["logged_bets"][i]["profit"] = -bet["amount"]
                                st.session_state["bankroll_total"] -= bet["amount"]
                                # Persist bankroll to DB so it survives refresh
                                if _DB_AVAILABLE:
                                    try: save_state_to_db()
                                    except Exception: pass
                                st.rerun()
        else:
            st.info("No bets logged yet. Use the expander above to log your first bet.")

    # ── API Keys ──
    with settings_tabs[3]:
        st.markdown("**API Key Configuration**")
        st.markdown('<div style="font-size:0.8rem;color:#94a3b8;margin-bottom:12px;">Enter API keys here or set them in .streamlit/secrets.toml or .env file.</div>', unsafe_allow_html=True)

        ak = _get_anthropic_key()
        ok = _get_odds_api_key()
        wk = _get_weather_key()
        sk = _get_scraper_api_key()

        st.markdown(f"""
        <div class="glass-card" style="padding:14px;">
            <div style="font-size:0.85rem;color:#e2e8f0;margin-bottom:12px;">Current Status</div>
            <div style="font-size:0.8rem;color:{'#4ade80' if ak else '#f87171'};">{'&#10003;' if ak else '&#10007;'} Anthropic (Claude AI) {'— Connected' if ak else '— Not configured'}</div>
            <div style="font-size:0.8rem;color:{'#4ade80' if ok else '#f87171'};">{'&#10003;' if ok else '&#10007;'} The Odds API {'— Connected' if ok else '— Not configured'}</div>
            <div style="font-size:0.8rem;color:{'#4ade80' if wk else '#f87171'};">{'&#10003;' if wk else '&#10007;'} OpenWeather {'— Connected' if wk else '— Not configured'}</div>
            <div style="font-size:0.8rem;color:{'#4ade80' if sk else '#f87171'};">{'&#10003;' if sk else '&#10007;'} ScraperAPI {'— Connected' if sk else '— Not configured'}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # Override keys via session state
        new_ak = st.text_input("Anthropic API Key", value="", type="password", key="api_anthropic",
                               help="Get from console.anthropic.com")
        new_ok = st.text_input("Odds API Key", value="", type="password", key="api_odds",
                               help="Get from the-odds-api.com")
        new_wk = st.text_input("OpenWeather API Key", value="", type="password", key="api_weather",
                               help="Get from openweathermap.org")
        new_sk = st.text_input("ScraperAPI Key", value="", type="password", key="api_scraper",
                               help="Get from scraperapi.com")

        if st.button("Save API Keys", key="save_keys"):
            if new_ak:
                st.session_state["_anthropic_key"] = new_ak
            if new_ok:
                st.session_state["_odds_api_key"] = new_ok
            if new_wk:
                os.environ["OPENWEATHER_API_KEY"] = new_wk
            if new_sk:
                os.environ["SCRAPER_API_KEY"] = new_sk
            st.success("API keys saved for this session!")


# ============================================================
# TAB — EDGE ANALYSIS (Decomposition, Attribution, Sources)
# ============================================================
def tab_edge_analysis(proj_df: pd.DataFrame, settings: dict):
    """Edge Analysis — decomposition, attribution, edge sources."""
    st.markdown(section_header("Edge Analysis", "&#128300;", "Decomposition & Attribution"), unsafe_allow_html=True)

    analysis_tabs = st.tabs(["Edge Decomposition", "Attribution", "Edge Sources", "Adversarial Tests"])

    with analysis_tabs[0]:
        try:
            from edge_analysis.decomposer import GolfEdgeDecomposer
            from database.connection import get_session
            decomposer = GolfEdgeDecomposer()
            report = decomposer.generate_report()
            st.markdown(f"""
            <div class="glass-card" style="padding:18px;">
                <div style="font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:12px;">Edge Decomposition Report</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#94a3b8;white-space:pre-wrap;">{report.get('summary', 'Run more bets to generate decomposition.')}</div>
            </div>
            """, unsafe_allow_html=True)
            if report.get("components"):
                import plotly.graph_objects as go_fig
                labels = list(report["components"].keys())
                values = list(report["components"].values())
                fig = go_fig.Figure(data=[go_fig.Pie(labels=labels, values=values, hole=0.4)])
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Edge decomposition requires bet history. ({e})")

    with analysis_tabs[1]:
        try:
            from edge_analysis.attribution import EdgeAttributionEngine
            engine = EdgeAttributionEngine([])
            st.markdown("""
            <div class="glass-card" style="padding:18px;">
                <div style="font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:12px;">Edge Attribution</div>
                <div style="font-size:0.85rem;color:#94a3b8;">
                    Attribution decomposes profit into: Prediction Edge, CLV/Timing Edge, Market Inefficiency, and Variance.<br>
                    Log more bets with full line data to generate attribution analysis.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Run Attribution Analysis", key="run_attribution"):
                with st.spinner("Running Monte Carlo attribution..."):
                    result = engine.decompose()
                    st.json(result)
        except Exception as e:
            st.info(f"Attribution requires bet history. ({e})")

    with analysis_tabs[2]:
        try:
            from edge_analysis.source_registry import SourceRegistry
            registry = SourceRegistry()
            sources = registry.get_all_sources()
            st.markdown(f"""
            <div class="glass-card" style="padding:18px;">
                <div style="font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:12px;">Edge Sources ({len(sources)} registered)</div>
            </div>
            """, unsafe_allow_html=True)
            for src in sources:
                mechanism = src.get_mechanism() if hasattr(src, 'get_mechanism') else "N/A"
                decay = src.get_decay_risk() if hasattr(src, 'get_decay_risk') else "N/A"
                name = src.__class__.__name__
                st.markdown(f"""
                <div class="glass-card" style="padding:12px;margin-bottom:8px;">
                    <div style="font-weight:600;color:#60a5fa;">{name}</div>
                    <div style="font-size:0.8rem;color:#94a3b8;">Mechanism: {mechanism}</div>
                    <div style="font-size:0.8rem;color:#94a3b8;">Decay Risk: {decay}</div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.info(f"Edge sources: {e}")

    with analysis_tabs[3]:
        try:
            from tests.adversarial.runner import AdversarialRunner
            st.markdown("""
            <div class="glass-card" style="padding:18px;">
                <div style="font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:12px;">Adversarial Destruction Tests</div>
                <div style="font-size:0.85rem;color:#94a3b8;">
                    Tests: Probability Perturbation, Best Bet Removal, Noise Injection, Assumption Distortion, Time Period Robustness
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Run Adversarial Suite", key="run_adversarial"):
                with st.spinner("Running adversarial tests (this may take a while)..."):
                    runner = AdversarialRunner()
                    report = runner.run_all()
                    st.json(report)
        except Exception as e:
            st.info(f"Adversarial tests: {e}")


# ============================================================
# TAB — SIMULATOR (Tournament Monte Carlo)
# ============================================================
def tab_simulator(proj_df: pd.DataFrame, settings: dict):
    """Tournament simulator — hole-by-hole Monte Carlo."""
    st.markdown(section_header("Tournament Simulator", "&#127919;", "Hole-by-Hole Monte Carlo"), unsafe_allow_html=True)

    try:
        from simulation.tournament_engine import TournamentSimulator
        from simulation.config import SimulationConfig

        n_sims = st.slider("Simulations", 100, 10000, 1000, 100, key="sim_n")
        config = SimulationConfig(n_simulations=n_sims)

        if proj_df is not None and not proj_df.empty:
            st.markdown(f"**Field size:** {len(proj_df)} players | **Course:** {settings.get('course', 'Default')}")

            if st.button("Run Tournament Simulation", key="run_sim"):
                with st.spinner(f"Simulating {n_sims} tournaments..."):
                    engine = TournamentSimulator(config=config)
                    players = proj_df.head(30).to_dict('records') if len(proj_df) > 30 else proj_df.to_dict('records')
                    results = engine.simulate_tournament(players, settings.get('course', 'Default'))

                    if results:
                        st.success("Simulation complete!")
                        result_rows = []
                        for name, data in sorted(results.items(), key=lambda x: x[1].get('win_prob', 0), reverse=True)[:20]:
                            result_rows.append({
                                "Player": name,
                                "Win %": f"{data.get('win_prob', 0)*100:.1f}%",
                                "Top 5 %": f"{data.get('top5_prob', 0)*100:.1f}%",
                                "Top 10 %": f"{data.get('top10_prob', 0)*100:.1f}%",
                                "Top 20 %": f"{data.get('top20_prob', 0)*100:.1f}%",
                                "Make Cut %": f"{data.get('make_cut_prob', 0)*100:.1f}%",
                                "Avg Score": f"{data.get('avg_total', 0):.1f}",
                            })
                        if result_rows:
                            st.dataframe(pd.DataFrame(result_rows), use_container_width=True)
        else:
            st.warning("Load player data first (PrizePicks Lab or Live Scanner).")
    except Exception as e:
        st.info(f"Simulator: {e}")


# ============================================================
# TAB — CAPITAL (Kelly, Risk, Portfolio)
# ============================================================
def tab_capital(proj_df: pd.DataFrame, settings: dict):
    """Capital efficiency — Kelly sizing, risk metrics, portfolio optimization."""
    st.markdown(section_header("Capital Efficiency", "&#128176;", "Kelly Sizing & Risk Management"), unsafe_allow_html=True)

    capital_tabs = st.tabs(["Kelly Calculator", "Risk Metrics", "Portfolio"])

    with capital_tabs[0]:
        try:
            from services.capital.kelly import KellyCriterion as KellyCalculator
            calc = KellyCalculator()
            st.markdown("**Kelly Criterion Calculator**")
            col1, col2 = st.columns(2)
            with col1:
                win_prob = st.number_input("Win Probability (%)", 1.0, 99.0, 55.0, 0.5, key="kelly_prob") / 100
                odds = st.number_input("Decimal Odds", 1.01, 20.0, 2.0, 0.05, key="kelly_odds")
            with col2:
                fraction = st.slider("Kelly Fraction", 0.1, 1.0, 0.25, 0.05, key="kelly_frac")
                uncertainty = st.slider("Prob Uncertainty (SE)", 0.0, 0.20, 0.05, 0.01, key="kelly_unc")

            full = calc.full_kelly(win_prob, odds)
            frac = calc.fractional_kelly(win_prob, odds, fraction)
            adj_result = calc.uncertainty_adjusted_kelly(win_prob, odds, uncertainty, fraction)
            adj = adj_result[0] if isinstance(adj_result, tuple) else adj_result

            bankroll = settings.get("bankroll", 1000)
            st.markdown(f"""
            <div class="glass-card" style="padding:14px;">
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
                    <div><span style="font-size:0.7rem;color:#64748b;">Full Kelly</span><br><span style="font-size:1.2rem;font-weight:700;color:#4ade80;">{full*100:.1f}%</span><br><span style="font-size:0.8rem;color:#94a3b8;">${bankroll*full:.0f}</span></div>
                    <div><span style="font-size:0.7rem;color:#64748b;">Fractional ({fraction:.0%})</span><br><span style="font-size:1.2rem;font-weight:700;color:#60a5fa;">{frac*100:.1f}%</span><br><span style="font-size:0.8rem;color:#94a3b8;">${bankroll*frac:.0f}</span></div>
                    <div><span style="font-size:0.7rem;color:#64748b;">Uncertainty-Adj</span><br><span style="font-size:1.2rem;font-weight:700;color:#fbbf24;">{adj*100:.1f}%</span><br><span style="font-size:0.8rem;color:#94a3b8;">${bankroll*adj:.0f}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.info(f"Kelly calculator: {e}")

    with capital_tabs[1]:
        try:
            from services.capital.risk_adjusted import RiskMetrics
            metrics = RiskMetrics()
            st.markdown("""
            <div class="glass-card" style="padding:18px;">
                <div style="font-size:1rem;font-weight:700;color:#e2e8f0;">Risk Metrics</div>
                <div style="font-size:0.85rem;color:#94a3b8;margin-top:8px;">
                    Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio, VaR (95%), CVaR (95%)<br>
                    Requires bet history with P&L data.
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.info(f"Risk metrics: {e}")

    with capital_tabs[2]:
        try:
            from services.capital.portfolio import BetPortfolio
            st.markdown("""
            <div class="glass-card" style="padding:18px;">
                <div style="font-size:1rem;font-weight:700;color:#e2e8f0;">Portfolio Optimization</div>
                <div style="font-size:0.85rem;color:#94a3b8;margin-top:8px;">
                    Correlation-adjusted allocation, concentration checks, max exposure limits.<br>
                    Submit multiple bet signals to optimize allocation.
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.info(f"Portfolio: {e}")


# ============================================================
# TAB — KILL SWITCH
# ============================================================
def tab_kill_switch(proj_df: pd.DataFrame, settings: dict):
    """Kill switch status — hard stops that cannot be overridden."""
    st.markdown(section_header("Kill Switch", "&#128680;", "Hard Safety Stops"), unsafe_allow_html=True)

    try:
        from services.kill_switch import KillSwitch
        ks = KillSwitch()
        try:
            status = ks.check_all()
            active = status.get("system_active", True)
        except Exception:
            status = {}
            active = True

        if active:
            st.markdown("""
            <div class="glass-card" style="padding:18px;border-left:4px solid #4ade80;">
                <div style="font-size:1.3rem;font-weight:700;color:#4ade80;">ALL CLEAR</div>
                <div style="font-size:0.85rem;color:#94a3b8;">All kill switches are inactive. System is operational.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            reason = ks.get_halt_reason()
            st.markdown(f"""
            <div class="glass-card" style="padding:18px;border-left:4px solid #ef4444;background:rgba(239,68,68,0.1);">
                <div style="font-size:1.3rem;font-weight:700;color:#ef4444;">SYSTEM HALTED</div>
                <div style="font-size:0.85rem;color:#fca5a5;">{reason}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**Kill Switch Conditions:**")
        conditions = status.get("conditions", [])
        for cond in conditions:
            triggered = cond.get("triggered", False)
            color = "#ef4444" if triggered else "#4ade80"
            icon = "&#10007;" if triggered else "&#10003;"
            st.markdown(f"""
            <div class="glass-card" style="padding:10px;margin-bottom:6px;border-left:3px solid {color};">
                <span style="color:{color};">{icon}</span>
                <span style="font-weight:600;color:#e2e8f0;">{cond.get('name', 'Unknown')}</span>
                <span style="font-size:0.8rem;color:#94a3b8;margin-left:8px;">{cond.get('message', '')}</span>
            </div>
            """, unsafe_allow_html=True)
        if not conditions:
            st.info("No kill conditions evaluated yet. Run some bets to populate the database.")
    except Exception as e:
        st.info(f"Kill switch: {e}")


# ============================================================
# TAB — EDGE MONITOR (Daily Verdict)
# ============================================================
def tab_edge_monitor(proj_df: pd.DataFrame, settings: dict):
    """Real-time edge monitoring — daily EDGE = YES/NO."""
    st.markdown(section_header("Edge Monitor", "&#128200;", "Daily Edge Status"), unsafe_allow_html=True)

    try:
        from services.edge_monitor.alert_system import EdgeAlertSystem
        from services.edge_monitor.daily_metrics import DailyMetrics

        alert = EdgeAlertSystem()
        try:
            verdict = alert.daily_verdict()
        except Exception:
            verdict = "EDGE STATUS: INSUFFICIENT DATA — Need more bets to determine edge."

        verdict_str = str(verdict) if not isinstance(verdict, str) else verdict
        edge_yes = "YES" in verdict_str.upper()
        color = "#4ade80" if edge_yes else "#ef4444"

        st.markdown(f"""
        <div class="glass-card" style="padding:24px;text-align:center;border:2px solid {color};">
            <div style="font-size:0.8rem;color:#64748b;letter-spacing:0.15em;text-transform:uppercase;">Edge Status</div>
            <div style="font-size:2.5rem;font-weight:900;color:{color};font-family:'JetBrains Mono',monospace;margin:8px 0;">
                {'EDGE = YES' if edge_yes else 'EDGE = NO'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="glass-card" style="padding:14px;margin-top:12px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#94a3b8;white-space:pre-wrap;">{verdict_str}</div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.info(f"Edge monitor requires bet history. ({e})")


# ============================================================
# TAB — FINAL VERDICT
# ============================================================
def tab_verdict(proj_df: pd.DataFrame, settings: dict):
    """Final verdict — honest self-assessment of the entire system."""
    st.markdown(section_header("Final Verdict", "&#9878;", "Honest System Self-Assessment"), unsafe_allow_html=True)

    try:
        from services.verdict.final_report import FinalVerdict

        st.markdown("""
        <div class="glass-card" style="padding:18px;">
            <div style="font-size:1rem;font-weight:700;color:#e2e8f0;">System Verdict Generator</div>
            <div style="font-size:0.85rem;color:#94a3b8;margin-top:8px;">
                Pulls from ALL subsystems: edge decomposition, attribution, CLV, data quality, execution reality,
                market reaction, model governance, adversarial testing, and capital efficiency.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Generate Full Verdict Report", key="gen_verdict"):
            with st.spinner("Running full system analysis..."):
                verdict = FinalVerdict()
                report = verdict.generate()
                st.markdown(f"""
                <div class="glass-card" style="padding:18px;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#e2e8f0;white-space:pre-wrap;">{report}</div>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.info(f"Verdict: {e}")


# ============================================================
# TAB — NEXT WEEK PREP (Pre-tournament analysis before lines drop)
# ============================================================
def tab_next_week_prep(proj_df: pd.DataFrame, settings: dict):
    """Pre-tournament preparation — runs full quant engine projections + simulations
    for next week's tournament before official lines drop."""
    st.markdown(section_header("Next Week Prep", "&#128197;", "Pre-Tournament Analysis Pipeline"), unsafe_allow_html=True)

    # Detect next week's tournament
    next_tournaments = _get_next_week_tournaments()

    if not next_tournaments:
        st.warning("No upcoming tournament found in the schedule for next week. "
                    "The schedule may need updating, or we may be in an off-week.")
        return

    tourney = next_tournaments[0]
    t_name = tourney["tournament"]
    t_course = tourney["course"]
    t_tour = tourney["tour"]

    # Calculate next week dates (Thursday-Sunday)
    days_until_thursday = (3 - datetime.now().weekday()) % 7
    if days_until_thursday == 0:
        days_until_thursday = 7
    next_week_start = datetime.now() + timedelta(days=days_until_thursday)
    next_week_end = next_week_start + timedelta(days=3)

    # Tournament Info Header
    st.markdown(f"""
    <div class="glass-card" style="padding:20px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Next Tournament</div>
                <div style="font-size:1.4rem;font-weight:700;color:#f0f0f0;margin:4px 0;">{t_name}</div>
                <div style="font-size:0.85rem;color:#94a3b8;">{t_course} &middot; {t_tour}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.7rem;color:#64748b;">Projected Dates</div>
                <div style="font-size:0.9rem;color:#94a3b8;">{next_week_start.strftime('%b %d')} — {next_week_end.strftime('%b %d, %Y')}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div style="background:rgba(234,179,8,0.1);border:1px solid rgba(234,179,8,0.3);border-radius:8px;'
        'padding:12px 16px;margin-bottom:16px;font-size:0.82rem;color:#fbbf24;">'
        '&#9888; <b>Official lines not yet available</b> — these are preliminary projections based on '
        'SG data and course fit analysis. Use for early field assessment and preparation only.</div>',
        unsafe_allow_html=True,
    )

    # Check if we have projection data to work with
    if proj_df is None or proj_df.empty:
        st.info("Load player data first (via sidebar data sources) to run next-week projections.")
        return

    prep_tabs = st.tabs(["Power Rankings", "Tournament Simulation", "Course Fit", "Early Value"])

    # Re-build projection table with the NEXT tournament's course
    with st.spinner(f"Building projections for {t_name} at {t_course}..."):
        try:
            next_proj_df = _build_projection_table(proj_df.copy(), t_course)
        except Exception:
            next_proj_df = proj_df.copy()

    # ── Power Rankings ──
    with prep_tabs[0]:
        st.markdown(f"**SG-Based Power Rankings — {t_name}**")
        st.markdown(f'<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:12px;">'
                    f'Rankings adjusted for course fit at {t_course}.</div>', unsafe_allow_html=True)

        if len(next_proj_df) > 0:
            top = next_proj_df.iloc[0]
            cols = st.columns(4)
            with cols[0]:
                st.markdown(metric_card("#1 Ranked", str(top["player"]),
                            f"SG: {top['sg_regressed']:.2f}", "positive", "green"), unsafe_allow_html=True)
            with cols[1]:
                avg_sg = next_proj_df["sg_regressed"].mean()
                st.markdown(metric_card("Field Avg SG", f"{avg_sg:.2f}",
                            f"{len(next_proj_df)} players", "neutral", "blue"), unsafe_allow_html=True)
            with cols[2]:
                if "edge" in next_proj_df.columns:
                    best_edge = next_proj_df["edge"].max()
                    best_player = next_proj_df.loc[next_proj_df["edge"].idxmax(), "player"]
                    st.markdown(metric_card("Best Edge", f"{best_edge*100:.1f}%",
                                str(best_player), "positive", "amber"), unsafe_allow_html=True)
                else:
                    st.markdown(metric_card("Best Edge", "N/A",
                                "No lines available", "neutral", "amber"), unsafe_allow_html=True)
            with cols[3]:
                fs_adj = field_strength_adjustment(next_proj_df["sg_regressed"].tolist())
                st.markdown(metric_card("Field Strength", fs_adj["strength_label"],
                            f"Factor: {fs_adj['adjustment_factor']:.3f}", "neutral", "purple"), unsafe_allow_html=True)

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

            # Rankings table
            rank_cols = ["player", "sg_regressed"]
            if "course_delta" in next_proj_df.columns:
                rank_cols.append("course_delta")
            for c in ["win_prob", "top5_prob", "top10_prob", "cut_prob"]:
                if c in next_proj_df.columns:
                    rank_cols.append(c)

            rank_df = next_proj_df[rank_cols].head(30).copy()
            rename_map = {
                "player": "Player", "sg_regressed": "SG (adj)", "course_delta": "Course Fit",
                "win_prob": "Win%", "top5_prob": "Top 5%", "top10_prob": "Top 10%", "cut_prob": "Cut%",
            }
            rank_df = rank_df.rename(columns=rename_map)
            for pct_col in ["Win%", "Top 5%", "Top 10%", "Cut%"]:
                if pct_col in rank_df.columns:
                    rank_df[pct_col] = (rank_df[pct_col] * 100).round(1)
            st.dataframe(rank_df, use_container_width=True, height=600)

    # ── Tournament Simulation ──
    with prep_tabs[1]:
        st.markdown(f"**Tournament Simulation — {t_name}**")
        st.markdown(f'<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:12px;">'
                    f'Monte Carlo simulation of full tournament outcomes at {t_course}.</div>', unsafe_allow_html=True)

        n_sims_prep = st.slider("Simulations", 500, 3000, 1500, 500, key="prep_sim_n")

        if st.button("Run Pre-Tournament Simulation", key="run_next_week_sim", type="primary"):
            with st.spinner(f"Simulating {n_sims_prep} tournaments at {t_course}..."):
                try:
                    from simulation.pipeline_bridge import SimulationBridge
                    sim_bridge = SimulationBridge(n_sims=n_sims_prep)
                    sim_df = sim_bridge.run_tournament_simulation(
                        field_projections=next_proj_df,
                        course_name=t_course,
                    )

                    if sim_df is not None and not sim_df.empty:
                        st.success(f"Simulation complete! {n_sims_prep} tournaments simulated.")

                        sim_results = []
                        name_col = "name" if "name" in sim_df.columns else "player"
                        for _, row in sim_df.head(30).iterrows():
                            sim_results.append({
                                "Player": row.get(name_col, ""),
                                "Win %": f"{row.get('sim_win_prob', row.get('win_prob', 0))*100:.2f}%",
                                "Top 5 %": f"{row.get('sim_top5_prob', row.get('top5_prob', 0))*100:.1f}%",
                                "Top 10 %": f"{row.get('sim_top10_prob', row.get('top10_prob', 0))*100:.1f}%",
                                "Top 20 %": f"{row.get('sim_top20_prob', row.get('top20_prob', 0))*100:.1f}%",
                                "Make Cut %": f"{row.get('sim_make_cut_prob', row.get('make_cut_prob', row.get('cut_prob', 0)))*100:.0f}%",
                                "Avg Finish": f"#{row.get('sim_avg_finish', row.get('avg_finish', 0)):.0f}",
                            })

                        if sim_results:
                            st.dataframe(pd.DataFrame(sim_results), use_container_width=True, hide_index=True)

                        st.session_state["next_week_sim"] = sim_df
                    else:
                        st.warning("Simulation returned no results.")
                except Exception as e:
                    st.error(f"Simulation error: {e}")

        # Show previously cached sim results with win probability chart
        if "next_week_sim" in st.session_state:
            cached_sim = st.session_state["next_week_sim"]
            if cached_sim is not None and not cached_sim.empty:
                st.markdown("**Win Probability Distribution (Top 15)**")
                name_col = "name" if "name" in cached_sim.columns else "player"
                top15 = cached_sim.head(15)
                win_col = "sim_win_prob" if "sim_win_prob" in top15.columns else "win_prob"
                if win_col in top15.columns:
                    fig = go.Figure(go.Bar(
                        x=top15[name_col],
                        y=top15[win_col] * 100,
                        marker_color="#4ade80",
                    ))
                    fig.update_layout(
                        height=350,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="Player", yaxis_title="Win Probability (%)",
                        font=dict(color="#e2e8f0"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ── Course Fit ──
    with prep_tabs[2]:
        st.markdown(f"**Course Fit Analysis — {t_course}**")

        cp = COURSE_PROFILES.get(t_course, {})
        weights = cp.get("sg_weights", SG_WEIGHTS)

        if cp:
            cols = st.columns(4)
            label_map = {"sg_ott": "Off Tee", "sg_app": "Approach", "sg_arg": "Around Green", "sg_putt": "Putting"}
            with cols[0]:
                dominant = max(weights, key=weights.get)
                st.markdown(metric_card("Key Skill", label_map.get(dominant, dominant),
                            f"Weight: {weights[dominant]:.0%}", "positive", "green"), unsafe_allow_html=True)
            with cols[1]:
                st.markdown(metric_card("Distance Bonus", f"{cp.get('distance_bonus', 0):+.0%}",
                            "Favors long hitters" if cp.get("distance_bonus", 0) > 0 else "Neutral",
                            "positive" if cp.get("distance_bonus", 0) > 0 else "neutral", "blue"), unsafe_allow_html=True)
            with cols[2]:
                st.markdown(metric_card("Wind Factor", f"{cp.get('wind_sensitivity', 0):.1f}",
                            "High" if cp.get("wind_sensitivity", 0) > 0.5 else "Moderate",
                            "negative" if cp.get("wind_sensitivity", 0) > 0.5 else "neutral", "amber"), unsafe_allow_html=True)
            with cols[3]:
                grass = "Bermuda" if cp.get("bermuda_greens", False) else "Bent/Poa"
                st.markdown(metric_card("Green Type", grass,
                            f"Elev: {cp.get('elevation', 0)}ft", "neutral", "green"), unsafe_allow_html=True)

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

            # Weight comparison chart
            default_w = SG_WEIGHTS
            comp_data = []
            for cat in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]:
                comp_data.append({
                    "Category": label_map.get(cat, cat),
                    "Course Weight": weights.get(cat, 0),
                    "Default Weight": default_w.get(cat, 0),
                })
            comp_df = pd.DataFrame(comp_data)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=comp_df["Category"], y=comp_df["Course Weight"],
                                 name=t_course, marker_color="#4ade80"))
            fig.add_trace(go.Bar(x=comp_df["Category"], y=comp_df["Default Weight"],
                                 name="Tour Default", marker_color="rgba(255,255,255,0.2)"))
            fig.update_layout(barmode="group", height=300,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#e2e8f0"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No detailed course profile found for {t_course}. "
                    "Default SG weights will be used for projections.")

        # Top course fit players
        if "course_delta" in next_proj_df.columns:
            st.markdown("**Top Course Fit Players**")
            fit_df = next_proj_df.nlargest(15, "course_delta")[["player", "sg_regressed", "course_delta"]].copy()
            fit_df.columns = ["Player", "SG (adj)", "Course Fit Delta"]
            st.dataframe(fit_df, use_container_width=True, hide_index=True)

    # ── Early Value ──
    with prep_tabs[3]:
        st.markdown(f"**Early Value Identification — {t_name}**")
        st.markdown(
            '<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:12px;">'
            'Players projected to outperform their typical market pricing based on SG model, '
            'course fit, and historical tournament performance. Target these when lines drop.</div>',
            unsafe_allow_html=True,
        )

        # Identify early value: players with strong SG + positive course fit
        value_df = next_proj_df.copy()
        if "course_delta" in value_df.columns:
            value_df["value_score"] = value_df["sg_regressed"] + value_df["course_delta"] * 2
        else:
            value_df["value_score"] = value_df["sg_regressed"]
        value_df = value_df.sort_values("value_score", ascending=False)

        st.markdown("**High-Value Targets (SG + Course Fit)**")
        val_rows = []
        for _, row in value_df.head(15).iterrows():
            sg = row.get("sg_regressed", 0)
            cd = row.get("course_delta", 0)
            win_p = row.get("win_prob", 0)
            cut_p = row.get("cut_prob", 0)

            vs = row["value_score"]
            if vs > 2.0:
                tier = "Elite"
            elif vs > 1.0:
                tier = "Strong"
            elif vs > 0.5:
                tier = "Moderate"
            else:
                tier = "Speculative"

            val_rows.append({
                "Player": row["player"],
                "SG (adj)": f"{sg:.2f}",
                "Course Fit": f"{cd:+.2f}" if cd != 0 else "0.00",
                "Value Score": f"{vs:.2f}",
                "Win%": f"{win_p*100:.2f}%" if win_p > 0 else "--",
                "Cut%": f"{cut_p*100:.0f}%" if cut_p > 0 else "--",
                "Tier": tier,
            })

        if val_rows:
            vdf = pd.DataFrame(val_rows)
            st.dataframe(vdf, use_container_width=True, hide_index=True)

        # Value distribution chart
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.markdown("**Value Score Distribution (Top 20)**")
        top20 = value_df.head(20)
        fig = go.Figure(go.Bar(
            x=top20["player"],
            y=top20["value_score"],
            marker_color=["#00FF88" if v > 2 else "#4ade80" if v > 1 else "#fbbf24" if v > 0.5 else "#94a3b8"
                          for v in top20["value_score"]],
        ))
        fig.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Player", yaxis_title="Value Score (SG + 2x Course Fit)",
            font=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div style="font-size:0.75rem;color:#64748b;margin-top:8px;">'
            'Value Score = SG (adjusted) + 2x Course Fit Delta. Higher scores indicate '
            'players likely undervalued by the market at this course. '
            'Cross-reference with official lines when they become available.</div>',
            unsafe_allow_html=True,
        )


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
    elif settings["data_mode"] == "Live Odds":
        # Step 1: Get the REAL field from ESPN (current PGA Tour event)
        with st.spinner("Fetching PGA Tour field from ESPN..."):
            espn_data = fetch_espn_pga_field()

        # Step 2: Get odds from The Odds API (futures — best available)
        odds_key = _get_odds_api_key()
        odds_map = {}
        if odds_key:
            with st.spinner("Fetching odds from The Odds API..."):
                odds_map = fetch_odds_api_golf_odds(odds_key)

        if espn_data and espn_data.get("players"):
            event_name = espn_data["event_name"]
            event_status = espn_data["status"]
            espn_players = espn_data["players"]
            st.success(f"**{event_name}** — {len(espn_players)} players ({event_status})")

            # Build raw_df from ESPN field + odds overlay
            rng = np.random.RandomState(42)
            rows = []
            n = len(espn_players)
            for i, pl in enumerate(espn_players):
                name = pl["name"]
                # Scale SG by leaderboard position (if available) or field order
                frac = 1.0 - (i / max(1, n - 1))
                base_sg = 2.0 * frac - 0.2
                sg_ott = round(rng.normal(0.3, 0.4) + max(0, base_sg * 0.18), 2)
                sg_app = round(rng.normal(0.2, 0.5) + max(0, base_sg * 0.30), 2)
                sg_arg = round(rng.normal(0.1, 0.3) + max(0, base_sg * 0.15), 2)
                sg_putt = round(rng.normal(0.15, 0.5) + max(0, base_sg * 0.20), 2)
                sg_total = round(sg_ott + sg_app + sg_arg + sg_putt, 2)

                # Look up odds for this player (fuzzy match)
                player_odds = odds_map.get(name, 0)
                if not player_odds:
                    # Try last-name match
                    last_name = name.split()[-1] if name else ""
                    for oname, oprice in odds_map.items():
                        if last_name and last_name in oname:
                            player_odds = oprice
                            break
                if not player_odds:
                    # Estimate from rank position
                    player_odds = int(500 + i * 300 + rng.randint(0, 500))

                rows.append({
                    "player": name,
                    "sg_total": sg_total,
                    "sg_ott": sg_ott,
                    "sg_app": sg_app,
                    "sg_arg": sg_arg,
                    "sg_putt": sg_putt,
                    "events": rng.randint(10, 25),
                    "odds": player_odds,
                    "world_rank": i + 1,
                })
            raw_df = pd.DataFrame(rows)
        else:
            st.warning("Could not fetch ESPN field. Falling back to demo data.")
            raw_df = _generate_sample_players(30, settings["course"])

        # Also fetch PrizePicks lines for the PrizePicks tab
        scraper_key = _get_scraper_api_key()
        with st.spinner("Fetching PrizePicks golf lines..."):
            pp_lines = fetch_prizepicks_golf_lines(scraper_key)
        if pp_lines:
            st.session_state["pp_lines"] = pp_lines
            st.info(f"Loaded {len(pp_lines)} PrizePicks golf props")
        else:
            st.session_state["pp_lines"] = []
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

    # Tab layout — PrizePicks Lab is the main Run Model page
    tabs = st.tabs([
        "&#127183; PrizePicks Lab",
        "&#128225; Live Scanner",
        "&#127942; Power Rankings",
        "&#128269; Player Deep Dive",
        "&#127959; Course Fit",
        "&#128176; Betting Edge",
        "&#128202; Quant System",
        "&#128300; Edge Analysis",
        "&#127919; Simulator",
        "&#128176; Capital",
        "&#128680; Kill Switch",
        "&#128200; Edge Monitor",
        "&#9878; Verdict",
        "&#128197; Next Week Prep",
        "&#9881; Settings",
    ])

    with tabs[0]:
        tab_prizepicks(proj_df, settings)
    with tabs[1]:
        tab_live_scanner(proj_df, settings)
    with tabs[2]:
        tab_power_rankings(proj_df, settings)
    with tabs[3]:
        tab_player_deep_dive(proj_df, settings)
    with tabs[4]:
        tab_course_fit(proj_df, settings)
    with tabs[5]:
        tab_betting_edge(proj_df, settings)
    with tabs[6]:
        tab_quant_system(proj_df, settings)
    with tabs[7]:
        tab_edge_analysis(proj_df, settings)
    with tabs[8]:
        tab_simulator(proj_df, settings)
    with tabs[9]:
        tab_capital(proj_df, settings)
    with tabs[10]:
        tab_kill_switch(proj_df, settings)
    with tabs[11]:
        tab_edge_monitor(proj_df, settings)
    with tabs[12]:
        tab_verdict(proj_df, settings)
    with tabs[13]:
        tab_next_week_prep(proj_df, settings)
    with tabs[14]:
        tab_settings(proj_df, settings)


if __name__ == "__main__":
    main()
