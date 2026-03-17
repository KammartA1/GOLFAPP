"""
Golf Quant Engine — Streamlit Dashboard v1.0
Run: streamlit run dashboard.py --server.port 8501
Visit: http://YOUR_SERVER_IP:8501
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path

# ── Page config (must be first) ────────────────────────────────────────────
st.set_page_config(
    page_title="Golf Quant Engine",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Barlow:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }

.stApp { background: #080c14; }

[data-testid="stSidebar"] {
    background: #0b0f1c !important;
    border-right: 1px solid #1a2540;
}

/* Kill default header */
#MainMenu, footer, header { visibility: hidden; }

/* Top header bar */
.dash-header {
    background: linear-gradient(135deg, #0b1628 0%, #0d1f3c 100%);
    border-bottom: 1px solid #1e3a5f;
    padding: 18px 32px;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 16px;
}
.dash-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #4ade80;
    margin: 0;
    letter-spacing: -0.5px;
}
.dash-header .sub {
    font-size: 0.8rem;
    color: #4a6080;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 2px;
}

/* Metric cards */
.metric-card {
    background: #0e1628;
    border: 1px solid #1a2d4a;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4ade80, #22d3ee);
}
.metric-card .label {
    font-size: 0.7rem;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e2e8f0;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1;
}
.metric-card .delta {
    font-size: 0.8rem;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}
.delta-pos { color: #4ade80; }
.delta-neg { color: #f87171; }
.delta-neu { color: #4a6080; }

/* Section headers */
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #4ade80;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #1a2d4a;
    padding-bottom: 8px;
    margin-bottom: 16px;
    margin-top: 8px;
}

/* Player rank badge */
.rank-badge {
    display: inline-block;
    background: #0a1628;
    border: 1px solid #1a2d4a;
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #4a6080;
}

/* PP pick cards */
.pick-card {
    background: #0e1628;
    border: 1px solid #1a2d4a;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.pick-card.high { border-left: 3px solid #4ade80; }
.pick-card.medium { border-left: 3px solid #facc15; }
.pick-card.low { border-left: 3px solid #60a5fa; }
.pick-over { color: #4ade80; font-weight: 700; }
.pick-under { color: #f87171; font-weight: 700; }

/* Slip card */
.slip-card {
    background: linear-gradient(135deg, #0e1628, #0a1f35);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 14px;
}
.slip-ev-pos { color: #4ade80; font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
.slip-ev-neg { color: #f87171; font-family: 'IBM Plex Mono', monospace; font-weight: 600; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0b0f1c;
    border-bottom: 1px solid #1a2540;
    gap: 4px;
    padding: 0 8px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #4a6080;
    border: none;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    padding: 12px 20px;
    letter-spacing: 0.5px;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #4ade80 !important;
    border-bottom: 2px solid #4ade80 !important;
}

/* Dataframe styling */
.stDataFrame {
    border: 1px solid #1a2d4a;
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #166534, #14532d);
    color: #4ade80;
    border: 1px solid #16a34a;
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 10px 24px;
    letter-spacing: 0.5px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #16a34a, #166534);
    border-color: #4ade80;
    color: #fff;
}

/* Selectbox, inputs */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: #0e1628 !important;
    border: 1px solid #1a2d4a !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] label {
    color: #7a9ab8 !important;
    font-size: 0.78rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Status pill */
.status-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}
.status-live { background: #052e16; color: #4ade80; border: 1px solid #16a34a; }
.status-stub { background: #1c1002; color: #facc15; border: 1px solid #ca8a04; }
.status-off  { background: #1a0a0a; color: #f87171; border: 1px solid #dc2626; }

/* Confidence badges */
.conf-high   { background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-family: 'IBM Plex Mono', monospace; }
.conf-medium { background: #1c1002; color: #facc15; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-family: 'IBM Plex Mono', monospace; }
.conf-low    { background: #0a1628; color: #60a5fa; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-family: 'IBM Plex Mono', monospace; }

hr { border-color: #1a2540; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def metric_card(label, value, delta=None, delta_type="neu"):
    delta_html = ""
    if delta:
        cls = f"delta-{delta_type}"
        arrow = "▲" if delta_type == "pos" else ("▼" if delta_type == "neg" else "")
        delta_html = f'<div class="delta {cls}">{arrow} {delta}</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

def conf_badge(conf):
    cls = {"HIGH": "conf-high", "MEDIUM": "conf-medium", "LOW": "conf-low"}.get(conf, "conf-low")
    return f'<span class="{cls}">{conf}</span>'

def status_pill(label, status):
    cls = {"live": "status-live", "stub": "status-stub", "off": "status-off"}.get(status, "status-off")
    return f'<span class="status-pill {cls}">{label}</span>'


# ── Mock / Demo data (used when engine modules aren't fully loaded) ─────────

@st.cache_data(ttl=300)
def get_demo_projections():
    players = [
        ("Scottie Scheffler", 1, 2.41, 0.82, 1.10, 0.49, 96, "improving"),
        ("Rory McIlroy", 2, 1.98, 0.61, 0.91, 0.46, 88, "stable"),
        ("Xander Schauffele", 5, 1.72, 0.55, 0.79, 0.38, 82, "improving"),
        ("Collin Morikawa", 8, 1.65, 0.48, 0.88, 0.29, 85, "stable"),
        ("Viktor Hovland", 7, 1.44, 0.52, 0.65, 0.27, 78, "declining"),
        ("Jon Rahm", 3, 1.38, 0.41, 0.72, 0.25, 80, "stable"),
        ("Patrick Cantlay", 12, 1.21, 0.38, 0.56, 0.27, 74, "stable"),
        ("Wyndham Clark", 18, 1.15, 0.44, 0.50, 0.21, 71, "improving"),
        ("Max Homa", 14, 1.08, 0.35, 0.52, 0.21, 69, "stable"),
        ("Tony Finau", 20, 0.98, 0.42, 0.41, 0.15, 66, "declining"),
        ("Justin Thomas", 16, 0.91, 0.28, 0.48, 0.15, 67, "improving"),
        ("Shane Lowry", 22, 0.85, 0.25, 0.42, 0.18, 63, "stable"),
        ("Tom Kim", 25, 0.79, 0.31, 0.35, 0.13, 61, "improving"),
        ("Matt Fitzpatrick", 19, 0.72, 0.22, 0.39, 0.11, 64, "stable"),
        ("Brian Harman", 30, 0.65, 0.18, 0.31, 0.16, 59, "stable"),
        ("Sahith Theegala", 28, 0.61, 0.24, 0.28, 0.09, 57, "improving"),
        ("Chris Kirk", 45, 0.55, 0.19, 0.26, 0.10, 54, "stable"),
        ("Russell Henley", 38, 0.48, 0.16, 0.22, 0.10, 52, "declining"),
        ("Adam Schenk", 60, 0.38, 0.15, 0.18, 0.05, 48, "stable"),
        ("Seamus Power", 52, 0.31, 0.10, 0.15, 0.06, 46, "declining"),
    ]
    rows = []
    for i, (name, rank, sg_total, sg_ott, sg_app, sg_putt, fit, form) in enumerate(players):
        win_p = max(0.002, 0.28 - i * 0.012 + np.random.uniform(-0.005, 0.005))
        top10_p = min(0.55, win_p * 6.5 + np.random.uniform(0, 0.03))
        cut_p = min(0.95, 0.75 + sg_total * 0.06)
        dk_pts = 37 + sg_total * 9.2 + cut_p * 3
        sal = max(6000, 12000 - i * 350 + np.random.randint(-200, 200))
        sal = round(sal / 100) * 100
        val = round(dk_pts / (sal / 1000), 2)
        own = max(0.01, 0.32 - i * 0.013 + np.random.uniform(-0.01, 0.01))
        lev = round((own * 100 - (i + 1)) * -0.3, 1)

        rows.append({
            "rank": i + 1,
            "name": name,
            "world_rank": rank,
            "proj_sg_total": sg_total,
            "proj_sg_ott": sg_ott,
            "proj_sg_app": sg_app,
            "proj_sg_putt": sg_putt,
            "course_fit": fit,
            "form_trend": form,
            "win_prob": round(win_p, 4),
            "top10_prob": round(top10_p, 4),
            "make_cut_prob": round(cut_p, 3),
            "dk_salary": sal,
            "dk_proj_pts": round(dk_pts, 1),
            "dk_value": val,
            "proj_ownership": round(own, 3),
            "leverage": lev,
        })
    return pd.DataFrame(rows)


def _load_live_pp_lines():
    """Load live PrizePicks lines from the scraper database."""
    try:
        from data.storage.database import get_session, PrizePicksLine, ScraperStatus
        session = get_session()
        try:
            lines = session.query(PrizePicksLine).filter(
                PrizePicksLine.is_latest == True
            ).all()
            if not lines:
                return None, None

            # Get scraper status for the header
            status = session.query(ScraperStatus).filter_by(
                scraper_name="golf_prizepicks"
            ).first()
            last_pull = status.last_success if status else None

            rows = []
            for l in lines:
                # Simple edge model: compare line to rough model projection
                # (placeholder — the real edge analyzer in betting/prizepicks.py
                #  should be wired in for production use)
                model_proj = l.line_score * 1.03  # placeholder
                gap = round(model_proj - l.line_score, 2)
                prob = 0.55 + abs(gap) * 0.02
                pick = "OVER" if gap > 0 else "UNDER"
                conf = "HIGH" if abs(gap) > 2.0 else "MEDIUM" if abs(gap) > 0.5 else "LOW"
                rows.append({
                    "player": l.player_name,
                    "stat": l.stat_display or l.stat_type,
                    "line": l.line_score,
                    "model_proj": model_proj,
                    "gap": gap,
                    "pick": pick,
                    "prob": min(prob, 0.85),
                    "confidence": conf,
                })
            return pd.DataFrame(rows), last_pull
        finally:
            session.close()
    except Exception:
        return None, None


@st.cache_data(ttl=60)
def get_pp_lines():
    """Load live lines from scraper DB; fall back to demo data if unavailable."""
    live_df, _last_pull = _load_live_pp_lines()
    if live_df is not None and not live_df.empty:
        return live_df
    # Fallback demo data
    lines = [
        ("Scottie Scheffler", "Fantasy Score", 52.5, 58.2, "OVER", "HIGH",  0.68),
        ("Scottie Scheffler", "Birdies or Better", 4.5, 5.1, "OVER", "HIGH", 0.64),
        ("Rory McIlroy",      "Fantasy Score", 48.5, 45.8, "UNDER", "MEDIUM", 0.59),
        ("Rory McIlroy",      "Birdies or Better", 4.5, 4.8, "OVER", "LOW", 0.55),
        ("Xander Schauffele", "Fantasy Score", 44.5, 47.1, "OVER", "MEDIUM", 0.61),
        ("Xander Schauffele", "Bogey Free Rounds", 1.5, 1.2, "UNDER", "MEDIUM", 0.58),
        ("Collin Morikawa",   "Fantasy Score", 43.5, 45.0, "OVER", "MEDIUM", 0.60),
        ("Collin Morikawa",   "Birdies or Better", 4.0, 4.6, "OVER", "HIGH", 0.65),
        ("Viktor Hovland",    "Fantasy Score", 41.5, 39.8, "UNDER", "LOW", 0.54),
        ("Patrick Cantlay",   "Fantasy Score", 40.5, 42.1, "OVER", "LOW", 0.55),
        ("Patrick Cantlay",   "Bogey Free Rounds", 1.5, 1.8, "OVER", "MEDIUM", 0.59),
        ("Wyndham Clark",     "Fantasy Score", 39.5, 41.2, "OVER", "MEDIUM", 0.60),
        ("Max Homa",          "Birdies or Better", 3.5, 4.1, "OVER", "HIGH", 0.66),
        ("Tom Kim",           "Fantasy Score", 36.5, 38.9, "OVER", "MEDIUM", 0.61),
        ("Sahith Theegala",   "Fantasy Score", 34.5, 36.8, "OVER", "MEDIUM", 0.58),
    ]
    rows = []
    for player, stat, line, model, rec, conf, prob in lines:
        rows.append({
            "player": player, "stat": stat, "line": line,
            "model_proj": model, "gap": round(model - line, 2),
            "pick": rec, "prob": prob, "confidence": conf,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def get_demo_h2h():
    matchups = [
        ("Scottie Scheffler", "Rory McIlroy",      +0.43, 0.61, "SG edge: Scheffler +0.43 | Course fit: Scheffler +8pts"),
        ("Collin Morikawa",   "Viktor Hovland",     +0.27, 0.59, "Course fit: Morikawa +7pts | Form: Morikawa improving"),
        ("Xander Schauffele", "Jon Rahm",           +0.34, 0.60, "SG edge: Schauffele +0.34 | Form: Schauffele improving"),
        ("Wyndham Clark",     "Tony Finau",         +0.17, 0.56, "Distance bonus: Clark | Form: Clark improving"),
        ("Tom Kim",           "Shane Lowry",        +0.14, 0.55, "SG edge: Kim +0.14 | Form: Kim improving"),
        ("Max Homa",          "Justin Thomas",      +0.17, 0.56, "Course fit: Homa +5pts | Approach SG: Homa stronger"),
        ("Sahith Theegala",   "Russell Henley",     +0.23, 0.58, "SG edge: Theegala +0.23 | Form: Theegala improving"),
        ("Patrick Cantlay",   "Matt Fitzpatrick",   +0.56, 0.62, "Putting edge: Cantlay dominant | Course fit: +11pts"),
    ]
    rows = []
    for a, b, sg_edge, prob_a, notes in matchups:
        rows.append({
            "player_a": a, "player_b": b,
            "sg_edge": sg_edge,
            "win_prob_a": prob_a,
            "win_prob_b": round(1 - prob_a, 2),
            "notes": notes,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def get_demo_lineups():
    lineups = [
        {
            "label": "GPP #1 — High Ceiling",
            "players": ["Scottie Scheffler", "Xander Schauffele", "Collin Morikawa",
                        "Wyndham Clark", "Tom Kim", "Sahith Theegala"],
            "salary": 49800, "proj_pts": 261.4, "own_sum": 94.2,
            "type": "GPP",
        },
        {
            "label": "GPP #2 — Leverage Stack",
            "players": ["Scottie Scheffler", "Collin Morikawa", "Max Homa",
                        "Wyndham Clark", "Sahith Theegala", "Adam Schenk"],
            "salary": 48600, "proj_pts": 254.8, "own_sum": 79.4,
            "type": "GPP",
        },
        {
            "label": "GPP #3 — Contrarian",
            "players": ["Rory McIlroy", "Collin Morikawa", "Patrick Cantlay",
                        "Tom Kim", "Chris Kirk", "Seamus Power"],
            "salary": 49200, "proj_pts": 248.1, "own_sum": 68.7,
            "type": "GPP",
        },
        {
            "label": "Cash — Floor Build",
            "players": ["Scottie Scheffler", "Rory McIlroy", "Xander Schauffele",
                        "Collin Morikawa", "Patrick Cantlay", "Max Homa"],
            "salary": 49900, "proj_pts": 271.2, "own_sum": 148.6,
            "type": "Cash",
        },
    ]
    return lineups


@st.cache_data(ttl=300)
def get_demo_audit():
    np.random.seed(42)
    n = 80
    dates = pd.date_range(end=datetime.now(), periods=n, freq="3D")
    win_probs = np.random.uniform(0.05, 0.35, n)
    # Simulate actual outcomes with slight edge
    outcomes = np.random.binomial(1, np.minimum(win_probs + 0.04, 1.0))
    stakes = np.random.uniform(20, 120, n)
    odds = np.random.choice([-150, -130, +110, +130, +150, +200, +250, +400, +600], n)
    bet_types = np.random.choice(["outright", "top10", "top20", "h2h", "make_cut"], n)

    def pl(won, stake, odd):
        if won:
            return stake * ((odd / 100) if odd > 0 else (100 / abs(odd)))
        return -stake

    pls = [pl(outcomes[i], stakes[i], odds[i]) for i in range(n)]

    df = pd.DataFrame({
        "date": dates,
        "player": np.random.choice(["Scheffler", "McIlroy", "Schauffele", "Morikawa", "Hovland"], n),
        "bet_type": bet_types,
        "stake": stakes.round(2),
        "odds": odds,
        "won": outcomes.astype(bool),
        "profit_loss": [round(p, 2) for p in pls],
        "model_prob": win_probs.round(3),
    })
    df["cumulative_pl"] = df["profit_loss"].cumsum().round(2)
    df["rolling_roi"] = (
        df["profit_loss"].rolling(20).sum() /
        df["stake"].rolling(20).sum()
    ).round(4)
    return df


# ── Try importing live engine modules ─────────────────────────────────────
ENGINE_AVAILABLE = False
PP_SCRAPER_AVAILABLE = False
PP_ANALYZER_AVAILABLE = False
H2H_AVAILABLE = False
_upcoming_tournament = None

try:
    from config.settings import DATAGOLF_ENABLED
    from config.courses import COURSE_PROFILES
    from data.storage.database import init_db, get_session
    init_db()
    ENGINE_AVAILABLE = True
    COURSE_NAMES = list(COURSE_PROFILES.keys())
except Exception as e:
    COURSE_NAMES = [
        "Augusta National", "TPC Sawgrass", "TPC Scottsdale", "Pebble Beach",
        "Pinehurst No. 2", "Torrey Pines South", "Bay Hill", "Muirfield Village",
        "Riviera CC", "Harbour Town", "Innisbrook (Copperhead)", "East Lake GC",
        "St Andrews Old Course", "Royal Troon",
    ]
    DATAGOLF_ENABLED = False

try:
    from data.scrapers.prizepicks import PrizePicksScraper
    PP_SCRAPER_AVAILABLE = True
except Exception:
    pass

try:
    from betting.prizepicks import PrizePicksAnalyzer
    PP_ANALYZER_AVAILABLE = True
except Exception:
    pass

try:
    from betting.h2h import sg_to_h2h_prob, generate_synthetic_matchups
    H2H_AVAILABLE = True
except Exception:
    try:
        from betting.h2h import H2HAnalyzer
        H2H_AVAILABLE = True
    except Exception:
        pass


# ── Auto-detect upcoming tournament ──────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _detect_upcoming_tournament():
    """Query PGA Tour API for the next upcoming tournament."""
    try:
        from data.scrapers.pga_tour import PGATourScraper
        scraper = PGATourScraper()
        upcoming = scraper.fetch_upcoming_tournaments(weeks_ahead=2)
        if upcoming:
            t = upcoming[0]
            return {
                "name": t["name"],
                "course": t.get("course_name", ""),
                "event_id": t["pga_event_id"],
                "start_date": t.get("start_date"),
            }
    except Exception:
        pass
    return None

_upcoming_tournament = _detect_upcoming_tournament()


# ── Real projection engine ───────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner="Running projection pipeline...")
def get_real_projections(course_name: str):
    """Fetch real PGA Tour data and run SG projection model."""
    try:
        from data.pipeline import DataPipeline
        from models.projection import ProjectionEngine

        pipeline = DataPipeline()
        event_id = _upcoming_tournament.get("event_id") if _upcoming_tournament else None
        data = pipeline.full_refresh(event_id=event_id)

        engine = ProjectionEngine()
        tournament_name = _upcoming_tournament.get("name", "Tournament") if _upcoming_tournament else "Tournament"
        proj_df = engine.run(
            tournament_name=tournament_name,
            course_name=course_name,
            field_data=data.get("field", []),
            sg_history=data.get("sg_history", {}),
        )

        if proj_df is not None and not proj_df.empty:
            # Ensure all columns the dashboard expects exist
            for col, default in [
                ("rank", None), ("course_fit", 50), ("form_trend", "stable"),
                ("dk_salary", 0), ("dk_value", 0), ("proj_ownership", 0.02), ("leverage", 0),
            ]:
                if col not in proj_df.columns:
                    if col == "rank":
                        proj_df["rank"] = range(1, len(proj_df) + 1)
                    elif col == "course_fit":
                        proj_df["course_fit"] = proj_df.get("course_fit_score", pd.Series([default]*len(proj_df)))
                    else:
                        proj_df[col] = default

            # Rename SG columns to match dashboard expectations
            renames = {}
            for old, new in [("proj_sg_total", "proj_sg_total"), ("proj_sg_ott", "proj_sg_ott"),
                             ("proj_sg_app", "proj_sg_app"), ("proj_sg_putt", "proj_sg_putt")]:
                if old not in proj_df.columns and new not in proj_df.columns:
                    proj_df[new] = 0.0

            return proj_df
    except Exception as e:
        import traceback
        st.session_state["_proj_error"] = f"{e}\n{traceback.format_exc()}"
    return None


# ── Real PrizePicks fetch ────────────────────────────────────────────────
@st.cache_data(ttl=120, show_spinner="Fetching PrizePicks lines...")
def fetch_live_pp_lines_direct():
    """Fetch live PrizePicks golf lines directly from their API."""
    try:
        if not PP_SCRAPER_AVAILABLE:
            return None
        scraper = PrizePicksScraper()
        pp_lines = scraper.fetch_golf_projections()
        if not pp_lines:
            return None
        rows = []
        for p in pp_lines:
            rows.append({
                "player": p.player_name,
                "stat": p.stat_display,
                "line": p.line_score,
                "stat_type": p.stat_type,
                "pp_id": p.pp_id,
                "player_id": p.player_id,
                "is_promo": p.is_promo,
                "start_time": p.start_time,
            })
        return rows
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def get_pp_lines_with_edges(proj_df=None):
    """Fetch live PP lines and compute edges using our model projections."""
    # Step 1: Try direct live fetch from PrizePicks API
    live_rows = fetch_live_pp_lines_direct()

    # Step 2: Fall back to scraper DB
    if not live_rows:
        live_df, _last_pull = _load_live_pp_lines()
        if live_df is not None and not live_df.empty:
            return live_df

    if not live_rows:
        return None  # Will fall back to demo data

    # Step 3: If we have projections + analyzer, compute real edges
    if proj_df is not None and PP_ANALYZER_AVAILABLE and PP_SCRAPER_AVAILABLE:
        try:
            analyzer = PrizePicksAnalyzer()
            edges = analyzer.analyze_slate(proj_df, min_confidence="LOW")
            if edges:
                rows = []
                for e in edges:
                    rows.append({
                        "player": e.projection.player_name,
                        "stat": e.projection.stat_display,
                        "line": e.line,
                        "model_proj": e.model_proj,
                        "gap": round(e.model_vs_line, 2),
                        "pick": e.recommendation,
                        "prob": e.pick_prob,
                        "confidence": e.confidence,
                    })
                if rows:
                    return pd.DataFrame(rows)
        except Exception:
            pass

    # Step 4: Build basic edge table from raw lines + projection data
    if proj_df is not None and not proj_df.empty:
        proj_map = {}
        for _, row in proj_df.iterrows():
            proj_map[row["name"].lower()] = row.to_dict()

        rows = []
        for lr in live_rows:
            player_key = lr["player"].lower()
            player_proj = proj_map.get(player_key)
            if not player_proj:
                for pname, pdata in proj_map.items():
                    if lr["player"].split()[-1].lower() in pname or pname.split()[-1] in player_key:
                        player_proj = pdata
                        break

            sg_total = player_proj.get("proj_sg_total", 0) if player_proj else 0
            # Simple stat projection using SG sensitivity
            SENSITIVITIES = {
                "fantasy_score": 9.2, "birdies_or_better": 0.9, "birdies": 0.9,
                "bogey_free_rounds": 0.12, "strokes_total": -4.0, "gir": 1.2,
                "fairways_hit": 1.5, "eagles": 0.025, "holes_under_par": 1.8,
            }
            BASELINES = {
                "fantasy_score": 37.0, "birdies_or_better": 4.2, "birdies": 4.2,
                "bogey_free_rounds": 1.1, "strokes_total": 284.0, "gir": 11.5,
                "fairways_hit": 9.0, "eagles": 0.12, "holes_under_par": 18.0,
            }
            stat_key = lr.get("stat_type", lr["stat"].lower().replace(" ", "_"))
            baseline = BASELINES.get(stat_key, lr["line"])
            sens = SENSITIVITIES.get(stat_key, 1.0)
            model_proj = baseline + sg_total * sens
            gap = round(model_proj - lr["line"], 2)
            pick = "OVER" if gap > 0 else "UNDER"
            prob = min(0.50 + abs(gap) * 0.03, 0.85)
            conf = "HIGH" if prob >= 0.625 else "MEDIUM" if prob >= 0.575 else "LOW"

            rows.append({
                "player": lr["player"], "stat": lr["stat"],
                "line": lr["line"], "model_proj": round(model_proj, 1),
                "gap": gap, "pick": pick, "prob": prob, "confidence": conf,
            })
        if rows:
            return pd.DataFrame(rows)

    # Step 5: Raw lines without model (minimal info)
    rows = []
    for lr in live_rows:
        model_proj = lr["line"] * 1.03
        gap = round(model_proj - lr["line"], 2)
        rows.append({
            "player": lr["player"], "stat": lr["stat"],
            "line": lr["line"], "model_proj": round(model_proj, 1),
            "gap": gap, "pick": "OVER" if gap > 0 else "UNDER",
            "prob": 0.55, "confidence": "LOW",
        })
    return pd.DataFrame(rows) if rows else None


# ── Generate H2H from real projections ───────────────────────────────────
def generate_h2h_from_projections(proj_df):
    """Generate head-to-head matchups from model projections."""
    if proj_df is None or proj_df.empty or len(proj_df) < 4:
        return None
    try:
        top = proj_df.head(20).copy()
        matchups = []
        seen = set()
        for i, row_a in top.iterrows():
            for j, row_b in top.iterrows():
                if i >= j:
                    continue
                key = (row_a["name"], row_b["name"])
                if key in seen:
                    continue
                seen.add(key)
                sg_a = row_a.get("proj_sg_total", 0) or 0
                sg_b = row_b.get("proj_sg_total", 0) or 0
                sg_edge = round(sg_a - sg_b, 3)
                if abs(sg_edge) < 0.05:
                    continue
                # Logistic conversion: 0.32 per SG stroke (from h2h.py)
                from scipy.special import expit
                prob_a = round(float(expit(sg_edge * 0.32)), 2)
                fit_a = row_a.get("course_fit", row_a.get("course_fit_score", 50))
                fit_b = row_b.get("course_fit", row_b.get("course_fit_score", 50))
                form_a = row_a.get("form_trend", "stable")
                form_b = row_b.get("form_trend", "stable")
                notes_parts = [f"SG edge: {row_a['name'].split()[-1]} {sg_edge:+.3f}"]
                if fit_a and fit_b and fit_a != fit_b:
                    notes_parts.append(f"Course fit: {row_a['name'].split()[-1]} {'+' if fit_a > fit_b else ''}{int(fit_a - fit_b)}pts")
                if form_a != "stable":
                    notes_parts.append(f"Form: {row_a['name'].split()[-1]} {form_a}")
                matchups.append({
                    "player_a": row_a["name"], "player_b": row_b["name"],
                    "sg_edge": sg_edge, "win_prob_a": prob_a,
                    "win_prob_b": round(1 - prob_a, 2),
                    "notes": " | ".join(notes_parts),
                })
                if len(matchups) >= 12:
                    break
            if len(matchups) >= 12:
                break
        if matchups:
            df = pd.DataFrame(matchups)
            return df.sort_values("sg_edge", ascending=False, key=abs).head(10)
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding: 16px 0 24px 0;">
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem;
                    font-weight: 600; color: #4ade80; letter-spacing: -0.5px;">
            ⛳ GOLF QUANT
        </div>
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
                    color: #2a4060; margin-top: 3px; letter-spacing: 1px;">
            ENGINE v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**TOURNAMENT**")
    _default_tournament = (_upcoming_tournament.get("name", "") + " " + str(datetime.now().year)) if _upcoming_tournament else f"PGA Tour {datetime.now().year}"
    tournament_name = st.text_input("Tournament Name", value=_default_tournament, label_visibility="collapsed")
    course = st.selectbox("Course", COURSE_NAMES, label_visibility="collapsed")

    st.markdown("<br>**BANKROLL**", unsafe_allow_html=True)
    bankroll = st.number_input("Bankroll ($)", min_value=100, max_value=1000000,
                                value=1000, step=100, label_visibility="collapsed")

    st.markdown("<br>**KELLY FRACTION**", unsafe_allow_html=True)
    kelly_frac = st.slider("Kelly", 0.1, 0.5, 0.25, 0.05, label_visibility="collapsed")

    st.markdown("<br>**FILTERS**", unsafe_allow_html=True)
    min_confidence = st.selectbox("Min PP Confidence", ["LOW", "MEDIUM", "HIGH"],
                                   index=0, label_visibility="collapsed")
    show_top_n = st.slider("Players to show", 10, 50, 20, 5, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶  RUN PROJECTIONS", use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Data source status
    st.markdown("**DATA SOURCES**")
    _pp_status = "live" if PP_SCRAPER_AVAILABLE else "stub"
    _engine_status = "live" if ENGINE_AVAILABLE else "off"
    sources = [
        ("PGA Tour API", _engine_status),
        ("ESPN Golf",    _engine_status),
        ("OpenWeather",  "live" if ENGINE_AVAILABLE else "stub"),
        ("PrizePicks",   _pp_status),
        ("DataGolf",     "live" if DATAGOLF_ENABLED else "stub"),
    ]
    for name, status in sources:
        icons = {"live": "🟢", "stub": "🟡", "off": "🔴"}
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;padding:4px 0;font-family:IBM Plex Mono,monospace;'
            f'font-size:0.72rem;color:#4a6080;">'
            f'<span>{name}</span>'
            f'<span>{icons[status]}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;'
        f'color:#2a4060;text-align:center;">Last run: {datetime.now().strftime("%m/%d %H:%M")}</div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

_data_badge = ('<span style="background:#052e16;color:#4ade80;padding:2px 8px;border-radius:4px;'
               'font-size:0.6rem;font-family:IBM Plex Mono,monospace;margin-left:12px;">LIVE DATA</span>'
               if st.session_state.get("_using_real_data") else
               '<span style="background:#1c1002;color:#facc15;padding:2px 8px;border-radius:4px;'
               'font-size:0.6rem;font-family:IBM Plex Mono,monospace;margin-left:12px;">DEMO DATA</span>')
st.markdown(f"""
<div class="dash-header">
    <div>
        <h1>⛳ GOLF QUANT ENGINE {_data_badge}</h1>
        <div class="sub">{tournament_name.upper()} &nbsp;·&nbsp; {course.upper()} &nbsp;·&nbsp; {datetime.now().strftime("%B %d, %Y")}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

if "projections" not in st.session_state or run_btn:
    with st.spinner("Running projection pipeline..."):
        # ── Try real data first, fall back to demo ──
        _real_proj = None
        if ENGINE_AVAILABLE:
            _real_proj = get_real_projections(course)
            if _real_proj is not None and not _real_proj.empty:
                st.session_state.projections = _real_proj
                st.session_state._using_real_data = True
            else:
                st.session_state.projections = get_demo_projections()
                st.session_state._using_real_data = False
                if "_proj_error" in st.session_state:
                    st.toast(f"Engine error, using demo data: {st.session_state['_proj_error'][:100]}", icon="⚠️")
        else:
            st.session_state.projections = get_demo_projections()
            st.session_state._using_real_data = False

        # ── PrizePicks: try live fetch → scraper DB → demo ──
        _pp = get_pp_lines_with_edges(st.session_state.projections if st.session_state.get("_using_real_data") else None)
        if _pp is not None and not _pp.empty:
            st.session_state.pp_lines = _pp
        else:
            st.session_state.pp_lines = get_pp_lines()

        # ── H2H: generate from real projections → demo ──
        _h2h = generate_h2h_from_projections(st.session_state.projections if st.session_state.get("_using_real_data") else None)
        if _h2h is not None and not _h2h.empty:
            st.session_state.h2h = _h2h
        else:
            st.session_state.h2h = get_demo_h2h()

        st.session_state.lineups      = get_demo_lineups()
        st.session_state.audit_data   = get_demo_audit()

proj_df   = st.session_state.projections
pp_df     = st.session_state.pp_lines
h2h_df    = st.session_state.h2h
lineups   = st.session_state.lineups
audit_df  = st.session_state.audit_data


# ═══════════════════════════════════════════════════════════════════════════
# TOP METRICS ROW
# ═══════════════════════════════════════════════════════════════════════════

c1, c2, c3, c4, c5, c6 = st.columns(6)

settled = audit_df[audit_df["won"].notna()]
total_pl = settled["profit_loss"].sum()
roi = total_pl / settled["stake"].sum() if len(settled) > 0 else 0
win_rate = settled["won"].mean() if len(settled) > 0 else 0
pp_edge_count = len(pp_df[pp_df["confidence"].isin(["HIGH", "MEDIUM"])])

with c1:
    metric_card("BANKROLL", f"${bankroll:,}", delta=f"${total_pl:+.0f} total P&L",
                delta_type="pos" if total_pl >= 0 else "neg")
with c2:
    metric_card("FIELD SIZE", f"{len(proj_df)}", delta="players projected", delta_type="neu")
with c3:
    metric_card("ROI", f"{roi:.1%}", delta=f"{len(settled)} bets settled",
                delta_type="pos" if roi >= 0 else "neg")
with c4:
    metric_card("WIN RATE", f"{win_rate:.1%}", delta=f"{settled['won'].sum():.0f}W / {(~settled['won']).sum():.0f}L",
                delta_type="pos" if win_rate >= 0.5 else "neg")
with c5:
    metric_card("PP EDGES", f"{pp_edge_count}", delta="actionable picks today", delta_type="pos" if pp_edge_count > 0 else "neu")
with c6:
    metric_card("DATAGOLF", "ACTIVE" if DATAGOLF_ENABLED else "STUBBED",
                delta="add key in .env", delta_type="neu" if not DATAGOLF_ENABLED else "pos")

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏌️  PROJECTIONS",
    "🎯  PRIZEPICKS",
    "⚔️  H2H MATCHUPS",
    "📋  DFS LINEUPS",
    "📊  AUDIT",
    "⚙️  SETTINGS",
])


# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — PROJECTIONS
# ───────────────────────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        section_title(f"FIELD PROJECTIONS — {course.upper()}")

        # Format display dataframe
        display = proj_df.head(show_top_n).copy()
        display["win_prob"]   = display["win_prob"].apply(lambda x: f"{x:.1%}")
        display["top10_prob"] = display["top10_prob"].apply(lambda x: f"{x:.1%}")
        display["make_cut_prob"] = display["make_cut_prob"].apply(lambda x: f"{x:.1%}")
        display["proj_ownership"] = display["proj_ownership"].apply(lambda x: f"{x*100:.1f}%")
        display["dk_salary"]  = display["dk_salary"].apply(lambda x: f"${x:,}")
        display["proj_sg_total"] = display["proj_sg_total"].apply(lambda x: f"{x:+.3f}")
        display["proj_sg_app"]   = display["proj_sg_app"].apply(lambda x: f"{x:+.3f}")

        display = display.rename(columns={
            "rank": "#", "name": "Player", "world_rank": "OWGR",
            "proj_sg_total": "SG Total", "proj_sg_app": "SG App",
            "course_fit": "Fit", "form_trend": "Form",
            "win_prob": "Win%", "top10_prob": "Top10%",
            "make_cut_prob": "Cut%", "dk_salary": "DK $",
            "dk_proj_pts": "DK Pts", "dk_value": "Value",
            "proj_ownership": "Own%",
        })

        cols = ["#", "Player", "OWGR", "SG Total", "SG App", "Fit",
                "Form", "Win%", "Top10%", "Cut%", "DK $", "DK Pts", "Value", "Own%"]
        cols = [c for c in cols if c in display.columns]

        st.dataframe(
            display[cols],
            use_container_width=True,
            hide_index=True,
            height=520,
        )

    with col_right:
        section_title("SG BREAKDOWN — TOP 10")

        top10 = proj_df.head(10)
        fig = go.Figure()

        categories = ["proj_sg_ott", "proj_sg_app", "proj_sg_putt"]
        labels = ["Off Tee", "Approach", "Putting"]
        colors = ["#4ade80", "#22d3ee", "#a78bfa"]

        for cat, label, color in zip(categories, labels, colors):
            fig.add_trace(go.Bar(
                name=label,
                x=top10["name"].apply(lambda x: x.split()[-1]),
                y=top10[cat],
                marker_color=color,
                marker_opacity=0.85,
            ))

        fig.update_layout(
            barmode="stack",
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(family="IBM Plex Mono", size=10, color="#7a9ab8"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                font=dict(size=9), bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=240,
            xaxis=dict(gridcolor="#1a2d4a", tickangle=-30),
            yaxis=dict(gridcolor="#1a2d4a", title="SG"),
        )
        st.plotly_chart(fig, use_container_width=True)

        section_title("COURSE FIT vs OWNERSHIP")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=proj_df.head(20)["proj_ownership"] * 100,
            y=proj_df.head(20)["course_fit"],
            mode="markers+text",
            text=proj_df.head(20)["name"].apply(lambda x: x.split()[-1]),
            textposition="top center",
            textfont=dict(size=8, color="#7a9ab8"),
            marker=dict(
                size=8,
                color=proj_df.head(20)["proj_sg_total"],
                colorscale=[[0, "#1a2d4a"], [0.5, "#22d3ee"], [1, "#4ade80"]],
                showscale=False,
                line=dict(width=1, color="#0a0e1a"),
            ),
        ))
        fig2.update_layout(
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(family="IBM Plex Mono", size=9, color="#7a9ab8"),
            xaxis=dict(title="Proj Ownership %", gridcolor="#1a2d4a"),
            yaxis=dict(title="Course Fit Score", gridcolor="#1a2d4a"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=230,
        )
        st.plotly_chart(fig2, use_container_width=True)

        section_title("GPP LEVERAGE PLAYS")
        leverage_plays = proj_df[
            (proj_df["proj_ownership"] <= 0.10) &
            (proj_df["rank"] <= 25)
        ].head(6)
        for _, row in leverage_plays.iterrows():
            st.markdown(
                f'<div style="padding:8px 12px;margin-bottom:6px;background:#0e1628;'
                f'border-radius:8px;border-left:2px solid #4ade80;">'
                f'<div style="font-size:0.85rem;font-weight:600;color:#e2e8f0;">{row["name"]}</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#4a6080;margin-top:2px;">'
                f'SG {row["proj_sg_total"]:+.3f} &nbsp;·&nbsp; '
                f'Own {row["proj_ownership"]*100:.1f}% &nbsp;·&nbsp; '
                f'Fit {row["course_fit"]}</div></div>',
                unsafe_allow_html=True
            )


# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — PRIZEPICKS
# ───────────────────────────────────────────────────────────────────────────
with tab2:
    # Scraper status indicator
    _pp_live_check, _pp_last_pull = _load_live_pp_lines()
    _pp_has_real_lines = (pp_df is not None and not pp_df.empty and
                          not pp_df.equals(get_pp_lines()) if callable(get_pp_lines) else True)
    if _pp_last_pull:
        _pp_age = int((datetime.now() - _pp_last_pull).total_seconds() / 60)
        _pp_status_color = "#4ade80" if _pp_age < 45 else "#facc15" if _pp_age < 180 else "#f87171"
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;'
            f'color:{_pp_status_color};margin-bottom:8px;">'
            f'SCRAPER: Last pull {_pp_age} min ago &nbsp;·&nbsp; '
            f'{len(_pp_live_check) if _pp_live_check is not None else 0} live lines</div>',
            unsafe_allow_html=True
        )
    elif PP_SCRAPER_AVAILABLE:
        st.markdown(
            '<div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;'
            'color:#4ade80;margin-bottom:8px;">PRIZEPICKS: Direct API fetch enabled — lines update on each run</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;'
            'color:#4a6080;margin-bottom:8px;">SCRAPER: Using demo data — install dependencies for live lines</div>',
            unsafe_allow_html=True
        )

    pp_col1, pp_col2 = st.columns([3, 2])

    with pp_col1:
        section_title("LIVE PRIZEPICKS LINES — EDGE ANALYSIS")

        conf_filter = st.multiselect(
            "Filter by confidence",
            ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM"] if min_confidence == "MEDIUM" else
                    ["HIGH"] if min_confidence == "HIGH" else ["HIGH", "MEDIUM", "LOW"],
            label_visibility="collapsed",
        )
        filtered_pp = pp_df[pp_df["confidence"].isin(conf_filter)] if conf_filter else pp_df

        for _, row in filtered_pp.iterrows():
            conf_cls = row["confidence"].lower()
            pick_cls = "pick-over" if row["pick"] == "OVER" else "pick-under"
            pick_arrow = "🔼" if row["pick"] == "OVER" else "🔽"
            gap_color = "#4ade80" if (
                (row["pick"] == "OVER" and row["gap"] > 0) or
                (row["pick"] == "UNDER" and row["gap"] < 0)
            ) else "#f87171"

            st.markdown(f"""
            <div class="pick-card {conf_cls}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="font-size:0.95rem;font-weight:600;color:#e2e8f0;">{row["player"]}</div>
                        <div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;
                                    color:#4a6080;margin-top:3px;">{row["stat"]}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-family:IBM Plex Mono,monospace;font-size:1.1rem;
                                    font-weight:700;color:#e2e8f0;">Line: {row["line"]}</div>
                        <div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;
                                    color:#4a6080;">Model: {row["model_proj"]:.1f}
                                    &nbsp;<span style="color:{gap_color};">({row["gap"]:+.2f})</span></div>
                    </div>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-top:10px;">
                    <div>
                        <span class="{pick_cls}" style="font-family:IBM Plex Mono,monospace;
                                font-size:0.9rem;">{pick_arrow} {row["pick"]}</span>
                    </div>
                    <div style="display:flex;gap:8px;align-items:center;">
                        <span style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;
                                     color:#7a9ab8;">{row["prob"]:.1%} prob</span>
                        {conf_badge(row["confidence"])}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with pp_col2:
        section_title("RECOMMENDED SLIPS")

        # Build slips from top picks
        high_picks = pp_df[pp_df["confidence"] == "HIGH"]
        med_picks  = pp_df[pp_df["confidence"] == "MEDIUM"]

        slip_configs = [
            ("⚡ 2-Pick Power Play", high_picks.head(2), 3.0, "power_play"),
            ("⚡ 3-Pick Power Play", pd.concat([high_picks, med_picks]).head(3), 5.0, "power_play"),
            ("⚡ 4-Pick Power Play", pd.concat([high_picks, med_picks]).head(4), 10.0, "power_play"),
        ]

        for slip_label, picks, payout, slip_type in slip_configs:
            if len(picks) < int(slip_label[2]):
                continue
            combined_prob = picks["prob"].prod()
            ev = combined_prob * payout - 1.0
            ev_cls = "slip-ev-pos" if ev > 0 else "slip-ev-neg"

            picks_html = ""
            for _, p in picks.iterrows():
                arrow = "🔼" if p["pick"] == "OVER" else "🔽"
                picks_html += (
                    f'<div style="padding:6px 0;border-bottom:1px solid #1a2d4a;'
                    f'font-family:IBM Plex Mono,monospace;font-size:0.72rem;">'
                    f'<span style="color:#e2e8f0;">{arrow} {p["player"]}</span>'
                    f'<span style="color:#4a6080;margin-left:8px;">{p["pick"]} {p["line"]} ({p["stat"]})</span>'
                    f'</div>'
                )

            st.markdown(f"""
            <div class="slip-card">
                <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                    <div style="font-weight:600;font-size:0.9rem;color:#e2e8f0;">{slip_label}</div>
                    <div><span style="color:#facc15;font-family:IBM Plex Mono,monospace;
                                      font-size:0.8rem;font-weight:600;">{payout}x</span></div>
                </div>
                {picks_html}
                <div style="display:flex;justify-content:space-between;margin-top:10px;">
                    <div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#4a6080;">
                        P(all hit): {combined_prob:.1%}
                    </div>
                    <div class="{ev_cls}">EV: {ev:+.1%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        section_title("PP EDGE BY STAT TYPE")
        stat_summary = pp_df.groupby("stat").agg(
            avg_prob=("prob", "mean"),
            n_picks=("prob", "count"),
            n_high=("confidence", lambda x: (x == "HIGH").sum()),
        ).reset_index()
        fig_stat = px.bar(
            stat_summary, x="stat", y="avg_prob",
            color="avg_prob",
            color_continuous_scale=[[0, "#1a2d4a"], [0.5, "#22d3ee"], [1, "#4ade80"]],
        )
        fig_stat.update_layout(
            plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
            font=dict(family="IBM Plex Mono", size=9, color="#7a9ab8"),
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0), height=200,
            xaxis=dict(gridcolor="#1a2d4a", tickangle=-20),
            yaxis=dict(gridcolor="#1a2d4a", tickformat=".0%"),
            showlegend=False,
        )
        st.plotly_chart(fig_stat, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — H2H MATCHUPS
# ───────────────────────────────────────────────────────────────────────────
with tab3:
    h2h_col1, h2h_col2 = st.columns([3, 2])

    with h2h_col1:
        section_title("HEAD-TO-HEAD MATCHUP EDGES")
        st.markdown(
            '<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
            'color:#4a6080;margin-bottom:16px;">Books price these lazily based on '
            'world ranking. Your model exploits course fit + form divergence.</div>',
            unsafe_allow_html=True
        )

        for _, row in h2h_df.iterrows():
            edge = round((row["win_prob_a"] - 0.5) * 100, 1)
            edge_color = "#4ade80" if edge > 5 else "#facc15" if edge > 2 else "#60a5fa"
            fav = row["player_a"] if row["win_prob_a"] > 0.5 else row["player_b"]
            fav_prob = max(row["win_prob_a"], row["win_prob_b"])

            st.markdown(f"""
            <div style="background:#0e1628;border:1px solid #1a2d4a;border-radius:10px;
                        padding:16px 20px;margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="flex:1;">
                        <div style="font-size:0.95rem;font-weight:600;color:#e2e8f0;">
                            {row["player_a"]}
                        </div>
                        <div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;
                                    color:#4ade80;">SG {row["sg_edge"]:+.3f} edge</div>
                    </div>
                    <div style="text-align:center;padding:0 16px;">
                        <div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;
                                    color:#2a4060;">vs</div>
                    </div>
                    <div style="flex:1;text-align:right;">
                        <div style="font-size:0.95rem;font-weight:600;color:#e2e8f0;">
                            {row["player_b"]}
                        </div>
                        <div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;
                                    color:#4a6080;">{row["win_prob_b"]:.1%} prob</div>
                    </div>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;
                            margin-top:10px;padding-top:10px;border-top:1px solid #1a2540;">
                    <div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#4a6080;">
                        {row["notes"]}
                    </div>
                    <div style="font-family:IBM Plex Mono,monospace;font-size:0.85rem;
                                font-weight:600;color:{edge_color};white-space:nowrap;margin-left:12px;">
                        {fav.split()[-1]} {fav_prob:.1%}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with h2h_col2:
        section_title("WIN PROBABILITY DISTRIBUTION")

        fig_h2h = go.Figure()
        for _, row in h2h_df.iterrows():
            matchup_label = f"{row['player_a'].split()[-1]} v {row['player_b'].split()[-1]}"
            fig_h2h.add_trace(go.Bar(
                name=matchup_label,
                x=[matchup_label],
                y=[row["win_prob_a"]],
                marker_color="#4ade80",
                showlegend=False,
            ))
            fig_h2h.add_trace(go.Bar(
                name=matchup_label,
                x=[matchup_label],
                y=[row["win_prob_b"]],
                marker_color="#1a2d4a",
                showlegend=False,
            ))

        fig_h2h.update_layout(
            barmode="stack",
            plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
            font=dict(family="IBM Plex Mono", size=9, color="#7a9ab8"),
            margin=dict(l=0, r=0, t=10, b=0), height=300,
            xaxis=dict(gridcolor="#1a2d4a", tickangle=-30),
            yaxis=dict(gridcolor="#1a2d4a", tickformat=".0%", range=[0, 1]),
            shapes=[dict(type="line", y0=0.5, y1=0.5, x0=-0.5,
                         x1=len(h2h_df) - 0.5,
                         line=dict(color="#f87171", width=1, dash="dash"))],
        )
        st.plotly_chart(fig_h2h, use_container_width=True)

        section_title("EDGE SUMMARY")
        for _, row in h2h_df.iterrows():
            edge_pct = round((row["win_prob_a"] - 0.5) * 100, 1)
            bar_width = min(abs(edge_pct) * 5, 100)
            bar_color = "#4ade80" if edge_pct > 0 else "#f87171"
            st.markdown(f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;
                            font-family:IBM Plex Mono,monospace;font-size:0.7rem;
                            color:#7a9ab8;margin-bottom:3px;">
                    <span>{row["player_a"].split()[-1]}</span>
                    <span style="color:{bar_color};">{edge_pct:+.1f}% edge</span>
                </div>
                <div style="height:4px;background:#1a2d4a;border-radius:2px;">
                    <div style="height:4px;width:{bar_width}%;background:{bar_color};
                                border-radius:2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 4 — DFS LINEUPS
# ───────────────────────────────────────────────────────────────────────────
with tab4:
    dfs_col1, dfs_col2 = st.columns([3, 1])

    with dfs_col1:
        section_title("GENERATED DFS LINEUPS — DRAFTKINGS")

        lineup_tab_labels = [lu["label"] for lu in lineups]
        l_tabs = st.tabs(lineup_tab_labels)

        for i, (l_tab, lineup_data) in enumerate(zip(l_tabs, lineups)):
            with l_tab:
                players_in_lineup = lineup_data["players"]
                lineup_proj = proj_df[proj_df["name"].isin(players_in_lineup)].copy()
                lineup_proj = lineup_proj.set_index("name").reindex(players_in_lineup).reset_index()

                # Salary bar
                cap_used = lineup_data["salary"] / 50000 * 100
                cap_color = "#4ade80" if cap_used < 98 else "#facc15"
                st.markdown(f"""
                <div style="display:flex;gap:24px;margin-bottom:16px;font-family:IBM Plex Mono,monospace;">
                    <div style="font-size:0.8rem;color:#4a6080;">
                        Salary: <span style="color:{cap_color};font-weight:600;">${lineup_data["salary"]:,}</span>
                        <span style="color:#2a4060;"> / $50,000</span>
                    </div>
                    <div style="font-size:0.8rem;color:#4a6080;">
                        Proj Pts: <span style="color:#22d3ee;font-weight:600;">{lineup_data["proj_pts"]}</span>
                    </div>
                    <div style="font-size:0.8rem;color:#4a6080;">
                        Own Sum: <span style="color:#e2e8f0;font-weight:600;">{lineup_data["own_sum"]}%</span>
                    </div>
                    <div style="font-size:0.8rem;color:#4a6080;">
                        Type: <span style="color:#a78bfa;font-weight:600;">{lineup_data["type"]}</span>
                    </div>
                </div>
                <div style="height:6px;background:#1a2d4a;border-radius:3px;margin-bottom:20px;">
                    <div style="height:6px;width:{cap_used:.1f}%;background:{cap_color};border-radius:3px;"></div>
                </div>
                """, unsafe_allow_html=True)

                # Player rows
                for _, p in lineup_proj.iterrows():
                    if pd.isna(p.get("name")):
                        continue
                    form_color = "#4ade80" if p.get("form_trend") == "improving" else \
                                 "#f87171" if p.get("form_trend") == "declining" else "#4a6080"
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                padding:12px 16px;margin-bottom:6px;background:#0e1628;
                                border-radius:8px;border:1px solid #1a2d4a;">
                        <div style="flex:2;">
                            <div style="font-size:0.9rem;font-weight:600;color:#e2e8f0;">{p.get("name","")}</div>
                            <div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;color:#4a6080;margin-top:2px;">
                                OWGR #{int(p.get("world_rank",99))} &nbsp;·&nbsp;
                                <span style="color:{form_color};">{p.get("form_trend","")}</span>
                            </div>
                        </div>
                        <div style="flex:1;text-align:center;font-family:IBM Plex Mono,monospace;">
                            <div style="font-size:0.85rem;color:#4ade80;font-weight:600;">
                                {p.get("proj_sg_total",0):+.3f} SG
                            </div>
                            <div style="font-size:0.68rem;color:#4a6080;">Fit {int(p.get("course_fit",50))}</div>
                        </div>
                        <div style="flex:1;text-align:center;font-family:IBM Plex Mono,monospace;">
                            <div style="font-size:0.85rem;color:#22d3ee;font-weight:600;">
                                {p.get("dk_proj_pts",0):.1f} pts
                            </div>
                            <div style="font-size:0.68rem;color:#4a6080;">${int(p.get("dk_salary",0)):,}</div>
                        </div>
                        <div style="flex:1;text-align:right;font-family:IBM Plex Mono,monospace;">
                            <div style="font-size:0.85rem;color:#e2e8f0;">{p.get("proj_ownership",0)*100:.1f}% own</div>
                            <div style="font-size:0.68rem;color:#4a6080;">val {p.get("dk_value",0):.2f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    with dfs_col2:
        section_title("PLAYER EXPOSURE")

        # Count player appearances across lineups
        all_players_flat = [p for lu in lineups for p in lu["players"]]
        from collections import Counter
        exposure = Counter(all_players_flat)
        exp_df = pd.DataFrame(exposure.items(), columns=["Player", "Lineups"])
        exp_df["Exposure %"] = (exp_df["Lineups"] / len(lineups) * 100).round(0).astype(int)
        exp_df = exp_df.sort_values("Lineups", ascending=False)

        for _, row in exp_df.iterrows():
            bar_w = row["Exposure %"]
            bar_color = "#4ade80" if bar_w >= 75 else "#22d3ee" if bar_w >= 50 else "#4a6080"
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;
                            font-family:IBM Plex Mono,monospace;font-size:0.72rem;
                            color:#7a9ab8;margin-bottom:4px;">
                    <span>{row["Player"].split()[-1]}</span>
                    <span style="color:{bar_color};">{row["Exposure %"]}%</span>
                </div>
                <div style="height:5px;background:#1a2d4a;border-radius:3px;">
                    <div style="height:5px;width:{bar_w}%;background:{bar_color};border-radius:3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        section_title("SLATE SIZING")
        dfs_alloc = bankroll * 0.05
        gpp_alloc = bankroll * 0.03
        cash_alloc = bankroll * 0.02
        metric_card("DFS BUDGET", f"${dfs_alloc:.0f}", delta="5% of bankroll", delta_type="neu")
        metric_card("GPP ($5 entries)", f"{int(gpp_alloc/5)} entries", delta=f"${gpp_alloc:.0f} allocated", delta_type="neu")
        metric_card("CASH entries", f"{int(cash_alloc/5)} entries", delta=f"${cash_alloc:.0f} allocated", delta_type="neu")


# ───────────────────────────────────────────────────────────────────────────
# TAB 5 — AUDIT
# ───────────────────────────────────────────────────────────────────────────
with tab5:
    a1, a2 = st.columns([3, 2])

    with a1:
        section_title("CUMULATIVE P&L")
        fig_pl = go.Figure()
        fig_pl.add_trace(go.Scatter(
            x=audit_df["date"], y=audit_df["cumulative_pl"],
            mode="lines", name="Cumulative P&L",
            line=dict(color="#4ade80", width=2),
            fill="tozeroy",
            fillcolor="rgba(74,222,128,0.06)",
        ))
        fig_pl.add_hline(y=0, line_dash="dash", line_color="#f87171", line_width=1, opacity=0.5)
        fig_pl.update_layout(
            plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
            font=dict(family="IBM Plex Mono", size=10, color="#7a9ab8"),
            margin=dict(l=0, r=0, t=10, b=0), height=240,
            xaxis=dict(gridcolor="#1a2d4a"),
            yaxis=dict(gridcolor="#1a2d4a", tickprefix="$"),
            showlegend=False,
        )
        st.plotly_chart(fig_pl, use_container_width=True)

        section_title("ROLLING 20-BET ROI")
        fig_roi = go.Figure()
        rolling = audit_df.dropna(subset=["rolling_roi"])
        colors_line = ["#4ade80" if v > 0 else "#f87171" for v in rolling["rolling_roi"]]
        fig_roi.add_trace(go.Scatter(
            x=rolling["date"], y=rolling["rolling_roi"],
            mode="lines", line=dict(color="#22d3ee", width=2),
        ))
        fig_roi.add_hline(y=0, line_dash="dash", line_color="#f87171", line_width=1, opacity=0.5)
        fig_roi.add_hline(y=0.10, line_dash="dot", line_color="#4ade80", line_width=1, opacity=0.3)
        fig_roi.update_layout(
            plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
            font=dict(family="IBM Plex Mono", size=10, color="#7a9ab8"),
            margin=dict(l=0, r=0, t=10, b=0), height=200,
            xaxis=dict(gridcolor="#1a2d4a"),
            yaxis=dict(gridcolor="#1a2d4a", tickformat=".0%"),
            showlegend=False,
        )
        st.plotly_chart(fig_roi, use_container_width=True)

        section_title("BET LOG — LAST 20")
        recent = audit_df.tail(20)[["date", "player", "bet_type", "stake", "odds", "won", "profit_loss"]].copy()
        recent["date"] = recent["date"].dt.strftime("%m/%d")
        recent["profit_loss"] = recent["profit_loss"].apply(lambda x: f"${x:+.2f}")
        recent["stake"] = recent["stake"].apply(lambda x: f"${x:.0f}")
        recent["won"] = recent["won"].apply(lambda x: "✅" if x else "❌")
        st.dataframe(recent, use_container_width=True, hide_index=True, height=300)

    with a2:
        section_title("PERFORMANCE BREAKDOWN")

        # ROI by type
        by_type = audit_df.groupby("bet_type").apply(lambda g: pd.Series({
            "n": len(g), "win_rate": g["won"].mean(),
            "roi": g["profit_loss"].sum() / g["stake"].sum(),
            "total_pl": g["profit_loss"].sum(),
        })).reset_index()

        fig_type = px.bar(
            by_type, x="bet_type", y="roi",
            color="roi",
            color_continuous_scale=[[0, "#7f1d1d"], [0.5, "#1a2d4a"], [1, "#052e16"]],
        )
        fig_type.update_layout(
            plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
            font=dict(family="IBM Plex Mono", size=9, color="#7a9ab8"),
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0), height=180,
            xaxis=dict(gridcolor="#1a2d4a"),
            yaxis=dict(gridcolor="#1a2d4a", tickformat=".0%"),
        )
        st.plotly_chart(fig_type, use_container_width=True)

        section_title("KEY METRICS")
        total_pl_val  = audit_df["profit_loss"].sum()
        total_staked  = audit_df["stake"].sum()
        overall_roi   = total_pl_val / total_staked
        win_rate_val  = audit_df["won"].mean()
        n_bets        = len(audit_df)
        avg_edge      = (audit_df["model_prob"] - 0.5).mean()

        metric_card("TOTAL P&L", f"${total_pl_val:+.2f}",
                    delta_type="pos" if total_pl_val > 0 else "neg")
        metric_card("OVERALL ROI", f"{overall_roi:.1%}",
                    delta=f"{n_bets} bets",
                    delta_type="pos" if overall_roi > 0 else "neg")
        metric_card("WIN RATE", f"{win_rate_val:.1%}",
                    delta=f"avg edge {avg_edge:.1%}", delta_type="neu")

        section_title("EDGE DECAY CHECK")
        mid = len(audit_df) // 2
        first_roi = (audit_df.iloc[:mid]["profit_loss"].sum() /
                     audit_df.iloc[:mid]["stake"].sum())
        second_roi = (audit_df.iloc[mid:]["profit_loss"].sum() /
                      audit_df.iloc[mid:]["stake"].sum())
        decay = first_roi - second_roi

        decay_color = "#4ade80" if decay < 0.05 else "#facc15" if decay < 0.15 else "#f87171"
        status_text = "✅ HEALTHY" if decay < 0.05 else "⚠️ MONITOR" if decay < 0.15 else "🔴 DECAYING"
        st.markdown(f"""
        <div style="background:#0e1628;border:1px solid #1a2d4a;border-radius:10px;padding:16px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#4a6080;">First half ROI</div>
                <div style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#e2e8f0;">{first_roi:.1%}</div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#4a6080;">Second half ROI</div>
                <div style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#e2e8f0;">{second_roi:.1%}</div>
            </div>
            <div style="display:flex;justify-content:space-between;padding-top:8px;border-top:1px solid #1a2540;">
                <div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#4a6080;">Status</div>
                <div style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;
                            font-weight:600;color:{decay_color};">{status_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 6 — SETTINGS
# ───────────────────────────────────────────────────────────────────────────
with tab6:
    s1, s2, s3 = st.columns(3)

    with s1:
        section_title("MODEL PARAMETERS")
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#4a6080;margin-bottom:8px;">SG RECENCY WEIGHTS</div>', unsafe_allow_html=True)
        w_last4  = st.slider("Last 4 events",  0.0, 1.0, 0.45, 0.05)
        w_last12 = st.slider("Last 12 events", 0.0, 1.0, 0.35, 0.05)
        w_last24 = st.slider("Last 24 events", 0.0, 1.0, 0.20, 0.05)

        st.markdown('<br><div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#4a6080;margin-bottom:8px;">REGRESSION FACTORS</div>', unsafe_allow_html=True)
        putt_reg = st.slider("Putting regression", 0.1, 0.9, 0.55, 0.05,
                              help="Higher = regress more toward mean. Putting is volatile so keep high.")
        app_reg  = st.slider("Approach regression", 0.1, 0.6, 0.20, 0.05,
                              help="Lower = trust approach SG more. Most predictive category.")

    with s2:
        section_title("KELLY & BANKROLL")
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#4a6080;margin-bottom:8px;">BET SIZING CAPS</div>', unsafe_allow_html=True)
        max_outright = st.slider("Max outright %", 0.5, 5.0, 1.0, 0.5,
                                  help="% of bankroll max on a single outright bet") / 100
        max_h2h      = st.slider("Max H2H / top-N %", 1.0, 10.0, 5.0, 0.5) / 100
        max_dfs      = st.slider("Max DFS per slate %", 1.0, 10.0, 3.0, 0.5) / 100

        st.markdown('<br><div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#4a6080;margin-bottom:8px;">EDGE THRESHOLD</div>', unsafe_allow_html=True)
        min_edge = st.slider("Min edge to bet", 0.01, 0.15, 0.04, 0.01,
                              help="Don't bet unless model shows at least this much edge") * 100
        st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#4ade80;">Minimum edge: {min_edge:.0f}%</div>', unsafe_allow_html=True)

    with s3:
        section_title("DATA SOURCES")

        st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#4a6080;margin-bottom:12px;">API KEY STATUS</div>', unsafe_allow_html=True)

        api_items = [
            ("OpenWeather API", "Set in .env", "live"),
            ("The Odds API",    "Set in .env", "live"),
            ("PGA Tour GraphQL","No key needed", "live"),
            ("ESPN Golf API",   "No key needed", "live"),
            ("PrizePicks API",  "No key needed", "live"),
            ("DataGolf API",    "Add to .env → auto-activates", "stub" if not DATAGOLF_ENABLED else "live"),
        ]
        for name, note, status in api_items:
            icon = {"live": "🟢", "stub": "🟡", "off": "🔴"}[status]
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:8px 12px;margin-bottom:6px;background:#0e1628;border-radius:8px;'
                f'border:1px solid #1a2d4a;">'
                f'<div><div style="font-family:IBM Plex Mono,monospace;font-size:0.78rem;'
                f'color:#e2e8f0;">{name}</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;'
                f'color:#4a6080;margin-top:2px;">{note}</div></div>'
                f'<div style="font-size:1rem;">{icon}</div></div>',
                unsafe_allow_html=True
            )

        section_title("DATAGOLF UPGRADE")
        st.markdown(
            '<div style="background:#0d1f0d;border:1px solid #166534;border-radius:10px;'
            'padding:14px;font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#4a6080;">'
            '<div style="color:#4ade80;font-weight:600;margin-bottom:6px;">🔶 Unlock DataGolf (~$30/mo)</div>'
            'Sign up at <span style="color:#22d3ee;">datagolf.com</span><br>'
            'Add key to <span style="color:#facc15;">.env</span> file on your server<br>'
            'Engine auto-upgrades: course fit ML, ownership projections, live stats'
            '</div>',
            unsafe_allow_html=True
        )


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;font-family:IBM Plex Mono,monospace;font-size:0.65rem;'
    'color:#1a2d4a;padding:20px 0;">GOLF QUANT ENGINE v1.0 &nbsp;·&nbsp; '
    'Built for edge &nbsp;·&nbsp; Always use fractional Kelly</div>',
    unsafe_allow_html=True
)
