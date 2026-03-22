"""
Golf Quant Engine — Main Streamlit App
The single-page app for PGA golf betting analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from database.db_manager import (
    init_db, get_active_pp_lines, get_pp_last_scraped,
    get_odds_consensus, get_odds_last_scraped,
    get_current_weather, get_weather_forecast, get_weather_last_fetched,
    get_current_tournament, get_leaderboard, get_bets,
    insert_bet, settle_bet, normalize_name,
)
from scrapers.tournament_detector import detect_current_tournament, get_live_leaderboard
from scrapers.prizepicks_scraper import PrizePicksScraper
from scrapers.odds_api_scraper import OddsAPIScraper
from scrapers.weather_scraper import WeatherScraper, wind_direction_str, weather_impact_factor
from models.probability_calculator import (
    analyze_line, project_stat, calc_probability,
    classify_confidence, kelly_stake, STAT_BASELINES,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
def get_secret(key: str, default: str = "") -> str:
    """Get secret from env vars OR Streamlit secrets."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

ODDS_API_KEY = get_secret("ODDS_API_KEY")
OPENWEATHER_KEY = get_secret("OPENWEATHER_API_KEY")
BANKROLL_DEFAULT = float(get_secret("DEFAULT_BANKROLL", "1000"))
KELLY_FRAC_DEFAULT = float(get_secret("KELLY_FRACTION", "0.25"))
MIN_EDGE_DEFAULT = float(get_secret("MIN_EDGE_THRESHOLD", "0.03"))

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Golf Quant Engine",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database
init_db()

# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
_defaults = {
    "tournament": None,
    "weather_current": None,
    "weather_forecast": None,
    "pp_lines": [],
    "pp_last_scraped": "Never",
    "odds_data": None,
    "odds_quota": {},
    "bankroll": BANKROLL_DEFAULT,
    "kelly_fraction": KELLY_FRAC_DEFAULT,
    "min_edge": MIN_EDGE_DEFAULT,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# DATA LOADING (with TTL caching)
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Detecting tournament...")
def _cached_detect_tournament():
    return detect_current_tournament()

@st.cache_data(ttl=900, show_spinner="Fetching PrizePicks lines...")
def _cached_pp_scrape():
    scraper = PrizePicksScraper()
    return scraper.fetch_golf_lines()

@st.cache_data(ttl=14400, show_spinner="Fetching Odds API...")
def _cached_odds_fetch():
    if not ODDS_API_KEY:
        return {"error": "No ODDS_API_KEY configured", "lines": [], "consensus": []}
    scraper = OddsAPIScraper(ODDS_API_KEY)
    return scraper.fetch_odds()

@st.cache_data(ttl=1800, show_spinner="Fetching weather...")
def _cached_weather_fetch(lat, lon):
    if not OPENWEATHER_KEY or (lat == 0 and lon == 0):
        return {"current": {"error": "No weather data"}, "forecast": []}
    scraper = WeatherScraper(OPENWEATHER_KEY)
    current = scraper.fetch_current(lat, lon)
    forecast = scraper.fetch_forecast(lat, lon)
    return {"current": current, "forecast": forecast[:16]}


# ─────────────────────────────────────────────
# LOAD TOURNAMENT CONTEXT
# ─────────────────────────────────────────────
tournament = _cached_detect_tournament()
if tournament and not tournament.get("error"):
    st.session_state["tournament"] = tournament


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⛳ Golf Quant Engine")

    # Tournament info
    t = st.session_state.get("tournament", {})
    if t and not t.get("error"):
        st.success(f"**{t.get('name', 'Unknown')}**")
        st.caption(f"📍 {t.get('course_name', 'Unknown Course')}")
        status_map = {
            "upcoming": "🟡 Upcoming",
            "in_progress": "🟢 In Progress",
            "completed": "🏁 Completed",
            "delayed": "⚠️ Delayed",
        }
        st.caption(f"Status: {status_map.get(t.get('status', ''), t.get('status', ''))}")
        if t.get("current_round"):
            st.caption(f"Round {t['current_round']}")
        st.caption(f"Field: {t.get('field_count', 0)} players | Par {t.get('par', 72)}")
    else:
        st.warning("No tournament detected")
        if t and t.get("error"):
            st.caption(f"Error: {t['error']}")

    st.divider()

    # Weather widget
    weather_data = None
    if t and t.get("course_lat") and t.get("course_lon"):
        weather_data = _cached_weather_fetch(t["course_lat"], t["course_lon"])
        w = weather_data.get("current", {})
        if w and not w.get("error"):
            wind_dir = wind_direction_str(w.get("wind_direction_deg", 0))
            impact = weather_impact_factor(w.get("wind_speed_mph", 0), w.get("precipitation_mm", 0))

            st.markdown("### 🌤️ Course Weather")
            col1, col2 = st.columns(2)
            col1.metric("Temp", f"{w.get('temp_f', 0):.0f}°F")
            col2.metric("Humidity", f"{w.get('humidity_pct', 0)}%")
            col1, col2 = st.columns(2)
            col1.metric("Wind", f"{w.get('wind_speed_mph', 0):.0f} mph {wind_dir}")
            gusts = w.get("wind_gust_mph", 0)
            if gusts:
                col2.metric("Gusts", f"{gusts:.0f} mph")
            else:
                col2.metric("Clouds", f"{w.get('cloud_cover_pct', 0)}%")
            st.caption(f"☁️ {w.get('description', '').title()}")

            if impact["is_significant"]:
                st.warning(f"⚠️ {impact['description']}")
            st.session_state["weather_current"] = w
            st.session_state["weather_impact"] = impact

    st.divider()

    # System status
    st.markdown("### 📡 System Status")

    # PrizePicks status
    pp_lines = _cached_pp_scrape()
    pp_count = len(pp_lines.get("lines", []))
    if pp_count > 0:
        st.markdown(f"🟢 **PrizePicks**: {pp_count} PGA lines")
    elif pp_lines.get("error"):
        st.markdown(f"🔴 **PrizePicks**: {pp_lines['error'][:80]}")
    else:
        st.markdown("🟡 **PrizePicks**: No PGA lines available")

    # Odds API status
    odds_data = _cached_odds_fetch()
    odds_count = len(odds_data.get("consensus", []))
    if odds_count > 0:
        st.markdown(f"🟢 **Odds API**: {odds_count} players")
        q = odds_data.get("quota", {})
        if q.get("remaining"):
            st.caption(f"Quota: {q['remaining']} remaining")
    elif odds_data.get("error"):
        st.markdown(f"🔴 **Odds API**: Error")
        st.caption(odds_data["error"][:80])
    else:
        st.markdown("🟡 **Odds API**: No data")

    # Weather status
    if weather_data and not weather_data.get("current", {}).get("error"):
        st.markdown("🟢 **Weather**: Live")
    elif OPENWEATHER_KEY:
        st.markdown("🟡 **Weather**: No coordinates")
    else:
        st.markdown("🔴 **Weather**: No API key")

    st.divider()

    # Settings
    st.markdown("### ⚙️ Settings")
    st.session_state["bankroll"] = st.number_input(
        "Bankroll ($)", min_value=10.0, value=float(st.session_state["bankroll"]),
        step=100.0, key="settings_bankroll"
    )
    st.session_state["min_edge"] = st.slider(
        "Min Edge %", 1, 20, int(st.session_state["min_edge"] * 100),
        key="settings_edge"
    ) / 100.0
    st.session_state["kelly_fraction"] = st.select_slider(
        "Kelly Fraction",
        options=[0.10, 0.15, 0.20, 0.25, 0.33, 0.50],
        value=st.session_state["kelly_fraction"],
        key="settings_kelly"
    )

    # Refresh buttons
    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("🔄 Refresh All", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    if col2.button("🔄 PP Lines", use_container_width=True):
        _cached_pp_scrape.clear()
        st.rerun()


# ─────────────────────────────────────────────
# MAIN CONTENT — TABS
# ─────────────────────────────────────────────
tab_pp, tab_rankings, tab_leaderboard, tab_bets, tab_status = st.tabs([
    "🎯 PrizePicks Lab", "🏆 Power Rankings", "📊 Live Leaderboard",
    "💰 Bet Tracker", "🔧 System Status"
])


# ═══════════════════════════════════════════════
# TAB 1: PRIZEPICKS LAB
# ═══════════════════════════════════════════════
with tab_pp:
    st.markdown("# 🎯 PrizePicks Lab")

    # Status bar
    t = st.session_state.get("tournament", {})
    t_name = t.get("name", "No Tournament") if t else "No Tournament"

    status_cols = st.columns(4)
    status_cols[0].metric("PGA Lines", pp_count)
    status_cols[1].metric("Books", odds_count)
    status_cols[2].metric("Tournament", t_name[:25])
    w_impact = st.session_state.get("weather_impact", {})
    if w_impact.get("is_significant"):
        status_cols[3].metric("Weather", f"⚠️ +{(w_impact['variance_mult']-1)*100:.0f}% var")
    else:
        status_cols[3].metric("Weather", "✅ Normal")

    # Build consensus lookup
    consensus_lookup = {}
    for c in odds_data.get("consensus", []):
        key = normalize_name(c.get("player_name", ""))
        consensus_lookup[key] = c.get("consensus_prob", 0)

    # Weather adjustment
    weather_adj = None
    wc = st.session_state.get("weather_current", {})
    if wc and not wc.get("error"):
        weather_adj = weather_impact_factor(
            wc.get("wind_speed_mph", 0),
            wc.get("precipitation_mm", 0)
        )

    # Analyze all PrizePicks lines
    pp_raw = pp_lines.get("lines", [])
    analyses = []
    for line in pp_raw:
        player = line.get("player_name", "Unknown")
        stat = line.get("stat_type", "unknown")
        line_val = line.get("line_value", 0)

        # Default SG profile (tour average) — will be enhanced with real data
        player_sg = {"sg_total": 0, "sg_app": 0, "sg_putt": 0, "sg_ott": 0, "sg_atg": 0}

        # If we have consensus odds, infer SG from outright probability
        player_norm = normalize_name(player)
        consensus = consensus_lookup.get(player_norm)
        if consensus and consensus > 0:
            # Higher outright prob → better player → positive SG
            # Top 10 player (~5-15% win prob) → sg_total ~1.0-2.0
            # Average field (~0.5-2%) → sg_total ~0
            # Longshot (<0.5%) → sg_total ~ -0.5 to -1.0
            import math
            if consensus > 0.001:
                sg_est = max(-2.0, min(3.0, math.log(consensus * 100) * 0.6))
            else:
                sg_est = -1.5
            player_sg = {
                "sg_total": sg_est,
                "sg_app": sg_est * 0.35,
                "sg_putt": sg_est * 0.25,
                "sg_ott": sg_est * 0.25,
                "sg_atg": sg_est * 0.15,
            }

        analysis = analyze_line(
            player_name=player,
            stat_type=stat,
            line_value=line_val,
            player_sg=player_sg,
            weather_adj=weather_adj,
            consensus_prob=consensus,
            bankroll=st.session_state["bankroll"],
            kelly_frac=st.session_state["kelly_fraction"],
        )
        analysis["stat_display"] = line.get("stat_type_display", stat)
        analyses.append(analysis)

    # Filter by minimum edge
    min_edge = st.session_state.get("min_edge", 0.03)
    bettable = [a for a in analyses if a.get("best_edge", 0) >= min_edge
                and a.get("confidence") != "NO_BET"
                and not a.get("error")]
    bettable.sort(key=lambda x: x.get("best_edge", 0), reverse=True)

    if not pp_raw:
        st.info(
            "📭 **No PrizePicks PGA lines available.**\n\n"
            "Lines typically go live Tuesday morning of tournament week. "
            f"Last check: {pp_lines.get('timestamp', 'Unknown')}\n\n"
            + (f"Error: {pp_lines.get('error', '')}" if pp_lines.get("error") else "")
        )
    else:
        # ── BEST BETS BOARD ──
        st.markdown("## 🔥 Best Bets")
        top_bets = bettable[:8]
        if top_bets:
            for i in range(0, len(top_bets), 4):
                cols = st.columns(min(4, len(top_bets) - i))
                for j, col in enumerate(cols):
                    if i + j >= len(top_bets):
                        break
                    bet = top_bets[i + j]
                    conf_colors = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🟠", "NO_BET": "🔴"}
                    conf_icon = conf_colors.get(bet["confidence"], "⚪")
                    mkt_icon = "✅" if bet.get("market_alignment") is True else (
                        "⚠️" if bet.get("market_alignment") is False else "—"
                    )
                    with col:
                        st.markdown(f"""
                        **{bet['player_name']}**
                        {bet.get('stat_display', bet['stat_type'])}

                        **{bet['best_side']}** {bet['line_value']}
                        Proj: {bet['projection']:.1f} (σ={bet['std_dev']:.1f})

                        Edge: **{bet['best_edge']*100:+.1f}%** | P: {bet['best_prob']*100:.0f}%
                        {conf_icon} {bet['confidence']} | Mkt: {mkt_icon}
                        Kelly: ${bet['kelly_stake']:.0f} ({bet['kelly_pct']:.1f}%)
                        """)
                        if bet.get("warnings"):
                            for w in bet["warnings"]:
                                st.warning(w)
        else:
            st.info("No bets meet the minimum edge threshold. Try lowering the edge filter.")

        # ── FULL EDGE TABLE ──
        st.markdown("## 📋 All Lines Analysis")

        # Filters
        filter_cols = st.columns(4)
        stat_types = sorted(set(a.get("stat_display", a.get("stat_type", "")) for a in analyses))
        stat_filter = filter_cols[0].selectbox("Stat Type", ["All"] + stat_types, key="stat_filter")
        conf_filter = filter_cols[1].selectbox("Confidence", ["All", "HIGH", "MEDIUM", "LOW"], key="conf_filter")
        side_filter = filter_cols[2].selectbox("Side", ["All", "OVER", "UNDER"], key="side_filter")
        sort_by = filter_cols[3].selectbox("Sort by", ["Edge %", "Confidence", "Player"], key="sort_by")

        # Apply filters
        filtered = analyses.copy()
        if stat_filter != "All":
            filtered = [a for a in filtered if a.get("stat_display") == stat_filter or a.get("stat_type") == stat_filter]
        if conf_filter != "All":
            filtered = [a for a in filtered if a.get("confidence") == conf_filter]
        if side_filter != "All":
            filtered = [a for a in filtered if a.get("best_side") == side_filter]

        # Sort
        if sort_by == "Edge %":
            filtered.sort(key=lambda x: x.get("best_edge", 0), reverse=True)
        elif sort_by == "Confidence":
            conf_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NO_BET": 3}
            filtered.sort(key=lambda x: (conf_order.get(x.get("confidence", "NO_BET"), 9), -x.get("best_edge", 0)))
        else:
            filtered.sort(key=lambda x: x.get("player_name", ""))

        if filtered:
            # Build DataFrame for display
            rows = []
            for a in filtered:
                if a.get("error"):
                    continue
                rows.append({
                    "Player": a["player_name"],
                    "Stat": a.get("stat_display", a["stat_type"]),
                    "Line": a["line_value"],
                    "Proj": f"{a['projection']:.1f}",
                    "σ": f"{a['std_dev']:.1f}",
                    "Over%": f"{a['prob_over']*100:.0f}%",
                    "Under%": f"{a['prob_under']*100:.0f}%",
                    "Side": a["best_side"],
                    "Edge": f"{a['best_edge']*100:+.1f}%",
                    "Conf": a["confidence"],
                    "Mkt": "✅" if a.get("market_alignment") is True else (
                        "⚠️" if a.get("market_alignment") is False else "—"
                    ),
                    "Kelly $": f"${a['kelly_stake']:.0f}",
                    "Z": f"{a['z_score']:.2f}",
                })

            df = pd.DataFrame(rows)

            # Color the dataframe
            def color_edge(val):
                try:
                    v = float(val.replace("%", "").replace("+", ""))
                    if v >= 8: return "color: #00FF88; font-weight: bold"
                    if v >= 5: return "color: #00AAFF"
                    if v >= 3: return "color: #FFB800"
                    return "color: #FF4444"
                except: return ""

            def color_conf(val):
                if val == "HIGH": return "color: #00FF88; font-weight: bold"
                if val == "MEDIUM": return "color: #FFB800"
                if val == "LOW": return "color: #FF8800"
                return "color: #FF4444"

            styled = df.style.map(color_edge, subset=["Edge"]).map(color_conf, subset=["Conf"])
            st.dataframe(styled, use_container_width=True, hide_index=True, height=600)

            # Expandable detail view
            with st.expander("📊 Detailed Analysis (click to expand)"):
                for a in filtered[:20]:
                    if a.get("error"):
                        continue
                    st.markdown(f"""
                    **{a['player_name']}** — {a.get('stat_display', a['stat_type'])} {a['best_side']} {a['line_value']}
                    - Projection: {a['projection']:.2f} ± {a['std_dev']:.2f} | Z-score: {a['z_score']:.3f}
                    - P(Over): {a['prob_over']*100:.1f}% | P(Under): {a['prob_under']*100:.1f}%
                    - Edge: {a['best_edge']*100:+.2f}% | Confidence: {a['confidence']}
                    - Market Consensus: {a.get('consensus_prob', 'N/A')}
                    - Weather: {a.get('weather_impact', 'None')}
                    - Kelly: ${a['kelly_stake']:.2f} ({a['kelly_pct']:.1f}%) | EV: ${a['ev_dollars']:.2f}
                    {'- ⚠️ ' + ', '.join(a['warnings']) if a.get('warnings') else ''}
                    ---
                    """)
        else:
            st.info("No lines match your filters.")

        # Stats summary
        st.markdown("## 📈 Session Summary")
        sum_cols = st.columns(4)
        sum_cols[0].metric("Total Lines", len(analyses))
        sum_cols[1].metric("Bettable", len(bettable))
        high = len([a for a in bettable if a.get("confidence") == "HIGH"])
        sum_cols[2].metric("HIGH Confidence", high)
        if bettable:
            avg_edge = np.mean([a["best_edge"] for a in bettable]) * 100
            sum_cols[3].metric("Avg Edge", f"{avg_edge:+.1f}%")


# ═══════════════════════════════════════════════
# TAB 2: POWER RANKINGS
# ═══════════════════════════════════════════════
with tab_rankings:
    st.markdown("# 🏆 Power Rankings")

    consensus_list = odds_data.get("consensus", [])
    if consensus_list:
        st.markdown(f"*Based on {len(consensus_list)} players from {odds_count} sportsbooks*")

        rows = []
        for i, c in enumerate(consensus_list[:50], 1):
            prob = c.get("consensus_prob", 0)
            implied_odds = f"+{int(1/prob * 100 - 100)}" if prob > 0.01 else "+99999"
            rows.append({
                "Rank": i,
                "Player": c["player_name"],
                "Win %": f"{prob*100:.1f}%",
                "Implied Odds": implied_odds,
                "Books": c.get("num_books", 0),
                "Min": f"{c.get('min_prob', 0)*100:.1f}%",
                "Max": f"{c.get('max_prob', 0)*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=700)
    else:
        st.info("No odds data available. Odds are fetched from The Odds API.")
        if odds_data.get("error"):
            st.error(odds_data["error"])


# ═══════════════════════════════════════════════
# TAB 3: LIVE LEADERBOARD
# ═══════════════════════════════════════════════
with tab_leaderboard:
    st.markdown("# 📊 Live Leaderboard")

    if st.button("🔄 Refresh Leaderboard"):
        leaderboard = get_live_leaderboard()
        st.session_state["_leaderboard"] = leaderboard

    leaderboard = st.session_state.get("_leaderboard")
    if leaderboard is None:
        leaderboard = get_live_leaderboard()
        st.session_state["_leaderboard"] = leaderboard

    if leaderboard:
        # Highlight players with PP lines
        pp_players = set(normalize_name(l.get("player_name", "")) for l in pp_raw)

        rows = []
        for entry in leaderboard[:50]:
            has_pp = normalize_name(entry.get("name", "")) in pp_players
            rows.append({
                "Pos": entry.get("position", ""),
                "Player": entry.get("name", ""),
                "Score": entry.get("total_score", 0),
                "Today": entry.get("today_score", 0),
                "Thru": entry.get("thru", ""),
                "R1": entry.get("round1", ""),
                "R2": entry.get("round2", ""),
                "R3": entry.get("round3", ""),
                "R4": entry.get("round4", ""),
                "PP": "🎯" if has_pp else "",
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=700)
        else:
            st.info("No leaderboard data. Tournament may not have started yet.")
    else:
        st.info("No leaderboard data available.")


# ═══════════════════════════════════════════════
# TAB 4: BET TRACKER
# ═══════════════════════════════════════════════
with tab_bets:
    st.markdown("# 💰 Bet Tracker")

    bets = get_bets()
    if bets:
        df_bets = pd.DataFrame(bets)
        settled = [b for b in bets if b.get("result") in ("won", "lost")]
        pending = [b for b in bets if b.get("result") == "pending"]

        if settled:
            total_staked = sum(b.get("bet_amount", 0) for b in settled)
            total_pl = sum(b.get("profit_loss", 0) for b in settled)
            wins = len([b for b in settled if b["result"] == "won"])
            roi = (total_pl / total_staked * 100) if total_staked > 0 else 0

            m_cols = st.columns(4)
            m_cols[0].metric("Bets Settled", len(settled))
            m_cols[1].metric("Win Rate", f"{wins/len(settled)*100:.0f}%")
            m_cols[2].metric("Total P&L", f"${total_pl:+.2f}")
            m_cols[3].metric("ROI", f"{roi:+.1f}%")

        if pending:
            st.markdown("### Pending Bets")
            for b in pending:
                col1, col2, col3 = st.columns([3, 1, 1])
                col1.write(f"**{b['player_name']}** {b['stat_type']} {b['pick_side']} {b['line_value']}")
                col2.write(f"${b['bet_amount']:.2f}")
                if col3.button("Settle", key=f"settle_{b['id']}"):
                    st.session_state[f"settling_{b['id']}"] = True

        st.markdown("### All Bets")
        display_cols = ["player_name", "stat_type", "pick_side", "line_value",
                       "edge_pct", "confidence", "bet_amount", "result", "profit_loss", "placed_at"]
        available_cols = [c for c in display_cols if c in df_bets.columns]
        st.dataframe(df_bets[available_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No bets tracked yet. Place bets from the PrizePicks Lab tab.")


# ═══════════════════════════════════════════════
# TAB 5: SYSTEM STATUS
# ═══════════════════════════════════════════════
with tab_status:
    st.markdown("# 🔧 System Status & Debug")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Tournament Context")
        t = st.session_state.get("tournament", {})
        if t and not t.get("error"):
            st.json({
                "name": t.get("name"),
                "course": t.get("course_name"),
                "status": t.get("status"),
                "round": t.get("current_round"),
                "par": t.get("par"),
                "lat": t.get("course_lat"),
                "lon": t.get("course_lon"),
                "field_size": t.get("field_count"),
                "espn_id": t.get("espn_id"),
            })
        else:
            st.error("No tournament context")
            if t:
                st.json(t)

        st.markdown("### PrizePicks Scraper")
        st.json({
            "lines_found": pp_count,
            "league_id": pp_lines.get("league_id"),
            "error": pp_lines.get("error"),
            "timestamp": pp_lines.get("timestamp"),
        })
        if pp_lines.get("raw_response"):
            with st.expander("Raw PP Response"):
                st.code(pp_lines["raw_response"][:3000])

    with col2:
        st.markdown("### Odds API")
        st.json({
            "players": odds_count,
            "sport_key": odds_data.get("sport_key"),
            "error": odds_data.get("error"),
            "quota": odds_data.get("quota"),
        })
        if odds_data.get("raw_response"):
            with st.expander("Raw Odds Response"):
                st.code(odds_data["raw_response"][:3000])

        st.markdown("### Weather")
        wc = st.session_state.get("weather_current", {})
        if wc:
            st.json(wc)

    st.markdown("### API Keys Status")
    st.json({
        "ODDS_API_KEY": f"{'✅ Set' if ODDS_API_KEY else '❌ Missing'} ({ODDS_API_KEY[:8]}...)" if ODDS_API_KEY else "❌ Missing",
        "OPENWEATHER_KEY": f"{'✅ Set' if OPENWEATHER_KEY else '❌ Missing'}",
        "DB_PATH": str(Path(__file__).parent / "data" / "golf_engine.db"),
    })
