"""
Golf Quant Engine — Master Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "cache"
DB_PATH  = BASE_DIR / "data" / "golf_quant.db"
LOG_DIR  = BASE_DIR / "logs"
AUDIT_DIR = BASE_DIR / "audit" / "reports"

for p in [DATA_DIR, LOG_DIR, AUDIT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# API KEYS (set in .env)
# ─────────────────────────────────────────────
WEATHER_API_KEY   = os.getenv("OPENWEATHER_API_KEY", "")   # free at openweathermap.org
ODDS_API_KEY      = os.getenv("ODDS_API_KEY", "")          # free tier at the-odds-api.com
DATAGOLF_API_KEY  = os.getenv("DATAGOLF_API_KEY", "")      # future — datagolf.com ~$30/mo
SPORTRADAR_KEY    = os.getenv("SPORTRADAR_KEY", "")        # optional

# ─────────────────────────────────────────────
# DATA SOURCES
# ─────────────────────────────────────────────
PGA_TOUR_BASE       = "https://www.pgatour.com"
PGA_STATS_URL       = "https://www.pgatour.com/stats"
PGA_GRAPHQL_URL     = "https://orchestrator.pgatour.com/graphql"
ESPN_GOLF_API       = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
ESPN_PLAYER_API     = "https://site.web.api.espn.com/apis/common/v3/sports/golf/pga/athletes/{player_id}/overview"
ODDS_API_BASE       = "https://api.the-odds-api.com/v4/sports/golf_pga_championship/odds"
WEATHER_API_BASE    = "https://api.openweathermap.org/data/2.5"
DATAGOLF_BASE       = "https://datagolf.com/api"            # future

# ─────────────────────────────────────────────
# MODEL WEIGHTS — SG CATEGORIES
# [v8.0] Research-validated: OTT > APP > ARG > PUTT for predictive power
# Putting has highest in-tournament impact but lowest week-to-week stability
# ─────────────────────────────────────────────
SG_WEIGHTS = {
    "sg_ott":  0.25,   # Off-the-tee — most stable, most predictive week-to-week
    "sg_app":  0.38,   # Approach — highest impact AND highly predictive
    "sg_atg":  0.22,   # Around-the-green — moderate predictive power
    "sg_putt": 0.15,   # Putting — high variance, LOW predictive power week-to-week
}

# [v8.0] Research-backed recency: ~10% on last week, ~90% on 2-year baseline
# DataGolf's empirically optimized: ~70% weight on most recent 50 rounds
# Counter-intuitive: markets OVER-weight recent form — edge in trusting baseline
FORM_WINDOWS = {
    "last_4":  0.10,   # Most recent ~1 month (was 0.55 — way too aggressive)
    "last_12": 0.25,   # Last ~3 months
    "last_24": 0.30,   # Last ~6 months
    "last_50": 0.35,   # Last ~12-18 months (NEW — the baseline that matters most)
}

# [v8.0] Regression factors — research: long game regresses LESS than short game
# Predictive hierarchy: OTT > APP > ARG > PUTT
PUTTING_REGRESSION_FACTOR = 0.55    # Regress hard — putting is mostly noise week-to-week
OTT_REGRESSION_FACTOR = 0.20        # Most stable — trust it (NEW)
APPROACH_REGRESSION_FACTOR = 0.15   # Most stable and highest impact
ATG_REGRESSION_FACTOR = 0.35        # Moderate regression (NEW)

# Cross-category predictive signal (from DataGolf research):
# A golfer with +1 SG:OTT will have future SG:APP predicted at ~+0.2
# because OTT signals general ball-striking ability
CROSS_CATEGORY_SIGNAL = {
    "sg_ott_to_sg_app": 0.20,   # OTT predicts APP
    "sg_app_to_sg_atg": 0.10,   # APP weakly predicts ATG
    "sg_ott_to_sg_atg": 0.08,   # OTT weakly predicts ATG
}

# Minimum data requirements (from DataGolf methodology)
MIN_ROUNDS_FOR_PROJECTION = 40   # Minimum ~40 rounds in last 2 years
MIN_ROUNDS_RECENT = 1            # At least 1 round in last 6 months
FULL_CONFIDENCE_EVENTS = 30      # Full confidence at 30+ events (was 20)

# ─────────────────────────────────────────────
# KELLY SIZING PARAMETERS
# ─────────────────────────────────────────────
# [v6.0] Aggressive Kelly for 200%+ ROI: higher fraction + higher edge threshold
KELLY_FRACTION       = 0.40     # 2/5 Kelly — aggressive for high-edge plays only
MAX_BET_PCT_BANKROLL = 0.08     # Up to 8% on elite single bets
MIN_EDGE_THRESHOLD   = 0.06     # Minimum 6% edge to place any bet (was 4%)
MAX_GPP_EXPOSURE     = 0.05     # Max 5% bankroll per DFS slate (was 3%)
MAX_H2H_EXPOSURE     = 0.08     # Max 8% on H2H matchups (was 5%)
MAX_OUTRIGHT_EXPOSURE= 0.02     # Max 2% on outright futures (was 1%)

# ─────────────────────────────────────────────
# DFS SETTINGS
# ─────────────────────────────────────────────
DK_SALARY_CAP    = 50000
FD_SALARY_CAP    = 60000
DK_ROSTER_SIZE   = 6
FD_ROSTER_SIZE   = 6

# GPP Lineup targets
TARGET_LINEUPS_GPP   = 20
TARGET_LINEUPS_CASH  = 3
MAX_LINEUP_OWNERSHIP = 130     # Combined % ownership ceiling for GPP lineups
CHALK_THRESHOLD      = 0.20   # Players above 20% owned = chalk

# ─────────────────────────────────────────────
# DATAGOLF INTEGRATION (FUTURE)
# ─────────────────────────────────────────────
DATAGOLF_ENABLED = bool(DATAGOLF_API_KEY)
DATAGOLF_ENDPOINTS = {
    "rankings":       "/preds/get-dg-rankings",
    "predictions":    "/preds/pre-tournament-predictions",
    "sg_categories":  "/historical-raw-data/event",
    "course_fit":     "/preds/apply-course-fit-adjustments",
    "ownership":      "/preds/fantasy-projection-defaults",
    "live_model":     "/preds/live-tournament-stats",
}

# ─────────────────────────────────────────────
# SCRAPING BEHAVIOR
# ─────────────────────────────────────────────
REQUEST_DELAY    = 2.0          # Seconds between requests
REQUEST_TIMEOUT  = 15
MAX_RETRIES      = 3
USER_AGENT       = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

# ─────────────────────────────────────────────
# AUDIT & LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL        = "INFO"
TRACK_ALL_BETS   = True
# [v6.0] Tighter audit controls
MIN_AUDIT_SAMPLE = 20          # Min bets before edge decay analysis (was 30)
ROI_ALERT_THRESHOLD = -0.05    # Alert if rolling ROI drops below -5% (was -10%)
