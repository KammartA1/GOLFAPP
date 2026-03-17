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
# Tuned from historical PGA correlation studies
# ─────────────────────────────────────────────
SG_WEIGHTS = {
    "sg_ott":  0.18,   # Off-the-tee
    "sg_app":  0.38,   # Approach — highest predictive power
    "sg_atg":  0.22,   # Around-the-green
    "sg_putt": 0.22,   # Putting — highest variance, regress hard
}

# Recency decay weights for form model
FORM_WINDOWS = {
    "last_4":  0.45,
    "last_12": 0.35,
    "last_24": 0.20,
}

# SG putting regression factor (pulled toward mean due to high variance)
PUTTING_REGRESSION_FACTOR = 0.55

# Approach SG regression factor (most stable — regress less)
APPROACH_REGRESSION_FACTOR = 0.20

# ─────────────────────────────────────────────
# KELLY SIZING PARAMETERS
# ─────────────────────────────────────────────
KELLY_FRACTION       = 0.25     # 1/4 Kelly — conservative for golf variance
MAX_BET_PCT_BANKROLL = 0.05     # Never more than 5% on any single bet
MIN_EDGE_THRESHOLD   = 0.04     # Minimum 4% edge to place a bet
MAX_GPP_EXPOSURE     = 0.03     # Max 3% bankroll per DFS slate
MAX_H2H_EXPOSURE     = 0.05     # Max 5% on H2H matchups
MAX_OUTRIGHT_EXPOSURE= 0.01     # Max 1% on outright futures

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
MIN_AUDIT_SAMPLE = 30          # Min bets before edge decay analysis
ROI_ALERT_THRESHOLD = -0.10    # Alert if rolling ROI drops below -10%
