"""
Golf Quant Engine — App Configuration
======================================
Page config, theme settings, and app-wide constants.
All user-specific settings live in the database via settings_service.
"""

# Page configuration for Streamlit
PAGE_CONFIG = {
    "page_title": "Golf Quant Engine",
    "page_icon": "\u26f3",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Tab definitions (label, icon)
TABS = [
    "\U0001f3af PrizePicks Lab",
    "\U0001f3c6 Power Rankings",
    "\U0001f4ca Live Leaderboard",
    "\U0001f4b0 Bet Tracker",
    "\U0001f527 System Status",
    "\U0001f9ee Quant System",
    "\U0001f4c8 CLV System",
]

# Default user ID (single-user mode)
DEFAULT_USER_ID = "default"

# Confidence level display config
CONFIDENCE_COLORS = {
    "HIGH": "\U0001f7e2",
    "MEDIUM": "\U0001f7e1",
    "LOW": "\U0001f7e0",
    "NO_BET": "\U0001f534",
}

CONFIDENCE_SORT_ORDER = {
    "HIGH": 0,
    "MEDIUM": 1,
    "LOW": 2,
    "NO_BET": 3,
}

# Status display map for tournaments
STATUS_DISPLAY = {
    "upcoming": "\U0001f7e1 Upcoming",
    "scheduled": "\U0001f7e1 Scheduled",
    "in_progress": "\U0001f7e2 In Progress",
    "live": "\U0001f7e2 Live",
    "completed": "\U0001f3c1 Completed",
    "delayed": "\u26a0\ufe0f Delayed",
    "cancelled": "\u274c Cancelled",
}

# System state display
SYSTEM_STATE_COLORS = {
    "ACTIVE": "\U0001f7e2",
    "REDUCED": "\U0001f7e1",
    "SUSPENDED": "\U0001f7e0",
    "KILLED": "\U0001f534",
    "UNKNOWN": "\u26aa",
}

# Premium CSS theme
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

.metric-card {
    background: linear-gradient(135deg, #0c1425 0%, #111b2e 50%, #0c1425 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 8px;
}
</style>
"""
