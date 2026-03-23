"""
Golf Quant Engine — Service Layer
===================================
All computation is performed here. Streamlit reads from DB and writes
user inputs to DB — no computation in the frontend.

Service modules:
  - bet_service:        Place, settle, query, and analyse bets. P&L and CLV.
  - event_service:      Tournament CRUD and field management.
  - odds_service:       Odds and line data reads/writes.
  - player_service:     Player records, SG stats, tournament field lookups.
  - projection_service: SG projection pipeline and probability calculator.
  - report_service:     Edge reports, calibration, system state.
  - settings_service:   User preferences, bankroll, model params, risk limits.
  - weather_service:    Weather data fetching, caching, scoring adjustments.

Usage:
    from services.bet_service import place_bet, get_pnl_summary
    from services.event_service import get_current_tournament
    from services.odds_service import get_prizepicks_lines
"""
