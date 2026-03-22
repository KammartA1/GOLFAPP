# Golf Quant Engine — Audit Findings

## Date: 2026-03-22

## What Works
- **ESPN Golf API**: Tournament detection via scoreboard endpoint works. Returns current event, field, status, round info.
- **PrizePicks API**: Live scraping works. 186 PGA lines returned for Valspar Championship (birdies, strokes, GIR, pars, matchups).
- **The Odds API**: Live key works. Returns Masters Tournament futures with 113 players across 5 bookmakers. Free tier only has major championship winner markets (no weekly event odds).
- **OpenWeather API**: Live key works. Returns current conditions and 5-day forecast for course coordinates.
- **SG Model (models/strokes_gained.py)**: Bayesian regression framework with recency weighting, mean regression, course fit. Structurally sound but has no live data feeding it.
- **Kelly Criterion (betting/kelly.py)**: Full/fractional Kelly with exposure caps. Working math.
- **Config system (config/settings.py)**: Loads from .env via python-dotenv. Has correct API endpoints.
- **Database (data/golf_quant.db)**: SQLite exists but only has empty/scaffold tables from old ORM.

## What Was Scaffolding/Placeholder
- **models/projection.py**: Has sg_to_win_prob() and sg_to_top_n_prob() but Monte Carlo simulation loop is a placeholder (`pass` in the loop body). Uses simplified normal approximation instead.
- **betting/prizepicks.py**: Had old PrizePicksScraper class with hardcoded stat sensitivities. Never connected to live API.
- **betting/h2h.py**: H2H probability model exists but fixed slope, no tournament-specific calibration.
- **models/ownership.py**: Ownership prediction model exists but hand-tuned weights, never validated.
- **dashboard.py**: 3,400-line Streamlit app with Power Rankings, Course Fit, Monte Carlo tabs. BUT all tabs were pulling from old/dummy projection data, not live scraped data.

## What Was Broken
- **config/settings.py line 38**: `ODDS_API_BASE` pointed to `golf_pga_championship` (wrong sport key). Odds API uses event-specific keys like `golf_masters_tournament_winner`.
- **No live data pipeline**: No scraper was connected to the database. No scraper was connected to the dashboard. Dashboard showed stale/dummy data.
- **No .env file existed**: API keys were only in .streamlit/secrets.toml. Local development had no way to load keys.
- **dashboard.py**: Main app had no PrizePicks tab, no live line analysis, no consensus comparison.

## What Was Missing Entirely
- **PrizePicks live scraper**: No working connection to api.prizepicks.com/projections
- **Odds API scraper**: No working connection (wrong sport key, no vig removal, no consensus calculation)
- **Weather integration**: No OpenWeather connection despite API key being available
- **Live leaderboard**: No ESPN leaderboard pulling
- **Stat-type specific projections**: Engine projected generic SG but didn't convert to PrizePicks stat types (fantasy score, birdies, GIR, etc.)
- **Probability calculator**: No over/under probability math for PrizePicks lines
- **Consensus engine**: No cross-reference between model projections and market odds
- **Confidence scoring system**: No HIGH/MEDIUM/LOW/NO_BET classification
- **Best Bets board**: No ranked, actionable picks display
- **Bet tracker**: No bet logging or P&L tracking
- **Database schema for scraped data**: No tables for PP lines, odds, weather, live scores

## What Was Built (v7.0 Rebuild)
1. **database/db_manager.py**: Full SQLite schema with tables for tournaments, PP lines, odds, weather, live scores, projections, bet tracker
2. **scrapers/tournament_detector.py**: ESPN API integration with course coordinate lookup and tournament-to-course mapping
3. **scrapers/prizepicks_scraper.py**: Live PrizePicks API scraper with JSON:API parsing, league auto-detection, stat type normalization
4. **scrapers/odds_api_scraper.py**: The Odds API integration with sport key discovery, vig removal, consensus probability calculation
5. **scrapers/weather_scraper.py**: OpenWeather integration with current conditions, 5-day forecast, wind impact factor calculation
6. **models/probability_calculator.py**: Full probability engine with stat-type baselines, SG sensitivities, normal distribution math, Kelly sizing, confidence scoring
7. **app.py**: Complete Streamlit app with PrizePicks Lab, Power Rankings, Live Leaderboard, Bet Tracker, System Status tabs

## API Key Status
| API | Key Status | Tier | Limits |
|-----|-----------|------|--------|
| The Odds API | ✅ Live | Free | 500 req/month (95,131 remaining on current key) |
| OpenWeather | ✅ Live | Free | 1,000 calls/day |
| PrizePicks | ✅ Public API | N/A | No key needed |
| ESPN Golf | ✅ Public API | N/A | No key needed |
| DataGolf | ❌ No key | Paid | $30/month |
| Anthropic Claude | ✅ Live | Paid | API key in secrets |
