# ⛳ Golf Quant Engine v1.0
### The World's Best Open-Source Golf Analytics Platform

Built for edge in DFS GPPs, H2H matchups, and outright betting markets.
Designed for Claude Code — run every command from your terminal.

---

## 🏗️ Architecture

```
golf_quant_engine/
├── config/
│   ├── settings.py          # All tunable parameters
│   └── courses.py           # Course fingerprint database (YOUR DOMAIN EDGE)
├── data/
│   ├── scrapers/
│   │   ├── pga_tour.py      # PGA Tour GraphQL API (free)
│   │   ├── espn.py          # ESPN Golf API (free)
│   │   ├── weather.py       # OpenWeatherMap (free tier)
│   │   ├── dfs_salaries.py  # DK/FD salary ingestion
│   │   └── datagolf.py      # DataGolf API (🔶 STUBBED → activate with key)
│   ├── storage/
│   │   └── database.py      # SQLite — all data persisted here
│   └── pipeline.py          # Orchestrates full data refresh
├── models/
│   ├── strokes_gained.py    # SG projection engine (5-step pipeline)
│   ├── course_fit.py        # Course fit matrix (uses your golf knowledge)
│   ├── ownership.py         # DFS ownership projection model
│   └── projection.py        # Master projection engine
├── betting/
│   ├── kelly.py             # Kelly criterion + bankroll management
│   ├── h2h.py               # H2H matchup analyzer
│   └── outrights.py         # Outright / futures model
├── dfs/
│   ├── optimizer.py         # LP-based lineup optimizer (PuLP)
│   └── stacking.py          # Correlation stacking logic
├── audit/
│   └── tracker.py           # Full performance audit system
└── main.py                  # CLI command center
```

---

## 🚀 Quick Start

### 1. Install
```bash
cd golf_quant_engine
pip install -r requirements.txt --break-system-packages
```

### 2. Configure
```bash
cp .env.example .env
# Add your free API keys to .env:
#   OPENWEATHER_API_KEY — openweathermap.org (free)
#   ODDS_API_KEY        — the-odds-api.com (free tier)
```

### 3. Run
```bash
# Check status
python main.py status

# Full projection run for current tournament
python main.py run

# With DFS salaries (export CSV from DK contest lobby)
python main.py run --dk-csv draftkings_salaries.csv --bankroll 500

# Specific H2H matchup analysis
python main.py matchup "Scottie Scheffler" "Rory McIlroy" --odds-a -150 --odds-b +120

# Performance audit
python main.py audit
```

---

## 🧠 The Model Pipeline

### 5-Step SG Projection
```
1. Recency-Weighted SG
   └── Last 4 events: 45%
       Last 12 events: 35%
       Last 24 events: 20%

2. Regression to Mean (Bayesian Shrinkage)
   └── sg_putt: 55% regression  ← most volatile
       sg_atg:  35% regression
       sg_ott:  30% regression
       sg_app:  20% regression  ← most predictive, regress least

3. Course Fit Adjustment
   └── Each course has a weight vector across SG categories
       (Your golf knowledge → most powerful edge source)

4. Surface Adjustment
   └── Bermuda vs Bentgrass putting split

5. Form Trend Signal
   └── Linear regression on recent SG values
       Improving / Stable / Declining
```

### Win Probability → Kelly Sizing
```
proj_sg_total → win_prob / top10_prob / make_cut_prob
→ Kelly fraction (1/4 Kelly default)
→ Position size capped by bet type:
   Outrights:  max 1% bankroll
   H2H / Top N: max 5% bankroll
   DFS GPP:    max 3% per slate
```

---

## 💰 Edge Sources (Ranked by Consistency)

| Source | Sustainability | Notes |
|--------|---------------|-------|
| Course Fit Model | Very High | Your golf knowledge is the moat |
| Ownership Leverage | High | Mechanical, scalable in GPPs |
| Weather/Tee Time | High | Systematic market underweighting |
| SG Trend Detection | Medium | Leading indicator |
| H2H Mispricing | Medium-High | Books are lazy |
| Outright Win | Low (variance) | Use sparingly |

---

## 📊 Data Sources

### Free (Available Now)
- **PGA Tour GraphQL API** — SG stats, field, schedule, results
- **ESPN Golf API** — Live scores, basic player data
- **OpenWeatherMap** — 5-day forecast, tee time analysis (free key)
- **The Odds API** — Sportsbook odds for Kelly sizing (free key, 500/mo)

### Future: DataGolf (~$30/mo)
Set `DATAGOLF_API_KEY` in `.env` — the entire DataGolf layer auto-activates:
- Course-fit adjusted projections (ML-based, not manual)
- Accurate DFS ownership projections
- Historical raw SG data (deep backtesting)
- Live in-tournament model updates

---

## 🔧 Customization (Your Domain Edge)

### Course Fingerprints (`config/courses.py`)
The most important file. Your golf knowledge lives here.
Every course you add/tune = edge over people using generic models.

```python
"Bay Hill": {
    "sg_weights": {"sg_ott": 0.20, "sg_app": 0.42, "sg_atg": 0.20, "sg_putt": 0.18},
    "distance_bonus": 0.50,
    "accuracy_penalty": 0.50,
    "bermuda_greens": True,
    "wind_sensitivity": 0.55,
    "notes": "Arnold Palmer Inv — tough, windy. App SG is king..."
}
```

### Model Parameters (`config/settings.py`)
Tune these as you collect data and validate performance:
```python
KELLY_FRACTION = 0.25         # Start conservative
PUTTING_REGRESSION_FACTOR = 0.55  # High — putting is volatile
APPROACH_REGRESSION_FACTOR = 0.20 # Low — approach is predictive
MIN_EDGE_THRESHOLD = 0.04     # 4% minimum edge to bet
```

---

## 📈 Audit Workflow

Run after every tournament:
```bash
python main.py audit
```

What it checks:
- ROI by bet type (where is edge coming from?)
- Model calibration (are our probabilities accurate?)
- Rank correlation (are we ranking players correctly?)
- Edge decay (is our edge shrinking over time?)
- Rolling ROI trends

---

## 🗺️ Roadmap

- [ ] Backtester against 3+ seasons of historical data
- [ ] Live in-tournament model (round-by-round SG updates)
- [ ] DP World Tour / LIV integration
- [ ] Ownership scraper (RotoGrinders)
- [ ] Streamlit dashboard
- [ ] DataGolf integration (activate when subscribed)
- [ ] Automated bet logging to sportsbooks
