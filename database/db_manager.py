"""
Golf Quant Engine — Database Manager
SQLite-backed storage for all scraped data.
"""
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "golf_engine.db"


def _ensure_dir():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_conn():
    """Thread-safe connection context manager."""
    _ensure_dir()
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist."""
    _ensure_dir()
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)
    log.info(f"Database initialized at {DB_PATH}")


# ─────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tournaments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    espn_id TEXT UNIQUE,
    name TEXT NOT NULL,
    course_name TEXT,
    course_lat REAL,
    course_lon REAL,
    par INTEGER,
    start_date TEXT,
    end_date TEXT,
    status TEXT DEFAULT 'upcoming',
    current_round INTEGER DEFAULT 0,
    detected_at TEXT DEFAULT (datetime('now')),
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS tournament_field (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_id INTEGER REFERENCES tournaments(id),
    player_name TEXT NOT NULL,
    player_name_normalized TEXT,
    espn_player_id TEXT,
    world_rank INTEGER,
    status TEXT DEFAULT 'active',
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS prizepicks_lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_name TEXT,
    player_name TEXT NOT NULL,
    player_name_normalized TEXT,
    stat_type TEXT NOT NULL,
    line_value REAL NOT NULL,
    start_time TEXT,
    is_active INTEGER DEFAULT 1,
    pp_projection_id TEXT,
    league_id TEXT,
    scraped_at TEXT DEFAULT (datetime('now')),
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS odds_api_lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_name TEXT,
    player_name TEXT NOT NULL,
    player_name_normalized TEXT,
    market_type TEXT NOT NULL,
    bookmaker TEXT,
    odds_american INTEGER,
    odds_decimal REAL,
    implied_prob REAL,
    no_vig_prob REAL,
    scraped_at TEXT DEFAULT (datetime('now')),
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS odds_consensus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_name TEXT,
    player_name TEXT NOT NULL,
    player_name_normalized TEXT,
    market_type TEXT NOT NULL,
    consensus_prob REAL,
    num_books INTEGER,
    min_prob REAL,
    max_prob REAL,
    calculated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS weather_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_name TEXT,
    course_name TEXT,
    lat REAL,
    lon REAL,
    forecast_time TEXT,
    temp_f REAL,
    wind_speed_mph REAL,
    wind_gust_mph REAL,
    wind_direction_deg INTEGER,
    humidity_pct INTEGER,
    precipitation_mm REAL,
    cloud_cover_pct INTEGER,
    description TEXT,
    fetched_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS live_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_id INTEGER REFERENCES tournaments(id),
    player_name TEXT NOT NULL,
    player_name_normalized TEXT,
    position TEXT,
    total_score INTEGER,
    today_score INTEGER,
    thru TEXT,
    round1 INTEGER,
    round2 INTEGER,
    round3 INTEGER,
    round4 INTEGER,
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS projections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_name TEXT,
    player_name TEXT NOT NULL,
    stat_type TEXT NOT NULL,
    projection REAL,
    std_dev REAL,
    weather_adj_projection REAL,
    weather_adj_std REAL,
    prob_over REAL,
    prob_under REAL,
    edge_pct REAL,
    confidence TEXT,
    calculated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS bet_tracker (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_name TEXT,
    player_name TEXT,
    stat_type TEXT,
    pick_side TEXT,
    line_value REAL,
    projection REAL,
    edge_pct REAL,
    confidence TEXT,
    bet_amount REAL,
    odds_decimal REAL,
    result TEXT DEFAULT 'pending',
    profit_loss REAL DEFAULT 0,
    placed_at TEXT DEFAULT (datetime('now')),
    settled_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_pp_lines_active ON prizepicks_lines(is_active, stat_type);
CREATE INDEX IF NOT EXISTS idx_pp_lines_player ON prizepicks_lines(player_name_normalized);
CREATE INDEX IF NOT EXISTS idx_odds_player ON odds_api_lines(player_name_normalized, market_type);
CREATE INDEX IF NOT EXISTS idx_weather_time ON weather_data(forecast_time);
CREATE INDEX IF NOT EXISTS idx_scores_tournament ON live_scores(tournament_id);
CREATE INDEX IF NOT EXISTS idx_projections_player ON projections(player_name, stat_type);
"""


# ─────────────────────────────────────────────
# PRIZEPICKS OPERATIONS
# ─────────────────────────────────────────────
def deactivate_old_pp_lines():
    """Mark all existing PP lines as inactive before a fresh scrape."""
    with get_conn() as conn:
        conn.execute("UPDATE prizepicks_lines SET is_active = 0 WHERE is_active = 1")


def insert_pp_line(player_name: str, stat_type: str, line_value: float,
                   tournament_name: str = "", start_time: str = "",
                   pp_projection_id: str = "", league_id: str = "",
                   player_norm: str = "", raw_json: str = ""):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO prizepicks_lines
               (tournament_name, player_name, player_name_normalized, stat_type,
                line_value, start_time, is_active, pp_projection_id, league_id, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
            (tournament_name, player_name, player_norm or normalize_name(player_name),
             stat_type, line_value, start_time, pp_projection_id, league_id, raw_json)
        )


def get_active_pp_lines() -> list[dict]:
    """Return all currently active PrizePicks lines."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM prizepicks_lines WHERE is_active = 1 ORDER BY stat_type, player_name"
        ).fetchall()
        return [dict(r) for r in rows]


def get_pp_last_scraped() -> str:
    """Return the timestamp of the most recent PP scrape."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(scraped_at) as last FROM prizepicks_lines"
        ).fetchone()
        return row["last"] if row and row["last"] else "Never"


# ─────────────────────────────────────────────
# ODDS API OPERATIONS
# ─────────────────────────────────────────────
def insert_odds_line(player_name: str, market_type: str, bookmaker: str,
                     odds_american: int, odds_decimal: float,
                     implied_prob: float, no_vig_prob: float = None,
                     tournament_name: str = "", raw_json: str = ""):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO odds_api_lines
               (tournament_name, player_name, player_name_normalized, market_type,
                bookmaker, odds_american, odds_decimal, implied_prob, no_vig_prob, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (tournament_name, player_name, normalize_name(player_name),
             market_type, bookmaker, odds_american, odds_decimal,
             implied_prob, no_vig_prob, raw_json)
        )


def upsert_odds_consensus(player_name: str, market_type: str,
                          consensus_prob: float, num_books: int,
                          min_prob: float, max_prob: float,
                          tournament_name: str = ""):
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO odds_consensus
               (tournament_name, player_name, player_name_normalized, market_type,
                consensus_prob, num_books, min_prob, max_prob)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (tournament_name, player_name, normalize_name(player_name),
             market_type, consensus_prob, num_books, min_prob, max_prob)
        )


def get_odds_consensus(market_type: str = "outrights") -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM odds_consensus WHERE market_type = ? ORDER BY consensus_prob DESC",
            (market_type,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_odds_last_scraped() -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(scraped_at) as last FROM odds_api_lines"
        ).fetchone()
        return row["last"] if row and row["last"] else "Never"


# ─────────────────────────────────────────────
# WEATHER OPERATIONS
# ─────────────────────────────────────────────
def insert_weather(tournament_name: str, course_name: str,
                   lat: float, lon: float, forecast_time: str,
                   temp_f: float, wind_speed: float, wind_gust: float,
                   wind_dir: int, humidity: int, precip_mm: float,
                   cloud_cover: int, description: str):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO weather_data
               (tournament_name, course_name, lat, lon, forecast_time,
                temp_f, wind_speed_mph, wind_gust_mph, wind_direction_deg,
                humidity_pct, precipitation_mm, cloud_cover_pct, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (tournament_name, course_name, lat, lon, forecast_time,
             temp_f, wind_speed, wind_gust, wind_dir, humidity,
             precip_mm, cloud_cover, description)
        )


def get_current_weather(tournament_name: str = "") -> dict:
    """Get most recent weather reading."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT * FROM weather_data
               WHERE tournament_name = ? OR ? = ''
               ORDER BY fetched_at DESC LIMIT 1""",
            (tournament_name, tournament_name)
        ).fetchone()
        return dict(row) if row else {}


def get_weather_forecast(tournament_name: str = "") -> list[dict]:
    """Get forecast data."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM weather_data
               WHERE (tournament_name = ? OR ? = '')
                 AND forecast_time >= datetime('now')
               ORDER BY forecast_time""",
            (tournament_name, tournament_name)
        ).fetchall()
        return [dict(r) for r in rows]


def get_weather_last_fetched() -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(fetched_at) as last FROM weather_data"
        ).fetchone()
        return row["last"] if row and row["last"] else "Never"


# ─────────────────────────────────────────────
# TOURNAMENT OPERATIONS
# ─────────────────────────────────────────────
def upsert_tournament(espn_id: str, name: str, course_name: str = "",
                      course_lat: float = 0, course_lon: float = 0,
                      par: int = 72, start_date: str = "", end_date: str = "",
                      status: str = "upcoming", current_round: int = 0,
                      raw_json: str = "") -> int:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO tournaments
               (espn_id, name, course_name, course_lat, course_lon, par,
                start_date, end_date, status, current_round, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(espn_id) DO UPDATE SET
                name=excluded.name, course_name=excluded.course_name,
                course_lat=excluded.course_lat, course_lon=excluded.course_lon,
                par=excluded.par, status=excluded.status,
                current_round=excluded.current_round, raw_json=excluded.raw_json""",
            (espn_id, name, course_name, course_lat, course_lon, par,
             start_date, end_date, status, current_round, raw_json)
        )
        row = conn.execute("SELECT id FROM tournaments WHERE espn_id = ?", (espn_id,)).fetchone()
        return row["id"] if row else 0


def get_current_tournament() -> dict:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM tournaments ORDER BY detected_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else {}


# ─────────────────────────────────────────────
# LIVE SCORES
# ─────────────────────────────────────────────
def upsert_live_score(tournament_id: int, player_name: str, position: str = "",
                      total_score: int = 0, today_score: int = 0, thru: str = "",
                      round1: int = None, round2: int = None,
                      round3: int = None, round4: int = None):
    norm = normalize_name(player_name)
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO live_scores
               (tournament_id, player_name, player_name_normalized, position,
                total_score, today_score, thru, round1, round2, round3, round4)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT DO NOTHING""",
            (tournament_id, player_name, norm, position,
             total_score, today_score, thru, round1, round2, round3, round4)
        )


def get_leaderboard(tournament_id: int = None) -> list[dict]:
    with get_conn() as conn:
        if tournament_id:
            rows = conn.execute(
                "SELECT * FROM live_scores WHERE tournament_id = ? ORDER BY total_score",
                (tournament_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM live_scores ORDER BY updated_at DESC, total_score LIMIT 100"
            ).fetchall()
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# BET TRACKER
# ─────────────────────────────────────────────
def insert_bet(tournament_name: str, player_name: str, stat_type: str,
               pick_side: str, line_value: float, projection: float,
               edge_pct: float, confidence: str, bet_amount: float,
               odds_decimal: float = 1.87):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO bet_tracker
               (tournament_name, player_name, stat_type, pick_side, line_value,
                projection, edge_pct, confidence, bet_amount, odds_decimal)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (tournament_name, player_name, stat_type, pick_side, line_value,
             projection, edge_pct, confidence, bet_amount, odds_decimal)
        )


def settle_bet(bet_id: int, result: str, profit_loss: float):
    with get_conn() as conn:
        conn.execute(
            """UPDATE bet_tracker SET result = ?, profit_loss = ?,
               settled_at = datetime('now') WHERE id = ?""",
            (result, profit_loss, bet_id)
        )


def get_bets(status: str = None) -> list[dict]:
    with get_conn() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM bet_tracker WHERE result = ? ORDER BY placed_at DESC",
                (status,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM bet_tracker ORDER BY placed_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
def normalize_name(name: str) -> str:
    """Normalize player names for consistent matching across sources."""
    if not name:
        return ""
    import re
    n = name.strip()
    # Remove suffixes
    n = re.sub(r'\s+(Jr\.|Sr\.|III|IV|II)\s*$', '', n, flags=re.IGNORECASE)
    # Lowercase, collapse whitespace
    n = re.sub(r'\s+', ' ', n).strip().lower()
    return n
