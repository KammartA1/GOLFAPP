"""
Golf Quant Engine — Tournament Auto-Detection
Uses ESPN Golf API to detect current PGA Tour tournament.
"""
import requests
import json
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"

# Known PGA Tour tournament → course mapping
TOURNAMENT_COURSE_MAP = {
    "valspar championship": "Innisbrook Copperhead",
    "the players championship": "TPC Sawgrass",
    "the masters": "Augusta National",
    "arnold palmer invitational": "Bay Hill",
    "rbc heritage": "Harbour Town",
    "wells fargo championship": "Quail Hollow",
    "pga championship": "Valhalla",
    "charles schwab challenge": "Colonial Country Club",
    "the memorial tournament": "Muirfield Village",
    "u.s. open": "Pinehurst No. 2",
    "travelers championship": "TPC River Highlands",
    "john deere classic": "TPC Deere Run",
    "the open championship": "Royal Troon",
    "3m open": "TPC Twin Cities",
    "wyndham championship": "Sedgefield Country Club",
    "fedex st. jude championship": "TPC Southwind",
    "bmw championship": "Caves Valley",
    "tour championship": "East Lake",
    "waste management phoenix open": "TPC Scottsdale",
    "wm phoenix open": "TPC Scottsdale",
    "genesis invitational": "Riviera Country Club",
    "at&t pebble beach pro-am": "Pebble Beach",
    "the sentry": "Kapalua Plantation",
    "sony open in hawaii": "Waialae Country Club",
    "houston open": "Memorial Park",
    "texas children's houston open": "Memorial Park",
    "rocket mortgage classic": "Detroit Golf Club",
    "the cj cup byron nelson": "TPC Craig Ranch",
    "zurich classic of new orleans": "TPC Louisiana",
    "cognizant classic": "PGA National",
    "mexico open": "El Camaleón",
    "valero texas open": "TPC San Antonio",
    "corales puntacana championship": "Corales Golf Club",
}

# Course coordinates lookup for weather API
COURSE_COORDINATES = {
    "tpc sawgrass": (30.1975, -81.3959),
    "augusta national": (33.5021, -82.0232),
    "innisbrook copperhead": (28.0394, -82.7891),
    "tpc scottsdale": (33.6420, -111.9261),
    "pebble beach": (36.5688, -121.9484),
    "riviera country club": (34.0478, -118.5001),
    "bay hill": (28.4600, -81.5100),
    "harbour town": (32.1310, -80.8100),
    "tpc southwind": (35.0567, -89.7939),
    "east lake": (33.7411, -84.3236),
    "torrey pines south": (32.8950, -117.2522),
    "muirfield village": (40.1175, -83.1710),
    "quail hollow": (35.1092, -80.8575),
    "valhalla": (38.2534, -85.5203),
    "congressional": (39.0002, -77.1458),
    "pinehurst no. 2": (35.1936, -79.4698),
    "royal troon": (55.5382, -4.8242),
    "st andrews": (56.3439, -2.8026),
    "tpc craig ranch": (33.0830, -96.7170),
    "colonial country club": (32.7310, -97.3860),
    "memorial park": (29.7633, -95.4167),
    "detroit golf club": (42.4230, -83.1400),
    "sedgefield country club": (36.0569, -79.9614),
    "tpc twin cities": (44.8800, -93.2200),
    "caves valley": (39.4347, -76.7731),
    "liberty national": (40.6919, -74.0552),
    "wilmington country club": (39.7850, -75.6089),
    "the olympic club": (37.7086, -122.4938),
    "oak hill": (43.1375, -77.5239),
    "los angeles country club": (34.0648, -118.4358),
    "medinah": (41.9656, -88.0348),
    "bethpage black": (40.7500, -73.4500),
    "tpc river highlands": (41.6389, -72.6400),
    "tpc deere run": (41.4111, -90.4267),
    "trinity forest": (32.6969, -96.7306),
    "tpc summerlin": (36.1531, -115.2900),
    "shadow creek": (36.1278, -115.1111),
    "kapalua plantation": (20.9950, -156.6497),
    "waialae country club": (21.2717, -157.7711),
    "el camaleón": (20.6719, -87.0147),
    "tpc san antonio": (29.5992, -98.6169),
    "congaree": (32.5789, -80.8144),
}


def detect_current_tournament() -> dict:
    """
    Auto-detect the current/most recent PGA Tour tournament from ESPN API.

    Returns dict with:
        name, course_name, course_lat, course_lon, par,
        start_date, end_date, status, current_round, espn_id,
        field (list of player dicts), raw_json
    """
    try:
        resp = requests.get(ESPN_SCOREBOARD, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"ESPN API error: {e}")
        return {"error": str(e)}

    events = data.get("events", [])
    if not events:
        return {"error": "No PGA events found on ESPN", "raw": json.dumps(data)[:500]}

    # Use the first (current/most recent) event
    event = events[0]
    competition = event.get("competitions", [{}])[0] if event.get("competitions") else {}

    name = event.get("name", "Unknown Tournament")
    short_name = event.get("shortName", name)
    espn_id = str(event.get("id", ""))

    # Course info — ESPN doesn't always include venue.
    # Use tournament-to-course lookup for known events.
    course_name = ""
    venue = competition.get("venue", {})
    if venue:
        course_name = venue.get("fullName", venue.get("shortName", ""))
        if not course_name:
            addr = venue.get("address", {})
            course_name = addr.get("summary", "")
    if event.get("courses"):
        for c in event["courses"]:
            if c.get("name"):
                course_name = c["name"]
                break
    # Fallback: known tournament → course mapping
    if not course_name:
        course_name = TOURNAMENT_COURSE_MAP.get(name.lower(), "")

    # Dates
    start_date = event.get("date", "")
    end_date = event.get("endDate", "")

    # Status — prefer competition.status (has round info) over event.status
    comp_status = competition.get("status", {})
    status_obj = comp_status if comp_status else event.get("status", {})
    status_type = status_obj.get("type", {}).get("name", "STATUS_SCHEDULED")
    status_map = {
        "STATUS_SCHEDULED": "upcoming",
        "STATUS_IN_PROGRESS": "in_progress",
        "STATUS_FINAL": "completed",
        "STATUS_DELAYED": "delayed",
        "STATUS_SUSPENDED": "suspended",
    }
    status = status_map.get(status_type, "upcoming")

    # Current round
    current_round = 0
    period = status_obj.get("period", 0)
    if period:
        current_round = int(period)

    # Par
    par = 72
    if event.get("courses"):
        for c in event["courses"]:
            if c.get("par"):
                par = int(c["par"])
                if not course_name:
                    course_name = c.get("name", "")
                break

    # Course coordinates
    course_lat, course_lon = _lookup_coordinates(course_name)

    # Field / competitors
    field = []
    competitors = competition.get("competitors", [])
    for comp in competitors:
        athlete = comp.get("athlete", {})
        player = {
            "name": athlete.get("displayName", comp.get("displayName", "Unknown")),
            "espn_id": str(athlete.get("id", "")),
            "position": comp.get("status", {}).get("position", {}).get("displayName", ""),
            "score": comp.get("score", ""),
            "status": comp.get("status", {}).get("type", {}).get("name", ""),
        }
        field.append(player)

    result = {
        "name": name,
        "short_name": short_name,
        "course_name": course_name,
        "course_lat": course_lat,
        "course_lon": course_lon,
        "par": par,
        "start_date": start_date,
        "end_date": end_date,
        "status": status,
        "current_round": current_round,
        "espn_id": espn_id,
        "field": field,
        "field_count": len(field),
        "raw_json": json.dumps(data)[:10000],
    }

    log.info(f"Detected tournament: {name} at {course_name} (status={status}, R{current_round})")
    return result


def _lookup_coordinates(course_name: str) -> tuple:
    """Lookup course lat/lon from our database."""
    if not course_name:
        return 0.0, 0.0
    cn = course_name.lower().strip()
    # Exact match
    if cn in COURSE_COORDINATES:
        return COURSE_COORDINATES[cn]
    # Partial match
    for key, coords in COURSE_COORDINATES.items():
        if key in cn or cn in key:
            return coords
        # Try matching just the main name
        key_parts = key.split()
        cn_parts = cn.split()
        if len(set(key_parts) & set(cn_parts)) >= 2:
            return coords
    log.warning(f"No coordinates found for course: {course_name}")
    return 0.0, 0.0


def get_live_leaderboard(espn_id: str = None) -> list[dict]:
    """Fetch live leaderboard from ESPN API."""
    try:
        resp = requests.get(ESPN_SCOREBOARD, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"ESPN leaderboard error: {e}")
        return []

    events = data.get("events", [])
    if not events:
        return []

    event = events[0]
    competition = event.get("competitions", [{}])[0] if event.get("competitions") else {}
    competitors = competition.get("competitors", [])

    leaderboard = []
    for comp in competitors:
        athlete = comp.get("athlete", {})
        stats = comp.get("linescores", [])

        rounds = {}
        for i, ls in enumerate(stats):
            rounds[f"round{i+1}"] = ls.get("value", None)

        entry = {
            "name": athlete.get("displayName", "Unknown"),
            "position": comp.get("status", {}).get("position", {}).get("displayName", ""),
            "total_score": _parse_score(comp.get("score", "")),
            "today_score": _parse_score(comp.get("linescores", [{}])[-1].get("displayValue", "E") if comp.get("linescores") else "E"),
            "thru": comp.get("status", {}).get("thru", ""),
            **rounds,
            "status": comp.get("status", {}).get("type", {}).get("name", ""),
        }
        leaderboard.append(entry)

    # Sort by total score (lowest first for stroke play)
    leaderboard.sort(key=lambda x: x.get("total_score", 999))
    return leaderboard


def _parse_score(score_str) -> int:
    """Parse score string like '+3', '-2', 'E' to integer."""
    if not score_str:
        return 0
    s = str(score_str).strip()
    if s in ("E", "e", "Even"):
        return 0
    try:
        return int(s.replace("+", ""))
    except (ValueError, TypeError):
        return 0
