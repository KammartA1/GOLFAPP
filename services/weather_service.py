"""
Weather Service — Golf-specific weather data layer
====================================================
Wraps weather fetching, storage, and adjustment factor computation.
All weather data is persisted to the database.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func

from database.models import Event
from services._db import get_session as _session

log = logging.getLogger(__name__)

# Course coordinate lookup (duplicated from data/scrapers/weather.py
# to avoid import-time side effects from that module)
_COURSE_COORDS = {
    "Augusta National":          (33.5021, -82.0205),
    "TPC Sawgrass":              (30.1975, -81.3964),
    "TPC Scottsdale":            (33.6609, -111.8906),
    "Pebble Beach":              (36.5681, -121.9469),
    "Pinehurst No. 2":           (35.1964, -79.4697),
    "Torrey Pines South":        (32.8997, -117.2522),
    "Bay Hill":                  (28.4636, -81.4956),
    "Muirfield Village":         (40.0867, -83.0563),
    "Riviera CC":                (34.0547, -118.4859),
    "Harbour Town":              (32.1367, -80.8064),
    "Innisbrook (Copperhead)":   (28.1556, -82.7295),
    "East Lake GC":              (33.7281, -84.3083),
    "St Andrews Old Course":     (56.3408, -2.8024),
    "Royal Troon":               (55.5372, -4.6789),
}

# We store weather in a lightweight table derived from LineMovement-style
# approach, but weather has its own schema in db_manager (weather_data table).
# This service uses SQLAlchemy models where possible and falls back to the
# raw db_manager weather_data table via raw SQL when needed.

# Import the atmospheric physics helpers
try:
    from data.scrapers.weather import (
        compute_wet_bulb_temperature,
        compute_air_density,
        compute_distance_factor,
        compute_wind_scoring_impact,
        WeatherModel,
    )
    _WEATHER_MODEL_AVAILABLE = True
except ImportError:
    _WEATHER_MODEL_AVAILABLE = False
    log.warning("Weather model not importable — weather service will use DB cache only")


def get_current_weather(course_name: str) -> dict:
    """
    Return the most recent weather observation for a course.

    First checks DB cache (weather_data table via raw SQL on the
    golf_engine.db or golf_quant.db). If stale (> 2 hours), attempts
    a live fetch via WeatherModel.
    """
    cached = _get_cached_weather(course_name, hours=2)
    if cached:
        return cached

    # Try live fetch
    if _WEATHER_MODEL_AVAILABLE:
        try:
            model = WeatherModel()
            coords = _COURSE_COORDS.get(course_name)
            if coords:
                forecasts = model.fetch_forecast(course_name, lat=coords[0], lon=coords[1])
                if forecasts:
                    # Store all forecast points
                    for f in forecasts:
                        store_weather(course_name, f)
                    # Return the nearest-time entry as "current"
                    now = datetime.utcnow()
                    nearest = min(forecasts, key=lambda f: abs((f.get("datetime", now) - now).total_seconds()))
                    return _normalize_weather(nearest, course_name)
        except Exception as exc:
            log.error("Live weather fetch failed for %s: %s", course_name, exc)

    return {"error": "No weather data available", "course": course_name}


def get_weather_forecast(
    course_name: str,
    hours: int = 48,
) -> list[dict]:
    """
    Return forecast data points for a course within the next N hours.

    Pulls from DB cache first. If empty or stale, attempts live fetch.
    """
    now = datetime.utcnow()
    future = now + timedelta(hours=hours)

    # Try DB cache
    cached_list = _get_cached_forecast(course_name, hours)
    if cached_list:
        return cached_list

    # Live fetch
    if _WEATHER_MODEL_AVAILABLE:
        try:
            model = WeatherModel()
            forecasts = model.fetch_forecast(course_name)
            if forecasts:
                for f in forecasts:
                    store_weather(course_name, f)
                # Filter to requested window
                result = []
                for f in forecasts:
                    dt = f.get("datetime", now)
                    if now <= dt <= future:
                        result.append(_normalize_weather(f, course_name))
                return result
        except Exception as exc:
            log.error("Forecast fetch failed for %s: %s", course_name, exc)

    return []


def get_weather_adjustment_factor(weather_data: dict) -> float:
    """
    Compute a single scoring adjustment factor from weather data.

    Returns a float representing expected strokes added to scoring
    due to weather conditions. Positive = harder conditions.

    Uses wind speed, precipitation, and temperature to compute the
    adjustment.
    """
    wind_mph = float(weather_data.get("wind_speed_mph", 0))
    precip_mm = float(weather_data.get("precipitation_mm", weather_data.get("rain_mm", 0)))
    temp_f = float(weather_data.get("temp_f", 72))
    gust_mph = float(weather_data.get("wind_gust_mph", 0))

    adjustment = 0.0

    # Wind impact (primary factor — explains most weather variance)
    effective_wind = max(wind_mph, gust_mph * 0.7)
    if effective_wind > 25:
        adjustment += 0.015 * (effective_wind - 10) + 0.5
    elif effective_wind > 20:
        adjustment += 0.015 * (effective_wind - 10) + 0.2
    elif effective_wind > 15:
        adjustment += 0.015 * (effective_wind - 10)
    elif effective_wind > 10:
        adjustment += 0.008 * (effective_wind - 10)

    # Rain penalty
    if precip_mm > 5.0:
        adjustment += 0.5
    elif precip_mm > 2.0:
        adjustment += 0.3
    elif precip_mm > 0.5:
        adjustment += 0.15

    # Extreme temperature penalty
    if temp_f > 95:
        adjustment += 0.1 * ((temp_f - 95) / 5)
    elif temp_f < 45:
        adjustment += 0.1 * ((45 - temp_f) / 5)

    return round(adjustment, 3)


def store_weather(course_name: str, weather_data: dict) -> None:
    """
    Persist a weather data point to the database.

    Uses raw SQL insert into the weather_data table (from db_manager schema)
    since that table is not represented in the SQLAlchemy ORM models.
    """
    try:
        from database.connection import get_engine
        engine = get_engine()

        forecast_dt = weather_data.get("datetime")
        if isinstance(forecast_dt, datetime):
            forecast_time = forecast_dt.isoformat()
        else:
            forecast_time = str(forecast_dt) if forecast_dt else datetime.utcnow().isoformat()

        coords = _COURSE_COORDS.get(course_name, (0, 0))

        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO weather_data
                        (tournament_name, course_name, lat, lon, forecast_time,
                         temp_f, wind_speed_mph, wind_gust_mph, wind_direction_deg,
                         humidity_pct, precipitation_mm, cloud_cover_pct, description,
                         fetched_at)
                    VALUES
                        (:tournament, :course, :lat, :lon, :forecast_time,
                         :temp, :wind, :gust, :wind_dir,
                         :humidity, :precip, :clouds, :desc,
                         :fetched)
                """),
                {
                    "tournament": weather_data.get("tournament_name", ""),
                    "course": course_name,
                    "lat": coords[0],
                    "lon": coords[1],
                    "forecast_time": forecast_time,
                    "temp": weather_data.get("temp_f", 0),
                    "wind": weather_data.get("wind_speed_mph", 0),
                    "gust": weather_data.get("wind_gust_mph", 0),
                    "wind_dir": weather_data.get("wind_dir_deg", weather_data.get("wind_direction_deg", 0)),
                    "humidity": weather_data.get("humidity_pct", 0),
                    "precip": weather_data.get("rain_mm", weather_data.get("precipitation_mm", 0)),
                    "clouds": weather_data.get("cloud_cover_pct", 0),
                    "desc": weather_data.get("conditions", weather_data.get("description", "")),
                    "fetched": datetime.utcnow().isoformat(),
                },
            )
            conn.commit()
    except Exception as exc:
        log.error("Failed to store weather for %s: %s", course_name, exc)


# ── internal helpers ──────────────────────────────────────────────

def _get_cached_weather(course_name: str, hours: int = 2) -> Optional[dict]:
    """Query weather_data table for recent readings."""
    try:
        from database.connection import get_engine
        from sqlalchemy import text

        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        engine = get_engine()

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT * FROM weather_data
                    WHERE course_name = :course
                      AND fetched_at >= :cutoff
                    ORDER BY fetched_at DESC
                    LIMIT 1
                """),
                {"course": course_name, "cutoff": cutoff},
            )
            row = result.mappings().first()
            if row:
                return dict(row)
    except Exception as exc:
        log.debug("Weather cache query failed: %s", exc)

    return None


def _get_cached_forecast(course_name: str, hours: int = 48) -> list[dict]:
    """Query weather_data table for forecast entries."""
    try:
        from database.connection import get_engine
        from sqlalchemy import text

        now = datetime.utcnow().isoformat()
        future = (datetime.utcnow() + timedelta(hours=hours)).isoformat()
        engine = get_engine()

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT * FROM weather_data
                    WHERE course_name = :course
                      AND forecast_time >= :now
                      AND forecast_time <= :future
                    ORDER BY forecast_time ASC
                """),
                {"course": course_name, "now": now, "future": future},
            )
            rows = result.mappings().all()
            return [dict(r) for r in rows]
    except Exception as exc:
        log.debug("Forecast cache query failed: %s", exc)

    return []


def _normalize_weather(raw: dict, course_name: str) -> dict:
    """Normalize a weather dict from various sources into a consistent shape."""
    return {
        "course": course_name,
        "temp_f": raw.get("temp_f", 0),
        "humidity_pct": raw.get("humidity_pct", 0),
        "wind_speed_mph": raw.get("wind_speed_mph", 0),
        "wind_gust_mph": raw.get("wind_gust_mph", 0),
        "wind_direction_deg": raw.get("wind_dir_deg", raw.get("wind_direction_deg", 0)),
        "precipitation_mm": raw.get("rain_mm", raw.get("precipitation_mm", 0)),
        "cloud_cover_pct": raw.get("cloud_cover_pct", 0),
        "description": raw.get("conditions", raw.get("description", "")),
        "wet_bulb_f": raw.get("wet_bulb_f"),
        "air_density": raw.get("air_density"),
        "distance_factor": raw.get("distance_factor"),
        "wind_scoring_penalty": raw.get("wind_scoring_penalty"),
        "forecast_time": str(raw.get("datetime", raw.get("forecast_time", ""))),
        "scoring_adjustment": get_weather_adjustment_factor(raw),
    }
