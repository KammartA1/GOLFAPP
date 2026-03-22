"""
Golf Quant Engine — OpenWeather Integration
Fetches live weather and 3-hour forecasts for tournament courses.
"""
import requests
import logging
from datetime import datetime

log = logging.getLogger(__name__)

WEATHER_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
WEATHER_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


class WeatherScraper:
    """Fetches weather data from OpenWeather API."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_current(self, lat: float, lon: float) -> dict:
        """Fetch current weather conditions."""
        if not self.api_key or (lat == 0 and lon == 0):
            return {"error": "No API key or coordinates"}

        try:
            resp = requests.get(WEATHER_CURRENT_URL, params={
                "lat": lat, "lon": lon,
                "appid": self.api_key,
                "units": "imperial",
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            wind = data.get("wind", {})
            main = data.get("main", {})
            weather = data.get("weather", [{}])[0]
            rain = data.get("rain", {})
            clouds = data.get("clouds", {})

            return {
                "temp_f": main.get("temp", 0),
                "feels_like_f": main.get("feels_like", 0),
                "humidity_pct": main.get("humidity", 0),
                "wind_speed_mph": wind.get("speed", 0),
                "wind_gust_mph": wind.get("gust", 0),
                "wind_direction_deg": wind.get("deg", 0),
                "precipitation_mm": rain.get("1h", 0),
                "cloud_cover_pct": clouds.get("all", 0),
                "description": weather.get("description", ""),
                "icon": weather.get("icon", ""),
                "forecast_time": datetime.utcnow().isoformat(),
                "error": None,
            }
        except Exception as e:
            log.error(f"Weather current error: {e}")
            return {"error": str(e)}

    def fetch_forecast(self, lat: float, lon: float) -> list[dict]:
        """Fetch 3-hour forecast (up to 5 days)."""
        if not self.api_key or (lat == 0 and lon == 0):
            return []

        try:
            resp = requests.get(WEATHER_FORECAST_URL, params={
                "lat": lat, "lon": lon,
                "appid": self.api_key,
                "units": "imperial",
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            forecasts = []
            for item in data.get("list", []):
                wind = item.get("wind", {})
                main = item.get("main", {})
                weather = item.get("weather", [{}])[0]
                rain = item.get("rain", {})
                clouds = item.get("clouds", {})

                forecasts.append({
                    "forecast_time": item.get("dt_txt", ""),
                    "temp_f": main.get("temp", 0),
                    "humidity_pct": main.get("humidity", 0),
                    "wind_speed_mph": wind.get("speed", 0),
                    "wind_gust_mph": wind.get("gust", 0),
                    "wind_direction_deg": wind.get("deg", 0),
                    "precipitation_mm": rain.get("3h", 0),
                    "cloud_cover_pct": clouds.get("all", 0),
                    "description": weather.get("description", ""),
                })

            return forecasts
        except Exception as e:
            log.error(f"Weather forecast error: {e}")
            return []

    def fetch_and_store(self, tournament_name: str, course_name: str,
                        lat: float, lon: float) -> dict:
        """Fetch current + forecast and store in database."""
        from database.db_manager import insert_weather

        current = self.fetch_current(lat, lon)
        forecast = self.fetch_forecast(lat, lon)

        # Store current
        if not current.get("error"):
            insert_weather(
                tournament_name=tournament_name,
                course_name=course_name,
                lat=lat, lon=lon,
                forecast_time=current.get("forecast_time", ""),
                temp_f=current.get("temp_f", 0),
                wind_speed=current.get("wind_speed_mph", 0),
                wind_gust=current.get("wind_gust_mph", 0),
                wind_dir=current.get("wind_direction_deg", 0),
                humidity=current.get("humidity_pct", 0),
                precip_mm=current.get("precipitation_mm", 0),
                cloud_cover=current.get("cloud_cover_pct", 0),
                description=current.get("description", ""),
            )

        # Store forecasts
        for fc in forecast:
            insert_weather(
                tournament_name=tournament_name,
                course_name=course_name,
                lat=lat, lon=lon,
                forecast_time=fc.get("forecast_time", ""),
                temp_f=fc.get("temp_f", 0),
                wind_speed=fc.get("wind_speed_mph", 0),
                wind_gust=fc.get("wind_gust_mph", 0),
                wind_dir=fc.get("wind_direction_deg", 0),
                humidity=fc.get("humidity_pct", 0),
                precip_mm=fc.get("precipitation_mm", 0),
                cloud_cover=fc.get("cloud_cover_pct", 0),
                description=fc.get("description", ""),
            )

        return {
            "current": current,
            "forecast_count": len(forecast),
            "forecast": forecast[:8],  # First 24 hours for display
        }


def wind_direction_str(deg: int) -> str:
    """Convert wind direction degrees to compass direction."""
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(deg / 22.5) % 16
    return dirs[idx]


def weather_impact_factor(wind_speed: float, precipitation: float = 0) -> dict:
    """
    Calculate weather impact on projections.

    Returns dict with variance_mult, projection_mult, description.
    """
    variance_mult = 1.0
    projection_mult = 1.0
    notes = []

    if wind_speed > 20:
        variance_mult = 1.0 + 0.015 * (wind_speed - 15)
        projection_mult = 0.97
        notes.append(f"Heavy wind ({wind_speed:.0f}mph): +{(variance_mult-1)*100:.0f}% variance, scoring harder")
    elif wind_speed > 15:
        variance_mult = 1.0 + 0.015 * (wind_speed - 15)
        notes.append(f"Moderate wind ({wind_speed:.0f}mph): +{(variance_mult-1)*100:.0f}% variance")
    elif wind_speed > 10:
        variance_mult = 1.05
        notes.append(f"Light wind ({wind_speed:.0f}mph): +5% variance")

    if precipitation > 2:
        variance_mult *= 1.10
        notes.append(f"Rain ({precipitation:.1f}mm): +10% variance, softer greens")
    elif precipitation > 0:
        notes.append(f"Light rain ({precipitation:.1f}mm): minimal impact")

    return {
        "variance_mult": round(variance_mult, 3),
        "projection_mult": round(projection_mult, 3),
        "description": " | ".join(notes) if notes else "Calm conditions",
        "is_significant": wind_speed > 15 or precipitation > 2,
    }
