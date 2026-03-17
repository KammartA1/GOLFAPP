"""
Weather Model
One of the most underpriced edges in golf betting.
Morning vs afternoon draw can be worth 1-2 strokes in bad weather.
"""
import logging
import requests
from datetime import datetime
from typing import Optional

from config.settings import WEATHER_API_KEY, WEATHER_API_BASE, REQUEST_TIMEOUT, USER_AGENT

log = logging.getLogger(__name__)

# Known coordinates for major golf venues
COURSE_COORDS = {
    "Augusta National":     (33.5021, -82.0205),
    "TPC Sawgrass":         (30.1975, -81.3964),
    "TPC Scottsdale":       (33.6609, -111.8906),
    "Pebble Beach":         (36.5681, -121.9469),
    "Pinehurst No. 2":      (35.1964, -79.4697),
    "Torrey Pines South":   (32.8997, -117.2522),
    "Bay Hill":             (28.4636, -81.4956),
    "Muirfield Village":    (40.0867, -83.0563),
    "Riviera CC":           (34.0547, -118.4859),
    "Harbour Town":         (32.1367, -80.8064),
    "Innisbrook (Copperhead)": (28.1556, -82.7295),
    "East Lake GC":         (33.7281, -84.3083),
    "St Andrews Old Course":(56.3408, -2.8024),
    "Royal Troon":          (55.5372, -4.6789),
}


class WeatherModel:
    """
    Fetches weather forecasts and generates tee-time advantage scores.

    Free tier: OpenWeatherMap (1000 calls/day, 5-day forecast)
    Get free key at: openweathermap.org/api
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.api_key = WEATHER_API_KEY

    def _has_api_key(self) -> bool:
        return bool(self.api_key)

    def fetch_forecast(self, course_name: str, lat: float = None, lon: float = None) -> list[dict]:
        """
        Fetch 5-day / 3-hour forecast for a course location.
        Returns list of forecast periods sorted by datetime.
        """
        if not self._has_api_key():
            log.warning("No OpenWeather API key set. Add OPENWEATHER_API_KEY to .env")
            return []

        if lat is None or lon is None:
            coords = COURSE_COORDS.get(course_name)
            if not coords:
                log.warning(f"No coordinates found for {course_name}")
                return []
            lat, lon = coords

        url = f"{WEATHER_API_BASE}/forecast"
        params = {
            "lat": lat, "lon": lon,
            "appid": self.api_key,
            "units": "imperial",
            "cnt": 40  # 5 days × 8 periods (3h intervals)
        }

        try:
            resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error(f"Weather fetch failed: {e}")
            return []

        forecasts = []
        for period in data.get("list", []):
            forecasts.append({
                "datetime": datetime.fromtimestamp(period["dt"]),
                "temp_f": period["main"]["temp"],
                "wind_speed_mph": period["wind"]["speed"] * 2.237,   # m/s to mph
                "wind_dir_deg": period["wind"].get("deg", 0),
                "wind_gust_mph": period["wind"].get("gust", 0) * 2.237,
                "conditions": period["weather"][0]["main"],           # Rain, Clear, Clouds, etc.
                "rain_mm": period.get("rain", {}).get("3h", 0),
                "humidity": period["main"]["humidity"],
                "visibility_m": period.get("visibility", 10000),
            })

        return sorted(forecasts, key=lambda x: x["datetime"])

    def get_round_windows(self, forecasts: list[dict], round_date: datetime) -> dict:
        """
        Get morning (AM draw) vs afternoon (PM draw) weather windows
        for a given round date.
        Returns comparative weather metrics for each window.
        """
        am_window = [f for f in forecasts
                     if f["datetime"].date() == round_date.date()
                     and 6 <= f["datetime"].hour <= 12]
        pm_window = [f for f in forecasts
                     if f["datetime"].date() == round_date.date()
                     and 12 < f["datetime"].hour <= 18]

        def summarize(window: list) -> dict:
            if not window:
                return {"avg_wind": 0, "max_wind": 0, "rain_prob": 0, "score": 0}
            avg_wind = sum(w["wind_speed_mph"] for w in window) / len(window)
            max_wind = max(w["wind_speed_mph"] for w in window)
            rain = any(w["rain_mm"] > 0.5 for w in window)
            # Lower score = better conditions
            condition_score = avg_wind * 0.6 + max_wind * 0.4 + (5 if rain else 0)
            return {
                "avg_wind_mph": round(avg_wind, 1),
                "max_wind_mph": round(max_wind, 1),
                "rain": rain,
                "condition_score": round(condition_score, 2),  # lower = better
                "periods": len(window),
            }

        am_summary = summarize(am_window)
        pm_summary = summarize(pm_window)

        # Positive = AM is better, Negative = PM is better
        draw_advantage = pm_summary["condition_score"] - am_summary["condition_score"]

        return {
            "am": am_summary,
            "pm": pm_summary,
            "am_advantage_score": round(draw_advantage, 2),
            "significant": abs(draw_advantage) > 5,  # > 5 points = significant edge
        }

    def score_player_tee_times(self, player_tee_times: dict, course_name: str,
                                tournament_dates: list[datetime]) -> dict[str, float]:
        """
        Given a dict of {player_name: tee_time (AM/PM per round)},
        return a weather adjustment score per player.

        Positive score = player has weather advantage
        Negative score = player has weather disadvantage
        """
        if not player_tee_times:
            return {}

        forecasts = self.fetch_forecast(course_name)
        if not forecasts:
            return {p: 0.0 for p in player_tee_times}

        player_adj = {}
        for player, draws in player_tee_times.items():
            # draws = list of "AM"/"PM" per round (r1, r2, etc.)
            total_adj = 0.0
            rounds_counted = 0

            for i, draw in enumerate(draws[:2]):  # Only R1/R2 matter for draw
                if i >= len(tournament_dates):
                    break
                windows = self.get_round_windows(forecasts, tournament_dates[i])
                if windows["significant"]:
                    if draw == "AM":
                        # AM player — positive if AM is better
                        total_adj += windows["am_advantage_score"]
                    else:
                        total_adj -= windows["am_advantage_score"]
                    rounds_counted += 1

            player_adj[player] = round(total_adj / max(rounds_counted, 1), 3)

        return player_adj

    def get_conditions_summary(self, course_name: str, date: datetime) -> str:
        """Human-readable summary of conditions for a given date."""
        forecasts = self.fetch_forecast(course_name)
        day_fcst = [f for f in forecasts if f["datetime"].date() == date.date()]

        if not day_fcst:
            return "No weather data available"

        avg_wind = sum(f["wind_speed_mph"] for f in day_fcst) / len(day_fcst)
        rain = any(f["rain_mm"] > 0 for f in day_fcst)
        conditions = day_fcst[0]["conditions"]

        severity = "calm" if avg_wind < 10 else "moderate wind" if avg_wind < 20 else "significant wind"
        rain_str = " with rain" if rain else ""

        return f"{conditions}{rain_str}, avg {avg_wind:.0f} mph {severity}"
