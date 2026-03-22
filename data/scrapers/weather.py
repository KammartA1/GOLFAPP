"""
Weather Model — v8.0
One of the most underpriced edges in golf betting.

v8.0 Research-backed improvements:
  1. Wet-bulb temperature (better predictor than air temp alone)
  2. Air density formula (elevation + temp + humidity + pressure)
  3. Wind is #1 factor — enhanced wind model
  4. Morning/afternoon wave splits (only reliable correlation strategy)
  5. Weather explains 44% of scoring variance (Int'l Journal of Biometeorology)
  6. Humid air is LIGHTER than dry air (counter-intuitive: ball goes farther)
"""
import math
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

# Course elevations in feet (for air density calculation)
COURSE_ELEVATIONS = {
    "Augusta National": 330,
    "TPC Sawgrass": 20,
    "TPC Scottsdale": 1500,
    "Pebble Beach": 50,
    "Pinehurst No. 2": 480,
    "Torrey Pines South": 300,
    "Bay Hill": 80,
    "Muirfield Village": 900,
    "Riviera CC": 200,
    "Harbour Town": 10,
    "Innisbrook (Copperhead)": 50,
    "East Lake GC": 1000,
    "St Andrews Old Course": 5,
    "Royal Troon": 10,
}


# ─────────────────────────────────────────────
# ATMOSPHERIC PHYSICS
# ─────────────────────────────────────────────

def compute_wet_bulb_temperature(temp_f: float, humidity_pct: float) -> float:
    """
    Compute wet-bulb temperature (°F).
    Research: wet-bulb temperature is a better predictor of golf scoring
    than air temperature alone because it incorporates humidity's effect
    on player fatigue and ball behavior.

    Uses Stull (2011) approximation.
    """
    temp_c = (temp_f - 32) * 5 / 9
    rh = humidity_pct

    # Stull approximation for wet-bulb temperature
    tw_c = temp_c * math.atan(0.151977 * math.sqrt(rh + 8.313659)) + \
           math.atan(temp_c + rh) - math.atan(rh - 1.676331) + \
           0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh) - 4.686035

    return tw_c * 9 / 5 + 32  # Convert back to °F


def compute_air_density(
    temp_f: float,
    humidity_pct: float,
    pressure_hpa: float = 1013.25,
    elevation_ft: float = 0,
) -> float:
    """
    Compute air density (kg/m³) incorporating all atmospheric factors.

    Research: Lower air density → less drag → ball flies farther.
    Factors that REDUCE density (more distance):
      - Higher temperature
      - Higher humidity (counter-intuitive: humid air is LIGHTER)
      - Higher altitude
      - Lower barometric pressure

    Returns: air_density in kg/m³ (sea level standard = 1.225)
    """
    temp_k = (temp_f - 32) * 5 / 9 + 273.15

    # Adjust pressure for elevation using barometric formula
    if elevation_ft > 0:
        elevation_m = elevation_ft * 0.3048
        # Standard atmospheric lapse rate
        pressure_hpa = pressure_hpa * (1 - 0.0000225577 * elevation_m) ** 5.25588

    # Saturation vapor pressure (Buck equation)
    temp_c = temp_k - 273.15
    e_sat = 6.1121 * math.exp((18.678 - temp_c / 234.5) * (temp_c / (257.14 + temp_c)))

    # Actual vapor pressure from humidity
    e = e_sat * humidity_pct / 100

    # Air density from ideal gas law with moisture correction
    # Rd = 287.05 (dry air gas constant), Rv = 461.495 (water vapor gas constant)
    p_pa = pressure_hpa * 100
    e_pa = e * 100

    density = (p_pa - e_pa) / (287.05 * temp_k) + e_pa / (461.495 * temp_k)
    return density


def compute_distance_factor(
    temp_f: float,
    humidity_pct: float,
    elevation_ft: float = 0,
    pressure_hpa: float = 1013.25,
) -> float:
    """
    Compute ball flight distance factor relative to standard conditions.

    Returns: multiplier (1.0 = standard, >1.0 = more distance, <1.0 = less)
    """
    standard_density = 1.225  # kg/m³ at sea level, 15°C, dry
    actual_density = compute_air_density(temp_f, humidity_pct, pressure_hpa, elevation_ft)

    # Distance is inversely proportional to air density (approximately)
    # ~1% more distance per 1% reduction in air density
    return standard_density / actual_density


def compute_wind_scoring_impact(
    wind_speed_mph: float,
    wind_gust_mph: float = 0,
) -> dict:
    """
    Compute wind's impact on scoring.

    Research: Wind is the #1 weather factor.
    - Add ~1% distance per 1 mph of headwind penalty
    - Subtract ~0.5% per 1 mph of tailwind benefit
    - Scoring increases ~0.015 strokes per 1 mph of wind above 10 mph
    """
    effective_wind = max(wind_speed_mph, wind_gust_mph * 0.7)

    # Scoring impact (strokes above baseline)
    if effective_wind > 25:
        scoring_penalty = 0.015 * (effective_wind - 10) + 0.5  # Severe
    elif effective_wind > 20:
        scoring_penalty = 0.015 * (effective_wind - 10) + 0.2  # Heavy
    elif effective_wind > 15:
        scoring_penalty = 0.015 * (effective_wind - 10)         # Moderate
    elif effective_wind > 10:
        scoring_penalty = 0.008 * (effective_wind - 10)         # Light
    else:
        scoring_penalty = 0.0                                    # Calm

    # Variance impact (scoring becomes more unpredictable)
    if effective_wind > 20:
        variance_mult = 1.0 + 0.02 * (effective_wind - 15)
    elif effective_wind > 15:
        variance_mult = 1.0 + 0.015 * (effective_wind - 15)
    elif effective_wind > 10:
        variance_mult = 1.05
    else:
        variance_mult = 1.0

    return {
        "scoring_penalty": round(scoring_penalty, 3),
        "variance_mult": round(variance_mult, 3),
        "effective_wind": round(effective_wind, 1),
    }


class WeatherModel:
    """
    Enhanced weather model — v8.0
    Research: Weather explains 44% of scoring variance.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.api_key = WEATHER_API_KEY

    def _has_api_key(self) -> bool:
        return bool(self.api_key)

    def fetch_forecast(self, course_name: str, lat: float = None, lon: float = None) -> list[dict]:
        """Fetch 5-day / 3-hour forecast for a course location."""
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
            "cnt": 40
        }

        try:
            resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error(f"Weather fetch failed: {e}")
            return []

        elevation = COURSE_ELEVATIONS.get(course_name, 0)
        forecasts = []

        for period in data.get("list", []):
            temp_f = period["main"]["temp"]
            humidity = period["main"]["humidity"]
            wind_speed = period["wind"]["speed"] * 2.237  # m/s to mph
            wind_gust = period["wind"].get("gust", 0) * 2.237
            pressure = period["main"].get("pressure", 1013.25)

            # v8.0: Compute advanced atmospheric metrics
            wet_bulb = compute_wet_bulb_temperature(temp_f, humidity)
            air_density = compute_air_density(temp_f, humidity, pressure, elevation)
            distance_factor = compute_distance_factor(temp_f, humidity, elevation, pressure)
            wind_impact = compute_wind_scoring_impact(wind_speed, wind_gust)

            forecasts.append({
                "datetime": datetime.fromtimestamp(period["dt"]),
                "temp_f": temp_f,
                "humidity_pct": humidity,
                "pressure_hpa": pressure,
                # Wind
                "wind_speed_mph": round(wind_speed, 1),
                "wind_dir_deg": period["wind"].get("deg", 0),
                "wind_gust_mph": round(wind_gust, 1),
                # Precipitation
                "rain_mm": period.get("rain", {}).get("3h", 0),
                "conditions": period["weather"][0]["main"],
                "visibility_m": period.get("visibility", 10000),
                # v8.0: Advanced metrics
                "wet_bulb_f": round(wet_bulb, 1),
                "air_density": round(air_density, 4),
                "distance_factor": round(distance_factor, 4),
                "wind_scoring_penalty": wind_impact["scoring_penalty"],
                "wind_variance_mult": wind_impact["variance_mult"],
                "elevation_ft": elevation,
            })

        return sorted(forecasts, key=lambda x: x["datetime"])

    # ─────────────────────────────────────────────
    # WAVE SPLIT MODEL (v8.0 ENHANCED)
    # Research: Morning/afternoon wave split is the ONLY
    # reliable correlation strategy in golf.
    # ─────────────────────────────────────────────

    def get_round_windows(self, forecasts: list[dict], round_date: datetime) -> dict:
        """
        Get morning (AM draw) vs afternoon (PM draw) weather windows.
        v8.0: Uses advanced atmospheric metrics for scoring prediction.
        """
        am_window = [f for f in forecasts
                     if f["datetime"].date() == round_date.date()
                     and 6 <= f["datetime"].hour <= 12]
        pm_window = [f for f in forecasts
                     if f["datetime"].date() == round_date.date()
                     and 12 < f["datetime"].hour <= 18]

        def summarize(window: list) -> dict:
            if not window:
                return {
                    "avg_wind": 0, "max_wind": 0, "rain_prob": False,
                    "scoring_penalty": 0, "variance_mult": 1.0,
                    "avg_wet_bulb": 70, "distance_factor": 1.0,
                    "condition_score": 0,
                }

            avg_wind = sum(w["wind_speed_mph"] for w in window) / len(window)
            max_wind = max(w["wind_speed_mph"] for w in window)
            rain = any(w["rain_mm"] > 0.5 for w in window)
            avg_wet_bulb = sum(w.get("wet_bulb_f", 70) for w in window) / len(window)
            avg_distance = sum(w.get("distance_factor", 1.0) for w in window) / len(window)

            # Composite scoring penalty from all weather factors
            wind_penalty = sum(w.get("wind_scoring_penalty", 0) for w in window) / len(window)
            rain_penalty = 0.3 if rain else 0.0
            # Extreme heat penalty (wet-bulb > 82°F = dangerous)
            heat_penalty = max(0, (avg_wet_bulb - 80) * 0.05) if avg_wet_bulb > 80 else 0

            total_penalty = wind_penalty + rain_penalty + heat_penalty
            total_variance = sum(w.get("wind_variance_mult", 1.0) for w in window) / len(window)
            if rain:
                total_variance *= 1.10

            # Lower condition_score = better conditions
            condition_score = total_penalty * 10 + (5 if rain else 0)

            return {
                "avg_wind_mph": round(avg_wind, 1),
                "max_wind_mph": round(max_wind, 1),
                "rain": rain,
                "scoring_penalty": round(total_penalty, 3),
                "variance_mult": round(total_variance, 3),
                "avg_wet_bulb_f": round(avg_wet_bulb, 1),
                "distance_factor": round(avg_distance, 4),
                "heat_penalty": round(heat_penalty, 3),
                "condition_score": round(condition_score, 2),
                "periods": len(window),
            }

        am_summary = summarize(am_window)
        pm_summary = summarize(pm_window)

        # Positive = AM is better, Negative = PM is better
        draw_advantage = pm_summary["condition_score"] - am_summary["condition_score"]

        # Scoring differential estimate (in strokes)
        scoring_diff = pm_summary["scoring_penalty"] - am_summary["scoring_penalty"]

        return {
            "am": am_summary,
            "pm": pm_summary,
            "am_advantage_score": round(draw_advantage, 2),
            "scoring_diff_strokes": round(scoring_diff, 3),
            "significant": abs(draw_advantage) > 3 or abs(scoring_diff) > 0.3,
        }

    def score_player_tee_times(self, player_tee_times: dict, course_name: str,
                                tournament_dates: list[datetime]) -> dict[str, float]:
        """
        Weather adjustment score per player based on tee time draw.
        v8.0: Uses advanced atmospheric model for more accurate scoring prediction.
        """
        if not player_tee_times:
            return {}

        forecasts = self.fetch_forecast(course_name)
        if not forecasts:
            return {p: 0.0 for p in player_tee_times}

        player_adj = {}
        for player, draws in player_tee_times.items():
            total_adj = 0.0
            rounds_counted = 0

            for i, draw in enumerate(draws[:2]):
                if i >= len(tournament_dates):
                    break
                windows = self.get_round_windows(forecasts, tournament_dates[i])
                if windows["significant"]:
                    if draw == "AM":
                        total_adj += windows["scoring_diff_strokes"]
                    else:
                        total_adj -= windows["scoring_diff_strokes"]
                    rounds_counted += 1

            player_adj[player] = round(total_adj / max(rounds_counted, 1), 3)

        return player_adj

    def get_comprehensive_weather(self, course_name: str, date: datetime) -> dict:
        """
        Get comprehensive weather analysis for a tournament day.
        v8.0: Includes all advanced metrics.
        """
        forecasts = self.fetch_forecast(course_name)
        day_fcst = [f for f in forecasts if f["datetime"].date() == date.date()]

        if not day_fcst:
            return {"available": False, "description": "No weather data available"}

        avg_wind = sum(f["wind_speed_mph"] for f in day_fcst) / len(day_fcst)
        max_wind = max(f["wind_speed_mph"] for f in day_fcst)
        avg_temp = sum(f["temp_f"] for f in day_fcst) / len(day_fcst)
        avg_wet_bulb = sum(f.get("wet_bulb_f", 70) for f in day_fcst) / len(day_fcst)
        avg_density = sum(f.get("air_density", 1.225) for f in day_fcst) / len(day_fcst)
        avg_distance = sum(f.get("distance_factor", 1.0) for f in day_fcst) / len(day_fcst)
        rain = any(f["rain_mm"] > 0 for f in day_fcst)
        total_rain = sum(f["rain_mm"] for f in day_fcst)
        avg_scoring_penalty = sum(f.get("wind_scoring_penalty", 0) for f in day_fcst) / len(day_fcst)
        avg_variance = sum(f.get("wind_variance_mult", 1.0) for f in day_fcst) / len(day_fcst)

        # Overall scoring impact
        weather_scoring_adj = avg_scoring_penalty + (0.3 if rain else 0)

        severity = ("calm" if avg_wind < 10
                    else "moderate wind" if avg_wind < 18
                    else "strong wind" if avg_wind < 25
                    else "extreme wind")
        rain_str = f" with rain ({total_rain:.1f}mm)" if rain else ""

        return {
            "available": True,
            "temp_f": round(avg_temp, 1),
            "wet_bulb_f": round(avg_wet_bulb, 1),
            "avg_wind_mph": round(avg_wind, 1),
            "max_wind_mph": round(max_wind, 1),
            "air_density": round(avg_density, 4),
            "distance_factor": round(avg_distance, 4),
            "rain": rain,
            "total_rain_mm": round(total_rain, 1),
            "scoring_adj_strokes": round(weather_scoring_adj, 3),
            "variance_mult": round(avg_variance, 3),
            "description": f"{day_fcst[0]['conditions']}{rain_str}, avg {avg_wind:.0f} mph {severity}",
            "elevation_ft": day_fcst[0].get("elevation_ft", 0),
        }

    def get_conditions_summary(self, course_name: str, date: datetime) -> str:
        """Human-readable summary of conditions for a given date."""
        weather = self.get_comprehensive_weather(course_name, date)
        if not weather.get("available"):
            return "No weather data available"
        return weather["description"]
