"""Weather Advantage Source — Wind skill, rain performance differential.

Some players perform significantly better in adverse weather.
This source captures:
  - Wind skill: performance differential in high-wind rounds
  - Rain performance: scoring relative to field in wet conditions
  - Temperature effects: cold/hot weather adaptability
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from edge_analysis.edge_sources import EdgeSource

logger = logging.getLogger(__name__)


class WeatherAdvantageSource(EdgeSource):
    """Identify players with weather-related performance advantages."""

    name = "weather_advantage"
    category = "informational"
    description = "Wind skill, rain performance differential, temperature adaptability"

    def get_signal(self, player: str, tournament_context: Dict[str, Any]) -> float:
        """Signal based on player's weather skill vs forecast conditions.

        Positive = player has advantage in forecasted conditions.
        """
        weather = tournament_context.get("weather", {})
        player_weather_stats = tournament_context.get("player_weather_stats", {})

        wind_speed = weather.get("wind_speed", 10)  # mph
        precip_chance = weather.get("precip_chance", 0.1)
        temp = weather.get("temperature", 75)  # Fahrenheit

        # Player's weather-specific performance
        wind_sg_diff = player_weather_stats.get("wind_sg_differential", 0.0)
        rain_sg_diff = player_weather_stats.get("rain_sg_differential", 0.0)
        cold_sg_diff = player_weather_stats.get("cold_sg_differential", 0.0)

        signal = 0.0

        # Wind advantage: significant above 15 mph
        if wind_speed > 15:
            wind_factor = min((wind_speed - 15) / 10.0, 1.0)  # 0-1 scale
            signal += wind_sg_diff * wind_factor * 1.5
        elif wind_speed > 10:
            wind_factor = (wind_speed - 10) / 10.0
            signal += wind_sg_diff * wind_factor * 0.5

        # Rain advantage
        if precip_chance > 0.4:
            rain_factor = min((precip_chance - 0.4) / 0.4, 1.0)
            signal += rain_sg_diff * rain_factor * 1.2

        # Temperature: some players struggle in cold/hot
        if temp < 55:
            cold_factor = min((55 - temp) / 20.0, 1.0)
            signal += cold_sg_diff * cold_factor * 0.8
        elif temp > 95:
            heat_factor = min((temp - 95) / 15.0, 1.0)
            # Most players suffer in extreme heat; any advantage is valuable
            heat_diff = player_weather_stats.get("heat_sg_differential", 0.0)
            signal += heat_diff * heat_factor * 0.8

        # Calm conditions: weather advantage sources provide no edge
        if wind_speed < 8 and precip_chance < 0.15 and 60 < temp < 90:
            signal *= 0.2  # Heavily discount in benign weather

        return round(float(np.clip(signal, -2.5, 2.5)), 4)

    def get_mechanism(self) -> str:
        return (
            "Markets price players based on average performance but underweight "
            "weather-specific skill differences. A player who gains +0.5 SG in wind "
            "vs calm is systematically undervalued when wind forecasts show 20mph. "
            "Weather forecasts are public but their player-specific impact is not "
            "immediately reflected in lines, creating a timing advantage."
        )

    def get_decay_risk(self) -> str:
        return (
            "medium — Weather forecasts are publicly available, but the player-specific "
            "impact modeling requires historical round-level weather data that most "
            "bettors do not have. Risk increases as weather-adjusted models become "
            "more common in commercial products."
        )

    def validate(self, historical_data: List[Dict[str, Any]]) -> dict:
        if len(historical_data) < 20:
            return {"is_valid": False, "reason": "insufficient data", "n_samples": len(historical_data)}

        signals = []
        outcomes = []
        for rec in historical_data:
            sig = self.get_signal(rec.get("player", ""), rec)
            signals.append(sig)
            outcomes.append(rec.get("actual_finish_pct", 0.5))

        sig_arr = np.array(signals)
        out_arr = np.array(outcomes)

        # Only validate on weather rounds (signal != 0)
        mask = np.abs(sig_arr) > 0.1
        if mask.sum() < 10:
            return {"is_valid": False, "reason": "insufficient weather rounds", "n_samples": int(mask.sum())}

        sig_weather = sig_arr[mask]
        out_weather = out_arr[mask]
        corr = float(np.corrcoef(sig_weather, out_weather)[0, 1]) if len(sig_weather) > 2 else 0.0
        returns = sig_weather * (out_weather - 0.5)
        sharpe = float(np.mean(returns) / max(np.std(returns), 0.001))
        hit_rate = float(np.mean((sig_weather > 0) == (out_weather > 0.5)))

        from scipy import stats
        _, p_val = stats.pearsonr(sig_weather, out_weather) if len(sig_weather) >= 10 else (0, 1.0)

        return {
            "is_valid": corr > 0.05,
            "sharpe": round(sharpe, 4),
            "hit_rate": round(hit_rate, 4),
            "correlation": round(corr, 4),
            "n_samples": int(mask.sum()),
            "p_value": round(float(p_val), 4),
        }
