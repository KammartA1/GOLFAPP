"""
Weather Advantage Edge Source
==============================
Player-specific weather skill ratings: wind, rain, temperature.
Market edge: sportsbooks adjust lines generically for weather; we have
per-player historical performance splits under different conditions.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import stats as sp_stats

log = logging.getLogger(__name__)

# Beaufort-like wind buckets (mph)
_WIND_CALM = 8.0
_WIND_MODERATE = 15.0
_WIND_STRONG = 22.0


class WeatherAdvantageSource:
    """Player-specific weather-skill edge."""

    name = "Weather Advantage"
    category = "conditions"
    version = "1.0"

    def get_signal(self, player: dict, tournament_context: dict) -> float:
        """
        Compute weather edge for a player given forecast conditions.

        player keys:
            wind_sg       – SG differential in high-wind (>15 mph) rounds
            rain_sg       – SG differential in rain rounds
            cold_sg       – SG differential in cold (<55F) rounds
            hot_sg        – SG differential in hot (>90F) rounds
            sg_total      – baseline total SG
        tournament_context keys:
            wind_mph      – forecast avg wind speed
            rain_prob     – probability of rain (0-1)
            temperature_f – forecast temperature (Fahrenheit)
        """
        wind_mph = tournament_context.get("wind_mph", 10.0)
        rain_prob = tournament_context.get("rain_prob", 0.0)
        temp_f = tournament_context.get("temperature_f", 72.0)

        # ── Wind component ──────────────────────────────────────────────
        wind_sg = player.get("wind_sg", 0.0)
        if wind_mph >= _WIND_STRONG:
            wind_edge = wind_sg * 1.0
        elif wind_mph >= _WIND_MODERATE:
            frac = (wind_mph - _WIND_MODERATE) / (_WIND_STRONG - _WIND_MODERATE)
            wind_edge = wind_sg * (0.5 + 0.5 * frac)
        elif wind_mph >= _WIND_CALM:
            frac = (wind_mph - _WIND_CALM) / (_WIND_MODERATE - _WIND_CALM)
            wind_edge = wind_sg * (0.1 + 0.4 * frac)
        else:
            wind_edge = 0.0  # calm, no wind advantage

        # ── Rain component ──────────────────────────────────────────────
        rain_sg = player.get("rain_sg", 0.0)
        # Scale by rain probability and severity
        rain_edge = rain_sg * rain_prob

        # ── Temperature component ───────────────────────────────────────
        cold_sg = player.get("cold_sg", 0.0)
        hot_sg = player.get("hot_sg", 0.0)
        if temp_f < 50:
            temp_edge = cold_sg * min((55 - temp_f) / 20.0, 1.0)
        elif temp_f > 90:
            temp_edge = hot_sg * min((temp_f - 88) / 15.0, 1.0)
        else:
            temp_edge = 0.0

        # ── Ball flight physics: cold = less carry, thin air ────────────
        # Standard: ball loses ~2 yards per 10F drop below 70F
        # This affects distance-dependent players more
        sg_ott = player.get("sg_ott", 0.0)
        distance_temp_adj = 0.0
        if temp_f < 60 and sg_ott > 0.3:
            # Long hitters lose proportionally less (technique > carry)
            distance_temp_adj = -0.02 * (60 - temp_f) / 10.0
        elif temp_f > 95 and sg_ott > 0.3:
            # Hot = ball flies further, long hitters benefit
            distance_temp_adj = 0.01 * (temp_f - 90) / 10.0

        # ── Humidity spin factor ────────────────────────────────────────
        humidity = tournament_context.get("humidity_pct", 50.0)
        # High humidity = less spin = approach advantage for low-spin players
        humidity_adj = 0.0
        if humidity > 80:
            spin_control = player.get("low_spin_skill", 0.0)
            humidity_adj = spin_control * 0.05 * (humidity - 80) / 20.0

        total_edge = wind_edge + rain_edge + temp_edge + distance_temp_adj + humidity_adj

        return round(float(total_edge), 4)

    def get_mechanism(self) -> str:
        return (
            "Each player has a unique weather skill profile built from historical "
            "round-level SG splits in high-wind (>15 mph), rain, cold (<55F), and "
            "hot (>90F) conditions.  We also model ball-flight physics: cold air "
            "reduces carry distance, high humidity reduces spin.  Sportsbooks "
            "adjust lines generically ('windy = harder') but do not capture "
            "player-specific weather skill, which can be worth 0.5-1.5 strokes "
            "per round in extreme conditions."
        )

    def get_decay_risk(self) -> str:
        return (
            "MEDIUM — Player weather skills are relatively stable (technique-based) "
            "but can shift with equipment changes or ageing.  Re-estimate weather "
            "splits every 6 months.  Risk: if DraftKings starts publishing "
            "player-weather data publicly, the edge compresses."
        )

    def validate(self, historical_data: list[dict]) -> dict:
        if len(historical_data) < 30:
            return {
                "sharpe": 0.0, "p_value": 1.0,
                "sample_size": len(historical_data),
                "correlation_with_other_signals": {},
                "status": "INSUFFICIENT_DATA",
            }

        signals, outcomes = [], []
        for rec in historical_data:
            sig = self.get_signal(rec.get("player", {}), rec.get("tournament_context", {}))
            finish = rec.get("actual_finish", 50)
            outcomes.append((50.0 - finish) / 50.0)
            signals.append(sig)

        signals = np.array(signals)
        outcomes = np.array(outcomes)
        n = len(signals)
        q = max(n // 5, 1)
        idx = np.argsort(signals)
        spread = float(np.mean(outcomes[idx[-q:]]) - np.mean(outcomes[idx[:q]]))
        pooled = float(np.std(np.concatenate([outcomes[idx[-q:]], outcomes[idx[:q]]]), ddof=1))
        sharpe = spread / pooled if pooled > 1e-9 else 0.0
        corr, p_val = sp_stats.spearmanr(signals, outcomes)
        if np.isnan(corr):
            corr, p_val = 0.0, 1.0

        return {
            "sharpe": round(sharpe, 4),
            "p_value": round(float(p_val), 6),
            "sample_size": n,
            "spearman_r": round(float(corr), 4),
            "quintile_spread": round(spread, 4),
            "correlation_with_other_signals": {},
            "status": "VALID" if p_val < 0.10 and n >= 30 else "MARGINAL",
        }
