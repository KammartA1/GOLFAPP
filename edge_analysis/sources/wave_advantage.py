"""Wave Advantage Source — AM/PM tee time advantage.

In golf tournaments, players are split into AM and PM waves for rounds 1-2.
When weather conditions differ between waves (e.g., wind picks up in afternoon),
one wave has a significant scoring advantage. This is often 1-2 strokes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from edge_analysis.edge_sources import EdgeSource

logger = logging.getLogger(__name__)


class WaveAdvantageSource(EdgeSource):
    """AM/PM wave tee time advantage based on weather differential."""

    name = "wave_advantage"
    category = "structural"
    description = "AM/PM wave scoring advantage from weather differential"

    def get_signal(self, player: str, tournament_context: Dict[str, Any]) -> float:
        """Signal based on player's wave assignment and expected conditions.

        Positive = player is in the advantaged wave.
        """
        wave = tournament_context.get("player_wave", "unknown")  # "AM" or "PM"
        round_num = tournament_context.get("round_num", 1)

        # Weather by wave
        am_weather = tournament_context.get("am_weather", {})
        pm_weather = tournament_context.get("pm_weather", {})

        if wave == "unknown" or not am_weather or not pm_weather:
            return 0.0

        # Calculate scoring difficulty for each wave
        am_difficulty = self._score_difficulty(am_weather)
        pm_difficulty = self._score_difficulty(pm_weather)

        # Difficulty differential (positive = PM harder)
        diff = pm_difficulty - am_difficulty

        # For rounds 1-2, players alternate waves
        # Round 1 AM -> Round 2 PM (and vice versa)
        if round_num == 1:
            player_difficulty = am_difficulty if wave == "AM" else pm_difficulty
            other_difficulty = pm_difficulty if wave == "AM" else am_difficulty
        elif round_num == 2:
            # Wave flips in round 2
            player_difficulty = pm_difficulty if wave == "AM" else am_difficulty
            other_difficulty = am_difficulty if wave == "AM" else pm_difficulty
        else:
            return 0.0  # No wave advantage in rounds 3-4

        # Signal = advantage from being in easier wave
        # Positive = player's wave is easier
        advantage = other_difficulty - player_difficulty

        # Convert to SG-scale signal
        # 1 point of difficulty ~ 0.5 SG advantage
        signal = advantage * 0.5

        # Discount for round 2 (waves flip, partially cancels)
        if round_num == 2:
            signal *= 0.4  # Round 2 partially offsets round 1

        # For 36-hole total, the net wave advantage is R1 + R2
        # Since waves flip, the net depends on which day has worse weather
        net_r1_r2 = tournament_context.get("net_wave_advantage_sg", None)
        if net_r1_r2 is not None:
            signal = net_r1_r2 * (1.0 if wave == "AM" else -1.0)

        return round(float(np.clip(signal, -2.0, 2.0)), 4)

    def _score_difficulty(self, weather: dict) -> float:
        """Convert weather conditions to a scoring difficulty score.

        0 = benign, 3+ = extremely difficult.
        """
        wind = weather.get("wind_speed", 10)
        precip = weather.get("precip_chance", 0.0)
        temp = weather.get("temperature", 75)

        difficulty = 0.0

        # Wind is the primary difficulty driver
        if wind > 20:
            difficulty += (wind - 10) * 0.12
        elif wind > 12:
            difficulty += (wind - 10) * 0.08

        # Rain increases difficulty
        if precip > 0.5:
            difficulty += precip * 1.5
        elif precip > 0.2:
            difficulty += precip * 0.8

        # Extreme temperatures
        if temp < 50:
            difficulty += (50 - temp) * 0.03
        elif temp > 95:
            difficulty += (temp - 90) * 0.02

        return round(difficulty, 2)

    def get_mechanism(self) -> str:
        return (
            "Wave advantage is a structural factor that markets consistently underprice. "
            "When afternoon wind picks up 10mph, the PM wave can score 1-2 strokes worse "
            "as a group. Outright and top-finish odds are set based on overall field "
            "strength, not wave-adjusted projections. This creates systematic mispricing "
            "of AM-wave players in windy forecasts."
        )

    def get_decay_risk(self) -> str:
        return (
            "low — Wave advantage is inherently structural and unpriceable by most "
            "market makers. It requires combining tee time sheets with hourly weather "
            "forecasts, which most books do not do. This edge source is durable."
        )

    def validate(self, historical_data: List[Dict[str, Any]]) -> dict:
        if len(historical_data) < 30:
            return {"is_valid": False, "reason": "insufficient data", "n_samples": len(historical_data)}

        am_scores = []
        pm_scores = []
        for rec in historical_data:
            wave = rec.get("player_wave", "unknown")
            score = rec.get("round_score_vs_field", 0.0)
            if wave == "AM":
                am_scores.append(score)
            elif wave == "PM":
                pm_scores.append(score)

        if len(am_scores) < 10 or len(pm_scores) < 10:
            return {"is_valid": False, "reason": "insufficient wave data", "n_samples": len(am_scores) + len(pm_scores)}

        am_arr = np.array(am_scores)
        pm_arr = np.array(pm_scores)

        from scipy import stats
        t_stat, p_val = stats.ttest_ind(am_arr, pm_arr)
        diff = float(np.mean(am_arr) - np.mean(pm_arr))

        signals = []
        outcomes = []
        for rec in historical_data:
            sig = self.get_signal(rec.get("player", ""), rec)
            signals.append(sig)
            outcomes.append(rec.get("actual_finish_pct", 0.5))

        sig_arr = np.array(signals)
        out_arr = np.array(outcomes)
        mask = np.abs(sig_arr) > 0.05
        if mask.sum() >= 10:
            returns = sig_arr[mask] * (out_arr[mask] - 0.5)
            sharpe = float(np.mean(returns) / max(np.std(returns), 0.001))
        else:
            sharpe = 0.0

        return {
            "is_valid": abs(diff) > 0.1 or p_val < 0.10,
            "sharpe": round(sharpe, 4),
            "hit_rate": 0.0,
            "am_mean": round(float(np.mean(am_arr)), 4),
            "pm_mean": round(float(np.mean(pm_arr)), 4),
            "wave_differential": round(diff, 4),
            "n_samples": len(am_scores) + len(pm_scores),
            "p_value": round(float(p_val), 4),
        }
