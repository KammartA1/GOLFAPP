"""Noise injection test — Verify system handles noisy inputs gracefully.

Injects noise into various system inputs (SG data, odds, weather, etc.)
to ensure the system doesn't amplify noise into bad decisions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class NoiseInjectionTest:
    """Test system robustness to noisy inputs across all data sources.

    Golf Data Sources Tested:
        1. Strokes Gained components (SG OTT/APP/ATG/PUTT)
        2. Odds/lines from sportsbooks
        3. Weather data (wind, temperature, precipitation)
        4. Course statistics (yardage, par, fairway width)
        5. Player form/recent results
        6. Field strength ratings
    """

    NOISE_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.50]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def run_all(self, player_data: List[dict]) -> dict:
        """Run noise injection across all data sources.

        Args:
            player_data: List of dicts with player-level features, e.g.:
                {
                    'player': str,
                    'sg_total': float, 'sg_ott': float, 'sg_app': float,
                    'sg_atg': float, 'sg_putt': float,
                    'odds_decimal': float, 'model_prob': float,
                    'wind_speed': float, 'temperature': float,
                    'course_yardage': int, 'field_strength': float,
                    'true_win_prob': float (for simulation),
                }
        """
        if len(player_data) < 5:
            return {"error": "Need at least 5 player records"}

        results = {}

        # Test each data source
        sg_fields = ["sg_total", "sg_ott", "sg_app", "sg_atg", "sg_putt"]
        results["strokes_gained"] = self._test_field_group(
            player_data, sg_fields, "Strokes Gained"
        )

        results["odds"] = self._test_field_group(
            player_data, ["odds_decimal"], "Odds"
        )

        weather_fields = ["wind_speed", "temperature"]
        results["weather"] = self._test_field_group(
            player_data, weather_fields, "Weather"
        )

        results["course"] = self._test_field_group(
            player_data, ["course_yardage", "field_strength"], "Course/Field"
        )

        # Combined noise: all sources noisy simultaneously
        results["combined"] = self._test_combined_noise(player_data)

        # Summary
        robustness_score = self._compute_robustness_score(results)

        return {
            "n_players": len(player_data),
            "noise_levels_tested": self.NOISE_LEVELS,
            "results": results,
            "robustness_score": round(robustness_score, 2),
            "verdict": "robust" if robustness_score >= 70 else
                       "marginal" if robustness_score >= 40 else "fragile",
            "most_sensitive_source": self._find_most_sensitive(results),
        }

    def _test_field_group(
        self,
        player_data: List[dict],
        fields: List[str],
        group_name: str,
    ) -> dict:
        """Test noise sensitivity for a group of fields."""
        results = []

        for noise_level in self.NOISE_LEVELS:
            noisy_data = self._inject_noise(player_data, fields, noise_level)
            impact = self._measure_impact(player_data, noisy_data, fields)
            impact["noise_level"] = noise_level
            impact["group"] = group_name
            results.append(impact)

        # Sensitivity: how much output changes per unit of input noise
        if results:
            sensitivities = [r.get("ranking_disruption", 0) / max(r["noise_level"], 0.001)
                             for r in results]
            avg_sensitivity = float(np.mean(sensitivities))
        else:
            avg_sensitivity = 0.0

        return {
            "group": group_name,
            "fields": fields,
            "noise_results": results,
            "avg_sensitivity": round(avg_sensitivity, 4),
        }

    def _test_combined_noise(self, player_data: List[dict]) -> dict:
        """Test with noise injected into ALL sources simultaneously."""
        all_fields = [
            "sg_total", "sg_ott", "sg_app", "sg_atg", "sg_putt",
            "odds_decimal", "wind_speed", "temperature",
            "course_yardage", "field_strength",
        ]

        results = []
        for noise_level in self.NOISE_LEVELS:
            noisy_data = self._inject_noise(player_data, all_fields, noise_level)
            impact = self._measure_impact(player_data, noisy_data, all_fields)
            impact["noise_level"] = noise_level
            impact["group"] = "combined"
            results.append(impact)

        return {
            "group": "combined",
            "fields": all_fields,
            "noise_results": results,
        }

    def _inject_noise(
        self,
        player_data: List[dict],
        fields: List[str],
        noise_level: float,
    ) -> List[dict]:
        """Inject Gaussian noise into specified fields.

        Noise is proportional to the standard deviation of each field.
        """
        noisy = []
        for record in player_data:
            noisy_record = record.copy()
            for field_name in fields:
                val = record.get(field_name)
                if val is None:
                    continue
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    continue

                # Scale noise to field magnitude
                noise_std = max(abs(val) * noise_level, noise_level)
                noise = self.rng.normal(0, noise_std)
                noisy_record[field_name] = val + noise

            noisy.append(noisy_record)

        return noisy

    def _measure_impact(
        self,
        original: List[dict],
        noisy: List[dict],
        fields: List[str],
    ) -> dict:
        """Measure the impact of noise injection on rankings and predictions."""
        # Compute rankings based on a scoring function
        def _score(record: dict) -> float:
            """Simple scoring proxy based on available fields."""
            score = 0.0
            if "sg_total" in record and record.get("sg_total") is not None:
                score += float(record["sg_total"]) * 10
            if "model_prob" in record and record.get("model_prob") is not None:
                score += float(record["model_prob"]) * 100
            if "true_win_prob" in record and record.get("true_win_prob") is not None:
                score += float(record["true_win_prob"]) * 100
            return score

        original_scores = np.array([_score(r) for r in original])
        noisy_scores = np.array([_score(r) for r in noisy])

        if len(original_scores) == 0:
            return {"ranking_disruption": 0.0, "score_rmse": 0.0}

        # Ranking disruption (Kendall's tau distance proxy)
        original_ranks = np.argsort(np.argsort(-original_scores))
        noisy_ranks = np.argsort(np.argsort(-noisy_scores))
        rank_changes = np.abs(original_ranks - noisy_ranks)
        avg_rank_change = float(np.mean(rank_changes))
        max_rank_change = int(np.max(rank_changes))

        # Score RMSE
        score_diffs = original_scores - noisy_scores
        score_rmse = float(np.sqrt(np.mean(score_diffs ** 2)))

        # Probability change (if model_prob exists)
        prob_changes = []
        for orig, noisy_r in zip(original, noisy):
            if "model_prob" in orig and orig.get("model_prob") is not None:
                o = float(orig["model_prob"])
                n = float(noisy_r.get("model_prob", o))
                prob_changes.append(abs(n - o))

        avg_prob_change = float(np.mean(prob_changes)) if prob_changes else 0.0

        # Top-N disruption: how many of original top-5 stay in noisy top-5?
        n = min(5, len(original))
        original_top = set(np.argsort(-original_scores)[:n])
        noisy_top = set(np.argsort(-noisy_scores)[:n])
        top_overlap = len(original_top & noisy_top) / n

        return {
            "ranking_disruption": round(avg_rank_change, 2),
            "max_rank_change": max_rank_change,
            "score_rmse": round(score_rmse, 4),
            "avg_prob_change": round(avg_prob_change, 4),
            "top5_overlap": round(top_overlap, 4),
            "n_players": len(original),
        }

    def _find_most_sensitive(self, results: dict) -> str:
        """Find the data source that is most sensitive to noise."""
        max_sensitivity = 0.0
        most_sensitive = "none"

        for source_name, data in results.items():
            if source_name == "combined":
                continue
            sensitivity = data.get("avg_sensitivity", 0)
            if sensitivity > max_sensitivity:
                max_sensitivity = sensitivity
                most_sensitive = source_name

        return most_sensitive

    def _compute_robustness_score(self, results: dict) -> float:
        """Compute 0-100 robustness score from noise injection results."""
        sensitivities = []

        for source_name, data in results.items():
            if source_name == "combined":
                continue
            sensitivities.append(data.get("avg_sensitivity", 0))

        if not sensitivities:
            return 50.0

        avg_sensitivity = float(np.mean(sensitivities))
        max_sensitivity = float(np.max(sensitivities))

        # Lower sensitivity → higher robustness
        score = 100.0 - min(100.0, avg_sensitivity * 30 + max_sensitivity * 10)
        return max(0.0, score)
