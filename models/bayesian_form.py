"""
Bayesian Player Form Index — v9.0 (R6)

Instead of recency-weighted windows, maintains a proper Bayesian posterior
for each player's current true skill level. After each tournament, updates:

    posterior_sg = (prior_sg * prior_prec + observed_sg * likelihood_prec) / (prior_prec + likelihood_prec)

This naturally handles:
  - Sample size (small samples = wide posterior, stays near prior)
  - Regression to mean (prior pulls toward Tour average)
  - Form detection (multiple strong results shift posterior quickly)
"""
import numpy as np
from typing import Optional


# Tour average prior
TOUR_PRIOR_MEAN = 0.0
TOUR_PRIOR_STD = 1.5      # Prior std for unknown player
OBSERVATION_STD = 2.75     # Per-tournament scoring noise


class BayesianFormIndex:
    """Maintains Bayesian posterior for player's current true SG."""

    def __init__(self, prior_mean: float = TOUR_PRIOR_MEAN, prior_std: float = TOUR_PRIOR_STD):
        self.prior_mean = prior_mean
        self.prior_std = prior_std

    def compute_posterior(
        self,
        player_history: list[dict],
        decay_rate: float = 0.95,
    ) -> dict:
        """Compute Bayesian posterior from player history.

        Each observation is weighted by recency (exponential decay).
        More recent observations have higher effective precision.

        Args:
            player_history: list of dicts sorted by date (oldest first),
                           each with 'sg_total' and optionally 'event_date'
            decay_rate: per-event recency decay (0.95 = 5% decay per event)

        Returns:
            dict with posterior_mean, posterior_std, confidence, form_signal
        """
        sg_values = []
        for h in player_history:
            sg = h.get("sg_total")
            if sg is not None:
                sg_values.append(float(sg))

        if not sg_values:
            return {
                "posterior_mean": self.prior_mean,
                "posterior_std": self.prior_std,
                "confidence": 0.0,
                "form_signal": "no_data",
                "n_events": 0,
            }

        n = len(sg_values)

        # Apply exponential recency decay to observations
        weights = np.array([decay_rate ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()

        # Weighted observation mean
        obs_mean = float(np.dot(sg_values, weights))

        # Effective sample size (accounts for weighting)
        eff_n = 1.0 / float(np.sum(weights ** 2))

        # Bayesian update
        prior_precision = 1.0 / (self.prior_std ** 2)
        obs_precision = eff_n / (OBSERVATION_STD ** 2)

        posterior_precision = prior_precision + obs_precision
        posterior_mean = (
            self.prior_mean * prior_precision + obs_mean * obs_precision
        ) / posterior_precision
        posterior_std = np.sqrt(1.0 / posterior_precision)

        # Confidence: 0-1 scale based on posterior precision
        confidence = min(1.0, obs_precision / (prior_precision + obs_precision))

        # Form signal: is the player trending up or down?
        form_signal = self._detect_form(sg_values, posterior_mean)

        return {
            "posterior_mean": round(float(posterior_mean), 3),
            "posterior_std": round(float(posterior_std), 3),
            "confidence": round(float(confidence), 3),
            "form_signal": form_signal,
            "n_events": n,
            "effective_n": round(float(eff_n), 1),
            "weighted_obs_mean": round(obs_mean, 3),
        }

    def _detect_form(self, sg_values: list[float], posterior_mean: float) -> str:
        """Detect form trend relative to posterior."""
        if len(sg_values) < 6:
            return "insufficient"

        recent_3 = np.mean(sg_values[-3:])
        older_3 = np.mean(sg_values[-6:-3])

        diff = recent_3 - older_3

        if diff > 0.5:
            return "hot_streak"
        elif diff > 0.2:
            return "improving"
        elif diff < -0.5:
            return "cold_streak"
        elif diff < -0.2:
            return "declining"
        return "stable"

    def project_next_tournament(
        self,
        posterior_mean: float,
        posterior_std: float,
        course_fit_adj: float = 0.0,
        weather_adj: float = 0.0,
    ) -> dict:
        """Project SG for next tournament from the posterior.

        Returns prediction interval for the next observation.
        """
        # Predictive distribution = posterior convolved with observation noise
        pred_mean = posterior_mean + course_fit_adj + weather_adj
        pred_std = np.sqrt(posterior_std ** 2 + OBSERVATION_STD ** 2)

        return {
            "predicted_sg": round(float(pred_mean), 3),
            "prediction_std": round(float(pred_std), 3),
            "p10": round(float(pred_mean - 1.28 * pred_std), 2),
            "p25": round(float(pred_mean - 0.67 * pred_std), 2),
            "p50": round(float(pred_mean), 2),
            "p75": round(float(pred_mean + 0.67 * pred_std), 2),
            "p90": round(float(pred_mean + 1.28 * pred_std), 2),
        }


def compute_closer_index(player_history: list[dict]) -> float:
    """Compute Sunday Closer Index from historical data (R9).

    Positive = thrives under Sunday pressure (closer)
    Negative = chokes under Sunday pressure

    Uses: Round 4 SG differential vs Rounds 1-3 average,
    weighted by contention level (top-10 entering R4).
    """
    r4_diffs = []

    for event in player_history:
        rounds = event.get("round_scores", [])
        if len(rounds) < 4:
            continue

        # Compare R4 to R1-3 average
        r1_3_avg = np.mean(rounds[:3])
        r4 = rounds[3]
        diff = r4 - r1_3_avg

        # Weight by contention: more weight when player was in contention
        position_entering_r4 = event.get("position_after_r3")
        if position_entering_r4 is not None and position_entering_r4 <= 10:
            weight = 2.0  # Double weight for contention rounds
        elif position_entering_r4 is not None and position_entering_r4 <= 25:
            weight = 1.5
        else:
            weight = 1.0

        r4_diffs.append(diff * weight)

    if len(r4_diffs) < 5:
        return 0.0

    # Negative closer_index = closes well (scores better in R4)
    # Flip sign so positive = good closer
    raw_index = -float(np.mean(r4_diffs))

    # Regress toward zero
    confidence = min(len(r4_diffs) / 15.0, 1.0)
    return round(float(raw_index * confidence * 0.5), 3)
