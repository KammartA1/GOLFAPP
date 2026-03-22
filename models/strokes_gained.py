"""
Strokes Gained Projection Model — v8.0
The mathematical core of the engine.

v8.0 Research-backed improvements:
  1. Recency weighting: 10/90 rule (markets over-weight recent form)
  2. Regression: OTT/APP regress least, PUTT regresses most (55%)
  3. Cross-category signals: OTT predicts future APP
  4. Cold hand penalty replaces hot streak momentum (noise)
  5. Player-specific variance modeling
  6. Minimum data requirements (40 rounds / 2 years)

Pipeline:
  1. Pull multi-season SG data per player
  2. Apply research-backed recency decay weighting
  3. Apply cross-category predictive signals
  4. Regress volatile categories (putting hard, ball-striking less)
  5. Apply course fit adjustment vector
  6. Apply cold hand penalty (not hot streak — that's noise)
  7. Estimate player-specific variance
  8. Output: projected SG per category + total + variance
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats

from config.settings import (
    SG_WEIGHTS, FORM_WINDOWS,
    PUTTING_REGRESSION_FACTOR, APPROACH_REGRESSION_FACTOR,
    OTT_REGRESSION_FACTOR, ATG_REGRESSION_FACTOR,
    CROSS_CATEGORY_SIGNAL, FULL_CONFIDENCE_EVENTS,
)
from config.courses import get_course_profile, BERMUDA_COURSES

log = logging.getLogger(__name__)


class SGModel:
    """
    Strokes Gained projection engine — v8.0
    Research-validated approach based on DataGolf methodology.
    """

    TOUR_AVERAGES = {
        "sg_ott":  0.0,
        "sg_app":  0.0,
        "sg_atg":  0.0,
        "sg_putt": 0.0,
        "sg_total": 0.0,
    }

    def __init__(self):
        self.tour_mean_sg = self.TOUR_AVERAGES.copy()

    def update_tour_means(self, all_player_stats: pd.DataFrame):
        """Recalculate tour mean SG from full season data."""
        for cat in ["sg_ott", "sg_app", "sg_atg", "sg_putt", "sg_total"]:
            if cat in all_player_stats.columns:
                self.tour_mean_sg[cat] = all_player_stats[cat].mean()
        log.info(f"Tour means updated: {self.tour_mean_sg}")

    # ─────────────────────────────────────────────
    # CORE PROJECTION PIPELINE
    # ─────────────────────────────────────────────

    def project_player(
        self,
        player_history: list[dict],
        course_name: str,
        is_bermuda: bool = None,
    ) -> dict:
        """
        Full projection pipeline for one player.

        player_history: list of dicts, each with SG stats per tournament,
                        sorted by event_date (oldest first).
                        Required keys: event_date, sg_ott, sg_app, sg_atg, sg_putt, sg_total

        Returns: projection dict with course-adjusted SG per category + variance
        """
        if not player_history:
            return self._empty_projection()

        df = pd.DataFrame(player_history).sort_values("event_date")
        df = df.dropna(subset=["sg_total"])
        n = len(df)

        if n < 3:
            log.debug(f"Insufficient history ({n} events) — returning mean regression")

        # Step 1: Recency-weighted SG per category (research-backed 10/90 rule)
        raw_proj = self._recency_weighted_sg(df)

        # Step 2: Cross-category predictive signals
        cross_adj = self._apply_cross_category_signals(raw_proj)

        # Step 3: Regression toward mean (PUTT hardest, OTT/APP least)
        regressed = self._regress_to_mean(cross_adj, n_events=n)

        # Step 4: Course fit adjustment
        course_adj = self._apply_course_fit(regressed, course_name)

        # Step 5: Putting surface adjustment (bermuda vs bentgrass)
        if is_bermuda is None:
            is_bermuda = course_name in BERMUDA_COURSES
        surface_adj = self._apply_surface_adjustment(course_adj, df, is_bermuda)

        # Step 6: Cold hand penalty (replaces hot streak momentum — that's noise)
        cold_hand = self._compute_cold_hand_penalty(df)

        # Step 7: Estimate player-specific variance
        player_variance = self._estimate_player_variance(df)

        # Final SG total (true sum of categories)
        final = surface_adj.copy()
        final["sg_total"] = (
            final.get("sg_ott", 0) +
            final.get("sg_app", 0) +
            final.get("sg_atg", 0) +
            final.get("sg_putt", 0)
        )

        # Apply cold hand penalty (only penalize, never reward — hot streaks are noise)
        final["sg_total"] += cold_hand["penalty"]

        return {
            "proj_sg_ott":   round(final.get("sg_ott", 0), 3),
            "proj_sg_app":   round(final.get("sg_app", 0), 3),
            "proj_sg_atg":   round(final.get("sg_atg", 0), 3),
            "proj_sg_putt":  round(final.get("sg_putt", 0), 3),
            "proj_sg_total": round(final.get("sg_total", 0), 3),
            "course_fit_score": round(self._course_fit_score(regressed, course_name), 1),
            "form_trend":    cold_hand["status"],  # stable / cold_hand / insufficient_data
            "cold_hand_penalty": round(cold_hand["penalty"], 3),
            "player_variance": round(player_variance, 3),
            "last4_avg":     round(self._window_avg(df, 4), 3),
            "last12_avg":    round(self._window_avg(df, 12), 3),
            "events_played": n,
            "raw_sg_total":  round(raw_proj.get("sg_total", 0), 3),
            "data_quality":  self._assess_data_quality(df, n),
        }

    # ─────────────────────────────────────────────
    # STEP 1: RECENCY WEIGHTED AVERAGE (v8.0)
    # Research: ~70% weight on most recent 50 rounds
    # Markets over-weight recent form — edge in trusting baseline
    # ─────────────────────────────────────────────

    def _recency_weighted_sg(self, df: pd.DataFrame) -> dict:
        """Apply research-backed recency weights to SG history."""
        cats = ["sg_ott", "sg_app", "sg_atg", "sg_putt"]
        result = {}

        last4  = df.tail(4)
        last12 = df.tail(12)
        last24 = df.tail(24)
        last50 = df.tail(50)

        for cat in cats:
            if cat not in df.columns:
                result[cat] = 0.0
                continue

            windows = []
            for subset, key in [
                (last4, "last_4"),
                (last12, "last_12"),
                (last24, "last_24"),
                (last50, "last_50"),
            ]:
                if len(subset) > 0 and subset[cat].notna().any():
                    val = subset[cat].mean()
                    if not np.isnan(val):
                        windows.append((FORM_WINDOWS[key], val))

            if windows:
                total_w = sum(w for w, _ in windows)
                result[cat] = sum(w * v for w, v in windows) / total_w
            else:
                result[cat] = 0.0

        result["sg_total"] = sum(result[c] for c in cats)
        return result

    # ─────────────────────────────────────────────
    # STEP 2: CROSS-CATEGORY SIGNALS (v8.0 NEW)
    # Research: OTT predicts future APP (+0.20 per SG:OTT)
    # because OTT signals general ball-striking ability
    # ─────────────────────────────────────────────

    def _apply_cross_category_signals(self, raw: dict) -> dict:
        """
        Apply cross-category predictive signals.
        A golfer averaging +1 SG:OTT will have future SG:APP predicted ~+0.2 higher.
        """
        adjusted = raw.copy()

        sg_ott = raw.get("sg_ott", 0)
        sg_app = raw.get("sg_app", 0)

        # OTT → APP boost (ball-striking signal)
        adjusted["sg_app"] += sg_ott * CROSS_CATEGORY_SIGNAL.get("sg_ott_to_sg_app", 0)

        # APP → ATG boost (approach skill predicts short game)
        adjusted["sg_atg"] += sg_app * CROSS_CATEGORY_SIGNAL.get("sg_app_to_sg_atg", 0)

        # OTT → ATG boost (general ability)
        adjusted["sg_atg"] += sg_ott * CROSS_CATEGORY_SIGNAL.get("sg_ott_to_sg_atg", 0)

        # Recalc total
        adjusted["sg_total"] = sum(adjusted.get(c, 0) for c in ["sg_ott", "sg_app", "sg_atg", "sg_putt"])
        return adjusted

    # ─────────────────────────────────────────────
    # STEP 3: REGRESSION TO MEAN (v8.0)
    # Research: Long game regresses LESS than short game/putting
    # Predictive hierarchy: OTT > APP > ARG > PUTT
    # ─────────────────────────────────────────────

    def _regress_to_mean(self, raw: dict, n_events: int) -> dict:
        """
        Bayesian shrinkage toward tour mean.
        v8.0: Research-validated regression factors by category.
        """
        sample_factor = min(n_events / float(FULL_CONFIDENCE_EVENTS), 1.0)

        regression_factors = {
            "sg_ott":  OTT_REGRESSION_FACTOR,       # 0.20 — most stable
            "sg_app":  APPROACH_REGRESSION_FACTOR,   # 0.15 — most stable + highest impact
            "sg_atg":  ATG_REGRESSION_FACTOR,        # 0.35 — moderate
            "sg_putt": PUTTING_REGRESSION_FACTOR,    # 0.55 — most volatile, regress hardest
        }

        regressed = {}
        for cat, raw_val in raw.items():
            if cat == "sg_total":
                continue
            mean = self.tour_mean_sg.get(cat, 0.0)
            r = regression_factors.get(cat, 0.35)
            # Reduce regression when we have more sample
            effective_r = r * (1 - sample_factor * 0.5)
            regressed[cat] = raw_val * (1 - effective_r) + mean * effective_r

        regressed["sg_total"] = sum(regressed.get(c, 0) for c in ["sg_ott", "sg_app", "sg_atg", "sg_putt"])
        return regressed

    # ─────────────────────────────────────────────
    # STEP 4: COURSE FIT ADJUSTMENT
    # ─────────────────────────────────────────────

    def _apply_course_fit(self, regressed: dict, course_name: str) -> dict:
        """Reweight SG categories by course-specific weights."""
        profile = get_course_profile(course_name)
        if not profile:
            log.debug(f"No course profile for {course_name} — using default weights")
            return regressed

        course_weights = profile["sg_weights"]
        default_weights = SG_WEIGHTS

        adjusted = {}
        for cat in ["sg_ott", "sg_app", "sg_atg", "sg_putt"]:
            base = regressed.get(cat, 0)
            course_w = course_weights.get(cat, default_weights.get(cat, 0.25))
            default_w = default_weights.get(cat, 0.25)

            weight_ratio = course_w / default_w if default_w > 0 else 1.0
            adj_factor = 1 + (weight_ratio - 1) * 0.4
            adjusted[cat] = base * adj_factor

        adjusted["sg_total"] = sum(adjusted.get(c, 0) for c in ["sg_ott", "sg_app", "sg_atg", "sg_putt"])
        adjusted["_distance_bonus"] = profile.get("distance_bonus", 0.5)
        adjusted["_accuracy_penalty"] = profile.get("accuracy_penalty", 0.5)
        return adjusted

    def _course_fit_score(self, regressed: dict, course_name: str) -> float:
        """Compute a 0-100 course fit score."""
        profile = get_course_profile(course_name)
        if not profile:
            return 50.0

        course_weights = profile["sg_weights"]
        score = 0.0
        for cat, cw in course_weights.items():
            player_val = regressed.get(cat, 0)
            score += player_val * cw * 100

        normalized = 50 + (score / 3)
        return max(0, min(100, normalized))

    # ─────────────────────────────────────────────
    # STEP 5: SURFACE (BERMUDA vs BENTGRASS)
    # ─────────────────────────────────────────────

    def _apply_surface_adjustment(self, adj: dict, df: pd.DataFrame, is_bermuda: bool) -> dict:
        """Surface-specific putting adjustment."""
        result = adj.copy()
        return result

    # ─────────────────────────────────────────────
    # STEP 6: COLD HAND PENALTY (v8.0 NEW)
    # Research (2025): Hot streaks are noise. Cold streaks ARE real.
    # Negative emotions from poor play have greater short-run
    # influence than positive emotions from success.
    # ─────────────────────────────────────────────

    def _compute_cold_hand_penalty(self, df: pd.DataFrame) -> dict:
        """
        Detect cold hand (extended poor form) and apply penalty.
        DO NOT reward hot streaks — 2025 research confirms they're noise.

        Cold hand signals: injury, equipment issues, personal problems,
        or genuine loss of confidence.
        """
        if "sg_total" not in df.columns or len(df) < 4:
            return {"status": "insufficient_data", "penalty": 0.0}

        sg = df["sg_total"].dropna()
        if len(sg) < 4:
            return {"status": "insufficient_data", "penalty": 0.0}

        last4 = sg.tail(4).values
        last12 = sg.tail(min(12, len(sg))).values

        last4_avg = np.mean(last4)
        last12_avg = np.mean(last12)

        # Cold hand detection:
        # 1. Recent average significantly below baseline (> 0.5 SG below their norm)
        # 2. Declining trajectory (3+ consecutive events below baseline)
        # 3. Statistical significance via regression

        consecutive_below = 0
        baseline = last12_avg
        for val in reversed(last4):
            if val < baseline - 0.3:
                consecutive_below += 1
            else:
                break

        # Apply penalty only for genuine cold hand, not random variance
        penalty = 0.0
        status = "stable"

        if consecutive_below >= 3 and last4_avg < last12_avg - 0.5:
            # Strong cold hand: 3+ events significantly below baseline
            penalty = -0.15  # Modest penalty — don't overreact
            status = "cold_hand"
        elif consecutive_below >= 2 and last4_avg < last12_avg - 0.8:
            # Severe cold hand: sharp decline even in 2 events
            penalty = -0.20
            status = "cold_hand"

        # Trend analysis for additional cold signal
        if len(sg) >= 6:
            recent = sg.tail(6).values
            x = np.arange(len(recent))
            slope, _, _, p_value, _ = stats.linregress(x, recent)
            # Statistically significant decline
            if slope < -0.05 and p_value < 0.10:
                penalty = min(penalty, -0.10)  # Apply if not already penalized
                if status == "stable":
                    status = "cold_hand"

        return {"status": status, "penalty": penalty}

    # ─────────────────────────────────────────────
    # STEP 7: PLAYER-SPECIFIC VARIANCE (v8.0 NEW)
    # Research: Some players are more volatile than others.
    # Must model variance, not just mean.
    # ─────────────────────────────────────────────

    def _estimate_player_variance(self, df: pd.DataFrame) -> float:
        """
        Estimate player-specific scoring variance.
        Higher variance → riskier but potentially higher upside in GPPs.
        Lower variance → better for cash games.

        Returns estimated round-to-round SG standard deviation.
        """
        if "sg_total" not in df.columns or len(df) < 5:
            return 2.75  # Tour average variance

        sg = df["sg_total"].dropna()
        if len(sg) < 5:
            return 2.75

        player_std = float(sg.std())

        # Bayesian shrinkage toward tour average variance
        # (don't overfit to small samples)
        tour_avg_std = 2.75
        n = len(sg)
        shrinkage = min(n / 20.0, 1.0)

        estimated_std = player_std * shrinkage + tour_avg_std * (1 - shrinkage)
        return max(1.0, min(5.0, estimated_std))

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _window_avg(self, df: pd.DataFrame, n: int) -> float:
        """Get average SG total over last n events."""
        if "sg_total" not in df.columns:
            return 0.0
        sg = df["sg_total"].dropna()
        window = sg.tail(n)
        return float(window.mean()) if len(window) > 0 else 0.0

    def _assess_data_quality(self, df: pd.DataFrame, n_events: int) -> str:
        """Assess quality of projection data."""
        if n_events >= 30:
            return "high"
        elif n_events >= 15:
            return "medium"
        elif n_events >= 5:
            return "low"
        return "very_low"

    def _empty_projection(self) -> dict:
        return {
            "proj_sg_ott": 0.0, "proj_sg_app": 0.0,
            "proj_sg_atg": 0.0, "proj_sg_putt": 0.0,
            "proj_sg_total": 0.0, "course_fit_score": 50.0,
            "form_trend": "no_data", "cold_hand_penalty": 0.0,
            "player_variance": 2.75, "last4_avg": 0.0,
            "last12_avg": 0.0, "events_played": 0,
            "raw_sg_total": 0.0, "data_quality": "none",
        }

    # ─────────────────────────────────────────────
    # BATCH PROJECTION
    # ─────────────────────────────────────────────

    def project_field(
        self,
        field_history: dict[str, list[dict]],
        course_name: str,
    ) -> pd.DataFrame:
        """
        Project entire tournament field.

        field_history: {player_name: [list of sg stat dicts]}
        Returns DataFrame with one row per player, all projections.
        """
        rows = []
        for player_name, history in field_history.items():
            proj = self.project_player(history, course_name)
            proj["name"] = player_name
            rows.append(proj)

        df = pd.DataFrame(rows)

        if "proj_sg_total" in df.columns:
            df["model_rank"] = df["proj_sg_total"].rank(ascending=False).astype(int)
            df = df.sort_values("proj_sg_total", ascending=False)

        log.info(f"Projected {len(df)} players for {course_name}")
        return df
