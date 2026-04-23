"""
Strokes Gained Projection Model — v9.0 (Revolutionary Overhaul)

v9.0 changes from v8.0:
  1. Non-overlapping recency windows (events 1-4, 5-12, 13-24, 25-50)
  2. Cross-category signals applied AFTER regression (not before)
  3. Proper Bayesian regression without sample-size double-counting
  4. Bermuda/bentgrass putting surface adjustment (was a no-op)
  5. Course history signal integration
  6. Player-specific variance with field-strength normalization
  7. Cold hand min/max logic fix
  8. Regression factor ordering: OTT < APP < ATG < PUTT (research-validated)
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
    """Strokes Gained projection engine — v9.0"""

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
        for cat in ["sg_ott", "sg_app", "sg_atg", "sg_putt", "sg_total"]:
            if cat in all_player_stats.columns:
                self.tour_mean_sg[cat] = all_player_stats[cat].mean()
        log.info(f"Tour means updated: {self.tour_mean_sg}")

    def project_player(
        self,
        player_history: list[dict],
        course_name: str,
        is_bermuda: bool = None,
        course_history: list[dict] = None,
        field_strength_adj: float = 0.0,
    ) -> dict:
        if not player_history:
            return self._empty_projection()

        df = pd.DataFrame(player_history).sort_values("event_date")
        df = df.dropna(subset=["sg_total"])
        n = len(df)

        if n < 3:
            log.debug(f"Insufficient history ({n} events) — returning mean regression")

        # Step 1: Non-overlapping recency-weighted SG
        raw_proj = self._recency_weighted_sg(df)

        # Step 2: Regression toward mean (BEFORE cross-category signals)
        regressed = self._regress_to_mean(raw_proj, n_events=n)

        # Step 3: Cross-category signals (AFTER regression — on cleaner values)
        cross_adj = self._apply_cross_category_signals(regressed)

        # Step 4: Course fit adjustment
        course_adj = self._apply_course_fit(cross_adj, course_name)

        # Step 5: Putting surface adjustment (bermuda vs bentgrass)
        if is_bermuda is None:
            is_bermuda = course_name in BERMUDA_COURSES
        surface_adj = self._apply_surface_adjustment(course_adj, df, is_bermuda)

        # Step 6: Course history bonus
        course_bonus = self._apply_course_history(course_history)

        # Step 7: Cold hand penalty
        cold_hand = self._compute_cold_hand_penalty(df)

        # Step 8: Player-specific variance
        player_variance = self._estimate_player_variance(df)

        # Final SG total
        final = surface_adj.copy()
        final["sg_total"] = (
            final.get("sg_ott", 0) +
            final.get("sg_app", 0) +
            final.get("sg_atg", 0) +
            final.get("sg_putt", 0)
        )

        final["sg_total"] += cold_hand["penalty"]
        final["sg_total"] += course_bonus
        final["sg_total"] += field_strength_adj

        return {
            "proj_sg_ott":   round(final.get("sg_ott", 0), 3),
            "proj_sg_app":   round(final.get("sg_app", 0), 3),
            "proj_sg_atg":   round(final.get("sg_atg", 0), 3),
            "proj_sg_putt":  round(final.get("sg_putt", 0), 3),
            "proj_sg_total": round(final.get("sg_total", 0), 3),
            "course_fit_score": round(self._course_fit_score(regressed, course_name), 1),
            "form_trend":    cold_hand["status"],
            "cold_hand_penalty": round(cold_hand["penalty"], 3),
            "player_variance": round(player_variance, 3),
            "last4_avg":     round(self._window_avg(df, 4), 3),
            "last12_avg":    round(self._window_avg(df, 12), 3),
            "events_played": n,
            "raw_sg_total":  round(raw_proj.get("sg_total", 0), 3),
            "data_quality":  self._assess_data_quality(df, n),
            "course_history_bonus": round(course_bonus, 3),
            "field_strength_adj": round(field_strength_adj, 3),
        }

    # ─────────────────────────────────────────────
    # STEP 1: NON-OVERLAPPING RECENCY WINDOWS (v9.0)
    # ─────────────────────────────────────────────

    def _recency_weighted_sg(self, df: pd.DataFrame) -> dict:
        """Non-overlapping windows: events 1-4, 5-12, 13-24, 25-50."""
        cats = ["sg_ott", "sg_app", "sg_atg", "sg_putt"]
        result = {}

        n = len(df)
        # Non-overlapping slices (most recent first)
        w1 = df.tail(4)                                    # events 1-4
        w2 = df.tail(12).head(max(0, min(8, n - 4)))       # events 5-12
        w3 = df.tail(24).head(max(0, min(12, n - 12)))     # events 13-24
        w4 = df.tail(50).head(max(0, min(26, n - 24)))     # events 25-50

        weights = {
            "last_4":  FORM_WINDOWS.get("last_4", 0.10),
            "last_12": FORM_WINDOWS.get("last_12", 0.25),
            "last_24": FORM_WINDOWS.get("last_24", 0.30),
            "last_50": FORM_WINDOWS.get("last_50", 0.35),
        }

        for cat in cats:
            if cat not in df.columns:
                result[cat] = 0.0
                continue

            windows = []
            for subset, key in [(w1, "last_4"), (w2, "last_12"), (w3, "last_24"), (w4, "last_50")]:
                if len(subset) > 0 and subset[cat].notna().any():
                    val = subset[cat].mean()
                    if not np.isnan(val):
                        windows.append((weights[key], val))

            if windows:
                total_w = sum(w for w, _ in windows)
                result[cat] = sum(w * v for w, v in windows) / total_w
            else:
                result[cat] = 0.0

        result["sg_total"] = sum(result[c] for c in cats)
        return result

    # ─────────────────────────────────────────────
    # STEP 2: REGRESSION TO MEAN (v9.0 — fixed)
    # Proper Bayesian shrinkage without sample-size double-counting
    # ─────────────────────────────────────────────

    def _regress_to_mean(self, raw: dict, n_events: int) -> dict:
        """Bayesian shrinkage — regression factor IS the sample-size function."""
        regression_factors = {
            "sg_ott":  OTT_REGRESSION_FACTOR,       # 0.15 — most stable
            "sg_app":  APPROACH_REGRESSION_FACTOR,   # 0.20
            "sg_atg":  ATG_REGRESSION_FACTOR,        # 0.35
            "sg_putt": PUTTING_REGRESSION_FACTOR,    # 0.55 — most volatile
        }

        # Bayesian: r = sigma_prior^2 / (sigma_prior^2 + sigma_obs^2/n)
        # Simplified: scale regression by sample completeness
        sample_factor = min(n_events / float(FULL_CONFIDENCE_EVENTS), 1.0)

        regressed = {}
        for cat, raw_val in raw.items():
            if cat == "sg_total":
                continue
            mean = self.tour_mean_sg.get(cat, 0.0)
            base_r = regression_factors.get(cat, 0.35)
            # At full sample, use base regression factor directly
            # At partial sample, regress MORE (higher r → more shrinkage)
            effective_r = base_r + (1.0 - base_r) * (1.0 - sample_factor)
            regressed[cat] = raw_val * (1 - effective_r) + mean * effective_r

        regressed["sg_total"] = sum(regressed.get(c, 0) for c in ["sg_ott", "sg_app", "sg_atg", "sg_putt"])
        return regressed

    # ─────────────────────────────────────────────
    # STEP 3: CROSS-CATEGORY SIGNALS (v9.0 — after regression)
    # ─────────────────────────────────────────────

    def _apply_cross_category_signals(self, regressed: dict) -> dict:
        adjusted = regressed.copy()

        sg_ott = regressed.get("sg_ott", 0)
        sg_app = regressed.get("sg_app", 0)

        adjusted["sg_app"] += sg_ott * CROSS_CATEGORY_SIGNAL.get("sg_ott_to_sg_app", 0)
        adjusted["sg_atg"] += sg_app * CROSS_CATEGORY_SIGNAL.get("sg_app_to_sg_atg", 0)
        adjusted["sg_atg"] += sg_ott * CROSS_CATEGORY_SIGNAL.get("sg_ott_to_sg_atg", 0)

        adjusted["sg_total"] = sum(adjusted.get(c, 0) for c in ["sg_ott", "sg_app", "sg_atg", "sg_putt"])
        return adjusted

    # ─────────────────────────────────────────────
    # STEP 4: COURSE FIT ADJUSTMENT
    # ─────────────────────────────────────────────

    def _apply_course_fit(self, regressed: dict, course_name: str) -> dict:
        profile = get_course_profile(course_name)
        if not profile:
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
        adjusted["_wind_sensitivity"] = profile.get("wind_sensitivity", 0.5)
        adjusted["_elevation_ft"] = profile.get("elevation_ft", 0)
        return adjusted

    def _course_fit_score(self, regressed: dict, course_name: str) -> float:
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
    # STEP 5: SURFACE ADJUSTMENT (v9.0 — IMPLEMENTED)
    # Bermuda putting is ~0.15 correlated with bentgrass
    # ─────────────────────────────────────────────

    def _apply_surface_adjustment(self, adj: dict, df: pd.DataFrame, is_bermuda: bool) -> dict:
        result = adj.copy()

        if "sg_putt" not in df.columns or len(df) < 8:
            return result

        # If we have surface-tagged putting data, use split
        if "surface" in df.columns:
            bermuda_df = df[df["surface"] == "bermuda"]
            bent_df = df[df["surface"] == "bentgrass"]

            if is_bermuda and len(bermuda_df) >= 4:
                bermuda_putt = bermuda_df["sg_putt"].mean()
                overall_putt = result.get("sg_putt", 0)
                # Blend: 40% surface-specific, 60% overall
                result["sg_putt"] = overall_putt * 0.60 + bermuda_putt * 0.40
            elif not is_bermuda and len(bent_df) >= 4:
                bent_putt = bent_df["sg_putt"].mean()
                overall_putt = result.get("sg_putt", 0)
                result["sg_putt"] = overall_putt * 0.60 + bent_putt * 0.40
        else:
            # Without surface tags, apply a small regression penalty
            # for bermuda courses (harder surface, more variance)
            if is_bermuda:
                result["sg_putt"] = result.get("sg_putt", 0) * 0.90

        result["sg_total"] = sum(result.get(c, 0) for c in ["sg_ott", "sg_app", "sg_atg", "sg_putt"])
        return result

    # ─────────────────────────────────────────────
    # STEP 6: COURSE HISTORY (v9.0 NEW — R3)
    # Players gain 0.3-0.5 SG at venues they've played 3+ times
    # ─────────────────────────────────────────────

    def _apply_course_history(self, course_history: list[dict] = None) -> float:
        if not course_history or len(course_history) < 2:
            return 0.0

        sg_at_course = []
        for event in course_history:
            sg = event.get("sg_total")
            if sg is not None:
                sg_at_course.append(float(sg))

        if len(sg_at_course) < 2:
            return 0.0

        # Recency-weighted course history
        weights = np.array([0.5 ** i for i in range(len(sg_at_course))])
        weights = weights[::-1]  # Most recent gets highest weight
        weights /= weights.sum()

        course_avg = float(np.dot(sg_at_course, weights))

        # Confidence scales with number of visits
        n_visits = len(sg_at_course)
        confidence = min(n_visits / 5.0, 1.0)

        # Bonus is the deviation from zero, scaled by confidence
        # Cap at ±0.5 SG to prevent outlier influence
        bonus = np.clip(course_avg * 0.30 * confidence, -0.5, 0.5)
        return float(bonus)

    # ─────────────────────────────────────────────
    # STEP 7: COLD HAND PENALTY (v9.0 — min/max fix)
    # ─────────────────────────────────────────────

    def _compute_cold_hand_penalty(self, df: pd.DataFrame) -> dict:
        if "sg_total" not in df.columns or len(df) < 4:
            return {"status": "insufficient_data", "penalty": 0.0}

        sg = df["sg_total"].dropna()
        if len(sg) < 4:
            return {"status": "insufficient_data", "penalty": 0.0}

        last4 = sg.tail(4).values
        last12 = sg.tail(min(12, len(sg))).values

        last4_avg = np.mean(last4)
        last12_avg = np.mean(last12)

        consecutive_below = 0
        baseline = last12_avg
        for val in reversed(last4):
            if val < baseline - 0.3:
                consecutive_below += 1
            else:
                break

        penalty = 0.0
        status = "stable"

        if consecutive_below >= 3 and last4_avg < last12_avg - 0.5:
            penalty = -0.15
            status = "cold_hand"
        elif consecutive_below >= 2 and last4_avg < last12_avg - 0.8:
            penalty = -0.20
            status = "cold_hand"

        # Trend analysis — FIXED: use max(penalty, -0.10) to not override a more severe penalty
        if len(sg) >= 6:
            recent = sg.tail(6).values
            x = np.arange(len(recent))
            slope, _, _, p_value, _ = stats.linregress(x, recent)
            if slope < -0.05 and p_value < 0.10:
                if penalty == 0.0:
                    penalty = -0.10
                    status = "cold_hand"

        return {"status": status, "penalty": penalty}

    # ─────────────────────────────────────────────
    # STEP 8: PLAYER-SPECIFIC VARIANCE
    # ─────────────────────────────────────────────

    def _estimate_player_variance(self, df: pd.DataFrame) -> float:
        if "sg_total" not in df.columns or len(df) < 5:
            return 2.75

        sg = df["sg_total"].dropna()
        if len(sg) < 5:
            return 2.75

        player_std = float(sg.std())
        tour_avg_std = 2.75
        n = len(sg)
        shrinkage = min(n / 20.0, 1.0)
        estimated_std = player_std * shrinkage + tour_avg_std * (1 - shrinkage)
        return max(1.0, min(5.0, estimated_std))

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _window_avg(self, df: pd.DataFrame, n: int) -> float:
        if "sg_total" not in df.columns:
            return 0.0
        sg = df["sg_total"].dropna()
        window = sg.tail(n)
        return float(window.mean()) if len(window) > 0 else 0.0

    def _assess_data_quality(self, df: pd.DataFrame, n_events: int) -> str:
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
            "course_history_bonus": 0.0, "field_strength_adj": 0.0,
        }

    def project_field(
        self,
        field_history: dict[str, list[dict]],
        course_name: str,
        course_histories: dict[str, list[dict]] = None,
        field_sg_values: list[float] = None,
    ) -> pd.DataFrame:
        rows = []
        # Compute field strength adjustment
        field_strength_adj = 0.0
        if field_sg_values:
            field_mean = np.mean(field_sg_values)
            field_strength_adj = -field_mean * 0.15  # Normalize to neutral field

        for player_name, history in field_history.items():
            ch = (course_histories or {}).get(player_name)
            proj = self.project_player(
                history, course_name,
                course_history=ch,
                field_strength_adj=field_strength_adj,
            )
            proj["name"] = player_name
            rows.append(proj)

        df = pd.DataFrame(rows)

        if "proj_sg_total" in df.columns:
            df["model_rank"] = df["proj_sg_total"].rank(ascending=False).astype(int)
            df = df.sort_values("proj_sg_total", ascending=False)

        log.info(f"Projected {len(df)} players for {course_name}")
        return df
