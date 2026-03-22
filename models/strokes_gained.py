"""
Strokes Gained Projection Model
The mathematical core of the engine.

Approach:
  1. Pull multi-season SG data per player
  2. Apply recency decay weighting
  3. Regress volatile categories (putting) toward mean
  4. Apply course fit adjustment vector
  5. Apply form trend signal
  6. Output: projected SG per category + total
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats

from config.settings import (
    SG_WEIGHTS, FORM_WINDOWS,
    PUTTING_REGRESSION_FACTOR, APPROACH_REGRESSION_FACTOR
)
from config.courses import get_course_profile, BERMUDA_COURSES

log = logging.getLogger(__name__)


class SGModel:
    """
    Strokes Gained projection engine.
    Produces per-player, course-adjusted SG projections.
    """

    # Tour averages (approximate — update with each season's data)
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

        Returns: projection dict with course-adjusted SG per category + win/top probabilities
        """
        if not player_history:
            return self._empty_projection()

        df = pd.DataFrame(player_history).sort_values("event_date")
        df = df.dropna(subset=["sg_total"])
        n = len(df)

        if n < 3:
            log.debug(f"Insufficient history ({n} events) — returning mean regression")

        # Step 1: Recency-weighted SG per category
        raw_proj = self._recency_weighted_sg(df)

        # Step 2: Regression toward mean
        regressed = self._regress_to_mean(raw_proj, n_events=n)

        # Step 3: Course fit adjustment
        course_adj = self._apply_course_fit(regressed, course_name)

        # Step 4: Putting surface adjustment (bermuda vs bentgrass)
        if is_bermuda is None:
            is_bermuda = course_name in BERMUDA_COURSES
        surface_adj = self._apply_surface_adjustment(course_adj, df, is_bermuda)

        # Step 5: Form trend signal
        form = self._compute_form_trend(df)

        # Step 6: Final SG total
        final = surface_adj.copy()
        final["sg_total"] = sum([
            final.get("sg_ott", 0) * SG_WEIGHTS["sg_ott"],
            final.get("sg_app", 0) * SG_WEIGHTS["sg_app"],
            final.get("sg_atg", 0) * SG_WEIGHTS["sg_atg"],
            final.get("sg_putt", 0) * SG_WEIGHTS["sg_putt"],
        ]) / sum(SG_WEIGHTS.values())  # normalize (should ~= 1.0 already)

        # Recompute as true sum:
        final["sg_total"] = (
            final.get("sg_ott", 0) +
            final.get("sg_app", 0) +
            final.get("sg_atg", 0) +
            final.get("sg_putt", 0)
        )

        return {
            "proj_sg_ott":   round(final.get("sg_ott", 0), 3),
            "proj_sg_app":   round(final.get("sg_app", 0), 3),
            "proj_sg_atg":   round(final.get("sg_atg", 0), 3),
            "proj_sg_putt":  round(final.get("sg_putt", 0), 3),
            "proj_sg_total": round(final.get("sg_total", 0), 3),
            "course_fit_score": round(self._course_fit_score(regressed, course_name), 1),
            "form_trend":    form["trend"],
            "last4_avg":     round(form["last4_avg"], 3),
            "last12_avg":    round(form["last12_avg"], 3),
            "events_played": n,
            "raw_sg_total":  round(raw_proj.get("sg_total", 0), 3),
        }

    # ─────────────────────────────────────────────
    # STEP 1: RECENCY WEIGHTED AVERAGE
    # ─────────────────────────────────────────────

    def _recency_weighted_sg(self, df: pd.DataFrame) -> dict:
        """Apply decaying recency weights to SG history."""
        n = len(df)
        cats = ["sg_ott", "sg_app", "sg_atg", "sg_putt"]
        result = {}

        last4  = df.tail(4)
        last12 = df.tail(12)
        last24 = df.tail(24)

        for cat in cats:
            col = cat if cat in df.columns else None
            if not col:
                result[cat] = 0.0
                continue

            v4  = last4[col].mean()  if len(last4)  > 0 and last4[col].notna().any()  else None
            v12 = last12[col].mean() if len(last12) > 0 and last12[col].notna().any() else None
            v24 = last24[col].mean() if len(last24) > 0 and last24[col].notna().any() else None

            weights, values = [], []
            if v4  is not None and not np.isnan(v4):
                weights.append(FORM_WINDOWS["last_4"])
                values.append(v4)
            if v12 is not None and not np.isnan(v12):
                weights.append(FORM_WINDOWS["last_12"])
                values.append(v12)
            if v24 is not None and not np.isnan(v24):
                weights.append(FORM_WINDOWS["last_24"])
                values.append(v24)

            if values:
                total_w = sum(weights)
                result[cat] = sum(w * v for w, v in zip(weights, values)) / total_w
            else:
                result[cat] = 0.0

        result["sg_total"] = sum(result[c] for c in cats)
        return result

    # ─────────────────────────────────────────────
    # STEP 2: REGRESSION TO MEAN
    # ─────────────────────────────────────────────

    def _regress_to_mean(self, raw: dict, n_events: int) -> dict:
        """
        Bayesian shrinkage toward tour mean.
        More regression with fewer events.
        Putting regresses hardest (most volatile).
        Approach regresses least (most predictive).
        """
        # Shrinkage factor: fewer events = more regression
        sample_factor = min(n_events / 20.0, 1.0)  # Full confidence at 20+ events

        regression_factors = {
            "sg_ott":  0.30,
            "sg_app":  APPROACH_REGRESSION_FACTOR,
            "sg_atg":  0.35,
            "sg_putt": PUTTING_REGRESSION_FACTOR,
        }

        regressed = {}
        for cat, raw_val in raw.items():
            if cat == "sg_total":
                continue
            mean = self.tour_mean_sg.get(cat, 0.0)
            r = regression_factors.get(cat, 0.35)
            # Reduce regression factor when we have more sample
            effective_r = r * (1 - sample_factor * 0.5)
            regressed[cat] = raw_val * (1 - effective_r) + mean * effective_r

        regressed["sg_total"] = sum(regressed.get(c, 0) for c in ["sg_ott", "sg_app", "sg_atg", "sg_putt"])
        return regressed

    # ─────────────────────────────────────────────
    # STEP 3: COURSE FIT ADJUSTMENT
    # ─────────────────────────────────────────────

    def _apply_course_fit(self, regressed: dict, course_name: str) -> dict:
        """
        Reweight SG categories by course-specific weights.
        This changes the projection to reflect what the course actually rewards.
        """
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

            # Boost/penalize based on course weight vs default weight
            weight_ratio = course_w / default_w
            # Dampen the adjustment (don't overfit to course weights)
            adj_factor = 1 + (weight_ratio - 1) * 0.4
            adjusted[cat] = base * adj_factor

        # Distance bonus: if player is long and course rewards it
        distance_bonus = profile.get("distance_bonus", 0.5)
        # This gets applied in the projection engine when we have distance data

        adjusted["sg_total"] = sum(adjusted.get(c, 0) for c in ["sg_ott", "sg_app", "sg_atg", "sg_putt"])
        adjusted["_distance_bonus"] = distance_bonus
        adjusted["_accuracy_penalty"] = profile.get("accuracy_penalty", 0.5)
        return adjusted

    def _course_fit_score(self, regressed: dict, course_name: str) -> float:
        """
        Compute a 0-100 course fit score for a player.
        Based on how well their skill profile matches the course's demands.
        """
        profile = get_course_profile(course_name)
        if not profile:
            return 50.0

        course_weights = profile["sg_weights"]
        score = 0.0
        max_score = 0.0

        for cat, cw in course_weights.items():
            player_val = regressed.get(cat, 0)
            # Positive SG in a heavily-weighted category = high fit score
            contribution = player_val * cw * 100
            score += contribution
            max_score += abs(cw) * 100

        # Normalize to 0-100 range
        normalized = 50 + (score / 3)  # Tour avg player = 50
        return max(0, min(100, normalized))

    # ─────────────────────────────────────────────
    # STEP 4: SURFACE (BERMUDA vs BENTGRASS)
    # ─────────────────────────────────────────────

    def _apply_surface_adjustment(self, adj: dict, df: pd.DataFrame, is_bermuda: bool) -> dict:
        """
        Players have different putting performance on bermuda vs bentgrass.
        If we have surface-split data, apply adjustment. Otherwise pass through.
        """
        # If we have bermuda/bentgrass split putting data in history, use it.
        # For now, apply a small regression penalty on putting if surface type
        # is different from what the player primarily plays on.
        # (Full implementation requires surface-tagged historical data)
        result = adj.copy()
        return result

    # ─────────────────────────────────────────────
    # STEP 5: FORM TREND
    # ─────────────────────────────────────────────

    def _compute_form_trend(self, df: pd.DataFrame) -> dict:
        """Detect whether player is on an improving/declining/stable trend."""
        if "sg_total" not in df.columns or len(df) < 4:
            return {"trend": "insufficient_data", "last4_avg": 0, "last12_avg": 0}

        sg = df["sg_total"].dropna()
        last4  = float(sg.tail(4).mean())  if len(sg) >= 4  else float(sg.mean())
        last12 = float(sg.tail(12).mean()) if len(sg) >= 12 else float(sg.mean())

        # Trend = slope of recent SG values
        if len(sg) >= 6:
            recent = sg.tail(6).values
            x = np.arange(len(recent))
            slope, _, r_value, p_value, _ = stats.linregress(x, recent)
            # [v6.0] Tightened p-value from 0.3 → 0.10 for statistical significance
            improving = slope > 0.03 and p_value < 0.10
            declining = slope < -0.03 and p_value < 0.10
        else:
            improving = last4 > last12 + 0.2
            declining = last4 < last12 - 0.2

        trend = "improving" if improving else "declining" if declining else "stable"
        return {
            "trend": trend,
            "last4_avg": last4,
            "last12_avg": last12,
        }

    def _empty_projection(self) -> dict:
        return {
            "proj_sg_ott": 0.0, "proj_sg_app": 0.0,
            "proj_sg_atg": 0.0, "proj_sg_putt": 0.0,
            "proj_sg_total": 0.0, "course_fit_score": 50.0,
            "form_trend": "no_data", "last4_avg": 0.0,
            "last12_avg": 0.0, "events_played": 0,
            "raw_sg_total": 0.0,
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

        # Rank by projected SG total
        if "proj_sg_total" in df.columns:
            df["model_rank"] = df["proj_sg_total"].rank(ascending=False).astype(int)
            df = df.sort_values("proj_sg_total", ascending=False)

        log.info(f"Projected {len(df)} players for {course_name}")
        return df
