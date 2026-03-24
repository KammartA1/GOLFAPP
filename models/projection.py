"""
Master Projection Engine
Orchestrates all model layers into final player projections for a tournament.

Pipeline:
  1. Fetch field from PGA Tour
  2. Pull SG history per player
  3. SG model → course-adjusted projections
  4. Ownership model → leverage scores
  5. Weather model → tee time adjustments
  6. DataGolf (if active) → supplement/override
  7. DFS scoring → DK/FD projected points
  8. Output: ranked DataFrame with all metrics
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from models.strokes_gained import SGModel
from models.ownership import OwnershipModel
from data.scrapers.weather import WeatherModel
from data.scrapers.datagolf import DataGolfClient
from data.storage.database import get_session, Player, Tournament, Projection, SGStats
from config.settings import DATAGOLF_ENABLED
from config.courses import get_course_profile
from simulation.pipeline_bridge import SimulationBridge

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DFS SCORING FORMULAS
# ─────────────────────────────────────────────

DK_SCORING = {
    "hole_in_one":      5.0,
    "eagle":            8.0,
    "birdie":           3.0,
    "par":              0.5,
    "bogey":           -0.5,
    "double_bogey":    -1.0,
    "streak_bonus":     3.0,   # 3+ consecutive birdies or better
    "bogey_free":       3.0,   # Full 18 holes bogey-free
    "made_cut":         3.0,
    "finish_1st":      30.0,
    "finish_2nd":      20.0,
    "finish_3rd":      18.0,
    "finish_4th":      16.0,
    "finish_5th":      14.0,
    "finish_6th":      12.0,
    "finish_7th":      10.0,
    "finish_8th":       9.0,
    "finish_9th":       8.5,
    "finish_10th":      8.0,
}

FD_SCORING = {
    "eagle":            8.0,
    "birdie":           4.0,
    "par":              0.0,
    "bogey":           -1.0,
    "double_bogey":    -2.0,
    "made_cut":         10.0,
    "finish_1st":      26.0,
    "finish_2nd":      20.0,
    "finish_3rd":      18.0,
    "finish_4th":      16.0,
    "finish_5th":      14.0,
}


def sg_to_dk_points(proj_sg_total: float, make_cut_prob: float = 0.65) -> float:
    """
    Convert projected SG total to estimated DraftKings fantasy points.

    Calibrated from historical SG vs DK points correlation:
    - Tour average (SG = 0) ≈ 35-40 pts per tournament
    - Each +1 SG ≈ +8-10 DK pts
    - Make cut bonus (3 pts if 2 rounds, ~65% of field makes cut)
    """
    base_pts = 37.0                           # Tour average baseline
    sg_pts = proj_sg_total * 9.2              # ~9.2 pts per SG stroke
    cut_pts = make_cut_prob * DK_SCORING["made_cut"]
    return round(base_pts + sg_pts + cut_pts, 2)


def sg_to_fd_points(proj_sg_total: float, make_cut_prob: float = 0.65) -> float:
    """FanDuel uses different scoring — slightly higher variance."""
    base_pts = 30.0
    sg_pts = proj_sg_total * 8.5
    cut_pts = make_cut_prob * FD_SCORING["made_cut"]
    return round(base_pts + sg_pts + cut_pts, 2)


def sg_to_win_prob(proj_sg_total: float, field_sg_values: list[float]) -> float:
    """
    Convert projected SG to win probability using field distribution.
    Uses Monte Carlo simulation of tournament variance.
    """
    if not field_sg_values:
        return 0.01

    # Tournament variance per round (empirically ~2.5-3.0 SG)
    round_std = 2.75
    tournament_std = round_std * np.sqrt(4) / 2  # 4 rounds, but correlated

    n_sim = 10000
    wins = 0
    field_arr = np.array(field_sg_values)

    player_scores = np.random.normal(proj_sg_total * 4, tournament_std * 4, n_sim)

    for _ in range(n_sim // 100):  # Vectorized per batch
        pass  # Placeholder for full Monte Carlo

    # Simplified normal approximation
    field_mean = np.mean(field_arr)
    field_std  = np.std(field_arr)
    combined_std = np.sqrt(field_std**2 + tournament_std**2)
    z = (proj_sg_total - field_mean) / combined_std if combined_std > 0 else 0

    from scipy import stats
    win_p = 1 - stats.norm.cdf(0, loc=z, scale=1) if z > 0 else 0.005
    # Normalize — exactly 1 winner per field
    n_players = len(field_sg_values)
    # [v6.0] Removed 40% cap — elite players in weak fields can genuinely exceed 40%
    win_prob = min(win_p / n_players * 1.5, 0.55)
    return round(max(0.001, win_prob), 4)


def sg_to_top_n_prob(proj_sg_total: float, field_sg_values: list[float], n: int) -> float:
    """Estimate probability of finishing top-N."""
    if not field_sg_values:
        return n / 156  # field average
    field_arr = np.array(field_sg_values)
    # Players below our projection
    pct_beaten = (field_arr < proj_sg_total).mean()
    # Scale for variance (golf is high variance)
    prob = pct_beaten ** (1 / np.sqrt(n))
    # Calibrate: top 10 in 156 = 6.4% if average
    base = n / len(field_arr)
    # Blend model probability with base rate
    blended = prob * 0.6 + base * 0.4 if proj_sg_total > 0 else base
    return round(min(blended, 0.95), 4)


def sg_to_make_cut_prob(proj_sg_total: float) -> float:
    """Make cut probability from projected SG total."""
    # Rough logistic calibration from historical data
    # SG = 0 (tour avg) → ~65% make cut
    # SG = +1 → ~80%, SG = -1 → ~45%
    base = 0.65
    adj = proj_sg_total * 0.15
    return round(min(0.97, max(0.05, base + adj)), 3)


# ─────────────────────────────────────────────
# MASTER PROJECTION ENGINE
# ─────────────────────────────────────────────

class ProjectionEngine:
    """
    Orchestrates the full projection pipeline for a tournament.
    """

    def __init__(self, enable_simulation: bool = False, n_sims: int = 10000, sim_weight: float = 0.5):
        self.sg_model   = SGModel()
        self.own_model  = OwnershipModel()
        self.weather    = WeatherModel()
        self.datagolf   = DataGolfClient()
        self.enable_simulation = enable_simulation
        self.sim_weight = sim_weight
        self.sim_bridge = SimulationBridge(n_sims=n_sims) if enable_simulation else None

    def run(
        self,
        tournament_name: str,
        course_name: str,
        field_data: list[dict],         # [{name, world_rank, pga_player_id, ...}]
        sg_history: dict[str, list],    # {player_name: [sg_stat_dicts]}
        salaries: dict[str, dict] = None,  # {player_name: {dk_salary, fd_salary}}
        tee_times: dict[str, list] = None, # {player_name: ["AM", "PM", "AM", "AM"]}
        tournament_dates: list[datetime] = None,
        model_version: str = "1.0",
    ) -> pd.DataFrame:
        """
        Full projection run for a tournament field.
        Returns complete DataFrame with all metrics, ranked by model.
        """
        log.info(f"🏌️ Running projections: {tournament_name} @ {course_name}")

        # ── Step 1: Update tour means from available history ───────────────
        all_history_flat = []
        for recs in sg_history.values():
            all_history_flat.extend(recs)
        if all_history_flat:
            self.sg_model.update_tour_means(pd.DataFrame(all_history_flat))

        # ── Step 2: Weather adjustment ─────────────────────────────────────
        weather_adj = {}
        if tee_times and tournament_dates:
            try:
                weather_adj = self.weather.score_player_tee_times(
                    tee_times, course_name, tournament_dates
                )
            except Exception as e:
                log.warning(f"Weather model failed: {e}")

        # ── Step 3: DataGolf supplement (if active) ────────────────────────
        dg_predictions = {}
        if DATAGOLF_ENABLED:
            try:
                dg_preds = self.datagolf.get_predictions(add_course_fit=True)
                dg_predictions = {p["name"]: p for p in dg_preds}
                log.info(f"DataGolf predictions loaded: {len(dg_predictions)} players")
            except Exception as e:
                log.warning(f"DataGolf fetch failed: {e}")

        # ── Step 4: Project each player ────────────────────────────────────
        rows = []
        for player_info in field_data:
            name = player_info.get("name", "")
            history = sg_history.get(name, [])

            # SG projection
            sg_proj = self.sg_model.project_player(history, course_name)

            # Weather adjustment
            tee_adv = weather_adj.get(name, 0.0)
            sg_proj["proj_sg_total"] += tee_adv * 0.15  # Weather converted to SG

            # DataGolf override (if active — trust DG course fit model)
            dg = dg_predictions.get(name, {})
            if dg:
                # Blend our model 40% / DataGolf 60% when available
                dg_sg = dg.get("sg_total", sg_proj["proj_sg_total"])
                sg_proj["proj_sg_total"] = sg_proj["proj_sg_total"] * 0.40 + dg_sg * 0.60
                sg_proj["win_prob_dg"]   = dg.get("win_prob")
                sg_proj["dg_course_fit"] = dg.get("course_fit_score")

            # Win / top-N probabilities (placeholder — filled after all projections)
            row = {
                "name":         name,
                "world_rank":   player_info.get("world_rank"),
                **sg_proj,
                "tee_time_adv": tee_adv,
                "weather_adj":  round(tee_adv * 0.15, 3),
                "recent_results": player_info.get("recent_results", []),
            }

            # Salaries
            if salaries and name in salaries:
                sal = salaries[name]
                row["dk_salary"] = sal.get("dk_salary")
                row["fd_salary"] = sal.get("fd_salary")

            rows.append(row)

        df = pd.DataFrame(rows)

        if df.empty:
            log.error("No projections generated — check field data and SG history")
            return df

        # ── Step 5: Compute win/top probabilities across field ─────────────
        field_sg = df["proj_sg_total"].dropna().tolist()
        df["win_prob"]     = df["proj_sg_total"].apply(lambda x: sg_to_win_prob(x, field_sg))
        df["top5_prob"]    = df["proj_sg_total"].apply(lambda x: sg_to_top_n_prob(x, field_sg, 5))
        df["top10_prob"]   = df["proj_sg_total"].apply(lambda x: sg_to_top_n_prob(x, field_sg, 10))
        df["top20_prob"]   = df["proj_sg_total"].apply(lambda x: sg_to_top_n_prob(x, field_sg, 20))
        df["make_cut_prob"]= df["proj_sg_total"].apply(sg_to_make_cut_prob)

        # ── Step 6: DFS Points ─────────────────────────────────────────────
        df["dk_proj_pts"] = df.apply(
            lambda r: sg_to_dk_points(r["proj_sg_total"], r["make_cut_prob"]), axis=1
        )
        df["fd_proj_pts"] = df.apply(
            lambda r: sg_to_fd_points(r["proj_sg_total"], r["make_cut_prob"]), axis=1
        )

        # Value ratios
        if "dk_salary" in df.columns:
            df["dk_value"] = df.apply(
                lambda r: round(r["dk_proj_pts"] / (r["dk_salary"] / 1000), 2)
                if r.get("dk_salary") else None, axis=1
            )
        if "fd_salary" in df.columns:
            df["fd_value"] = df.apply(
                lambda r: round(r["fd_proj_pts"] / (r["fd_salary"] / 1000), 2)
                if r.get("fd_salary") else None, axis=1
            )

        # ── Step 7: Model rank ─────────────────────────────────────────────
        df["model_rank"] = df["proj_sg_total"].rank(ascending=False).astype(int)

        # ── Step 8: Ownership projections ──────────────────────────────────
        player_dicts = df.to_dict("records")
        own_df = self.own_model.project_field_ownership(player_dicts, df)
        if "proj_ownership" in own_df.columns:
            df["proj_ownership"] = own_df["proj_ownership"].values
            df["leverage_score"]  = own_df.get("leverage_score", pd.Series([0]*len(df))).values

        # ── Step 9: Tournament Simulation (if enabled) ──────────────────────
        if self.enable_simulation and self.sim_bridge is not None:
            try:
                log.info("Running tournament simulation (%d sims)...", self.sim_bridge.n_sims)
                sim_df = self.sim_bridge.run_tournament_simulation(
                    field_projections=df,
                    course_name=course_name,
                )
                df = self.sim_bridge.blend_projections(
                    analytical=df,
                    simulated=sim_df,
                    sim_weight=self.sim_weight,
                )
                log.info("Simulation blended into projections (weight=%.0f%%)", self.sim_weight * 100)
            except Exception as e:
                log.warning(f"Tournament simulation failed (non-fatal): {e}")

        # ── Step 10: Sort and finalize ──────────────────────────────────────
        df = df.sort_values("proj_sg_total", ascending=False).reset_index(drop=True)

        log.info(f"Projections complete: {len(df)} players | "
                 f"Leader: {df.iloc[0]['name']} ({df.iloc[0]['proj_sg_total']:+.3f} SG)")
        return df

    def top_plays(self, df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Return top N players by projected SG total."""
        cols = [c for c in [
            "name", "world_rank", "model_rank",
            "proj_sg_total", "proj_sg_app", "proj_sg_putt",
            "course_fit_score", "form_trend",
            "win_prob", "top10_prob", "make_cut_prob",
            "dk_salary", "dk_proj_pts", "dk_value",
            "proj_ownership", "leverage_score",
        ] if c in df.columns]
        return df[cols].head(n)

    def gpp_targets(self, df: pd.DataFrame, max_own: float = 0.12) -> pd.DataFrame:
        """
        Players to target in GPP lineups:
        Model likes them AND public underweights them.
        """
        if "proj_ownership" not in df.columns:
            return df.head(10)
        return df[df["proj_ownership"] <= max_own].sort_values(
            "proj_sg_total", ascending=False
        ).head(15)

    def cash_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Safe plays for cash games (50/50s, double-ups).
        High floor, consistent SG performers, likely make cut.
        """
        return df[df["make_cut_prob"] >= 0.75].sort_values(
            "dk_value" if "dk_value" in df.columns else "proj_sg_total",
            ascending=False
        ).head(12)
