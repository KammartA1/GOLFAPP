"""
Player Fingerprint System — v9.0 (R2)

Creates multi-dimensional vectors for each player's strengths,
then computes cosine similarity against course requirement vectors.

Fingerprint dimensions:
  1. Shot shape tendency (draw/fade/straight)
  2. Ball flight trajectory (high/low)
  3. Distance band excellence (proximity by distance range)
  4. Par-3/4/5 scoring splits
  5. Bermuda vs bentgrass putting differential
  6. Pressure response coefficient
  7. Wind performance split
  8. Scrambling ability
"""
import numpy as np
from typing import Optional


def build_player_fingerprint(player_proj: dict, player_history: list[dict] = None) -> dict:
    """Build a normalized fingerprint vector for a player.

    Args:
        player_proj: Projection dict with SG components
        player_history: Optional historical event data for deeper signals

    Returns:
        dict with fingerprint dimensions (each normalized 0-1)
    """
    sg_ott = float(player_proj.get("proj_sg_ott", player_proj.get("sg_ott", 0)) or 0)
    sg_app = float(player_proj.get("proj_sg_app", player_proj.get("sg_app", 0)) or 0)
    sg_atg = float(player_proj.get("proj_sg_atg", player_proj.get("sg_atg", 0)) or 0)
    sg_putt = float(player_proj.get("proj_sg_putt", player_proj.get("sg_putt", 0)) or 0)
    sg_total = sg_ott + sg_app + sg_atg + sg_putt
    variance = float(player_proj.get("player_variance", 2.75))

    # Normalize SG values to 0-1 scale (clamp to [-3, +3] range)
    def _norm_sg(val, low=-2.0, high=2.0):
        return max(0, min(1, (val - low) / (high - low)))

    fingerprint = {
        # Core skill dimensions
        "power": _norm_sg(sg_ott, -1.5, 1.5),
        "approach_precision": _norm_sg(sg_app, -1.5, 1.5),
        "short_game": _norm_sg(sg_atg, -1.5, 1.5),
        "putting": _norm_sg(sg_putt, -1.5, 1.5),

        # Derived dimensions
        "ball_striking": _norm_sg((sg_ott + sg_app) / 2, -1.5, 1.5),
        "scoring_ability": _norm_sg((sg_app + sg_putt) / 2, -1.5, 1.5),
        "scrambling": _norm_sg((sg_atg * 0.7 + sg_putt * 0.3), -1.0, 1.0),

        # Volatility profile
        "consistency": 1.0 - min(1.0, max(0, (variance - 1.5) / 3.0)),
        "upside": min(1.0, max(0, (variance - 2.0) / 2.0)),

        # Overall strength
        "overall": _norm_sg(sg_total, -3.0, 3.0),
    }

    # If we have historical data, compute deeper signals
    if player_history and len(player_history) >= 5:
        _enrich_from_history(fingerprint, player_history)

    return fingerprint


def _enrich_from_history(fingerprint: dict, history: list[dict]):
    """Add historical signals to the fingerprint."""
    # Trend: is player improving or declining?
    if len(history) >= 8:
        recent_4 = [h.get("sg_total", 0) for h in history[-4:] if h.get("sg_total") is not None]
        older_4 = [h.get("sg_total", 0) for h in history[-8:-4] if h.get("sg_total") is not None]
        if recent_4 and older_4:
            trend = np.mean(recent_4) - np.mean(older_4)
            fingerprint["form_trend"] = max(0, min(1, (trend + 1.0) / 2.0))

    # Wind performance (if tagged)
    windy = [h.get("sg_total", 0) for h in history if h.get("wind_speed", 0) > 15 and h.get("sg_total") is not None]
    calm = [h.get("sg_total", 0) for h in history if h.get("wind_speed", 0) <= 10 and h.get("sg_total") is not None]
    if len(windy) >= 3 and len(calm) >= 3:
        wind_diff = np.mean(windy) - np.mean(calm)
        fingerprint["wind_skill"] = max(0, min(1, (wind_diff + 1.0) / 2.0))


def build_course_requirement_vector(course_profile: dict) -> dict:
    """Build a requirement vector from a course profile.

    Maps course attributes to the same dimensions as player fingerprint.
    """
    if not course_profile:
        return _default_course_vector()

    weights = course_profile.get("sg_weights", {})
    dist_bonus = course_profile.get("distance_bonus", 0.5)
    acc_penalty = course_profile.get("accuracy_penalty", 0.5)
    wind_sens = course_profile.get("wind_sensitivity", 0.5)

    return {
        "power": dist_bonus,
        "approach_precision": weights.get("sg_app", 0.38),
        "short_game": weights.get("sg_atg", 0.22) + acc_penalty * 0.2,
        "putting": weights.get("sg_putt", 0.15),
        "ball_striking": (dist_bonus + weights.get("sg_app", 0.38)) / 2,
        "scoring_ability": weights.get("sg_app", 0.38) * 0.6 + weights.get("sg_putt", 0.15) * 0.4,
        "scrambling": weights.get("sg_atg", 0.22) * 0.8 + acc_penalty * 0.2,
        "consistency": 1.0 - dist_bonus * 0.3,  # Bomber courses reward variance
        "upside": dist_bonus * 0.4,
        "overall": 0.5,
        "wind_skill": wind_sens,
    }


def _default_course_vector() -> dict:
    return {
        "power": 0.5, "approach_precision": 0.5, "short_game": 0.5,
        "putting": 0.5, "ball_striking": 0.5, "scoring_ability": 0.5,
        "scrambling": 0.5, "consistency": 0.5, "upside": 0.5,
        "overall": 0.5, "wind_skill": 0.5,
    }


def course_fit_cosine(player_fp: dict, course_req: dict) -> float:
    """Compute cosine similarity between player fingerprint and course requirements.

    Returns 0-100 score where 100 = perfect fit.
    """
    # Aligned dimensions
    dims = sorted(set(player_fp.keys()) & set(course_req.keys()))
    if not dims:
        return 50.0

    p_vec = np.array([player_fp[d] for d in dims])
    c_vec = np.array([course_req[d] for d in dims])

    # Cosine similarity
    dot = np.dot(p_vec, c_vec)
    norm_p = np.linalg.norm(p_vec)
    norm_c = np.linalg.norm(c_vec)

    if norm_p == 0 or norm_c == 0:
        return 50.0

    cosine_sim = dot / (norm_p * norm_c)

    # Also consider magnitude alignment (a strong player with matching profile)
    magnitude_bonus = (np.mean(p_vec) - 0.5) * 20  # +10 for strong, -10 for weak

    score = cosine_sim * 50 + 50 + magnitude_bonus
    return round(max(0, min(100, score)), 1)


def compute_field_fingerprints(
    projections: list[dict],
    course_profile: dict,
    histories: dict = None,
) -> list[dict]:
    """Compute fingerprints and course fit for entire field.

    Returns list of dicts with fingerprint + course_fit_cosine for each player.
    """
    course_vec = build_course_requirement_vector(course_profile)
    results = []

    for proj in projections:
        name = proj.get("name", "Unknown")
        history = (histories or {}).get(name, [])
        fp = build_player_fingerprint(proj, history)
        fit = course_fit_cosine(fp, course_vec)

        results.append({
            "name": name,
            "fingerprint": fp,
            "course_fit_cosine": fit,
            "power_score": round(fp.get("power", 0.5) * 100, 1),
            "precision_score": round(fp.get("approach_precision", 0.5) * 100, 1),
            "short_game_score": round(fp.get("short_game", 0.5) * 100, 1),
            "putting_score": round(fp.get("putting", 0.5) * 100, 1),
            "consistency_score": round(fp.get("consistency", 0.5) * 100, 1),
        })

    return results
