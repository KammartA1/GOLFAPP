"""
Hole-Level Probability Distribution Model — v9.0

v9.0 fixes:
  1. Fully vectorized Monte Carlo (no Python inner loop)
  2. Removed post-hoc noise injection that was inconsistent with hole outcomes
  3. Per-hole SG sensitivity scaled properly (not * 18)
  4. Course-specific par mix support
"""
import logging
import numpy as np
from scipy import stats
from typing import Optional

log = logging.getLogger(__name__)

HOLE_SCORE_DIST = {
    "par3": {
        "eagle_or_better": 0.003,
        "birdie": 0.120,
        "par": 0.680,
        "bogey": 0.160,
        "double_or_worse": 0.037,
    },
    "par4": {
        "eagle_or_better": 0.005,
        "birdie": 0.100,
        "par": 0.640,
        "bogey": 0.195,
        "double_or_worse": 0.060,
    },
    "par5": {
        "eagle_or_better": 0.035,
        "birdie": 0.280,
        "par": 0.540,
        "bogey": 0.115,
        "double_or_worse": 0.030,
    },
}

DEFAULT_PAR_MIX = {"par3": 4, "par4": 10, "par5": 4}

SCORE_VALUES = {
    "eagle_or_better": -2,
    "birdie": -1,
    "par": 0,
    "bogey": 1,
    "double_or_worse": 2,
}


def adjust_hole_dist_for_sg(
    base_dist: dict,
    sg_per_hole: float,
    hole_type: str = "par4",
) -> dict:
    """Adjust hole-level score distribution based on player's SG PER HOLE.

    v9.0: sg_per_hole is already divided by 18. Sensitivity coefficients
    are calibrated to per-hole scale. +1 SG:Total per round = +0.055 per hole.
    Each +0.055 SG/hole ≈ +3.5% birdie rate increase.
    """
    # Per-hole sensitivities (calibrated to sg_per_hole scale)
    sensitivity = {
        "eagle_or_better": 0.15,
        "birdie": 0.65,
        "par": -0.10,
        "bogey": -0.55,
        "double_or_worse": -0.15,
    }

    if hole_type == "par5":
        sensitivity["birdie"] = 0.90
        sensitivity["eagle_or_better"] = 0.30
        sensitivity["bogey"] = -0.65

    adjusted = {}
    for score_type, base_prob in base_dist.items():
        adj = base_prob + sensitivity.get(score_type, 0) * sg_per_hole
        adjusted[score_type] = max(0.001, min(0.999, adj))

    total = sum(adjusted.values())
    adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted


def compute_round_score_distribution(
    sg_total: float,
    sg_ott: float = 0,
    sg_app: float = 0,
    sg_atg: float = 0,
    sg_putt: float = 0,
    par_mix: dict = None,
    player_variance: float = 2.75,
    course_par: int = 72,
) -> dict:
    if par_mix is None:
        par_mix = DEFAULT_PAR_MIX

    sg_per_hole = sg_total / 18.0

    total_expected_birdies = 0.0
    total_expected_eagles = 0.0
    total_expected_pars = 0.0
    total_expected_bogeys = 0.0
    total_expected_doubles = 0.0
    total_expected_score = 0.0
    total_bogey_free_prob = 1.0

    hole_dists = {}

    for hole_type, count in par_mix.items():
        base_dist = HOLE_SCORE_DIST.get(hole_type, HOLE_SCORE_DIST["par4"])
        adj_dist = adjust_hole_dist_for_sg(base_dist, sg_per_hole, hole_type)
        hole_dists[hole_type] = adj_dist

        total_expected_eagles += adj_dist["eagle_or_better"] * count
        total_expected_birdies += adj_dist["birdie"] * count
        total_expected_pars += adj_dist["par"] * count
        total_expected_bogeys += adj_dist["bogey"] * count
        total_expected_doubles += adj_dist["double_or_worse"] * count

        for score_type, prob in adj_dist.items():
            total_expected_score += prob * SCORE_VALUES[score_type] * count

        bogey_free_per_hole = 1.0 - adj_dist["bogey"] - adj_dist["double_or_worse"]
        total_bogey_free_prob *= bogey_free_per_hole ** count

    expected_strokes = course_par + total_expected_score
    expected_birdies_or_better = total_expected_birdies + total_expected_eagles

    return {
        "expected_score": round(expected_strokes, 2),
        "score_vs_par": round(total_expected_score, 2),
        "expected_birdies": round(total_expected_birdies, 2),
        "expected_birdies_or_better": round(expected_birdies_or_better, 2),
        "expected_eagles": round(total_expected_eagles, 3),
        "expected_pars": round(total_expected_pars, 2),
        "expected_bogeys": round(total_expected_bogeys, 2),
        "expected_double_or_worse": round(total_expected_doubles, 2),
        "expected_bogey_free_holes": round(18 - total_expected_bogeys - total_expected_doubles, 2),
        "bogey_free_round_prob": round(total_bogey_free_prob, 4),
        "score_std": round(player_variance, 2),
        "hole_distributions": hole_dists,
    }


def simulate_round(
    sg_total: float,
    player_variance: float = 2.75,
    par_mix: dict = None,
    course_par: int = 72,
    n_simulations: int = 10000,
) -> dict:
    """Fully vectorized Monte Carlo simulation."""
    if par_mix is None:
        par_mix = DEFAULT_PAR_MIX

    rng = np.random.default_rng()
    sg_per_hole = sg_total / 18.0

    scores = np.zeros(n_simulations)
    birdies = np.zeros(n_simulations)
    bogey_free = np.full(n_simulations, 18.0)

    for hole_type, count in par_mix.items():
        base_dist = HOLE_SCORE_DIST.get(hole_type, HOLE_SCORE_DIST["par4"])
        adj_dist = adjust_hole_dist_for_sg(base_dist, sg_per_hole, hole_type)

        outcomes = list(adj_dist.keys())
        probs = np.array([adj_dist[o] for o in outcomes])
        score_vals = np.array([SCORE_VALUES[o] for o in outcomes])
        is_birdie = np.array([1.0 if o in ("birdie", "eagle_or_better") else 0.0 for o in outcomes])
        is_bogey_plus = np.array([1.0 if o in ("bogey", "double_or_worse") else 0.0 for o in outcomes])

        for _ in range(count):
            samples = rng.choice(len(outcomes), size=n_simulations, p=probs)
            scores += score_vals[samples]
            birdies += is_birdie[samples]
            bogey_free -= is_bogey_plus[samples]

    total_scores = course_par + scores

    return {
        "scores": {
            "mean": round(float(np.mean(total_scores)), 2),
            "std": round(float(np.std(total_scores)), 2),
            "p10": round(float(np.percentile(total_scores, 10)), 1),
            "p25": round(float(np.percentile(total_scores, 25)), 1),
            "p50": round(float(np.percentile(total_scores, 50)), 1),
            "p75": round(float(np.percentile(total_scores, 75)), 1),
            "p90": round(float(np.percentile(total_scores, 90)), 1),
        },
        "birdies": {
            "mean": round(float(np.mean(birdies)), 2),
            "std": round(float(np.std(birdies)), 2),
            "p10": round(float(np.percentile(birdies, 10)), 1),
            "p25": round(float(np.percentile(birdies, 25)), 1),
            "p50": round(float(np.percentile(birdies, 50)), 1),
            "p75": round(float(np.percentile(birdies, 75)), 1),
            "p90": round(float(np.percentile(birdies, 90)), 1),
        },
        "bogey_free_holes": {
            "mean": round(float(np.mean(bogey_free)), 2),
            "std": round(float(np.std(bogey_free)), 2),
            "p10": round(float(np.percentile(bogey_free, 10)), 1),
            "p50": round(float(np.percentile(bogey_free, 50)), 1),
            "p90": round(float(np.percentile(bogey_free, 90)), 1),
        },
        "n_simulations": n_simulations,
    }


def prop_probability(
    prop_type: str,
    line_value: float,
    sg_total: float,
    player_variance: float = 2.75,
    par_mix: dict = None,
    course_par: int = 72,
    n_simulations: int = 10000,
) -> dict:
    sim = simulate_round(sg_total, player_variance, par_mix, course_par, n_simulations)

    if prop_type in ("birdies", "birdies_or_better"):
        stat = sim["birdies"]
    elif prop_type in ("strokes", "strokes_total"):
        stat = sim["scores"]
    elif prop_type in ("bogey_free_holes", "bogey_free"):
        stat = sim["bogey_free_holes"]
    elif prop_type == "fantasy_score":
        round_dist = compute_round_score_distribution(
            sg_total, player_variance=player_variance,
            par_mix=par_mix, course_par=course_par,
        )
        proj = (round_dist["expected_birdies_or_better"] * 3 +
                round_dist["expected_eagles"] * 5 +
                round_dist["expected_pars"] * 0.5 +
                round_dist["expected_bogeys"] * -0.5 +
                round_dist["expected_double_or_worse"] * -1.0)
        std = player_variance * 2.5
        z = (line_value - proj) / std if std > 0 else 0
        return {
            "prob_over": round(float(1 - stats.norm.cdf(z)), 4),
            "prob_under": round(float(stats.norm.cdf(z)), 4),
            "projection": round(proj, 2),
            "std": round(std, 2),
            "method": "hole_level_simulation",
        }
    else:
        return {
            "prob_over": 0.5, "prob_under": 0.5,
            "projection": line_value, "std": 0,
            "method": "fallback",
        }

    projection = stat["mean"]
    std = stat["std"]

    if std <= 0:
        return {
            "prob_over": 0.5, "prob_under": 0.5,
            "projection": projection, "std": 0,
            "method": "hole_level_simulation",
        }

    z = (line_value - projection) / std
    prob_over = float(1 - stats.norm.cdf(z))
    prob_under = float(stats.norm.cdf(z))

    return {
        "prob_over": round(prob_over, 4),
        "prob_under": round(prob_under, 4),
        "projection": round(projection, 2),
        "std": round(std, 2),
        "method": "hole_level_simulation",
    }
