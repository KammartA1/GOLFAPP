"""
Hole-Level Probability Distribution Model — v8.0
The technical heart of prop modeling.

Research-backed approach (DataGolf methodology):
  1. Establish round-level SG distribution (mean + player-specific variance)
  2. Convert round-level to hole-level score probabilities
  3. Derive prop-specific projections from hole distributions
  4. Simulate thousands of rounds for any prop market

This replaces the simple baseline + sensitivity approach with a
proper probability distribution model.

Key conversion benchmarks:
  - PGA Tour average: ~3.63 birdies per round
  - Each +1 SG:Total ≈ scoring 1 stroke better than field
  - A player at +2.0 SG:Total on par-72 ≈ expected score of ~70
"""
import math
import logging
import numpy as np
from scipy import stats
from typing import Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# HOLE DIFFICULTY BASELINES
# Average score distribution by hole type (PGA Tour)
# ─────────────────────────────────────────────

# P(score) on each hole type for TOUR AVERAGE player
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

# Standard 18-hole par mix
DEFAULT_PAR_MIX = {"par3": 4, "par4": 10, "par5": 4}  # Standard par 72

# Score values relative to par
SCORE_VALUES = {
    "eagle_or_better": -2,
    "birdie": -1,
    "par": 0,
    "bogey": 1,
    "double_or_worse": 2,
}


def adjust_hole_dist_for_sg(
    base_dist: dict,
    sg_total: float,
    hole_type: str = "par4",
) -> dict:
    """
    Adjust hole-level score distribution based on player's SG.

    A +1.0 SG:Total player makes ~15% more birdies and ~15% fewer bogeys
    than a tour-average player.
    """
    # SG sensitivity by score type (empirically calibrated)
    # Positive SG → more birdies/eagles, fewer bogeys
    sensitivity = {
        "eagle_or_better": 0.008,
        "birdie": 0.035,
        "par": -0.005,
        "bogey": -0.030,
        "double_or_worse": -0.008,
    }

    # Par 5s are more sensitive to skill (more scoring variance)
    if hole_type == "par5":
        sensitivity["birdie"] = 0.050
        sensitivity["eagle_or_better"] = 0.015
        sensitivity["bogey"] = -0.035

    adjusted = {}
    for score_type, base_prob in base_dist.items():
        adj = base_prob + sensitivity.get(score_type, 0) * sg_total
        adjusted[score_type] = max(0.001, min(0.999, adj))

    # Normalize to sum to 1.0
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
    """
    Compute expected score distribution for a full round.

    Returns dict with:
      - expected_score: expected total strokes
      - expected_birdies: expected birdie count
      - expected_bogeys: expected bogey count
      - expected_bogey_free_holes: expected holes without bogey
      - score_std: round-level standard deviation
      - hole_distributions: per-hole-type probability dicts
    """
    if par_mix is None:
        par_mix = DEFAULT_PAR_MIX

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
        adj_dist = adjust_hole_dist_for_sg(base_dist, sg_total, hole_type)
        hole_dists[hole_type] = adj_dist

        # Accumulate per count of this hole type
        total_expected_eagles += adj_dist["eagle_or_better"] * count
        total_expected_birdies += adj_dist["birdie"] * count
        total_expected_pars += adj_dist["par"] * count
        total_expected_bogeys += adj_dist["bogey"] * count
        total_expected_doubles += adj_dist["double_or_worse"] * count

        # Expected score relative to par
        for score_type, prob in adj_dist.items():
            total_expected_score += prob * SCORE_VALUES[score_type] * count

        # Bogey-free probability (per hole)
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
    """
    Monte Carlo simulation of rounds for deriving any prop probability.

    Returns dict with:
      - score_distribution: histogram of total scores
      - birdie_distribution: histogram of birdie counts
      - bogey_free_distribution: histogram of bogey-free hole counts
      - percentiles: {10th, 25th, 50th, 75th, 90th} for each stat
    """
    if par_mix is None:
        par_mix = DEFAULT_PAR_MIX

    rng = np.random.default_rng()

    scores = np.zeros(n_simulations)
    birdies = np.zeros(n_simulations)
    bogey_free = np.zeros(n_simulations)

    for hole_type, count in par_mix.items():
        base_dist = HOLE_SCORE_DIST.get(hole_type, HOLE_SCORE_DIST["par4"])
        adj_dist = adjust_hole_dist_for_sg(base_dist, sg_total, hole_type)

        # Build CDF for sampling
        outcomes = list(adj_dist.keys())
        probs = [adj_dist[o] for o in outcomes]

        for _ in range(count):
            # Sample outcomes for all simulations at once
            samples = rng.choice(len(outcomes), size=n_simulations, p=probs)

            for i, outcome_idx in enumerate(samples):
                outcome = outcomes[outcome_idx]
                scores[i] += SCORE_VALUES[outcome]
                if outcome in ("birdie", "eagle_or_better"):
                    birdies[i] += 1
                if outcome not in ("bogey", "double_or_worse"):
                    bogey_free[i] += 1

    # Add player-specific noise to scores
    noise = rng.normal(0, player_variance * 0.3, n_simulations)
    scores += noise

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
    """
    Calculate over/under probability for any PrizePicks prop type
    using hole-level simulation.

    prop_type: "birdies", "strokes", "bogey_free_holes", "fantasy_score", etc.
    line_value: the PrizePicks line (e.g., 3.5 for birdies)

    Returns: {prob_over, prob_under, projection, std}
    """
    sim = simulate_round(sg_total, player_variance, par_mix, course_par, n_simulations)

    if prop_type in ("birdies", "birdies_or_better"):
        stat = sim["birdies"]
    elif prop_type in ("strokes", "strokes_total"):
        stat = sim["scores"]
    elif prop_type in ("bogey_free_holes", "bogey_free"):
        stat = sim["bogey_free_holes"]
    elif prop_type == "fantasy_score":
        # DK fantasy: approximate from score distribution
        # Each birdie ≈ 3 pts, par ≈ 0.5, bogey ≈ -0.5
        round_dist = compute_round_score_distribution(
            sg_total, player_variance=player_variance,
            par_mix=par_mix, course_par=course_par,
        )
        proj = (round_dist["expected_birdies_or_better"] * 3 +
                round_dist["expected_eagles"] * 5 +
                round_dist["expected_pars"] * 0.5 +
                round_dist["expected_bogeys"] * -0.5 +
                round_dist["expected_double_or_worse"] * -1.0)
        std = player_variance * 2.5  # Fantasy points have higher variance
        z = (line_value - proj) / std if std > 0 else 0
        return {
            "prob_over": round(float(1 - stats.norm.cdf(z)), 4),
            "prob_under": round(float(stats.norm.cdf(z)), 4),
            "projection": round(proj, 2),
            "std": round(std, 2),
            "method": "hole_level_simulation",
        }
    else:
        # Fallback to normal approximation for unknown types
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

    # For strokes, OVER means higher score (worse), so the edge logic is inverted
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
