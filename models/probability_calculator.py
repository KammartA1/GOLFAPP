"""
Golf Quant Engine — Probability Calculator — v8.0
Calculates over/under probabilities and edges for PrizePicks lines.

v8.0: Integrates hole-level probability distributions for birdies,
bogey-free holes, and strokes props (DataGolf methodology).
Falls back to baseline+sensitivity for stat types without hole-level model.
"""
import math
import logging
from scipy import stats

log = logging.getLogger(__name__)

# Try to import hole-level model
try:
    from models.hole_level import prop_probability as hole_level_prop
    HOLE_LEVEL_AVAILABLE = True
except ImportError:
    HOLE_LEVEL_AVAILABLE = False
    log.warning("Hole-level model not available — using baseline+sensitivity")

# PrizePicks Fantasy Score rubric (DraftKings-style for PGA)
PP_FANTASY_SCORING = {
    "eagle": 8,
    "birdie": 3,
    "par": 0.5,
    "bogey": -1,
    "double_bogey": -2,
    "worse": -2,
    "streak_bonus_3": 3,  # 3 consecutive birdies+
}

# Tour average baselines per stat type (per round unless noted)
STAT_BASELINES = {
    "fantasy_score": {"mean": 37.0, "std": 10.0, "per": "tournament"},
    "birdies": {"mean": 3.8, "std": 1.5, "per": "round"},
    "birdies_or_better": {"mean": 3.8, "std": 1.5, "per": "round"},
    "bogey_free_holes": {"mean": 13.5, "std": 2.5, "per": "round"},
    "pars_or_better": {"mean": 14.2, "std": 2.0, "per": "round"},
    "pars": {"mean": 10.5, "std": 2.0, "per": "round"},
    "strokes_total": {"mean": 71.0, "std": 3.0, "per": "round"},
    "holes_under_par": {"mean": 15.0, "std": 4.0, "per": "tournament"},
    "gir": {"mean": 11.5, "std": 2.5, "per": "round"},
    "greens_in_regulation": {"mean": 11.5, "std": 2.5, "per": "round"},
    "fairways_hit": {"mean": 8.5, "std": 3.0, "per": "round"},
    "eagles": {"mean": 0.3, "std": 0.5, "per": "tournament"},
    "longest_drive": {"mean": 305, "std": 12.0, "per": "round"},
    "made_cuts": {"mean": 0.70, "std": 0.15, "per": "tournament"},
    # Matchup props (H2H: difference between two players)
    "birdies_or_better_matchup": {"mean": 0.0, "std": 1.5, "per": "round"},
}

# SG → stat sensitivity coefficients
# v9.0: Use ONLY component sensitivities (NOT sg_total + components, which double-counts)
# sg_total = sg_ott + sg_app + sg_atg + sg_putt, so using both is double-counting
SG_STAT_SENSITIVITY = {
    "fantasy_score": {
        "sg_app": 3.8, "sg_putt": 2.8, "sg_ott": 1.6, "sg_atg": 1.3,
    },
    "birdies": {
        "sg_app": 0.40, "sg_putt": 0.30, "sg_ott": 0.10, "sg_atg": 0.05,
    },
    "bogey_free_holes": {
        "sg_app": 0.35, "sg_putt": 0.25, "sg_ott": 0.10, "sg_atg": 0.10,
    },
    "pars_or_better": {
        "sg_app": 0.30, "sg_putt": 0.20, "sg_ott": 0.10, "sg_atg": 0.10,
    },
    "strokes_total": {
        "sg_app": -1.5, "sg_putt": -1.0, "sg_ott": -0.8, "sg_atg": -0.7,
    },
    "holes_under_par": {
        "sg_app": 1.5, "sg_putt": 1.0, "sg_ott": 0.5, "sg_atg": 0.5,
    },
    "gir": {
        "sg_app": 1.5, "sg_ott": 0.5,
    },
    "fairways_hit": {
        "sg_ott": 1.8,
    },
    "eagles": {
        "sg_ott": 0.10, "sg_app": 0.08,
    },
    "longest_drive": {
        "sg_ott": 5.0,
    },
    "birdies_or_better": {
        "sg_app": 0.40, "sg_putt": 0.30, "sg_ott": 0.10, "sg_atg": 0.05,
    },
    "greens_in_regulation": {
        "sg_app": 1.5, "sg_ott": 0.5,
    },
    "pars": {
        "sg_app": 0.25, "sg_putt": 0.15, "sg_ott": 0.10, "sg_atg": 0.10,
    },
    "birdies_or_better_matchup": {
        "sg_app": 0.40, "sg_putt": 0.30, "sg_ott": 0.10,
    },
}


def project_stat(stat_type: str, player_sg: dict,
                 weather_adj: dict = None) -> tuple:
    """
    Project a specific stat value for a player based on their SG profile.

    v8.0: Uses hole-level probability model for birdies, bogey_free_holes,
    and strokes props. Falls back to baseline+sensitivity for other types.

    Args:
        stat_type: e.g. "fantasy_score", "birdies"
        player_sg: dict with keys like sg_total, sg_app, sg_putt, sg_ott, sg_atg
        weather_adj: dict with variance_mult, projection_mult from weather

    Returns: (projection, std_dev, weather_adj_projection, weather_adj_std)
    """
    # v8.0: Use hole-level model for supported prop types
    hole_level_types = {"birdies", "birdies_or_better", "bogey_free_holes",
                        "strokes_total", "fantasy_score"}

    if HOLE_LEVEL_AVAILABLE and stat_type in hole_level_types:
        sg_total = float(player_sg.get("sg_total", player_sg.get("proj_sg_total", 0)) or 0)
        player_variance = float(player_sg.get("player_variance", 2.75))

        # Use hole-level simulation for projection
        try:
            result = hole_level_prop(
                prop_type=stat_type,
                line_value=0,  # We just want the projection, not probability vs a line
                sg_total=sg_total,
                player_variance=player_variance,
                n_simulations=5000,  # Reduced for speed in bulk analysis
            )
            projection = result["projection"]
            std = result["std"]

            # Weather adjustments
            w_proj = projection
            w_std = std
            if weather_adj:
                w_proj *= weather_adj.get("projection_mult", 1.0)
                w_std *= weather_adj.get("variance_mult", 1.0)

            return round(projection, 2), round(std, 2), round(w_proj, 2), round(w_std, 2)
        except Exception as e:
            log.warning(f"Hole-level model failed for {stat_type}: {e}")
            # Fall through to baseline+sensitivity

    # Baseline + sensitivity approach (original method, for non-hole-level types)
    baseline_info = STAT_BASELINES.get(stat_type)
    if not baseline_info:
        log.warning(f"Unknown stat type: {stat_type}")
        return None, None, None, None

    baseline = baseline_info["mean"]
    std = baseline_info["std"]

    # Calculate SG contribution
    sensitivity = SG_STAT_SENSITIVITY.get(stat_type, {"sg_total": 1.0})
    sg_contrib = 0.0
    for sg_cat, sens in sensitivity.items():
        sg_val = player_sg.get(sg_cat, player_sg.get(f"proj_{sg_cat}", 0)) or 0
        sg_contrib += float(sg_val) * sens

    projection = baseline + sg_contrib

    # Weather adjustments
    w_proj = projection
    w_std = std
    if weather_adj:
        w_proj *= weather_adj.get("projection_mult", 1.0)
        w_std *= weather_adj.get("variance_mult", 1.0)

    return round(projection, 2), round(std, 2), round(w_proj, 2), round(w_std, 2)


def calc_probability(projection: float, std: float, line: float) -> dict:
    """
    Calculate over/under probability using normal distribution.

    Returns dict with: prob_over, prob_under, z_score, edge_over, edge_under
    """
    if std <= 0 or projection is None:
        return {
            "prob_over": 0.5, "prob_under": 0.5,
            "z_score": 0, "edge_over": 0, "edge_under": 0,
        }

    z = (line - projection) / std
    prob_over = 1 - stats.norm.cdf(z)
    prob_under = stats.norm.cdf(z)

    # PrizePicks breakeven varies: 2-pick power play ~57.7%, standard ~53.1%
    breakeven = 0.531
    edge_over = prob_over - breakeven
    edge_under = prob_under - breakeven

    return {
        "prob_over": round(float(prob_over), 4),
        "prob_under": round(float(prob_under), 4),
        "z_score": round(float(z), 3),
        "edge_over": round(float(edge_over), 4),
        "edge_under": round(float(edge_under), 4),
    }


def classify_confidence(edge: float, market_alignment: bool = None,
                        weather_significant: bool = False) -> str:
    """
    Classify confidence level based on edge and market alignment.

    HIGH: edge > 8% AND market aligns
    MEDIUM: edge > 5% AND (market aligns OR no data)
    LOW: edge > 3%
    NO BET: edge < 3%
    """
    abs_edge = abs(edge)

    if abs_edge >= 0.08:
        if market_alignment is True or market_alignment is None:
            return "HIGH"
        return "MEDIUM"
    elif abs_edge >= 0.05:
        if market_alignment is True:
            return "MEDIUM"
        elif market_alignment is None:
            return "MEDIUM"
        return "LOW"
    elif abs_edge >= 0.03:
        return "LOW"
    return "NO_BET"


def kelly_stake(prob: float, odds_decimal: float = 1.87,
                kelly_fraction: float = 0.20, bankroll: float = 1000) -> dict:
    """
    Calculate Kelly criterion bet size.

    PrizePicks is roughly -115 odds = 1.87 decimal.
    """
    if prob <= 0 or prob >= 1 or odds_decimal <= 1:
        return {"stake": 0, "kelly_pct": 0, "ev": 0}

    b = odds_decimal - 1
    q = 1 - prob
    f_star = (b * prob - q) / b
    f_star = max(0, f_star)

    # Apply Kelly fraction
    f = f_star * kelly_fraction
    f = min(f, 0.08)  # Hard cap at 8% of bankroll

    stake = round(bankroll * f, 2)
    ev = round(stake * (odds_decimal * prob - 1), 2)

    return {
        "stake": stake,
        "kelly_pct": round(f * 100, 2),
        "full_kelly_pct": round(f_star * 100, 2),
        "ev": ev,
    }


def analyze_line(player_name: str, stat_type: str, line_value: float,
                 player_sg: dict, weather_adj: dict = None,
                 consensus_prob: float = None, bankroll: float = 1000,
                 kelly_frac: float = 0.25) -> dict:
    """
    Full analysis of a single PrizePicks line.

    Returns complete analysis dict ready for display.
    """
    # Project the stat
    proj, std, w_proj, w_std = project_stat(stat_type, player_sg, weather_adj)

    if proj is None:
        return {
            "player_name": player_name,
            "stat_type": stat_type,
            "line_value": line_value,
            "error": f"Cannot project stat type: {stat_type}",
        }

    # Use weather-adjusted values if available
    eff_proj = w_proj if w_proj else proj
    eff_std = w_std if w_std else std

    # Calculate probabilities
    probs = calc_probability(eff_proj, eff_std, line_value)

    # Determine best side
    best_side = "OVER" if probs["edge_over"] > probs["edge_under"] else "UNDER"
    best_edge = max(probs["edge_over"], probs["edge_under"])
    best_prob = probs["prob_over"] if best_side == "OVER" else probs["prob_under"]

    # Market alignment check
    market_alignment = None
    if consensus_prob is not None:
        # If consensus says player is strong (high win prob) → supports OVER
        # If consensus says player is weak → supports UNDER
        if consensus_prob > 0.05:  # Top player
            market_alignment = (best_side == "OVER")
        elif consensus_prob < 0.01:  # Longshot
            market_alignment = (best_side == "UNDER")

    # Confidence
    confidence = classify_confidence(
        best_edge, market_alignment,
        weather_adj.get("is_significant", False) if weather_adj else False
    )

    # Kelly sizing
    kelly = kelly_stake(best_prob, 1.87, kelly_frac, bankroll)

    # Sanity checks
    warnings = []
    if abs(best_edge) > 0.30:
        warnings.append("EXTREME EDGE — VERIFY MODEL")
    if best_prob > 0.97 or best_prob < 0.03:
        warnings.append("MODEL WARNING — likely data error or stat type mismatch")
    if abs(proj - line_value) / max(line_value, 1) > 2.0:
        warnings.append("SCALE MISMATCH — projection and line on different scales")

    return {
        "player_name": player_name,
        "stat_type": stat_type,
        "line_value": line_value,
        "projection": proj,
        "std_dev": std,
        "weather_projection": w_proj,
        "weather_std": w_std,
        "prob_over": probs["prob_over"],
        "prob_under": probs["prob_under"],
        "z_score": probs["z_score"],
        "edge_over": probs["edge_over"],
        "edge_under": probs["edge_under"],
        "best_side": best_side,
        "best_edge": round(best_edge, 4),
        "best_prob": round(best_prob, 4),
        "confidence": confidence,
        "market_alignment": market_alignment,
        "consensus_prob": consensus_prob,
        "kelly_stake": kelly["stake"],
        "kelly_pct": kelly["kelly_pct"],
        "ev_dollars": kelly["ev"],
        "warnings": warnings,
        "weather_impact": weather_adj.get("description", "") if weather_adj else "",
    }
