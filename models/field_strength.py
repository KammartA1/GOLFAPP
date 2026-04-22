"""
Field Strength Normalization — v9.0 (R10)

A +2.0 SG performance at the Bermuda Championship (weak field) is NOT
the same as +2.0 SG at the Masters (elite field). This module normalizes
historical SG by field strength.

Also computes field-strength-adjusted win/placement probabilities.
"""
import numpy as np
from typing import Optional


# Known field strength tiers (average field SG vs Tour mean)
FIELD_STRENGTH_TIERS = {
    "major": 0.45,             # Majors: field is ~0.45 SG above Tour mean
    "signature": 0.35,         # Signature events (elevated purse)
    "playoff": 0.50,           # FedExCup playoffs (top 70/50/30)
    "invitational": 0.25,      # Strong invitational fields
    "regular": 0.0,            # Average field
    "alternate": -0.30,        # Opposite-field / weak-field events
}


def estimate_field_strength(field_sg_values: list[float]) -> dict:
    """Estimate field strength from the SG distribution of the field.

    Returns:
        {
            field_mean_sg: average SG of the field,
            field_std_sg: standard deviation,
            field_tier: estimated tier name,
            adjustment: SG adjustment to normalize to average field,
        }
    """
    if not field_sg_values:
        return {
            "field_mean_sg": 0.0,
            "field_std_sg": 2.75,
            "field_tier": "regular",
            "adjustment": 0.0,
        }

    arr = np.array(field_sg_values)
    mean_sg = float(np.mean(arr))
    std_sg = float(np.std(arr))

    # Classify tier
    if mean_sg > 0.40:
        tier = "playoff"
    elif mean_sg > 0.30:
        tier = "major"
    elif mean_sg > 0.20:
        tier = "signature"
    elif mean_sg > 0.05:
        tier = "invitational"
    elif mean_sg > -0.15:
        tier = "regular"
    else:
        tier = "alternate"

    # Adjustment: how much to shift SG to normalize to "regular" field
    adjustment = -mean_sg * 0.15

    return {
        "field_mean_sg": round(mean_sg, 3),
        "field_std_sg": round(std_sg, 3),
        "field_tier": tier,
        "adjustment": round(adjustment, 3),
    }


def normalize_historical_sg(
    sg_total: float,
    field_strength: float,
) -> float:
    """Normalize a historical SG value by field strength.

    A +1.0 SG in a weak field (-0.3 mean) is worth less than
    +1.0 SG in a strong field (+0.4 mean).

    The normalization adds a fraction of the field mean to the raw SG,
    so performances in strong fields get boosted.
    """
    return sg_total + field_strength * 0.20


def adjusted_win_probability(
    raw_win_prob: float,
    field_tier: str,
) -> float:
    """Adjust win probability based on field strength.

    In a weak field, the same SG gives higher win probability.
    In a strong field, it gives lower.
    """
    tier_multipliers = {
        "major": 0.70,       # Harder to win with same SG
        "signature": 0.80,
        "playoff": 0.65,
        "invitational": 0.90,
        "regular": 1.00,
        "alternate": 1.30,   # Easier to win with same SG
    }

    mult = tier_multipliers.get(field_tier, 1.0)
    adjusted = raw_win_prob * mult

    # Cap at 30% (no golfer should be >30% to win any event)
    return round(min(adjusted, 0.30), 4)
