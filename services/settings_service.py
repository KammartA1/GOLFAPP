"""
Settings Service — User settings layer
========================================
Read/write user preferences, bankroll, model params, and risk limits.
All settings are stored in the user_settings table as key-value pairs
with JSON-serialised values.
"""
import json
import logging
from datetime import datetime
from typing import Any, Optional

from database.models import UserSetting
from services._db import get_session as _session

log = logging.getLogger(__name__)

# Default settings applied when a key has never been set
_DEFAULTS = {
    "bankroll": 1000.0,
    "kelly_fraction": 0.25,
    "min_edge": 0.03,
    "max_bet_pct": 0.08,
    "max_daily_bets": 10,
    "model_version": "v8.0",
    # Model params
    "putting_regression": 0.55,
    "approach_regression": 0.15,
    "ott_regression": 0.20,
    "atg_regression": 0.35,
    "form_weight_last4": 0.10,
    "form_weight_last12": 0.25,
    "form_weight_last24": 0.30,
    "form_weight_last50": 0.35,
    "cross_cat_ott_to_app": 0.20,
    # Risk limits
    "max_outright_exposure": 0.02,
    "max_h2h_exposure": 0.08,
    "max_gpp_exposure": 0.05,
    "max_drawdown_pct": 0.25,
    "stop_loss_daily": 200.0,
}


def get_setting(user_id: str, key: str, default: Any = None) -> Any:
    """
    Read a single setting. Returns the deserialised value, or ``default``
    if the key has never been set (falling back to _DEFAULTS if default
    is None).
    """
    with _session() as session:
        row = (
            session.query(UserSetting)
            .filter(
                UserSetting.user_id == user_id,
                UserSetting.setting_key == key,
            )
            .first()
        )

        if row and row.setting_value is not None:
            return _deserialize(row.setting_value)

    # Fallback chain: caller default -> global default
    if default is not None:
        return default
    return _DEFAULTS.get(key)


def set_setting(user_id: str, key: str, value: Any) -> None:
    """Write a single setting (upsert)."""
    serialized = _serialize(value)

    with _session() as session:
        row = (
            session.query(UserSetting)
            .filter(
                UserSetting.user_id == user_id,
                UserSetting.setting_key == key,
            )
            .first()
        )

        if row:
            row.setting_value = serialized
        else:
            row = UserSetting(
                user_id=user_id,
                setting_key=key,
                setting_value=serialized,
            )
            session.add(row)

    log.debug("Set %s.%s = %s", user_id, key, serialized[:80])


def get_all_settings(user_id: str) -> dict:
    """
    Return all settings for a user as a flat dict, merged with defaults
    so every expected key is present.
    """
    result = dict(_DEFAULTS)  # start with defaults

    with _session() as session:
        rows = (
            session.query(UserSetting)
            .filter(UserSetting.user_id == user_id)
            .all()
        )
        for row in rows:
            if row.setting_value is not None:
                result[row.setting_key] = _deserialize(row.setting_value)

    return result


def get_bankroll(user_id: str) -> float:
    """Convenience: read the bankroll setting."""
    val = get_setting(user_id, "bankroll", _DEFAULTS["bankroll"])
    return float(val)


def set_bankroll(user_id: str, amount: float) -> None:
    """Convenience: write the bankroll setting."""
    set_setting(user_id, "bankroll", amount)


def get_model_params(user_id: str) -> dict:
    """
    Return model-tuning parameters for this user.

    Keys: putting_regression, approach_regression, ott_regression,
    atg_regression, form_weight_last4, form_weight_last12,
    form_weight_last24, form_weight_last50, cross_cat_ott_to_app,
    model_version.
    """
    all_settings = get_all_settings(user_id)

    param_keys = [
        "putting_regression", "approach_regression",
        "ott_regression", "atg_regression",
        "form_weight_last4", "form_weight_last12",
        "form_weight_last24", "form_weight_last50",
        "cross_cat_ott_to_app", "model_version",
    ]

    return {k: all_settings.get(k, _DEFAULTS.get(k)) for k in param_keys}


def get_risk_limits(user_id: str) -> dict:
    """
    Return risk management limits for this user.

    Keys: max_outright_exposure, max_h2h_exposure, max_gpp_exposure,
    max_drawdown_pct, stop_loss_daily, max_bet_pct, max_daily_bets,
    kelly_fraction, min_edge.
    """
    all_settings = get_all_settings(user_id)

    risk_keys = [
        "max_outright_exposure", "max_h2h_exposure", "max_gpp_exposure",
        "max_drawdown_pct", "stop_loss_daily", "max_bet_pct",
        "max_daily_bets", "kelly_fraction", "min_edge",
    ]

    return {k: all_settings.get(k, _DEFAULTS.get(k)) for k in risk_keys}


# ── serialisation helpers ──────────────────────────────────────────

def _serialize(value: Any) -> str:
    """Serialise a Python value to a string for storage."""
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _deserialize(raw: str) -> Any:
    """Deserialise a stored string back to a Python value."""
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
