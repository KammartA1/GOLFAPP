"""
DataGolf API Integration
STATUS: STUBBED — activate by setting DATAGOLF_API_KEY in .env

DataGolf is the Bloomberg terminal of golf analytics (~$30/mo).
This module is fully implemented and ready to use when you subscribe.

What you unlock with DataGolf:
  - Pre-tournament skill ratings (their proprietary SG model)
  - Course fit adjustments (ML-based, not manual like our config)
  - Accurate ownership projections for DFS
  - Live in-tournament model updates
  - Historical raw data going back years
  - Field quality / strength of field ratings

All methods return the same schema as our free scrapers —
the rest of the engine works identically with or without DataGolf.
"""
import logging
import requests
from config.settings import DATAGOLF_API_KEY, DATAGOLF_BASE, DATAGOLF_ENDPOINTS, DATAGOLF_ENABLED

log = logging.getLogger(__name__)


class DataGolfClient:
    """
    DataGolf API client.
    Fully stubbed until DATAGOLF_API_KEY is set.
    All methods return empty/mock data when not activated.
    """

    def __init__(self):
        self.api_key = DATAGOLF_API_KEY
        self.enabled = DATAGOLF_ENABLED
        self.session = requests.Session()
        if self.enabled:
            log.info("✅ DataGolf API ACTIVE")
        else:
            log.info("🔶 DataGolf API STUBBED — set DATAGOLF_API_KEY to activate")

    def _get(self, endpoint: str, params: dict = None) -> dict:
        if not self.enabled:
            log.debug(f"DataGolf stubbed: {endpoint}")
            return {}
        url = f"{DATAGOLF_BASE}{endpoint}"
        params = params or {}
        params["key"] = self.api_key
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.error(f"DataGolf API error: {e}")
            return {}

    # ── RANKINGS ──────────────────────────────────────────────────

    def get_rankings(self) -> list[dict]:
        """
        DataGolf's proprietary player skill rankings.
        Better than OWGR for predicting performance.

        Returns: [{"player_name", "dg_id", "dg_rank", "sg_total", "sg_ott", "sg_app", "sg_atg", "sg_putt"}, ...]
        """
        data = self._get(DATAGOLF_ENDPOINTS["rankings"])
        if not data:
            return []
        rankings = data.get("rankings", [])
        return [{
            "name": r.get("player_name"),
            "dg_id": r.get("dg_id"),
            "dg_rank": r.get("datagolf_rank"),
            "sg_total": r.get("sg_total"),
            "sg_ott": r.get("sg_ott"),
            "sg_app": r.get("sg_app"),
            "sg_atg": r.get("sg_atg"),
            "sg_putt": r.get("sg_putt"),
        } for r in rankings]

    # ── PRE-TOURNAMENT PREDICTIONS ────────────────────────────────

    def get_predictions(self, tour: str = "pga", add_course_fit: bool = True,
                        odds_format: str = "percent") -> list[dict]:
        """
        Pre-tournament win/top-5/top-10/top-20 probabilities WITH course fit.
        This is DataGolf's crown jewel — their model is excellent.

        Returns: [{"player_name", "win", "top_5", "top_10", "top_20", "make_cut",
                   "sg_total", "course_fit_score"}, ...]
        """
        params = {
            "tour": tour,
            "add_course_fit": "yes" if add_course_fit else "no",
            "odds_format": odds_format,
        }
        data = self._get(DATAGOLF_ENDPOINTS["predictions"], params)
        if not data:
            return []

        players = []
        for p in data.get("baseline_history_fit", []):
            players.append({
                "name": p.get("player_name"),
                "win_prob": p.get("win"),
                "top5_prob": p.get("top_5"),
                "top10_prob": p.get("top_10"),
                "top20_prob": p.get("top_20"),
                "make_cut_prob": p.get("make_cut"),
                "sg_total": p.get("total"),
                "course_fit_score": p.get("course_fit"),
            })
        return players

    # ── COURSE FIT ────────────────────────────────────────────────

    def get_course_fit_adjustments(self, tour: str = "pga") -> list[dict]:
        """
        DataGolf's ML-derived course fit adjustments.
        Replaces/enhances our manual COURSE_PROFILES when activated.
        """
        data = self._get(DATAGOLF_ENDPOINTS["course_fit"], {"tour": tour})
        if not data:
            return []
        return data.get("players", [])

    # ── DFS OWNERSHIP PROJECTIONS ─────────────────────────────────

    def get_ownership_projections(self, site: str = "dk", slate: str = "main") -> list[dict]:
        """
        DataGolf's DFS ownership projections for DraftKings / FanDuel.
        This is GOLD for ownership leverage modeling.

        site: "dk" or "fd"
        Returns: [{"player_name", "proj_ownership", "salary", "proj_pts"}, ...]
        """
        params = {"site": site, "slate": slate, "tour": "pga"}
        data = self._get(DATAGOLF_ENDPOINTS["ownership"], params)
        if not data:
            return []

        return [{
            "name": p.get("player_name"),
            "proj_ownership": p.get("proj_own"),
            "salary": p.get("salary"),
            "proj_pts": p.get("proj_points"),
        } for p in data.get("players", [])]

    # ── HISTORICAL RAW DATA ───────────────────────────────────────

    def get_historical_sg(self, tour: str = "pga", event_id: str = None,
                           year: int = None) -> list[dict]:
        """
        Historical SG data by event — for training/backtesting.
        Far more complete than PGA Tour's public data.
        """
        params = {"tour": tour}
        if event_id:
            params["event_id"] = event_id
        if year:
            params["year"] = year
        data = self._get(DATAGOLF_ENDPOINTS["sg_categories"], params)
        if not data:
            return []
        return data.get("data", [])

    # ── LIVE TOURNAMENT STATS ─────────────────────────────────────

    def get_live_stats(self, tour: str = "pga") -> list[dict]:
        """
        Live in-tournament SG stats.
        Use during the tournament for live-model updates.
        """
        data = self._get(DATAGOLF_ENDPOINTS["live_model"], {"tour": tour})
        if not data:
            return []
        return data.get("data", [])

    def status(self) -> str:
        return "ACTIVE ✅" if self.enabled else "STUBBED 🔶 (set DATAGOLF_API_KEY in .env)"
