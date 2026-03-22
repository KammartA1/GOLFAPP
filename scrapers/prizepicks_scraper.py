"""
Golf Quant Engine — PrizePicks Scraper
Fetches live PGA golf lines from the PrizePicks API.
"""
import requests
import json
import logging
from datetime import datetime

log = logging.getLogger(__name__)

PP_PROJECTIONS_URL = "https://api.prizepicks.com/projections"
PP_LEAGUES_URL = "https://api.prizepicks.com/leagues"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://app.prizepicks.com/",
    "Origin": "https://app.prizepicks.com",
}

# Known PrizePicks stat type mappings
PP_STAT_NORMALIZE = {
    "Fantasy Score": "fantasy_score",
    "Birdies or Better": "birdies_or_better",
    "Birdies": "birdies",
    "Bogey-Free Holes": "bogey_free_holes",
    "Bogey Free Holes": "bogey_free_holes",
    "Pars or Better": "pars_or_better",
    "Pars": "pars",
    "Strokes": "strokes_total",
    "Total Strokes": "strokes_total",
    "Strokes (Round)": "strokes_total",
    "Eagles": "eagles",
    "Longest Drive": "longest_drive",
    "FedExCup Points": "fedex_points",
    "Holes Under Par": "holes_under_par",
    "Greens in Regulation": "greens_in_regulation",
    "GIR": "greens_in_regulation",
    "Fairways Hit": "fairways_hit",
    "Aces": "holes_in_one",
    "Made Cuts": "made_cuts",
    # Matchup variants
    "Birdies or Better (Matchup)": "birdies_or_better_matchup",
}


class PrizePicksScraper:
    """Scrapes PrizePicks API for PGA golf lines."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._pga_league_id = None

    def find_pga_league_id(self) -> str:
        """Find the PGA/Golf league ID from PrizePicks leagues endpoint."""
        if self._pga_league_id:
            return self._pga_league_id

        try:
            resp = self.session.get(PP_LEAGUES_URL, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            leagues = data.get("data", [])
            golf_keywords = ["pga", "golf", "pga tour"]

            for league in leagues:
                attrs = league.get("attributes", {})
                name = (attrs.get("name", "") or "").lower()
                if any(kw in name for kw in golf_keywords):
                    self._pga_league_id = league.get("id", "")
                    log.info(f"Found PGA league: id={self._pga_league_id}, name={attrs.get('name')}")
                    return self._pga_league_id

            # Log all leagues for debugging
            all_names = [l.get("attributes", {}).get("name", "?") for l in leagues]
            log.warning(f"No PGA league found. Available leagues: {all_names[:20]}")
            return ""

        except Exception as e:
            log.error(f"Failed to fetch PrizePicks leagues: {e}")
            return ""

    def fetch_golf_lines(self) -> dict:
        """
        Fetch all PGA golf projections from PrizePicks.

        Returns dict with:
            lines: list of parsed line dicts
            raw_response: raw API response for debugging
            error: error message if any
            league_id: the PGA league ID used
            timestamp: when the scrape happened
        """
        result = {
            "lines": [],
            "raw_response": None,
            "error": None,
            "league_id": None,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Find PGA league ID
        league_id = self.find_pga_league_id()
        result["league_id"] = league_id

        try:
            params = {"per_page": 250, "single_stat": "true"}
            if league_id:
                params["league_id"] = league_id

            resp = self.session.get(PP_PROJECTIONS_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            result["raw_response"] = json.dumps(data)[:50000]

        except requests.exceptions.HTTPError as e:
            result["error"] = f"PrizePicks API HTTP error: {e.response.status_code}"
            log.error(result["error"])
            return result
        except requests.exceptions.ConnectionError:
            result["error"] = "PrizePicks API connection error — service may be down"
            log.error(result["error"])
            return result
        except Exception as e:
            result["error"] = f"PrizePicks API error: {str(e)}"
            log.error(result["error"])
            return result

        # Parse JSON:API format
        projections = data.get("data", [])
        included = data.get("included", [])

        # Build lookup maps from included resources
        players_map = {}   # id -> player info
        leagues_map = {}   # id -> league info

        for item in included:
            item_type = item.get("type", "")
            item_id = item.get("id", "")
            attrs = item.get("attributes", {})

            if item_type == "new_player":
                players_map[item_id] = {
                    "name": attrs.get("display_name", attrs.get("name", "")),
                    "team": attrs.get("team", ""),
                    "position": attrs.get("position", ""),
                    "image_url": attrs.get("image_url", ""),
                }
            elif item_type == "league":
                leagues_map[item_id] = {
                    "name": attrs.get("name", ""),
                    "sport": attrs.get("sport", ""),
                }

        # Parse each projection
        pga_lines = []
        for proj in projections:
            attrs = proj.get("attributes", {})
            relationships = proj.get("relationships", {})

            # Get player info
            player_rel = relationships.get("new_player", {}).get("data", {})
            player_id = player_rel.get("id", "")
            player_info = players_map.get(player_id, {})

            # Get league info
            league_rel = relationships.get("league", {}).get("data", {})
            proj_league_id = league_rel.get("id", "")
            league_info = leagues_map.get(proj_league_id, {})

            # Filter for golf/PGA only
            league_name = (league_info.get("name", "") or "").lower()
            is_golf = any(kw in league_name for kw in ["pga", "golf", "lpga", "liv"])

            # If we have a league_id filter, also check by ID
            if league_id and proj_league_id != league_id:
                if not is_golf:
                    continue

            if not is_golf and league_id:
                continue

            stat_type_raw = attrs.get("stat_type", "")
            stat_type = PP_STAT_NORMALIZE.get(stat_type_raw, stat_type_raw.lower().replace(" ", "_"))

            line = {
                "player_name": player_info.get("name", "Unknown"),
                "stat_type": stat_type,
                "stat_type_display": stat_type_raw,
                "line_value": float(attrs.get("line_score", 0)),
                "start_time": attrs.get("start_time", ""),
                "description": attrs.get("description", ""),
                "is_active": attrs.get("status", "") == "pre_game" or attrs.get("status", "") not in ("closed", "suspended"),
                "flash_sale": bool(attrs.get("flash_sale_line_score")),
                "projection_id": proj.get("id", ""),
                "league_id": proj_league_id,
                "league_name": league_info.get("name", ""),
                "odds_type": attrs.get("odds_type", ""),
                "raw": json.dumps({
                    "id": proj.get("id"),
                    "attributes": attrs,
                    "player": player_info,
                    "league": league_info,
                }),
            }
            pga_lines.append(line)

        result["lines"] = pga_lines

        if not pga_lines:
            if not league_id:
                result["error"] = (
                    "No PGA league found on PrizePicks. Golf lines may not be available yet. "
                    "Lines typically go live Tuesday morning of tournament week."
                )
            else:
                result["error"] = (
                    f"No PGA lines currently available (league_id={league_id}). "
                    "Lines typically go live Tuesday morning of tournament week."
                )

        log.info(f"PrizePicks scrape: {len(pga_lines)} PGA lines found")
        return result

    def scrape_and_store(self) -> dict:
        """Scrape PrizePicks and store results in database."""
        from database.db_manager import (
            deactivate_old_pp_lines, insert_pp_line, normalize_name
        )

        result = self.fetch_golf_lines()

        if result["lines"]:
            deactivate_old_pp_lines()
            for line in result["lines"]:
                insert_pp_line(
                    player_name=line["player_name"],
                    stat_type=line["stat_type"],
                    line_value=line["line_value"],
                    tournament_name="",
                    start_time=line.get("start_time", ""),
                    pp_projection_id=line.get("projection_id", ""),
                    league_id=line.get("league_id", ""),
                    player_norm=normalize_name(line["player_name"]),
                    raw_json=line.get("raw", ""),
                )
            log.info(f"Stored {len(result['lines'])} PrizePicks lines in database")

        return result
