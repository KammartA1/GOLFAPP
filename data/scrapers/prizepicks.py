"""
PrizePicks Scraper
Pulls live golf player prop lines from PrizePicks via their unofficial API.

PrizePicks API is undocumented but stable — used by their own web app.
No API key needed. Rate limit gently (2s between calls).

Golf-specific stats available on PrizePicks:
  - Fantasy Score (DK-style scoring)
  - Bogey Free Rounds
  - Birdies or Better (per round)
  - Eagles
  - Strokes (total score)
  - Holes Under Par
  - Longest Drive (featured slates)
  - Greens in Regulation
  - Fairways Hit
"""
import time
import logging
import requests
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ── PrizePicks API constants ───────────────────────────────────────────────
PP_API_BASE    = "https://api.prizepicks.com"
PP_PROJ_URL    = f"{PP_API_BASE}/projections"
PP_LEAGUES_URL = f"{PP_API_BASE}/leagues"

# Golf league IDs on PrizePicks
# PGA Tour = 9 (verify — can shift between seasons)
GOLF_LEAGUE_IDS = {
    "PGA": 9,
    "LIV": 9,          # Often grouped
    "DP World": 10,    # European Tour — may vary
}

STAT_CATEGORY_MAP = {
    "Fantasy Score":         "fantasy_score",
    "Bogey Free Rounds":     "bogey_free_rounds",
    "Birdies or Better":     "birdies_or_better",
    "Birdies":               "birdies",
    "Eagles":                "eagles",
    "Strokes":               "strokes_total",
    "Holes Under Par":       "holes_under_par",
    "Longest Drive":         "longest_drive",
    "Greens in Regulation":  "gir",
    "Fairways Hit":          "fairways_hit",
    "Holes in One":          "holes_in_one",
    "Rounds":                "rounds_played",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://app.prizepicks.com/",
    "Origin":          "https://app.prizepicks.com",
    "x-device-id":    "golf-quant-engine",
}


@dataclass
class PPProjection:
    """A single PrizePicks player projection line."""
    pp_id:          str
    player_name:    str
    player_id:      str
    stat_type:      str          # normalized stat key (e.g. "fantasy_score")
    stat_display:   str          # raw display name from PP (e.g. "Fantasy Score")
    line_score:     float        # the over/under line
    is_promo:       bool = False # promo lines are less reliable
    start_time:     Optional[datetime] = None
    description:    str = ""
    league:         str = "PGA"
    flash_sale_line_score: Optional[float] = None  # discounted line (power play)

    # Filled in by analyzer
    model_proj:     Optional[float] = None    # our model's projection for this stat
    edge_over:      Optional[float] = None    # % edge on the OVER
    edge_under:     Optional[float] = None    # % edge on the UNDER
    recommendation: Optional[str]  = None    # "OVER", "UNDER", or None
    confidence:     Optional[str]  = None    # "HIGH", "MEDIUM", "LOW"

    @property
    def normalized_stat(self) -> str:
        return STAT_CATEGORY_MAP.get(self.stat_display, self.stat_display.lower().replace(" ", "_"))


class PrizePicksScraper:
    """
    Fetches live PrizePicks golf projections.
    Uses their undocumented but stable REST API.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._league_cache: dict = {}

    def _get(self, url: str, params: dict = None, retries: int = 3) -> Optional[dict]:
        """Resilient GET with retry."""
        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                time.sleep(2)
                return resp.json()
            except requests.HTTPError as e:
                log.warning(f"HTTP {e.response.status_code} on attempt {attempt+1}: {url}")
                if e.response.status_code == 429:
                    time.sleep(10 * (attempt + 1))
            except Exception as e:
                log.warning(f"Request error attempt {attempt+1}: {e}")
                time.sleep(2 * (attempt + 1))
        log.error(f"All retries failed: {url}")
        return None

    def fetch_leagues(self) -> list[dict]:
        """Fetch all active leagues to find golf league IDs."""
        data = self._get(PP_LEAGUES_URL)
        if not data:
            return []
        leagues = data.get("data", [])
        result = []
        for league in leagues:
            attrs = league.get("attributes", {})
            name = attrs.get("name", "")
            if any(g in name.upper() for g in ["GOLF", "PGA", "LIV", "DP WORLD"]):
                result.append({
                    "id": league.get("id"),
                    "name": name,
                    "active": attrs.get("active", False),
                })
                log.info(f"Found golf league: {name} (id={league.get('id')})")
        return result

    def fetch_golf_projections(
        self,
        league_id: int = None,
        per_page: int = 250,
        single_stat_only: bool = False,
    ) -> list[PPProjection]:
        """
        Fetch all live golf projections from PrizePicks.

        league_id: PGA = 9 by default (auto-discovers if None)
        single_stat_only: True = only standard lines, False = includes combos
        """
        if league_id is None:
            league_id = self._discover_golf_league_id()

        params = {
            "league_id":   league_id,
            "per_page":    per_page,
            "single_stat": "true" if single_stat_only else "false",
        }

        log.info(f"Fetching PrizePicks golf projections (league_id={league_id})...")
        data = self._get(PP_PROJ_URL, params=params)
        if not data:
            log.error("Failed to fetch PrizePicks projections")
            return []

        projections = self._parse_projections(data)
        log.info(f"✅ Fetched {len(projections)} PrizePicks golf lines")
        return projections

    def _discover_golf_league_id(self) -> int:
        """Auto-discover the golf league ID — handles PP changing IDs."""
        leagues = self.fetch_leagues()
        for league in leagues:
            if "PGA" in league["name"].upper() and league.get("active"):
                return int(league["id"])
        log.warning("Could not auto-discover golf league ID — defaulting to 9")
        return 9

    def _parse_projections(self, data: dict) -> list[PPProjection]:
        """
        Parse raw PrizePicks API response into PPProjection objects.

        PrizePicks uses JSON:API format:
          data[]: projection objects
          included[]: related objects (players, leagues, etc.) by id
        """
        included = data.get("included", [])

        # Build lookup maps from included resources
        player_map = {}
        for obj in included:
            if obj.get("type") == "new_player":
                attrs = obj.get("attributes", {})
                player_map[obj["id"]] = {
                    "name": attrs.get("display_name") or attrs.get("name", "Unknown"),
                    "team": attrs.get("team", ""),
                    "position": attrs.get("position", ""),
                }

        projections = []
        for item in data.get("data", []):
            if item.get("type") != "projection":
                continue

            attrs  = item.get("attributes", {})
            rels   = item.get("relationships", {})

            # Resolve player
            player_rel = rels.get("new_player", {}).get("data", {})
            player_id  = player_rel.get("id", "")
            player_info = player_map.get(player_id, {})
            player_name = player_info.get("name", attrs.get("description", "Unknown"))

            stat_display = attrs.get("stat_type", "")
            line_score   = attrs.get("line_score")
            if line_score is None:
                continue  # Skip if no line

            try:
                start_time = datetime.fromisoformat(
                    attrs.get("start_time", "").replace("Z", "+00:00")
                )
            except:
                start_time = None

            proj = PPProjection(
                pp_id        = item.get("id", ""),
                player_name  = player_name,
                player_id    = player_id,
                stat_type    = STAT_CATEGORY_MAP.get(stat_display,
                                   stat_display.lower().replace(" ", "_")),
                stat_display = stat_display,
                line_score   = float(line_score),
                is_promo     = attrs.get("is_promo", False),
                start_time   = start_time,
                description  = attrs.get("description", ""),
                flash_sale_line_score = float(f) if (f := attrs.get("flash_sale_line_score")) else None,
            )
            projections.append(proj)

        return projections

    def fetch_by_stat(
        self,
        stat_type: str,
        league_id: int = None,
    ) -> list[PPProjection]:
        """
        Fetch projections filtered to a specific stat type.

        stat_type: "fantasy_score", "birdies_or_better", "bogey_free_rounds", etc.
        """
        all_projs = self.fetch_golf_projections(league_id=league_id)
        filtered = [p for p in all_projs if p.stat_type == stat_type]
        log.info(f"Filtered to {len(filtered)} '{stat_type}' lines")
        return filtered

    def fetch_player_lines(self, player_name: str, league_id: int = None) -> list[PPProjection]:
        """Fetch all PrizePicks lines for a specific player."""
        all_projs = self.fetch_golf_projections(league_id=league_id)
        name_lower = player_name.lower()
        matches = [
            p for p in all_projs
            if name_lower in p.player_name.lower() or p.player_name.lower() in name_lower
        ]
        if not matches:
            log.warning(f"No PrizePicks lines found for: {player_name}")
        return matches

    def get_available_stats(self, league_id: int = None) -> dict[str, int]:
        """Return dict of {stat_type: count_of_lines} for current slate."""
        projs = self.fetch_golf_projections(league_id=league_id)
        stat_counts = {}
        for p in projs:
            stat_counts[p.stat_type] = stat_counts.get(p.stat_type, 0) + 1
        return dict(sorted(stat_counts.items(), key=lambda x: x[1], reverse=True))

    def print_slate(self, projections: list[PPProjection]):
        """Pretty-print the current PrizePicks golf slate."""
        from rich.table import Table
        from rich.console import Console
        console = Console()

        table = Table(title="🏌️  PrizePicks Live Golf Slate", show_lines=True)
        table.add_column("Player",     style="bold")
        table.add_column("Stat",       style="cyan")
        table.add_column("Line",       justify="center", style="yellow")
        table.add_column("Promo",      justify="center")
        table.add_column("Starts",     style="dim")

        for p in sorted(projections, key=lambda x: (x.stat_display, x.player_name)):
            promo_str = "⭐ PROMO" if p.is_promo else ""
            start_str = p.start_time.strftime("%a %m/%d %I:%M%p") if p.start_time else ""
            table.add_row(
                p.player_name,
                p.stat_display,
                str(p.line_score),
                promo_str,
                start_str,
            )

        console.print(table)
        console.print(f"\n[dim]{len(projections)} total lines on slate[/dim]")
