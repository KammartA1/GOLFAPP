"""
PGA Tour Scraper
Primary free data source for SG stats, player data, tournament results.

Endpoints used:
  - PGA Tour GraphQL API (unofficial but stable)
  - PGA Tour stats pages
  - ESPN Golf API (free, no key needed)
"""
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional

from config.settings import (
    PGA_GRAPHQL_URL, ESPN_GOLF_API, ESPN_PLAYER_API,
    REQUEST_DELAY, REQUEST_TIMEOUT, MAX_RETRIES, USER_AGENT, DATA_DIR
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# BASE SESSION
# ─────────────────────────────────────────────

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "x-api-key": "da2-gsrx5bibzbb4njvhl7t37wqyl4",   # PGA Tour public API key (from their web app)
        "x-amz-user-agent": "aws-amplify/3.0.7",
    })
    return s


def _request(session: requests.Session, method: str, url: str, **kwargs) -> Optional[dict]:
    """Resilient request with retry + delay."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.json()
        except requests.HTTPError as e:
            log.warning(f"HTTP {e.response.status_code} on attempt {attempt+1}: {url}")
            if e.response.status_code == 429:
                time.sleep(10 * (attempt + 1))
        except Exception as e:
            log.warning(f"Request error attempt {attempt+1}: {e}")
            time.sleep(REQUEST_DELAY * (attempt + 1))
    log.error(f"All retries failed for {url}")
    return None


# ─────────────────────────────────────────────
# PGA TOUR GRAPHQL QUERIES
# ─────────────────────────────────────────────

PLAYER_LIST_QUERY = """
query PlayerDirectory($tourCode: TourCode!, $active: Boolean) {
  players(tourCode: $tourCode, active: $active) {
    id
    firstName
    lastName
    country
    worldRankings { current }
    playerBio { age birthplace }
  }
}
"""

TOURNAMENT_SCHEDULE_QUERY = """
query Schedule($tourCode: TourCode!, $year: Int) {
  schedule(tourCode: $tourCode, year: $year) {
    completed {
      id name purse
      courses { name }
      dates { start end }
    }
    upcoming {
      id name purse
      courses { name }
      dates { start end }
    }
  }
}
"""

TOURNAMENT_RESULTS_QUERY = """
query TournamentResults($id: ID!) {
  tournamentPastResults(id: $id) {
    id
    players {
      id
      player { id firstName lastName country }
      status
      total
      position { short }
      rounds { score }
      earnings
    }
  }
}
"""

SG_STATS_QUERY = """
query StatDetails($tourCode: TourCode!, $statId: String!, $year: Int) {
  statDetails(tourCode: $tourCode, statId: $statId, year: $year) {
    displayName
    rows {
      player { id firstName lastName }
      stats { statValue }
      rank
    }
  }
}
"""

# PGA Tour stat IDs for SG categories
SG_STAT_IDS = {
    "sg_total":   "02675",
    "sg_ott":     "02567",
    "sg_app":     "02568",
    "sg_atg":     "02569",
    "sg_putt":    "02564",
    "sg_t2g":     "02674",
    "driving_dist": "101",
    "driving_acc":  "102",
    "gir":          "103",
    "scrambling":   "130",
    "birdie_avg":   "156",
    "proximity_100_125":  "331",
    "proximity_125_150":  "332",
    "proximity_150_175":  "333",
    "proximity_175_200":  "334",
}


class PGATourScraper:
    """Main scraper for PGA Tour data via their unofficial GraphQL API."""

    def __init__(self):
        self.session = _make_session()
        self.tour_code = "R"  # R = PGA Tour

    def _gql(self, query: str, variables: dict) -> Optional[dict]:
        return _request(
            self.session,
            "POST",
            PGA_GRAPHQL_URL,
            json={"query": query, "variables": variables}
        )

    # ── PLAYER DATA ────────────────────────────────────────────────

    def fetch_player_list(self, active_only: bool = True) -> list[dict]:
        """Fetch all active PGA Tour players."""
        log.info("Fetching PGA Tour player list...")
        data = self._gql(PLAYER_LIST_QUERY, {"tourCode": self.tour_code, "active": active_only})
        if not data or "data" not in data:
            log.error("Failed to fetch player list")
            return []
        players = data["data"].get("players", [])
        result = []
        for p in players:
            result.append({
                "pga_player_id": p["id"],
                "name": f"{p['firstName']} {p['lastName']}",
                "country": p.get("country", ""),
                "world_rank": p.get("worldRankings", {}).get("current"),
            })
        log.info(f"Fetched {len(result)} players")
        return result

    # ── TOURNAMENT SCHEDULE ────────────────────────────────────────

    def fetch_schedule(self, year: int = None) -> dict:
        """Fetch tournament schedule for a given year."""
        year = year or datetime.now().year
        log.info(f"Fetching {year} schedule...")
        data = self._gql(TOURNAMENT_SCHEDULE_QUERY, {"tourCode": self.tour_code, "year": year})
        if not data or "data" not in data:
            return {}
        return data["data"].get("schedule", {})

    def fetch_upcoming_tournaments(self, weeks_ahead: int = 4) -> list[dict]:
        """Get the next N weeks of tournaments."""
        schedule = self.fetch_schedule()
        upcoming = schedule.get("upcoming", [])
        cutoff = datetime.now() + timedelta(weeks=weeks_ahead)
        result = []
        for t in upcoming:
            start_str = t.get("dates", {}).get("start", "")
            try:
                start = datetime.fromisoformat(start_str.replace("Z", ""))
                if start <= cutoff:
                    result.append({
                        "pga_event_id": t["id"],
                        "name": t["name"],
                        "course_name": t.get("courses", [{}])[0].get("name", ""),
                        "start_date": start,
                        "purse": t.get("purse", 0),
                    })
            except:
                pass
        return result

    # ── TOURNAMENT RESULTS ─────────────────────────────────────────

    def fetch_tournament_results(self, event_id: str) -> list[dict]:
        """Fetch full results for a completed tournament."""
        log.info(f"Fetching results for event {event_id}...")
        data = self._gql(TOURNAMENT_RESULTS_QUERY, {"id": event_id})
        if not data or "data" not in data:
            return []
        raw = data["data"].get("tournamentPastResults", {})
        results = []
        for p in raw.get("players", []):
            player_info = p.get("player", {})
            rounds = [r.get("score") for r in p.get("rounds", [])]
            results.append({
                "pga_player_id": player_info.get("id"),
                "name": f"{player_info.get('firstName','')} {player_info.get('lastName','')}".strip(),
                "finish_str": p.get("position", {}).get("short", ""),
                "made_cut": p.get("status") not in ["MC", "WD", "DQ"],
                "score_total": p.get("total"),
                "score_r1": rounds[0] if len(rounds) > 0 else None,
                "score_r2": rounds[1] if len(rounds) > 1 else None,
                "score_r3": rounds[2] if len(rounds) > 2 else None,
                "score_r4": rounds[3] if len(rounds) > 3 else None,
                "earnings": p.get("earnings"),
            })
        log.info(f"Fetched {len(results)} results")
        return results

    # ── STROKES GAINED STATS ───────────────────────────────────────

    def fetch_sg_stats(self, stat_key: str, year: int = None, season_year: int = None) -> list[dict]:
        """
        Fetch a specific SG stat for all players in a given season.

        stat_key: one of SG_STAT_IDS keys (e.g. 'sg_app', 'sg_putt')
        """
        year = year or datetime.now().year
        stat_id = SG_STAT_IDS.get(stat_key)
        if not stat_id:
            log.error(f"Unknown stat key: {stat_key}")
            return []

        log.info(f"Fetching {stat_key} stats for {year}...")
        data = self._gql(SG_STATS_QUERY, {
            "tourCode": self.tour_code,
            "statId": stat_id,
            "year": year
        })
        if not data or "data" not in data:
            return []

        rows = data["data"].get("statDetails", {}).get("rows", [])
        result = []
        for row in rows:
            p = row.get("player", {})
            stats = row.get("stats", [{}])
            val = stats[0].get("statValue") if stats else None
            try:
                val = float(val) if val else None
            except:
                val = None

            result.append({
                "pga_player_id": p.get("id"),
                "name": f"{p.get('firstName','')} {p.get('lastName','')}".strip(),
                "rank": row.get("rank"),
                stat_key: val,
                "season": year,
            })

        log.info(f"Fetched {len(result)} players for {stat_key}")
        return result

    def fetch_all_sg_stats(self, year: int = None) -> dict[str, list]:
        """Fetch all SG categories for a season."""
        year = year or datetime.now().year
        all_stats = {}
        for stat_key in SG_STAT_IDS:
            all_stats[stat_key] = self.fetch_sg_stats(stat_key, year=year)
            time.sleep(REQUEST_DELAY)
        return all_stats

    # ── CURRENT FIELD / ENTRY LIST ─────────────────────────────────

    def fetch_field(self, event_id: str) -> list[dict]:
        """Fetch the field (entry list) for an upcoming tournament."""
        FIELD_QUERY = """
        query TournamentEntryList($id: ID!) {
          tournamentEntryList(id: $id) {
            players {
              player { id firstName lastName country worldRankings { current } }
              status
            }
          }
        }
        """
        log.info(f"Fetching field for event {event_id}...")
        data = self._gql(FIELD_QUERY, {"id": event_id})
        if not data or "data" not in data:
            return []

        raw = data["data"].get("tournamentEntryList", {})
        players = []
        for entry in raw.get("players", []):
            p = entry.get("player", {})
            players.append({
                "pga_player_id": p.get("id"),
                "name": f"{p.get('firstName','')} {p.get('lastName','')}".strip(),
                "world_rank": p.get("worldRankings", {}).get("current"),
                "status": entry.get("status"),
            })
        log.info(f"Field has {len(players)} players")
        return players


# ─────────────────────────────────────────────
# ESPN GOLF API (backup / supplementary)
# ─────────────────────────────────────────────

class ESPNGolfScraper:
    """ESPN Golf API — free, no key needed, good for live scores & basic data."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def fetch_current_scoreboard(self) -> dict:
        """Get current tournament scoreboard from ESPN."""
        log.info("Fetching ESPN scoreboard...")
        data = _request(self.session, "GET", ESPN_GOLF_API)
        if not data:
            return {}
        return data

    def parse_scoreboard(self, data: dict) -> list[dict]:
        """Parse ESPN scoreboard into structured player list."""
        players = []
        events = data.get("events", [])
        for event in events:
            comps = event.get("competitions", [])
            for comp in comps:
                competitors = comp.get("competitors", [])
                for c in competitors:
                    athlete = c.get("athlete", {})
                    linescores = c.get("linescores", [])
                    rounds = [ls.get("value") for ls in linescores]
                    players.append({
                        "espn_id": athlete.get("id"),
                        "name": athlete.get("displayName"),
                        "position": c.get("status", {}).get("position", {}).get("displayName"),
                        "score": c.get("score"),
                        "rounds": rounds,
                        "made_cut": c.get("status", {}).get("type", {}).get("name") not in ["cut", "wd"],
                    })
        return players
