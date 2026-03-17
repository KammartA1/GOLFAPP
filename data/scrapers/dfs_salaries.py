"""
DFS Salary Scraper
Pulls DraftKings and FanDuel golf salaries.
DK publishes CSV export — FD requires scraping.
"""
import io
import csv
import logging
import requests
from typing import Optional
from config.settings import REQUEST_TIMEOUT, USER_AGENT

log = logging.getLogger(__name__)


class DFSSalaryScraper:
    """
    Scrapes DFS salaries for golf tournaments.

    DraftKings: Download salary CSV from their contest lobby
        URL format: https://www.draftkings.com/lineup/getavailableplayerscsv?contestTypeId=21&draftGroupId={id}
    FanDuel: Requires their contest CSV export from contest page.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Referer": "https://www.draftkings.com",
        })

    def fetch_dk_salaries_from_csv(self, draft_group_id: str) -> list[dict]:
        """
        Fetch DraftKings golf salary CSV by draft group ID.
        Find draft_group_id in the URL when viewing a contest lobby.
        """
        url = f"https://www.draftkings.com/lineup/getavailableplayerscsv?contestTypeId=21&draftGroupId={draft_group_id}"
        log.info(f"Fetching DK salaries for draft group {draft_group_id}...")

        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            content = resp.text
        except Exception as e:
            log.error(f"Failed to fetch DK salary CSV: {e}")
            return []

        return self._parse_dk_csv(content)

    def load_dk_csv_file(self, filepath: str) -> list[dict]:
        """Load a locally saved DK salary CSV file (manual download)."""
        try:
            with open(filepath, "r") as f:
                return self._parse_dk_csv(f.read())
        except Exception as e:
            log.error(f"Failed to load DK CSV: {e}")
            return []

    def _parse_dk_csv(self, content: str) -> list[dict]:
        """Parse DraftKings salary CSV format."""
        players = []
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            try:
                # DK CSV columns vary slightly — handle both formats
                name = row.get("Name") or row.get("name") or ""
                salary = row.get("Salary") or row.get("salary") or "0"
                avg_pts = row.get("AvgPointsPerGame") or row.get("avg_points_per_game") or "0"
                team = row.get("TeamAbbrev") or row.get("team") or ""

                players.append({
                    "name": name.strip(),
                    "platform": "DraftKings",
                    "salary": int(salary.replace("$", "").replace(",", "")),
                    "avg_pts": float(avg_pts) if avg_pts else 0.0,
                    "team": team.strip(),
                })
            except Exception as e:
                log.debug(f"Skipped row {row}: {e}")

        log.info(f"Parsed {len(players)} DK players")
        return players

    def load_fd_csv_file(self, filepath: str) -> list[dict]:
        """Load a locally saved FanDuel salary CSV file."""
        players = []
        try:
            with open(filepath, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("Nickname") or row.get("Name") or ""
                    salary = row.get("Salary") or "0"
                    fppg = row.get("FPPG") or "0"
                    players.append({
                        "name": name.strip(),
                        "platform": "FanDuel",
                        "salary": int(salary.replace("$", "").replace(",", "")),
                        "avg_pts": float(fppg) if fppg else 0.0,
                    })
        except Exception as e:
            log.error(f"Failed to load FD CSV: {e}")

        log.info(f"Parsed {len(players)} FD players")
        return players

    def normalize_name(self, name: str) -> str:
        """Normalize player names for cross-source matching (DK vs PGA vs ESPN)."""
        # Handle "Last, First" format
        if "," in name:
            parts = name.split(",", 1)
            name = f"{parts[1].strip()} {parts[0].strip()}"
        # Remove suffixes
        for suffix in [" Jr.", " Sr.", " III", " II"]:
            name = name.replace(suffix, "")
        return name.strip().lower()

    def merge_salaries_with_projections(
        self,
        salaries: list[dict],
        projections: list[dict],
        platform: str = "DraftKings"
    ) -> list[dict]:
        """
        Merge salary data with model projections on player name.
        Returns merged list with salary + projection data.
        """
        salary_map = {
            self.normalize_name(s["name"]): s
            for s in salaries
            if s.get("platform") == platform or not s.get("platform")
        }

        merged = []
        for proj in projections:
            norm = self.normalize_name(proj.get("name", ""))
            sal = salary_map.get(norm)

            if sal:
                entry = {**proj, **sal}
                salary = sal["salary"]
                pts = proj.get("dk_proj_pts") if platform == "DraftKings" else proj.get("fd_proj_pts")
                if salary and pts:
                    entry["value"] = round(pts / (salary / 1000), 2)
                merged.append(entry)
            else:
                log.debug(f"No salary match for: {proj.get('name')}")

        log.info(f"Merged {len(merged)}/{len(projections)} players with {platform} salaries")
        return merged
