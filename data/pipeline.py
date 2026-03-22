"""
Data Pipeline
Orchestrates all data collection: PGA Tour, weather, salaries, DataGolf.
Run this before every tournament to refresh all data.
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from data.scrapers.pga_tour import PGATourScraper, ESPNGolfScraper
from data.scrapers.weather import WeatherModel
from data.scrapers.dfs_salaries import DFSSalaryScraper
from data.scrapers.datagolf import DataGolfClient
from data.scrapers.shotlink import ShotLinkScraper
from data.storage.database import get_session, Player, Tournament, SGStats, TournamentResult
from config.settings import DATA_DIR

log = logging.getLogger(__name__)


class DataPipeline:
    """
    Full data pipeline for tournament prep.
    Pulls and stores all required data before running projections.
    """

    def __init__(self):
        self.pga      = PGATourScraper()
        self.espn     = ESPNGolfScraper()
        self.weather  = WeatherModel()
        self.dfs_sal  = DFSSalaryScraper()
        self.dg       = DataGolfClient()
        self.shotlink = ShotLinkScraper()
        self.session  = get_session()

    # ─────────────────────────────────────────────
    # FULL REFRESH
    # ─────────────────────────────────────────────

    def full_refresh(self, event_id: str = None) -> dict:
        """
        Full data refresh for an upcoming tournament.

        Steps:
          1. Refresh player database
          2. Fetch tournament field
          3. Pull multi-season SG stats per player
          4. Supplement with DataGolf (if active)
          5. Cache weather data

        Returns dict ready for ProjectionEngine.run()
        """
        log.info("🔄 Starting full data refresh...")

        results = {}

        # 1. Player list refresh
        log.info("Step 1/4: Refreshing player database...")
        players = self.pga.fetch_player_list()
        self._upsert_players(players)

        # 2. Field for this event
        log.info("Step 2/4: Fetching tournament field...")
        if event_id:
            field = self.pga.fetch_field(event_id)
        else:
            # Use upcoming tournament
            upcoming = self.pga.fetch_upcoming_tournaments(weeks_ahead=1)
            if upcoming:
                event_id = upcoming[0]["pga_event_id"]
                field = self.pga.fetch_field(event_id)
                results["tournament"] = upcoming[0]
            else:
                log.warning("No upcoming tournaments found")
                field = []

        results["field"] = field
        log.info(f"  Field: {len(field)} players")

        # 3. SG history per player in the field
        # v8.0: Use ShotLink scraper for event-level SG history (real data)
        log.info("Step 3/4: Pulling SG history via ShotLink scraper...")
        player_names = [p["name"] for p in field]
        seasons = [datetime.now().year, datetime.now().year - 1, datetime.now().year - 2]
        try:
            sg_history = self.shotlink.build_player_event_history(player_names, seasons)
            log.info(f"  ShotLink: event-level SG for {sum(1 for v in sg_history.values() if v)} players")
        except Exception as e:
            log.warning(f"ShotLink scraper failed, falling back to season-level: {e}")
            sg_history = self._build_sg_history(player_names=player_names, seasons=seasons)
        results["sg_history"] = sg_history

        # 4. DataGolf supplement
        if self.dg.enabled:
            log.info("Step 4/4: Fetching DataGolf projections...")
            dg_preds = self.dg.get_predictions(add_course_fit=True)
            results["dg_predictions"] = {p["name"]: p for p in dg_preds}
        else:
            log.info("Step 4/4: DataGolf STUBBED — skipping")
            results["dg_predictions"] = {}

        log.info("✅ Data refresh complete")
        return results

    def _upsert_players(self, players: list[dict]):
        """Insert or update players in database."""
        for p in players:
            existing = self.session.query(Player).filter_by(
                pga_player_id=p.get("pga_player_id")
            ).first()
            if existing:
                existing.world_rank = p.get("world_rank")
                existing.updated_at = datetime.utcnow()
            else:
                new_player = Player(
                    pga_player_id=p.get("pga_player_id", ""),
                    name=p.get("name", ""),
                    country=p.get("country", ""),
                    world_rank=p.get("world_rank"),
                )
                self.session.add(new_player)
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            log.error(f"Player upsert error: {e}")

    def _build_sg_history(
        self,
        player_names: list[str],
        seasons: list[int],
    ) -> dict[str, list[dict]]:
        """
        Build SG history dict for each player in the field.
        Tries database first, falls back to PGA Tour API.
        """
        history = {}

        # Fetch from PGA Tour for each season
        all_season_data = {}
        for season in seasons:
            season_stats = self._fetch_season_sg(season)
            if season_stats is not None:
                all_season_data[season] = season_stats

        # Organize by player
        for name in player_names:
            player_records = []
            for season, df in all_season_data.items():
                # Match player by name (fuzzy if needed)
                match = df[df["name"].str.lower() == name.lower()]
                if not match.empty:
                    row = match.iloc[0].to_dict()
                    row["event_date"] = datetime(season, 6, 1)  # approximate mid-season date
                    row["season"] = season
                    player_records.append(row)

            history[name] = sorted(player_records, key=lambda x: x.get("event_date", datetime.min))

        log.info(f"Built SG history for {len([k for k, v in history.items() if v])} players")
        return history

    def _fetch_season_sg(self, season: int) -> Optional[pd.DataFrame]:
        """Fetch all SG categories for a season and merge into single DataFrame."""
        cache_file = DATA_DIR / f"sg_stats_{season}.parquet"

        # Use cache if available and fresh
        if cache_file.exists():
            age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
            if age_days < 7:
                log.info(f"  Using cached SG data for {season}")
                return pd.read_parquet(cache_file)

        log.info(f"  Fetching SG stats for {season} from PGA Tour...")
        try:
            all_stats = self.pga.fetch_all_sg_stats(year=season)
        except Exception as e:
            log.error(f"Failed to fetch SG stats for {season}: {e}")
            return None

        if not all_stats:
            return None

        # Merge all stat categories on player ID + name
        merged = None
        for stat_key, records in all_stats.items():
            df = pd.DataFrame(records)
            if df.empty:
                continue
            if merged is None:
                merged = df[["pga_player_id", "name", stat_key]].copy()
            else:
                merged = merged.merge(
                    df[["pga_player_id", stat_key]],
                    on="pga_player_id", how="outer"
                )

        if merged is not None and not merged.empty:
            merged["season"] = season
            # Cache it
            merged.to_parquet(cache_file, index=False)
            log.info(f"  Cached {len(merged)} player stats for {season}")

        return merged

    # ─────────────────────────────────────────────
    # HISTORICAL DATA LOAD (for backtesting)
    # ─────────────────────────────────────────────

    def load_historical_results(self, seasons: list[int]) -> pd.DataFrame:
        """
        Load historical tournament results for backtesting.
        Fetches from PGA Tour API and merges with SG data.
        """
        all_results = []
        for season in seasons:
            schedule = self.pga.fetch_schedule(year=season)
            completed = schedule.get("completed", [])
            log.info(f"Loading {len(completed)} completed events from {season}...")

            for event in completed[:10]:  # Limit to avoid rate limiting
                event_id = event.get("id")
                if not event_id:
                    continue
                results = self.pga.fetch_tournament_results(event_id)
                for r in results:
                    r["season"] = season
                    r["event_name"] = event.get("name", "")
                    r["course_name"] = event.get("courses", [{}])[0].get("name", "")
                all_results.extend(results)

        return pd.DataFrame(all_results)

    # ─────────────────────────────────────────────
    # SALARY INTEGRATION
    # ─────────────────────────────────────────────

    def load_salaries(
        self,
        dk_path: str = None,
        fd_path: str = None,
        dk_draft_group_id: str = None,
    ) -> dict:
        """
        Load DFS salaries from files or DK API.

        Instructions:
          DraftKings: Go to contest lobby → Export CSV → save to dk_path
          FanDuel: Go to contest → Export → save to fd_path
          Or provide dk_draft_group_id to auto-fetch from DK
        """
        salaries = {}

        if dk_draft_group_id:
            dk_data = self.dfs_sal.fetch_dk_salaries_from_csv(dk_draft_group_id)
        elif dk_path:
            dk_data = self.dfs_sal.load_dk_csv_file(dk_path)
        else:
            dk_data = []

        if fd_path:
            fd_data = self.dfs_sal.load_fd_csv_file(fd_path)
        else:
            fd_data = []

        # Normalize into {player_name: {dk_salary, fd_salary}}
        for p in dk_data:
            name = self.dfs_sal.normalize_name(p["name"])
            salaries.setdefault(p["name"], {})["dk_salary"] = p["salary"]

        for p in fd_data:
            name = self.dfs_sal.normalize_name(p["name"])
            salaries.setdefault(p["name"], {})["fd_salary"] = p["salary"]

        log.info(f"Loaded salaries: {len([k for k,v in salaries.items() if 'dk_salary' in v])} DK, "
                 f"{len([k for k,v in salaries.items() if 'fd_salary' in v])} FD")
        return salaries
