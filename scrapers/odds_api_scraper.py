"""
Golf Quant Engine — The Odds API Scraper
Fetches consensus sportsbook odds for PGA Tour golf.
"""
import requests
import json
import logging
from datetime import datetime

log = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
SPORTS_ENDPOINT = f"{ODDS_API_BASE}"


class OddsAPIScraper:
    """Fetches and processes odds from The Odds API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.requests_remaining = None
        self.requests_used = None
        self._golf_sport_key = None

    def _read_quota(self, resp):
        """Read API quota headers."""
        self.requests_remaining = resp.headers.get("x-requests-remaining")
        self.requests_used = resp.headers.get("x-requests-used")

    def get_quota(self) -> dict:
        return {
            "remaining": self.requests_remaining,
            "used": self.requests_used,
        }

    def find_golf_sport_key(self) -> str:
        """Discover available golf sport keys."""
        if self._golf_sport_key:
            return self._golf_sport_key

        try:
            resp = requests.get(
                SPORTS_ENDPOINT,
                params={"apiKey": self.api_key},
                timeout=15
            )
            resp.raise_for_status()
            self._read_quota(resp)
            sports = resp.json()

            golf_sports = [s for s in sports if "golf" in s.get("key", "").lower()
                           or "golf" in s.get("title", "").lower()
                           or "pga" in s.get("title", "").lower()]

            if golf_sports:
                # Prefer "golf_pga_tour" over event-specific keys
                for gs in golf_sports:
                    if gs.get("active", False):
                        self._golf_sport_key = gs["key"]
                        log.info(f"Found active golf sport: {gs['key']} — {gs.get('title')}")
                        return self._golf_sport_key
                # Fall back to first golf sport
                self._golf_sport_key = golf_sports[0]["key"]
                log.info(f"Using golf sport: {self._golf_sport_key}")
                return self._golf_sport_key

            all_keys = [s.get("key") for s in sports]
            log.warning(f"No golf sport found. Available: {all_keys[:30]}")
            return ""

        except Exception as e:
            log.error(f"Odds API sports list error: {e}")
            return ""

    def fetch_odds(self, markets: str = "outrights",
                   regions: str = "us", odds_format: str = "american") -> dict:
        """
        Fetch odds for the current PGA golf event.

        Args:
            markets: comma-separated market types (outrights, h2h, totals)
            regions: us, uk, eu, au
            odds_format: american, decimal

        Returns dict with: lines, consensus, quota, error, raw_response
        """
        result = {
            "lines": [],
            "consensus": [],
            "quota": {},
            "error": None,
            "raw_response": None,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sport_key": None,
        }

        sport_key = self.find_golf_sport_key()
        if not sport_key:
            result["error"] = "No active golf sport found on The Odds API"
            return result
        result["sport_key"] = sport_key

        try:
            url = f"{ODDS_API_BASE}/{sport_key}/odds"
            params = {
                "apiKey": self.api_key,
                "markets": markets,
                "regions": regions,
                "oddsFormat": odds_format,
            }
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            self._read_quota(resp)
            result["quota"] = self.get_quota()
            data = resp.json()
            result["raw_response"] = json.dumps(data)[:50000]

        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response else "?"
            result["error"] = f"Odds API HTTP {code}: {e}"
            if code == 401:
                result["error"] += " — Invalid API key"
            elif code == 429:
                result["error"] += " — Rate limit exceeded"
            return result
        except Exception as e:
            result["error"] = f"Odds API error: {str(e)}"
            return result

        # Parse events
        all_lines = []
        player_probs = {}  # player -> list of implied probs for consensus

        for event in data if isinstance(data, list) else [data]:
            event_name = event.get("home_team", event.get("id", ""))

            for bookmaker in event.get("bookmakers", []):
                book_name = bookmaker.get("title", bookmaker.get("key", "Unknown"))

                for market in bookmaker.get("markets", []):
                    market_key = market.get("key", "")

                    for outcome in market.get("outcomes", []):
                        player_name = outcome.get("name", "Unknown")
                        odds_val = outcome.get("price", 0)

                        # Convert to decimal and implied probability
                        odds_decimal = self._american_to_decimal(odds_val)
                        implied_prob = 1.0 / odds_decimal if odds_decimal > 0 else 0

                        line_data = {
                            "player_name": player_name,
                            "market_type": market_key,
                            "bookmaker": book_name,
                            "odds_american": odds_val,
                            "odds_decimal": round(odds_decimal, 4),
                            "implied_prob": round(implied_prob, 6),
                            "event_name": event_name,
                        }
                        all_lines.append(line_data)

                        # Accumulate for consensus
                        key = (player_name, market_key)
                        if key not in player_probs:
                            player_probs[key] = []
                        player_probs[key].append(implied_prob)

        result["lines"] = all_lines

        # Calculate consensus (no-vig)
        consensus = []
        for (player, market_type), probs in player_probs.items():
            if not probs:
                continue
            # Remove vig: normalize so all probs sum to 1
            total = sum(probs)
            avg_raw = sum(probs) / len(probs)
            # Simple vig removal: divide by overround
            # For individual player, just use average across books
            no_vig = avg_raw  # Individual player — vig removal needs full market

            consensus.append({
                "player_name": player,
                "market_type": market_type,
                "consensus_prob": round(no_vig, 6),
                "num_books": len(probs),
                "min_prob": round(min(probs), 6),
                "max_prob": round(max(probs), 6),
            })

        # Now do proper vig removal for outright markets
        if markets == "outrights":
            outright_probs = {c["player_name"]: c["consensus_prob"]
                             for c in consensus if c["market_type"] in ("outrights", "outright_winner")}
            total_prob = sum(outright_probs.values())
            if total_prob > 1.0:
                overround = total_prob
                for c in consensus:
                    if c["market_type"] in ("outrights", "outright_winner"):
                        c["consensus_prob"] = round(c["consensus_prob"] / overround, 6)
                log.info(f"Removed vig: overround was {overround:.2%}, normalized to 100%")

        result["consensus"] = sorted(consensus, key=lambda x: x["consensus_prob"], reverse=True)

        if not all_lines:
            result["error"] = "No golf odds returned — tournament may not have odds available yet"

        log.info(f"Odds API: {len(all_lines)} lines, {len(consensus)} players, "
                 f"quota remaining: {self.requests_remaining}")
        return result

    def fetch_and_store(self, tournament_name: str = "") -> dict:
        """Fetch odds and store in database."""
        from database.db_manager import (
            insert_odds_line, upsert_odds_consensus, normalize_name
        )

        result = self.fetch_odds()

        if result["lines"]:
            for line in result["lines"]:
                insert_odds_line(
                    player_name=line["player_name"],
                    market_type=line["market_type"],
                    bookmaker=line["bookmaker"],
                    odds_american=line["odds_american"],
                    odds_decimal=line["odds_decimal"],
                    implied_prob=line["implied_prob"],
                    tournament_name=tournament_name,
                )

        if result["consensus"]:
            for c in result["consensus"]:
                upsert_odds_consensus(
                    player_name=c["player_name"],
                    market_type=c["market_type"],
                    consensus_prob=c["consensus_prob"],
                    num_books=c["num_books"],
                    min_prob=c["min_prob"],
                    max_prob=c["max_prob"],
                    tournament_name=tournament_name,
                )

        return result

    @staticmethod
    def _american_to_decimal(american: int) -> float:
        if american >= 100:
            return 1.0 + american / 100.0
        elif american <= -100:
            return 1.0 + 100.0 / abs(american)
        return 2.0  # fallback
