"""Line shading detection — Identify when books shade lines against us."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class LineShadingDetector:
    """Detect if books are shading lines in response to our activity."""

    def detect_shading(
        self,
        our_bets: List[dict],
        market_lines: List[dict],
    ) -> dict:
        """Detect line shading patterns.

        Args:
            our_bets: List of {player, market_type, bet_time, direction, book}
            market_lines: List of {player, market_type, time, line, source}

        Returns:
            Analysis of potential shading patterns.
        """
        if len(our_bets) < 10 or len(market_lines) < 20:
            return {"has_data": False, "n_bets": len(our_bets)}

        # Track line movements after our bets
        post_bet_movements = []

        for bet in our_bets:
            # Find lines from this book within 2 hours after our bet
            matching_lines = [
                m for m in market_lines
                if m["player"] == bet["player"]
                and m["market_type"] == bet["market_type"]
                and m["source"] == bet.get("book", "")
            ]

            if len(matching_lines) < 2:
                continue

            # Sort by time
            matching_lines.sort(key=lambda x: x["time"])

            # Find line before and after our bet
            pre_bet = [m for m in matching_lines if m["time"] < bet["bet_time"]]
            post_bet = [m for m in matching_lines if m["time"] > bet["bet_time"]]

            if pre_bet and post_bet:
                pre_line = pre_bet[-1]["line"]
                post_line = post_bet[0]["line"]
                movement = post_line - pre_line

                # Shading: line moves against our position after bet
                is_adverse = (
                    (bet["direction"] == "over" and movement < 0) or
                    (bet["direction"] == "under" and movement > 0)
                )

                post_bet_movements.append({
                    "movement": movement,
                    "is_adverse": is_adverse,
                    "book": bet.get("book", "unknown"),
                })

        if not post_bet_movements:
            return {"has_data": False, "n_analyzed": 0}

        n_adverse = sum(1 for m in post_bet_movements if m["is_adverse"])
        adverse_rate = n_adverse / len(post_bet_movements)

        # By book
        by_book = defaultdict(list)
        for m in post_bet_movements:
            by_book[m["book"]].append(m)

        book_shading = {}
        for book, movements in by_book.items():
            book_adverse = sum(1 for m in movements if m["is_adverse"])
            book_shading[book] = {
                "n_bets": len(movements),
                "adverse_rate": round(book_adverse / len(movements), 4),
                "is_shading": book_adverse / len(movements) > 0.6,
            }

        return {
            "has_data": True,
            "n_analyzed": len(post_bet_movements),
            "overall_adverse_rate": round(adverse_rate, 4),
            "is_being_shaded": adverse_rate > 0.55,
            "by_book": book_shading,
            "expected_random_adverse_rate": 0.50,
        }
