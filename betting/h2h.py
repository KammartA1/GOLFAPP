"""
H2H Matchup Analyzer
Books price head-to-head matchups lazily based on world ranking.
This is the most consistent ROI market in golf.

Key edges:
  - Course fit disparity (model-driven vs rank-driven)
  - Recent form divergence the book hasn't priced
  - Tee time / weather split between the two players
  - Playing history between specific players at specific venues
"""
import logging
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from betting.kelly import KellyModel, american_to_implied_prob

log = logging.getLogger(__name__)


@dataclass
class H2HMatchup:
    player_a: str
    player_b: str
    course: str

    # Model projections
    sg_a: float
    sg_b: float
    edge_sg: float          # sg_a - sg_b

    # Model win probabilities for this matchup
    win_prob_a: float       # probability A beats B
    win_prob_b: float       # probability B beats A

    # Odds from book
    odds_a: Optional[int] = None
    odds_b: Optional[int] = None
    book: str = "TBD"

    # Edge calculations
    implied_a: Optional[float] = None
    implied_b: Optional[float] = None
    edge_a: Optional[float] = None
    edge_b: Optional[float] = None

    # Context
    world_rank_a: Optional[int] = None
    world_rank_b: Optional[int] = None
    form_a: str = ""
    form_b: str = ""
    course_fit_a: float = 50.0
    course_fit_b: float = 50.0
    tee_time_adv_a: float = 0.0
    notes: str = ""

    def best_side(self) -> tuple[str, float, float]:
        """Return (best_side_player, edge, odds) or None if no edge."""
        if self.edge_a is not None and self.edge_b is not None:
            if self.edge_a >= self.edge_b and self.edge_a > 0.04:
                return (self.player_a, self.edge_a, self.odds_a)
            elif self.edge_b > self.edge_a and self.edge_b > 0.04:
                return (self.player_b, self.edge_b, self.odds_b)
        return None


class H2HAnalyzer:
    """
    Analyze and find edge in head-to-head matchup markets.
    """

    # SG edge to win probability conversion for H2H
    # Derived from: historical matchup data shows each +0.1 SG ≈ +3-4% win probability in 72hr H2H
    # [v6.0] Increased sensitivity from 0.32 → 0.38 for sharper H2H edge detection
    SG_TO_H2H_WIN_SLOPE = 0.38   # per 1 SG stroke difference

    def __init__(self):
        pass

    def sg_to_h2h_prob(self, sg_a: float, sg_b: float, n_rounds: int = 4) -> float:
        """
        Convert SG projections to H2H win probability.

        Uses logistic function scaled to golf's round-to-round variance.
        4-round H2H is more predictive than single-round.
        """
        from scipy.special import expit
        import numpy as np

        sg_diff = sg_a - sg_b
        # Scale by sqrt(rounds) — more rounds = more predictive = sharper distribution
        rounds_factor = np.sqrt(n_rounds) / 2
        # Logistic sigmoid: 50% at SG diff = 0, grows with edge
        win_prob = expit(sg_diff * self.SG_TO_H2H_WIN_SLOPE * rounds_factor * 2.5)
        return round(float(win_prob), 4)

    def analyze_matchup(
        self,
        player_a: dict,
        player_b: dict,
        course: str,
        odds_a: int = None,
        odds_b: int = None,
        book: str = "TBD",
    ) -> H2HMatchup:
        """
        Analyze a single H2H matchup.

        player_a/b: projection dicts from ProjectionEngine.run()
            Expected keys: name, proj_sg_total, world_rank, form_trend,
                           course_fit_score, tee_time_adv
        """
        sg_a = player_a.get("proj_sg_total", 0)
        sg_b = player_b.get("proj_sg_total", 0)

        win_prob_a = self.sg_to_h2h_prob(sg_a, sg_b)
        win_prob_b = 1 - win_prob_a

        matchup = H2HMatchup(
            player_a=player_a.get("name", ""),
            player_b=player_b.get("name", ""),
            course=course,
            sg_a=sg_a,
            sg_b=sg_b,
            edge_sg=round(sg_a - sg_b, 3),
            win_prob_a=win_prob_a,
            win_prob_b=win_prob_b,
            odds_a=odds_a,
            odds_b=odds_b,
            book=book,
            world_rank_a=player_a.get("world_rank"),
            world_rank_b=player_b.get("world_rank"),
            form_a=player_a.get("form_trend", ""),
            form_b=player_b.get("form_trend", ""),
            course_fit_a=player_a.get("course_fit_score", 50),
            course_fit_b=player_b.get("course_fit_score", 50),
            tee_time_adv_a=player_a.get("tee_time_adv", 0),
        )

        # Calculate edge if odds provided
        if odds_a:
            matchup.implied_a = american_to_implied_prob(odds_a)
            matchup.edge_a = round(win_prob_a - matchup.implied_a, 4)
        if odds_b:
            matchup.implied_b = american_to_implied_prob(odds_b)
            matchup.edge_b = round(win_prob_b - matchup.implied_b, 4)

        # Build notes
        notes = []
        if abs(sg_a - sg_b) > 0.3:
            leader = player_a["name"] if sg_a > sg_b else player_b["name"]
            notes.append(f"SG edge: {leader} by {abs(sg_a - sg_b):.2f}")
        fit_diff = abs(matchup.course_fit_a - matchup.course_fit_b)
        if fit_diff > 10:
            better = player_a["name"] if matchup.course_fit_a > matchup.course_fit_b else player_b["name"]
            notes.append(f"Course fit advantage: {better} (+{fit_diff:.0f} pts)")
        if matchup.form_a != matchup.form_b:
            notes.append(f"Form: {player_a['name']}={matchup.form_a} vs {player_b['name']}={matchup.form_b}")
        matchup.notes = " | ".join(notes)

        return matchup

    def scan_field_matchups(
        self,
        projections_df: pd.DataFrame,
        matchup_odds: list[dict],     # [{"player_a": str, "player_b": str, "odds_a": int, "odds_b": int, "book": str}]
        course: str,
        min_edge: float = 0.04,
    ) -> list[H2HMatchup]:
        """
        Analyze all available H2H matchups and return those with edge.

        matchup_odds: scraped or manually input H2H odds from sportsbook
        """
        proj_map = {row["name"]: row for _, row in projections_df.iterrows()}
        edged_matchups = []

        for m in matchup_odds:
            pa_name = m.get("player_a", "")
            pb_name = m.get("player_b", "")
            pa = proj_map.get(pa_name)
            pb = proj_map.get(pb_name)

            if pa is None or pb is None:
                log.debug(f"Missing projection for {pa_name} or {pb_name}")
                continue

            matchup = self.analyze_matchup(
                dict(pa), dict(pb), course,
                odds_a=m.get("odds_a"),
                odds_b=m.get("odds_b"),
                book=m.get("book", "TBD"),
            )

            best = matchup.best_side()
            if best:
                edged_matchups.append(matchup)

        edged_matchups.sort(key=lambda m: max(
            m.edge_a or 0, m.edge_b or 0
        ), reverse=True)

        log.info(f"Found {len(edged_matchups)} H2H matchups with edge > {min_edge:.0%}")
        return edged_matchups

    def generate_synthetic_matchups(
        self,
        projections_df: pd.DataFrame,
        course: str,
        similar_rank_range: int = 10,
    ) -> pd.DataFrame:
        """
        Generate H2H matchup analysis for players near each other in rank.
        Useful even without book odds — shows where you'd want to look for lines.
        """
        df = projections_df.sort_values("proj_sg_total", ascending=False).reset_index(drop=True)
        rows = []

        for i in range(len(df)):
            for j in range(i + 1, min(i + similar_rank_range, len(df))):
                a = df.iloc[i].to_dict()
                b = df.iloc[j].to_dict()

                matchup = self.analyze_matchup(a, b, course)
                rows.append({
                    "player_a":       matchup.player_a,
                    "player_b":       matchup.player_b,
                    "sg_a":           matchup.sg_a,
                    "sg_b":           matchup.sg_b,
                    "sg_edge":        matchup.edge_sg,
                    "win_prob_a":     matchup.win_prob_a,
                    "win_prob_b":     matchup.win_prob_b,
                    "course_fit_a":   matchup.course_fit_a,
                    "course_fit_b":   matchup.course_fit_b,
                    "form_a":         matchup.form_a,
                    "form_b":         matchup.form_b,
                    "rank_diff":      abs((a.get("world_rank") or 999) - (b.get("world_rank") or 999)),
                    "notes":          matchup.notes,
                })

        result = pd.DataFrame(rows)
        if not result.empty:
            # Find cases where rank-based pricing would be wrong
            # (lower-ranked player is the MODEL's favorite)
            result["upset_flag"] = (
                (result["world_rank_a"] if "world_rank_a" in result.columns else result.get("rank_diff", 0)) > 0
            )

        log.info(f"Generated {len(rows)} synthetic H2H matchups")
        return result

    def print_top_matchups(self, matchups: list[H2HMatchup], n: int = 10):
        """Pretty print top H2H edges."""
        from rich.table import Table
        from rich.console import Console
        console = Console()

        table = Table(title=f"⚔️  Top {n} H2H Matchup Edges", show_lines=True)
        table.add_column("Side", style="bold green")
        table.add_column("vs.")
        table.add_column("Model%", justify="right")
        table.add_column("Implied%", justify="right")
        table.add_column("Edge", justify="right", style="green")
        table.add_column("Odds")
        table.add_column("SG Diff", justify="right")
        table.add_column("Notes")

        shown = 0
        for m in matchups:
            if shown >= n:
                break
            best = m.best_side()
            if not best:
                continue
            player, edge, odds = best
            is_a = player == m.player_a
            model_prob = m.win_prob_a if is_a else m.win_prob_b
            implied = m.implied_a if is_a else m.implied_b
            opponent = m.player_b if is_a else m.player_a

            table.add_row(
                player, opponent,
                f"{model_prob:.1%}",
                f"{implied:.1%}" if implied else "N/A",
                f"{edge:+.1%}",
                f"{odds:+d}" if odds else "N/A",
                f"{m.edge_sg:+.2f}",
                m.notes[:40],
            )
            shown += 1

        console.print(table)
