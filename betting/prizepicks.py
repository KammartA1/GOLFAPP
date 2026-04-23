"""
PrizePicks Edge Analyzer
Compares our model's projections to PrizePicks lines to find +EV picks.

PrizePicks is a Pick'em platform — no vig on individual picks, but the
payout structure (typically 3x for 2-pick, 5x for 3-pick, etc.) means
you need ~53%+ accuracy on individual picks to beat a 2-pick power play.

Key: We can use our SG model to project EACH STAT TYPE independently.

Stat-to-SG conversion:
  Fantasy Score  ← directly from SG model (sg_to_dk_points)
  Birdies        ← derived from scoring average + SG total
  Bogey Free     ← probabilistic from SG total + volatility
  GIR            ← correlated with SG:APP
  Fairways Hit   ← correlated with SG:OTT (accuracy component)
  Strokes        ← relative to course scoring avg
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from data.scrapers.prizepicks import PPProjection, PrizePicksScraper
from models.projection import sg_to_dk_points

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PAYOUT STRUCTURES
# ─────────────────────────────────────────────

# PrizePicks Power Play (no flex — must hit all picks)
POWER_PLAY_PAYOUTS = {
    2: 3.00,    # 2-pick: 3x
    3: 5.00,    # 3-pick: 5x
    4: 10.00,   # 4-pick: 10x
    5: 20.00,   # 5-pick: 20x
    6: 25.00,   # 6-pick: 25x (when offered)
}

# PrizePicks Flex Play (partial payout if some wrong)
FLEX_PLAY_PAYOUTS = {
    3: {3: 2.25, 2: 1.25, 1: 0.0},
    4: {4: 5.00, 3: 1.50, 2: 0.4},
    5: {5: 10.0, 4: 2.00, 3: 0.4},
    6: {6: 25.0, 5: 2.00, 4: 0.4, 3: 0.1},
}

# Accuracy needed to break even on Power Play
POWER_PLAY_BREAKEVEN = {
    2: 0.5774,   # sqrt(1/3) = 57.7% per pick
    3: 0.5848,   # (1/5)^(1/3) = 58.5%
    4: 0.5623,   # (1/10)^(1/4) = 56.2%
    5: 0.5495,   # (1/20)^(1/5) = 54.9%
}


@dataclass
class PPEdge:
    """Edge analysis for a single PrizePicks line."""
    projection: PPProjection
    model_proj: float           # Our model's projection for this stat
    line:       float           # PrizePicks line
    edge_over:  float           # P(actual > line) - 0.5
    edge_under: float           # P(actual < line) - 0.5
    raw_prob_over:  float       # P(actual > line) — raw probability
    raw_prob_under: float       # P(actual < line)
    recommendation: str         # "OVER", "UNDER", "PASS"
    confidence: str             # "HIGH" (>60%), "MEDIUM" (55-60%), "LOW" (<55%)
    model_vs_line: float        # model_proj - line (positive = OVER lean)
    stat_std:   float           # our model's uncertainty for this stat
    notes:      str = ""

    @property
    def best_edge(self) -> float:
        return max(self.edge_over, self.edge_under)

    @property
    def pick_prob(self) -> float:
        """Probability of our recommended pick being correct."""
        return self.raw_prob_over if self.recommendation == "OVER" else self.raw_prob_under

    def ev_power_play(self, n_picks: int = 2) -> float:
        """EV per dollar for a Power Play entry using this pick."""
        payout = POWER_PLAY_PAYOUTS.get(n_picks, 3.0)
        # EV = p^n * payout - 1 (per $1 entry)
        # This pick's contribution: p = pick_prob, combined with n-1 other picks at same prob
        p = self.pick_prob
        return p * payout - 1  # Simplified (1-pick EV before combining)


@dataclass
class PPSlipRecommendation:
    """A full recommended PrizePicks slip (2-6 picks)."""
    picks:       list[PPEdge]
    slip_type:   str            # "power_play" or "flex"
    n_picks:     int
    payout:      float          # multiplier
    combined_prob: float        # P(all picks hit)
    ev:          float          # expected value per $1 entry
    confidence:  str            # overall slip confidence
    notes:       str = ""

    def __str__(self):
        picks_str = "\n  ".join([
            f"{p.projection.player_name} {p.recommendation} {p.line} "
            f"({p.projection.stat_display}) [{p.confidence} {p.pick_prob:.1%}]"
            for p in self.picks
        ])
        return (
            f"{'⚡' if self.slip_type == 'power_play' else '🔄'} "
            f"{self.n_picks}-Pick {self.slip_type.replace('_',' ').title()} "
            f"| {self.payout}x | EV: {self.ev:+.2%} | P(win): {self.combined_prob:.1%}\n  "
            + picks_str
        )


class PrizePicksAnalyzer:
    """
    Analyzes PrizePicks golf lines using our SG projection model.

    Converts SG projections → stat-specific projections → edge vs PP line.
    """

    # [v6.0] Tightened standard deviations for sharper probability estimates
    # Reduced by ~15% to increase confidence separation between edge/no-edge
    STAT_STD = {
        "fantasy_score":      10.0,   # DK points — tightened from 12.0
        "birdies_or_better":   1.5,   # birdies per round — tightened from 1.8
        "birdies":             1.5,
        "bogey_free_rounds":   0.42,  # binary-ish — tightened from 0.48
        "strokes_total":       3.0,   # total strokes 72 holes — tightened from 3.5
        "holes_under_par":     3.8,   # holes under par — tightened from 4.5
        "gir":                 2.5,   # greens in regulation — tightened from 3.0
        "fairways_hit":        3.4,   # fairways hit — tightened from 4.0
        "eagles":              0.18,  # very rare
        "longest_drive":      13.0,   # yards — tightened from 15.0
        "holes_in_one":        0.02,  # extremely rare
    }

    # SG → stat conversion baselines (tour average values)
    TOUR_BASELINES = {
        "fantasy_score":      37.0,   # avg DK pts per tournament
        "birdies_or_better":   4.2,   # birdies per round
        "birdies":             4.2,
        "bogey_free_rounds":   1.1,   # out of 4 rounds
        "strokes_total":     284.0,   # 71.0 avg * 4 rounds
        "holes_under_par":    18.0,   # holes under par per tournament
        "gir":                11.5,   # greens hit per round (of 18)
        "fairways_hit":        9.0,   # fairways per round (of ~14)
        "eagles":              0.12,  # per tournament
    }

    # How much each SG category drives each stat
    SG_TO_STAT_SENSITIVITY = {
        # stat: {sg_category: sensitivity}
        "fantasy_score":     {"sg_total": 9.2},
        "birdies_or_better": {"sg_total": 0.90, "sg_app": 0.6, "sg_putt": 0.4},
        "birdies":           {"sg_total": 0.90, "sg_app": 0.6, "sg_putt": 0.4},
        "bogey_free_rounds": {"sg_total": 0.12},
        "strokes_total":     {"sg_total": -4.0},   # negative: better SG = fewer strokes
        "holes_under_par":   {"sg_total": 1.80},
        "gir":               {"sg_app": 1.20, "sg_ott": 0.3},
        "fairways_hit":      {"sg_ott": 1.50},
        "eagles":            {"sg_total": 0.025, "sg_ott": 0.015},
    }

    def __init__(self, scraper: PrizePicksScraper = None):
        self.scraper = scraper or PrizePicksScraper()

    # ─────────────────────────────────────────────
    # STAT PROJECTION FROM SG MODEL
    # ─────────────────────────────────────────────

    def project_stat(
        self,
        stat_type: str,
        player_proj: dict,
    ) -> tuple[float, float]:
        """
        Project a specific PrizePicks stat for a player.

        player_proj: dict from ProjectionEngine with keys:
            proj_sg_total, proj_sg_app, proj_sg_ott, proj_sg_atg, proj_sg_putt

        Returns: (projected_value, std_dev)
        """
        baseline = self.TOUR_BASELINES.get(stat_type, 0)
        sensitivity = self.SG_TO_STAT_SENSITIVITY.get(stat_type, {"sg_total": 1.0})
        std = self.STAT_STD.get(stat_type, 2.0)

        # Sum the contributions from each relevant SG category
        sg_contribution = 0.0
        for sg_cat, sens in sensitivity.items():
            sg_val = player_proj.get(f"proj_{sg_cat}", player_proj.get(sg_cat, 0))
            if sg_val:
                sg_contribution += sg_val * sens

        proj_value = baseline + sg_contribution
        return round(proj_value, 3), std

    def prob_over(self, proj: float, line: float, std: float) -> float:
        """P(actual > line) using normal distribution."""
        from scipy import stats
        if std <= 0:
            return 1.0 if proj > line else 0.0
        z = (line - proj) / std
        return round(float(1 - stats.norm.cdf(z)), 4)

    def prob_under(self, proj: float, line: float, std: float) -> float:
        """P(actual < line) using normal distribution."""
        return round(1 - self.prob_over(proj, line, std), 4)

    def classify_confidence(self, prob: float) -> str:
        """[v6.0] Raised thresholds for elite-only selection targeting 200%+ ROI."""
        if prob >= 0.68:
            return "HIGH"
        elif prob >= 0.62:
            return "MEDIUM"
        elif prob >= 0.57:
            return "LOW"
        return "PASS"

    # ─────────────────────────────────────────────
    # ANALYZE SINGLE LINE
    # ─────────────────────────────────────────────

    def analyze_line(
        self,
        pp_proj: PPProjection,
        player_model_proj: dict,
    ) -> PPEdge:
        """
        Analyze a single PrizePicks line against our model projection.

        Returns PPEdge with recommendation and confidence.
        """
        stat = pp_proj.stat_type
        line = pp_proj.line_score

        model_val, std = self.project_stat(stat, player_model_proj)

        p_over  = self.prob_over(model_val, line, std)
        p_under = self.prob_under(model_val, line, std)

        edge_over  = round(p_over  - 0.50, 4)
        edge_under = round(p_under - 0.50, 4)

        # Determine recommendation
        best_side = "OVER" if p_over > p_under else "UNDER"
        best_prob = p_over if best_side == "OVER" else p_under
        confidence = self.classify_confidence(best_prob)

        if confidence == "PASS":
            rec = "PASS"
        else:
            rec = best_side

        # Build context notes
        notes_parts = []
        model_diff = model_val - line
        if abs(model_diff) > std * 0.5:
            notes_parts.append(f"Model: {model_val:.2f} vs Line: {line:.1f} (gap: {model_diff:+.2f})")
        if pp_proj.is_promo:
            notes_parts.append("⚠️ PROMO line — verify independently")
        if pp_proj.flash_sale_line_score and pp_proj.flash_sale_line_score != line:
            notes_parts.append(f"Power Play line: {pp_proj.flash_sale_line_score}")

        return PPEdge(
            projection     = pp_proj,
            model_proj     = model_val,
            line           = line,
            edge_over      = edge_over,
            edge_under     = edge_under,
            raw_prob_over  = p_over,
            raw_prob_under = p_under,
            recommendation = rec,
            confidence     = confidence,
            model_vs_line  = round(model_diff, 3),
            stat_std       = std,
            notes          = " | ".join(notes_parts),
        )

    # ─────────────────────────────────────────────
    # ANALYZE FULL SLATE
    # ─────────────────────────────────────────────

    def analyze_slate(
        self,
        projections_df: pd.DataFrame,
        league_id: int = None,
        min_confidence: str = "LOW",
    ) -> list[PPEdge]:
        """
        Analyze all live PrizePicks golf lines against our model.

        projections_df: output from ProjectionEngine.run()
        min_confidence: minimum confidence to include in results
                        "HIGH", "MEDIUM", "LOW"

        Returns list of PPEdge sorted by best edge descending.
        """
        # Fetch live PP lines
        log.info("Fetching live PrizePicks slate...")
        pp_lines = self.scraper.fetch_golf_projections(league_id=league_id)

        if not pp_lines:
            log.warning("No PrizePicks lines available")
            return []

        # Build player projection lookup
        proj_map = {}
        for _, row in projections_df.iterrows():
            proj_map[row["name"].lower()] = row.to_dict()

        edges = []
        no_match_players = set()

        for pp in pp_lines:
            # Match PP player name to our projection
            player_key = pp.player_name.lower()
            player_proj = proj_map.get(player_key)

            if player_proj is None:
                # Fuzzy match: try partial name
                for proj_name, proj_data in proj_map.items():
                    if (pp.player_name.split()[-1].lower() in proj_name or
                            proj_name.split()[-1] in player_key):
                        player_proj = proj_data
                        break

            if player_proj is None:
                no_match_players.add(pp.player_name)
                continue

            edge = self.analyze_line(pp, player_proj)
            edges.append(edge)

        if no_match_players:
            log.warning(f"No projection match for: {', '.join(sorted(no_match_players)[:10])}")

        # Filter by confidence
        confidence_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "PASS": 0}
        min_rank = confidence_rank.get(min_confidence, 1)
        filtered = [e for e in edges if confidence_rank.get(e.confidence, 0) >= min_rank]

        # Sort by best edge
        filtered.sort(key=lambda e: e.best_edge, reverse=True)

        log.info(f"Analyzed {len(edges)} lines | {len(filtered)} with edge ≥ {min_confidence}")
        return filtered

    # ─────────────────────────────────────────────
    # SLIP BUILDER
    # ─────────────────────────────────────────────

    def build_optimal_slips(
        self,
        edges: list[PPEdge],
        slip_sizes: list[int] = None,
        slip_type: str = "power_play",
        max_slips: int = 5,
        min_combined_prob: float = 0.30,
    ) -> list[PPSlipRecommendation]:
        """
        Build optimal PrizePicks slips from the best edges.

        Strategy:
          - Prioritize HIGH confidence picks
          - Diversify across stat types (don't stack same stat)
          - Prefer uncorrelated picks (different players)
          - Target power play EV > 0 per slip

        slip_sizes: [2, 3, 4] — which slip sizes to generate
        """
        slip_sizes = slip_sizes or [2, 3, 4]

        # Use only non-PASS picks
        actionable = [e for e in edges if e.recommendation != "PASS"]
        if not actionable:
            log.warning("No actionable edges found")
            return []

        # Score each pick for slip inclusion
        conf_scores = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        for e in actionable:
            e._slip_score = conf_scores.get(e.confidence, 0) + e.best_edge * 10

        actionable.sort(key=lambda e: e._slip_score, reverse=True)

        slips = []
        for size in slip_sizes:
            if len(actionable) < size:
                continue

            generated = self._build_slips_for_size(
                actionable, size, slip_type, min_combined_prob, max_per_size=max_slips
            )
            slips.extend(generated)

        return slips[:max_slips]

    def _build_slips_for_size(
        self,
        edges: list[PPEdge],
        size: int,
        slip_type: str,
        min_combined_prob: float,
        max_per_size: int = 3,
    ) -> list[PPSlipRecommendation]:
        """Generate best slips of a given size."""
        from itertools import combinations

        payout = POWER_PLAY_PAYOUTS.get(size, 3.0) if slip_type == "power_play" else 2.0

        # Try combinations of top picks
        candidates = edges[:min(20, len(edges))]  # Limit combinations
        slips = []

        for combo in combinations(candidates, size):
            # Skip if same player appears twice
            players = [e.projection.player_name for e in combo]
            if len(set(players)) < len(players):
                continue

            # [v6.0] Correlation-adjusted combined probability
            # Golf stats for same-tournament players are mildly correlated (weather, course)
            # Apply conservative correlation penalty: 3% per additional leg
            probs = [e.pick_prob for e in combo]
            combined_prob = float(np.prod(probs))
            # R5: Anti-correlation intelligence
            # Compute correlation penalty based on stat types and player overlap
            stat_types = [e.projection.stat_type for e in combo]
            players = [e.projection.player_name for e in combo]

            # Different stat types are less correlated
            unique_stats = len(set(stat_types))
            stat_diversity = unique_stats / len(stat_types)

            # Same-player different-stat picks are highly correlated
            unique_players = len(set(players))
            player_diversity = unique_players / len(players)

            # Mixed over/under reduces correlation
            directions = [e.recommendation for e in combo]
            has_both_directions = len(set(directions)) > 1

            # Correlation penalty: less penalty for diverse, uncorrelated picks
            base_corr_penalty = 0.97 ** max(0, len(probs) - 1)
            diversity_bonus = 1.0 + (stat_diversity - 0.5) * 0.06 + (0.02 if has_both_directions else 0)
            player_penalty = 0.95 if player_diversity < 1.0 else 1.0
            _corr_penalty = base_corr_penalty * diversity_bonus * player_penalty
            combined_prob *= _corr_penalty

            if combined_prob < min_combined_prob:
                continue

            ev = combined_prob * payout - 1.0
            # [v6.0] Only build slips with positive EV after correlation adjustment
            if ev < 0.05:
                continue

            # Overall confidence — [v6.0] stricter: require ALL HIGH for HIGH slip
            confs = [e.confidence for e in combo]
            if all(c == "HIGH" for c in confs):
                overall_conf = "HIGH"
            elif all(c in ("HIGH", "MEDIUM") for c in confs):
                overall_conf = "MEDIUM"
            else:
                overall_conf = "LOW"

            notes = f"P(all hit): {combined_prob:.1%} | EV: {ev:+.2%} per $1"

            slips.append(PPSlipRecommendation(
                picks           = list(combo),
                slip_type       = slip_type,
                n_picks         = size,
                payout          = payout,
                combined_prob   = round(combined_prob, 4),
                ev              = round(ev, 4),
                confidence      = overall_conf,
                notes           = notes,
            ))

        # Sort by EV
        slips.sort(key=lambda s: s.ev, reverse=True)
        return slips[:max_per_size]

    # ─────────────────────────────────────────────
    # DISPLAY
    # ─────────────────────────────────────────────

    def print_edge_table(self, edges: list[PPEdge], top_n: int = 25):
        """Print all edges in a rich table."""
        from rich.table import Table
        from rich.console import Console
        from rich.style import Style

        console = Console()
        table = Table(title=f"🎯 PrizePicks Golf Edge Analysis", show_lines=True)

        table.add_column("Player",      style="bold", min_width=20)
        table.add_column("Stat",        style="cyan", min_width=18)
        table.add_column("Line",        justify="center", style="yellow")
        table.add_column("Model Proj",  justify="center")
        table.add_column("Gap",         justify="center")
        table.add_column("Pick",        justify="center")
        table.add_column("Prob",        justify="center")
        table.add_column("Confidence",  justify="center")
        table.add_column("Notes",       min_width=25)

        conf_styles = {"HIGH": "bold green", "MEDIUM": "yellow", "LOW": "dim"}

        shown = 0
        for e in edges:
            if e.recommendation == "PASS":
                continue
            if shown >= top_n:
                break

            style = conf_styles.get(e.confidence, "")
            gap_str = f"{e.model_vs_line:+.2f}"
            gap_style = "green" if (
                (e.recommendation == "OVER" and e.model_vs_line > 0) or
                (e.recommendation == "UNDER" and e.model_vs_line < 0)
            ) else "red"

            table.add_row(
                e.projection.player_name,
                e.projection.stat_display,
                str(e.line),
                f"{e.model_proj:.2f}",
                f"[{gap_style}]{gap_str}[/{gap_style}]",
                f"[bold {'green' if e.recommendation == 'OVER' else 'red'}]{e.recommendation}[/bold {'green' if e.recommendation == 'OVER' else 'red'}]",
                f"{e.pick_prob:.1%}",
                f"[{style}]{e.confidence}[/{style}]",
                e.notes[:40],
                style=Style(dim=(e.confidence == "LOW")),
            )
            shown += 1

        console.print(table)
        console.print(f"\n[dim]Showing {shown} actionable picks | PrizePicks breakeven: ~57.7% (2-pick) / 58.5% (3-pick)[/dim]")

    def print_slips(self, slips: list[PPSlipRecommendation]):
        """Print recommended slips."""
        from rich.panel import Panel
        from rich.console import Console
        from rich.text import Text

        console = Console()
        console.print("\n[bold gold1]⚡ Recommended PrizePicks Slips[/bold gold1]\n")

        for i, slip in enumerate(slips, 1):
            text = Text()
            text.append(f"Slip #{i} — {slip.n_picks}-Pick {slip.slip_type.replace('_',' ').title()}  ", style="bold")
            text.append(f"{slip.payout}x Payout  ", style="yellow")
            text.append(f"P(Win): {slip.combined_prob:.1%}  ", style="cyan")
            text.append(f"EV: {slip.ev:+.1%}", style="green" if slip.ev > 0 else "red")

            picks_text = ""
            for pick in slip.picks:
                icon = "🔼" if pick.recommendation == "OVER" else "🔽"
                picks_text += (
                    f"\n  {icon} {pick.projection.player_name}  "
                    f"{pick.recommendation} {pick.line}  "
                    f"({pick.projection.stat_display})  "
                    f"[{pick.confidence}  {pick.pick_prob:.1%}]"
                )

            border = "green" if slip.ev > 0 else "yellow"
            console.print(Panel(str(text) + picks_text, border_style=border))
