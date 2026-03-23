"""Structural edge component — Field correlation and wave advantage.

Golf-specific structural factors that create betting edge:
  - Bet correlation across field (diversification vs concentration)
  - Wave advantage (AM vs PM tee times + weather)
  - Course type advantage (specific course types where model excels)
  - Field strength effects (weak vs strong fields)
  - Tournament structure (cut effects, 36-hole events, etc.)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List

import numpy as np
from scipy import stats as scipy_stats

from edge_analysis.schemas import GolfBetRecord, EdgeComponent

logger = logging.getLogger(__name__)


class StructuralAnalyzer:
    """Analyze structural/systematic edge factors specific to golf."""

    def analyze(self, records: List[GolfBetRecord]) -> EdgeComponent:
        """Compute structural edge from golf-specific factors."""
        if not records:
            return EdgeComponent(
                name="structural", value=0.0, confidence=0.0,
                verdict="No data to analyze.",
            )

        # Wave advantage analysis
        wave_analysis = self._wave_advantage(records)

        # Field correlation (are we overexposed to correlated outcomes?)
        correlation_analysis = self._field_correlation(records)

        # Weather condition analysis
        weather_analysis = self._weather_conditions(records)

        # Course type analysis
        course_analysis = self._course_type_analysis(records)

        # Tournament structure
        structure_analysis = self._tournament_structure(records)

        # Aggregate structural edge
        structural_edge = (
            wave_analysis.get("wave_edge_cents", 0.0) * 0.3 +
            correlation_analysis.get("diversification_benefit_cents", 0.0) * 0.2 +
            weather_analysis.get("weather_edge_cents", 0.0) * 0.3 +
            course_analysis.get("course_edge_cents", 0.0) * 0.2
        )

        confidence = min(len(records) / 150.0, 1.0)

        if structural_edge > 2.0:
            verdict = f"Strong structural edge ({structural_edge:.1f}c) — golf-specific factors contributing"
        elif structural_edge > 0.5:
            verdict = f"Moderate structural edge ({structural_edge:.1f}c)"
        else:
            verdict = f"Minimal structural edge ({structural_edge:.1f}c)"

        return EdgeComponent(
            name="structural",
            value=round(structural_edge, 2),
            confidence=round(confidence, 3),
            details={
                "wave_advantage": wave_analysis,
                "field_correlation": correlation_analysis,
                "weather_conditions": weather_analysis,
                "course_type": course_analysis,
                "tournament_structure": structure_analysis,
            },
            verdict=verdict,
        )

    def _wave_advantage(self, records: List[GolfBetRecord]) -> dict:
        """Analyze AM vs PM wave advantage in betting outcomes."""
        am_bets = [r for r in records if r.wave == "AM"]
        pm_bets = [r for r in records if r.wave == "PM"]

        if len(am_bets) < 5 or len(pm_bets) < 5:
            return {"has_data": False, "wave_edge_cents": 0.0}

        clv_am = float(np.mean([r.clv_cents for r in am_bets]))
        clv_pm = float(np.mean([r.clv_cents for r in pm_bets]))
        wr_am = float(np.mean([r.actual_outcome for r in am_bets]))
        wr_pm = float(np.mean([r.actual_outcome for r in pm_bets]))

        # Wave edge: how much better do we perform in one wave
        wave_edge = abs(clv_am - clv_pm)

        # Statistical test
        am_clvs = np.array([r.clv_cents for r in am_bets])
        pm_clvs = np.array([r.clv_cents for r in pm_bets])
        if len(am_clvs) >= 10 and len(pm_clvs) >= 10:
            t_stat, p_val = scipy_stats.ttest_ind(am_clvs, pm_clvs)
        else:
            p_val = 1.0

        return {
            "has_data": True,
            "n_am_bets": len(am_bets),
            "n_pm_bets": len(pm_bets),
            "clv_am": round(clv_am, 2),
            "clv_pm": round(clv_pm, 2),
            "win_rate_am": round(wr_am, 4),
            "win_rate_pm": round(wr_pm, 4),
            "wave_edge_cents": round(wave_edge, 2),
            "p_value": round(p_val, 4),
            "better_wave": "AM" if clv_am > clv_pm else "PM",
        }

    def _field_correlation(self, records: List[GolfBetRecord]) -> dict:
        """Analyze correlation of bets within same tournament fields."""
        # Group by tournament
        by_tournament = defaultdict(list)
        for r in records:
            by_tournament[r.tournament].append(r)

        concentration_scores = []
        intra_tournament_corrs = []

        for tourn, recs in by_tournament.items():
            if len(recs) < 2:
                continue

            # Concentration: how many bets in one tournament
            concentration_scores.append(len(recs))

            # Intra-tournament outcome correlation
            outcomes = [r.actual_outcome for r in recs]
            if len(set(outcomes)) > 1:
                # Win rate within tournament vs overall
                tourn_wr = float(np.mean(outcomes))
                overall_wr = float(np.mean([r.actual_outcome for r in records]))
                intra_tournament_corrs.append(tourn_wr - overall_wr)

        avg_concentration = float(np.mean(concentration_scores)) if concentration_scores else 0
        max_concentration = max(concentration_scores) if concentration_scores else 0

        # Diversification benefit: spread bets reduce variance
        n_tournaments = len(by_tournament)
        n_bets = len(records)
        div_ratio = n_tournaments / max(n_bets, 1)

        # Optimal: ~3-5 bets per tournament for golf
        # Over-concentration penalty
        if avg_concentration > 8:
            div_benefit = -1.0  # Too concentrated
        elif avg_concentration < 2:
            div_benefit = 0.5  # Well diversified
        else:
            div_benefit = 1.0  # Good balance

        return {
            "n_tournaments": n_tournaments,
            "avg_bets_per_tournament": round(avg_concentration, 1),
            "max_bets_per_tournament": max_concentration,
            "diversification_ratio": round(div_ratio, 4),
            "diversification_benefit_cents": round(div_benefit, 2),
            "intra_tournament_corr": round(float(np.mean(intra_tournament_corrs)), 4) if intra_tournament_corrs else 0.0,
        }

    def _weather_conditions(self, records: List[GolfBetRecord]) -> dict:
        """Analyze performance by weather conditions."""
        by_weather = defaultdict(list)
        for r in records:
            by_weather[r.weather_conditions].append(r)

        result = {}
        overall_clv = float(np.mean([r.clv_cents for r in records]))

        for condition, recs in by_weather.items():
            clvs = [r.clv_cents for r in recs]
            avg_clv = float(np.mean(clvs))
            result[condition] = {
                "n_bets": len(recs),
                "avg_clv": round(avg_clv, 2),
                "win_rate": round(float(np.mean([r.actual_outcome for r in recs])), 4),
                "edge_vs_overall": round(avg_clv - overall_clv, 2),
            }

        # Weather edge: how much more edge in non-normal conditions
        non_normal = [r for r in records if r.weather_conditions != "normal"]
        normal = [r for r in records if r.weather_conditions == "normal"]

        if non_normal and normal:
            weather_edge = float(np.mean([r.clv_cents for r in non_normal])) - float(np.mean([r.clv_cents for r in normal]))
        else:
            weather_edge = 0.0

        return {
            "by_condition": result,
            "weather_edge_cents": round(weather_edge, 2),
            "n_weather_bets": len(non_normal),
        }

    def _course_type_analysis(self, records: List[GolfBetRecord]) -> dict:
        """Analyze performance by course type."""
        by_course = defaultdict(list)
        for r in records:
            by_course[r.course_id].append(r)

        course_clvs = {}
        for course, recs in by_course.items():
            if len(recs) >= 3:
                course_clvs[course] = {
                    "n_bets": len(recs),
                    "avg_clv": round(float(np.mean([r.clv_cents for r in recs])), 2),
                    "win_rate": round(float(np.mean([r.actual_outcome for r in recs])), 4),
                }

        # Course edge: variance in CLV across courses indicates structural advantage
        if len(course_clvs) >= 3:
            clv_values = [v["avg_clv"] for v in course_clvs.values()]
            course_edge = max(clv_values) - min(clv_values)
        else:
            course_edge = 0.0

        return {
            "n_courses": len(by_course),
            "by_course": course_clvs,
            "course_edge_cents": round(course_edge, 2),
        }

    def _tournament_structure(self, records: List[GolfBetRecord]) -> dict:
        """Analyze how tournament structure affects edge."""
        by_market = defaultdict(list)
        for r in records:
            by_market[r.market_type].append(r)

        # Make cut bets: structural advantage from modeling cut dynamics
        cut_bets = by_market.get("make_cut", [])
        non_cut = [r for r in records if r.market_type != "make_cut"]

        cut_clv = float(np.mean([r.clv_cents for r in cut_bets])) if cut_bets else 0.0
        other_clv = float(np.mean([r.clv_cents for r in non_cut])) if non_cut else 0.0

        return {
            "n_cut_bets": len(cut_bets),
            "cut_bet_clv": round(cut_clv, 2),
            "other_bet_clv": round(other_clv, 2),
            "cut_structural_edge": round(cut_clv - other_clv, 2),
        }
