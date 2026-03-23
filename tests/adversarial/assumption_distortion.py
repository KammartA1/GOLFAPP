"""Assumption distortion test — What breaks when fundamental assumptions are wrong?

Tests the system under scenarios where key modeling assumptions are violated:
1. Distribution assumption failures (fat tails, skewness)
2. Independence assumption failures (correlated outcomes)
3. Stationarity assumption failures (regime changes)
4. Market efficiency assumption failures
5. Golf-specific assumption failures (SG additivity, weather independence)
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class AssumptionDistortionTest:
    """Test system behavior when fundamental assumptions are violated.

    Every quant model makes assumptions. This test verifies the system
    degrades gracefully when those assumptions break down — which they
    inevitably will in practice.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def run_all(self, bet_records: List[dict], n_simulations: int = 1000) -> dict:
        """Run all assumption distortion scenarios.

        Args:
            bet_records: Historical bet data with model_prob, odds_decimal,
                        stake, pnl, status.
            n_simulations: Number of Monte Carlo paths per scenario.
        """
        if len(bet_records) < 20:
            return {"error": "Need at least 20 bet records"}

        results = {}

        results["fat_tails"] = self._test_fat_tails(bet_records, n_simulations)
        results["correlated_outcomes"] = self._test_correlated_outcomes(
            bet_records, n_simulations
        )
        results["regime_change"] = self._test_regime_change(bet_records, n_simulations)
        results["market_adaptation"] = self._test_market_adaptation(
            bet_records, n_simulations
        )
        results["sg_non_additivity"] = self._test_sg_non_additivity(bet_records)
        results["weather_correlation"] = self._test_weather_correlated_outcomes(
            bet_records, n_simulations
        )

        # Summary
        n_scenarios = len(results)
        n_failing = sum(
            1 for r in results.values()
            if r.get("verdict") in ("fragile", "broken")
        )

        overall_score = self._compute_overall_score(results)

        return {
            "n_scenarios": n_scenarios,
            "n_failing": n_failing,
            "results": results,
            "overall_score": round(overall_score, 2),
            "verdict": "robust" if overall_score >= 70 else
                       "marginal" if overall_score >= 40 else "fragile",
        }

    # ══════════════════════════════════════════════════════════════════════
    #  SCENARIO 1: Fat Tails
    # ══════════════════════════════════════════════════════════════════════

    def _test_fat_tails(
        self, bet_records: List[dict], n_simulations: int
    ) -> dict:
        """Test: What if PnL has fat tails (extreme outcomes more frequent)?

        Golf is inherently fat-tailed: outright bets have huge variance.
        Tests if Kelly sizing and risk management handle extreme outcomes.
        """
        pnl = np.array([r["pnl"] for r in bet_records])
        stakes = np.array([r["stake"] for r in bet_records])
        n_bets = len(pnl)

        # Normal distribution baseline
        normal_drawdowns = []
        for _ in range(n_simulations):
            sim_pnl = self.rng.normal(np.mean(pnl), np.std(pnl), n_bets)
            cumsum = np.cumsum(sim_pnl)
            peak = np.maximum.accumulate(cumsum)
            dd = float(np.max(peak - cumsum))
            normal_drawdowns.append(dd)

        # Student-t distribution (fat tails, df=3)
        fat_tail_drawdowns = []
        for _ in range(n_simulations):
            sim_pnl = self.rng.standard_t(df=3, size=n_bets) * np.std(pnl) + np.mean(pnl)
            cumsum = np.cumsum(sim_pnl)
            peak = np.maximum.accumulate(cumsum)
            dd = float(np.max(peak - cumsum))
            fat_tail_drawdowns.append(dd)

        normal_dd = np.array(normal_drawdowns)
        fat_dd = np.array(fat_tail_drawdowns)

        dd_ratio = float(np.median(fat_dd)) / max(float(np.median(normal_dd)), 1)

        # If fat tails increase max drawdown by more than 2x, system is sensitive
        if dd_ratio > 3.0:
            verdict = "fragile"
        elif dd_ratio > 2.0:
            verdict = "marginal"
        else:
            verdict = "robust"

        return {
            "assumption": "PnL follows normal distribution",
            "distortion": "Student-t with df=3 (fat tails)",
            "normal_median_max_dd": round(float(np.median(normal_dd)), 2),
            "fat_tail_median_max_dd": round(float(np.median(fat_dd)), 2),
            "drawdown_ratio": round(dd_ratio, 2),
            "normal_p95_dd": round(float(np.percentile(normal_dd, 95)), 2),
            "fat_tail_p95_dd": round(float(np.percentile(fat_dd, 95)), 2),
            "ruin_prob_normal": round(float(np.mean(normal_dd > np.sum(stakes) * 0.5)), 4),
            "ruin_prob_fat_tail": round(float(np.mean(fat_dd > np.sum(stakes) * 0.5)), 4),
            "verdict": verdict,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  SCENARIO 2: Correlated Outcomes
    # ══════════════════════════════════════════════════════════════════════

    def _test_correlated_outcomes(
        self, bet_records: List[dict], n_simulations: int
    ) -> dict:
        """Test: What if bet outcomes are correlated (not independent)?

        In golf, bets on the same tournament ARE correlated — if conditions
        favor one player, they may disfavor another. This tests the impact
        of outcome correlation on portfolio risk.
        """
        probs = np.array([r["model_prob"] for r in bet_records])
        odds = np.array([r["odds_decimal"] for r in bet_records])
        stakes = np.array([r["stake"] for r in bet_records])
        n_bets = len(probs)

        # Independent outcomes baseline
        independent_pnls = []
        for _ in range(n_simulations):
            outcomes = self.rng.binomial(1, probs)
            sim_pnl = float(np.sum(outcomes * stakes * (odds - 1) - (1 - outcomes) * stakes))
            independent_pnls.append(sim_pnl)

        # Correlated outcomes (rho=0.3)
        correlated_pnls = []
        rho = 0.3
        for _ in range(n_simulations):
            # Generate correlated uniform variables via Gaussian copula
            z_common = self.rng.normal()
            z_individual = self.rng.normal(size=n_bets)
            z_correlated = rho * z_common + np.sqrt(1 - rho**2) * z_individual

            from scipy import stats as sp_stats
            u_correlated = sp_stats.norm.cdf(z_correlated)
            outcomes = (u_correlated < probs).astype(float)

            sim_pnl = float(np.sum(outcomes * stakes * (odds - 1) - (1 - outcomes) * stakes))
            correlated_pnls.append(sim_pnl)

        ind = np.array(independent_pnls)
        corr = np.array(correlated_pnls)

        # Variance ratio: how much more volatile under correlation?
        var_ratio = float(np.var(corr)) / max(float(np.var(ind)), 1)

        # Loss probability increase
        loss_prob_ind = float(np.mean(ind < 0))
        loss_prob_corr = float(np.mean(corr < 0))

        if var_ratio > 3.0 or (loss_prob_corr - loss_prob_ind) > 0.20:
            verdict = "fragile"
        elif var_ratio > 1.8 or (loss_prob_corr - loss_prob_ind) > 0.10:
            verdict = "marginal"
        else:
            verdict = "robust"

        return {
            "assumption": "Bet outcomes are independent",
            "distortion": f"Gaussian copula correlation rho={rho}",
            "independent_mean_pnl": round(float(np.mean(ind)), 2),
            "correlated_mean_pnl": round(float(np.mean(corr)), 2),
            "independent_std_pnl": round(float(np.std(ind)), 2),
            "correlated_std_pnl": round(float(np.std(corr)), 2),
            "variance_ratio": round(var_ratio, 2),
            "loss_prob_independent": round(loss_prob_ind, 4),
            "loss_prob_correlated": round(loss_prob_corr, 4),
            "verdict": verdict,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  SCENARIO 3: Regime Change
    # ══════════════════════════════════════════════════════════════════════

    def _test_regime_change(
        self, bet_records: List[dict], n_simulations: int
    ) -> dict:
        """Test: What if the market regime changes mid-stream?

        Simulates edge decay: edge starts positive, then drops to zero
        or negative midway through. Tests if the system detects and
        responds to the regime change.
        """
        probs = np.array([r["model_prob"] for r in bet_records])
        market_probs = np.array([r["market_prob"] for r in bet_records])
        odds = np.array([r["odds_decimal"] for r in bet_records])
        stakes = np.array([r["stake"] for r in bet_records])
        n_bets = len(probs)

        # Baseline: edge constant throughout
        baseline_pnls = []
        for _ in range(n_simulations):
            outcomes = self.rng.binomial(1, probs)
            pnl = float(np.sum(outcomes * stakes * (odds - 1) - (1 - outcomes) * stakes))
            baseline_pnls.append(pnl)

        # Regime change: edge disappears at midpoint
        regime_pnls = []
        midpoint = n_bets // 2
        for _ in range(n_simulations):
            # First half: use model probs (edge exists)
            outcomes_1 = self.rng.binomial(1, probs[:midpoint])
            pnl_1 = np.sum(outcomes_1 * stakes[:midpoint] * (odds[:midpoint] - 1) -
                           (1 - outcomes_1) * stakes[:midpoint])

            # Second half: use market probs (edge gone — model is wrong)
            outcomes_2 = self.rng.binomial(1, market_probs[midpoint:])
            pnl_2 = np.sum(outcomes_2 * stakes[midpoint:] * (odds[midpoint:] - 1) -
                           (1 - outcomes_2) * stakes[midpoint:])

            regime_pnls.append(float(pnl_1 + pnl_2))

        base = np.array(baseline_pnls)
        regime = np.array(regime_pnls)

        pnl_drop = float(np.mean(base) - np.mean(regime))
        loss_increase = float(np.mean(regime < 0)) - float(np.mean(base < 0))

        # How quickly would a monitor detect the change?
        # Simulate rolling CLV: edge = probs - market_probs
        edges = probs - market_probs
        rolling_window = 25
        if n_bets > rolling_window * 2:
            pre_edge = float(np.mean(edges[:midpoint]))
            post_edge_model = float(np.mean(edges[midpoint:]))  # Model still thinks edge exists
            detection_delay = rolling_window  # At minimum, need a full window
        else:
            pre_edge = float(np.mean(edges))
            post_edge_model = pre_edge
            detection_delay = n_bets

        if pnl_drop > float(np.mean(base)) * 0.5:
            verdict = "fragile"
        elif pnl_drop > float(np.mean(base)) * 0.25:
            verdict = "marginal"
        else:
            verdict = "robust"

        return {
            "assumption": "Edge is stationary over time",
            "distortion": "Edge disappears at midpoint",
            "baseline_mean_pnl": round(float(np.mean(base)), 2),
            "regime_change_mean_pnl": round(float(np.mean(regime)), 2),
            "pnl_drop": round(pnl_drop, 2),
            "loss_probability_increase": round(loss_increase, 4),
            "estimated_detection_delay_bets": detection_delay,
            "pre_regime_avg_edge": round(pre_edge, 4),
            "verdict": verdict,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  SCENARIO 4: Market Adaptation
    # ══════════════════════════════════════════════════════════════════════

    def _test_market_adaptation(
        self, bet_records: List[dict], n_simulations: int
    ) -> dict:
        """Test: What if the market adapts to our strategy?

        Simulates progressive edge erosion as the market learns to
        incorporate our signals.
        """
        probs = np.array([r["model_prob"] for r in bet_records])
        market_probs = np.array([r["market_prob"] for r in bet_records])
        odds = np.array([r["odds_decimal"] for r in bet_records])
        stakes = np.array([r["stake"] for r in bet_records])
        n_bets = len(probs)

        # Decay rates to test
        decay_rates = [0.001, 0.005, 0.01, 0.02, 0.05]
        results_by_decay = []

        for decay_rate in decay_rates:
            # Edge decays exponentially over time
            time_indices = np.arange(n_bets)
            decay_factor = np.exp(-decay_rate * time_indices)

            # Effective edge shrinks over time
            original_edge = probs - market_probs
            decayed_edge = original_edge * decay_factor
            effective_probs = market_probs + decayed_edge

            # Simulate
            sim_pnls = []
            for _ in range(min(n_simulations, 500)):
                outcomes = self.rng.binomial(1, np.clip(effective_probs, 0.01, 0.99))
                pnl = float(np.sum(outcomes * stakes * (odds - 1) - (1 - outcomes) * stakes))
                sim_pnls.append(pnl)

            sim_arr = np.array(sim_pnls)

            # When does cumulative edge become negative?
            cumulative_edge = np.cumsum(decayed_edge)
            bets_until_negative = n_bets  # Default: never
            for i in range(n_bets):
                if np.mean(decayed_edge[:i+1]) < 0:
                    bets_until_negative = i
                    break

            results_by_decay.append({
                "decay_rate": decay_rate,
                "half_life_bets": round(np.log(2) / max(decay_rate, 1e-6), 0),
                "mean_pnl": round(float(np.mean(sim_arr)), 2),
                "loss_probability": round(float(np.mean(sim_arr < 0)), 4),
                "bets_until_negative_edge": bets_until_negative,
                "final_edge_pct": round(float(decay_factor[-1]) * 100, 1),
            })

        # Find critical decay rate (where loss prob > 50%)
        critical_decay = None
        for r in results_by_decay:
            if r["loss_probability"] > 0.50:
                critical_decay = r["decay_rate"]
                break

        if critical_decay and critical_decay <= 0.005:
            verdict = "fragile"
        elif critical_decay and critical_decay <= 0.02:
            verdict = "marginal"
        else:
            verdict = "robust"

        return {
            "assumption": "Market does not adapt to our signals",
            "distortion": "Exponential edge decay over time",
            "decay_scenarios": results_by_decay,
            "critical_decay_rate": critical_decay,
            "verdict": verdict,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  SCENARIO 5: SG Non-Additivity
    # ══════════════════════════════════════════════════════════════════════

    def _test_sg_non_additivity(self, bet_records: List[dict]) -> dict:
        """Test: What if SG components are NOT additive?

        The SG model assumes SG_total = SG_OTT + SG_APP + SG_ATG + SG_PUTT.
        In practice, there are interaction effects (e.g., great driving
        makes approach easier). Test the impact of non-additivity.
        """
        # Extract SG data from records that have it
        sg_records = []
        for r in bet_records:
            if all(k in r for k in ["sg_ott", "sg_app", "sg_atg", "sg_putt"]):
                sg_records.append(r)

        if len(sg_records) < 10:
            return {
                "assumption": "SG components are additive",
                "verdict": "untestable",
                "reason": "Insufficient SG data in bet records",
            }

        sg_ott = np.array([r["sg_ott"] for r in sg_records])
        sg_app = np.array([r["sg_app"] for r in sg_records])
        sg_atg = np.array([r["sg_atg"] for r in sg_records])
        sg_putt = np.array([r["sg_putt"] for r in sg_records])

        additive_total = sg_ott + sg_app + sg_atg + sg_putt

        # Simulate interaction effects
        # OTT * APP interaction: good driving makes approach easier
        interaction_ott_app = 0.1 * sg_ott * sg_app
        # Putting under pressure: non-linear when leading
        interaction_putt_pressure = 0.05 * np.where(additive_total > 1, sg_putt ** 2, 0)

        non_additive_total = additive_total + interaction_ott_app + interaction_putt_pressure

        # How much does non-additivity change rankings?
        additive_ranks = np.argsort(np.argsort(-additive_total))
        non_additive_ranks = np.argsort(np.argsort(-non_additive_total))

        rank_changes = np.abs(additive_ranks - non_additive_ranks)
        avg_rank_change = float(np.mean(rank_changes))
        max_rank_change = int(np.max(rank_changes))

        # Total score difference
        score_diff = non_additive_total - additive_total
        avg_diff = float(np.mean(np.abs(score_diff)))
        max_diff = float(np.max(np.abs(score_diff)))

        # Correlation between additive and non-additive
        corr = float(np.corrcoef(additive_total, non_additive_total)[0, 1])

        if corr < 0.90 or avg_rank_change > 3:
            verdict = "fragile"
        elif corr < 0.95 or avg_rank_change > 1.5:
            verdict = "marginal"
        else:
            verdict = "robust"

        return {
            "assumption": "SG components are additive",
            "distortion": "OTT*APP interaction + non-linear putting under pressure",
            "n_records": len(sg_records),
            "additive_vs_non_additive_corr": round(corr, 4),
            "avg_rank_change": round(avg_rank_change, 2),
            "max_rank_change": max_rank_change,
            "avg_score_difference": round(avg_diff, 4),
            "max_score_difference": round(max_diff, 4),
            "verdict": verdict,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  SCENARIO 6: Weather-Correlated Outcomes
    # ══════════════════════════════════════════════════════════════════════

    def _test_weather_correlated_outcomes(
        self, bet_records: List[dict], n_simulations: int
    ) -> dict:
        """Test: What if weather creates correlated outcomes across bets?

        In golf, weather affects ALL players in a wave simultaneously.
        This violates the independence assumption when betting on
        multiple players in the same tournament.
        """
        probs = np.array([r["model_prob"] for r in bet_records])
        odds = np.array([r["odds_decimal"] for r in bet_records])
        stakes = np.array([r["stake"] for r in bet_records])
        n_bets = len(probs)

        # Baseline: independent
        independent_results = []
        for _ in range(n_simulations):
            outcomes = self.rng.binomial(1, probs)
            pnl = float(np.sum(outcomes * stakes * (odds - 1) - (1 - outcomes) * stakes))
            independent_results.append(pnl)

        # Weather-correlated: shared weather shock per "tournament"
        # Group bets into pseudo-tournaments of 10
        tournament_size = min(10, n_bets)
        n_tournaments = max(1, n_bets // tournament_size)

        weather_results = []
        for _ in range(n_simulations):
            total_pnl = 0.0
            for t in range(n_tournaments):
                start = t * tournament_size
                end = min(start + tournament_size, n_bets)
                t_probs = probs[start:end]
                t_odds = odds[start:end]
                t_stakes = stakes[start:end]

                # Weather shock: shifts all probabilities in same direction
                weather_shock = self.rng.normal(0, 0.03)  # +-3% probability shift
                adjusted_probs = np.clip(t_probs + weather_shock, 0.01, 0.99)

                outcomes = self.rng.binomial(1, adjusted_probs)
                total_pnl += float(np.sum(
                    outcomes * t_stakes * (t_odds - 1) - (1 - outcomes) * t_stakes
                ))

            weather_results.append(total_pnl)

        ind = np.array(independent_results)
        weather = np.array(weather_results)

        var_ratio = float(np.var(weather)) / max(float(np.var(ind)), 1)
        tail_risk_increase = (
            float(np.percentile(weather, 5)) - float(np.percentile(ind, 5))
        )

        if var_ratio > 2.0:
            verdict = "fragile"
        elif var_ratio > 1.3:
            verdict = "marginal"
        else:
            verdict = "robust"

        return {
            "assumption": "Weather does not correlate outcomes across bets",
            "distortion": "Shared weather shock within tournament groups",
            "independent_pnl_std": round(float(np.std(ind)), 2),
            "weather_correlated_pnl_std": round(float(np.std(weather)), 2),
            "variance_ratio": round(var_ratio, 2),
            "tail_risk_change_p5": round(tail_risk_increase, 2),
            "verdict": verdict,
        }

    def _compute_overall_score(self, results: dict) -> float:
        """Compute overall assumption robustness score (0-100)."""
        scores = []
        verdict_to_score = {"robust": 90, "marginal": 50, "fragile": 15, "broken": 0}

        for scenario, data in results.items():
            verdict = data.get("verdict", "marginal")
            scores.append(verdict_to_score.get(verdict, 50))

        return float(np.mean(scores)) if scores else 50.0
