"""Adversarial test runner — Orchestrate all adversarial tests and produce unified report."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from database.connection import DatabaseManager
from database.models import Bet, CLVLog, AuditLog

logger = logging.getLogger(__name__)


class AdversarialTestRunner:
    """Run all adversarial tests and produce a unified robustness report.

    Usage:
        runner = AdversarialTestRunner(sport="golf")
        report = runner.run_full_suite()
        print(runner.format_report(report))
    """

    def __init__(self, sport: str = "golf", seed: int = 42):
        self.sport = sport
        self.seed = seed

    def run_full_suite(self, n_simulations: int = 1000) -> dict:
        """Run the complete adversarial test suite.

        Returns:
            {
                'overall_robustness': float (0-100),
                'overall_verdict': str,
                'test_results': dict,
                'critical_findings': list,
                'recommendations': list,
                'timestamp': str,
            }
        """
        # Load bet data
        bet_records = self._load_bet_records()

        if not bet_records or len(bet_records) < 20:
            return {
                "overall_robustness": 0.0,
                "overall_verdict": "untestable",
                "error": f"Insufficient bet records ({len(bet_records) if bet_records else 0}). Need at least 20.",
                "timestamp": datetime.utcnow().isoformat(),
            }

        test_results = {}
        critical_findings = []

        # 1. Probability Perturbation
        from tests.adversarial.probability_perturbation import ProbabilityPerturbationTest
        try:
            pp = ProbabilityPerturbationTest(seed=self.seed)
            test_results["probability_perturbation"] = pp.run_all(bet_records)
            if test_results["probability_perturbation"].get("verdict") == "fragile":
                critical_findings.append(
                    "System is FRAGILE to probability estimation errors — "
                    "small errors in model probabilities cause large PnL swings"
                )
        except Exception as e:
            logger.error("Probability perturbation test failed: %s", e)
            test_results["probability_perturbation"] = {"error": str(e)}

        # 2. Best Bet Removal
        from tests.adversarial.best_bet_removal import BestBetRemovalTest
        try:
            bbr = BestBetRemovalTest()
            test_results["best_bet_removal"] = bbr.run(bet_records)
            verdict = test_results["best_bet_removal"].get("verdict", "")
            if verdict in ("fragile", "illusory"):
                critical_findings.append(
                    f"Profitability is {verdict.upper()} — "
                    f"concentrated in a few lucky bets. "
                    f"Unprofitable after removing top "
                    f"{test_results['best_bet_removal'].get('first_unprofitable_removal_pct', '?')}%"
                )
        except Exception as e:
            logger.error("Best bet removal test failed: %s", e)
            test_results["best_bet_removal"] = {"error": str(e)}

        # 3. Noise Injection
        from tests.adversarial.noise_injection import NoiseInjectionTest
        try:
            ni = NoiseInjectionTest(seed=self.seed)
            player_data = self._prepare_player_data(bet_records)
            test_results["noise_injection"] = ni.run_all(player_data)
            if test_results["noise_injection"].get("verdict") == "fragile":
                most_sensitive = test_results["noise_injection"].get("most_sensitive_source", "unknown")
                critical_findings.append(
                    f"System is FRAGILE to noisy inputs — "
                    f"most sensitive to {most_sensitive} data"
                )
        except Exception as e:
            logger.error("Noise injection test failed: %s", e)
            test_results["noise_injection"] = {"error": str(e)}

        # 4. Assumption Distortion
        from tests.adversarial.assumption_distortion import AssumptionDistortionTest
        try:
            ad = AssumptionDistortionTest(seed=self.seed)
            test_results["assumption_distortion"] = ad.run_all(
                bet_records, n_simulations=n_simulations
            )
            if test_results["assumption_distortion"].get("verdict") == "fragile":
                # Find which assumptions are broken
                broken = []
                for scenario, data in test_results["assumption_distortion"].get("results", {}).items():
                    if data.get("verdict") in ("fragile", "broken"):
                        broken.append(scenario)
                if broken:
                    critical_findings.append(
                        f"System FRAGILE under assumption violations: {', '.join(broken)}"
                    )
        except Exception as e:
            logger.error("Assumption distortion test failed: %s", e)
            test_results["assumption_distortion"] = {"error": str(e)}

        # Compute overall robustness
        scores = []
        for test_name, result in test_results.items():
            score = result.get("robustness_score") or result.get("overall_score")
            if score is not None:
                scores.append(score)

        overall_robustness = float(np.mean(scores)) if scores else 0.0

        if overall_robustness >= 70:
            overall_verdict = "robust"
        elif overall_robustness >= 50:
            overall_verdict = "acceptable"
        elif overall_robustness >= 30:
            overall_verdict = "marginal"
        else:
            overall_verdict = "fragile"

        # Generate recommendations
        recommendations = self._generate_recommendations(
            test_results, critical_findings
        )

        # Log results
        self._log_results(overall_robustness, overall_verdict, critical_findings)

        return {
            "overall_robustness": round(overall_robustness, 2),
            "overall_verdict": overall_verdict,
            "n_tests_run": len(test_results),
            "n_critical_findings": len(critical_findings),
            "test_results": {
                name: {
                    "robustness_score": r.get("robustness_score") or r.get("overall_score"),
                    "verdict": r.get("verdict", "unknown"),
                    "error": r.get("error"),
                }
                for name, r in test_results.items()
            },
            "full_results": test_results,
            "critical_findings": critical_findings,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def format_report(self, report: dict) -> str:
        """Format adversarial test results as human-readable report."""
        lines = [
            "=" * 70,
            "  ADVERSARIAL TEST REPORT",
            f"  Overall Robustness: {report['overall_robustness']:.0f}/100",
            f"  Verdict: {report['overall_verdict'].upper()}",
            "=" * 70,
            "",
        ]

        lines.append("TEST RESULTS:")
        for name, result in report.get("test_results", {}).items():
            score = result.get("robustness_score")
            verdict = result.get("verdict", "?")
            error = result.get("error")

            if error:
                lines.append(f"  {name:<35} ERROR: {error}")
            elif score is not None:
                bar = "#" * int(score / 5) + "." * (20 - int(score / 5))
                lines.append(f"  {name:<35} [{bar}] {score:.0f}/100 ({verdict})")
            else:
                lines.append(f"  {name:<35} {verdict}")

        if report.get("critical_findings"):
            lines.append("")
            lines.append("CRITICAL FINDINGS:")
            for finding in report["critical_findings"]:
                lines.append(f"  [!!] {finding}")

        if report.get("recommendations"):
            lines.append("")
            lines.append("RECOMMENDATIONS:")
            for rec in report["recommendations"]:
                lines.append(f"  ->  {rec}")

        lines.append("")
        lines.append(f"Generated: {report.get('timestamp', 'unknown')}")
        lines.append("=" * 70)

        return "\n".join(lines)

    # ── Internal ────────────────────────────────────────────────────────

    def _load_bet_records(self, limit: int = 500) -> List[dict]:
        """Load settled bet records from database."""
        with DatabaseManager.session_scope() as session:
            bets = (
                session.query(Bet)
                .filter(
                    Bet.sport == self.sport,
                    Bet.status.in_(["won", "lost"]),
                )
                .order_by(Bet.settled_at.desc())
                .limit(limit)
                .all()
            )

            records = []
            for b in bets:
                record = {
                    "bet_id": b.bet_id,
                    "player": b.player,
                    "bet_type": b.bet_type,
                    "market_type": b.bet_type,
                    "model_prob": b.model_prob,
                    "market_prob": b.market_prob,
                    "odds_decimal": b.odds_decimal,
                    "stake": b.stake,
                    "pnl": b.pnl,
                    "status": b.status,
                    "edge": b.edge,
                }

                # Parse feature snapshot for SG data
                if b.features_snapshot:
                    try:
                        features = json.loads(b.features_snapshot)
                        for key in ["sg_ott", "sg_app", "sg_atg", "sg_putt",
                                     "sg_total", "wind_speed", "temperature",
                                     "course_yardage", "field_strength"]:
                            if key in features:
                                record[key] = features[key]
                    except (json.JSONDecodeError, TypeError):
                        pass

                records.append(record)

            return records

    def _prepare_player_data(self, bet_records: List[dict]) -> List[dict]:
        """Prepare player-level data for noise injection test."""
        # Use bet records directly — they contain player-level features
        player_data = []
        for r in bet_records:
            data = {
                "player": r.get("player", "unknown"),
                "model_prob": r.get("model_prob", 0.5),
                "odds_decimal": r.get("odds_decimal", 2.0),
            }
            # Add SG and other features if available
            for key in ["sg_total", "sg_ott", "sg_app", "sg_atg", "sg_putt",
                         "wind_speed", "temperature", "course_yardage",
                         "field_strength"]:
                if key in r:
                    data[key] = r[key]
            player_data.append(data)
        return player_data

    def _generate_recommendations(
        self, test_results: dict, critical_findings: List[str]
    ) -> List[str]:
        """Generate actionable recommendations from test results."""
        recs = []

        # Probability perturbation
        pp = test_results.get("probability_perturbation", {})
        if pp.get("verdict") in ("fragile", "marginal"):
            recs.append(
                "Increase probability uncertainty buffer in Kelly criterion sizing "
                "to account for model estimation error"
            )

        # Best bet removal
        bbr = test_results.get("best_bet_removal", {})
        if bbr.get("verdict") in ("fragile", "illusory"):
            concentration = bbr.get("concentration_index", 0)
            recs.append(
                f"PnL concentration index is {concentration:.0%} in top 5% of bets. "
                f"Diversify bet types and reduce outright bet sizing."
            )
            recs.append(
                "Consider increasing allocation to higher-frequency markets "
                "(top20, make_cut) vs low-frequency outrights"
            )

        # Noise injection
        ni = test_results.get("noise_injection", {})
        if ni.get("verdict") in ("fragile", "marginal"):
            source = ni.get("most_sensitive_source", "unknown")
            recs.append(
                f"Most noise-sensitive input: {source}. "
                f"Add data validation and smoothing for this source."
            )

        # Assumption distortion
        ad = test_results.get("assumption_distortion", {})
        if ad.get("verdict") in ("fragile", "marginal"):
            ad_results = ad.get("results", {})
            if ad_results.get("correlated_outcomes", {}).get("verdict") == "fragile":
                recs.append(
                    "Bet outcomes are sensitive to correlation. "
                    "Limit per-tournament exposure and diversify across tournaments."
                )
            if ad_results.get("regime_change", {}).get("verdict") == "fragile":
                recs.append(
                    "System slow to detect regime changes. "
                    "Reduce CUSUM detection thresholds for faster response."
                )
            if ad_results.get("fat_tails", {}).get("verdict") == "fragile":
                recs.append(
                    "PnL has fat-tail sensitivity. "
                    "Use fractional Kelly (<=25%) and increase max drawdown buffer."
                )

        if not recs:
            recs.append("System passed all adversarial tests. Continue monitoring.")

        return recs

    def _log_results(
        self,
        robustness: float,
        verdict: str,
        critical_findings: List[str],
    ) -> None:
        """Log adversarial test results to audit log."""
        try:
            with DatabaseManager.session_scope() as session:
                session.add(AuditLog(
                    sport=self.sport,
                    category="adversarial_test",
                    level="warning" if verdict != "robust" else "info",
                    message=(
                        f"Adversarial test: {verdict} "
                        f"(robustness={robustness:.0f}/100, "
                        f"{len(critical_findings)} critical findings)"
                    ),
                    details=json.dumps({
                        "robustness": round(robustness, 2),
                        "verdict": verdict,
                        "critical_findings": critical_findings,
                    }),
                ))
        except Exception as e:
            logger.error("Failed to log adversarial test results: %s", e)
