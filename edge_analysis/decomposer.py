"""
edge_analysis/decomposer.py
=============================
Master EdgeDecomposer for Golf — orchestrates all 5 edge components.

Usage:
    decomposer = EdgeDecomposer(sport="golf")
    report = decomposer.run()
    print(report.verdict)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from edge_analysis.schemas import GolfBetRecord, EdgeReport
from edge_analysis.predictive import compute_predictive_edge
from edge_analysis.informational import compute_informational_edge
from edge_analysis.market_inefficiency import compute_market_inefficiency_edge
from edge_analysis.execution import compute_execution_edge
from edge_analysis.structural import compute_structural_edge

log = logging.getLogger(__name__)


def _american_to_implied(odds: int) -> float:
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


class EdgeDecomposer:
    """Decomposes total golf system performance into 5 edge components.

    Components:
      1. Predictive — Brier score per market type (outright/top5/top10/top20/cut/matchup)
      2. Informational — Weather timing, injury/WD timing, wave advantage
      3. Market Inefficiency — CLV per market type (outrights are notoriously inefficient)
      4. Execution — Line shopping effectiveness in golf markets
      5. Structural — Field-level diversification, wave correlation
    """

    def __init__(self, sport: str = "golf", db_path: str | None = None):
        self.sport = sport.lower()
        self._db_path = db_path

    def load_bets_from_db(self) -> List[GolfBetRecord]:
        """Load all historical bets from the database."""
        from quant_system.db.schema import get_session, BetLog, CLVLog

        session = get_session(self._db_path)
        try:
            rows = (
                session.query(BetLog)
                .filter(
                    BetLog.sport == self.sport,
                    BetLog.status.in_(["won", "lost", "push"]),
                )
                .order_by(BetLog.timestamp.asc())
                .all()
            )

            records = []
            for row in rows:
                clv = session.query(CLVLog).filter_by(bet_id=row.bet_id).first()
                opening_line = clv.opening_line if clv else row.line
                closing_line = clv.closing_line if clv else (row.closing_line or row.line)

                won = None
                if row.status == "won":
                    won = True
                elif row.status == "lost":
                    won = False

                market_prob_at_close = None
                if row.closing_odds:
                    market_prob_at_close = _american_to_implied(row.closing_odds)

                # Extract golf-specific context from features_snapshot
                tournament = ""
                wave = None
                weather_conditions = None
                course_id = None
                try:
                    import json
                    features = json.loads(row.features_snapshot) if row.features_snapshot else {}
                    tournament = features.get("tournament", row.notes or "")
                    wave = features.get("wave")
                    weather_conditions = features.get("weather")
                    course_id = features.get("course_id")
                except Exception:
                    tournament = row.notes or ""

                records.append(GolfBetRecord(
                    bet_id=row.bet_id,
                    timestamp=row.timestamp,
                    signal_generated_at=row.timestamp,
                    tournament=tournament,
                    player=row.player,
                    course_id=course_id,
                    market_type=row.stat_type or "outright",
                    direction=row.direction or "over",
                    wave=wave,
                    weather_conditions=weather_conditions,
                    signal_line=opening_line,
                    bet_line=row.line,
                    closing_line=closing_line,
                    opening_line=opening_line,
                    predicted_prob=row.model_prob,
                    market_prob_at_bet=row.market_prob,
                    market_prob_at_close=market_prob_at_close,
                    actual_outcome=row.actual_result,
                    won=won,
                    stake=row.stake,
                    pnl=row.pnl or 0.0,
                    odds_american=row.odds_american,
                    odds_decimal=row.odds_decimal,
                    kelly_fraction=row.kelly_fraction,
                    model_projection=row.model_projection,
                    model_std=row.model_std,
                    confidence_score=row.confidence_score or 0.0,
                ))

            log.info("Loaded %d settled golf bets for edge decomposition", len(records))
            return records
        finally:
            session.close()

    def load_bets_from_golf_db(self) -> List[GolfBetRecord]:
        """Load from the Golf-specific database (database/models.py Bet table)."""
        try:
            from database.connection import get_session as get_golf_session
            session = get_golf_session()
        except Exception:
            try:
                from database.connection import init_db, SessionLocal
                init_db()
                session = SessionLocal()
            except Exception:
                return []

        try:
            from database.models import Bet
            rows = (
                session.query(Bet)
                .filter(Bet.status.in_(["won", "lost", "push"]))
                .order_by(Bet.timestamp.asc())
                .all()
            )

            records = []
            for row in rows:
                won = None
                if row.status == "won":
                    won = True
                elif row.status == "lost":
                    won = False

                market_prob_close = None
                if row.closing_line and row.closing_line != 0:
                    # Approximate: use odds if available
                    pass

                # Extract features
                tournament = row.event or ""
                wave = None
                weather_conditions = None
                course_id = None
                try:
                    import json
                    features = json.loads(row.features_snapshot_json) if row.features_snapshot_json else {}
                    wave = features.get("wave")
                    weather_conditions = features.get("weather")
                    course_id = features.get("course_id")
                except Exception:
                    pass

                records.append(GolfBetRecord(
                    bet_id=str(row.id),
                    timestamp=row.timestamp or datetime.now(timezone.utc),
                    signal_generated_at=row.timestamp,
                    tournament=tournament,
                    player=row.player or "",
                    course_id=course_id,
                    market_type=row.market or "outright",
                    direction=row.direction or "over",
                    wave=wave,
                    weather_conditions=weather_conditions,
                    signal_line=row.signal_line or 0.0,
                    bet_line=row.bet_line or 0.0,
                    closing_line=row.closing_line or 0.0,
                    predicted_prob=row.predicted_prob or 0.5,
                    market_prob_at_bet=0.5,
                    actual_outcome=row.actual_outcome,
                    won=won,
                    stake=row.stake or 0.0,
                    pnl=row.profit or 0.0,
                    odds_american=row.odds_american or -110,
                    odds_decimal=row.odds_decimal or 1.909,
                    kelly_fraction=0.0,
                    model_projection=row.model_projection or 0.0,
                    model_std=row.model_std or 1.0,
                    confidence_score=row.confidence_score or 0.0,
                ))

            return records
        except Exception as exc:
            log.warning("Could not load from golf DB: %s", exc)
            return []
        finally:
            session.close()

    def run(self, bets: Optional[List[GolfBetRecord]] = None) -> EdgeReport:
        """Execute full edge decomposition.

        Tries multiple data sources: quant_system DB, then golf-specific DB.
        """
        if bets is None:
            bets = self.load_bets_from_db()
            if not bets:
                bets = self.load_bets_from_golf_db()

        if len(bets) < 10:
            return EdgeReport(
                generated_at=datetime.now(timezone.utc),
                sport=self.sport,
                total_roi=0.0, total_bets=len(bets), total_pnl=0.0,
                predictive_pct=0.0, informational_pct=0.0,
                market_pct=0.0, execution_pct=0.0, structural_pct=0.0,
                verdict="Insufficient data for edge decomposition (need 10+ bets)",
            )

        # Compute total ROI
        total_pnl = sum(b.pnl for b in bets if b.pnl is not None)
        total_staked = sum(b.stake for b in bets if b.stake > 0)
        total_roi = (total_pnl / total_staked * 100.0) if total_staked > 0 else 0.0

        # Run each component
        predictive = compute_predictive_edge(bets, total_roi)
        informational = compute_informational_edge(bets, total_roi)
        market = compute_market_inefficiency_edge(bets, total_roi)
        execution = compute_execution_edge(bets, total_roi)
        structural = compute_structural_edge(bets, total_roi)

        # Normalize percentages
        raw_pcts = [
            predictive.edge_pct_of_roi,
            informational.edge_pct_of_roi,
            market.edge_pct_of_roi,
            execution.edge_pct_of_roi,
            structural.edge_pct_of_roi,
        ]
        raw_total = sum(abs(p) for p in raw_pcts)
        if raw_total > 0:
            scale = 100.0 / raw_total
            pred_pct = predictive.edge_pct_of_roi * scale
            info_pct = informational.edge_pct_of_roi * scale
            mkt_pct = market.edge_pct_of_roi * scale
            exec_pct = execution.edge_pct_of_roi * scale
            struct_pct = structural.edge_pct_of_roi * scale
        else:
            pred_pct = info_pct = mkt_pct = exec_pct = struct_pct = 20.0

        predictive.edge_pct_of_roi = round(pred_pct, 2)
        informational.edge_pct_of_roi = round(info_pct, 2)
        market.edge_pct_of_roi = round(mkt_pct, 2)
        execution.edge_pct_of_roi = round(exec_pct, 2)
        structural.edge_pct_of_roi = round(struct_pct, 2)

        # Calibration curve
        cal_curve = []
        if predictive.details.get("calibration_curve"):
            from edge_analysis.schemas import CalibrationPoint
            for pt in predictive.details["calibration_curve"]:
                bounds = pt["bucket"].replace("%", "").split("-")
                cal_curve.append(CalibrationPoint(
                    bucket_lower=float(bounds[0]) / 100.0,
                    bucket_upper=float(bounds[1]) / 100.0,
                    predicted_avg=pt["predicted"],
                    actual_rate=pt["actual"],
                    n_bets=pt["n"],
                    calibration_error=pt["error"],
                ))

        # Heavy lifter / illusions
        components = [
            ("Predictive", pred_pct, predictive.is_positive, predictive.is_significant),
            ("Informational", info_pct, informational.is_positive, informational.is_significant),
            ("Market Inefficiency", mkt_pct, market.is_positive, market.is_significant),
            ("Execution", exec_pct, execution.is_positive, execution.is_significant),
            ("Structural", struct_pct, structural.is_positive, structural.is_significant),
        ]

        positive_sig = [(n, p) for n, p, pos, sig in components if pos and sig]
        if positive_sig:
            heavy_lifter = max(positive_sig, key=lambda x: x[1])[0]
        else:
            positive_any = [(n, p) for n, p, pos, sig in components if pos]
            heavy_lifter = max(positive_any, key=lambda x: x[1])[0] if positive_any else "None detected"

        illusions = [n for n, p, pos, sig in components if not pos or (abs(p) > 10 and not sig)]

        verdict = _build_final_verdict(
            total_roi, total_pnl, len(bets),
            predictive, informational, market, execution, structural,
            heavy_lifter, illusions,
        )

        return EdgeReport(
            generated_at=datetime.now(timezone.utc),
            sport=self.sport,
            total_roi=round(total_roi, 4),
            total_bets=len(bets),
            total_pnl=round(total_pnl, 2),
            predictive_pct=round(pred_pct, 2),
            informational_pct=round(info_pct, 2),
            market_pct=round(mkt_pct, 2),
            execution_pct=round(exec_pct, 2),
            structural_pct=round(struct_pct, 2),
            predictive=predictive,
            informational=informational,
            market_inefficiency=market,
            execution=execution,
            structural=structural,
            calibration_curve=cal_curve,
            brier_score=predictive.details.get("brier_model", 0.0),
            log_loss=predictive.details.get("logloss_model", 0.0),
            brier_baseline=predictive.details.get("brier_market", 0.25),
            log_loss_baseline=predictive.details.get("logloss_market", 0.693),
            verdict=verdict,
            heavy_lifter=heavy_lifter,
            illusions=illusions,
        )


def _build_final_verdict(
    total_roi, total_pnl, n_bets,
    predictive, informational, market, execution, structural,
    heavy_lifter, illusions,
) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("GOLF EDGE DECOMPOSITION VERDICT")
    lines.append("=" * 70)
    lines.append(f"Total ROI: {total_roi:+.2f}% across {n_bets} bets (P&L: ${total_pnl:+,.2f})")
    lines.append("")

    lines.append("COMPONENT ATTRIBUTION:")
    for name, comp in [
        ("Predictive", predictive),
        ("Informational", informational),
        ("Market Inefficiency", market),
        ("Execution", execution),
        ("Structural", structural),
    ]:
        status = "REAL" if comp.is_positive and comp.is_significant else (
            "POSSIBLE" if comp.is_positive else "ILLUSION"
        )
        lines.append(f"  {name:25s} {comp.edge_pct_of_roi:6.1f}%  [{status}]  p={comp.p_value:.4f}")

    lines.append("")
    lines.append(f"HEAVY LIFTER: {heavy_lifter}")

    if illusions:
        lines.append(f"ILLUSIONS: {', '.join(illusions)}")
    else:
        lines.append("ILLUSIONS: None — all components contribute.")

    lines.append("")
    lines.append("WHICH COMPONENT IS DOING THE HEAVY LIFTING — AND WHICH ARE ILLUSIONS?")
    lines.append("-" * 70)

    if market.is_positive and market.is_significant:
        lines.append(
            "CLV is positive and significant. Genuine market inefficiency exploitation confirmed. "
            "Golf outright markets are inherently inefficient — this system capitalizes on it."
        )
    elif total_roi > 0 and not market.is_positive:
        lines.append(
            "WARNING: Positive ROI but NEGATIVE CLV. In golf's high-variance markets, "
            "this is almost certainly luck. Outright winners can mask negative CLV for "
            "hundreds of bets. Without positive CLV, this edge is an illusion."
        )

    if predictive.is_positive and predictive.is_significant:
        lines.append(
            "The model produces better probabilities than the market across golf market types. "
            "This is the foundation of quantitative golf betting."
        )

    # Golf-specific market type insights
    by_market = predictive.details.get("by_market_type", {})
    if "outright" in by_market and not by_market["outright"].get("insufficient"):
        skill = by_market["outright"].get("skill_vs_market", 0)
        if skill > 0:
            lines.append(
                f"Outright prediction skill is POSITIVE (skill={skill:.1%}). "
                f"This is extremely difficult to achieve in golf. Strong signal."
            )

    lines.append("=" * 70)
    return "\n".join(lines)
