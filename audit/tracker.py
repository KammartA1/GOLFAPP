"""
Audit & Performance Tracking System
The most important module for long-term profitability.

Tracks:
  - Every projection vs actual finish
  - Edge decay over time (is our model staying sharp?)
  - ROI by bet type, player tier, course type
  - Model calibration (are our probabilities accurate?)
  - Kelly sizing accuracy
  - DFS lineup performance

Inspired by the NBA quant engine's audit philosophy: 
every edge source must be proven in data, not assumed.
"""
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from config.settings import AUDIT_DIR, ROI_ALERT_THRESHOLD, MIN_AUDIT_SAMPLE

log = logging.getLogger(__name__)


@dataclass
class ModelCalibration:
    """Calibration results for a probability estimate bucket."""
    bucket_label: str       # e.g., "10-20% win prob"
    predicted_prob: float
    actual_rate: float
    n_samples: int
    calibration_error: float  # |predicted - actual|
    is_overconfident: bool


class AuditEngine:
    """
    Full performance audit engine for the Golf Quant Engine.

    Stores and analyzes:
      - Bet records (placed, won/lost, P&L)
      - Projection accuracy (model rank vs actual finish)
      - DFS lineup results
      - Model calibration
      - Edge decay signals
    """

    def __init__(self, db_session=None):
        self.session = db_session
        self.audit_path = AUDIT_DIR

    # ─────────────────────────────────────────────
    # BET TRACKING
    # ─────────────────────────────────────────────

    def log_bet(self, bet_data: dict) -> bool:
        """Log a placed bet to the audit system."""
        from data.storage.database import Bet, get_session
        if self.session is None:
            self.session = get_session()
        try:
            bet = Bet(**{k: v for k, v in bet_data.items() if hasattr(Bet, k)})
            self.session.add(bet)
            self.session.commit()
            log.info(f"Bet logged: {bet_data.get('player')} {bet_data.get('bet_type')}")
            return True
        except Exception as e:
            log.error(f"Failed to log bet: {e}")
            self.session.rollback()
            return False

    def settle_bet(self, bet_id: int, won: bool, actual_finish: int = None) -> dict:
        """Settle a bet and update P&L."""
        from data.storage.database import Bet
        from betting.kelly import american_to_decimal

        bet = self.session.get(Bet, bet_id)
        if not bet:
            return {}

        decimal_odds = american_to_decimal(bet.odds_american)
        if won:
            pl = bet.stake * (decimal_odds - 1)
        else:
            pl = -bet.stake

        bet.won = won
        bet.profit_loss = round(pl, 2)
        bet.settled = True
        bet.settlement_date = datetime.utcnow()
        self.session.commit()

        log.info(f"Settled bet {bet_id}: {'WON' if won else 'LOST'} ${pl:+.2f}")
        return {"bet_id": bet_id, "won": won, "profit_loss": pl}

    # ─────────────────────────────────────────────
    # PROJECTION ACCURACY
    # ─────────────────────────────────────────────

    def log_projection_accuracy(
        self,
        tournament_name: str,
        projections_df: pd.DataFrame,
        actual_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compare projected SG / model rank to actual finish.
        Computes accuracy metrics and logs them.

        projections_df: output of ProjectionEngine.run()
        actual_results: DataFrame with columns [name, finish_position, sg_total_actual]
        """
        merged = pd.merge(
            projections_df[["name", "model_rank", "proj_sg_total", "win_prob", "top10_prob", "course_fit_score"]],
            actual_results[["name", "finish_position", "sg_total_actual"]].rename(
                columns={"finish_position": "actual_finish"}
            ),
            on="name", how="inner"
        )

        if merged.empty:
            log.warning("No players matched between projections and results")
            return pd.DataFrame()

        # Rank correlation
        merged["actual_rank"] = merged["actual_finish"].rank()
        rank_corr = merged["model_rank"].corr(merged["actual_rank"], method="spearman")

        # SG accuracy (if we have actual SG)
        if "sg_total_actual" in merged.columns and merged["sg_total_actual"].notna().any():
            merged["sg_error"] = merged["proj_sg_total"] - merged["sg_total_actual"]
            mae = merged["sg_error"].abs().mean()
            rmse = np.sqrt((merged["sg_error"]**2).mean())
        else:
            mae, rmse = None, None

        # Top-10 hit rate (did players we projected top-10 make top-10?)
        if "actual_finish" in merged.columns:
            our_top10 = merged[merged["model_rank"] <= 10]["name"].tolist()
            actual_top10 = merged[merged["actual_finish"] <= 10]["name"].tolist()
            top10_hits = len(set(our_top10) & set(actual_top10))
            top10_accuracy = top10_hits / max(len(our_top10), 1)
        else:
            top10_accuracy = None

        accuracy_report = {
            "tournament": tournament_name,
            "date": datetime.utcnow().isoformat(),
            "n_players": len(merged),
            "rank_correlation_spearman": round(rank_corr, 3),
            "sg_mae": round(mae, 3) if mae else None,
            "sg_rmse": round(rmse, 3) if rmse else None,
            "top10_accuracy": round(top10_accuracy, 3) if top10_accuracy else None,
            "top10_hits": top10_hits if top10_accuracy else None,
        }

        # Save report
        report_file = self.audit_path / f"accuracy_{tournament_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, "w") as f:
            json.dump(accuracy_report, f, indent=2)

        log.info(f"Accuracy report saved: Rank corr={rank_corr:.3f}, Top10 hit={top10_accuracy}")
        return merged

    # ─────────────────────────────────────────────
    # CALIBRATION ANALYSIS
    # ─────────────────────────────────────────────

    def calibration_analysis(self, bet_records: list[dict]) -> list[ModelCalibration]:
        """
        Check if our probability estimates are well-calibrated.
        e.g., when we say 15% win probability, do players win ~15% of the time?

        bet_records: list of {model_prob, actual_outcome (bool), bet_type}
        """
        df = pd.DataFrame(bet_records)
        if df.empty or "model_prob" not in df.columns:
            return []

        df["outcome"] = df["actual_outcome"].astype(int)

        # Create probability buckets
        bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]
        labels = ["0-5%","5-10%","10-15%","15-20%","20-30%","30-40%","40-50%","50-70%","70%+"]
        df["bucket"] = pd.cut(df["model_prob"], bins=bins, labels=labels)

        calibrations = []
        for label in labels:
            bucket = df[df["bucket"] == label]
            if len(bucket) < 5:
                continue
            pred = bucket["model_prob"].mean()
            actual = bucket["outcome"].mean()
            error = abs(pred - actual)
            calibrations.append(ModelCalibration(
                bucket_label=label,
                predicted_prob=round(pred, 3),
                actual_rate=round(actual, 3),
                n_samples=len(bucket),
                calibration_error=round(error, 3),
                is_overconfident=pred > actual,
            ))

        return calibrations

    # ─────────────────────────────────────────────
    # ROI ANALYSIS
    # ─────────────────────────────────────────────

    def compute_roi(
        self,
        bet_records: list[dict],
        group_by: str = None,    # "bet_type", "course_type", "player_tier"
    ) -> pd.DataFrame:
        """
        Compute ROI across all settled bets, optionally grouped.

        Returns DataFrame with ROI, win rate, avg edge, etc.
        """
        df = pd.DataFrame(bet_records)
        settled = df[df.get("settled", pd.Series([True]*len(df))).astype(bool)]

        if settled.empty:
            return pd.DataFrame()

        def roi_stats(group: pd.DataFrame) -> pd.Series:
            total_stake = group["stake"].sum()
            total_pl = group["profit_loss"].sum()
            wins = (group["won"] == True).sum()
            n = len(group)
            return pd.Series({
                "n_bets":       n,
                "wins":         wins,
                "win_rate":     round(wins / n, 3),
                "total_staked": round(total_stake, 2),
                "total_pl":     round(total_pl, 2),
                "roi":          round(total_pl / total_stake, 4) if total_stake else 0,
                "avg_edge":     round(group.get("edge_pct", pd.Series([0]*n)).mean(), 3),
                "avg_odds":     round(group.get("odds_american", pd.Series([0]*n)).mean(), 1),
            })

        if group_by and group_by in settled.columns:
            result = settled.groupby(group_by).apply(roi_stats).reset_index()
        else:
            result = roi_stats(settled).to_frame("overall").T

        return result

    def rolling_roi(self, bet_records: list[dict], window: int = 20) -> pd.DataFrame:
        """Compute rolling ROI to detect edge decay."""
        df = pd.DataFrame(bet_records)
        df = df[df.get("settled", pd.Series([True]*len(df))).astype(bool)]
        df = df.sort_values("placed_at") if "placed_at" in df.columns else df

        if len(df) < window:
            return pd.DataFrame()

        df["rolling_roi"] = (
            df["profit_loss"].rolling(window).sum() /
            df["stake"].rolling(window).sum()
        )

        if df["rolling_roi"].iloc[-1] < ROI_ALERT_THRESHOLD:
            log.warning(f"⚠️  EDGE ALERT: Rolling ROI ({df['rolling_roi'].iloc[-1]:.1%}) "
                        f"below threshold ({ROI_ALERT_THRESHOLD:.1%})")

        return df[["placed_at", "profit_loss", "stake", "rolling_roi"]]

    # ─────────────────────────────────────────────
    # FULL AUDIT REPORT
    # ─────────────────────────────────────────────

    def generate_report(self, bet_records: list[dict], title: str = "Golf Quant Audit") -> str:
        """Generate full text audit report."""
        from rich.console import Console
        from rich.table import Table
        from io import StringIO

        output = StringIO()
        console = Console(file=output, width=100)

        console.rule(f"[bold gold1]{title}[/bold gold1]")
        console.print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

        # Overall ROI
        overall = self.compute_roi(bet_records)
        if not overall.empty:
            console.print("[bold]📊 Overall Performance[/bold]")
            console.print(overall.to_string())
            console.print()

        # ROI by bet type
        by_type = self.compute_roi(bet_records, group_by="bet_type")
        if not by_type.empty:
            console.print("[bold]📈 ROI by Bet Type[/bold]")
            console.print(by_type.to_string())
            console.print()

        # Calibration
        cal = self.calibration_analysis(bet_records)
        if cal:
            console.print("[bold]🎯 Probability Calibration[/bold]")
            cal_data = [{
                "Bucket": c.bucket_label,
                "Predicted": f"{c.predicted_prob:.1%}",
                "Actual": f"{c.actual_rate:.1%}",
                "Error": f"{c.calibration_error:.1%}",
                "N": c.n_samples,
                "Status": "OVERCONF" if c.is_overconfident else "UNDERCONF",
            } for c in cal]
            console.print(pd.DataFrame(cal_data).to_string())

        report_text = output.getvalue()

        # Save to file
        fname = self.audit_path / f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(fname, "w") as f:
            f.write(report_text)
        log.info(f"Audit report saved: {fname}")

        return report_text

    # ─────────────────────────────────────────────
    # EDGE DECAY DETECTION
    # ─────────────────────────────────────────────

    def detect_edge_decay(self, bet_records: list[dict]) -> dict:
        """
        Compare first half vs second half ROI.
        If ROI is declining over time, our edge may be getting priced out.
        """
        df = pd.DataFrame(bet_records)
        df = df[df.get("settled", pd.Series([True]*len(df))).astype(bool)]

        if len(df) < MIN_AUDIT_SAMPLE:
            return {"status": "insufficient_data", "n_bets": len(df)}

        mid = len(df) // 2
        first_half = df.iloc[:mid]
        second_half = df.iloc[mid:]

        def roi(g): 
            return g["profit_loss"].sum() / g["stake"].sum() if g["stake"].sum() else 0

        first_roi = roi(first_half)
        second_roi = roi(second_half)
        decay = first_roi - second_roi

        status = "healthy"
        if decay > 0.10:
            status = "decaying"
            log.warning(f"⚠️  Edge decay detected: {first_roi:.1%} → {second_roi:.1%}")
        elif decay < -0.05:
            status = "improving"

        return {
            "status": status,
            "first_half_roi": round(first_roi, 4),
            "second_half_roi": round(second_roi, 4),
            "decay_rate": round(decay, 4),
            "recommendation": (
                "Review model inputs — edge may be getting priced in" if status == "decaying"
                else "Model performing well" if status == "healthy"
                else "Edge improving — scale up sizing"
            )
        }
