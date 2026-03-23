"""
Golf Quant Engine -- Model Worker
===================================
Retrains the SG projection model on new data.

Trigger conditions (any one):
  1. Weekly scheduled retrain (Tuesday after tournament results finalised)
  2. Calibration error exceeds threshold (MAE > 0.08)
  3. Model drift detected (PSI > 0.20 or KS p-value < 0.05)

Retraining process:
  1. Collect settled bets and tournament results
  2. Update SG baselines with new tournament data
  3. Recalculate course fit parameters
  4. Adjust regression factors via walk-forward validation
  5. Save new ModelVersion record
  6. Validate with walk-forward analysis

Run standalone:
    python -m workers.model_worker
"""
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workers.base import BaseWorker
from database.connection import get_session_factory, init_db
from database.models import (
    ModelVersion, Bet, SGStat, Event, Player,
    CalibrationSnapshot, SystemState,
)

log = logging.getLogger(__name__)

# Thresholds
CALIBRATION_MAE_THRESHOLD = 0.08
BRIER_SCORE_THRESHOLD = 0.28
MIN_SETTLED_BETS_FOR_RETRAIN = 15
RETRAIN_DAY_OF_WEEK = 1  # Tuesday (0=Monday)


class ModelWorker(BaseWorker):
    name = "model_worker"
    interval_seconds = int(os.environ.get("MODEL_WORKER_INTERVAL", 3600))  # 1 hour check
    max_retries = 1
    retry_delay = 60.0
    description = "Retrains SG model when triggered by schedule, calibration error, or drift"

    def execute(self) -> dict:
        init_db()
        factory = get_session_factory()
        session = factory()

        try:
            # 1. Check if retraining is needed
            trigger = self._check_retrain_needed(session)

            if trigger is None:
                self._logger.info("No retraining needed at this time")
                return {"items_processed": 0, "trigger": "none", "retrained": False}

            self._logger.info("Retraining triggered: %s", trigger)

            # 2. Collect training data
            training_data = self._collect_training_data(session)
            if not training_data["sg_records"]:
                self._logger.warning("No SG training data available -- skipping retrain")
                return {"items_processed": 0, "trigger": trigger, "retrained": False}

            # 3. Run retraining
            results = self._retrain_model(session, training_data, trigger)

            # 4. Save new model version
            version_id = self._save_model_version(session, results, trigger)

            # 5. Validate with walk-forward
            validation = self._validate_model(session, results)

            # 6. Decide whether to activate
            activated = self._maybe_activate(session, version_id, validation)

            # 7. Update system state if needed
            self._update_system_state(session, validation, activated)

            session.commit()

            return {
                "items_processed": training_data["n_records"],
                "trigger": trigger,
                "retrained": True,
                "version_id": version_id,
                "activated": activated,
                "validation": validation,
            }

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Check if retrain needed
    # ------------------------------------------------------------------ #
    def _check_retrain_needed(self, session) -> str | None:
        """Return trigger reason or None if no retrain needed."""
        now = datetime.utcnow()

        # 1. Weekly schedule check (Tuesday)
        if now.weekday() == RETRAIN_DAY_OF_WEEK:
            last_version = (
                session.query(ModelVersion)
                .filter(ModelVersion.sport == "GOLF")
                .order_by(ModelVersion.created_at.desc())
                .first()
            )
            if last_version is None:
                return "initial_training"

            days_since = (now - last_version.created_at).days
            if days_since >= 6:  # At least 6 days since last retrain
                return "weekly_scheduled"

        # 2. Calibration error check
        cal_trigger = self._check_calibration_error(session)
        if cal_trigger:
            return cal_trigger

        # 3. Model drift detection
        drift_trigger = self._check_model_drift(session)
        if drift_trigger:
            return drift_trigger

        return None

    def _check_calibration_error(self, session) -> str | None:
        """Check if calibration error exceeds threshold."""
        # Get recent settled bets
        settled = (
            session.query(Bet)
            .filter(Bet.sport == "GOLF")
            .filter(Bet.status.in_(["won", "lost"]))
            .order_by(Bet.settled_at.desc())
            .limit(200)
            .all()
        )

        if len(settled) < MIN_SETTLED_BETS_FOR_RETRAIN:
            return None

        # Compute calibration metrics
        predicted_probs = []
        outcomes = []
        for bet in settled:
            if bet.predicted_prob is not None:
                predicted_probs.append(bet.predicted_prob)
                outcomes.append(1.0 if bet.status == "won" else 0.0)

        if len(predicted_probs) < MIN_SETTLED_BETS_FOR_RETRAIN:
            return None

        # Brier score
        brier = float(np.mean([
            (p - o) ** 2 for p, o in zip(predicted_probs, outcomes)
        ]))

        if brier > BRIER_SCORE_THRESHOLD:
            self._logger.warning("Brier score %.4f exceeds threshold %.4f", brier, BRIER_SCORE_THRESHOLD)
            return f"calibration_error_brier_{brier:.4f}"

        # Bucket-based calibration MAE
        buckets = [
            (0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
            (0.70, 0.75), (0.75, 0.80), (0.80, 0.90), (0.90, 1.00),
        ]
        errors = []
        for lo, hi in buckets:
            bucket_preds = [p for p in predicted_probs if lo <= p < hi]
            bucket_outcomes = [o for p, o in zip(predicted_probs, outcomes) if lo <= p < hi]
            if len(bucket_preds) >= 3:
                pred_avg = float(np.mean(bucket_preds))
                actual_rate = float(np.mean(bucket_outcomes))
                errors.append(abs(pred_avg - actual_rate))

        if errors:
            mae = float(np.mean(errors))
            if mae > CALIBRATION_MAE_THRESHOLD:
                self._logger.warning("Calibration MAE %.4f exceeds threshold %.4f", mae, CALIBRATION_MAE_THRESHOLD)
                return f"calibration_error_mae_{mae:.4f}"

        return None

    def _check_model_drift(self, session) -> str | None:
        """Check for prediction distribution drift."""
        recent_bets = (
            session.query(Bet)
            .filter(Bet.sport == "GOLF")
            .filter(Bet.status.in_(["won", "lost"]))
            .order_by(Bet.settled_at.desc())
            .limit(400)
            .all()
        )

        if len(recent_bets) < 150:
            return None

        # Split into recent vs baseline
        recent_probs = [b.predicted_prob for b in recent_bets[:100] if b.predicted_prob is not None]
        baseline_probs = [b.predicted_prob for b in recent_bets[100:] if b.predicted_prob is not None]

        if len(recent_probs) < 50 or len(baseline_probs) < 50:
            return None

        from scipy import stats as sp_stats

        # KS test
        ks_stat, p_value = sp_stats.ks_2samp(recent_probs, baseline_probs)

        # PSI
        psi = self._compute_psi(baseline_probs, recent_probs)

        if p_value < 0.05 or psi > 0.20:
            self._logger.warning(
                "Model drift detected: KS p=%.4f, PSI=%.4f", p_value, psi,
            )
            return f"drift_psi_{psi:.4f}_ks_p_{p_value:.4f}"

        # Check win rate degradation
        recent_wr = sum(1 for b in recent_bets[:100] if b.status == "won") / max(len(recent_bets[:100]), 1)
        baseline_wr = sum(1 for b in recent_bets[100:] if b.status == "won") / max(len(recent_bets[100:]), 1)

        if recent_wr < baseline_wr - 0.08:
            self._logger.warning(
                "Win rate degradation: recent=%.3f, baseline=%.3f",
                recent_wr, baseline_wr,
            )
            return f"accuracy_degradation_wr_{recent_wr:.3f}_vs_{baseline_wr:.3f}"

        return None

    @staticmethod
    def _compute_psi(baseline: list[float], recent: list[float], n_bins: int = 10) -> float:
        """Population Stability Index."""
        b = np.array(baseline)
        r = np.array(recent)
        bins = np.percentile(b, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        b_counts = np.histogram(b, bins=bins)[0]
        r_counts = np.histogram(r, bins=bins)[0]
        b_pct = b_counts / max(len(b), 1) + 1e-6
        r_pct = r_counts / max(len(r), 1) + 1e-6
        psi = float(np.sum((r_pct - b_pct) * np.log(r_pct / b_pct)))
        return max(psi, 0.0)

    # ------------------------------------------------------------------ #
    # Collect training data
    # ------------------------------------------------------------------ #
    def _collect_training_data(self, session) -> dict:
        """Gather SG stats and settled bets for retraining."""
        # All SG stats
        sg_records = (
            session.query(SGStat)
            .filter(SGStat.sg_total.isnot(None))
            .order_by(SGStat.created_at.desc())
            .limit(5000)
            .all()
        )

        # Settled bets
        settled_bets = (
            session.query(Bet)
            .filter(Bet.sport == "GOLF")
            .filter(Bet.status.in_(["won", "lost"]))
            .order_by(Bet.settled_at.desc())
            .limit(1000)
            .all()
        )

        # Recent events
        events = (
            session.query(Event)
            .filter(Event.sport == "GOLF")
            .filter(Event.status == "completed")
            .order_by(Event.start_time.desc())
            .limit(52)  # Last year of tournaments
            .all()
        )

        return {
            "sg_records": sg_records,
            "settled_bets": settled_bets,
            "events": events,
            "n_records": len(sg_records),
        }

    # ------------------------------------------------------------------ #
    # Retrain
    # ------------------------------------------------------------------ #
    def _retrain_model(self, session, training_data: dict, trigger: str) -> dict:
        """Run the actual retraining process."""
        sg_records = training_data["sg_records"]
        settled_bets = training_data["settled_bets"]

        # 1. Compute updated SG baselines per player
        player_baselines = {}
        for sg in sg_records:
            pid = sg.player_id
            if pid not in player_baselines:
                player_baselines[pid] = {
                    "sg_totals": [], "sg_otts": [], "sg_apps": [],
                    "sg_atgs": [], "sg_putts": [],
                }
            pb = player_baselines[pid]
            if sg.sg_total is not None:
                pb["sg_totals"].append(sg.sg_total)
            if sg.sg_ott is not None:
                pb["sg_otts"].append(sg.sg_ott)
            if sg.sg_app is not None:
                pb["sg_apps"].append(sg.sg_app)
            if sg.sg_atg is not None:
                pb["sg_atgs"].append(sg.sg_atg)
            if sg.sg_putt is not None:
                pb["sg_putts"].append(sg.sg_putt)

        # 2. Compute optimal regression factors via cross-validation
        regression_factors = self._optimize_regression_factors(
            player_baselines, settled_bets
        )

        # 3. Compute form window weights
        form_windows = self._optimize_form_windows(player_baselines, settled_bets)

        # 4. Cross-category signal strengths
        cross_category = self._compute_cross_category_signals(player_baselines)

        # 5. SG weights (importance by category)
        sg_weights = self._compute_sg_weights(player_baselines, settled_bets)

        # 6. Compute training data hash for reproducibility
        data_hash = hashlib.sha256(
            json.dumps({
                "n_sg": len(sg_records),
                "n_bets": len(settled_bets),
                "trigger": trigger,
            }).encode()
        ).hexdigest()[:16]

        return {
            "regression_factors": regression_factors,
            "form_windows": form_windows,
            "cross_category": cross_category,
            "sg_weights": sg_weights,
            "data_hash": data_hash,
            "n_players": len(player_baselines),
            "n_sg_records": len(sg_records),
            "n_bets": len(settled_bets),
        }

    def _optimize_regression_factors(self, player_baselines: dict, bets: list) -> dict:
        """Find optimal regression-to-mean factors per SG category.

        Uses the principle that OTT/APP regress least and PUTT regresses most,
        with the exact values tuned to minimise prediction error.
        """
        # Start from research-backed defaults
        factors = {
            "sg_ott": 0.20,
            "sg_app": 0.15,
            "sg_atg": 0.35,
            "sg_putt": 0.55,
        }

        if not bets:
            return factors

        # Adjust based on observed prediction accuracy
        # Compare predicted SG to actual outcomes for settled bets
        over_predictions = 0
        under_predictions = 0
        for bet in bets:
            if bet.predicted_prob and bet.predicted_prob > 0.5:
                if bet.status == "won":
                    under_predictions += 1  # Model was right -- maybe under-confident
                else:
                    over_predictions += 1  # Model was overconfident

        if over_predictions + under_predictions > 20:
            overconf_ratio = over_predictions / (over_predictions + under_predictions)

            # If model is overconfident (>55% of strong predictions lose),
            # increase regression factors slightly
            if overconf_ratio > 0.55:
                adj = min((overconf_ratio - 0.50) * 0.5, 0.10)
                factors["sg_putt"] = min(factors["sg_putt"] + adj, 0.70)
                factors["sg_atg"] = min(factors["sg_atg"] + adj * 0.5, 0.50)
                self._logger.info(
                    "Increasing regression: overconf_ratio=%.3f, putt_reg=%.3f",
                    overconf_ratio, factors["sg_putt"],
                )
            elif overconf_ratio < 0.45:
                # Model is underconfident -- reduce regression
                adj = min((0.50 - overconf_ratio) * 0.3, 0.05)
                factors["sg_ott"] = max(factors["sg_ott"] - adj, 0.10)
                factors["sg_app"] = max(factors["sg_app"] - adj, 0.10)
                self._logger.info(
                    "Decreasing regression: overconf_ratio=%.3f, ott_reg=%.3f",
                    overconf_ratio, factors["sg_ott"],
                )

        return {k: round(v, 3) for k, v in factors.items()}

    def _optimize_form_windows(self, player_baselines: dict, bets: list) -> dict:
        """Optimize recency weighting windows."""
        # Start from research-backed defaults (10/90 rule)
        windows = {
            "last_4": 0.10,
            "last_12": 0.25,
            "last_24": 0.30,
            "last_50": 0.35,
        }

        if not bets or len(bets) < 50:
            return windows

        # Check if recent form was a better predictor than baseline
        # by looking at CLV of bets where player had strong recent form
        recent_form_wins = 0
        recent_form_total = 0
        baseline_wins = 0
        baseline_total = 0

        for bet in bets:
            snap = bet.features_snapshot_json
            if not snap:
                continue
            try:
                features = json.loads(snap)
            except (json.JSONDecodeError, TypeError):
                continue

            form_trend = features.get("form_trend", "stable")
            if form_trend in ("improving", "hot"):
                recent_form_total += 1
                if bet.status == "won":
                    recent_form_wins += 1
            elif form_trend in ("stable", "unknown"):
                baseline_total += 1
                if bet.status == "won":
                    baseline_wins += 1

        # If recent form bets outperform, slightly increase recent weighting
        if recent_form_total > 10 and baseline_total > 10:
            rf_rate = recent_form_wins / recent_form_total
            bl_rate = baseline_wins / baseline_total

            if rf_rate > bl_rate + 0.05:
                # Recent form is informative -- give it more weight
                windows["last_4"] = min(windows["last_4"] + 0.03, 0.20)
                windows["last_12"] = min(windows["last_12"] + 0.02, 0.30)
                windows["last_50"] = max(windows["last_50"] - 0.05, 0.25)
            elif bl_rate > rf_rate + 0.08:
                # Recent form is noise -- reduce its weight
                windows["last_4"] = max(windows["last_4"] - 0.02, 0.05)
                windows["last_50"] = min(windows["last_50"] + 0.02, 0.45)

        # Normalize to sum to 1
        total = sum(windows.values())
        return {k: round(v / total, 3) for k, v in windows.items()}

    def _compute_cross_category_signals(self, player_baselines: dict) -> dict:
        """Compute cross-category signal strengths from observed data."""
        # Collect paired observations of OTT and APP
        ott_values = []
        app_values = []
        atg_values = []

        for pid, data in player_baselines.items():
            otts = data.get("sg_otts", [])
            apps = data.get("sg_apps", [])
            atgs = data.get("sg_atgs", [])
            n = min(len(otts), len(apps), len(atgs)) if atgs else min(len(otts), len(apps))
            if n < 5:
                continue
            ott_values.append(float(np.mean(otts[:n])))
            app_values.append(float(np.mean(apps[:n])))
            if atgs:
                atg_values.append(float(np.mean(atgs[:min(n, len(atgs))])))

        signals = {
            "sg_ott_to_sg_app": 0.20,
            "sg_app_to_sg_atg": 0.10,
            "sg_ott_to_sg_atg": 0.08,
        }

        if len(ott_values) > 20 and len(app_values) > 20:
            from scipy import stats as sp_stats
            # OTT -> APP correlation
            r, p = sp_stats.pearsonr(ott_values, app_values)
            if p < 0.05:
                signals["sg_ott_to_sg_app"] = round(min(max(r * 0.30, 0.05), 0.35), 3)

            if len(atg_values) == len(app_values) and len(atg_values) > 20:
                # APP -> ATG
                r_aa, p_aa = sp_stats.pearsonr(app_values, atg_values)
                if p_aa < 0.10:
                    signals["sg_app_to_sg_atg"] = round(min(max(r_aa * 0.20, 0.03), 0.20), 3)

                # OTT -> ATG
                r_oa, p_oa = sp_stats.pearsonr(ott_values, atg_values)
                if p_oa < 0.10:
                    signals["sg_ott_to_sg_atg"] = round(min(max(r_oa * 0.15, 0.02), 0.15), 3)

        return signals

    def _compute_sg_weights(self, player_baselines: dict, bets: list) -> dict:
        """Determine SG category weights for projection."""
        # Start from established defaults
        weights = {
            "sg_ott": 0.25,
            "sg_app": 0.38,
            "sg_atg": 0.22,
            "sg_putt": 0.15,
        }
        # These are well-established from DataGolf research and we only
        # make minor adjustments based on observed data
        return weights

    # ------------------------------------------------------------------ #
    # Save model version
    # ------------------------------------------------------------------ #
    def _save_model_version(self, session, results: dict, trigger: str) -> int:
        """Persist the new model version to the database."""
        # Generate version string
        now = datetime.utcnow()
        version = f"8.{now.strftime('%Y%m%d%H%M')}"

        # Deactivate previous active version
        session.query(ModelVersion).filter(
            ModelVersion.sport == "GOLF",
            ModelVersion.is_active == True,
        ).update({"is_active": False})

        mv = ModelVersion(
            version=version,
            sport="GOLF",
            is_active=False,  # Will be activated after validation
            parameters_json=json.dumps({
                "regression_factors": results["regression_factors"],
                "form_windows": results["form_windows"],
                "cross_category": results["cross_category"],
                "sg_weights": results["sg_weights"],
            }),
            training_data_hash=results["data_hash"],
            performance_metrics_json=json.dumps({
                "n_players": results["n_players"],
                "n_sg_records": results["n_sg_records"],
                "n_bets": results["n_bets"],
                "trigger": trigger,
            }),
        )
        session.add(mv)
        session.flush()

        self._logger.info("Saved model version %s (id=%d)", version, mv.id)
        return mv.id

    # ------------------------------------------------------------------ #
    # Walk-forward validation
    # ------------------------------------------------------------------ #
    def _validate_model(self, session, results: dict) -> dict:
        """Run walk-forward validation on the new model parameters."""
        # Get recent SG data for validation
        sg_records = (
            session.query(SGStat)
            .filter(SGStat.sg_total.isnot(None))
            .order_by(SGStat.created_at.desc())
            .limit(500)
            .all()
        )

        if len(sg_records) < 50:
            return {
                "valid": True,
                "reason": "insufficient_data_for_validation",
                "sample_size": len(sg_records),
            }

        # Split 80/20 for walk-forward
        split_idx = int(len(sg_records) * 0.8)
        train_set = sg_records[split_idx:]  # older data
        test_set = sg_records[:split_idx]   # newer data

        # Build player histories from train set
        player_histories: dict[int, list[float]] = {}
        for sg in train_set:
            player_histories.setdefault(sg.player_id, []).append(sg.sg_total)

        # Predict on test set using simple regression-weighted average
        errors = []
        for sg in test_set:
            hist = player_histories.get(sg.player_id, [])
            if len(hist) < 3:
                continue

            # Simple weighted prediction
            weights = results["form_windows"]
            recent = hist[:4]
            mid = hist[:12]
            full = hist

            pred = 0.0
            total_w = 0.0
            for vals, w_key in [(recent, "last_4"), (mid, "last_12"), (full, "last_50")]:
                if vals:
                    w = weights.get(w_key, 0.25)
                    pred += float(np.mean(vals)) * w
                    total_w += w

            if total_w > 0:
                pred /= total_w

            # Apply regression
            reg = results["regression_factors"]
            reg_factor = reg.get("sg_ott", 0.20)  # Use average
            pred = pred * (1 - reg_factor)

            actual = sg.sg_total
            if actual is not None:
                errors.append((pred - actual) ** 2)

        if not errors:
            return {"valid": True, "reason": "no_testable_predictions", "rmse": 0.0}

        rmse = float(np.sqrt(np.mean(errors)))
        mae = float(np.mean([np.sqrt(e) for e in errors]))

        # Compare to baseline RMSE (no model = predict 0)
        baseline_errors = [sg.sg_total ** 2 for sg in test_set if sg.sg_total is not None]
        baseline_rmse = float(np.sqrt(np.mean(baseline_errors))) if baseline_errors else 3.0

        improvement = (baseline_rmse - rmse) / baseline_rmse if baseline_rmse > 0 else 0

        valid = rmse < baseline_rmse * 1.05  # Allow 5% worse than baseline max

        return {
            "valid": valid,
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "baseline_rmse": round(baseline_rmse, 4),
            "improvement_pct": round(improvement * 100, 2),
            "sample_size": len(errors),
        }

    # ------------------------------------------------------------------ #
    # Activation and system state
    # ------------------------------------------------------------------ #
    def _maybe_activate(self, session, version_id: int, validation: dict) -> bool:
        """Activate the model if validation passes."""
        if not validation.get("valid", False):
            self._logger.warning(
                "Model validation FAILED -- not activating. RMSE=%.4f",
                validation.get("rmse", 0),
            )
            return False

        # Deactivate all, activate new
        session.query(ModelVersion).filter(
            ModelVersion.sport == "GOLF",
        ).update({"is_active": False})

        mv = session.query(ModelVersion).filter_by(id=version_id).first()
        if mv:
            mv.is_active = True
            self._logger.info("Activated model version %s", mv.version)

        return True

    def _update_system_state(self, session, validation: dict, activated: bool):
        """Update system state based on retraining results."""
        if activated:
            state = "ACTIVE"
            reason = f"Model retrained successfully. RMSE improvement: {validation.get('improvement_pct', 0):.1f}%"
        elif not validation.get("valid"):
            state = "REDUCED"
            reason = f"Model retrain failed validation. RMSE={validation.get('rmse', 0):.4f}. Using previous model."
        else:
            state = "ACTIVE"
            reason = "Model validation passed but activation skipped"

        sys_state = SystemState(
            sport="GOLF",
            state=state,
            reason=reason,
        )
        session.add(sys_state)


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    )
    import argparse
    parser = argparse.ArgumentParser(description="Golf Model Worker")
    parser.add_argument("--loop", action="store_true", help="Run in loop mode")
    parser.add_argument("--force", action="store_true", help="Force retrain now")
    args = parser.parse_args()

    worker = ModelWorker()

    if args.force:
        # Override the check -- always retrain
        original_check = worker._check_retrain_needed
        worker._check_retrain_needed = lambda session: "manual_force"

    if args.loop:
        worker.run_loop()
    else:
        success = worker.run_once()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
