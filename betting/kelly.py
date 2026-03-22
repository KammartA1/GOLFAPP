"""
Kelly Criterion & Bankroll Management
Golf-specific sizing model.

Golf is high variance even with edge — always use fractional Kelly.
The key insight: you need a large sample to validate edge, so preserve bankroll
long enough to get there.
"""
import logging
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

from config.settings import (
    KELLY_FRACTION, MAX_BET_PCT_BANKROLL, MIN_EDGE_THRESHOLD,
    MAX_GPP_EXPOSURE, MAX_H2H_EXPOSURE, MAX_OUTRIGHT_EXPOSURE,
    AUDIT_DIR
)

log = logging.getLogger(__name__)


@dataclass
class BetRecommendation:
    player: str
    bet_type: str               # "outright", "h2h", "top5", "top10", "top20", "make_cut"
    market: str                 # human-readable market description
    book: str
    odds_american: int
    model_prob: float           # our model's win probability
    implied_prob: float         # sportsbook implied probability
    edge_pct: float             # model_prob - implied_prob
    kelly_fraction: float       # optimal Kelly fraction
    kelly_rec_stake: float      # Kelly recommended stake in $
    capped_stake: float         # After applying bankroll caps
    bankroll: float             # current bankroll
    pct_bankroll: float         # capped_stake / bankroll
    expected_value: float       # EV in dollars
    rationale: str = ""
    placed: bool = False
    result: Optional[bool] = None
    profit_loss: Optional[float] = None


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal."""
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Implied probability from decimal odds."""
    return 1 / decimal_odds


def american_to_implied_prob(odds: int) -> float:
    return decimal_to_implied_prob(american_to_decimal(odds))


def kelly_fraction_calc(win_prob: float, decimal_odds: float) -> float:
    """
    Full Kelly fraction.
    f* = (bp - q) / b
    where b = net odds (decimal - 1), p = win prob, q = 1 - p
    """
    b = decimal_odds - 1
    p = win_prob
    q = 1 - p
    if b <= 0:
        return 0
    f_star = (b * p - q) / b
    return max(0, f_star)


class KellyModel:
    """
    Golf betting Kelly calculator with bankroll management.
    Tracks all recommendations and settles results.
    """

    def __init__(self, bankroll: float, kelly_frac: float = KELLY_FRACTION):
        self.bankroll = bankroll
        self.kelly_frac = kelly_frac
        self.bets: list[BetRecommendation] = []
        self.session_log_path = AUDIT_DIR / f"kelly_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def evaluate(
        self,
        player: str,
        bet_type: str,
        market: str,
        book: str,
        odds_american: int,
        model_prob: float,
    ) -> Optional[BetRecommendation]:
        """
        Evaluate a bet and return a recommendation if edge exists.

        Returns None if edge is below threshold or Kelly = 0.
        """
        implied_prob = american_to_implied_prob(odds_american)
        edge_pct = model_prob - implied_prob

        if edge_pct < MIN_EDGE_THRESHOLD:
            log.debug(f"No edge on {player} {bet_type}: {edge_pct:.2%} < {MIN_EDGE_THRESHOLD:.2%}")
            return None

        decimal_odds = american_to_decimal(odds_american)
        full_kelly = kelly_fraction_calc(model_prob, decimal_odds)
        fractional_kelly = full_kelly * self.kelly_frac

        # Raw Kelly stake
        kelly_stake = fractional_kelly * self.bankroll

        # Apply type-specific caps
        type_caps = {
            "outright":  MAX_OUTRIGHT_EXPOSURE,
            "h2h":       MAX_H2H_EXPOSURE,
            "top5":      MAX_H2H_EXPOSURE,
            "top10":     MAX_H2H_EXPOSURE,
            "top20":     MAX_H2H_EXPOSURE,
            "make_cut":  MAX_H2H_EXPOSURE,
            "dfs_gpp":   MAX_GPP_EXPOSURE,
            "dfs_cash":  MAX_GPP_EXPOSURE,
        }
        max_pct = type_caps.get(bet_type, MAX_BET_PCT_BANKROLL)
        max_stake = self.bankroll * max_pct
        capped_stake = min(kelly_stake, max_stake)

        # Minimum bet floor ($5)
        if capped_stake < 5:
            log.debug(f"Stake too small ({capped_stake:.2f}) for {player} — skipping")
            return None

        ev = capped_stake * (decimal_odds - 1) * model_prob - capped_stake * (1 - model_prob)

        rec = BetRecommendation(
            player=player,
            bet_type=bet_type,
            market=market,
            book=book,
            odds_american=odds_american,
            model_prob=round(model_prob, 4),
            implied_prob=round(implied_prob, 4),
            edge_pct=round(edge_pct, 4),
            kelly_fraction=round(fractional_kelly, 4),
            kelly_rec_stake=round(kelly_stake, 2),
            capped_stake=round(capped_stake, 2),
            bankroll=self.bankroll,
            pct_bankroll=round(capped_stake / self.bankroll, 4),
            expected_value=round(ev, 2),
        )
        return rec

    def evaluate_field(
        self,
        player_projections: list[dict],
        odds_data: dict,      # {player_name: {bet_type: american_odds}}
        bet_types: list[str] = None,
    ) -> list[BetRecommendation]:
        """
        Evaluate bets across entire field for given bet types.

        player_projections: list of dicts with model probabilities
        odds_data: {player: {bet_type: odds_american}}
        bet_types: which markets to evaluate ["outright", "top10", "h2h", ...]
        """
        bet_types = bet_types or ["outright", "top10", "top20", "make_cut"]
        recommendations = []

        PROB_FIELDS = {
            "outright":  "win_prob",
            "top5":      "top5_prob",
            "top10":     "top10_prob",
            "top20":     "top20_prob",
            "make_cut":  "make_cut_prob",
        }

        for proj in player_projections:
            name = proj.get("name", "")
            player_odds = odds_data.get(name, {})

            for bt in bet_types:
                odds = player_odds.get(bt)
                if not odds:
                    continue
                prob_field = PROB_FIELDS.get(bt)
                if not prob_field:
                    continue
                model_prob = proj.get(prob_field)
                if not model_prob:
                    continue

                rec = self.evaluate(
                    player=name,
                    bet_type=bt,
                    market=f"{name} {bt.replace('_', ' ').title()}",
                    book="TBD",
                    odds_american=odds,
                    model_prob=model_prob,
                )
                if rec:
                    recommendations.append(rec)

        # Sort by EV descending
        recommendations.sort(key=lambda r: r.expected_value, reverse=True)
        log.info(f"Found {len(recommendations)} bets with edge across {len(player_projections)} players")
        return recommendations

    def size_dfs_slate(self, bankroll: float = None) -> dict:
        """
        Return recommended DFS allocation for a slate.
        Splits between GPP and cash game exposure.
        """
        bk = bankroll or self.bankroll
        return {
            "total_dfs_allocation": round(bk * 0.05, 2),    # Max 5% per slate
            "gpp_allocation":       round(bk * 0.03, 2),    # 3% to GPPs
            "cash_allocation":      round(bk * 0.02, 2),    # 2% to cash
            "gpp_entries":          max(1, int(bk * 0.03 / 5)),   # Assumes $5 GPP entry
            "cash_entries":         max(1, int(bk * 0.02 / 5)),
        }

    def settle_bet(self, bet: BetRecommendation, won: bool, book: str = None) -> float:
        """Settle a bet and update bankroll."""
        decimal_odds = american_to_decimal(bet.odds_american)
        if won:
            profit = bet.capped_stake * (decimal_odds - 1)
            self.bankroll += profit
            bet.profit_loss = round(profit, 2)
        else:
            self.bankroll -= bet.capped_stake
            bet.profit_loss = round(-bet.capped_stake, 2)

        bet.result = won
        bet.placed = True
        self.bankroll = round(self.bankroll, 2)
        log.info(f"{'✅ WON' if won else '❌ LOST'} | {bet.player} {bet.bet_type} | "
                 f"P&L: ${bet.profit_loss:+.2f} | Bankroll: ${self.bankroll:.2f}")
        return bet.profit_loss

    def summary(self) -> dict:
        """Return bankroll and performance summary."""
        settled = [b for b in self.bets if b.result is not None]
        if not settled:
            return {"bankroll": self.bankroll, "bets_settled": 0}

        wins = [b for b in settled if b.result]
        total_staked = sum(b.capped_stake for b in settled)
        total_pl = sum(b.profit_loss for b in settled if b.profit_loss)
        roi = total_pl / total_staked if total_staked > 0 else 0

        return {
            "bankroll":     self.bankroll,
            "bets_placed":  len(settled),
            "wins":         len(wins),
            "losses":       len(settled) - len(wins),
            "win_rate":     round(len(wins) / len(settled), 3),
            "total_staked": round(total_staked, 2),
            "total_pl":     round(total_pl, 2),
            "roi":          round(roi, 4),
        }

    def print_recommendations(self, recs: list[BetRecommendation], top_n: int = 10):
        """Pretty print top betting recommendations."""
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(title=f"🏌️ Top {top_n} Betting Recommendations", show_lines=True)
        table.add_column("Player", style="bold")
        table.add_column("Type")
        table.add_column("Odds")
        table.add_column("Model%", justify="right")
        table.add_column("Implied%", justify="right")
        table.add_column("Edge", justify="right", style="green")
        table.add_column("Stake", justify="right", style="yellow")
        table.add_column("EV", justify="right", style="cyan")

        for rec in recs[:top_n]:
            table.add_row(
                rec.player,
                rec.bet_type,
                f"{rec.odds_american:+d}",
                f"{rec.model_prob:.1%}",
                f"{rec.implied_prob:.1%}",
                f"{rec.edge_pct:+.1%}",
                f"${rec.capped_stake:.2f}",
                f"${rec.expected_value:+.2f}",
            )
        console.print(table)


def run_golf_monte_carlo_backtest(
    n_bets=1000, bankroll_start=1000.0,
    min_edge=0.06, kelly_frac=0.40, max_cap=0.08,
    edge_mean=0.09, edge_std=0.035,
    win_prob_mean=0.60, win_prob_std=0.07,
    price_decimal=2.0, seed=42,
):
    """
    [v6.0] Monte Carlo backtest for golf quant engine.
    Simulates n_bets with realistic edge distributions from SG-model data.
    Returns ROI, Sharpe, max drawdown, win rate, bankroll curve.
    """
    import math
    rng = np.random.default_rng(seed)
    bankroll = float(bankroll_start)
    total_staked = 0.0
    total_profit = 0.0
    wins = 0
    losses = 0
    bets_placed = 0
    bankroll_curve = [bankroll]
    max_bankroll = bankroll
    max_drawdown = 0.0
    per_bet = []

    for i in range(n_bets * 4):
        if bets_placed >= n_bets or bankroll <= 10:
            break
        _ev = float(rng.normal(edge_mean, edge_std))
        _prob = float(np.clip(rng.normal(win_prob_mean, win_prob_std), 0.35, 0.88))
        if _ev < min_edge:
            continue
        b = price_decimal - 1.0
        q = 1.0 - _prob
        k_full = max(0, (b * _prob - q) / b)
        if k_full <= 0:
            continue
        # Confidence-based multiplier
        if _prob >= 0.68:
            _mult = 1.50
        elif _prob >= 0.62:
            _mult = 1.10
        elif _prob >= 0.57:
            _mult = 0.65
        else:
            _mult = 0.0
        if _mult == 0:
            continue
        f = min(kelly_frac * _mult * k_full, max_cap)
        stake = bankroll * f
        if stake < 1.0:
            continue
        won = rng.random() < _prob
        if won:
            profit = stake * (price_decimal - 1.0)
            wins += 1
        else:
            profit = -stake
            losses += 1
        bankroll += profit
        total_staked += stake
        total_profit += profit
        bets_placed += 1
        max_bankroll = max(max_bankroll, bankroll)
        _dd = (max_bankroll - bankroll) / max_bankroll if max_bankroll > 0 else 0
        max_drawdown = max(max_drawdown, _dd)
        bankroll_curve.append(bankroll)
        per_bet.append({
            "bet_num": bets_placed, "ev": round(_ev, 4), "prob": round(_prob, 4),
            "stake": round(stake, 2), "won": won, "profit": round(profit, 2),
            "bankroll": round(bankroll, 2),
        })

    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    win_rate = (wins / bets_placed * 100) if bets_placed > 0 else 0.0
    if per_bet:
        returns = [b["profit"] / b["stake"] if b["stake"] > 0 else 0 for b in per_bet]
        _mean_r = float(np.mean(returns))
        _std_r = float(np.std(returns)) if len(returns) > 1 else 1.0
        sharpe = (_mean_r / _std_r * math.sqrt(600)) if _std_r > 0 else 0.0
    else:
        sharpe = 0.0
    return {
        "roi_pct": round(roi, 2), "total_profit": round(total_profit, 2),
        "total_staked": round(total_staked, 2), "bets_placed": bets_placed,
        "bets_filtered": i + 1 - bets_placed,
        "win_rate_pct": round(win_rate, 2), "wins": wins, "losses": losses,
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "final_bankroll": round(bankroll, 2),
        "bankroll_curve": bankroll_curve,
    }
