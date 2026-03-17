"""
DFS Lineup Optimizer
Linear programming-based optimizer for DraftKings / FanDuel golf.

Key features:
  - Salary cap compliance
  - GPP ownership leverage constraints
  - Cash game floor optimization
  - Correlation stacking (same tee time, co-variance plays)
  - Multi-lineup generation with player exposure controls
"""
import logging
import itertools
import random
import pandas as pd
import numpy as np

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    logging.warning("PuLP not installed. Run: pip install pulp --break-system-packages")

from config.settings import (
    DK_SALARY_CAP, FD_SALARY_CAP, DK_ROSTER_SIZE, FD_ROSTER_SIZE,
    TARGET_LINEUPS_GPP, TARGET_LINEUPS_CASH, MAX_LINEUP_OWNERSHIP,
    CHALK_THRESHOLD
)

log = logging.getLogger(__name__)


class LineupOptimizer:
    """
    LP-based DFS lineup optimizer for golf.
    Supports both DraftKings and FanDuel.
    """

    def __init__(self, platform: str = "DraftKings"):
        self.platform = platform
        self.salary_cap = DK_SALARY_CAP if platform == "DraftKings" else FD_SALARY_CAP
        self.roster_size = DK_ROSTER_SIZE if platform == "DraftKings" else FD_ROSTER_SIZE
        self.salary_col = "dk_salary" if platform == "DraftKings" else "fd_salary"
        self.pts_col = "dk_proj_pts" if platform == "DraftKings" else "fd_proj_pts"

    def _prep_pool(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter and validate player pool."""
        required = [self.salary_col, self.pts_col, "name"]
        pool = df.copy()

        for col in required:
            if col not in pool.columns:
                raise ValueError(f"Missing column: {col}")

        pool = pool.dropna(subset=[self.salary_col, self.pts_col])
        pool = pool[pool[self.salary_col] > 0]
        pool = pool.reset_index(drop=True)
        log.info(f"Player pool: {len(pool)} eligible players")
        return pool

    def optimize_single(
        self,
        df: pd.DataFrame,
        locked_players: list[str] = None,
        banned_players: list[str] = None,
        max_ownership_sum: float = None,
        min_pts_floor: float = None,
        pts_col_override: str = None,
    ) -> list[dict]:
        """
        Generate a single optimal lineup using LP.

        Returns list of player dicts making up the lineup.
        """
        if not PULP_AVAILABLE:
            log.error("PuLP not available — using greedy fallback")
            return self._greedy_optimize(df, locked_players, banned_players)

        pool = self._prep_pool(df)
        pts_col = pts_col_override or self.pts_col

        locked = set(locked_players or [])
        banned = set(banned_players or [])

        n = len(pool)
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

        prob = pulp.LpProblem("golf_dfs", pulp.LpMaximize)

        # Objective: maximize projected points
        prob += pulp.lpSum([pool.iloc[i][pts_col] * x[i] for i in range(n)])

        # Roster size constraint
        prob += pulp.lpSum(x) == self.roster_size

        # Salary cap
        prob += pulp.lpSum([pool.iloc[i][self.salary_col] * x[i] for i in range(n)]) <= self.salary_cap

        # Locked players must be included
        for player in locked:
            idx_list = pool.index[pool["name"] == player].tolist()
            if idx_list:
                prob += x[idx_list[0]] == 1

        # Banned players excluded
        for player in banned:
            idx_list = pool.index[pool["name"] == player].tolist()
            if idx_list:
                prob += x[idx_list[0]] == 0

        # Ownership cap (GPP only)
        if max_ownership_sum and "proj_ownership" in pool.columns:
            prob += pulp.lpSum([
                pool.iloc[i]["proj_ownership"] * 100 * x[i] for i in range(n)
            ]) <= max_ownership_sum

        # Min points floor
        if min_pts_floor:
            prob += pulp.lpSum([pool.iloc[i][pts_col] * x[i] for i in range(n)]) >= min_pts_floor

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] != "Optimal":
            log.warning("LP did not find optimal — trying greedy fallback")
            return self._greedy_optimize(df, locked_players, banned_players)

        lineup = []
        for i in range(n):
            if pulp.value(x[i]) == 1:
                lineup.append(pool.iloc[i].to_dict())

        return lineup

    def _greedy_optimize(
        self,
        df: pd.DataFrame,
        locked: list[str] = None,
        banned: list[str] = None,
    ) -> list[dict]:
        """Greedy optimizer fallback (value-based selection)."""
        pool = self._prep_pool(df).copy()
        locked = set(locked or [])
        banned = set(banned or [])

        pool = pool[~pool["name"].isin(banned)]
        pool["value"] = pool[self.pts_col] / (pool[self.salary_col] / 1000)
        pool = pool.sort_values("value", ascending=False)

        lineup = []
        remaining_cap = self.salary_cap

        # Add locked players first
        for player in locked:
            row = pool[pool["name"] == player]
            if not row.empty:
                r = row.iloc[0].to_dict()
                lineup.append(r)
                remaining_cap -= r[self.salary_col]
                pool = pool[pool["name"] != player]

        # Fill remaining spots greedily
        for _, row in pool.iterrows():
            if len(lineup) >= self.roster_size:
                break
            if row[self.salary_col] <= remaining_cap:
                lineup.append(row.to_dict())
                remaining_cap -= row[self.salary_col]

        return lineup

    def generate_gpp_lineups(
        self,
        df: pd.DataFrame,
        n_lineups: int = None,
        max_player_exposure: float = 0.60,
        lock_chalk: int = 1,
    ) -> list[list[dict]]:
        """
        Generate multiple diverse GPP lineups.

        Strategy:
          - Vary locked chalk player
          - Rotate leverage plays in/out
          - Control max player exposure (no player in >60% of lineups by default)
          - Target total lineup ownership under MAX_LINEUP_OWNERSHIP
        """
        n_lineups = n_lineups or TARGET_LINEUPS_GPP
        pool = self._prep_pool(df)

        # Identify chalk plays
        chalk = []
        if "proj_ownership" in pool.columns:
            chalk = pool[pool["proj_ownership"] >= CHALK_THRESHOLD]["name"].tolist()
            log.info(f"Chalk players: {chalk}")

        # Leverage plays (model likes, public underweights)
        leverage = []
        if "leverage_score" in pool.columns:
            leverage = pool[pool["leverage_score"] > 5].sort_values(
                "leverage_score", ascending=False
            )["name"].tolist()[:15]
            log.info(f"Leverage plays: {leverage}")

        lineups = []
        player_counts = {}

        # High-ceiling lineup first (pure LP optimal)
        base = self.optimize_single(df)
        if base:
            lineups.append(base)
            for p in base:
                player_counts[p["name"]] = player_counts.get(p["name"], 0) + 1

        attempts = 0
        while len(lineups) < n_lineups and attempts < n_lineups * 5:
            attempts += 1

            # Pick a random lock and ban to create diversity
            banned = []
            locked = []

            # Rotate chalk plays
            if chalk:
                if random.random() > 0.3:  # 70% of lineups include at least 1 chalk
                    locked.append(random.choice(chalk))

            # Force leverage plays into some lineups
            if leverage and random.random() > 0.4:
                locked.append(random.choice(leverage))

            # Ban over-exposed players to force diversity
            for player, count in player_counts.items():
                if count / max(len(lineups), 1) > max_player_exposure:
                    banned.append(player)

            # Ownership constraint for GPP
            own_cap = MAX_LINEUP_OWNERSHIP + random.uniform(-15, 10)

            try:
                lineup = self.optimize_single(
                    df,
                    locked_players=locked,
                    banned_players=banned,
                    max_ownership_sum=own_cap,
                )
                if lineup and len(lineup) == self.roster_size:
                    # Check if lineup is unique enough
                    lineup_names = frozenset(p["name"] for p in lineup)
                    existing = [frozenset(p["name"] for p in l) for l in lineups]
                    if lineup_names not in existing:
                        lineups.append(lineup)
                        for p in lineup:
                            player_counts[p["name"]] = player_counts.get(p["name"], 0) + 1
            except Exception as e:
                log.debug(f"Lineup generation error: {e}")

        log.info(f"Generated {len(lineups)} GPP lineups")
        return lineups

    def generate_cash_lineups(self, df: pd.DataFrame, n_lineups: int = 3) -> list[list[dict]]:
        """
        Cash game lineups: optimize for floor (consistent scorers, high make-cut prob).
        Use make_cut_prob-weighted points instead of raw pts.
        """
        pool = self._prep_pool(df).copy()

        if "make_cut_prob" in pool.columns:
            pool["cash_pts"] = pool[self.pts_col] * pool["make_cut_prob"]
        else:
            pool["cash_pts"] = pool[self.pts_col]

        lineups = []
        for _ in range(n_lineups):
            lineup = self.optimize_single(df, pts_col_override="cash_pts")
            if lineup:
                lineups.append(lineup)

        log.info(f"Generated {len(lineups)} cash lineups")
        return lineups

    def export_to_csv(self, lineups: list[list[dict]], filepath: str):
        """Export lineups to DraftKings/FanDuel upload format."""
        rows = []
        for i, lineup in enumerate(lineups):
            row = {"lineup_num": i + 1}
            for j, player in enumerate(lineup):
                row[f"G{j+1}"] = player.get("name", "")
            row["total_salary"] = sum(p.get(self.salary_col, 0) for p in lineup)
            row["proj_pts"] = round(sum(p.get(self.pts_col, 0) for p in lineup), 2)
            row["total_ownership"] = round(
                sum(p.get("proj_ownership", 0) * 100 for p in lineup), 1
            )
            rows.append(row)

        pd.DataFrame(rows).to_csv(filepath, index=False)
        log.info(f"Exported {len(lineups)} lineups to {filepath}")

    def print_lineup(self, lineup: list[dict], label: str = ""):
        """Print a single lineup nicely."""
        from rich.table import Table
        from rich.console import Console
        console = Console()

        sal_col = self.salary_col
        pts_col = self.pts_col

        table = Table(title=f"🏌️ {self.platform} Lineup {label}", show_lines=True)
        table.add_column("Player", style="bold")
        table.add_column("Salary", justify="right")
        table.add_column("Proj Pts", justify="right")
        table.add_column("Value", justify="right")
        table.add_column("Own%", justify="right")
        table.add_column("SG Proj", justify="right")

        total_sal = 0
        total_pts = 0

        for p in lineup:
            sal = p.get(sal_col, 0)
            pts = p.get(pts_col, 0)
            val = round(pts / (sal / 1000), 2) if sal else 0
            own = f"{p.get('proj_ownership', 0)*100:.1f}%"
            sg = p.get("proj_sg_total", 0)
            total_sal += sal
            total_pts += pts
            table.add_row(
                p.get("name", ""),
                f"${sal:,}",
                f"{pts:.1f}",
                f"{val:.2f}",
                own,
                f"{sg:+.3f}",
            )

        table.add_row(
            "[bold]TOTAL[/bold]", f"${total_sal:,}", f"{total_pts:.1f}", "", "", "",
            style="bold yellow"
        )
        console.print(table)
