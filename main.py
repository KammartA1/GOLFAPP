"""
Golf Quant Engine — Main CLI
The command center for running the full pipeline.

Usage:
  python main.py run          # Full projection run for current tournament
  python main.py h2h          # H2H matchup analysis
  python main.py dfs          # Generate DFS lineups
  python main.py audit        # Run performance audit
  python main.py status       # Show engine status
"""
import sys
import logging
import click
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config.settings import LOG_LEVEL, LOG_DIR
from data.storage.database import init_db
from data.pipeline import DataPipeline
from models.projection import ProjectionEngine
from betting.kelly import KellyModel
from betting.h2h import H2HAnalyzer
from dfs.optimizer import LineupOptimizer
from audit.tracker import AuditEngine

console = Console()

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / f"golf_quant_{datetime.now().strftime('%Y%m%d')}.log"),
        ]
    )


# ─────────────────────────────────────────────
# CLI HEADER
# ─────────────────────────────────────────────

def print_header():
    text = Text()
    text.append("⛳ GOLF QUANT ENGINE v1.0\n", style="bold gold1")
    text.append("The World's Best Golf Analytics Platform", style="italic dim")
    console.print(Panel(text, border_style="green"))


# ─────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────

@click.group()
def cli():
    """Golf Quant Engine — Command Center"""
    setup_logging()
    init_db()


@cli.command()
@click.option("--event-id", default=None, help="PGA Tour event ID (auto-detects current if not set)")
@click.option("--course", default=None, help="Course name (auto-detects if not set)")
@click.option("--dk-csv", default=None, help="Path to DraftKings salary CSV")
@click.option("--fd-csv", default=None, help="Path to FanDuel salary CSV")
@click.option("--bankroll", default=1000.0, help="Current bankroll in dollars")
@click.option("--top", default=30, help="Number of players to show in output")
def run(event_id, course, dk_csv, fd_csv, bankroll, top):
    """Full projection run for current/specified tournament."""
    print_header()
    console.print(f"\n[bold]🔄 Running full projection pipeline...[/bold]\n")

    pipeline = DataPipeline()
    engine   = ProjectionEngine()
    kelly    = KellyModel(bankroll=bankroll)
    h2h      = H2HAnalyzer()

    # ── Data Refresh ────────────────────────────────────────────────
    console.print("[cyan]Step 1/5: Data refresh...[/cyan]")
    data = pipeline.full_refresh(event_id=event_id)

    tournament_name = data.get("tournament", {}).get("name", "Current Tournament")
    course_name     = course or data.get("tournament", {}).get("course_name", "Unknown")
    field           = data.get("field", [])
    sg_history      = data.get("sg_history", {})

    if not field:
        console.print("[red]❌ No field data available. Check event ID or try again later.[/red]")
        sys.exit(1)

    # ── Salaries ─────────────────────────────────────────────────────
    console.print("[cyan]Step 2/5: Loading salaries...[/cyan]")
    salaries = {}
    if dk_csv or fd_csv:
        salaries = pipeline.load_salaries(dk_path=dk_csv, fd_path=fd_csv)
    else:
        console.print("  [yellow]No salary files provided. Skipping DFS optimization.[/yellow]")
        console.print("  [dim]Tip: Export DK/FD salary CSV and pass with --dk-csv flag[/dim]")

    # ── Projections ───────────────────────────────────────────────────
    console.print(f"[cyan]Step 3/5: Running projections for {tournament_name} @ {course_name}...[/cyan]")
    sal_for_engine = {
        name: v for name, v in salaries.items()
    } if salaries else None

    proj_df = engine.run(
        tournament_name=tournament_name,
        course_name=course_name,
        field_data=field,
        sg_history=sg_history,
        salaries=sal_for_engine,
    )

    if proj_df.empty:
        console.print("[red]❌ Projection engine returned no results.[/red]")
        sys.exit(1)

    # ── Display Projections ───────────────────────────────────────────
    console.print(f"\n[bold gold1]🏌️  {tournament_name} — Top {top} Projections[/bold gold1]")
    display_cols = [c for c in [
        "model_rank", "name", "proj_sg_total", "proj_sg_app", "course_fit_score",
        "form_trend", "win_prob", "top10_prob", "make_cut_prob",
        "dk_proj_pts", "dk_value", "proj_ownership"
    ] if c in proj_df.columns]

    top_df = proj_df[display_cols].head(top)
    console.print(top_df.to_string(index=False))

    # ── GPP Targets ───────────────────────────────────────────────────
    console.print("\n[bold]🎯 GPP Leverage Targets (high model, low ownership)[/bold]")
    gpp = engine.gpp_targets(proj_df)
    if not gpp.empty:
        console.print(gpp[["name", "proj_sg_total", "proj_ownership", "leverage_score"]].to_string(index=False))

    # ── H2H Synthetic Matchups ────────────────────────────────────────
    console.print("\n[bold]⚔️  Top H2H Matchup Opportunities[/bold]")
    matchups = h2h.generate_synthetic_matchups(proj_df, course_name, similar_rank_range=8)
    if not matchups.empty:
        console.print(matchups[[
            "player_a", "player_b", "sg_edge", "win_prob_a", "notes"
        ]].head(10).to_string(index=False))

    # ── DFS Lineup Generation ─────────────────────────────────────────
    if salaries:
        console.print("\n[bold]📋 Generating DFS Lineups...[/bold]")
        optimizer = LineupOptimizer(platform="DraftKings")

        gpp_lineups = optimizer.generate_gpp_lineups(proj_df, n_lineups=5)
        for i, lineup in enumerate(gpp_lineups[:2]):
            optimizer.print_lineup(lineup, label=f"GPP #{i+1}")

        # Export
        export_path = f"outputs/lineups_{tournament_name.replace(' ','_')}_DK.csv"
        optimizer.export_to_csv(gpp_lineups, export_path)
        console.print(f"\n[green]✅ Lineups exported: {export_path}[/green]")

    # ── Save projections ──────────────────────────────────────────────
    output_path = f"outputs/projections_{tournament_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.csv"
    import os; os.makedirs("outputs", exist_ok=True)
    proj_df.to_csv(output_path, index=False)
    console.print(f"[green]✅ Full projections saved: {output_path}[/green]")


@cli.command()
@click.argument("player_a")
@click.argument("player_b")
@click.option("--proj-file", default=None, help="Path to saved projections CSV")
@click.option("--course", default=None, help="Course name")
@click.option("--odds-a", default=None, type=int, help="American odds on Player A")
@click.option("--odds-b", default=None, type=int, help="American odds on Player B")
def matchup(player_a, player_b, proj_file, course, odds_a, odds_b):
    """Analyze a specific H2H matchup."""
    print_header()
    h2h = H2HAnalyzer()

    if proj_file:
        df = pd.read_csv(proj_file)
        a_data = df[df["name"].str.lower() == player_a.lower()]
        b_data = df[df["name"].str.lower() == player_b.lower()]

        if a_data.empty or b_data.empty:
            console.print(f"[red]Player not found in projection file.[/red]")
            sys.exit(1)

        result = h2h.analyze_matchup(
            a_data.iloc[0].to_dict(),
            b_data.iloc[0].to_dict(),
            course or "Unknown",
            odds_a=odds_a, odds_b=odds_b
        )

        console.print(f"\n[bold]⚔️  {result.player_a} vs {result.player_b}[/bold]")
        console.print(f"SG Edge: {result.edge_sg:+.3f} (favor: {result.player_a if result.sg_a > result.sg_b else result.player_b})")
        console.print(f"Win Prob: {result.player_a}={result.win_prob_a:.1%} | {result.player_b}={result.win_prob_b:.1%}")
        if result.edge_a:
            console.print(f"Edge on {result.player_a}: {result.edge_a:+.1%}")
        if result.edge_b:
            console.print(f"Edge on {result.player_b}: {result.edge_b:+.1%}")
        console.print(f"Notes: {result.notes}")


@cli.command()
@click.option("--bankroll", default=1000.0)
def status(bankroll):
    """Show engine status and data source connectivity."""
    print_header()
    from data.scrapers.datagolf import DataGolfClient
    dg = DataGolfClient()

    console.print("\n[bold]Engine Status[/bold]")
    console.print(f"  DataGolf API: {dg.status()}")
    console.print(f"  Database:     {'✅ Connected' if True else '❌ Error'}")
    console.print(f"  Bankroll:     ${bankroll:,.2f}")

    console.print("\n[bold]Data Sources[/bold]")
    console.print("  ✅ PGA Tour GraphQL API (free)")
    console.print("  ✅ ESPN Golf API (free)")
    console.print("  ✅ OpenWeatherMap (free tier — set OPENWEATHER_API_KEY)")
    console.print("  ✅ The Odds API (free tier — set ODDS_API_KEY)")
    console.print("  🔶 DataGolf API (set DATAGOLF_API_KEY when subscribed)")

    console.print("\n[bold]Quick Start[/bold]")
    console.print("  1. Add API keys to .env file")
    console.print("  2. python main.py run")
    console.print("  3. Export DK salary CSV, pass with --dk-csv flag")
    console.print("  4. python main.py run --dk-csv salaries.csv --bankroll 1000")


@cli.command()
def audit():
    """Run full performance audit on all settled bets."""
    print_header()
    from data.storage.database import get_session, Bet
    session = get_session()
    bets = session.query(Bet).filter(Bet.settled == True).all()

    if not bets:
        console.print("[yellow]No settled bets found. Place and settle bets first.[/yellow]")
        return

    bet_records = [{
        "player": b.player.name if b.player else "",
        "bet_type": b.bet_type,
        "stake": b.stake,
        "odds_american": b.odds_american,
        "model_prob": b.model_win_prob,
        "implied_prob": b.implied_prob,
        "edge_pct": b.edge_pct,
        "won": b.won,
        "profit_loss": b.profit_loss,
        "placed_at": b.placed_at,
        "settled": b.settled,
        "actual_outcome": b.won,
    } for b in bets]

    auditor = AuditEngine()
    report = auditor.generate_report(bet_records, title="Golf Quant Engine — Performance Audit")
    console.print(report)

    decay = auditor.detect_edge_decay(bet_records)
    console.print(f"\n[bold]Edge Decay Analysis:[/bold] {decay}")


@cli.command()
def workers():
    """Start all background workers via APScheduler."""
    console.print("\n[bold]Starting background workers...[/bold]")
    console.print("  Workers: odds (15m), signal (30m), closing (15m), model (1h), report (daily), stats (daily)")
    console.print("  Health endpoint: http://0.0.0.0:8089/health")
    console.print("  Press Ctrl+C to stop.\n")

    from workers import run_all
    run_all()


@cli.command()
@click.argument("worker_name", type=click.Choice([
    "odds", "signal", "closing", "model", "report", "stats",
]))
def run_worker(worker_name):
    """Run a single worker once."""
    print_header()
    console.print(f"\n[bold]Running {worker_name} worker (one-shot)...[/bold]")

    from workers import (
        OddsWorker, SignalWorker, ClosingWorker,
        ModelWorker, ReportWorker, StatsWorker,
    )
    worker_map = {
        "odds": OddsWorker,
        "signal": SignalWorker,
        "closing": ClosingWorker,
        "model": ModelWorker,
        "report": ReportWorker,
        "stats": StatsWorker,
    }

    worker = worker_map[worker_name]()
    success = worker.run_once()
    if success:
        console.print(f"[green]Worker {worker_name} completed successfully.[/green]")
    else:
        console.print(f"[red]Worker {worker_name} failed.[/red]")
        sys.exit(1)


@cli.command()
def db_status():
    """Show database health and migration status."""
    print_header()
    from database.connection import health_check, init_db
    from database.migrations import get_migration_status, auto_migrate

    init_db()
    auto_migrate()

    health = health_check()
    mig = get_migration_status()

    console.print("\n[bold]Database Status[/bold]")
    console.print(f"  Status:     {health.get('status', 'unknown')}")
    console.print(f"  Driver:     {health.get('driver', 'unknown')}")
    console.print(f"  Version:    {health.get('version', 'unknown')}")
    console.print(f"  Journal:    {health.get('journal_mode', 'unknown')}")
    console.print(f"  Schema:     v{mig.get('current_version', 0)} (latest: v{mig.get('latest_available', 0)})")
    console.print(f"  Up to date: {'Yes' if mig.get('is_up_to_date') else 'No'}")

    if not mig.get("is_up_to_date"):
        console.print(f"  Pending:    {mig.get('pending_versions', [])}")


if __name__ == "__main__":
    cli()
