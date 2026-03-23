"""
Golf Quant Engine — Tournament Simulator Tab
==============================================
Full Monte Carlo tournament simulation UI.
Select players, courses, weather, run simulation, view results.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional

from simulation.config import SimulationConfig
from simulation.course_model import (
    CourseModel,
    DEFAULT_COURSES,
    create_generic_course,
)
from simulation.player_model import (
    PlayerModel,
    DEFAULT_PLAYER_MODELS,
    create_elite_player,
    create_tour_average_player,
)
from simulation.tournament_engine import TournamentEngine, TournamentResult
from simulation.volatility_model import VolatilityModel


def render() -> None:
    """Render the Tournament Simulator tab."""
    st.markdown("# Tournament Simulator")
    st.markdown(
        "Monte Carlo simulation engine. Configure field, course, and weather, "
        "then run thousands of simulations to generate win/top-N/cut probabilities."
    )

    # Initialize session state
    if "sim_result" not in st.session_state:
        st.session_state.sim_result = None

    # --- Sidebar configuration ---
    with st.sidebar:
        st.markdown("### Simulation Settings")
        n_sims = st.slider(
            "Number of Simulations",
            min_value=100,
            max_value=50_000,
            value=5_000,
            step=100,
            help="More simulations = more accurate probabilities but slower",
        )
        seed = st.number_input("Random Seed", value=42, min_value=0)

    # --- Main layout ---
    config_col, run_col = st.columns([3, 1])

    with config_col:
        tab_field, tab_course, tab_weather = st.tabs([
            "Field Selection",
            "Course Selection",
            "Weather Settings",
        ])

        with tab_field:
            selected_players = _render_field_selection()

        with tab_course:
            selected_course = _render_course_selection()

        with tab_weather:
            weather_params = _render_weather_settings()

    with run_col:
        st.markdown("### Run")
        run_disabled = len(selected_players) < 2
        if run_disabled:
            st.warning("Select at least 2 players")

        if st.button(
            "Run Simulation",
            type="primary",
            disabled=run_disabled,
            use_container_width=True,
        ):
            _run_simulation(
                players=selected_players,
                course=selected_course,
                n_sims=n_sims,
                seed=seed,
                weather_params=weather_params,
            )

    st.divider()

    # --- Results display ---
    if st.session_state.sim_result is not None:
        _render_results(st.session_state.sim_result)


def _render_field_selection() -> list[PlayerModel]:
    """Field selection UI. Returns list of selected PlayerModels."""
    st.markdown("#### Select Players")

    # Preset fields
    preset = st.selectbox(
        "Preset Field",
        ["Custom", "Default Stars", "Full Default Field"],
    )

    if preset == "Default Stars":
        default_selected = [
            "Scottie Scheffler", "Rory McIlroy", "Jon Rahm",
            "Xander Schauffele", "Collin Morikawa",
        ]
    elif preset == "Full Default Field":
        default_selected = list(DEFAULT_PLAYER_MODELS.keys())
    else:
        default_selected = []

    # Multi-select from default players
    available = list(DEFAULT_PLAYER_MODELS.keys())
    selected_names = st.multiselect(
        "Select from known players",
        options=available,
        default=[n for n in default_selected if n in available],
    )

    selected = [DEFAULT_PLAYER_MODELS[n] for n in selected_names]

    # Custom player addition
    with st.expander("Add Custom Player"):
        custom_name = st.text_input("Player Name", value="Custom Player")
        col1, col2 = st.columns(2)
        with col1:
            sg_ott = st.number_input("SG: Off-the-Tee", value=0.0, step=0.1, format="%.2f")
            sg_app = st.number_input("SG: Approach", value=0.0, step=0.1, format="%.2f")
            sg_atg = st.number_input("SG: Around Green", value=0.0, step=0.1, format="%.2f")
            sg_putt = st.number_input("SG: Putting", value=0.0, step=0.1, format="%.2f")
        with col2:
            volatility = st.slider("Volatility", 0.5, 2.0, 1.0, 0.05)
            pressure = st.slider("Pressure Coeff", -1.0, 1.0, 0.0, 0.05)
            form = st.number_input("Recent Form Adj", value=0.0, step=0.05, format="%.2f")

        if st.button("Add Custom Player"):
            custom = PlayerModel(
                name=custom_name,
                sg_ott_mean=sg_ott,
                sg_app_mean=sg_app,
                sg_atg_mean=sg_atg,
                sg_putt_mean=sg_putt,
                volatility=volatility,
                pressure_coeff=pressure,
                recent_form=form,
            )
            selected.append(custom)
            st.success(f"Added {custom_name} (SG Total: {custom.sg_total_mean:+.2f})")

    # Generate filler players
    with st.expander("Add Random Filler Players"):
        n_fillers = st.number_input(
            "Number of filler players",
            value=0, min_value=0, max_value=150, step=5,
        )
        if n_fillers > 0:
            rng = np.random.default_rng(99)
            for i in range(n_fillers):
                sg = rng.normal(0, 0.6)
                filler = create_tour_average_player(f"Player_{i+1}")
                filler.sg_ott_mean = sg * 0.25
                filler.sg_app_mean = sg * 0.38
                filler.sg_atg_mean = sg * 0.22
                filler.sg_putt_mean = sg * 0.15
                filler.volatility = rng.uniform(0.7, 1.4)
                selected.append(filler)

    st.info(f"Field size: {len(selected)} players")
    return selected


def _render_course_selection() -> CourseModel:
    """Course selection UI."""
    st.markdown("#### Select Course")

    course_options = list(DEFAULT_COURSES.keys()) + ["Generic Course"]
    choice = st.selectbox("Course", course_options)

    if choice == "Generic Course":
        course_type = st.selectbox(
            "Course Type",
            ["parkland", "links", "desert", "coastal_parkland"],
        )
        bermuda = st.checkbox("Bermuda Greens", value=False)
        course = create_generic_course(
            name="Generic Course",
            course_type=course_type,
            bermuda=bermuda,
        )
    else:
        course = DEFAULT_COURSES[choice]

    # Display course info
    st.markdown(f"**Par:** {course.par} | **Yardage:** {course.total_yardage:,}")
    st.markdown(f"**Type:** {course.course_type} | **Bermuda:** {course.bermuda_greens}")
    st.markdown(f"**Wind Sensitivity:** {course.wind_sensitivity:.2f}")

    return course


def _render_weather_settings() -> dict:
    """Weather settings UI."""
    st.markdown("#### Weather Conditions")

    col1, col2 = st.columns(2)
    with col1:
        base_wind = st.slider("Base Wind (mph)", 0.0, 30.0, 10.0, 1.0)
        wind_dir = st.slider("Wind Direction (deg)", 0, 360, 180, 15)
    with col2:
        base_temp = st.slider("Temperature (F)", 40.0, 105.0, 72.0, 1.0)
        rain_prob = st.slider("Rain Probability", 0.0, 1.0, 0.15, 0.05)

    return {
        "base_wind": base_wind,
        "base_temp": base_temp,
        "rain_prob": rain_prob,
        "wind_direction": float(wind_dir),
    }


def _run_simulation(
    players: list[PlayerModel],
    course: CourseModel,
    n_sims: int,
    seed: int,
    weather_params: dict,
) -> None:
    """Execute the simulation with a progress bar."""
    config = SimulationConfig(n_simulations=n_sims, random_seed=seed)
    engine = TournamentEngine(config)

    progress_bar = st.progress(0, text="Running simulation...")

    def update_progress(current: int, total: int) -> None:
        progress_bar.progress(
            current / total,
            text=f"Simulation {current:,}/{total:,}",
        )

    result = engine.run_simulation(
        players=players,
        course=course,
        n_simulations=n_sims,
        seed=seed,
        weather_base_wind=weather_params["base_wind"],
        weather_base_temp=weather_params["base_temp"],
        weather_rain_prob=weather_params["rain_prob"],
        weather_wind_direction=weather_params["wind_direction"],
        tournament_name="Custom Simulation",
        progress_callback=update_progress,
    )

    progress_bar.progress(1.0, text="Complete!")
    st.session_state.sim_result = result
    st.success(
        f"Simulation complete! {n_sims:,} simulations, "
        f"{len(players)} players at {course.name}"
    )


def _render_results(result: TournamentResult) -> None:
    """Render all simulation results."""
    st.markdown("## Simulation Results")
    st.markdown(
        f"**{result.tournament_name}** at {result.course_name} | "
        f"{result.n_simulations:,} simulations | {result.n_players} players"
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Winning Score", f"{result.avg_winning_score:+.1f}")
    with col2:
        st.metric("Winning Score Std", f"{result.winning_score_std:.1f}")
    with col3:
        st.metric("Avg Cut Line", f"{result.avg_cut_line:+.1f}")
    with col4:
        st.metric("Cut Line Std", f"{result.cut_line_std:.1f}")

    # Tabs for different views
    tab_lb, tab_charts, tab_h2h, tab_rounds, tab_edge = st.tabs([
        "Leaderboard",
        "Probability Charts",
        "Head-to-Head",
        "Round Distributions",
        "Edge Detection",
    ])

    with tab_lb:
        _render_leaderboard(result)

    with tab_charts:
        _render_probability_charts(result)

    with tab_h2h:
        _render_h2h_tool(result)

    with tab_rounds:
        _render_round_distributions(result)

    with tab_edge:
        _render_edge_detection(result)


def _render_leaderboard(result: TournamentResult) -> None:
    """Display the simulation leaderboard."""
    sort_by = st.selectbox(
        "Sort by",
        ["win_prob", "top5_prob", "top10_prob", "top20_prob", "avg_score", "avg_finish"],
        index=0,
    )
    df = result.get_leaderboard(sort_by=sort_by)
    st.dataframe(df, use_container_width=True, height=600)


def _render_probability_charts(result: TournamentResult) -> None:
    """Bar charts for win%, top5%, top10%, top20%."""
    # Get top 20 players by win probability
    sorted_players = sorted(
        result.player_results.values(),
        key=lambda p: p.win_prob,
        reverse=True,
    )[:20]

    names = [p.player_name for p in sorted_players]

    # Win probability chart
    fig_win = go.Figure()
    fig_win.add_trace(go.Bar(
        x=names,
        y=[p.win_prob * 100 for p in sorted_players],
        marker_color="gold",
        text=[f"{p.win_prob*100:.1f}%" for p in sorted_players],
        textposition="outside",
    ))
    fig_win.update_layout(
        title="Win Probability",
        yaxis_title="Probability (%)",
        xaxis_tickangle=-45,
        height=400,
    )
    st.plotly_chart(fig_win, use_container_width=True)

    # Stacked top-N chart
    fig_topn = go.Figure()
    for label, key, color in [
        ("Win", "win_prob", "#FFD700"),
        ("Top 5", "top5_prob", "#3498db"),
        ("Top 10", "top10_prob", "#2ecc71"),
        ("Top 20", "top20_prob", "#9b59b6"),
    ]:
        fig_topn.add_trace(go.Bar(
            name=label,
            x=names,
            y=[getattr(p, key) * 100 for p in sorted_players],
            marker_color=color,
        ))
    fig_topn.update_layout(
        title="Position Probabilities (Top 20 Players)",
        barmode="group",
        yaxis_title="Probability (%)",
        xaxis_tickangle=-45,
        height=450,
    )
    st.plotly_chart(fig_topn, use_container_width=True)

    # Make cut probability
    fig_cut = go.Figure()
    sorted_by_cut = sorted(sorted_players, key=lambda p: p.make_cut_prob, reverse=True)
    fig_cut.add_trace(go.Bar(
        x=[p.player_name for p in sorted_by_cut],
        y=[p.make_cut_prob * 100 for p in sorted_by_cut],
        marker_color="#e74c3c",
        text=[f"{p.make_cut_prob*100:.0f}%" for p in sorted_by_cut],
        textposition="outside",
    ))
    fig_cut.update_layout(
        title="Make Cut Probability",
        yaxis_title="Probability (%)",
        xaxis_tickangle=-45,
        height=400,
    )
    st.plotly_chart(fig_cut, use_container_width=True)


def _render_h2h_tool(result: TournamentResult) -> None:
    """Head-to-head comparison tool."""
    st.markdown("#### Head-to-Head Comparison")

    player_names = list(result.player_results.keys())
    if len(player_names) < 2:
        st.warning("Need at least 2 players for H2H comparison")
        return

    col1, col2 = st.columns(2)
    with col1:
        player_a = st.selectbox("Player A", player_names, index=0)
    with col2:
        default_b = 1 if len(player_names) > 1 else 0
        player_b = st.selectbox("Player B", player_names, index=default_b)

    if player_a == player_b:
        st.warning("Select two different players")
        return

    prob_a = result.get_h2h_probability(player_a, player_b)
    prob_b = 1.0 - prob_a

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(player_a, f"{prob_a*100:.1f}%")
    with col2:
        edge = abs(prob_a - 0.5) * 2
        st.metric("Edge", f"{edge*100:.1f}%")
    with col3:
        st.metric(player_b, f"{prob_b*100:.1f}%")

    # Visual bar
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[prob_a * 100],
        y=["H2H"],
        orientation="h",
        name=player_a,
        marker_color="#3498db",
        text=[f"{player_a}: {prob_a*100:.1f}%"],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=[prob_b * 100],
        y=["H2H"],
        orientation="h",
        name=player_b,
        marker_color="#e74c3c",
        text=[f"{player_b}: {prob_b*100:.1f}%"],
        textposition="inside",
    ))
    fig.update_layout(barmode="stack", height=150, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed comparison
    res_a = result.player_results[player_a]
    res_b = result.player_results[player_b]

    comp_df = pd.DataFrame({
        "Metric": ["Win%", "Top 5%", "Top 10%", "Top 20%", "Make Cut%", "Avg Score", "Score Std", "Avg Finish"],
        player_a: [
            f"{res_a.win_prob*100:.2f}", f"{res_a.top5_prob*100:.2f}",
            f"{res_a.top10_prob*100:.2f}", f"{res_a.top20_prob*100:.2f}",
            f"{res_a.make_cut_prob*100:.1f}", f"{res_a.avg_total_score:+.1f}",
            f"{res_a.score_std:.1f}", f"{res_a.avg_finish_position:.0f}",
        ],
        player_b: [
            f"{res_b.win_prob*100:.2f}", f"{res_b.top5_prob*100:.2f}",
            f"{res_b.top10_prob*100:.2f}", f"{res_b.top20_prob*100:.2f}",
            f"{res_b.make_cut_prob*100:.1f}", f"{res_b.avg_total_score:+.1f}",
            f"{res_b.score_std:.1f}", f"{res_b.avg_finish_position:.0f}",
        ],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # H2H matrix for top players
    with st.expander("Full H2H Matrix (Top 10 Players)"):
        top_names = [
            p.player_name for p in sorted(
                result.player_results.values(),
                key=lambda p: p.win_prob,
                reverse=True,
            )[:10]
        ]
        if len(top_names) >= 2:
            matrix = result.get_h2h_matrix(top_names)
            st.dataframe(
                matrix.style.format("{:.1%}").background_gradient(
                    cmap="RdYlGn", vmin=0.3, vmax=0.7
                ),
                use_container_width=True,
            )


def _render_round_distributions(result: TournamentResult) -> None:
    """Round-by-round scoring distributions."""
    st.markdown("#### Round Score Distributions")

    player_names = sorted(result.player_results.keys())
    selected = st.multiselect(
        "Select players to compare",
        player_names,
        default=player_names[:3] if len(player_names) >= 3 else player_names,
    )

    if not selected:
        return

    for round_num, round_label in [(1, "Round 1"), (2, "Round 2"), (3, "Round 3"), (4, "Round 4")]:
        fig = go.Figure()
        for name in selected:
            pr = result.player_results[name]
            scores_attr = f"round_scores_r{round_num}"
            scores = getattr(pr, scores_attr, [])
            if not scores:
                continue
            scores_clean = [s for s in scores if not np.isnan(s)]
            if not scores_clean:
                continue
            fig.add_trace(go.Histogram(
                x=scores_clean,
                name=name,
                opacity=0.6,
                nbinsx=25,
            ))
        fig.update_layout(
            title=round_label,
            xaxis_title="Score Relative to Par",
            yaxis_title="Frequency",
            barmode="overlay",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_edge_detection(result: TournamentResult) -> None:
    """Edge detection: overlay simulated probabilities with sportsbook odds."""
    st.markdown("#### Edge Detection")
    st.markdown(
        "Enter sportsbook odds to compare against simulated probabilities. "
        "Positive edge = our model sees value the book is missing."
    )

    player_names = sorted(result.player_results.keys())

    # Market type
    market = st.selectbox(
        "Market Type",
        ["Outright Winner", "Top 5", "Top 10", "Top 20", "Make Cut"],
    )

    market_map = {
        "Outright Winner": "win_prob",
        "Top 5": "top5_prob",
        "Top 10": "top10_prob",
        "Top 20": "top20_prob",
        "Make Cut": "make_cut_prob",
    }
    prob_key = market_map[market]

    st.markdown("Enter American odds for each player (e.g., +2500, -150):")

    edges = []
    with st.form("edge_form"):
        for name in player_names[:20]:
            pr = result.player_results[name]
            sim_prob = getattr(pr, prob_key)
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            with col1:
                st.text(name)
            with col2:
                st.text(f"Sim: {sim_prob*100:.1f}%")
            with col3:
                odds = st.number_input(
                    f"Odds ({name})",
                    value=0,
                    step=50,
                    key=f"odds_{name}",
                    label_visibility="collapsed",
                )
            with col4:
                if odds != 0:
                    implied = _american_to_prob(odds)
                    edge = sim_prob - implied
                    edges.append({
                        "Player": name,
                        "Sim Prob": f"{sim_prob*100:.1f}%",
                        "Book Prob": f"{implied*100:.1f}%",
                        "Edge": f"{edge*100:+.1f}%",
                        "edge_val": edge,
                    })
                    if edge > 0.02:
                        st.markdown(f"**+{edge*100:.1f}%**")
                    elif edge < -0.02:
                        st.markdown(f"{edge*100:.1f}%")
                    else:
                        st.text("~0")

        st.form_submit_button("Calculate Edges")

    if edges:
        st.markdown("#### Detected Edges")
        edge_df = pd.DataFrame(edges).drop(columns=["edge_val"])
        positive_edges = [e for e in edges if e["edge_val"] > 0.02]
        if positive_edges:
            st.success(f"Found {len(positive_edges)} positive edges!")
        st.dataframe(edge_df, use_container_width=True, hide_index=True)


def _american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds == 0:
        return 0.0
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)
