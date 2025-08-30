# ui/graphs.py

import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Import the necessary data structures from grading.py
from tools.grading import (
    DEFAULT_WEIGHTS,
    SUBROLE_WEIGHTS,
    ROLE_PRETTY,
    _normalize_role,
    _match_metric_name,
    _invert_if_negative
)

def _get_role_metrics(role_hint: str) -> List[str]:
    """
    Returns a list of metrics relevant to a given role from the grading weights.
    """
    base, sub = _normalize_role(role_hint)
    
    metrics = list(DEFAULT_WEIGHTS.get(base, {}).keys())
    if sub and sub in SUBROLE_WEIGHTS:
        metrics.extend(list(SUBROLE_WEIGHTS[sub].keys()))

    return list(dict.fromkeys(metrics))

def create_spider_graph(
    player_data: pd.DataFrame,
    player_name: str,
    role_hint: str,
    language: str,
    threshold: float = 75.0
) -> go.Figure:
    """
    Creates an interactive spider graph for a player based on their position.
    
    Args:
        player_data: A DataFrame with 'Metric' as index and 'Percentile' as a column.
        player_name: The name of the player.
        role_hint: The player's inferred position (e.g., "fw:w", "df:cb").
        language: The selected language ('English' or 'Français').
        threshold: The percentile value for the static threshold line (e.g., 50.0 or 75.0).
    """
    
    # ------------------ Configuration & Data Prep ------------------
    metrics_to_plot = _get_role_metrics(role_hint)
    
    if language.lower().startswith("fr"):
        plot_title = f"{player_name}"
        axis_title = "Percentile"
        threshold_name = f"Seuil {int(threshold)}%"
    else:
        plot_title = f"{player_name}"
        axis_title = "Percentile"
        threshold_name = f"{int(threshold)}th Percentile"

    if "Percentile" in player_data.columns:
        player_data["Percentile"] = pd.to_numeric(player_data["Percentile"], errors="coerce")

    player_series = pd.Series(index=metrics_to_plot, dtype='float64')

    for metric in metrics_to_plot:
        actual_metric_name = _match_metric_name(list(player_data.index), metric)
        if actual_metric_name and actual_metric_name in player_data.index:
            pct = player_data.loc[actual_metric_name, "Percentile"]
            if not pd.isna(pct):
                inverted_pct = _invert_if_negative(actual_metric_name, pct)
                player_series[metric] = inverted_pct

    player_series = player_series.dropna()
    
    player_metrics = player_series.index.tolist()
    player_values = player_series.values.tolist()

    # ------------------ Create the Plot ------------------
    fig = go.Figure()

    # Add player's trace
    fig.add_trace(go.Scatterpolar(
        r=player_values,
        theta=player_metrics,
        fill="toself",
        name=player_name,
        line_color="#45c264",
        mode="lines+markers"
    ))

    # Add the static percentile line.
    # We now use the same filtered metrics list to ensure alignment.
    fig.add_trace(go.Scatterpolar(
        r=[threshold] * len(player_metrics),
        theta=player_metrics,
        mode="lines",
        name=threshold_name,
        line=dict(color="red", width=2, dash="dash")
    ))

    # Get a pretty role name for the subtitle
    base, sub = _normalize_role(role_hint)
    pretty_role = ROLE_PRETTY.get(f"{base}:{sub}" if sub else base, role_hint.upper())
    
    fig.update_layout(
        title=dict(text=plot_title, x=0.5, font=dict(size=20)),
        title_xanchor="center",
        title_yanchor="top",
        annotations=[
            dict(
                text=f"Role: {pretty_role}",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=1.0,
                xanchor="center", yanchor="bottom",
                font=dict(size=10)
            )
        ],
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                title=dict(text=axis_title, font=dict(size=12)),
                tickmode="linear",
                dtick=25
            )
        ),
        showlegend=True
    )
    
    return fig