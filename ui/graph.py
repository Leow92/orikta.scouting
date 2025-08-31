# ui/graphs.py

import plotly.graph_objects as go
import pandas as pd
from typing import List
import numpy as np

from tools.grading import (
    DEFAULT_WEIGHTS,
    SUBROLE_WEIGHTS,
    ROLE_PRETTY,
    _normalize_role,
    _match_metric_name,
    _invert_if_negative
)

# ------------------ Color & UX Presets ------------------
# Okabe–Ito palette (color-blind–friendly)
PALETTE = {
    "orange":   "#E69F00",
    "sky":      "#56B4E9",
    "green":    "#009E73",
    "yellow":   "#F0E442",
    "blue":     "#0072B2",
    "vermillion":"#D55E00",
    "purple":   "#CC79A7",
    "black":    "#000000"
}

# Dark theme defaults
THEME = {
    "player_line": PALETTE["sky"],
    "player_fill": "rgba(86,180,233,0.25)",   # translucent sky blue
    "player_marker_line": "#1F2937",          # dark gray outline for markers
    "threshold_line": PALETTE["vermillion"],
    "axis_grid": "#374151",                   # muted gray grid
    "axis_tick": "#D1D5DB",                   # light gray text
    "font": "#F9FAFB",                        # almost white text
    "polar_bg": "#111827",                    # near-black panel
    "paper_bg": "#111827",                    # same as polar for seamless look
    "legend_bg": "rgba(17,24,39,0.8)"         # semi-transparent dark box
}

def _get_role_metrics(role_hint: str) -> List[str]:
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
    metrics_to_plot = _get_role_metrics(role_hint)

    if language.lower().startswith("fr"):
        plot_title = f"{player_name} Percentiles (365 derniers jours) {role_hint}"
        axis_title = "Percentile"
        threshold_name = f"{int(threshold)}% des joueurs"
        #role_label = "Rôle"
    else:
        plot_title = f"{player_name} Percentiles (Last 365 days) {role_hint}"
        axis_title = "Percentile"
        threshold_name = f"{int(threshold)}% of players"
        #role_label = "Role"

    if "Percentile" in player_data.columns:
        player_data["Percentile"] = pd.to_numeric(player_data["Percentile"], errors="coerce")

    player_series = pd.Series(index=metrics_to_plot, dtype="float64")

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

    # ✅ Close the polygon by repeating the first point
    player_metrics_closed = player_metrics + [player_metrics[0]]
    player_values_closed = player_values + [player_values[0]]

    # ------------------ Create the Plot ------------------
    fig = go.Figure()

    # Player polygon
    fig.add_trace(go.Scatterpolar(
        r=player_values_closed,
        theta=player_metrics_closed,
        fill="toself",
        name=player_name,
        line=dict(color=THEME["player_line"], width=3),
        fillcolor=THEME["player_fill"],
        mode="lines+markers",
        marker=dict(
            size=7,
            symbol="circle",
            color=THEME["player_line"],
            line=dict(color=THEME["player_marker_line"], width=1.6)
        ),
        hovertemplate="<b>%{theta}</b><br>%{r:.1f} " + axis_title + "<extra></extra>"
    ))

    # Threshold polygon (dashed line, also closed)
    fig.add_trace(go.Scatterpolar(
        r=[threshold] * len(player_metrics) + [threshold],
        theta=player_metrics_closed,
        mode="lines",
        name=threshold_name,
        line=dict(color=THEME["threshold_line"], width=2.5, dash="dash"),
        hoverinfo="skip"
    ))

    base, sub = _normalize_role(role_hint)
    pretty_role = ROLE_PRETTY.get(f"{base}:{sub}" if sub else base, role_hint.upper())

    # ------------------ Layout ------------------
    fig.update_layout(
        title=dict(
            text=plot_title,
            font=dict(size=18, color=THEME["font"])
        ),
        title_x=0.5,
        polar=dict(
            bgcolor=THEME["polar_bg"],
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                title=dict(text=axis_title, font=dict(size=10, color=THEME["axis_tick"])),
                tickmode="linear",
                dtick=25,
                gridcolor=THEME["axis_grid"],
                gridwidth=1,
                tickfont=dict(color=THEME["axis_tick"])
            ),
            angularaxis=dict(
                gridcolor=THEME["axis_grid"],
                gridwidth=1,
                tickfont=dict(color=THEME["font"])
            )
        ),
        paper_bgcolor=THEME["paper_bg"],
        font=dict(color=THEME["font"]),
        legend=dict(
            bgcolor=THEME["legend_bg"],
            bordercolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="top",
            y=-0.22,   # ✅ pushed further down for more gap
            xanchor="center",
            x=0.5,
            font=dict(size=12, color=THEME["font"])
        ),
        margin=dict(l=40, r=40, t=70, b=120),  # ✅ extra bottom margin
        hoverlabel=dict(
            bgcolor="#1F2937",
            font_size=12,
            font_color="#F9FAFB",
            namelength=-1
        ),
        showlegend=True
    )

    return fig

def create_spider_graph_duo(
    playerA_data: pd.DataFrame,
    playerB_data: pd.DataFrame,
    playerA_name: str,
    playerB_name: str,
    role_hint: str,
    language: str,
    threshold: float = 75.0,
    show_threshold: bool = True
) -> go.Figure:
    """
    Two-player spider chart on the same axes.
    - Uses the role_hint to pick the metric set (same as single-player).
    - Only plots metrics that exist for BOTH players (to keep polygons aligned).
    """
    metrics_all = _get_role_metrics(role_hint)

    # i18n bits
    if language.lower().startswith("fr"):
        plot_title = f"{playerA_name} vs {playerB_name} — Percentiles (365 jours) {role_hint}"
        axis_title = "Percentile"
        threshold_name = f"{int(threshold)}% des joueurs"
    else:
        plot_title = f"{playerA_name} vs {playerB_name} — Percentiles (Last 365 days) {role_hint}"
        axis_title = "Percentile"
        threshold_name = f"{int(threshold)}% of players"

    # ensure numeric
    for df in (playerA_data, playerB_data):
        if "Percentile" in df.columns:
            df["Percentile"] = pd.to_numeric(df["Percentile"], errors="coerce")

    # build aligned series (only metrics present for both)
    labels, valsA, valsB = [], [], []
    for metric in metrics_all:
        a_name = _match_metric_name(list(playerA_data.index), metric)
        b_name = _match_metric_name(list(playerB_data.index), metric)
        if not a_name or not b_name:
            continue
        a_pct = playerA_data.loc[a_name, "Percentile"] if a_name in playerA_data.index else np.nan
        b_pct = playerB_data.loc[b_name, "Percentile"] if b_name in playerB_data.index else np.nan
        if pd.isna(a_pct) or pd.isna(b_pct):
            continue
        a_val = _invert_if_negative(a_name, float(a_pct))
        b_val = _invert_if_negative(b_name, float(b_pct))
        labels.append(metric)
        valsA.append(a_val)
        valsB.append(b_val)

    if not labels:
        # fallback: avoid crash if nothing overlaps (shouldn't happen after your table alignment)
        labels = ["No common metrics"]
        valsA = [0.0]
        valsB = [0.0]

    # close polygons
    labels_c = labels + [labels[0]]
    valsA_c  = valsA  + [valsA[0]]
    valsB_c  = valsB  + [valsB[0]]

    # colors for A/B (Okabe–Ito palette, same vibe as single)
    A_line = PALETTE["sky"]       # blue
    A_fill = "rgba(86,180,233,0.22)"
    B_line = PALETTE["orange"]    # orange
    B_fill = "rgba(230,159,0,0.22)"

    fig = go.Figure()

    # A
    fig.add_trace(go.Scatterpolar(
        r=valsA_c,
        theta=labels_c,
        fill="toself",
        name=playerA_name,
        line=dict(color=A_line, width=3),
        fillcolor=A_fill,
        mode="lines+markers",
        marker=dict(size=7, symbol="circle", color=A_line,
                    line=dict(color=THEME["player_marker_line"], width=1.6)),
        hovertemplate=f"<b>%{{theta}}</b><br>%{{r:.1f}} {axis_title}<extra>{playerA_name}</extra>"
    ))

    # B
    fig.add_trace(go.Scatterpolar(
        r=valsB_c,
        theta=labels_c,
        fill="toself",
        name=playerB_name,
        line=dict(color=B_line, width=3),
        fillcolor=B_fill,
        mode="lines+markers",
        marker=dict(size=7, symbol="circle", color=B_line,
                    line=dict(color=THEME["player_marker_line"], width=1.6)),
        hovertemplate=f"<b>%{{theta}}</b><br>%{{r:.1f}} {axis_title}<extra>{playerB_name}</extra>"
    ))

    # Optional threshold ring
    if show_threshold:
        fig.add_trace(go.Scatterpolar(
            r=[threshold] * len(labels_c),
            theta=labels_c,
            mode="lines",
            name=threshold_name,
            line=dict(color=THEME["threshold_line"], width=2.5, dash="dash"),
            hoverinfo="skip"
        ))

    # layout
    fig.update_layout(
        title=dict(text=plot_title, font=dict(size=18, color=THEME["font"])),
        title_x=0.5,
        polar=dict(
            bgcolor=THEME["polar_bg"],
            radialaxis=dict(visible=True, range=[0, 100],
                            title=dict(text=axis_title, font=dict(size=10, color=THEME["axis_tick"])),
                            tickmode="linear", dtick=25,
                            gridcolor=THEME["axis_grid"], gridwidth=1,
                            tickfont=dict(color=THEME["axis_tick"])),
            angularaxis=dict(gridcolor=THEME["axis_grid"], gridwidth=1,
                             tickfont=dict(color=THEME["font"]))
        ),
        paper_bgcolor=THEME["paper_bg"],
        font=dict(color=THEME["font"]),
        legend=dict(
            bgcolor=THEME["legend_bg"],
            bordercolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="top", y=-0.22,
            xanchor="center", x=0.5,
            font=dict(size=12, color=THEME["font"])
        ),
        margin=dict(l=40, r=40, t=70, b=120),
        hoverlabel=dict(bgcolor="#1F2937", font_size=12, font_color="#F9FAFB", namelength=-1),
        showlegend=True
    )
    return fig
