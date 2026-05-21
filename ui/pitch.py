# ui/pitch.py
#
# Plotly pitch diagram for the Team Builder.

from __future__ import annotations
import plotly.graph_objects as go


def _grade_color(grade: float | None) -> str:
    if grade is None:
        return "#607D8B"   # slate-grey — not yet fetched
    if grade >= 70:
        return "#00C853"   # bright green
    if grade >= 60:
        return "#27AE60"   # green
    if grade >= 50:
        return "#F39C12"   # amber
    if grade >= 40:
        return "#E67E22"   # orange
    return "#E74C3C"       # red


def _pitch_shapes() -> list[dict]:
    """Return white Plotly shape dicts for standard pitch markings.

    Coordinate system: x 0→1 (left→right), y 0→1 (GK goal→attacking goal).
    Proportions approximate a 105m × 68m pitch displayed in portrait.
    """
    W = 1.0   # pitch width (normalised)
    H = 1.0   # pitch height (normalised)

    # Penalty area depth  ≈ 16.5 m / 105 m ≈ 0.157
    PA_DEPTH = 0.158
    # Penalty area width  ≈ 40.32 m / 68 m  ≈ 0.593 → side margins ≈ 0.204
    PA_MARGIN = 0.185
    # Goal area depth  ≈ 5.5 m / 105 m ≈ 0.052
    GA_DEPTH = 0.056
    # Goal area width  ≈ 18.32 m / 68 m ≈ 0.269 → side margins ≈ 0.365
    GA_MARGIN = 0.365

    line = dict(color="rgba(255,255,255,0.80)", width=1.5)

    shapes = [
        # Pitch outline
        dict(type="rect", x0=0, y0=0, x1=W, y1=H,
             line=dict(color="rgba(255,255,255,0.9)", width=2),
             fillcolor="rgba(0,0,0,0)", layer="below"),
        # Centre line
        dict(type="line", x0=0, y0=0.5, x1=W, y1=0.5, line=line),
        # Centre circle (ellipse to match normalised aspect ratio)
        dict(type="circle",
             x0=0.5 - 0.115, y0=0.5 - 0.078,
             x1=0.5 + 0.115, y1=0.5 + 0.078,
             line=line, fillcolor="rgba(0,0,0,0)"),
        # Centre spot
        dict(type="circle",
             x0=0.5 - 0.006, y0=0.5 - 0.004,
             x1=0.5 + 0.006, y1=0.5 + 0.004,
             line=dict(color="white", width=1), fillcolor="white"),
        # Bottom (GK) penalty area
        dict(type="rect",
             x0=PA_MARGIN, y0=0, x1=W - PA_MARGIN, y1=PA_DEPTH,
             line=line, fillcolor="rgba(0,0,0,0)"),
        # Bottom goal area
        dict(type="rect",
             x0=GA_MARGIN, y0=0, x1=W - GA_MARGIN, y1=GA_DEPTH,
             line=line, fillcolor="rgba(0,0,0,0)"),
        # Bottom penalty spot
        dict(type="circle",
             x0=0.5 - 0.005, y0=0.115 - 0.004,
             x1=0.5 + 0.005, y1=0.115 + 0.004,
             line=dict(color="white", width=1), fillcolor="white"),
        # Top (attacking) penalty area
        dict(type="rect",
             x0=PA_MARGIN, y0=H - PA_DEPTH, x1=W - PA_MARGIN, y1=H,
             line=line, fillcolor="rgba(0,0,0,0)"),
        # Top goal area
        dict(type="rect",
             x0=GA_MARGIN, y0=H - GA_DEPTH, x1=W - GA_MARGIN, y1=H,
             line=line, fillcolor="rgba(0,0,0,0)"),
        # Top penalty spot
        dict(type="circle",
             x0=0.5 - 0.005, y0=H - 0.115 - 0.004,
             x1=0.5 + 0.005, y1=H - 0.115 + 0.004,
             line=dict(color="white", width=1), fillcolor="white"),
        # Bottom goal (GK end) — sticks out below the pitch line
        dict(type="rect",
             x0=0.5 - 0.054, y0=-0.028, x1=0.5 + 0.054, y1=0,
             line=dict(color="rgba(255,255,255,0.9)", width=2),
             fillcolor="rgba(255,255,255,0.08)"),
        # Top goal (attacking end) — sticks out above the pitch line
        dict(type="rect",
             x0=0.5 - 0.054, y0=H, x1=0.5 + 0.054, y1=H + 0.028,
             line=dict(color="rgba(255,255,255,0.9)", width=2),
             fillcolor="rgba(255,255,255,0.08)"),
    ]
    return shapes


def create_pitch_figure(
    slots: list,
    results: list,
    formation_name: str = "",
    height: int = 600,
) -> go.Figure:
    """Build a Plotly pitch diagram with players positioned by slot.

    Args:
        slots:          list[Slot] from tools.team_builder
        results:        list[SlotResult] aligned with slots
        formation_name: string shown in the figure title
        height:         pixel height of the figure
    """
    result_by_id = {r.slot.id: r for r in results}

    xs, ys, marker_colors, hover_texts, display_texts = [], [], [], [], []

    for slot in slots:
        r = result_by_id.get(slot.id)
        grade = r.grade if r else None
        name = (r.found_name or r.query) if r else ""
        error = r.error if r else ""

        name_short = name.split()[-1] if name else "?"
        grade_str = f"{grade:.0f}" if grade is not None else ("?" if not error else "⚠")

        display_texts.append(
            f"<b>{slot.label}</b><br>"
            f"<span style='font-size:9px'>{name_short}</span><br>"
            f"<b style='font-size:11px'>{grade_str}</b>"
        )

        hover_line = (
            f"<b>{slot.label}</b> — {name or '(empty)'}<br>"
            f"Grade: <b>{grade_str}/80</b><br>"
            f"Role: {slot.role}"
        )
        if error:
            hover_line += f"<br>⚠️ {error}"
        hover_texts.append(hover_line)

        xs.append(slot.x)
        ys.append(slot.y)
        marker_colors.append(_grade_color(grade))

    fig = go.Figure()

    # Grass stripe background (alternating light/dark green bands)
    stripe_count = 8
    for i in range(stripe_count):
        fig.add_shape(
            type="rect",
            x0=0, x1=1,
            y0=i / stripe_count, y1=(i + 1) / stripe_count,
            fillcolor="#2E7D32" if i % 2 == 0 else "#388E3C",
            line_width=0,
            layer="below",
        )

    # Player markers
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        marker=dict(
            size=48,
            color=marker_colors,
            line=dict(color="white", width=2),
            symbol="circle",
        ),
        text=display_texts,
        textposition="middle center",
        textfont=dict(color="white", size=9.5, family="Arial"),
        hovertext=hover_texts,
        hoverinfo="text",
        hoverlabel=dict(
            bgcolor="#0D1B2A",
            font_color="white",
            bordercolor="#444",
        ),
    ))

    fig.update_layout(
        title=dict(
            text=formation_name or "⚽ Team Formation",
            font=dict(color="white", size=15, family="Arial"),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#2E7D32",
        xaxis=dict(
            range=[-0.04, 1.04],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
        ),
        yaxis=dict(
            range=[-0.04, 1.04],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
        ),
        height=height,
        margin=dict(l=8, r=8, t=44, b=8),
        shapes=_pitch_shapes(),
        showlegend=False,
        dragmode=False,
    )

    return fig
