"""Plotly-Abbildungen der Novelty-Search-Demo: Lauf im Labyrinth (Endpositionen, besuchte Zellen, Archiv), Verlauf des Abstands und der besuchten Zellen, Erfolgsquoten je Variante, Budgetkurven,
Einstellungs-Raster, verlängerter Lauf. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen. Das Raster `grid[i, j]` wird mit i als x-Achse gezeichnet."""

import numpy as np
import plotly.graph_objects as go

import ns_constants as C

WALL_COLOR = "#3a3f47"
VISITED_COLOR = "#d6e4f5"
START_COLOR, GOAL_COLOR, ROBOT_COLOR, PATH_COLOR, ARCHIVE_COLOR, TRAP_COLOR = "#54a24b", "#e45756", "#4c78a8", "#f58518", "#b279a2", "#b279a2"
VARIANT_COLORS = {"fitness": "#e45756", "novelty": "#4c78a8", "archive_top": "#54a24b", "archive_threshold": "#b279a2"}
LEVEL_COLORS = {0: "#54a24b", 1: "#72b7b2", 2: "#4c78a8", 3: "#f58518", 4: "#e45756"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25, top=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _arena_axes(fig):
    """Beide Achsen 0 bis N mit gleichem Maßstab (quadratische Zellen), Zeichenfläche passt sich der kleineren Seite an (constrain=domain)."""
    fig.update_xaxes(range=[0, C.N], showgrid=False, zeroline=False, title_text="x (Zellen)", constrain="domain")
    fig.update_yaxes(range=[0, C.N], showgrid=False, zeroline=False, title_text="y (Zellen)", scaleanchor="x", scaleratio=1, constrain="domain")


def _markers(fig):
    fig.add_trace(go.Scatter(x=[C.START[0]], y=[C.START[1]], mode="markers", marker=dict(color=START_COLOR, size=13, symbol="square", line=dict(color="white", width=1.5)), name="Start"))
    fig.add_trace(go.Scatter(x=[C.GOAL[0]], y=[C.GOAL[1]], mode="markers", marker=dict(color=GOAL_COLOR, size=16, symbol="star", line=dict(color="white", width=1.5)), name="Ziel"))
    fig.add_shape(type="circle", xref="x", yref="y", x0=C.GOAL[0] - C.SUCCESS_RADIUS, x1=C.GOAL[0] + C.SUCCESS_RADIUS, y0=C.GOAL[1] - C.SUCCESS_RADIUS, y1=C.GOAL[1] + C.SUCCESS_RADIUS,
                  line=dict(color=GOAL_COLOR, dash="dot", width=1.5))


def build_run_figure(grid, end_positions, track, visited, archive=None):
    """Ein Lauf: Wände (dunkel), alle besuchten Zellen (hellblau), Endpositionen der letzten Generation (blau), Archiv (violett, falls vorhanden) und die beste Fahrt des Laufs (orange)."""
    z = np.where(grid, 2.0, np.where(visited, 1.0, np.nan))
    fig = go.Figure(go.Heatmap(z=z.T, x0=0.5, dx=1, y0=0.5, dy=1, zmin=1, zmax=2, colorscale=[[0, VISITED_COLOR], [0.5, VISITED_COLOR], [0.5, WALL_COLOR], [1, WALL_COLOR]], showscale=False, hoverinfo="skip",
                               xgap=0, ygap=0))
    if archive is not None and len(archive):
        fig.add_trace(go.Scatter(x=archive[:, 0], y=archive[:, 1], mode="markers", marker=dict(color=ARCHIVE_COLOR, size=3, opacity=0.45), name=f"Archiv ({C.fmt_int(len(archive))} Punkte)"))
    fig.add_trace(go.Scatter(x=end_positions[:, 0], y=end_positions[:, 1], mode="markers", marker=dict(color=ROBOT_COLOR, size=6, opacity=0.7), name="Endpositionen der letzten Generation"))
    fig.add_trace(go.Scatter(x=track[:, 0], y=track[:, 1], mode="lines", line=dict(color=PATH_COLOR, width=2.5), name="beste Fahrt"))
    _markers(fig)
    _arena_axes(fig)
    return _base(fig, 420, legend_y=-0.3, top=0)


def build_progress_chart(fit_best, nov_best, variant, trap):
    """Kleinster Abstand zum Ziel je Generation, Fitness gegen Novelty; gestrichelt: Zielkreis und (wenn vorhanden) die Tiefe der tiefsten Falle."""
    gens = list(range(1, len(fit_best) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=fit_best, mode="lines", line=dict(color=VARIANT_COLORS["fitness"], width=2.5), name=C.VARIANT_LABELS["fitness"]))
    fig.add_trace(go.Scatter(x=gens, y=nov_best, mode="lines", line=dict(color=VARIANT_COLORS[variant], width=2.5), name=C.VARIANT_LABELS[variant]))
    fig.add_hline(y=C.SUCCESS_RADIUS, line=dict(color=GOAL_COLOR, dash="dot"), annotation_text="Zielkreis")
    if trap is not None:
        fig.add_hline(y=trap, line=dict(color=TRAP_COLOR, dash="dash"), annotation_text="tiefste Falle")
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="bester Abstand zum Ziel (Zellen)", rangemode="tozero")
    return _base(fig, 320, legend_y=-0.35)


def build_coverage_chart(fit_visited, nov_visited, variant, free):
    """Anteil der freien Zellen, die die Suche bis zur jeweiligen Generation besucht hat (Endpositionen), Fitness gegen Novelty."""
    gens = list(range(1, len(fit_visited) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=[v / free for v in fit_visited], mode="lines", line=dict(color=VARIANT_COLORS["fitness"], width=2.5), name=C.VARIANT_LABELS["fitness"]))
    fig.add_trace(go.Scatter(x=gens, y=[v / free for v in nov_visited], mode="lines", line=dict(color=VARIANT_COLORS[variant], width=2.5), name=C.VARIANT_LABELS[variant]))
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="besuchte Zellen (Anteil der freien)", tickformat=".0%", range=[0, 1.03])
    return _base(fig, 320, legend_y=-0.35)


def build_success_chart(pre):
    """Erfolgsquote der 20 Läufe je Stufe und Variante (innerhalb des vollen Budgets von 40 000 Evaluationen)."""
    n = pre["seeds"]
    fig = go.Figure()
    for v in C.VARIANTS:
        counts = [sum(1 for r in pre["levels"][str(lv)]["variants"][v]["runs"] if r is not None) for lv in C.LEVELS]
        fig.add_trace(go.Bar(x=[C.LEVEL_LABELS[lv] for lv in C.LEVELS], y=[c / n for c in counts], marker_color=VARIANT_COLORS[v], name=C.VARIANT_LABELS[v], text=[str(c) for c in counts], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text=f"Anteil erfolgreicher Läufe (Zahl über dem Balken: von {n})", tickformat=".0%", range=[0, 1.18])
    return _base(fig, 380, legend_y=-0.3)


def build_budget_chart(pre, level):
    """Anteil erfolgreicher Läufe über dem Evaluationsbudget auf der gewählten Stufe, je Variante."""
    budgets = list(range(0, pre["budget"] + 1, 250))
    fig = go.Figure()
    for v in C.VARIANTS:
        runs = pre["levels"][str(level)]["variants"][v]["runs"]
        ys = [sum(1 for r in runs if r is not None and r <= b) / len(runs) for b in budgets]
        fig.add_trace(go.Scatter(x=budgets, y=ys, mode="lines", line=dict(color=VARIANT_COLORS[v], width=2.5), name=C.VARIANT_LABELS[v]))
    fig.update_xaxes(title_text="Evaluationen")
    fig.update_yaxes(title_text="Anteil erfolgreicher Läufe", tickformat=".0%", range=[-0.02, 1.05])
    return _base(fig, 340, legend_y=-0.35)


def build_study_coverage_chart(pre, level):
    """Mittlere Zahl besuchter Zellen (Anteil der freien Zellen) über den Generationen, 20 Läufe je Variante."""
    free = pre["levels"][str(level)]["metrics"]["free"]
    fig = go.Figure()
    for v in C.VARIANTS:
        curve = pre["levels"][str(level)]["variants"][v]["mean_visited"]
        fig.add_trace(go.Scatter(x=list(range(1, len(curve) + 1)), y=[c / free for c in curve], mode="lines", line=dict(color=VARIANT_COLORS[v], width=2.5), name=C.VARIANT_LABELS[v]))
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="besuchte Zellen (Anteil der freien)", tickformat=".0%", range=[0, 1.03])
    return _base(fig, 340, legend_y=-0.35)


def build_sweep_chart(pre, level):
    """Zahl erfolgreicher Läufe (von 20) für k und Populationsgröße der Variante „Novelty, nur Population“ bei gleichem Gesamtbudget."""
    ks, pops = list(C.SWEEP_KS), list(C.SWEEP_POPS)
    z = [[sum(1 for r in next(x for x in pre["sweep"] if x["level"] == level and x["k"] == k and x["pop"] == p)["runs"] if r is not None) for k in ks] for p in pops]
    fig = go.Figure(go.Heatmap(z=z, x=[f"k = {k}" for k in ks], y=[f"{p} Individuen" for p in pops], text=[[f"{v} von {pre['seeds']}" for v in row] for row in z], texttemplate="%{text}",
                               colorscale="Blues", zmin=0, zmax=pre["seeds"], showscale=False))
    fig.update_xaxes(title_text="Zahl der Nachbarn k")
    return _base(fig, 260)


def build_long_chart(pre):
    """Kleinster wahrer Restweg (Zellen) über 400 000 Evaluationen auf Stufe 4, Mittel über 20 Läufe je Variante; senkrecht: das Budget der Studie."""
    fig = go.Figure()
    for v in C.LONG_VARIANTS:
        curve = pre["long"]["variants"][v]["mean_progress"]
        fig.add_trace(go.Scatter(x=[(i + 1) * C.LONG_STEP * C.POP for i in range(len(curve))], y=curve, mode="lines", line=dict(color=VARIANT_COLORS[v], width=2.5), name=C.VARIANT_LABELS[v]))
    fig.add_vline(x=pre["budget"], line=dict(color="#888", dash="dot"), annotation_text="Budget der Studie")
    fig.update_xaxes(title_text="Evaluationen", type="log")
    fig.update_yaxes(title_text="kleinster wahrer Restweg bis zum Ziel (Zellen)", rangemode="tozero")
    return _base(fig, 340, legend_y=-0.35)
