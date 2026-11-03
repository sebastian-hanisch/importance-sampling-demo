"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import imp_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_estimates(a, truth):
    """Verteilung der Schätzungen aus REPS Wiederholungen (logarithmische Achse); gestrichelt die Wahrheit. Nullschätzungen erscheinen nicht, ihr Anteil steht in der Beschriftung."""
    vals = [x for x in a.est if x > 0]
    fig = go.Figure()
    fig.add_trace(go.Box(y=vals, name="Importance Sampling", marker_color=C.COLOR_IS, boxmean="sd", boxpoints="outliers"))
    fig.add_hline(y=truth, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="Wahrheit", annotation_position="top right")
    fig.update_yaxes(title_text="Schätzung der Ausfallwahrscheinlichkeit", type="log")
    fig = _base(fig, 340)
    fig.update_layout(showlegend=False)
    return fig


def build_qp_curve(exp, qp_now):
    rows = exp["rows"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["qp"] for r in rows], y=[r["cv2"] for r in rows], mode="lines", name="CV² (exakt)", line=dict(color=C.COLOR_IS)))
    fig.add_hline(y=exp["cv2_plain"], line=dict(color=C.COLOR_PLAIN, dash="dash"), annotation_text="einfaches Ziehen", annotation_position="top right")
    if exp["qp_opt"] == exp["qp_opt"]:
        fig.add_trace(go.Scatter(x=[exp["qp_opt"]], y=[exp["cv2_opt"]], mode="markers", name="Optimum", marker=dict(size=12, color=C.COLOR_ACCENT)))
    fig.add_vline(x=qp_now, line=dict(color=C.COLOR_OTHER, dash="dot"), annotation_text="Regler q'", annotation_position="bottom right")
    fig.update_xaxes(title_text="Kippung q' (Ausfallwahrscheinlichkeit je Kante in der Hilfsverteilung)", type="log")
    fig.update_yaxes(title_text="relatives Varianzquadrat CV² je Ziehung", type="log")
    return _base(fig, 360)


def build_rarity(exp):
    rows = exp["rows"]
    fig = go.Figure()
    xs = [str(r["q"]).replace(".", ",") for r in rows]
    fig.add_trace(go.Bar(x=xs, y=[r["n_plain"] for r in rows], name="einfaches Ziehen", marker_color=C.COLOR_PLAIN))
    fig.add_trace(go.Bar(x=xs, y=[r["n_opt"] for r in rows], name="optimale Kippung", marker_color=C.COLOR_IS))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Ausfallwahrscheinlichkeit q je Kante", type="category")
    fig.update_yaxes(title_text="Ziehungen für 10 % relativen Fehler", type="log")
    return _base(fig, 330)


def build_sweep(exp):
    rows = exp["rows"]
    fig = go.Figure()
    xs = [f"{r['qp']:.3f}".replace(".", ",") for r in rows]
    fig.add_trace(go.Bar(x=xs, y=[r["med_rel"] for r in rows], name="mediane Abweichung", marker_color=C.COLOR_IS))
    fig.add_trace(go.Scatter(x=xs, y=[r["cover"] for r in rows], mode="markers+lines", name="Überdeckung des 95-%-Intervalls", yaxis="y2", line=dict(color=C.COLOR_ACCENT)))
    fig.add_trace(go.Scatter(x=xs, y=[r["zero"] for r in rows], mode="markers+lines", name="Anteil Läufe ohne Ausfall", yaxis="y2", line=dict(color=C.COLOR_BAD, dash="dot")))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 1.02], title="Überdeckung / Anteil", fixedrange=True))
    fig.update_xaxes(title_text="Kippung q' (erste Spalte: einfaches Ziehen)", type="category")
    fig.update_yaxes(title_text="mediane relative Abweichung", range=[0, 1.05])
    return _base(fig, 360)


def build_continuous(exp):
    rows = exp["rows"]
    fig = go.Figure()
    xs = [str(r["theta"]).replace(".", ",") for r in rows]
    fig.add_trace(go.Scatter(x=xs, y=[r["k_med"] for r in rows], mode="markers+lines", name="Pareto-k̂ (Median)", line=dict(color=C.COLOR_IS)))
    fig.add_trace(go.Scatter(x=xs, y=[r["k_exact"] for r in rows], mode="lines", name="exaktes k = (θ−1)/θ", line=dict(color=C.COLOR_TRUTH, dash="dash")))
    fig.add_trace(go.Scatter(x=xs, y=[r["cover"] for r in rows], mode="markers+lines", name="Überdeckung des 95-%-Intervalls", line=dict(color=C.COLOR_ACCENT, dash="dot")))
    fig.add_hline(y=C.K_FINITE_VARIANCE, line=dict(color=C.COLOR_BAD, dash="dot"), annotation_text="k = 1/2: ab hier unendliche Varianz", annotation_position="top left")
    fig.update_xaxes(title_text="θ (Rate der Hilfsverteilung; das Ziel hat Rate 1)", type="category")
    fig.update_yaxes(title_text="k̂ und Überdeckung", range=[-1.1, 1.1])
    return _base(fig, 360)


def build_netz(netz):
    fig = go.Figure()
    xs, ys = [c[0] for c in netz.coords], [c[1] for c in netz.coords]
    ex, ey = [], []
    for u, v in netz.edges:
        ex += [xs[u], xs[v], None]
        ey += [ys[u], ys[v], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#666", width=3), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=22, color="white", line=dict(color="#222", width=2)), showlegend=False, hoverinfo="skip"))
    for i, (x, y) in enumerate(zip(xs, ys)):
        fig.add_annotation(x=x, y=y, text=str(i), showarrow=False, font=dict(size=10, color="#222"))
    pad = 0.5
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad])
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
    fig = _base(fig, 280)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    return fig
