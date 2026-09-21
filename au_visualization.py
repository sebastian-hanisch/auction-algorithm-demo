"""Plotly-Abbildungen des Auktionsalgorithmus: Karte mit Preisen und Geboten, Fortschritt, Preiskrieg (Gebote gegen eps), Phasen, Verteilung, Sweeps, Strafe.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen. Punkte auf einer Geraden werden mit Bögen gezeichnet."""

import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import au_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.08), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _collinear(sc):
    """Liegen alle Punkte auf einer Geraden? Dann würden sich die Paar-Linien überdecken - sie werden gebogen gezeichnet."""
    pts = list(sc.vehicles + sc.orders)
    (x0, y0), (x1, y1) = pts[0], next((p for p in pts if p != pts[0]), pts[0])
    return all((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0) == 0 for x, y in pts)


def _edge_path(sc, i, j, curved, steps=14):
    """Punkte einer Paar-Linie: gerade, oder (bei Punkten auf einer Geraden) als Bogen, dessen Seite je Paar wechselt."""
    (vx, vy), (ox, oy) = sc.vehicles[i], sc.orders[j]
    if not curved:
        return [vx, ox], [vy, oy]
    dx, dy = ox - vx, oy - vy
    side = 1 if (i + j) % 2 == 0 else -1
    cx, cy = (vx + ox) / 2 - side * 0.35 * dy, (vy + oy) / 2 + side * 0.35 * dx      # Kontrollpunkt senkrecht zur Verbindung
    ts = [k / steps for k in range(steps + 1)]
    return ([(1 - t) ** 2 * vx + 2 * (1 - t) * t * cx + t * t * ox for t in ts], [(1 - t) ** 2 * vy + 2 * (1 - t) * t * cy + t * t * oy for t in ts])


def _segments(sc, pairs, curved=False):
    """Linienspur für eine Menge von Paaren (None trennt die Segmente)."""
    x, y = [], []
    for i, j in pairs:
        px, py = _edge_path(sc, i, j, curved)
        x += px + [None]
        y += py + [None]
    return x, y


def _feasible_pairs(sc):
    return [(i, j) for i in range(sc.n) for j in range(sc.m) if sc.feasible[i, j]]


def _map_layout(fig, sc, height):
    xs = [p[0] for p in sc.vehicles + sc.orders]
    ys = [p[1] for p in sc.vehicles + sc.orders]
    pad = 8
    if _collinear(sc):
        # Punkte auf einer Geraden: nur die Bögen brauchen Höhe. Das Seitenverhältnis wird freigegeben, sonst wird eine lange Kette zu einem dünnen Streifen.
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        if max(xs) - min(xs) >= max(ys) - min(ys):
            fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad])
            fig.update_yaxes(visible=False, range=[cy - span * 0.22, cy + span * 0.22])
        else:
            fig.update_xaxes(visible=False, range=[cx - span * 0.22, cx + span * 0.22])
            fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
        return _base(fig, min(height, 320))
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
    return _base(fig, height)


def _scaled(values, vmax, lo, hi):
    return [lo + (hi - lo) * min(v, vmax) / vmax if vmax else lo for v in values]


def _pair_lines(fig, sc, curved, pairs, color, width, name, dash=None):
    x, y = _segments(sc, pairs, curved)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width, dash=dash), hoverinfo="skip", name=name))


def build_auction_map(sc, state, p_max, S, event=None, height=430):
    """Karte im Zustand `state`: Aufträge nach Preis (Größe und Farbe, Minuten), gewählte Paare blau, Verzichter violett umrandet;
    das aktuelle Ereignis hervorgehoben (Gebot grün, Verdrängter rot umringt, Preissenkung mit Übernahme gestrichelt)."""
    curved = _collinear(sc)
    fig = go.Figure()
    ex, ey = _segments(sc, _feasible_pairs(sc), curved)
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(150,150,150,0.3)", width=1), hoverinfo="skip", name="mögliche Paare"))
    assign, owner, p, idle = state["assign"], state["owner"], state["p"], state["idle"]
    pairs = [(i, j) for i, j in enumerate(assign) if j >= 0]
    _pair_lines(fig, sc, curved, pairs, C.COLORS["matched"], 3.5, "zugeordnet")
    ring = []
    if event is not None:
        kind = event[0]
        if kind == "bid":
            _, i, j, _p0, _p1, _v1, _v2, old, won = event
            _pair_lines(fig, sc, curved, [(i, j)], C.COLORS["bid"] if won else "#999", 5, "Gebot" if won else "Gebot (verloren)")
            if old >= 0:
                ring.append(("Fahrzeug", old, C.COLORS["evicted"], "verdrängt"))
        elif kind == "idle":
            ring.append(("Fahrzeug", event[1], C.COLORS["idle"], "verzichtet"))
        elif kind == "lower":
            _, j, _p0, _p1, taker, freed, _was = event
            ring.append(("Auftrag", j, C.COLORS["idle"], "Preis gesenkt"))
            if taker >= 0:
                _pair_lines(fig, sc, curved, [(taker, j)], C.COLORS["idle"], 5, "übernimmt", dash="dash")
    small = sc.n + sc.m <= 24
    vmax = max(p_max, 1)
    for kind, pts, symbol, prefix, tpos in (("Fahrzeug", sc.vehicles, "square", "F", "top center"), ("Auftrag", sc.orders, "circle", "A", "bottom center")):
        if kind == "Fahrzeug":
            groups = (("zugeordnet", [k for k in range(sc.n) if assign[k] >= 0], "#2e7d32", symbol),
                      ("verzichtet", [k for k in range(sc.n) if assign[k] < 0 and idle[k]], C.COLORS["idle"], symbol + "-open"),
                      ("bietet noch", [k for k in range(sc.n) if assign[k] < 0 and not idle[k]], "#555", symbol + "-open"))
            for name, idx, color, sym in groups:
                if idx:
                    fig.add_trace(go.Scatter(x=[pts[k][0] for k in idx], y=[pts[k][1] for k in idx], mode="markers+text" if small else "markers", name=f"Fahrzeug {name}",
                                             text=[f"{prefix}{k + 1}" for k in idx] if small else None, textposition=tpos, hovertext=[f"Fahrzeug {k + 1}: {name}" for k in idx], hoverinfo="text",
                                             marker=dict(symbol=sym, size=12, color=color, line=dict(width=2, color=color))))
        else:
            for on in (True, False):
                idx = [k for k in range(sc.m) if (owner[k] >= 0) == on]
                if not idx:
                    continue
                prices = [p[k] / S for k in idx]
                fig.add_trace(go.Scatter(x=[pts[k][0] for k in idx], y=[pts[k][1] for k in idx], mode="markers+text" if small else "markers", name=f"Auftrag {'vergeben' if on else 'frei'}",
                                         text=[f"{prefix}{k + 1}: {round(p[k] / S)}" for k in idx] if small else None, textposition=tpos,
                                         hovertext=[f"Auftrag {k + 1}: Preis {p[k] / S:.1f} min" + ("" if on else " (frei)") for k in idx], hoverinfo="text",
                                         marker=dict(symbol=symbol if on else symbol + "-open", size=_scaled(prices, vmax, 9, 22), color=prices, colorscale=[[0, "#fdd0a2"], [1, "#7f2704"]], cmin=0, cmax=vmax,
                                                     line=dict(width=2, color="#333"))))
    for kind, k, color, name in ring:
        pt = sc.vehicles[k] if kind == "Fahrzeug" else sc.orders[k]
        fig.add_trace(go.Scatter(x=[pt[0]], y=[pt[1]], mode="markers", name=name, hoverinfo="skip", marker=dict(symbol="circle-open", size=28, color=color, line=dict(width=3, color=color))))
    fig = _map_layout(fig, sc, height)
    fig.update_layout(showlegend=False)          # die Legende würde in schmalen Spalten die Zeichenfläche auf fast null drücken; die Farben stehen in der Erklärung unter der Karte
    return fig


def build_progress(unassigned, price_sum, k, phases, height=260):
    """Über die Ereignisse: Fahrzeuge ohne Auftrag (Treppe) und Summe der Preise [min]; senkrecht der aktuelle Schritt, gestrichelt die Phasenwechsel."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    xs = list(range(len(unassigned)))
    fig.add_trace(go.Scatter(x=xs, y=unassigned, mode="lines", line=dict(color=C.COLORS["evicted"], shape="hv"), name="Fahrzeuge ohne Auftrag"), secondary_y=False)
    fig.add_trace(go.Scatter(x=xs, y=price_sum, mode="lines", line=dict(color=C.COLORS["order"]), name="Summe der Preise [min]"), secondary_y=True)
    for pos in phases:
        fig.add_vline(x=pos, line=dict(color="#999", dash="dot", width=1))
    fig.add_vline(x=k, line=dict(color="#333", width=2))
    fig.update_xaxes(title="Ereignis")
    fig.update_yaxes(title="ohne Auftrag", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title="Preise [min]", secondary_y=True, rangemode="tozero", showgrid=False)
    fig = _base(fig, height)
    fig.update_layout(showlegend=False)          # die Achsentitel sagen, was die beiden Linien sind
    return fig


def build_phases(bids_by_phase, eps_min, height=260):
    """Gebote je eps-Phase (Balken); die Achse nennt die Schrittweite der Phase in Minuten."""
    labels = [f"{e:.3g}" for e in eps_min]
    fig = go.Figure(go.Bar(x=labels, y=list(bids_by_phase), marker_color=C.COLORS["scaled"], opacity=0.75, name="Gebote"))
    fig.update_xaxes(title="Schrittweite ε der Phase [min]", type="category")
    fig.update_yaxes(title="Gebote", rangemode="tozero")
    return _base(fig, height)


def build_war(rows, ref, height=380):
    """Oben: Gebote gegen die feste Schrittweite (logarithmisch), die ε-Skalierung als waagerechte Linie; unten: größte und mittlere Lücke zum Optimum gegen die Schranke n·ε [min]."""
    labels = ["exakt" if r["eps"] == 0 else f"{r['eps']} min" for r in rows]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=("Gebote (Mittel über die Karten)", "Abstand zum Optimum [min]"))
    fig.add_trace(go.Scatter(x=labels, y=[r["bids"] for r in rows], mode="lines+markers", name="fest", line=dict(color=C.COLORS["fixed"])), row=1, col=1)
    if ref is not None:
        fig.add_trace(go.Scatter(x=labels, y=[ref] * len(labels), mode="lines", name="ε-Skalierung", line=dict(color=C.COLORS["scaled"], dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=labels, y=[r["max_gap"] for r in rows], mode="lines+markers", name="größte Lücke", line=dict(color=C.COLORS["evicted"])), row=2, col=1)
    fig.add_trace(go.Scatter(x=labels, y=[r["mean_gap"] for r in rows], mode="lines+markers", name="mittlere Lücke", line=dict(color=C.COLORS["matched"])), row=2, col=1)
    fig.add_trace(go.Scatter(x=labels, y=[r["bound"] for r in rows], mode="lines", name="Schranke n·ε", line=dict(color="#555", dash="dot")), row=2, col=1)
    fig.update_xaxes(title="feste Schrittweite ε", type="category", row=2, col=1)
    fig.update_yaxes(type="log", title="Gebote", row=1, col=1)
    fig.update_yaxes(title="min", rangemode="tozero", row=2, col=1)
    fig = _base(fig, height)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22))
    return fig


def build_bid_hist(bids_by_label, current=None, height=300):
    """Verteilung der Gebote über die Karten (logarithmische Achse): fest gegen ε-Skalierung; die Verteilung des festen Laufs ist stark rechtsschief."""
    colors = [C.COLORS["scaled"], C.COLORS["fixed"]]
    fig = go.Figure()
    for k, (label, bids) in enumerate(bids_by_label.items()):
        fig.add_trace(go.Histogram(x=[math.log10(max(b, 1)) for b in bids], xbins=dict(size=0.2), name=label, marker_color=colors[k % 2], opacity=0.65))
    fig.update_layout(barmode="overlay")
    ticks = [0, 1, 2, 3, 4, 5]
    fig.update_xaxes(title="Gebote je Karte", tickvals=ticks, ticktext=[f"{10 ** t:,}".replace(",", " ") for t in ticks])
    fig.update_yaxes(title="Karten")
    if current is not None:
        fig.add_vline(x=math.log10(max(current, 1)), line=dict(color="#555", dash="dash"), annotation_text="Ihre Ziehung", annotation_position="top")
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.35), height=height + 50)
    return fig


def build_reach_sweep(rows, current=None, height=420):
    """Oben: Gebote (fest, ε-Skalierung); unten: durchsuchte Kanten von Ungarischer Methode, Auktion mit ε-Skalierung und fester Auktion (alles logarithmisch)."""
    x = [r["x"] for r in rows]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Gebote (Mittel)", "Durchsuchte Kanten (Mittel)"))
    fig.add_trace(go.Scatter(x=x, y=[r["fixed_bids"] for r in rows], mode="lines+markers", name="fest ε klein", line=dict(color=C.COLORS["fixed"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[r["scaled_bids"] for r in rows], mode="lines+markers", name="ε-Skalierung", line=dict(color=C.COLORS["scaled"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[r["hung_scans"] for r in rows], mode="lines+markers", name="Ungarische Methode", line=dict(color=C.COLORS["hungarian"])), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=[r["scaled_scans"] for r in rows], mode="lines+markers", name="Auktion, ε-Skalierung", line=dict(color=C.COLORS["scaled"]), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=[r["fixed_scans"] for r in rows], mode="lines+markers", name="Auktion, fest", line=dict(color=C.COLORS["fixed"]), showlegend=False), row=2, col=1)
    if current is not None and min(x) <= current <= max(x):
        fig.add_vline(x=current, line=dict(color="#555", dash="dot"))
    fig.update_xaxes(title="Reichweite [min]", row=2, col=1)
    fig.update_yaxes(title="Gebote", type="log", row=1, col=1)
    fig.update_yaxes(title="Kanten", type="log", row=2, col=1)
    fig = _base(fig, height)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


def build_scaling(rows, height=340):
    """Durchsuchte Kanten gegen die Kartengröße (doppelt logarithmisch): Ungarische Methode, Auktion mit ε-Skalierung, feste Auktion (nur bis zur Größe, die sich noch rechnen lässt)."""
    fig = go.Figure()
    ns = [r["n"] for r in rows]
    fixed = [r for r in rows if r["fixed_scans"] is not None]
    fig.add_trace(go.Scatter(x=ns, y=[r["hung_scans"] for r in rows], mode="lines+markers", name="Ungarische Methode", line=dict(color=C.COLORS["hungarian"])))
    fig.add_trace(go.Scatter(x=ns, y=[r["scaled_scans"] for r in rows], mode="lines+markers", name="Auktion, ε-Skalierung", line=dict(color=C.COLORS["scaled"])))
    fig.add_trace(go.Scatter(x=[r["n"] for r in fixed], y=[r["fixed_scans"] for r in fixed], mode="lines+markers", name="Auktion, fest", line=dict(color=C.COLORS["fixed"])))
    fig.update_xaxes(title="Fahrzeuge = Aufträge", type="log")
    fig.update_yaxes(title="durchsuchte Kanten", type="log")
    return _base(fig, height)


def build_penalty(rows, height=300):
    """Gebote gegen die Strafe M für Verzicht (Vielfache des Standards): die feste Auktion wächst mit M, die ε-Skalierung kaum."""
    labels = [f"{r['factor']}×M" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["fixed_bids"] for r in rows], name="fest ε klein", marker_color=C.COLORS["fixed"]))
    fig.add_trace(go.Bar(x=labels, y=[r["scaled_bids"] for r in rows], name="ε-Skalierung", marker_color=C.COLORS["scaled"]))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Strafe für Verzicht", type="category")
    fig.update_yaxes(title="Gebote (Mittel)", type="log")
    return _base(fig, height)
