"""Auswertung: eine Karte (`analyse`, `verdict`, `compare_table`), viele Karten (`distribution`, `effort_table`, `eps_sweep`), Sweeps.

Alle Größen kommen aus ganzen Zahlen und deterministischen Verfahren; nur die Anzeige-Statistiken (Anteile, Mittel) sind Gleitkomma.
Aufwand wird in Geboten und durchsuchten Kanten gezählt (Vorwärts: Kantenliste eines Fahrzeugs je Entscheidung, Rückwärts: Spalte
eines Auftrags), nie in Sekunden. Vergleichsgröße der Ungarischen Methode: ihre durchsuchten Fahrzeug->Auftrag-Kanten. Die Auktion
braucht keinen Heap, die Ungarische Methode schon (Heap-Operationen kommen bei ihr mit einem Log-Faktor hinzu).
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import au_constants as C
from au_algorithm import auction, certificate
from au_hungarian import hungarian
from au_scenario import build, generate

OPTIMAL, NEAR, MISMATCH, NONE = "optimal", "near", "mismatch", "none"
VARIANTS = (("sequential", "scaled"), ("simultaneous", "scaled"), ("sequential", "fixed"), ("simultaneous", "fixed"))
VARIANT_LABELS = {("sequential", "scaled"): "Nacheinander, ε-Skalierung", ("simultaneous", "scaled"): "Gleichzeitig, ε-Skalierung",
                  ("sequential", "fixed"): "Nacheinander, fest ε = 1/(n+1) min", ("simultaneous", "fixed"): "Gleichzeitig, fest ε = 1/(n+1) min"}


def eps_units(epsv, n):
    """Feste Schrittweite in 1/S Minute (S = n + 1): epsv = 0 heißt exakt (1), sonst epsv Minuten."""
    return 1 if epsv == 0 else (n + 1) * epsv


def run_choice(sc, bid, eps, epsv, record=True, **kw):
    return auction(sc, bid, eps, eps_units(epsv, sc.n) if eps == "fixed" else 1, record=record, **kw)


@dataclass
class Analysis:
    scenario: object
    result: object       # gewählte Auktion (mit Protokoll)
    hung: object         # Ungarische Methode (Stück 3): Referenz und Vergleichsverfahren
    runs: dict           # (Gebotsart, eps-Modus) -> Result ohne Protokoll, alle vier exakten Varianten


def analyse(sc, bid=C.DEFAULT_BID, eps=C.DEFAULT_EPS, epsv=C.DEFAULT_EPSV):
    res = run_choice(sc, bid, eps, epsv)
    runs = {v: auction(sc, v[0], v[1], 1, record=False) for v in VARIANTS}
    return Analysis(sc, res, hungarian(sc, record=False), runs)


def penalised_gap(res, hung, n):
    """Kosten des Ergebnisses abzüglich der des Optimums, beides einschließlich der Strafe für Verzichte (in Minuten)."""
    return res.penalised - (hung.cost + res.penalty * (n - hung.count))


def classify(res, hung, cert, n):
    if hung.count == 0:
        return NONE
    if (res.count, res.cost) == (hung.count, hung.cost):
        return OPTIMAL
    gap = penalised_gap(res, hung, n)
    return NEAR if 0 < gap <= cert["bound"] / res.scale else MISMATCH


def verdict(a):
    """(Stufe, Code, Zahlen) für die Anzeige; `Zahlen` enthält jede Zahl, die der Text nennt."""
    r, h, sc = a.result, a.hung, a.scenario
    cert = certificate(sc, r)
    code = classify(r, h, cert, sc.n)
    c = r.counters
    data = {"count": r.count, "opt_count": h.count, "cost": r.cost, "opt_cost": h.cost, "gap": penalised_gap(r, h, sc.n), "bound_min": cert["bound"] / r.scale, "n_idle": r.n_idle,
            "bids": c["bids"], "wasted": c["wasted"], "rounds": c["rounds"], "phases": c["phases"], "takeovers": c["takeovers"], "fwd_scans": c["fwd_scans"], "rev_scans": c["rev_scans"],
            "scans": r.scans_total, "hung_scans": h.scanned_total, "max_price_min": c["max_price"] / r.scale, "penalty": r.penalty, "scale": r.scale, "eps_final": r.eps_final,
            "eps_min": r.eps_final / r.scale, "cert": cert, "edges": int(sc.feasible.sum()), "bids_by_phase": list(c["bids_by_phase"]),
            "fixed_bids": a.runs[("sequential", "fixed")].counters["bids"], "scaled_bids": a.runs[("sequential", "scaled")].counters["bids"]}
    level = {NONE: "info", OPTIMAL: "success", NEAR: "warning", MISMATCH: "error"}[code]
    return level, code, data


def compare_table(a):
    """Zeilen für die Ungarische Methode und die vier exakten Auktionsvarianten auf der aktuellen Karte."""
    h = a.hung
    rows = [{"label": "Ungarische Methode (zentral)", "count": h.count, "cost": h.cost, "bids": None, "rounds": len(h.rounds), "scans": h.scanned_total, "wasted": None}]
    for v in VARIANTS:
        r = a.runs[v]
        rows.append({"label": "Auktion: " + VARIANT_LABELS[v], "count": r.count, "cost": r.cost, "bids": r.counters["bids"],
                     "rounds": r.counters["rounds"] if v[0] == "simultaneous" else None, "scans": r.scans_total, "wasted": r.counters["wasted"] if v[0] == "simultaneous" else None})
    return rows


# --- viele Karten --------------------------------------------------------------------------------------------------------------

def _row(r):
    c = r.counters
    return (r.count, r.cost, c["bids"], r.scans_total, c["rounds"], c["wasted"], c["takeovers"], c["rev_scans"], c["max_price"] // r.scale)


@lru_cache(maxsize=256)
def cell_rows(n, m, reach, ballung, seeds):
    """Je Seed: Ungarische Methode (Paare, Kosten, Kantensuchen) und die vier exakten Auktionsvarianten als Ganzzahlen. Rückgabe: Tupel von Dicts."""
    rows = []
    for s in seeds:
        sc = generate(n, m, reach, ballung, s)
        h = hungarian(sc, record=False)
        rows.append({"hung": (h.count, h.cost, h.scanned_total), "edges": int(sc.feasible.sum()), **{v: _row(auction(sc, v[0], v[1], 1, record=False)) for v in VARIANTS}})
    return tuple(rows)


@lru_cache(maxsize=256)
def coarse_rows(n, m, reach, ballung, epsv, seeds, bid="sequential"):
    """Je Seed bei fester Schrittweite epsv (Minuten, 0 = exakt): (Paare, Kosten, Gebote, Lücke einschließlich Strafe in Minuten, Schranke n*eps in Minuten)."""
    rows = []
    for s in seeds:
        sc = generate(n, m, reach, ballung, s)
        h = hungarian(sc, record=False)
        r = auction(sc, bid, "fixed", eps_units(epsv, n), record=False)
        rows.append((r.count, r.cost, r.counters["bids"], penalised_gap(r, h, n), n * r.eps_final / r.scale, h.count, h.cost))
    return tuple(rows)


def _mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def distribution(n, m, reach, ballung, seeds=C.DIST_SEEDS):
    """Verteilung über viele Karten: Exaktheit, Gebote, Kantensuchen und Runden der vier Auktionsvarianten gegen die Ungarische Methode."""
    rows = [r for r in cell_rows(n, m, reach, ballung, tuple(seeds)) if r["hung"][0] > 0]
    nv = len(rows)
    if nv == 0:
        return {"n_seeds": len(seeds), "n_valid": 0}
    out = {"n_seeds": len(seeds), "n_valid": nv, "hung_scanned_mean": _mean(r["hung"][2] for r in rows), "edges_mean": _mean(r["edges"] for r in rows), "variants": {}}
    for v in VARIANTS:
        vs = [r[v] for r in rows]
        out["variants"][v] = {"share_optimal": float(np.mean([x[:2] == r["hung"][:2] for x, r in zip(vs, rows)])), "bids_mean": _mean(x[2] for x in vs), "bids_median": float(np.median([x[2] for x in vs])),
                              "scans_mean": _mean(x[3] for x in vs), "rounds_mean": _mean(x[4] for x in vs), "wasted_mean": _mean(x[5] for x in vs), "takeovers_mean": _mean(x[6] for x in vs),
                              "rev_scans_mean": _mean(x[7] for x in vs), "bids": [x[2] for x in vs], "max_price_mean": _mean(x[8] for x in vs)}
    sq, fx = out["variants"][VARIANTS[0]], out["variants"][VARIANTS[2]]
    out["fixed_worse_share"] = float(np.mean([r[VARIANTS[2]][2] > r[VARIANTS[0]][2] for r in rows]))
    out["scaled_vs_hung"] = sq["scans_mean"] / out["hung_scanned_mean"] if out["hung_scanned_mean"] else None
    out["fixed_vs_scaled_bids"] = fx["bids_mean"] / sq["bids_mean"] if sq["bids_mean"] else None
    return out


def effort_table(n, m, reach, ballung, seeds=C.DIST_SEEDS):
    """Zeilen für die Ungarische Methode und die vier Auktionsvarianten: Anteil Optimum, Gebote, Runden, Kantensuchen (Mittel über die festen Karten)."""
    rows = [r for r in cell_rows(n, m, reach, ballung, tuple(seeds)) if r["hung"][0] > 0]
    if not rows:
        return []
    out = [{"label": "Ungarische Methode (zentral)", "optimal": 1.0, "bids": None, "rounds": None, "scans": _mean(r["hung"][2] for r in rows)}]
    for v in VARIANTS:
        out.append({"label": "Auktion: " + VARIANT_LABELS[v], "optimal": float(np.mean([r[v][:2] == r["hung"][:2] for r in rows])), "bids": _mean(r[v][2] for r in rows),
                    "rounds": _mean(r[v][4] for r in rows) if v[0] == "simultaneous" else None, "scans": _mean(r[v][3] for r in rows)})
    return out


def eps_sweep(n, m, reach, ballung, seeds=C.DIST_SEEDS, values=C.EPS_SWEEP):
    """Je fester Schrittweite (Minuten; 0 = exakt): mittlere Gebote, Anteil Optimum, größte Lücke, Schranke n*eps; dazu die Skalierung als Bezug."""
    out = []
    for k in values:
        rows = [r for r in coarse_rows(n, m, reach, ballung, k, tuple(seeds)) if r[5] > 0]
        if not rows:
            continue
        out.append({"eps": k, "bids": _mean(r[2] for r in rows), "optimal": float(np.mean([(r[0], r[1]) == (r[5], r[6]) for r in rows])), "max_gap": max(r[3] for r in rows),
                    "mean_gap": _mean(r[3] for r in rows), "bound": rows[0][4], "pairs_lost": sum(r[5] - r[0] for r in rows)})
    d = distribution(n, m, reach, ballung, seeds)
    ref = d["variants"][VARIANTS[0]]["bids_mean"] if d["n_valid"] else None
    return out, ref


def reach_sweep(n, m, ballung, values=C.REACH_SWEEP, seeds=C.SWEEP_SEEDS):
    """Je Reichweite: Gebote (fest, skaliert) und Kantensuchen (Ungarisch, skaliert, fest)."""
    out = []
    for reach in values:
        d = distribution(n, m, reach, ballung, seeds)
        if d["n_valid"] == 0:
            out.append({"x": reach, "fixed_bids": None, "scaled_bids": None, "hung_scans": None, "scaled_scans": None, "fixed_scans": None})
            continue
        sq, fx = d["variants"][VARIANTS[0]], d["variants"][VARIANTS[2]]
        out.append({"x": reach, "fixed_bids": fx["bids_mean"], "scaled_bids": sq["bids_mean"], "hung_scans": d["hung_scanned_mean"], "scaled_scans": sq["scans_mean"], "fixed_scans": fx["scans_mean"]})
    return out


def scaling(ns=C.SCALE_NS, seeds=C.SCALE_SEEDS):
    """Aufwand wächst mit der Karte: n = m bei konstantem mittleren Grad; Kantensuchen der Ungarischen Methode, der skalierten und (bis n = 80) der festen Auktion."""
    out = []
    for n in ns:
        reach = max(5, round(C.SCALE_REACH_AT_20 * math.sqrt(20 / n)))
        acc = []
        for s in seeds:
            sc = generate(n, n, reach, 0, s)
            h = hungarian(sc, record=False)
            a = auction(sc, record=False)
            f = auction(sc, "sequential", "fixed", 1, record=False) if n <= C.SCALE_FIXED_MAX_N else None
            acc.append((h.scanned_total, a.scans_total, a.counters["bids"], f.scans_total if f else float("nan"), f.counters["bids"] if f else float("nan")))
        arr = np.array(acc, dtype=float).mean(axis=0)
        out.append({"n": n, "reach": reach, "hung_scans": float(arr[0]), "scaled_scans": float(arr[1]), "scaled_bids": float(arr[2]),
                    "fixed_scans": None if math.isnan(arr[3]) else float(arr[3]), "fixed_bids": None if math.isnan(arr[4]) else float(arr[4])})
    return out


def penalty_experiment(n, m, reach, ballung, seeds=C.SWEEP_SEEDS, factors=C.M_FACTORS):
    """Wie hängt der Aufwand von der Strafe M für Verzicht ab (Vielfache des Standards)? Feste Auktion gegen ε-Skalierung, mittlere Gebote."""
    out = []
    for f in factors:
        fx, sq = [], []
        for s in seeds:
            sc = generate(n, m, reach, ballung, s)
            base = min(n, m) * max([int(sc.cost[i, j]) for i in range(n) for j in range(m) if sc.feasible[i, j]] or [1]) + 1
            fx.append(auction(sc, "sequential", "fixed", 1, penalty=f * base, record=False).counters["bids"])
            sq.append(auction(sc, penalty=f * base, record=False).counters["bids"])
        out.append({"factor": f, "fixed_bids": _mean(fx), "scaled_bids": _mean(sq)})
    return out


def scenario_from_settings(net, n, m, reach, ballung, seed):
    return build(net, n, m, reach, ballung, seed)


def progress_series(res, n, m):
    """Je Ereignis (Index 0 = Anfang): Fahrzeuge ohne Auftrag und Summe der Preise in Minuten; dazu die Ereignisindizes der Phasenwechsel."""
    from au_algorithm import apply_event, initial_state
    st = initial_state(n, m)
    unassigned, price_sum, phases = [n], [0.0], []
    for k, e in enumerate(res.events, start=1):
        apply_event(st, e)
        if e[0] == "phase":
            phases.append(k)
        unassigned.append(sum(1 for i in range(n) if st["assign"][i] < 0 and not st["idle"][i]))
        price_sum.append(sum(st["p"]) / res.scale)
    return unassigned, price_sum, phases
