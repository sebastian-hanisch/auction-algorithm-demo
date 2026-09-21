"""Ungarische Methode als kürzeste Verbesserungswege mit Potenzialen (Successive Shortest Paths auf dem Restgraphen).

Netz: s -> Fahrzeug (Kosten 0) -> Auftrag (Kosten c) -> t (Kosten 0), alle Kapazitäten 1. Nach k Runden ist das Matching das billigste
mit k Paaren; endet das Verfahren (kein freier Auftrag mehr erreichbar), hat es die größtmögliche Paarzahl und unter allen solchen die
geringsten Kosten - lexikografisch, ohne Big-M.

Potenziale: pi_v[i] (Fahrzeuge), pi_o[j] (Aufträge), P (t), pi_s = 0. Restkanten: nicht gewählte mögliche Paare i -> j mit Kosten c, gewählte
j -> i mit -c; reduzierte Kosten r = w + pi_a - pi_b >= 0. Jede Runde: ein Dijkstra von ALLEN freien Fahrzeugen (Abstand 0), Abbruch beim
Festlegen des ersten freien Auftrags (Abstand d_t). Danach `pi_x += d_x` für festgelegte Ecken (d_x <= d_t), `pi_x += d_t` für alle anderen,
`P += d_t`; die Wegkosten sind `W = d_t + P_alt` und `P = W` sind die Grenzkosten des neuen Paars.

Aufwand wird gezählt (durchsuchte Fahrzeug->Auftrag-Kanten, Rückkanten-Schritte, Heap-Einfügungen und -Entnahmen), nie in Sekunden.
Alles ganzzahlig und deterministisch: Gleichstand im Heap entscheidet (Abstand, Art, Index), Aufträge vor Fahrzeugen.
"""

import heapq
from collections import deque
from dataclasses import dataclass

from au_augment import _adjacency, _apply, _path_from_orders, _snapshot

KIND_ORDER, KIND_VEHICLE = 0, 1


@dataclass(frozen=True)
class Round:
    path: tuple            # ((i, j, "add" | "drop"), ...) wie in der Verbesserungswege-Demo
    length: int
    w: int                 # Wegkosten = Grenzkosten des neuen Paars
    d_t: int               # reduzierter Abstand des Zielauftrags
    p_before: int
    p_after: int
    settled: tuple         # ((art, index, d, rang), ...) in der Reihenfolge des Festlegens; art 0 = Auftrag, 1 = Fahrzeug
    tree: tuple            # Vorgängerbaum: ((i, j), ...) für jeden festgelegten Auftrag j, über den Fahrzeug i entdeckt wurde
    pi_v_before: tuple
    pi_v_after: tuple
    pi_o_before: tuple
    pi_o_after: tuple
    scanned: int           # durchsuchte Fahrzeug->Auftrag-Kanten
    matched_arcs: int      # Rückkanten-Schritte (Auftrag -> sein Fahrzeug)
    pushes: int
    pops: int
    pairs_after: int


@dataclass(frozen=True)
class Result:
    pairs: tuple
    cost: int
    rounds: tuple
    final: dict            # die erfolglose letzte Suche: scanned, matched_arcs, pushes, pops, settled
    states: tuple          # Zustand vor Runde 1, nach Runde 1, ... (Tupel von Paaren)
    pi_v: tuple
    pi_o: tuple
    p: int
    cover_v: tuple
    cover_o: tuple

    @property
    def count(self):
        return len(self.pairs)

    @property
    def marginal(self):
        return tuple(r.w for r in self.rounds)

    @property
    def scanned_total(self):
        return sum(r.scanned for r in self.rounds) + self.final["scanned"]

    @property
    def pops_total(self):
        return sum(r.pops for r in self.rounds) + self.final["pops"]

    @property
    def pushes_total(self):
        return sum(r.pushes for r in self.rounds) + self.final["pushes"]

    @property
    def matched_arcs_total(self):
        return sum(r.matched_arcs for r in self.rounds) + self.final["matched_arcs"]

    def key(self):
        return self.count, -self.cost


def _dijkstra(sc, adj, match_v, match_o, pi_v, pi_o, stop_at_free_order=True):
    """Ein Mehrfachstart-Dijkstra auf reduzierten Kosten. Rückgabe: dict mit Weg-Bausteinen, Abständen, Festlegungsliste und Zählern."""
    n, m = sc.n, sc.m
    INF = 1 << 60
    dist_v, dist_o = [INF] * n, [INF] * m
    pred_o = {}
    settled_v, settled_o = [False] * n, [False] * m
    heap = []
    for i in range(n):
        if match_v[i] < 0:
            dist_v[i] = 0
            heap.append((0, KIND_VEHICLE, i))
    heapq.heapify(heap)
    pushes, pops, scanned, matched_arcs = len(heap), 0, 0, 0
    order, target, d_t = [], -1, 0
    while heap:
        d, kind, x = heapq.heappop(heap)
        pops += 1
        if kind == KIND_VEHICLE:
            if settled_v[x] or d > dist_v[x]:
                continue
            settled_v[x] = True
            order.append((kind, x, d))
            for j in adj[x]:
                scanned += 1
                if match_v[x] == j:
                    continue
                nd = d + int(sc.cost[x, j]) + pi_v[x] - pi_o[j]
                if nd < dist_o[j]:
                    dist_o[j] = nd
                    pred_o[j] = x
                    heapq.heappush(heap, (nd, KIND_ORDER, j))
                    pushes += 1
        else:
            if settled_o[x] or d > dist_o[x]:
                continue
            settled_o[x] = True
            order.append((kind, x, d))
            if match_o[x] < 0:
                if stop_at_free_order:
                    target, d_t = x, d
                    break
                continue
            i2 = match_o[x]                      # Rückkante: gewählte Kanten sind straff, reduzierte Kosten 0
            matched_arcs += 1
            if d < dist_v[i2]:
                dist_v[i2] = d
                heapq.heappush(heap, (d, KIND_VEHICLE, i2))
                pushes += 1
    return {"target": target, "d_t": d_t, "order": order, "pred_o": pred_o, "dist_v": dist_v, "dist_o": dist_o, "settled_v": settled_v, "settled_o": settled_o,
            "scanned": scanned, "matched_arcs": matched_arcs, "pushes": pushes, "pops": pops}


def hungarian(sc, record=True):
    """Ungarische Methode; Rückgabe `Result` mit allen Runden (Weg, Grenzkosten, Potenziale vorher/nachher, Festlegungsreihenfolge, Aufwand)."""
    n, m = sc.n, sc.m
    adj = _adjacency(sc)
    match_v, match_o = [-1] * n, [-1] * m
    pi_v, pi_o, p = [0] * n, [0] * m, 0
    states = [()] if record else []
    rounds = []
    while True:
        s = _dijkstra(sc, adj, match_v, match_o, pi_v, pi_o)
        if s["target"] < 0:
            final = {"scanned": s["scanned"], "matched_arcs": s["matched_arcs"], "pushes": s["pushes"], "pops": s["pops"],
                     "settled": tuple((k, x, d, r) for r, (k, x, d) in enumerate(s["order"]))}
            break
        j_end, d_t = s["target"], s["d_t"]
        chain_v, chain_o, cur = [], [], j_end
        while True:                                          # Weg rückwärts über die Vorgänger
            v = s["pred_o"][cur]
            chain_v.append(v)
            chain_o.append(cur)
            if match_v[v] < 0:
                break
            cur = match_v[v]
        path = _path_from_orders(chain_v[::-1], chain_o[::-1])
        pv_b, po_b, p_b = tuple(pi_v), tuple(pi_o), p
        for i in range(n):
            pi_v[i] += s["dist_v"][i] if s["settled_v"][i] else d_t
        for j in range(m):
            pi_o[j] += s["dist_o"][j] if s["settled_o"][j] else d_t
        p += d_t
        _apply(path, match_v, match_o)
        tree = tuple((s["pred_o"][x], x) for k, x, d in s["order"] if k == KIND_ORDER)
        rounds.append(Round(path, len(path), d_t + p_b, d_t, p_b, p, tuple((k, x, d, r) for r, (k, x, d) in enumerate(s["order"])), tree,
                            pv_b, tuple(pi_v), po_b, tuple(pi_o), s["scanned"], s["matched_arcs"], s["pushes"], s["pops"], sum(1 for j in match_v if j >= 0)))
        if record:
            states.append(_snapshot(match_v))
    final_pairs = _snapshot(match_v)
    seen_v = {x for k, x, d, r in final["settled"] if k == KIND_VEHICLE}
    seen_o = {x for k, x, d, r in final["settled"] if k == KIND_ORDER}
    cost = int(sum(sc.cost[i, j] for i, j in final_pairs))
    return Result(final_pairs, cost, tuple(rounds), final, tuple(states), tuple(pi_v), tuple(pi_o), p,
                  tuple(i for i in range(n) if i not in seen_v), tuple(sorted(seen_o)))


def certificate(sc, res):
    """Optimalitätsbeweis in Zahlen (Dualzulässigkeit des Gesamtnetzes). C1 gewählte Kanten straff, C2 pi_o - pi_v <= c auf allen möglichen Kanten,
    C3 freie Fahrzeuge pi = 0 und alle pi >= 0, C4 freie Aufträge pi = P und alle pi <= P, C5 kein freier Auftrag erreichbar (Überdeckung mit |M| Ecken);
    dazu die Kostenidentität cost = sum(pi_o) - sum(pi_v) - (m - nu) * P und Kennzahlen für die Anzeige."""
    matched_v = {i: j for i, j in res.pairs}
    free_v = [i for i in range(sc.n) if i not in matched_v]
    free_o = [j for j in range(sc.m) if j not in {j for _, j in res.pairs}]
    slack = {(i, j): int(sc.cost[i, j]) - (res.pi_o[j] - res.pi_v[i]) for i in range(sc.n) for j in range(sc.m) if sc.feasible[i, j]}
    matched_slacks = [slack[p] for p in res.pairs]
    other = [v for k, v in slack.items() if k not in set(res.pairs)]
    c1 = all(v == 0 for v in matched_slacks)
    c2 = all(v >= 0 for v in slack.values())
    c3 = all(res.pi_v[i] == 0 for i in free_v) and all(v >= 0 for v in res.pi_v)
    c4 = all(res.pi_o[j] == res.p for j in free_o) and all(v <= res.p for v in res.pi_o)
    c5 = (all(i in res.cover_v or j in res.cover_o for i in range(sc.n) for j in range(sc.m) if sc.feasible[i, j])
          and len(res.cover_v) + len(res.cover_o) == res.count)
    identity_rhs = sum(res.pi_o) - sum(res.pi_v) - (sc.m - res.count) * res.p
    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5, "identity": res.cost == identity_rhs, "identity_rhs": identity_rhs,
            "n_tight": sum(1 for v in slack.values() if v == 0), "n_edges": len(slack), "min_slack": min(other) if other else None,
            "p": res.p, "cover_size": len(res.cover_v) + len(res.cover_o), "all_ok": c1 and c2 and c3 and c4 and c5 and res.cost == identity_rhs}
