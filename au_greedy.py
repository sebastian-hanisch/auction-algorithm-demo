"""Aus der Greedy-Matching-Demo (Stück 1 der Linie) übernommen: zwei Greedy-Regeln, die exakte Messlatte und die Symmetrische Differenz zweier Matchings.

Ein Matching ist eine Menge von Paaren (Fahrzeug i, Auftrag j) ohne gemeinsame Fahrzeuge oder Aufträge. Bewertet wird
lexikografisch: erst möglichst viele Paare, dann die geringste Summe der Anfahrtszeiten.

Der exakte Löser ist hier nur die Messlatte, an der sich Greedy messen lässt - das Verfahren selbst (ungarische
Methode) ist ein späteres Stück der Linie. Er rechnet mit ganzen Zahlen und ist deterministisch, daher liefert er auf jeder
Plattform dieselben Paare.
"""

from dataclasses import dataclass

import numpy as np

RULE_EDGE = "edge"    # billigste Kante zuerst
RULE_ORDER = "order"  # Auftrag für Auftrag, nächstes freies Fahrzeug
RULES = (RULE_EDGE, RULE_ORDER)


@dataclass(frozen=True)
class Matching:
    pairs: tuple   # ((i, j), ...) nach Auswahlreihenfolge; der exakte Löser liefert sie nach Fahrzeug sortiert
    cost: int

    @property
    def count(self):
        return len(self.pairs)

    def key(self):
        """Ordnungsschlüssel: größer ist besser (mehr Paare, dann geringere Kosten)."""
        return self.count, -self.cost


def _matching(sc, pairs):
    return Matching(tuple(pairs), int(sum(sc.cost[i, j] for i, j in pairs)))


def greedy_cheapest_edge(sc):
    """Regel A: alle möglichen Kanten nach (Kosten, Fahrzeug, Auftrag) sortieren, jede nehmen, deren Enden beide frei sind."""
    used_v, used_o, pairs = set(), set(), []
    for _c, i, j in sc.edges():
        if i not in used_v and j not in used_o:
            used_v.add(i)
            used_o.add(j)
            pairs.append((i, j))
    return _matching(sc, pairs)


def greedy_by_order(sc):
    """Regel B: Aufträge in der Reihenfolge ihres Index; jeder nimmt das freie mögliche Fahrzeug mit den geringsten Kosten."""
    used_v, pairs = set(), []
    for j in range(sc.m):
        best = None
        for i in range(sc.n):
            if i not in used_v and sc.feasible[i, j] and (best is None or int(sc.cost[i, j]) < best[0]):
                best = (int(sc.cost[i, j]), i)
        if best is not None:
            used_v.add(best[1])
            pairs.append((best[1], j))
    return _matching(sc, pairs)


def run_rule(sc, rule):
    return greedy_cheapest_edge(sc) if rule == RULE_EDGE else greedy_by_order(sc)


def is_maximal(sc, pairs):
    """Maximal heißt: keine mögliche Kante hat zwei freie Enden (nicht zu verwechseln mit 'größtmöglich')."""
    used_v = {i for i, _ in pairs}
    used_o = {j for _, j in pairs}
    return not any(sc.feasible[i, j] for i in range(sc.n) if i not in used_v for j in range(sc.m) if j not in used_o)


def _assign_min_cost(a):
    """Kürzeste augmentierende Wege mit Potenzialen auf einer quadratischen ganzzahligen Kostenmatrix (Jonker-Volgenant-Art).
    Liefert `col_to_row` (Länge N, Zeilenindex je Spalte)."""
    N = a.shape[0]
    INF = np.iinfo(np.int64).max // 4
    u = np.zeros(N + 1, dtype=np.int64)
    v = np.zeros(N + 1, dtype=np.int64)
    p = np.zeros(N + 1, dtype=np.int64)     # p[j] = Zeile (1-basiert), die Spalte j hält
    way = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        p[0] = i
        j0 = 0
        minv = np.full(N + 1, INF, dtype=np.int64)
        used = np.zeros(N + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = a[i0 - 1] - u[i0] - v[1:]
            free = ~used[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free, minv[1:], INF)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
    return p[1:] - 1


def optimum(sc):
    """Exaktes Optimum (erst Paarzahl, dann Kosten). Trick: eine quadratische Zuordnung, in der jedes nicht mögliche Paar
    BIG kostet; BIG übersteigt jede Kostensumme aller echten Paare, daher wird zuerst die Zahl der BIG-Paare minimiert
    (= Paarzahl maximiert) und erst danach die echten Kosten."""
    N = max(sc.n, sc.m)
    cmax = int(sc.cost.max()) if sc.cost.size else 0
    big = cmax * min(sc.n, sc.m) + 1
    a = np.full((N, N), big, dtype=np.int64)
    real = np.where(sc.feasible, sc.cost, big)
    a[:sc.n, :sc.m] = real
    col_to_row = _assign_min_cost(a)
    pairs = sorted((int(i), j) for j, i in enumerate(col_to_row) if i < sc.n and j < sc.m and sc.feasible[i, j])
    return _matching(sc, pairs)


def alternating_components(g_pairs, o_pairs):
    """Symmetrische Differenz zweier Matchings: jede Ecke hat höchstens Grad 2, also zerfällt sie in alternierende Wege und
    gerade Kreise. Rückgabe: Liste von Komponenten {"kind", "edges": [(i, j, "G"|"O"), ...]}. `kind` ist
    "g_augment" (ein Weg, der G um ein Paar verbessern würde), "o_augment" (umgekehrt, kann bei optimalem O nicht vorkommen),
    "path" (gleich viele Kanten, Kosten-Tausch) oder "cycle"."""
    g, o = set(g_pairs), set(o_pairs)
    edges = [(i, j, "G") for i, j in sorted(g - o)] + [(i, j, "O") for i, j in sorted(o - g)]
    adj = {}
    for idx, (i, j, _s) in enumerate(edges):
        adj.setdefault(("v", i), []).append(idx)
        adj.setdefault(("o", j), []).append(idx)
    seen, comps = set(), []
    for start in range(len(edges)):
        if start in seen:
            continue
        stack, members = [start], []
        seen.add(start)
        while stack:
            e = stack.pop()
            members.append(e)
            i, j, _s = edges[e]
            for node in (("v", i), ("o", j)):
                for f in adj[node]:
                    if f not in seen:
                        seen.add(f)
                        stack.append(f)
        comp = sorted(members)
        n_g = sum(1 for e in comp if edges[e][2] == "G")
        n_o = len(comp) - n_g
        if n_o == n_g + 1:
            kind = "g_augment"
        elif n_g == n_o + 1:
            kind = "o_augment"
        else:
            degrees = {}
            for e in comp:
                i, j, _s = edges[e]
                degrees[("v", i)] = degrees.get(("v", i), 0) + 1
                degrees[("o", j)] = degrees.get(("o", j), 0) + 1
            kind = "cycle" if all(d == 2 for d in degrees.values()) else "path"
        comps.append({"kind": kind, "edges": [edges[e] for e in comp]})
    return comps
