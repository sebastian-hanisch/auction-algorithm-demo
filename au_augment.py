"""Augmentierende Pfade: aus einem Matching (leer oder das Ergebnis von Greedy) durch wiederholtes Umklappen von Verbesserungswegen ein
größtmögliches machen. Breitensuche (kürzester Weg) und Tiefensuche (Kuhn) als Wegesuche.

Ein *Verbesserungsweg* beginnt an einem freien Fahrzeug, geht über eine nicht gewählte Kante zu einem Auftrag, von dort über die gewählte Kante
zurück zu dessen Fahrzeug, wieder über eine nicht gewählte Kante zu einem Auftrag, ... und endet an einem freien Auftrag. Er hat immer eine
nicht gewählte Kante mehr als gewählte; klappt man ihn um (gewählt <-> nicht gewählt), hat das Matching ein Paar mehr (Berge, 1957).

Die Nachbarn jedes Fahrzeugs stehen in Indexreihenfolge - die Suche kennt die Kosten nicht. Das ist gewollt: sie zeigt, was Verbesserungswege
für die Paarzahl leisten und was nicht für die Kosten. Aufwand wird in durchsuchten Kanten gezählt (maschinenunabhängig), nie in Sekunden.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from au_greedy import run_rule

SEARCH_BFS = "bfs"
SEARCH_DFS = "dfs"
SEARCHES = (SEARCH_BFS, SEARCH_DFS)
START_EMPTY = "empty"
STARTS = (START_EMPTY, "edge", "order")     # leer, Greedy 'Billigste Kante zuerst', Greedy 'Auftrag für Auftrag'


@dataclass(frozen=True)
class Round:
    path: tuple        # ((i, j, "add" | "drop"), ...): "add" = Kante wird gewählt, "drop" = gewählte Kante wird freigegeben; beginnt und endet mit "add"
    length: int        # Zahl der Kanten des Wegs (2k+1)
    scanned: int       # in dieser Runde durchsuchte Kanten (bei Tiefensuche einschließlich vorher gescheiterter Startfahrzeuge)
    pairs_after: int


@dataclass(frozen=True)
class Result:
    start: tuple           # Paare zu Beginn
    pairs: tuple           # Paare am Ende (nach Fahrzeug sortiert)
    cost: int
    rounds: tuple          # erfolgreiche Runden
    final_scanned: int     # durchsuchte Kanten der letzten, erfolglosen Suche (der Beweis)
    states: tuple          # Zustand vor Runde 1, nach Runde 1, ... (Tupel von Paaren); leer, wenn nicht aufgezeichnet
    cover_v: tuple         # Knotenüberdeckung aus der erfolglosen Suche: Fahrzeuge ...
    cover_o: tuple         # ... und Aufträge; genau |M| Ecken, jede mögliche Kante hat mindestens eine

    @property
    def count(self):
        return len(self.pairs)

    @property
    def scanned_total(self):
        return sum(r.scanned for r in self.rounds) + self.final_scanned

    def key(self):
        return self.count, -self.cost


def start_pairs(sc, start):
    """Startpaare: leer oder das Ergebnis einer Greedy-Regel."""
    return () if start == START_EMPTY else run_rule(sc, start).pairs


def _adjacency(sc):
    return [[j for j in range(sc.m) if sc.feasible[i, j]] for i in range(sc.n)]


def _path_from_orders(chain_v, chain_o):
    """Weg aus Fahrzeugen v0, v1, ... und Aufträgen o0, o1, ...: (v0,o0) wird gewählt, (v1,o0) freigegeben, (v1,o1) gewählt, ..."""
    path = []
    for k, (v, o) in enumerate(zip(chain_v, chain_o)):
        path.append((v, o, "add"))
        if k + 1 < len(chain_v):
            path.append((chain_v[k + 1], o, "drop"))
    return tuple(path)


def _bfs_round(adj, match_v, match_o, n):
    """Kürzester Verbesserungsweg: alle freien Fahrzeuge gleichzeitig als Schicht 0. Rückgabe (Weg oder None, durchsuchte Kanten)."""
    scanned = 0
    parent_v = {}                     # Auftrag -> Fahrzeug, über das er entdeckt wurde
    seen_o = set()
    queue = deque(i for i in range(n) if match_v[i] < 0)
    while queue:
        i = queue.popleft()
        for j in adj[i]:
            scanned += 1
            if j in seen_o:
                continue
            seen_o.add(j)
            parent_v[j] = i
            if match_o[j] < 0:                       # freier Auftrag: Weg rückwärts über die Vorgänger zusammensetzen
                chain_v, chain_o, cur = [], [], j
                while True:
                    v = parent_v[cur]
                    chain_v.append(v)
                    chain_o.append(cur)
                    if match_v[v] < 0:
                        break
                    cur = match_v[v]
                return _path_from_orders(chain_v[::-1], chain_o[::-1]), scanned
            queue.append(match_o[j])
    return None, scanned


def _dfs_try(adj, match_v, match_o, start):
    """Kuhn: Tiefensuche ab einem freien Fahrzeug (iterativ). Rückgabe (Weg oder None, durchsuchte Kanten)."""
    scanned = 0
    visited_o = set()
    stack_v, stack_pos, stack_o = [start], [0], []
    while stack_v:
        v, pos = stack_v[-1], stack_pos[-1]
        if pos >= len(adj[v]):
            stack_v.pop()
            stack_pos.pop()
            if stack_v:
                stack_o.pop()
            continue
        stack_pos[-1] += 1
        j = adj[v][pos]
        scanned += 1
        if j in visited_o:
            continue
        visited_o.add(j)
        if match_o[j] < 0:
            return _path_from_orders(stack_v, stack_o + [j]), scanned
        stack_o.append(j)
        stack_v.append(match_o[j])
        stack_pos.append(0)
    return None, scanned


def _reachable(adj, match_v, match_o, n):
    """Von allen freien Fahrzeugen aus alternierend erreichbare Fahrzeuge und Aufträge (Grundlage der Knotenüberdeckung)."""
    seen_v, seen_o = set(), set()
    queue = deque(i for i in range(n) if match_v[i] < 0)
    seen_v.update(queue)
    while queue:
        i = queue.popleft()
        for j in adj[i]:
            if j in seen_o:
                continue
            seen_o.add(j)
            if match_o[j] >= 0 and match_o[j] not in seen_v:
                seen_v.add(match_o[j])
                queue.append(match_o[j])
    return seen_v, seen_o


def _apply(path, match_v, match_o):
    """Weg umklappen: die 'add'-Kanten setzen genügt, die freigegebenen Kanten werden dabei überschrieben."""
    for i, j, kind in path:
        if kind == "add":
            match_v[i] = j
            match_o[j] = i


def _snapshot(match_v):
    return tuple((i, j) for i, j in enumerate(match_v) if j >= 0)


def augment(sc, pairs=(), search=SEARCH_BFS, record=True):
    """Größtmögliches Matching aus `pairs` (leer oder Greedy) durch Verbesserungswege; `search` = Breiten- oder Tiefensuche."""
    n, m = sc.n, sc.m
    adj = _adjacency(sc)
    match_v, match_o = [-1] * n, [-1] * m
    for i, j in pairs:
        assert sc.feasible[i, j] and match_v[i] < 0 and match_o[j] < 0, "Startpaare sind kein Matching"
        match_v[i], match_o[j] = j, i
    start = _snapshot(match_v)
    states = [start] if record else []
    rounds, pending = [], 0
    if search == SEARCH_BFS:
        while True:
            path, scanned = _bfs_round(adj, match_v, match_o, n)
            if path is None:
                pending = scanned
                break
            _apply(path, match_v, match_o)
            rounds.append(Round(path, len(path), scanned, sum(1 for j in match_v if j >= 0)))
            if record:
                states.append(_snapshot(match_v))
    else:
        for i in range(n):                   # jedes Fahrzeug nur einmal: scheitert es, scheitert es auch später (Kuhn)
            if match_v[i] >= 0:
                continue
            path, scanned = _dfs_try(adj, match_v, match_o, i)
            pending += scanned
            if path is None:
                continue
            _apply(path, match_v, match_o)
            rounds.append(Round(path, len(path), pending, sum(1 for j in match_v if j >= 0)))
            pending = 0
            if record:
                states.append(_snapshot(match_v))
    seen_v, seen_o = _reachable(adj, match_v, match_o, n)
    final = _snapshot(match_v)
    cost = int(sum(sc.cost[i, j] for i, j in final))
    return Result(start, final, cost, tuple(rounds), pending, tuple(states),
                  tuple(i for i in range(n) if i not in seen_v), tuple(sorted(seen_o)))


def has_augmenting_path(sc, pairs):
    """Gibt es zu diesem Matching noch einen Verbesserungsweg? (Unabhängige Prüfung für Tests und Verdict.)"""
    adj = _adjacency(sc)
    match_v, match_o = [-1] * sc.n, [-1] * sc.m
    for i, j in pairs:
        match_v[i], match_o[j] = j, i
    return _bfs_round(adj, match_v, match_o, sc.n)[0] is not None


def edge_count(sc):
    return int(np.count_nonzero(sc.feasible))
