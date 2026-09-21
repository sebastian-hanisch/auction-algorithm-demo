"""Szenario: Fahrzeuge und Aufträge auf einer Karte.

Alles ist ganzzahlig und läuft über einen eigenen Zufallsgenerator (SplitMix64 auf Python-Ints) statt über
`numpy.random`: numpy garantiert keine über Versionen stabilen Zufallsströme, die CI installiert aber wöchentlich die
neueste Version. So sind Voreinstellungen, Seeds und jede im Text genannte Zahl auf Windows und Linux dieselben.
Kosten = auf ganze Minuten aufgerundete Entfernung (per `isqrt`, ohne Gleitkomma); ein Paar ist möglich, wenn die
Entfernung höchstens die Reichweite beträgt.
"""

from dataclasses import dataclass
from math import isqrt

import numpy as np

_MASK = (1 << 64) - 1
MAP_SIZE = 100
N_CENTRES = 3


class SplitMix64:
    """Kleiner, gut gemischter 64-Bit-Zufallsgenerator (Vigna); reine Ganzzahl-Arithmetik."""

    def __init__(self, seed):
        self.state = seed & _MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def below(self, n):
        """Ganzzahl in 0..n-1 (die Modulo-Verzerrung bei n <= 101 liegt um 1e-17)."""
        return self.next() % n


def travel_cost(dx, dy):
    """Aufgerundete Entfernung in Minuten und das Quadrat der Entfernung (beides ganzzahlig)."""
    d2 = dx * dx + dy * dy
    r = isqrt(d2)
    return r + (1 if d2 > r * r else 0), d2


@dataclass(frozen=True)
class Scenario:
    vehicles: tuple      # ((x, y), ...)
    orders: tuple
    reach: int
    cost: np.ndarray     # (n, m) int64, Anfahrtszeit in Minuten
    feasible: np.ndarray  # (n, m) bool, Entfernung <= Reichweite

    @property
    def n(self):
        return len(self.vehicles)

    @property
    def m(self):
        return len(self.orders)

    def edges(self):
        """Alle möglichen Paare als (Kosten, Fahrzeug, Auftrag), aufsteigend sortiert (Gleichstand: kleinster Index)."""
        return sorted((int(self.cost[i, j]), i, j) for i in range(self.n) for j in range(self.m) if self.feasible[i, j])


def from_points(vehicles, orders, reach):
    n, m = len(vehicles), len(orders)
    cost = np.zeros((n, m), dtype=np.int64)
    feasible = np.zeros((n, m), dtype=bool)
    for i, (vx, vy) in enumerate(vehicles):
        for j, (ox, oy) in enumerate(orders):
            c, d2 = travel_cost(vx - ox, vy - oy)
            cost[i, j] = c
            feasible[i, j] = d2 <= reach * reach
    return Scenario(tuple(map(tuple, vehicles)), tuple(map(tuple, orders)), int(reach), cost, feasible)


def generate(n, m, reach, ballung, seed):
    """Zufällige Karte. `ballung` in ganzen Prozent: 0 = gleichmäßig verteilt, 100 = alle Punkte um drei Zentren."""
    rng = SplitMix64(seed)
    lo, hi = 15, MAP_SIZE - 15
    centres = [(lo + rng.below(hi - lo + 1), lo + rng.below(hi - lo + 1)) for _ in range(N_CENTRES)]

    def point():
        ux, uy = rng.below(MAP_SIZE + 1), rng.below(MAP_SIZE + 1)
        cx, cy = centres[rng.below(N_CENTRES)]
        jx, jy = rng.below(21) - 10, rng.below(21) - 10
        x = ((100 - ballung) * ux + ballung * (cx + jx)) // 100
        y = ((100 - ballung) * uy + ballung * (cy + jy)) // 100
        return min(max(x, 0), MAP_SIZE), min(max(y, 0), MAP_SIZE)

    vehicles = [point() for _ in range(n)]
    orders = [point() for _ in range(m)]
    return from_points(vehicles, orders, reach)


# --- feste Karten (Lehrbuchfälle), dieselbe Geometrie wie die Zufallskarten --------------------------------------------------

def steal_2x2():
    """Die billigste Kante 'klaut': V1-O1 kostet 4, danach bleibt nur V2-O2 für 14 (zusammen 18); optimal wären 5 + 5."""
    return from_points(vehicles=[(5, 50), (14, 50)], orders=[(9, 50), (0, 50)], reach=20)


def p4_chain(k=1):
    """k getrennte Pfade V1-O1-V2-O2 (Versatz 30 in y, außer Reichweite füreinander). Die mittlere Kante ist die
    billigste, beide Regeln nehmen sie zuerst: Greedy findet k Paare, das Optimum 2k (genau die Hälfte)."""
    vehicles, orders = [], []
    for c in range(k):
        y = 5 + 30 * c
        vehicles += [(0, y), (12, y)]
        orders += [(10, y), (22, y)]
    return from_points(vehicles, orders, reach=10)


def long_chain(k=6):
    """Eine Kette F1-A1-F2-A2-...: die Kanten (F_i, A_i) kosten 10, die Kanten (F_(i+1), A_i) nur 2 - Greedy nimmt alle billigen
    und lässt F1 und der letzte Auftrag übrig. Der einzige Verbesserungsweg läuft durch die ganze Kette: 2k+1 Kanten.
    Alle Punkte liegen auf einer Geraden (Abstand 12 je Glied), nur die Nachbarn sind in Reichweite."""
    return from_points(vehicles=[(12 * i, 50) for i in range(k + 1)], orders=[(12 * i + 10, 50) for i in range(k + 1)], reach=10)


NETS = {
    "steal": lambda: steal_2x2(),
    "p4": lambda: p4_chain(1),
    "paths": lambda: p4_chain(3),
    "chain": lambda: long_chain(6),
}


def build(net, n, m, reach, ballung, seed):
    """Karte zu den Einstellungen; feste Karten ignorieren die Zufallsparameter."""
    if net in NETS:
        return NETS[net]()
    return generate(n, m, reach, ballung, seed)
