"""Orakel-Regressionstest (Auktion): beliebige, gleichstandsreiche Kostenmatrizen (nicht nur Karten) gegen Brute Force.

Unabhängig vom Code der Demo: das Orakel zählt alle Matchings auf (größte Paarzahl, darunter kleinste Kosten) und prüft
- exaktes ε: alle vier Varianten treffen (Paare, Kosten) genau, die Ungarische Methode ebenfalls;
- grobes ε: Kosten einschließlich Verzichtstrafe liegen höchstens n·ε Minuten über dem Optimum (ε-Komplementarität)."""

import random

import numpy as np

import au_evaluation as ev
from au_algorithm import auction, certificate
from au_hungarian import hungarian
from au_scenario import Scenario

MODES = [("sequential", "scaled"), ("simultaneous", "scaled"), ("sequential", "fixed"), ("simultaneous", "fixed")]


def _scenario(rng):
    n, m = rng.randint(1, 6), rng.randint(1, 6)
    cmax = rng.choice([0, 1, 2, 3, 10, 40])
    p = rng.choice([0.3, 0.6, 1.0])
    cost = np.array([[rng.randint(0, cmax) for _ in range(m)] for _ in range(n)], dtype=np.int64)
    feas = np.array([[rng.random() < p for _ in range(m)] for _ in range(n)], dtype=bool)
    return Scenario(((0, 0),) * n, ((0, 0),) * m, 0, cost, feas)


def _brute(sc):
    best = (-1, 0)

    def rec(i, used, cnt, c):
        nonlocal best
        if i == sc.n:
            if cnt > best[0] or (cnt == best[0] and c < best[1]):
                best = (cnt, c)
            return
        rec(i + 1, used, cnt, c)
        for j in range(sc.m):
            if sc.feasible[i, j] and not (used >> j) & 1:
                rec(i + 1, used | (1 << j), cnt + 1, c + int(sc.cost[i, j]))

    rec(0, 0, 0, 0)
    return best


def test_all_variants_exact_and_coarse_eps_within_bound_on_tie_heavy_matrices():
    rng = random.Random(2026)
    for _ in range(120):
        sc = _scenario(rng)
        opt = _brute(sc)
        h = hungarian(sc, record=False)
        assert (h.count, h.cost) == opt
        for mode in MODES:
            res = auction(sc, *mode, 1, record=False)
            assert (res.count, res.cost) == opt
            assert certificate(sc, res)["all_ok"]
        for epsv in (1, 5, 20):
            res = ev.run_choice(sc, "sequential", "fixed", epsv, record=False)
            gap = res.penalised - (opt[1] + res.penalty * (sc.n - opt[0]))
            assert 0 <= gap <= sc.n * epsv
