"""Eigenschaftstests (metamorph) für den Auktionsalgorithmus: Aussagen über die lexikografische Zuordnung (erst möglichst viele Paare, dann minimale Kosten), die gelten MÜSSEN,
ohne dass man das Ergebnis vorher kennt. Das vorhandene Orakel (test_oracle_auction.py) vergleicht gegen Brute Force, die Invarianten je Ereignis stehen in test_algorithm.py.
Hier wird die EINGABE verändert und die Beziehung der Ausgaben geprüft (feste Seeds, je Eigenschaft 30 bis 60 Zufallsmatrizen, auch mit Gleichständen, plus Randfälle):

 A1 Umnummerieren der Fahrzeuge und Aufträge ändert (Paarzahl, Kosten) nicht, in allen vier Varianten (Gebotsart x Schrittweite), und alle Varianten stimmen miteinander überein.
 A2 Spiegelung (Fahrzeuge und Aufträge tauschen die Rollen = transponierte Matrizen): gleiche (Paarzahl, Kosten).
 A3 Kosten mit ganzem c >= 1 multipliziert: dieselbe Paarzahl, Kosten mal c; jeder mögliche Kostenwert um d >= 0 verschoben: Paarzahl gleich, Kosten + d * Paarzahl
    (alle größten Matchings haben gleich viele Paare). Beides nur zulässig, weil das Ziel lexikografisch ist.
 A4 Dominanz: eine zusätzliche mögliche Kante, ein zusätzliches Fahrzeug oder ein zusätzlicher Auftrag verschlechtern (Paarzahl, -Kosten) lexikografisch nie; eine entfernte Kante
    verbessert sie nie; eine gesenkte Einzelkost senkt die Gesamtkosten um höchstens die Senkung und erhöht sie nie.
 A5 Schwache Dualität in JEDEM Zwischenstand des Protokolls: D(p) = Summe der Preise + Summe der besten Werte der Fahrzeuge >= Wert des Optimums (Preise sind >= 0);
    im Endstand liegt D(p) höchstens n * eps über dem Optimum (Bertsekas: Preise liegen innerhalb n * eps der Dual-Optimallösung).
 A6 Grobes eps: auch bei gleichzeitigem Bieten, bei den festen Schrittweiten eps = 1, 3, S, 3 S (Einheiten 1/S Minute, S = n + 1) liegen die Kosten einschließlich
    Verzichtsstrafe höchstens n * eps / S Minuten über dem Optimum.
 A7 Ein größeres M (Strafe für Verzicht) ändert das exakte Ergebnis nicht (M ist bewiesen sicher, jedes größere auch).
 A8 Determinismus: dieselbe Eingabe, dieselben Paare, Preise und Ereignisse.
 A9 Randfälle: keine Fahrzeuge, keine Aufträge, keine möglichen Paare, alle Kosten 0, ein Fahrzeug, ein Auftrag, viele Gleichstände.
"""

import functools
import random

import numpy as np

from au_algorithm import all_states, auction, certificate
from au_scenario import Scenario

MODES = [("sequential", "scaled"), ("simultaneous", "scaled"), ("sequential", "fixed"), ("simultaneous", "fixed")]


# --- Hilfen ----------------------------------------------------------------------------------------------------------------------------------

def _sc(cost, feas):
    feas = np.array(feas, dtype=bool)
    cost = np.array(cost, dtype=np.int64).reshape(feas.shape)
    n, m = feas.shape
    return Scenario(((0, 0),) * n, ((0, 0),) * m, 0, cost, feas)


def _random(rng, nmax=6, mmax=6, cmaxes=(0, 1, 3, 10, 40)):
    n, m = rng.randint(1, nmax), rng.randint(1, mmax)
    cmax = rng.choice(cmaxes)
    p = rng.choice([0.3, 0.6, 1.0])
    cost = [[rng.randint(0, cmax) for _ in range(m)] for _ in range(n)]
    feas = [[rng.random() < p for _ in range(m)] for _ in range(n)]
    return cost, feas


def _opt(cost, feas):
    """Lexikografisches Optimum (Paarzahl, Kosten) durch Aufzählen mit Gedächtnis (Brute Force)."""
    n = len(cost)
    m = len(cost[0]) if n else 0

    @functools.lru_cache(maxsize=None)
    def rec(i, used):
        if i == n:
            return (0, 0)
        best = rec(i + 1, used)
        for j in range(m):
            if feas[i][j] and not (used >> j) & 1:
                c, k = rec(i + 1, used | (1 << j))
                cand = (c + 1, k - cost[i][j])            # Kosten negativ, damit das Maximum "mehr Paare, dann billiger" ist
                best = max(best, cand)
        return best

    c, negk = rec(0, 0)
    return c, -negk


def _run(cost, feas, mode, **kw):
    res = auction(_sc(cost, feas), *mode, 1, record=False, **kw)
    return res.count, res.cost


def _key(r):
    """Lexikografische Güte: mehr Paare, dann billiger."""
    return r[0], -r[1]


# --- A1 Umnummerieren ------------------------------------------------------------------------------------------------------------------------

def test_a1_relabelling_vehicles_and_orders_keeps_count_and_cost_in_all_variants():
    rng = random.Random(201)
    for _ in range(40):
        cost, feas = _random(rng)
        n, m = len(cost), len(cost[0])
        qv, qo = rng.sample(range(n), n), rng.sample(range(m), m)
        cost2 = [[0] * m for _ in range(n)]
        feas2 = [[False] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                cost2[qv[i]][qo[j]], feas2[qv[i]][qo[j]] = cost[i][j], feas[i][j]
        want = _opt(cost, feas)
        for mode in MODES:
            assert _run(cost, feas, mode) == want == _run(cost2, feas2, mode)


# --- A2 Spiegelung ---------------------------------------------------------------------------------------------------------------------------

def test_a2_swapping_vehicles_and_orders_keeps_count_and_cost():
    rng = random.Random(202)
    for _ in range(40):
        cost, feas = _random(rng)
        ct, ft = np.array(cost).T.tolist(), np.array(feas).T.tolist()
        for mode in MODES:
            assert _run(cost, feas, mode) == _run(ct, ft, mode)


# --- A3 Skalierung und Verschiebung der Kosten -----------------------------------------------------------------------------------------------

def test_a3_scaling_and_shifting_the_costs_acts_linearly_on_the_optimum():
    rng = random.Random(203)
    for _ in range(40):
        cost, feas = _random(rng, cmaxes=(1, 3, 10))
        n, m = len(cost), len(cost[0])
        base = _run(cost, feas, MODES[0])
        c, d = rng.randint(1, 5), rng.randint(0, 9)
        scaled = [[c * x for x in row] for row in cost]
        shifted = [[x + d for x in row] for row in cost]
        for mode in (MODES[0], MODES[1], MODES[3]):
            cnt, tot = _run(scaled, feas, mode)
            assert (cnt, tot) == (base[0], c * base[1])
            cnt, tot = _run(shifted, feas, mode)
            assert (cnt, tot) == (base[0], base[1] + d * base[0])


# --- A4 Dominanz -----------------------------------------------------------------------------------------------------------------------------

def test_a4_extra_options_never_hurt_and_removing_options_never_helps():
    rng = random.Random(204)
    for _ in range(50):
        cost, feas = _random(rng, 5, 5, cmaxes=(3, 10, 40))
        n, m = len(cost), len(cost[0])
        base = _run(cost, feas, MODES[0])
        i, j = rng.randrange(n), rng.randrange(m)
        other = [list(r) for r in feas]
        other[i][j] = not other[i][j]
        new = _run(cost, other, MODES[0])
        assert (_key(new) >= _key(base)) if other[i][j] else (_key(new) <= _key(base))
        row_c, row_f = [rng.randint(0, 40) for _ in range(m)], [rng.random() < 0.5 for _ in range(m)]
        assert _key(_run(cost + [row_c], feas + [row_f], MODES[0])) >= _key(base)             # zusätzliches Fahrzeug
        col_c, col_f = [rng.randint(0, 40) for _ in range(n)], [rng.random() < 0.5 for _ in range(n)]
        assert _key(_run([r + [col_c[k]] for k, r in enumerate(cost)], [r + [col_f[k]] for k, r in enumerate(feas)], MODES[1])) >= _key(base)    # zusätzlicher Auftrag
        if feas[i][j]:
            cut = rng.randint(1, max(1, cost[i][j])) if cost[i][j] > 0 else 0
            cheaper = [list(r) for r in cost]
            cheaper[i][j] -= cut
            cnt, tot = _run(cheaper, feas, MODES[1])
            assert cnt == base[0] and base[1] - cut <= tot <= base[1]                          # Senkung einer Einzelkost: höchstens um die Senkung billiger, nie teurer


# --- A5 Schwache Dualität in jedem Zwischenstand ---------------------------------------------------------------------------------------------

def _dual(res, p):
    W = -res.scale * res.penalty
    return sum(p) + sum(max([W] + [res.a[i][j] - p[j] for j in res.a[i]]) for i in range(len(res.a)))


def test_a5_weak_duality_holds_in_every_state_and_the_final_prices_are_within_n_eps_of_the_dual_optimum():
    rng = random.Random(205)
    states_checked = 0
    for _ in range(30):
        cost, feas = _random(rng, 5, 5, cmaxes=(3, 10))
        sc = _sc(cost, feas)
        n, m = sc.n, sc.m
        cnt, tot = _opt(cost, feas)
        for mode in MODES:
            res = auction(sc, *mode, 1)
            v_opt = -res.scale * (tot + res.penalty * (n - cnt))              # Wert des Optimums in den Einheiten der Auktion (Verzichte mit W)
            for st in all_states(res, n, m):
                assert all(x >= 0 for x in st["p"])
                assert _dual(res, st["p"]) >= v_opt                             # Schranke gilt bei jedem Zwischenstand
                states_checked += 1
            gap = _dual(res, list(res.prices)) - v_opt
            assert 0 <= gap <= n * res.eps_final
            assert certificate(sc, res)["all_ok"]
    assert states_checked > 1500


# --- A6 grobes eps ---------------------------------------------------------------------------------------------------------------------------

def test_a6_coarse_eps_stays_within_n_eps_of_the_optimum_for_both_bidding_modes():
    rng = random.Random(206)
    for _ in range(40):
        cost, feas = _random(rng, 5, 5, cmaxes=(3, 10))
        sc = _sc(cost, feas)
        n = sc.n
        cnt, tot = _opt(cost, feas)
        S = n + 1
        for bid in ("sequential", "simultaneous"):
            for e in (1, 3, S, 3 * S):
                res = auction(sc, bid, "fixed", e, record=False)
                opt_pen = tot + res.penalty * (n - cnt)
                gap_units = S * (res.penalised - opt_pen)                        # Lücke in 1/S Minute
                assert 0 <= gap_units <= n * e


# --- A7 größeres M ---------------------------------------------------------------------------------------------------------------------------

def test_a7_a_larger_penalty_for_idling_does_not_change_the_exact_result():
    rng = random.Random(207)
    for _ in range(30):
        cost, feas = _random(rng, 5, 5, cmaxes=(1, 3, 10))
        base = _run(cost, feas, MODES[0])
        assert base == _opt(cost, feas)
        n, m = len(cost), len(cost[0])
        M0 = min(n, m) * (max([cost[i][j] for i in range(n) for j in range(m) if feas[i][j]] or [1]) or 1) + 1
        for mult in (2, 3):
            for mode in (MODES[0], MODES[1], MODES[2]):
                assert _run(cost, feas, mode, penalty=mult * M0) == base


# --- A8 Determinismus ------------------------------------------------------------------------------------------------------------------------

def test_a8_same_input_same_output():
    rng = random.Random(208)
    for _ in range(15):
        cost, feas = _random(rng, 6, 6, cmaxes=(3, 10))
        sc = _sc(cost, feas)
        for mode in MODES:
            a, b = auction(sc, *mode, 1), auction(sc, *mode, 1)
            assert (a.pairs, a.prices, a.events, a.counters["bids"]) == (b.pairs, b.prices, b.events, b.counters["bids"])


# --- A9 Randfälle ----------------------------------------------------------------------------------------------------------------------------

def test_a9_edge_cases():
    for mode in MODES:
        assert _run(np.zeros((0, 0), dtype=int), np.zeros((0, 0), dtype=bool), mode) == (0, 0)
        assert _run(np.zeros((3, 0), dtype=int), np.zeros((3, 0), dtype=bool), mode) == (0, 0)
        assert _run(np.zeros((0, 3), dtype=int), np.zeros((0, 3), dtype=bool), mode) == (0, 0)
        assert _run([[5, 5], [5, 5]], [[False, False], [False, False]], mode) == (0, 0)             # kein Paar möglich
        assert _run([[0, 0, 0], [0, 0, 0]], [[True] * 3, [True] * 3], mode) == (2, 0)               # alle Kosten 0
        assert _run([[7]], [[True]], mode) == (1, 7)
        assert _run([[9, 4, 6]], [[True, True, True]], mode) == (1, 4)                              # ein Fahrzeug nimmt den billigsten Auftrag
        assert _run([[9], [4], [6]], [[True], [True], [True]], mode) == (1, 4)                      # ein Auftrag geht an das billigste Fahrzeug
        assert _run([[3, 3, 3], [3, 3, 3], [3, 3, 3]], [[True] * 3] * 3, mode) == (3, 9)            # lauter Gleichstände
        assert _run([[1, 0], [0, 1]], [[True, False], [False, True]], mode) == (2, 2)               # nur die Diagonale ist möglich, die billigen Kosten stehen daneben
