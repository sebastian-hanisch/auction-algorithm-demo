"""Literatur-Referenzwerte für den Auktionsalgorithmus (alle Quellen am 2026-10-17 eingesehen).

 L1  Dokumentation der Referenzsoftware SciPy 1.18.0, `scipy.optimize.linear_sum_assignment`,
     https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html, Abschnitt "Examples":
     cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]; Ausgabe col_ind = [1, 0, 2] (Zeilen 0, 1, 2 in dieser Reihenfolge), Gesamtkosten 5.
     Das Optimum ist eindeutig (die anderen fünf Zuordnungen kosten 6, 6, 7, 9 und 11, von Hand gerechnet), also ist die dokumentierte Zuordnung die einzige mögliche.
 L2  Wikipedia-Beispiel (gegengerechnet), "Hungarian algorithm", Abschnitt zum Beispiel (Alice, Bob, Carol; Bad putzen, Böden kehren, Fenster waschen):
     Kosten [[8, 4, 7], [5, 2, 3], [9, 4, 8]] (Zeilen Alice, Bob, Carol; Spalten Bad, Böden, Fenster); genannt werden 15 Euro, "Alice cleans the bathroom, Carol sweeps the floors,
     Bob washes the windows". Wikipedia ist keine Primärquelle; die 15 und die Eindeutigkeit wurden hier durch Aufzählen aller sechs Zuordnungen nachgerechnet (15, 16, 16, 17, 18, 18).
 L3  Bertsekas, "Auction Algorithms" (Encyclopedia of Optimization), MIT-Kopie https://www.mit.edu/~dimitrib/Auction_Encycl.pdf, Abschnitt "Optimality Properties at Termination"
     (Seite 3 der Kopie): (a) der Gesamtnutzen der Endzuordnung liegt höchstens n * eps unter dem Optimum; (b) sind die Nutzen a_ij ganzzahlig und eps < 1/n, ist die Endzuordnung
     optimal; (c) die Endpreise liegen innerhalb n * eps einer Optimallösung des Dualproblems  min  sum_j p_j + sum_i max_j (a_ij - p_j)  (Gleichung (9) der Quelle).
     In der Demo sind die Nutzen -S * Kosten (S = n + 1), eps in Einheiten 1/S Minute, also eps = 1 entspricht 1/(n + 1) < 1/n. Geprüft wird auf quadratischen, vollständigen
     ganzzahligen Matrizen, das Optimum kommt aus SciPy (Referenzsoftware, `linear_sum_assignment`), nicht aus dem Code der Demo.

NICHT als Literaturwert verwendet (Quelle nicht einsehbar): Bertsekas 1988 (Annals of Operations Research 14) und die Berichte von 1979/1987 (nur als Bildscan ohne Text erreichbar), Bertsekas'
Lehrbücher; deshalb gibt es kein Zahlenbeispiel mit Preisfolge aus der Originalarbeit.
"""

import random

import numpy as np
import pytest

from au_algorithm import auction
from au_scenario import Scenario

MODES = [("sequential", "scaled"), ("simultaneous", "scaled"), ("sequential", "fixed"), ("simultaneous", "fixed")]


def _full(cost):
    c = np.array(cost, dtype=np.int64)
    n, m = c.shape
    return Scenario(((0, 0),) * n, ((0, 0),) * m, 0, c, np.ones((n, m), dtype=bool))


# --- L1 SciPy-Dokumentation --------------------------------------------------------------------------------------------------------------------

def test_l1_scipy_documentation_cost_matrix_gives_col_ind_1_0_2_and_total_cost_5():
    sc = _full([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    for mode in MODES:
        res = auction(sc, *mode, 1, record=False)
        assert res.pairs == ((0, 1), (1, 0), (2, 2)) and res.cost == 5 and res.count == 3 and res.n_idle == 0


# --- L2 Wikipedia-Beispiel (gegengerechnet) ----------------------------------------------------------------------------------------------------

def test_l2_wikipedia_hungarian_example_alice_bathroom_carol_floors_bob_windows_for_15():
    sc = _full([[8, 4, 7], [5, 2, 3], [9, 4, 8]])
    for mode in MODES:
        res = auction(sc, *mode, 1, record=False)
        assert res.pairs == ((0, 0), (1, 2), (2, 1)) and res.cost == 15


# --- L3 Bertsekas: n * eps, eps < 1/n, Dualproblem ---------------------------------------------------------------------------------------------

def _dual(res, p):
    """Gleichung (9) der Quelle in den Einheiten der Demo (Nutzen a_ij = -S * Kosten, Verzicht W)."""
    W = -res.scale * res.penalty
    return sum(p) + sum(max([W] + [res.a[i][j] - p[j] for j in res.a[i]]) for i in range(len(res.a)))


def test_l3_bertsekas_final_assignment_is_within_n_eps_and_exact_below_one_over_n_with_dual_within_n_eps():
    lsa = pytest.importorskip("scipy.optimize").linear_sum_assignment
    rng = random.Random(1988)
    exact_checked = coarse_checked = 0
    for _ in range(40):
        n = rng.randint(2, 7)
        cost = np.array([[rng.randint(0, rng.choice([3, 10, 40])) for _ in range(n)] for _ in range(n)])
        r, c = lsa(cost)
        opt = int(cost[r, c].sum())                                             # Optimum der Referenzsoftware
        sc = _full(cost)
        S = n + 1
        for bid in ("sequential", "simultaneous"):
            res = auction(sc, bid, "fixed", 1)                                  # eps = 1 Einheit = 1 / (n + 1) < 1 / n Minute (b): exakt
            assert res.cost == opt and res.count == n
            assert 0 <= _dual(res, list(res.prices)) + S * opt <= n * 1          # (c): Dualwert innerhalb n * eps des Optimalwertes -S * opt
            exact_checked += 1
            for e in (S, 3 * S):                                                # grobes eps = 1 bzw. 3 Minuten: (a)
                coarse = auction(sc, bid, "fixed", e, record=False)
                assert coarse.count == n and 0 <= S * (coarse.cost - opt) <= n * e
                coarse_checked += 1
    assert exact_checked == 80 and coarse_checked == 160
