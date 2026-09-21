"""Auktionsalgorithmus: Wächter für die kopierten Vorgänger, Handrechnung, Invarianten je Ereignis, Orakel (Ungarische Methode, scipy, networkx,
Brute Force), Zertifikat mit Negativtest, Determinismus, Negativkontrollen (was ohne Preissenkung, ohne Verzicht-Option, mit falschem M oder S passiert)."""

import itertools

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import au_constants as C
from au_algorithm import all_states, apply_event, auction, certificate, eps_plan, initial_state, state_at
from au_hungarian import hungarian
from au_scenario import SplitMix64, generate, long_chain, p4_chain, steal_2x2

MODES = [("sequential", "scaled"), ("simultaneous", "scaled"), ("sequential", "fixed"), ("simultaneous", "fixed")]


def exact(sc, res, h=None):
    h = h or hungarian(sc, record=False)
    return (res.count, res.cost) == (h.count, h.cost)


# --- Wächter für die kopierten Dateien der Vorgänger ---------------------------------------------------------------------------------

def test_copied_predecessor_files_reproduce_their_numbers():
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(2)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4]                # Referenzvektor von Vigna
    sc = generate(20, 20, 40, 0, 2)                                          # Seed-2-Karte der Vorgänger: Ungarische Methode 20 Paare, 316 Minuten
    h = hungarian(sc, record=False)
    assert (h.count, h.cost, len(h.rounds)) == (20, 316, 20)


# --- Handrechnung -----------------------------------------------------------------------------------------------------------------------

def test_steal_2x2_by_hand_with_fixed_eps():
    """Kosten [[4, 5], [5, 14]], n = m = 2: S = 3, M = min(2, 2)*14 + 1 = 29, Verzicht W = -87, eps = 1 (Drittelminute).
    F1 (Index 0): Werte -12 (A1) und -15 (A2), zweitbester max(-15, W) = -15: Preis A1 = 0 + (-12 + 15) + 1 = 4.
    F2: A1 kostet mit Preis -15 - 4 = -19, A2 -42: Preis A1 = 4 + (-19 + 42) + 1 = 28, F1 wird verdrängt.
    F1: A1 = -12 - 28 = -40, A2 = -15: Preis A2 = 0 + (-15 + 40) + 1 = 26. Ergebnis F1-A2 (5) und F2-A1 (5) = 10 Minuten."""
    res = auction(steal_2x2(), "sequential", "fixed", 1)
    assert (res.scale, res.penalty) == (3, 29)
    assert [e[:5] for e in res.events if e[0] == "bid"] == [("bid", 0, 0, 0, 4), ("bid", 1, 0, 4, 28), ("bid", 0, 1, 0, 26)]
    assert res.pairs == ((0, 1), (1, 0)) and res.cost == 10 and res.prices == (28, 26) and res.bids == 3


def test_p4_and_chain_fixed_runs_by_hand():
    """Pfad aus vier Punkten (Kosten [[10, -], [2, 10]], S = 3, M = 21): F1 bietet auf A1 (Wert -30, zweitbester W = -63): Preis 0 + 33 + 1 = 34.
    F2 sieht A1 mit -6 - 34 = -40 und A2 mit -30: Preis A2 = 0 + (-30 + 40) + 1 = 11. Ergebnis F1-A1, F2-A2 = 20 Minuten."""
    res = auction(p4_chain(1), "sequential", "fixed", 1)
    assert res.pairs == ((0, 0), (1, 1)) and res.cost == 20 and res.prices == (34, 11) and res.bids == 2
    chain = auction(long_chain(6), "sequential", "fixed", 1)
    assert chain.cost == 70 and chain.count == 7 and chain.bids == 7 and chain.prices == (489, 426, 363, 300, 237, 174, 111) and (chain.scale, chain.penalty) == (8, 71)


def test_scaled_run_on_the_tiny_maps_is_pinned():
    steal, chain = auction(steal_2x2()), auction(long_chain(6))
    assert (steal.bids, steal.counters["phases"], steal.prices, steal.cost) == (5, 3, (71, 68), 10)
    assert (chain.bids, chain.counters["phases"], chain.prices, chain.cost) == (20, 4, (487, 424, 361, 298, 235, 172, 109), 70)


def test_eps_plan():
    assert eps_plan(21, 40, "scaled") == (840, 168, 33, 6, 1) and eps_plan(21, 40, "fixed", 105) == (105,) and eps_plan(3, 1, "scaled") == (3, 1)


# --- Invarianten je Ereignis --------------------------------------------------------------------------------------------------------------

def _best(res, i, p):
    return max([-res.scale * res.penalty] + [res.a[i][j] - p[j] for j in res.a[i]])


def _check_state(res, st, eps, phase_end):
    n, m = len(st["assign"]), len(st["owner"])
    W = -res.scale * res.penalty
    assert all(x >= 0 for x in st["p"])
    used = [j for j in st["assign"] if j >= 0]
    assert len(used) == len(set(used))
    for j, i in enumerate(st["owner"]):
        assert i < 0 or st["assign"][i] == j
    for i in range(n):
        j = st["assign"][i]
        if j >= 0:
            assert st["owner"][j] == i
            assert res.a[i][j] - st["p"][j] >= _best(res, i, st["p"]) - eps           # eps-Komplementarität der zugeordneten Fahrzeuge
        elif phase_end:
            assert st["idle"][i] and W >= _best(res, i, st["p"]) - eps                 # am Phasenende: Verzichter liegen höchstens eps unter ihrer besten Wahl
    if phase_end:
        assert all(st["p"][j] == 0 for j in range(m) if st["owner"][j] < 0)            # unversorgte Aufträge kosten nichts


@pytest.mark.parametrize("mode", MODES)
def test_invariants_after_every_event(mode):
    checked = 0
    for n, m, R, seed in ((6, 6, 40, 1), (5, 8, 40, 2), (8, 5, 40, 3), (7, 7, 15, 4), (9, 9, 60, 5), (10, 10, 40, 6)):
        sc = generate(n, m, R, 0, seed)
        res = auction(sc, *mode, 1)
        states = all_states(res, n, m)
        prev_p, prev_phase = [0] * m, None
        for k, (e, st) in enumerate(zip(res.events, states[1:]), start=1):
            if e[0] == "phase":
                prev_phase, eps = e[2], e[1]
                continue
            nxt_phase_end = k == len(res.events) or res.events[k][0] == "phase"
            _check_state(res, st, eps, phase_end=nxt_phase_end)
            changed = [j for j in range(m) if st["p"][j] != prev_p[j]]
            if e[0] == "bid":
                assert changed == ([e[2]] if e[8] else [])                            # ein Gebot ändert genau einen Preis, und nur nach oben
                assert all(st["p"][j] > prev_p[j] for j in changed)
            elif e[0] == "lower":
                assert changed in ([], [e[1]]) and all(st["p"][j] < prev_p[j] for j in changed)   # Preise fallen nur im Preissenkungs-Schritt
            else:
                assert not changed
            if e[0] == "bid" and e[8]:
                assert e[4] >= e[3] + eps                                              # jedes Gebot hebt den Preis um mindestens eps
            prev_p = list(st["p"])
            checked += 1
        assert states[-1]["assign"] == [next((j for i2, j in res.pairs if i2 == i), -1) for i in range(n)]
    assert checked > 100


def test_replay_of_the_log_never_lowers_a_price_outside_reverse_steps():
    sc = generate(20, 20, 40, 0, 165)
    res = auction(sc)
    st = initial_state(sc.n, sc.m)
    for e in res.events:
        before = list(st["p"])
        apply_event(st, e)
        if e[0] != "lower":
            assert all(a >= b for a, b in zip(st["p"], before))
        else:
            assert e[3] < e[2]                                                         # eine Preissenkung senkt wirklich


def test_replay_matches_the_final_result():
    sc = generate(20, 20, 40, 0, 165)
    res = auction(sc, "simultaneous", "scaled")
    st = state_at(res, sc.n, sc.m, len(res.events))
    assert tuple(st["p"]) == res.prices and tuple(i for i in st["assign"]) == tuple(next((j for i2, j in res.pairs if i2 == i), -1) for i in range(sc.n))


# --- Orakel -----------------------------------------------------------------------------------------------------------------------------

def brute_force(sc):
    """Größtmögliche Paarzahl, darunter kleinste Kosten, durch Aufzählen aller Matchings (nur für winzige Karten)."""
    best = (-1, 0)

    def rec(i, used, count, cost):
        nonlocal best
        if i == sc.n:
            if count > best[0] or (count == best[0] and cost < best[1]):
                best = (count, cost)
            return
        rec(i + 1, used, count, cost)
        for j in range(sc.m):
            if sc.feasible[i, j] and j not in used:
                rec(i + 1, used | {j}, count + 1, cost + int(sc.cost[i, j]))

    rec(0, frozenset(), 0, 0)
    return best


def scipy_reference(sc):
    big = 10 ** 6
    mat = np.where(sc.feasible, sc.cost, big).astype(np.int64)
    r, c = linear_sum_assignment(mat)
    pairs = [(i, j) for i, j in zip(r, c) if sc.feasible[i, j]]
    return len(pairs), int(sum(sc.cost[i, j] for i, j in pairs))


@pytest.mark.parametrize("mode", MODES)
def test_all_modes_match_brute_force_on_tiny_maps(mode):
    for seed in range(60):
        for n, m in ((3, 3), (4, 3), (3, 5), (5, 5)):
            sc = generate(n, m, 30 + 5 * (seed % 5), 0, seed)
            res = auction(sc, *mode, 1, record=False)
            assert (res.count, res.cost) == brute_force(sc), (n, m, seed)


@pytest.mark.parametrize("mode", MODES)
def test_all_modes_match_hungarian_scipy_and_networkx_on_larger_maps(mode):
    import networkx as nx
    for n, m, R, seed in ((12, 12, 40, 7), (12, 12, 10, 8), (15, 9, 40, 9), (9, 15, 40, 10), (14, 14, 150, 11), (16, 16, 40, 12), (12, 12, 40, 13)):
        sc = generate(n, m, R, 0, seed)
        res = auction(sc, *mode, 1, record=False)
        assert exact(sc, res) and (res.count, res.cost) == scipy_reference(sc)
        g = nx.Graph()
        big = 1000
        g.add_nodes_from([("v", i) for i in range(n)] + [("o", j) for j in range(m)])
        g.add_weighted_edges_from([(("v", i), ("o", j), big - int(sc.cost[i, j])) for i in range(n) for j in range(m) if sc.feasible[i, j]])
        mate = nx.max_weight_matching(g, maxcardinality=True)
        cost = sum(int(sc.cost[next(x[1] for x in e if x[0] == "v"), next(x[1] for x in e if x[0] == "o")]) for e in mate)
        assert (res.count, res.cost) == (len(mate), cost)


def test_random_maps_including_isolated_vehicles_and_no_edges():
    empty = generate(3, 3, 10, 0, next(s for s in range(500) if not generate(3, 3, 10, 0, s).feasible.any()))
    for mode in MODES:
        res = auction(empty, *mode, 1)
        assert res.count == 0 and res.cost == 0 and res.n_idle == 3 and certificate(empty, res)["all_ok"]
    rng = SplitMix64(7)
    for _ in range(150):
        n, m, R, seed = 3 + rng.below(6), 3 + rng.below(6), 10 + 5 * rng.below(9), rng.below(10 ** 6)
        sc = generate(n, m, R, 0, seed)
        h = hungarian(sc, record=False)
        for mode in MODES:
            assert exact(sc, auction(sc, *mode, 1, record=False), h), (n, m, R, seed, mode)


# --- Zertifikat ---------------------------------------------------------------------------------------------------------------------------

def test_certificate_is_independent_of_the_hungarian_method_and_proves_exactness():
    for seed in range(100000, 100020):
        sc = generate(20, 20, 40, 0, seed)
        for mode in MODES:
            res = auction(sc, *mode, 1)
            c = certificate(sc, res)
            assert c["all_ok"] and c["exact_guaranteed"] and c["g"] < res.scale and c["g"] <= c["bound"]


def test_certificate_for_a_coarse_run_gives_only_a_bound():
    sc = generate(20, 20, 40, 0, 68)
    res = auction(sc, "sequential", "fixed", 21 * 5)
    c = certificate(sc, res)
    assert c["all_ok"] and not c["exact_guaranteed"] and c["g"] <= c["bound"] == 20 * 105 and c["gap_minutes_bound"] <= 100
    h = hungarian(sc, record=False)
    assert 0 <= res.penalised - (h.cost + res.penalty * (sc.n - h.count)) <= c["gap_minutes_bound"]


def test_certificate_rejects_a_tampered_result():
    """Ein Preis um 3 Minuten erhöht: der Schlupf des Halters übersteigt eps, E1 schlägt fehl - der Beweis fängt einen falschen Preisvektor."""
    import dataclasses
    sc = generate(20, 20, 40, 0, 165)
    res = auction(sc)
    j = res.pairs[0][1]
    bad = dataclasses.replace(res, prices=tuple(p + 3 * res.scale if k == j else p for k, p in enumerate(res.prices)))
    assert certificate(sc, res)["all_ok"] and not certificate(sc, bad)["e1"]
    stale = dataclasses.replace(res, prices=tuple(p + 5 if k not in {j2 for _, j2 in res.pairs} else p for k, p in enumerate(res.prices))) if res.count < sc.m else None
    assert stale is None or not certificate(sc, stale)["e3"]


# --- Determinismus und Aufwandszähler -------------------------------------------------------------------------------------------------------

def test_determinism_and_counters():
    sc = generate(20, 20, 40, 0, 165)
    for mode in MODES:
        a, b = auction(sc, *mode, 1), auction(sc, *mode, 1)
        assert a.events == b.events and a.prices == b.prices and a.counters == b.counters
        c = a.counters
        assert c["bids"] == sum(1 for e in a.events if e[0] == "bid") and c["wasted"] == sum(1 for e in a.events if e[0] == "bid" and not e[8])
        assert c["bids_by_phase"] and sum(c["bids_by_phase"]) == c["bids"] and c["phases"] == len(a.eps_schedule)
        if mode[0] == "sequential":
            assert c["wasted"] == 0 and c["rounds"] == 0
        else:
            assert c["rounds"] == sum(1 for e in a.events if e[0] == "round")
        if mode[1] == "fixed":
            assert c["rev_scans"] == 0 and c["takeovers"] == 0 and c["phases"] == 1                # kein Preissenkungs-Schritt im festen Modus
    assert auction(sc, record=False).counters == auction(sc).counters


# --- Negativkontrollen -----------------------------------------------------------------------------------------------------------------------

def wrong(n, m, R, seeds=C.DIST_SEEDS, **kw):
    bad, capped = 0, 0
    for s in seeds:
        sc = generate(n, m, R, 0, s)
        res = auction(sc, record=False, **kw)
        capped += res.capped
        bad += (res.count, res.cost) != (lambda h: (h.count, h.cost))(hungarian(sc, record=False))
    return bad, capped


def test_scaling_without_the_price_reduction_step_is_wrong():
    """Naive eps-Skalierung mit behaltenen Preisen (ohne Preissenkung am Phasenende) ist falsch: unversorgte Aufträge behalten alte Preise."""
    assert wrong(20, 20, 40, reverse=False)[0] == 42
    assert wrong(20, 10, 40, reverse=False)[0] == 100 and wrong(10, 20, 40, reverse=False)[0] == 100 and wrong(20, 20, 10, reverse=False)[0] == 100


def test_keeping_idle_vehicles_after_a_price_reduction_is_wrong_on_small_maps():
    bad = 0
    for seed in range(3000):
        rng = SplitMix64(seed)
        n, m, R, s = 3 + rng.below(4), 3 + rng.below(4), 10 + 5 * rng.below(9), rng.below(10 ** 6)
        sc = generate(n, m, R, 0, s)
        h = hungarian(sc, record=False)
        bad += (lambda r: (r.count, r.cost) != (h.count, h.cost))(auction(sc, reset_idle=False, record=False))
    assert bad > 0


def test_a_too_small_penalty_loses_pairs():
    """Lange Kette: M = 58 gibt alle 7 Paare (das siebte kostet 58 mehr), M = 57 nur 6."""
    assert [auction(long_chain(6), penalty=M, record=False).count for M in (57, 58)] == [6, 7]


def test_without_the_scale_factor_the_result_is_wrong():
    """S = 1 statt n + 1: die Lücke n*eps ist nicht mehr kleiner als eine Kosteneinheit, das Ergebnis ist auf vielen Karten falsch."""
    assert wrong(20, 20, 40, eps_mode="fixed", scale=1)[0] == 25 and wrong(20, 20, 40, scale=1)[0] == 31


def test_a_naive_auction_without_the_outside_option_never_ends_when_not_everyone_can_be_served():
    """Ohne Verzicht-Option und mit einer Gebotsgrenze: bricht auf 35 von 100 Karten (20 x 20, Reichweite 40) und auf allen 100 mit mehr Fahrzeugen als Aufträgen ab."""
    assert wrong(20, 20, 40, eps_mode="fixed", outside=False, bid_cap=20000)[1] == 35
    assert wrong(20, 10, 40, eps_mode="fixed", outside=False, bid_cap=20000)[1] == 100
