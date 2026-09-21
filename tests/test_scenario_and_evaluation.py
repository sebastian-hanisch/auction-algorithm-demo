"""Auswertung: Einordnung, Verdict, Vergleichstabelle, Verteilung über viele Karten, Aufwandstabelle, Schrittweiten-Sweep, Reichweite-Sweep, Aufwand-Experiment, Strafe."""

import dataclasses

import pytest

import au_constants as C
import au_evaluation as ev
from au_algorithm import auction
from au_scenario import build, generate


def test_build_fixed_nets_ignore_random_parameters():
    a, b = build("chain", 5, 5, 99, 100, 123), build("chain", 30, 30, 10, 0, 1)
    assert a.vehicles == b.vehicles and a.n == 7
    assert build("steal", 1, 1, 1, 1, 1).n == 2 and build("p4", 1, 1, 1, 1, 1).n == 2 and build("paths", 1, 1, 1, 1, 1).n == 6


def test_eps_units():
    assert ev.eps_units(0, 20) == 1 and ev.eps_units(5, 20) == 105 and ev.eps_units(1, 2) == 3


@pytest.mark.parametrize("net,count,cost", [("steal", 2, 10), ("p4", 2, 20), ("chain", 7, 70)])
def test_verdict_of_the_fixed_nets(net, count, cost):
    for bid in C.BID_LABELS:
        for eps in C.EPS_LABELS:
            level, code, d = ev.verdict(ev.analyse(build(net, 20, 20, 40, 0, 2), bid, eps, 0))
            assert (level, code) == ("success", ev.OPTIMAL) and (d["count"], d["cost"]) == (count, cost) and d["gap"] == 0
            assert d["cert"]["all_ok"] and d["cert"]["exact_guaranteed"] and d["bids"] > 0 and d["scans"] > 0


def test_verdict_near_and_mismatch_and_none():
    sc = generate(20, 20, 40, 0, 68)
    level, code, d = ev.verdict(ev.analyse(sc, "sequential", "fixed", 5))
    assert (level, code) == ("warning", ev.NEAR) and 0 < d["gap"] <= d["bound_min"] and not d["cert"]["exact_guaranteed"]
    a = ev.analyse(sc)
    forged = dataclasses.replace(a.result, cost=a.result.cost - 1)                                 # billiger als das Optimum: unmöglich, also Fehler
    assert ev.verdict(dataclasses.replace(a, result=forged))[1] == ev.MISMATCH
    worse = dataclasses.replace(a.result, cost=a.result.cost + 10 ** 5)
    assert ev.verdict(dataclasses.replace(a, result=worse))[1] == ev.MISMATCH                        # weit jenseits der Schranke
    empty = next(s for s in (generate(3, 3, 10, 0, k) for k in range(200)) if not s.feasible.any())
    assert ev.verdict(ev.analyse(empty))[:2] == ("info", ev.NONE)


def test_penalised_gap_is_never_negative_for_any_step():
    for seed in range(30):
        sc = generate(12, 9, 40, 0, seed)
        h = ev.hungarian(sc, record=False)
        for epsv in (0, 1, 5, 20):
            r = auction(sc, "sequential", "fixed", ev.eps_units(epsv, sc.n), record=False)
            assert ev.penalised_gap(r, h, sc.n) >= 0


def test_compare_table_rows():
    rows = ev.compare_table(ev.analyse(generate(20, 20, 40, 0, 165)))
    assert [r["label"] for r in rows][0] == "Ungarische Methode (zentral)" and len(rows) == 5
    assert all((r["count"], r["cost"]) == (20, 380) for r in rows)
    assert rows[0]["bids"] is None and rows[1]["rounds"] is None and rows[2]["rounds"] > 0 and rows[2]["wasted"] > 0 and rows[3]["wasted"] is None


def test_distribution_fields_and_shares():
    d = ev.distribution(20, 20, 40, 0)
    assert d["n_seeds"] == 100 == d["n_valid"] and set(d["variants"]) == set(ev.VARIANTS)
    assert all(len(v["bids"]) == 100 for v in d["variants"].values()) and d["hung_scanned_mean"] > 0 and 0 <= d["fixed_worse_share"] <= 1


def test_distribution_is_repeatable():
    assert ev.distribution(15, 15, 40, 25) == ev.distribution(15, 15, 40, 25)


def test_distribution_without_feasible_pairs():
    bad = tuple(s for s in range(60) if not generate(3, 3, 10, 0, s).feasible.any())
    assert len(bad) >= 3
    assert ev.distribution(3, 3, 10, 0, seeds=bad)["n_valid"] == 0 and ev.effort_table(3, 3, 10, 0, seeds=bad) == []


def test_effort_table_rows():
    rows = ev.effort_table(20, 20, 40, 0)
    assert [r["label"] for r in rows][0] == "Ungarische Methode (zentral)" and len(rows) == 5 and all(r["optimal"] == 1.0 for r in rows)
    assert rows[0]["bids"] is None and rows[3]["rounds"] is None and rows[2]["rounds"] > 0


def test_cell_rows_are_integers():
    rows = ev.cell_rows(10, 10, 40, 0, tuple(C.SWEEP_SEEDS[:5]))
    for row in rows:
        for key, val in row.items():
            assert all(isinstance(v, int) for v in (val if isinstance(val, tuple) else (val,))), key


def test_coarse_rows_shape_and_bound():
    rows = ev.coarse_rows(10, 10, 40, 0, 5, tuple(C.SWEEP_SEEDS[:10]))
    assert len(rows) == 10 and all(0 <= r[3] <= r[4] for r in rows)


def test_eps_sweep_reach_sweep_scaling_and_penalty_rows():
    rows, ref = ev.eps_sweep(10, 10, 40, 0, seeds=C.SWEEP_SEEDS[:10], values=(0, 5))
    assert [r["eps"] for r in rows] == [0, 5] and ref > 0 and rows[0]["optimal"] == 1.0
    sweep = ev.reach_sweep(15, 15, 0, values=(10, 40, 150), seeds=C.SWEEP_SEEDS[:10])
    assert [r["x"] for r in sweep] == [10, 40, 150] and all(r["scaled_bids"] > 0 for r in sweep)
    scal = ev.scaling(ns=(10, 20), seeds=C.SCALE_SEEDS[:3])
    assert [r["n"] for r in scal] == [10, 20] and [r["reach"] for r in scal] == [57, 40] and all(r["fixed_scans"] is not None for r in scal)
    assert ev.scaling(ns=(160,), seeds=C.SCALE_SEEDS[:1])[0]["fixed_scans"] is None                    # über der Rechengrenze wird die feste Auktion nicht gerechnet
    pen = ev.penalty_experiment(10, 10, 40, 0, seeds=C.SWEEP_SEEDS[:5], factors=(1, 2))
    assert [r["factor"] for r in pen] == [1, 2] and pen[1]["fixed_bids"] >= pen[0]["fixed_bids"]


def test_progress_series():
    sc = generate(20, 20, 40, 0, 165)
    res = auction(sc)
    unassigned, price_sum, phases = ev.progress_series(res, sc.n, sc.m)
    assert len(unassigned) == len(price_sum) == len(res.events) + 1 and unassigned[0] == 20 and unassigned[-1] == 0
    assert len(phases) == res.counters["phases"] and price_sum[0] == 0 and max(price_sum) >= price_sum[-1] > 0
