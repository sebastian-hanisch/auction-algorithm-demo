"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die 100 festen Karten (DIST_SEEDS) belegt.
Alles rechnet mit ganzen Zahlen und einem eigenen Zufallsgenerator - die Werte sind auf jeder Plattform dieselben; die Toleranzen decken nur die Rundung auf die im Text genannten Stellen.
Positive UND negative Aussagen: wo die Auktion gewinnt (dezentral, exakt, ε-Skalierung) und wo sie verliert (feste Schrittweite bei mittlerer Reichweite, mehr Kantensuchen als die Ungarische Methode)."""

import numpy as np
import pytest

import au_constants as C
import au_evaluation as ev
from au_algorithm import auction
from au_greedy import run_rule
from au_scenario import generate, long_chain, steal_2x2

SQ, SIM, FX, FXSIM = ev.VARIANTS


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


@pytest.fixture(scope="module")
def mid():
    return ev.distribution(20, 20, 40, 0)


# --- feste Karten ----------------------------------------------------------------------------------------------------------------------

def test_steal_and_chain_numbers():
    steal, chain = auction(steal_2x2()), auction(long_chain(6))
    fixed = auction(long_chain(6), "sequential", "fixed", 1)
    assert steal.cost == 10 and run_rule(steal_2x2(), "edge").cost == 18                        # "10 Minuten (Greedy zahlt 18)"
    assert steal.bids == 5 and steal.prices == (71, 68) and steal.scale == 3                   # "fünf Gebote", Preise 71 und 68 in Drittelminuten
    assert chain.cost == 70 and chain.bids == 20 and chain.counters["phases"] == 4 and fixed.bids == 7      # "20 Gebote in 4 Phasen, fest nur 7"


# --- Standardkarte (20 x 20, Reichweite 40) -------------------------------------------------------------------------------------------------

def test_every_variant_is_exact_on_every_map_at_every_reach_and_clustering():
    for reach in C.REACH_SWEEP:
        d = ev.distribution(20, 20, reach, 0, C.SWEEP_SEEDS)
        assert all(v["share_optimal"] == 1.0 for v in d["variants"].values()), reach
    for ballung in (50, 100):
        assert all(v["share_optimal"] == 1.0 for v in ev.distribution(20, 20, 40, ballung)["variants"].values())


def test_mid_reach_numbers(mid):
    assert all(v["share_optimal"] == 1.0 for v in mid["variants"].values())                     # "liefert jede der vier Auktionsvarianten das Optimum"
    near(mid["variants"][SQ]["bids_mean"], 209, 0.6)                                            # "im Mittel 209 Gebote"
    near(mid["variants"][SQ]["scans_mean"], 2210, 0.6)                                          # "2 210 durchsuchte Kanten"
    near(mid["hung_scanned_mean"], 1698, 0.6)                                                   # "die Ungarische Methode 1 698"
    near(mid["scaled_vs_hung"], 1.3, 0.005)                                                     # "rund 1,3-mal so viele"
    assert mid["scaled_vs_hung"] > 1                                                            # negativ: die Auktion sucht MEHR Kanten als die Ungarische Methode


def test_the_price_war_is_a_skewed_distribution(mid):
    fx = mid["variants"][FX]
    near(fx["bids_median"], 428, 0.6)                                                           # "Median 428"
    near(fx["bids_mean"], 3036, 0.6)                                                            # "Mittelwert 3 036"
    assert fx["bids_mean"] > 5 * fx["bids_median"] and max(fx["bids"]) > 40000 and 9000 < np.percentile(fx["bids"], 90) < 10000     # "Zehntausende Gebote auf einzelnen Karten"
    near(mid["fixed_worse_share"], 0.69, 0.001)                                                 # "auf 69 von 100 Karten teurer als die Skalierung"
    near(mid["fixed_vs_scaled_bids"], 14.5, 0.05)


def test_the_war_preset_map():
    sc = generate(20, 20, 40, 0, 177)
    assert auction(sc, "sequential", "fixed", 1, record=False).bids == 434 and auction(sc, record=False).bids == 200      # "434 Gebote gegen 200"


def test_simultaneous_bidding_numbers(mid):
    sim, seq = mid["variants"][SIM], mid["variants"][SQ]
    near(sim["rounds_mean"], 97, 0.5)                                                           # "im Mittel 97 Runden"
    near(sim["bids_mean"], 268, 0.5)                                                            # "268 Gebote"
    near(sim["wasted_mean"], 58, 0.5)                                                           # "58 Gebote verlieren an ein höheres Gebot"
    assert sim["rounds_mean"] < seq["bids_mean"] < sim["bids_mean"]                             # weniger Runden als Einzelgebote, aber mehr Gebote insgesamt (negativ)


# --- andere Reichweiten und Größen -------------------------------------------------------------------------------------------------------

def test_short_reach_the_fixed_step_wins():
    d = ev.distribution(20, 20, 10, 0)
    fx, sq = d["variants"][FX], d["variants"][SQ]
    assert round(fx["bids_mean"]) == 9 and round(sq["bids_mean"]) == 19                         # "9 Gebote, die ε-Skalierung 19"
    assert round(fx["scans_mean"]) == 13 and round(sq["scans_mean"]) == 91 and round(d["hung_scanned_mean"]) == 51     # "13 Kanten, 91, Ungarisch 51"
    assert fx["scans_mean"] < d["hung_scanned_mean"] < sq["scans_mean"]                         # hier verliert die Skalierung sogar gegen die Ungarische Methode
    near(d["fixed_worse_share"], 0.01, 0.001)


def test_the_short_reach_preset_map():
    sc = generate(20, 20, 10, 0, 13)
    assert auction(sc, "sequential", "fixed", 1, record=False).bids == 8 and auction(sc, record=False).bids == 18


def test_all_reachable_numbers():
    d = ev.distribution(20, 20, 150, 0)
    assert round(d["variants"][FX]["bids_mean"]) == 497 and round(d["variants"][SQ]["bids_mean"]) == 262      # "497 gegen 262"
    near(d["fixed_vs_scaled_bids"], 1.9, 0.05)
    assert round(d["variants"][SQ]["scans_mean"]) == 5234 and round(d["hung_scanned_mean"]) == 5322        # Skalierung ≈ Ungarisch (0,98)
    near(d["scaled_vs_hung"], 0.98, 0.005)
    assert d["variants"][SQ]["takeovers_mean"] == 0                                             # keine Preissenkungs-Übernahme, wenn alles erreichbar ist


def test_more_vehicles_than_orders():
    d = ev.distribution(20, 10, 40, 0)
    assert round(d["variants"][SQ]["bids_mean"]) == 80 and round(d["variants"][FX]["bids_mean"]) == 1677     # "80 mit Skalierung, 1 677 fest"
    assert d["fixed_worse_share"] == 1.0                                                        # auf allen 100 Karten
    assert d["variants"][SQ]["scans_mean"] > d["hung_scanned_mean"]                             # 893 gegen 614


def test_more_orders_than_vehicles_the_fixed_step_wins():
    d = ev.distribution(10, 20, 40, 0)
    assert round(d["variants"][FX]["bids_mean"]) == 16 and round(d["variants"][SQ]["bids_mean"]) == 56
    near(d["fixed_worse_share"], 0.02, 0.001)                                                   # kein Preiskrieg ohne zu wenige Aufträge


def test_big_map_numbers():
    d = ev.distribution(40, 40, 60, 0)
    assert round(d["variants"][SQ]["scans_mean"]) == 14567 and round(d["hung_scanned_mean"]) == 21124   # hier sucht die Auktion WENIGER Kanten
    near(d["scaled_vs_hung"], 0.69, 0.005)
    assert round(d["variants"][FX]["scans_mean"]) == 83717


# --- Schrittweite ------------------------------------------------------------------------------------------------------------------------------

def test_coarse_epsilon_numbers():
    rows, ref = ev.eps_sweep(20, 20, 40, 0)
    by = {r["eps"]: r for r in rows}
    assert [round(by[k]["bids"]) for k in (0, 1, 2, 5, 10, 20)] == [3036, 701, 465, 251, 152, 92]          # Gebote fallen mit ε
    assert [round(100 * by[k]["optimal"]) for k in (0, 1, 2, 5, 10, 20)] == [100, 75, 39, 9, 2, 0]         # "auf 9 von 100 optimal" bei 5 Minuten
    assert [by[k]["max_gap"] for k in (0, 1, 2, 5, 10, 20)] == [0, 3, 7, 32, 44, 84]                       # größte Lücke, "höchstens 32 statt Schranke 100"
    assert [by[k]["bound"] for k in (1, 2, 5, 10, 20)] == [20, 40, 100, 200, 400]                          # Schranke n·ε
    assert all(r["max_gap"] <= r["bound"] for r in rows) and all(r["pairs_lost"] == 0 for r in rows)      # Schranke nie verletzt; gemessen (nicht bewiesen): keine Paare verloren
    near(ref, 209.43, 0.01)


def test_the_coarse_preset_map():
    sc = generate(20, 20, 40, 0, 68)
    coarse = auction(sc, "sequential", "fixed", 21 * 5, record=False)
    assert coarse.bids == 63 and auction(sc, "sequential", "fixed", 1, record=False).bids == 348           # "63 Gebote statt 348"
    a = ev.analyse(sc, "sequential", "fixed", 5)
    _, code, d = ev.verdict(a)
    assert code == ev.NEAR and d["gap"] == 5 and d["bound_min"] == 100 and d["cost"] == 451 and d["opt_cost"] == 446


# --- Sweeps, Strafe, Größe ------------------------------------------------------------------------------------------------------------------------

def test_reach_sweep_the_war_peaks_at_mid_reach_and_the_scaling_stays_flat():
    rows = ev.reach_sweep(20, 20, 0)
    fixed = {r["x"]: r["fixed_bids"] for r in rows}
    scaled = [r["scaled_bids"] for r in rows]
    assert max(fixed, key=fixed.get) == 40 and fixed[30] > 1000 and fixed[40] > 2000 and fixed[150] < 500     # Gipfel bei Reichweite 40
    assert max(scaled) < 300 and max(scaled) / min(scaled) < 15                                 # ε-Skalierung: bei jeder Reichweite unter 300 Geboten (nicht streng steigend)
    assert fixed[10] < scaled[0]                                                                # bei knapper Reichweite ist fest günstiger


def test_penalty_experiment():
    rows = ev.penalty_experiment(20, 20, 40, 0)
    fx = [r["fixed_bids"] for r in rows]
    assert [round(x) for x in fx] == [2096, 4071, 8025] and [round(r["scaled_bids"]) for r in rows] == [202, 233, 296]
    assert fx[1] / fx[0] > 1.9 and fx[2] / fx[0] > 3.8                                          # "etwa im gleichen Verhältnis wie M"
    assert rows[2]["scaled_bids"] / rows[0]["scaled_bids"] < 1.5                                # ε-Skalierung wächst kaum


def test_scaling_experiment():
    rows = ev.scaling()
    ratios = [r["scaled_scans"] / r["hung_scans"] for r in rows]
    assert all(1.2 < x < 1.7 for x in ratios)                                                   # ein festes Vielfaches (gemessen 1,3 bis 1,6)
    slope = np.polyfit(np.log([r["n"] for r in rows]), np.log([r["scaled_scans"] for r in rows]), 1)[0]
    assert 1.6 < slope < 2.2                                                                    # etwa quadratisch
    by = {r["n"]: r for r in rows}
    assert by[80]["fixed_scans"] > 100 * by[80]["scaled_scans"] and by[160]["fixed_scans"] is None and by[320]["fixed_scans"] is None
    assert by[80]["fixed_scans"] > 4_000_000                                                    # "dort schon Millionen Kantensuchen"
    assert 31_000 < by[80]["scaled_scans"] < 33_000                                            # "rund 32 000 mit Skalierung"
