"""Presets: vollständig, in den Grenzen, und jede Beispielkarte zeigt, was ihr Hilfetext behauptet (typische Ziehung, Median der 100 festen Karten)."""

import numpy as np
import pytest

import au_constants as C
import au_evaluation as ev
import au_presets as P
from au_algorithm import auction
from au_scenario import build, generate

KEYS = set(P.PRESET_KEYS)


def _sc(p):
    return build(p["net"], p["n"], p["m"], p["reach"], p["ballung"], p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 8
    assert all(C.PRESET_HELP[name].strip() for name in C.PRESETS)
    for name, p in C.PRESETS.items():
        assert set(p) == KEYS, name


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_values_are_inside_the_bounds_and_on_the_step_grid(name):
    p = C.PRESETS[name]
    assert p["net"] in C.NETS and p["bid"] in C.BID_LABELS and p["eps"] in C.EPS_LABELS and p["epsv"] in C.EPSV_LABELS
    for key, state_key in P.PRESET_KEYS.items():
        spec = P.SETTING_SPECS[state_key]
        if spec.lo is not None:
            assert spec.lo <= p[key] <= spec.hi, (name, key)
    assert (p["reach"] - C.REACH_MIN) % 5 == 0 and p["ballung"] % 25 == 0
    assert p["eps"] == "fixed" or p["epsv"] == 0                     # ein Schrittweiten-Wert gehört nur zur festen Schrittweite (kein toter Wert im Preset)


def test_setting_specs_have_room_to_move():
    """Ein Regler mit lo == hi würde Streamlit abstürzen lassen."""
    assert all(spec.lo < spec.hi for spec in P.SETTING_SPECS.values() if spec.lo is not None)


def test_presets_use_seeds_outside_the_distribution_set():
    for name, p in C.PRESETS.items():
        assert p["seed"] not in C.DIST_SEEDS, name


def test_the_mid_reach_map_is_the_map_of_the_previous_pieces():
    mid = C.PRESETS["🗺️ Mittlere Reichweite"]
    assert (mid["n"], mid["m"], mid["reach"], mid["ballung"], mid["seed"]) == (20, 20, 40, 0, 165) == (C.DEFAULT_N, C.DEFAULT_M, C.DEFAULT_REACH, C.DEFAULT_BALLUNG, C.DEFAULT_SEED)
    sim = C.PRESETS["🔀 Gleichzeitig"]
    assert all(sim[k] == mid[k] for k in ("net", "n", "m", "reach", "ballung", "seed", "eps")) and sim["bid"] == "simultaneous"


def test_every_preset_is_optimal_or_within_its_bound():
    for name, p in C.PRESETS.items():
        level, code, d = ev.verdict(ev.analyse(_sc(p), p["bid"], p["eps"], p["epsv"]))
        assert code == (ev.NEAR if p["epsv"] else ev.OPTIMAL) and d["cert"]["all_ok"], name
        if p["epsv"] == 0:
            assert d["cert"]["exact_guaranteed"] and d["gap"] == 0


def _bids(sc, mode, epsv=0):
    return auction(sc, "sequential", mode, ev.eps_units(epsv, sc.n), record=False).counters["bids"]


def _median_of(cfg, f):
    return float(np.median([f(generate(*cfg, s)) for s in C.DIST_SEEDS]))


@pytest.mark.parametrize("name,tol,modes", [("🗺️ Mittlere Reichweite", 0.1, ("scaled",)), ("⚔️ Preiskrieg", 0.1, ("scaled", "fixed")), ("📡 Knappe Reichweite", 0.15, ("scaled", "fixed")),
                                             ("🚚 Mehr Fahrzeuge als Aufträge", 0.1, ("scaled", "fixed"))])
def test_random_preset_is_a_typical_draw(name, tol, modes):
    """Die gezeigte Karte liegt bei den Geboten der Schrittweiten, die das Preset zeigt, nahe dem Median der 100 festen Karten (die Verteilung der festen ist rechtsschief: Median, nicht Mittel).
    Die Standardkarte (Seed 165, dieselbe wie in den Vorgängern) ist für die feste Schrittweite untypisch leicht (288 gegen Median 428) und zeigt deshalb nur die Skalierung."""
    p = C.PRESETS[name]
    cfg = (p["n"], p["m"], p["reach"], p["ballung"])
    sc = _sc(p)
    for mode in modes:
        med = _median_of(cfg, lambda s, mode=mode: _bids(s, mode))
        assert abs(_bids(sc, mode) - med) <= tol * med + 1, (name, mode)


def test_the_coarse_preset_has_a_typical_gap():
    p = C.PRESETS["🎯 Zu grobes ε"]
    cfg = (p["n"], p["m"], p["reach"], p["ballung"])
    rows = [r for r in ev.coarse_rows(*cfg, p["epsv"], tuple(C.DIST_SEEDS))]
    median_gap, median_bids = float(np.median([r[3] for r in rows])), float(np.median([r[2] for r in rows]))
    mine = ev.coarse_rows(*cfg, p["epsv"], (p["seed"],))[0]
    assert abs(mine[3] - median_gap) <= 1 and abs(mine[2] - median_bids) <= 0.1 * median_bids + 1


def test_fixed_presets_hide_the_random_controls():
    assert {n for n, p in C.PRESETS.items() if p["net"] in C.FIXED_NETS} == {"⚖️ Die billigste Kante klaut", "⛓️ Lange Kette"}
