"""Auktionsalgorithmus - dezentral bieten statt zentral rechnen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - den Auktionsalgorithmus (Bertsekas) - und lässt stattdessen das Beispiel wachsen.
Viertes Stück der Matching-Linie der "Konzepte"-Reihe, Nachfolger der Ungarischen Methode: dieselben Preise, aber dezentral über Gebote. Nicht zu verwechseln mit der Demo zu kombinatorischen Auktionen. Siehe README.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import au_constants as C
import au_evaluation as ev
from au_algorithm import certificate, state_at
from au_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from au_scenario import build
from au_visualization import build_auction_map, build_bid_hist, build_penalty, build_phases, build_progress, build_reach_sweep, build_scaling, build_war

st.set_page_config(page_title="Auktionsalgorithmus – Sebastian Hanisch", layout="wide")


def _pct(x, digits=0):
    return "–" if x is None else f"{x:.{digits}f} %".replace(".", ",")


def _share(x):
    """Anteil (0..1) als 'nn %'."""
    return f"{100 * x:.0f} %"


def _f(x, digits=1):
    return "–" if x is None else f"{x:.{digits}f}".replace(".", ",")


def _int(x):
    """Ganzzahl mit Leerzeichen als Tausendertrenner."""
    return f"{x:,.0f}".replace(",", " ")


@st.cache_resource(show_spinner=False, max_entries=16)
def _analysis(params):
    net, n, m, reach, ballung, seed, bid, eps, epsv = params
    return ev.analyse(build(net, n, m, reach, ballung, seed), bid, eps, epsv)


@st.cache_data(show_spinner=False)
def _distribution(n, m, reach, ballung):
    return ev.distribution(n, m, reach, ballung)


@st.cache_data(show_spinner=False)
def _effort(n, m, reach, ballung):
    return ev.effort_table(n, m, reach, ballung)


@st.cache_data(show_spinner=False)
def _eps_sweep(n, m, reach, ballung):
    return ev.eps_sweep(n, m, reach, ballung)


@st.cache_data(show_spinner=False)
def _reach_sweep(n, m, ballung):
    return ev.reach_sweep(n, m, ballung)


@st.cache_data(show_spinner=False)
def _scaling():
    return ev.scaling()


@st.cache_data(show_spinner=False)
def _penalty(n, m, reach, ballung):
    return ev.penalty_experiment(n, m, reach, ballung)


@st.cache_data(show_spinner=False, max_entries=8)
def _progress(params, n_events):
    a = _analysis(params)
    return ev.progress_series(a.result, a.scenario.n, a.scenario.m)


st.title("🔨 Auktionsalgorithmus – dezentral bieten statt zentral rechnen")
st.markdown(
    """
Die **Ungarische Methode** findet das billigste Matching mit den meisten Paaren - aber in einer **zentralen Rechnung**, die alle Kosten kennt. Der **Auktionsalgorithmus** (Bertsekas) löst dieselbe Aufgabe **dezentral**:
jedes Fahrzeug kennt nur seine eigene Kostenzeile und die aktuellen Preise, bietet auf den Auftrag, der es inklusive Preis am wenigsten kostet, und hebt dessen Preis um den Vorsprung vor der zweitbesten Wahl plus ε an. Wer überboten wird, sucht sich einen neuen Auftrag.
Ist die Schrittweite ε kleiner als 1/(n+1) Minute, ist das Ergebnis **exakt optimal** - auch hier "erst möglichst viele Paare, dann minimale Kosten".
Der Preis dafür ist der **Preiskrieg**: bei fester kleiner Schrittweite streiten Fahrzeuge in Mini-Schritten um denselben Auftrag; **ε-Skalierung** (grob anfangen, ε schrittweise verkleinern) macht das im Mittel um ein Vielfaches billiger.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - viertes Stück der Matching-Linie der \"Konzepte\"-Reihe, Nachfolger der Ungarischen Methode - **ein** Verfahren an einem wachsenden Beispiel. "
    "Nicht zu verwechseln mit der Demo zu **kombinatorischen Auktionen** (Multi-Agenten-Linie): dort bieten Fahrzeuge auf Bündel von Aufträgen, hier auf einzelne Aufträge. "
    "Die Schwächen dieses Stücks sind die Ansatzpunkte der nächsten: **Blossom** und **Gewichteter Blossom** (allgemeine Graphen), **Gale–Shapley** (Vorlieben statt Kosten) und **Online-Matching** - alle inzwischen gebaut. "
    "Die Referenz \"Optimum\" ist die Ungarische Methode aus der Demo davor, deren Ergebnis zusätzlich in den Tests gegen scipy, networkx und Brute Force geprüft ist."
)

with st.expander("So funktioniert der Auktionsalgorithmus", expanded=True):
    st.markdown(
        """
1. **Preise:** jeder Auftrag hat einen Preis, am Anfang 0. Ein Fahrzeug bewertet einen Auftrag mit **Anfahrtszeit + Preis** (kleiner ist besser).
2. **Bieten:** ein Fahrzeug ohne Auftrag nimmt den Auftrag mit dem kleinsten Wert und bietet: der Preis steigt um den **Vorsprung** vor dem zweitbesten Auftrag **plus ε**. Der bisherige Halter wird **verdrängt** und bietet später neu.
3. **Verzicht:** jedes Fahrzeug darf auf einen Auftrag verzichten, gegen eine hohe Strafe M (so groß, dass sich jedes zusätzliche Paar lohnt). Das macht "Paare zuerst" exakt und beendet den Streit um Aufträge, die nicht alle bekommen können.
4. **ε:** je kleiner ε, desto genauer - bei ε < 1/(n+1) Minute exakt. Fest klein: viele kleine Preisschritte (**Preiskrieg**). **Skalierung:** mit großem ε anfangen, dann durch 5 teilen; am Ende jeder Phase werden die Preise nicht vergebener Aufträge wieder gesenkt (sonst ist das Ergebnis falsch).
5. **Nacheinander oder gleichzeitig:** ein Fahrzeug pro Schritt, oder alle Fahrzeuge ohne Auftrag in derselben Runde (das höchste Gebot je Auftrag gewinnt, die übrigen sind vergeblich).
6. **Beweis:** aus dem Ergebnis allein folgt eine Schranke für den Abstand zum Optimum; bei ε < 1/(n+1) Minute ist er null.
        """
    )

st.caption("🎯 Schnellstart – eine Beispielkarte laden:")
names = list(C.PRESETS.keys())
for row in range(0, len(names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, names[row:row + 4]):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Karte", list(C.NETS), key="net_select", format_func=lambda k: C.NETS[k],
        help="Eine zufällige Karte mit Fahrzeugen und Aufträgen, oder eine der festen Lehrbuchkarten, an denen sich die Rechnung von Hand nachvollziehen lässt.",
    )
    if net_key == "random":
        n = st.slider("Fahrzeuge", *bounds("n_slider"), key="n_slider", help="Anzahl der Fahrzeuge (die Bieter).")
        st.session_state[KEPT["n_slider"]] = n
        m = st.slider("Aufträge", *bounds("m_slider"), key="m_slider", help="Anzahl der Aufträge. Weniger Aufträge als Fahrzeuge heißt: ein Teil der Fahrzeuge muss verzichten, der Preiskrieg wird länger (bei 10 Aufträgen im Mittel 1 677 Gebote mit fester Schrittweite gegen 80 mit Skalierung).")
        st.session_state[KEPT["m_slider"]] = m
        reach = st.slider(
            "Reichweite [min]", *bounds("reach_slider"), key="reach_slider", step=5,
            help="Wie weit ein Fahrzeug höchstens fahren darf. Bei 10 braucht die feste Schrittweite im Mittel 9 Gebote und die Skalierung 19, bei 40 sind es 3 036 und 209 (Median der festen: 428) - der Preiskrieg tritt vor allem bei mittlerer Reichweite auf.",
        )
        st.session_state[KEPT["reach_slider"]] = reach
        ballung = st.slider("Ballung [%]", *bounds("ballung_slider"), key="ballung_slider", step=25, help="0 = Fahrzeuge und Aufträge gleichmäßig verteilt, 100 = alle um drei Stadtteile gruppiert.")
        st.session_state[KEPT["ballung_slider"]] = ballung
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neue Karte generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed. Die Verteilung über 100 feste Karten weiter unten ändert sich dabei nicht.")
    else:
        n = int(st.session_state.get(KEPT["n_slider"], C.DEFAULT_N))
        m = int(st.session_state.get(KEPT["m_slider"], C.DEFAULT_M))
        reach = int(st.session_state.get(KEPT["reach_slider"], C.DEFAULT_REACH))
        ballung = int(st.session_state.get(KEPT["ballung_slider"], C.DEFAULT_BALLUNG))
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Diese Karte ist fest - es gibt nichts zu erzeugen. Zahl der Fahrzeuge und Aufträge, Reichweite, Ballung und Seed gehören zur zufälligen Karte.")

# --- Die Auktion in Aktion ------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Auktion in Aktion")
step_col, bid_col, eps_col, play_col = st.columns([4, 3, 3, 2])
with bid_col:
    bid = st.radio("Gebote", list(C.BID_LABELS), key="bid_radio", format_func=lambda k: C.BID_LABELS[k], horizontal=True,
                   help="Nacheinander: ein Fahrzeug pro Schritt. Gleichzeitig: alle Fahrzeuge ohne Auftrag bieten in derselben Runde auf die Preise vom Rundenanfang; pro Auftrag gewinnt das höchste Gebot, die übrigen sind vergeblich.")
with eps_col:
    eps = st.radio("Schrittweite ε", list(C.EPS_LABELS), key="eps_radio", format_func=lambda k: C.EPS_LABELS[k], horizontal=True,
                   help="Skalierung: mit großem ε anfangen und schrittweise verkleinern, exakt am Ende. Fest klein: sofort exakt (ε = 1/(n+1) Minute), aber ein langer Preiskrieg; oder grob (mehr Minuten): schnell, aber nur noch fast optimal.")
    if eps == "fixed":
        if not st.session_state.get("_epsv_shown") and KEPT["epsv_select"] in st.session_state:
            st.session_state["epsv_select"] = st.session_state[KEPT["epsv_select"]]      # der zuletzt gewählte Wert kommt zurück, wenn der Regler wieder erscheint
        epsv = st.selectbox("Feste Schrittweite", list(C.EPSV_LABELS), key="epsv_select", format_func=lambda k: C.EPSV_LABELS[k],
                            help="exakt = 1/(n+1) Minute: garantiert das Optimum. Mehr Minuten: weniger Gebote, aber der Abstand zum Optimum kann bis zu n·ε betragen (gemessen viel weniger).")
        st.session_state[KEPT["epsv_select"]] = epsv
        st.session_state["_epsv_shown"] = True
    else:
        st.session_state["_epsv_shown"] = False
        epsv = int(st.session_state.get(KEPT["epsv_select"], C.DEFAULT_EPSV))

params = (net_key, int(n), int(m), int(reach), int(ballung), int(seed))
if net_key in C.FIXED_NETS:
    params = (net_key, C.DEFAULT_N, C.DEFAULT_M, C.DEFAULT_REACH, C.DEFAULT_BALLUNG, C.DEFAULT_SEED)
params = params + (bid, eps, int(epsv) if eps == "fixed" else 0)
with st.spinner("Rechne..."):
    a = _analysis(params)
sc, res, hung = a.scenario, a.result, a.hung
level, code, d = ev.verdict(a)
n_events = len(res.events)
if st.session_state.get("au_step_owner") != params:
    st.session_state["au_step"] = n_events
    st.session_state["au_step_owner"] = params
with step_col:
    step = st.slider("Schritt", 0, n_events, key="au_step", help="Wie viele Ereignisse (Phasenwechsel, Gebote, Verzichte, Preissenkungen) schon abgelaufen sind. Ganz rechts das fertige Ergebnis und der Beweis.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
sync_query_params({"net_select": net_key, "bid_radio": bid, "eps_radio": eps, "epsv_select": int(epsv), "n_slider": int(n), "m_slider": int(m), "reach_slider": int(reach),
                   "ballung_slider": int(ballung), "seed_input": int(seed)})
S = res.scale
p_max = max(res.counters["max_price"] / S, 1)
view_slot = st.empty()
unassigned, price_sum, phase_marks = _progress(params, n_events)


def _event_text(e):
    kind = e[0]
    if kind == "phase":
        return f"Phase {e[2] + 1}: ε = {e[1] / S:.3g} min - alle Zuordnungen werden aufgehoben, die Preise bleiben."
    if kind == "round":
        return f"Runde {e[1]}: alle Fahrzeuge ohne Auftrag bieten gleichzeitig auf die Preise vom Rundenanfang."
    if kind == "bid":
        _, i, j, p0, p1, v1, v2, old, won = e
        base = f"F{i + 1} bietet auf A{j + 1}: mit Preis kostet der Auftrag {-v1 / S:.1f} min, der zweitbeste {-v2 / S:.1f} min; Preis {p0 / S:.1f} → {p1 / S:.1f} min"
        if not won:
            return base + " - vergeblich, ein höheres Gebot auf denselben Auftrag gewinnt."
        return base + (f"; F{old + 1} wird verdrängt." if old >= 0 else ".")
    if kind == "idle":
        return f"F{e[1] + 1} verzichtet: jeder Auftrag ist ihm inklusive Preis teurer als die Strafe M = {res.penalty} min."
    _, j, p0, p1, taker, freed, was = e
    text = f"Preissenkung: A{j + 1} von {p0 / S:.1f} auf {p1 / S:.1f} min"
    if taker >= 0:
        text += f"; F{taker + 1} übernimmt ihn" + (f", A{freed + 1} wird frei." if freed >= 0 else " und beendet damit seinen Verzicht.")
    return text + "."


def _cert_table():
    cert = certificate(sc, res)
    ok = lambda b: "✅" if b else "❌"
    rows = [
        ("E1 ε-Schlupf", f"{ok(cert['e1'])} jedes Fahrzeug liegt höchstens ε = {res.eps_final / S:.3g} min unter seiner besten Wahl (größter Schlupf {cert['max_slack'] / S:.3g} min)"),
        ("E2 Preise nie negativ", f"{ok(cert['e2'])} kleinster Preis {min(res.prices, default=0) / S:.3g} min"),
        ("E3 Freie Aufträge: Preis 0", f"{ok(cert['e3'])} {sc.m - res.count} freie(r) Auftrag/Aufträge"),
        ("E4 Jeder Auftrag höchstens ein Fahrzeug", f"{ok(cert['e4'])} {res.count} Paare, {res.n_idle} Verzicht(e)"),
        ("E5 Lücke G ≤ n·ε", f"{ok(cert['e5'])} G = {cert['g'] / S:.3g} min ≤ {cert['bound'] / S:.3g} min"),
        ("Exakt garantiert (G < 1 Minute-Einheit)", ("✅ Die Lücke ist kleiner als eine Einheit (1/(n+1) Minute): das Ergebnis ist optimal." if cert["exact_guaranteed"]
                                                        else f"⚠️ nein: das Ergebnis liegt höchstens {cert['gap_minutes_bound']:.3g} min über dem Optimum")),
    ]
    return {"Bedingung": [r[0] for r in rows], "Prüfung": [r[1] for r in rows]}


def _render(k):
    """Zustand nach den ersten k Ereignissen: links die Karte mit dem letzten Ereignis, rechts der Verlauf; am Ende der Beweis."""
    with view_slot.container():
        st_k = state_at(res, sc.n, sc.m, k)
        ev_k = res.events[k - 1] if k > 0 else None
        c1, c2 = st.columns([3, 2])
        c1.markdown(f"**Nach {k} von {n_events} Ereignissen** - " + ("Anfang: alle Preise 0, kein Fahrzeug hat einen Auftrag." if k == 0 else _event_text(ev_k)))
        c1.plotly_chart(build_auction_map(sc, st_k, p_max, S, ev_k), width="stretch", key=f"auction_map_{k}")
        c2.markdown("**Verlauf**")
        c2.plotly_chart(build_progress(unassigned, price_sum, k, phase_marks), width="stretch", key=f"progress_chart_{k}")
        lo = max(0, k - 7)
        c2.table({"Schritt": list(range(lo + 1, k + 1)), "Ereignis": [_event_text(e) for e in res.events[lo:k]]} if k > 0 else {"Schritt": [0], "Ereignis": ["Anfang"]})
        if k == n_events:
            st.markdown("**Beweis:** aus dem Ergebnis allein folgt der Abstand zum Optimum")
            st.table(_cert_table())
            st.caption("Der Beweis kommt ohne die Ungarische Methode aus (schwache Dualität): jedes Fahrzeug liegt höchstens ε unter seiner besten Wahl, nicht vergebene Aufträge kosten nichts; daraus folgt, dass kein Matching mehr als G günstiger sein kann. "
                       "Alle Werte sind Vielfache von 1/(n+1) Minute: bei G < 1 Einheit ist die Zuordnung optimal.")


if auto_play:
    frames = sorted({int(round(x)) for x in np.linspace(0, n_events, min(n_events, 40) + 1)})
    for k in frames:
        _render(k)
        time.sleep(min(0.6, 6.0 / max(len(frames), 1)))
    step = n_events
else:
    _render(step)

if eps == "scaled" and len(res.counters["bids_by_phase"]) > 1:
    st.plotly_chart(build_phases(res.counters["bids_by_phase"], [e / S for e in res.eps_schedule]), width="stretch", key="phase_chart")
st.caption("Orangefarbene Kreise sind Aufträge (Größe und Farbe: Preis in Minuten; leer = frei), Quadrate Fahrzeuge (grün: haben einen Auftrag, violett umrandet: verzichtet, grau: bieten noch), blaue Linien die aktuelle Zuordnung. "
           "Das letzte Ereignis ist hervorgehoben: grüne Linie = Gebot, roter Ring = verdrängtes Fahrzeug, violetter Ring = Verzicht oder Preissenkung (gestrichelt: der Übernehmer). "
           "Im Verlauf: rote Treppe = Fahrzeuge ohne Auftrag, orange = Summe der Preise, senkrechte Striche = Phasenwechsel. "
           "Die Preise steigen weit über jede Fahrzeit hinaus, bis Verzichten billiger ist als Überbieten: sie sind Zertifikate für das Ergebnis, keine Grenzkosten wie bei der Ungarischen Methode.")

st.markdown("---")

# --- Optimal - und was kostet das? --------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Optimal – und was kostet das?")
st.caption("Verglichen wird mit der Ungarischen Methode (zentral, exakt). **Gebote** und **durchsuchte Kanten** sind maschinenunabhängige Zähler, keine Sekunden. Eine Auktion durchsucht die Kantenliste eines Fahrzeugs je Entscheidung und braucht keinen Heap; "
           "die Ungarische Methode zählt ihre durchsuchten Kanten im Dijkstra und braucht zusätzlich Heap-Operationen (mit einem Log-Faktor). Die Zähler sind also vergleichbar, aber nicht identisch in ihrer Bedeutung.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Paare (Ergebnis)", f"{res.count} von {hung.count}", help="Die Auktion vergibt so viele Paare wie möglich; der Rest verzichtet.")
m2.metric("Kosten", f"{res.cost} min", delta=f"{res.cost - hung.cost:+d} min gegenüber der Ungarischen Methode" if res.cost != hung.cost else "so billig wie die Ungarische Methode", delta_color="inverse" if res.cost != hung.cost else "off",
          help=f"Summe der Anfahrtszeiten. Ungarische Methode: {hung.cost} min.")
m3.metric("Gebote", _int(d["bids"]), delta=(f"davon vergeblich: {d['wasted']}, Runden: {d['rounds']}" if bid == "simultaneous" else f"{d['phases']} Phase(n), {d['takeovers']} Preissenkungs-Übernahmen"), delta_color="off",
          help=f"Fest klein hätte {_int(d['fixed_bids'])} Gebote gebraucht, die ε-Skalierung {_int(d['scaled_bids'])} (nacheinander, auf dieser Karte).")
m4.metric("Durchsuchte Kanten", _int(d["scans"]), delta=f"Ungarische Methode: {_int(d['hung_scans'])}", delta_color="off",
          help=f"Vorwärts {_int(d['fwd_scans'])} (Kantenliste des bietenden Fahrzeugs), rückwärts {_int(d['rev_scans'])} (Preissenkungs-Schritt); das Netz hat {d['edges']} Kanten.")

if code == "none":
    st.info("ℹ️ Keine einzige Kante ist möglich – die Reichweite ist zu klein. Es gibt nichts zuzuordnen.")
elif code == "mismatch":
    st.error(f"Abweichung von der Ungarischen Methode: {res.count} Paare für {res.cost} Minuten gegen {hung.count} Paare für {hung.cost} Minuten. Das dürfte nicht vorkommen.")
elif code == "near":
    st.warning(f"⚠️ Nahe am Optimum: {res.count} Paare für {res.cost} Minuten statt {hung.cost} (+{d['gap']} min, garantiert höchstens n·ε = {_f(d['bound_min'], 0)} min). "
               f"Mit ε = {d['eps_min']:.3g} min waren es nur {_int(d['bids'])} Gebote statt {_int(d['fixed_bids'])} bei exaktem ε - die Genauigkeit wurde gegen Tempo getauscht.")
else:
    how = {("sequential", "scaled"): f"in {d['phases']} Phasen mit ε-Skalierung", ("simultaneous", "scaled"): f"in {d['rounds']} Runden mit ε-Skalierung",
           ("sequential", "fixed"): "mit fester Schrittweite", ("simultaneous", "fixed"): f"in {d['rounds']} Runden mit fester Schrittweite"}[(bid, eps)]
    st.success(f"✅ Optimal: {res.count} Paare für {res.cost} Minuten – genau wie die Ungarische Methode. Die Auktion brauchte {_int(d['bids'])} Gebote {how} und {_int(d['scans'])} durchsuchte Kanten (Ungarische Methode: {_int(d['hung_scans'])}). "
               f"Mit fester Schrittweite wären es {_int(d['fixed_bids'])} Gebote gewesen, mit ε-Skalierung {_int(d['scaled_bids'])}.")

st.markdown("**Die Verfahren im Vergleich auf dieser Karte** (alle exakt)")
cmp_rows = ev.compare_table(a)
st.table({"Verfahren": [r["label"] for r in cmp_rows], "Paare": [r["count"] for r in cmp_rows], "Kosten [min]": [r["cost"] for r in cmp_rows],
          "Gebote": [_int(r["bids"]) if r["bids"] is not None else "–" for r in cmp_rows], "Runden": [str(r["rounds"]) if r["rounds"] is not None else "–" for r in cmp_rows],
          "vergeblich": [str(r["wasted"]) if r["wasted"] is not None else "–" for r in cmp_rows], "durchsuchte Kanten": [_int(r["scans"]) for r in cmp_rows]})

if net_key in C.FIXED_NETS:
    st.info("Feste Karte: es gibt nur diese eine Ziehung. Für die Verteilung über viele Karten eine zufällige Karte wählen.")
else:
    st.markdown(f"**Nicht nur diese eine Karte:** {len(C.DIST_SEEDS)} feste Karten mit denselben Einstellungen (Fahrzeuge {n}, Aufträge {m}, Reichweite {reach}, Ballung {ballung} %), getrennt vom Seed oben.")
    dist = _distribution(int(n), int(m), int(reach), int(ballung))
    if dist["n_valid"] == 0:
        st.info("ℹ️ Bei dieser Reichweite gibt es auf keiner der Karten ein mögliches Paar.")
    else:
        vs, vf = dist["variants"][ev.VARIANTS[0]], dist["variants"][ev.VARIANTS[2]]
        share_min = min(v["share_optimal"] for v in dist["variants"].values())
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Optimum getroffen", _share(share_min), help="Schlechtester Anteil der vier Varianten (nacheinander/gleichzeitig × Skalierung/fest, alle bei exaktem ε): hier immer alle.")
        p2.metric("Gebote, fest (Median | Mittel)", f"{_int(vf['bids_median'])} | {_int(vf['bids_mean'])}", delta=f"Skalierung: {_int(vs['bids_median'])} | {_int(vs['bids_mean'])}", delta_color="off",
                  help="Die Verteilung der festen Auktion ist stark rechtsschief: wenige Karten kosten sehr viele Gebote, deshalb liegt der Mittelwert weit über dem Median.")
        p3.metric("Fest teurer als Skalierung", _share(dist["fixed_worse_share"]), help="Anteil der Karten, auf denen die feste Schrittweite mehr Gebote braucht als die ε-Skalierung.")
        p4.metric("Kantensuchen: Skalierung gegen Ungarisch", f"{_f(dist['scaled_vs_hung'], 2)}×" if dist["scaled_vs_hung"] else "–", delta=f"{_int(vs['scans_mean'])} gegen {_int(dist['hung_scanned_mean'])}", delta_color="off",
                  help="Mittlere durchsuchte Kanten der Auktion mit ε-Skalierung geteilt durch die der Ungarischen Methode: über 1 heißt, die Auktion sucht mehr.")
        st.success(f"✅ Auf {_share(share_min)} der {dist['n_seeds']} Karten dieser Einstellung liefert jede der vier Auktionsvarianten das Optimum. "
                   f"Der Preis: fest klein {_int(vf['bids_mean'])} Gebote im Mittel (Median {_int(vf['bids_median'])}), ε-Skalierung {_int(vs['bids_mean'])}; gegenüber der Ungarischen Methode {_f(dist['scaled_vs_hung'], 2)}-mal so viele durchsuchte Kanten.")
        h1, h2 = st.columns([3, 2])
        h1.plotly_chart(build_bid_hist({"ε-Skalierung": vs["bids"], "fest klein": vf["bids"]}, current=d["scaled_bids"] if eps == "scaled" else d["fixed_bids"]), width="stretch", key="bid_hist")
        h1.caption("Gebote je Karte (logarithmische Achse). Die Marke ist Ihre Ziehung mit der oben gewählten Schrittweite; bei „fest klein“ reicht die Verteilung über mehrere Zehnerpotenzen.")
        eff = _effort(int(n), int(m), int(reach), int(ballung))
        h2.markdown("**Aufwand** (Mittel über die Karten)")
        h2.table({"Verfahren": [r["label"] for r in eff], "Optimum": [_share(r["optimal"]) for r in eff], "Gebote": [_int(r["bids"]) if r["bids"] is not None else "–" for r in eff],
                  "Runden": [_f(r["rounds"]) if r["rounds"] is not None else "–" for r in eff], "durchsuchte Kanten": [_int(r["scans"]) for r in eff]})

# --- Experimente auf Abruf ------------------------------------------------------------------------------------------------------------------

st.markdown("**Preiskrieg und Genauigkeit: wie viel spart ein größeres ε?**")
if st.button("ε von exakt bis 20 Minuten durchfahren (100 Karten je Wert, dauert einige Sekunden)", key="eps_start"):
    st.session_state["eps_on"] = (int(n), int(m), int(reach), int(ballung))
if st.session_state.get("eps_on") == (int(n), int(m), int(reach), int(ballung)):
    with st.spinner(f"Rechne {len(C.EPS_SWEEP)} Schrittweiten × {len(C.DIST_SEEDS)} Karten..."):
        eps_rows, eps_ref = _eps_sweep(int(n), int(m), int(reach), int(ballung))
    if eps_rows:
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(build_war(eps_rows, eps_ref), width="stretch", key="war_chart")
        c2.table({"ε": ["exakt" if r["eps"] == 0 else f"{r['eps']} min" for r in eps_rows], "Gebote": [_int(r["bids"]) for r in eps_rows], "Optimum": [_share(r["optimal"]) for r in eps_rows],
                  "größte Lücke [min]": [r["max_gap"] for r in eps_rows], "Schranke n·ε [min]": [_int(r["bound"]) for r in eps_rows], "Paare verloren": [r["pairs_lost"] for r in eps_rows]})
        st.caption("Sequentiell, fest ε, Mittel über 100 feste Karten. Ein größeres ε spart Gebote (logarithmische Achse), kostet aber Genauigkeit: die Lücke zum Optimum bleibt weit unter der garantierten Schranke n·ε. "
                   "Dass dabei keine Paare verloren gehen, ist gemessen, nicht bewiesen; bewiesen ist die Schranke auf den Gesamtkosten einschließlich der Strafe für Verzichte.")

st.markdown("**Wie hängt der Aufwand von der Reichweite ab?**")
if st.button("Reichweite von 10 bis 150 durchfahren (40 Karten je Wert, dauert wenige Sekunden)", key="sweep_start"):
    st.session_state["sweep_on"] = (int(n), int(m), int(ballung))
if st.session_state.get("sweep_on") == (int(n), int(m), int(ballung)):
    with st.spinner(f"Rechne {len(C.REACH_SWEEP)} Reichweiten × {len(C.SWEEP_SEEDS)} Karten..."):
        sweep_rows = _reach_sweep(int(n), int(m), int(ballung))
    st.plotly_chart(build_reach_sweep(sweep_rows, current=int(reach) if net_key == "random" else None), width="stretch", key="sweep_chart")
    st.caption("Mittel über 40 feste Karten je Reichweite; Fahrzeuge, Aufträge und Ballung wie oben, alles logarithmisch. Der Preiskrieg der festen Schrittweite hat seinen Gipfel bei mittlerer Reichweite - bei knapper gibt es kaum Streit, bei großer sind viele Aufträge für jedes Fahrzeug erreichbar. "
               "Die ε-Skalierung bleibt bei jeder Reichweite unter 300 Geboten.")

st.markdown("**Was kostet die Strafe für Verzicht?**")
if st.button("Strafe M ×1, ×2, ×4 durchrechnen (40 Karten je Wert)", key="penalty_start"):
    st.session_state["penalty_on"] = (int(n), int(m), int(reach), int(ballung))
if st.session_state.get("penalty_on") == (int(n), int(m), int(reach), int(ballung)):
    with st.spinner("Rechne 3 Strafen × 40 Karten × 2 Verfahren..."):
        pen_rows = _penalty(int(n), int(m), int(reach), int(ballung))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_penalty(pen_rows), width="stretch", key="penalty_chart")
    c2.table({"Strafe": [f"{r['factor']}×M" for r in pen_rows], "Gebote fest": [_int(r["fixed_bids"]) for r in pen_rows], "Gebote Skalierung": [_int(r["scaled_bids"]) for r in pen_rows]})
    st.caption("M ist die kleinste sichere Strafe (Anzahl möglicher Paare × größte Fahrzeit + 1): kleiner kann das Ergebnis Paare kosten (auf der langen Kette gibt M − 1 nur 6 statt 7 Paare). "
               "Ein größeres M verlängert den Preiskrieg der festen Schrittweite etwa im gleichen Verhältnis, weil Fahrzeuge, die um zu wenige Aufträge streiten, bis zu diesem Preis hochbieten; die ε-Skalierung wächst kaum.")

st.markdown("---")

# --- Aufwand -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Aufwand: wie wächst die Suche mit der Karte?")
if st.button("Karten von 10 bis 320 Fahrzeugen durchrechnen (dauert einige Sekunden)", key="scaling_start"):
    st.session_state["scaling_on"] = True
if st.session_state.get("scaling_on"):
    with st.spinner("Rechne 6 Kartengrößen × 10 Karten × 3 Verfahren..."):
        sc_rows = _scaling()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_scaling(sc_rows), width="stretch", key="scaling_chart")
    c2.table({"Fahrzeuge = Aufträge": [r["n"] for r in sc_rows], "Reichweite": [r["reach"] for r in sc_rows], "Ungarisch": [_int(r["hung_scans"]) for r in sc_rows], "Skalierung": [_int(r["scaled_scans"]) for r in sc_rows],
              "fest": [_int(r["fixed_scans"]) if r["fixed_scans"] is not None else "nicht gerechnet" for r in sc_rows], "Skalierung / Ungarisch": [_f(r["scaled_scans"] / r["hung_scans"], 2) + "×" for r in sc_rows]})
    st.caption(f"Fahrzeuge = Aufträge bei konstantem mittleren Grad (die Reichweite sinkt mit der Wurzel der Kartengröße), Mittel über 10 feste Karten. Gezählt werden **durchsuchte Kanten**, nicht Sekunden. "
               f"Die ε-Skalierung wächst wie die Ungarische Methode etwa mit dem Quadrat der Kartengröße und liegt hier bei einem festen Vielfachen; die feste Schrittweite wächst viel steiler und ist ab {C.SCALE_FIXED_MAX_N} Fahrzeugen nicht mehr gerechnet (dort schon Millionen Kantensuchen).")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Strafe M ist groß genug, aber nicht zu groß** | Zu klein: das Ergebnis verliert Paare. Groß genug (M = möglich Paare × größte Fahrzeit + 1) ist bewiesen, aber ein größeres M verlängert den Preiskrieg der festen Schrittweite etwa im gleichen Verhältnis. | ε-Skalierung, oder ein Vorlauf, der die Paarzahl zuerst festlegt (Verbesserungswege) |
| **ε ist klein genug** | Ein grobes ε ist schnell, aber nur noch fast optimal: die Lücke ist höchstens n·ε, gemessen weit darunter. Exakt ist es erst unter 1/(n+1) Minute - dafür ist die ε-Skalierung da. | ε-Skalierung, hier schon eingebaut |
| **Preise werden an alle übermittelt** | Die Auktion ist dezentral in der Rechnung, aber jedes Fahrzeug muss die Preise seiner Aufträge kennen, und gleichzeitiges Bieten braucht gemeinsame Runden. Ohne Takt und ohne Preisübertragung gilt das Ergebnis so nicht. | asynchrone Auktionen, Nachrichtenverluste (nicht gebaut) |
| **Es gibt zwei getrennte Seiten** | Fahrzeuge und Aufträge bilden zwei Gruppen. Sollen sich Fahrer untereinander paaren, gibt es Zyklen ungerader Länge. | **Blossom**, dann **Gewichteter Blossom** |
| **Nur die Summe der Kosten zählt** | Die Auktion minimiert die Gesamtkosten und fragt nicht, ob ein Fahrzeug lieber einen anderen Auftrag hätte. Haben beide Seiten Vorlieben, ist ein stabiles Ergebnis oft teurer als dieses Optimum. | **Gale–Shapley**: der Preis der Stabilität gegen dieses Optimum |
| **Alles ist vorab bekannt** | Aufträge kommen hier alle vor der Auktion an. Kommen sie nacheinander und sind schon zugesagt, darf nicht mehr umgeboten werden. | **Online-Matching** |
"""
)
st.caption("Die Matching-Linie ist inzwischen vollständig gebaut (13 Stücke): die Wurzel (Greedy-Matching), die Verbesserungswege, die Ungarische Methode, diese Demo, Hopcroft–Karp, Blossom, Gewichteter Blossom, Gale–Shapley, Stabile Mitbewohner, Krankenhaus-Zulassung, Top Trading Cycles, Nierentausch und Online-Matching.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Bipartiter Graph mit Fahrzeugen $V$ (die Bieter), Aufträgen $O$, möglichen Paaren $E$ und ganzzahligen Kosten $c_{ij}\ge 0$. Gesucht ist ein Matching mit größtmöglicher Paarzahl und darunter kleinsten Kosten.

**Verzicht.** Jedes Fahrzeug $i$ hat einen privaten, nicht überbietbaren Ausweg mit Wert $-S M$; $S=n+1$ skaliert die Kosten, $M=\min(n,m)\,c_{\max}+1$. Dann ist das Problem ein gewöhnliches Zuordnungsproblem mit den Werten $a_{ij}=-S c_{ij}$ und der Gesamtwert eines Matchings ist $-S\,(c(M)+M\cdot\#\text{Verzichte})$: jedes zusätzliche Paar lohnt sich, weil $M>\min(n,m)\,c_{\max}$ größer ist als die Kosten aller Paare zusammen.

**Gebot.** Bei Preisen $p_j\ge 0$ ist der Wert von Auftrag $j$ für $i$ gleich $a_{ij}-p_j$, der beste $v_1$, der zweitbeste (einschließlich des Verzichts) $v_2$. Fahrzeug $i$ nimmt den besten Auftrag $j^*$ und setzt
$$p_{j^*}\leftarrow p_{j^*}+(v_1-v_2)+\varepsilon,$$
der bisherige Halter wird verdrängt. Ist der Verzicht mindestens so gut wie $v_1$, verzichtet $i$.

**$\varepsilon$-Komplementarität.** Nach jedem Gebot gilt für jedes Fahrzeug: sein Wert liegt höchstens $\varepsilon$ unter seinem besten Wert. Mit $\pi_i$ = Wert des eigenen Auftrags und $\pi_i^*=\max(-SM,\max_j a_{ij}-p_j)$:
$$A \;\ge\; A^*-\sum_i(\pi_i^*-\pi_i)-\sum_{j\ \text{unversorgt}}p_j \;\ge\; A^*-n\varepsilon,$$
wenn alle nicht vergebenen Aufträge Preis 0 haben. Sind alle Werte Vielfache von $S$ und $n\varepsilon<S$, ist $A=A^*$: das Ergebnis ist optimal (schwache Dualität, Preise nie negativ).

**$\varepsilon$-Skalierung.** Phasen mit $\varepsilon_0=S c_{\max},\ \varepsilon_{k+1}=\max(1,\lfloor\varepsilon_k/5\rfloor)$ bis $\varepsilon=1$; die Preise bleiben, die Zuordnung wird zurückgesetzt. Weil hier nicht alle Aufträge vergeben werden, müssen am Ende jeder Phase die Preise unversorgter Aufträge gesenkt werden: $p_j\leftarrow\max\bigl(0,\max_i(a_{ij}-\pi_i-\varepsilon)\bigr)$; wer dabei straff ist, übernimmt den Auftrag. Ohne diesen Schritt ist das Ergebnis auf vielen Karten falsch.

**Preiskrieg.** Jedes Gebot hebt einen Preis um mindestens $\varepsilon$, Preise sind höchstens $\approx S M+\varepsilon$: höchstens $m\,(SM/\varepsilon+1)$ Gebote je Phase. Streiten $k$ Fahrzeuge um weniger als $k$ Aufträge, steigt der Preis in $\varepsilon$-Schritten bis in die Nähe der Strafe: das ist der Preiskrieg der festen Schrittweite.

**Gleichzeitig.** Alle Fahrzeuge ohne Auftrag bieten auf den Preisen vom Rundenanfang; pro Auftrag gewinnt das höchste Gebot (Gleichstand: kleinster Index). Die Ungleichung oben bleibt gültig.

**Grenzen.** (1) Strafe M. (2) ε. (3) Preise werden übermittelt, Takt. (4) Zwei Seiten. (5) Nur die Summe der Kosten zählt. (6) Alles vorab bekannt.

Implementiert in `au_scenario.py` (Karten, eigener Zufallsgenerator), `au_greedy.py`, `au_augment.py` und `au_hungarian.py` (Greedy, Verbesserungswege und Ungarische Methode aus den Vorgängerdemos), `au_algorithm.py` (Auktion, Preissenkung, Beweis), `au_evaluation.py` (Kennzahlen, Verteilung, Sweeps).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
