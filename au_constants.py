"""Konstanten, Regler-Grenzen, Presets und feste Seed-Mengen der Demo "Auktionsalgorithmus"."""

# --- Regler (wie in den Vorgängerdemos) ---------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 3, 40, 20          # Fahrzeuge
M_MIN, M_MAX, DEFAULT_M = 3, 40, 20          # Aufträge
REACH_MIN, REACH_MAX, DEFAULT_REACH = 10, 150, 40   # Reichweite in Minuten; ab 142 ist auf der 100x100-Karte alles erreichbar
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG = 0, 100, 0   # ganze Prozent, Schritt 25
DEFAULT_SEED = 165
SEED_MAX = 2_000_000_000

NETS = {"random": "Zufällige Karte", "steal": "Billigste Kante klaut (2×2)", "p4": "Pfad aus vier Punkten", "paths": "Drei Pfade hintereinander", "chain": "Lange Kette (13 Kanten)"}
DEFAULT_NET = "random"
FIXED_NETS = ("steal", "p4", "paths", "chain")

BID_LABELS = {"sequential": "Nacheinander", "simultaneous": "Gleichzeitig"}
DEFAULT_BID = "sequential"
EPS_LABELS = {"scaled": "ε-Skalierung", "fixed": "Fest klein"}
DEFAULT_EPS = "scaled"
# feste Schrittweite in Minuten; 0 = exakt (1/(n+1) Minute, das Optimum ist garantiert)
EPSV_LABELS = {0: "exakt (1/(n+1) min)", 1: "1 min", 2: "2 min", 5: "5 min", 10: "10 min", 20: "20 min"}
DEFAULT_EPSV = 0

# --- feste Seed-Mengen (dieselben wie in den Vorgängerdemos; unabhängig vom Nutzer-Seed) --------------------------------------------
DIST_SEEDS = tuple(range(100000, 100100))
SWEEP_SEEDS = DIST_SEEDS[:40]
REACH_SWEEP = (10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150)
SCALE_NS = (10, 20, 40, 80, 160, 320)      # Aufwand-Experiment: Fahrzeuge = Aufträge, mittlerer Grad konstant (Reichweite ~ 1/sqrt(n))
SCALE_FIXED_MAX_N = 80                      # der feste Lauf ist darüber nicht mehr zu bezahlen (bei n = 80 schon Millionen Kantensuchen)
SCALE_SEEDS = DIST_SEEDS[:10]
SCALE_REACH_AT_20 = 40
EPS_SWEEP = (0, 1, 2, 5, 10, 20)
M_FACTORS = (1, 2, 4)

COLORS = {"matched": "#1f77b4", "bid": "#2ca02c", "evicted": "#d62728", "vehicle": "#111111", "order": "#ff7f0e", "idle": "#9467bd",
          "hungarian": "#d62728", "scaled": "#1f77b4", "fixed": "#ff7f0e", "sim": "#2ca02c"}

# --- Presets ------------------------------------------------------------------------------------------------------------------
_BASE = dict(net="random", bid=DEFAULT_BID, eps=DEFAULT_EPS, epsv=DEFAULT_EPSV, n=DEFAULT_N, m=DEFAULT_M, reach=DEFAULT_REACH, ballung=DEFAULT_BALLUNG, seed=DEFAULT_SEED)
PRESETS = {
    "⚖️ Die billigste Kante klaut": {**_BASE, "net": "steal"},
    "⛓️ Lange Kette": {**_BASE, "net": "chain"},
    "🗺️ Mittlere Reichweite": {**_BASE},
    "⚔️ Preiskrieg": {**_BASE, "eps": "fixed", "seed": 177},
    "🔀 Gleichzeitig": {**_BASE, "bid": "simultaneous"},
    "📡 Knappe Reichweite": {**_BASE, "eps": "fixed", "reach": 10, "seed": 13},
    "🚚 Mehr Fahrzeuge als Aufträge": {**_BASE, "m": 10, "seed": 147},
    "🎯 Zu grobes ε": {**_BASE, "eps": "fixed", "epsv": 5, "seed": 68},
}
# Jede Zahl in diesen Texten ist in tests/test_claims.py über die 100 festen Karten (DIST_SEEDS) belegt
PRESET_HELP = {
    "⚖️ Die billigste Kante klaut": "Zwei Fahrzeuge, zwei Aufträge. Die Auktion kommt auf 10 Minuten wie die Ungarische Methode (Greedy zahlt 18): in fünf Geboten steigen die Preise so weit, bis das Fahrzeug mit der billigsten Kante den Auftrag lieber dem anderen überlässt. Die Preise (71 und 68, in Drittelminuten) sind nur ein Nebenprodukt - was zählt, ist das Ergebnis.",
    "⛓️ Lange Kette": "Sieben Fahrzeuge in einer Kette: die Auktion findet die 70 Minuten der Ungarischen Methode. Mit ε-Skalierung braucht sie 20 Gebote in 4 Phasen, mit fester Schrittweite nur 7 - hier gibt es keinen Streit, den die Skalierung entschärfen müsste.",
    "🗺️ Mittlere Reichweite": "Auf allen 100 Karten (20 Fahrzeuge, 20 Aufträge, Reichweite 40) liefert jede der vier Auktionsvarianten das Optimum. Mit ε-Skalierung braucht sie im Mittel 209 Gebote und 2 210 durchsuchte Kanten, die Ungarische Methode 1 698 Kanten: rund 1,3-mal so viele - dafür ohne zentrale Rechnung.",
    "⚔️ Preiskrieg": "Feste Schrittweite ε = 1/(n+1) Minute: derselbe Auftrag wird in Mini-Schritten überboten. Auf dieser Karte 434 Gebote gegen 200 mit ε-Skalierung. Über die 100 Karten liegt der Median bei 428 Geboten, der Mittelwert aber bei 3 036: einzelne Karten kosten Zehntausende Gebote.",
    "🔀 Gleichzeitig": "Alle Fahrzeuge ohne Auftrag bieten in derselben Runde, wie es dezentral wirklich liefe: im Mittel 97 Runden statt 209 Einzelgebote, dafür 268 statt 209 Gebote, weil 58 Gebote an ein höheres Gebot verlieren.",
    "📡 Knappe Reichweite": "Wenige mögliche Paare, kaum Streit: die feste Schrittweite braucht im Mittel 9 Gebote und 13 durchsuchte Kanten, die ε-Skalierung 19 Gebote und 91 Kanten, die Ungarische Methode 51 Kanten. Hier verliert die Skalierung.",
    "🚚 Mehr Fahrzeuge als Aufträge": "20 Fahrzeuge, nur 10 Aufträge: die Hälfte der Fahrzeuge muss verzichten (Verzicht-Option). Mit ε-Skalierung im Mittel 80 Gebote, mit fester Schrittweite 1 677: der Preiskrieg um zu wenige Aufträge ist besonders lang.",
    "🎯 Zu grobes ε": "Feste Schrittweite von 5 Minuten: nur 63 Gebote statt 348, aber nicht mehr optimal - hier 5 Minuten über dem Optimum, garantiert höchstens n·ε = 100. Über die 100 Karten ist die grobe Auktion auf 9 optimal.",
}
