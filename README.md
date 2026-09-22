# Auktionsalgorithmus – dezentral bieten statt zentral rechnen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-auction-algorithm-demo.streamlit.app/)**

Viertes Stück der **Matching-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Nachfolger der [Ungarischen Methode](https://github.com/sebastian-hanisch/hungarian-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – den **Auktionsalgorithmus** (Bertsekas) – an einem wachsenden Beispiel.
Die Ungarische Methode findet das billigste Matching mit den meisten Paaren, aber in einer **zentralen Rechnung**, die alle Kosten kennt. Hier kennt jedes Fahrzeug nur seine eigene Kostenzeile und die aktuellen Preise, bietet auf den Auftrag, der es inklusive Preis am wenigsten kostet, und hebt dessen Preis um den Vorsprung vor der zweitbesten Wahl plus ε an; wer überboten wird, sucht sich einen neuen Auftrag.
Ziel ist wieder lexikografisch: erst möglichst viele Paare, dann minimale Kosten. Ist ε kleiner als 1/(n+1) Minute, ist das Ergebnis **exakt optimal**.

**Nicht zu verwechseln** mit der Demo [auction-demo](https://github.com/sebastian-hanisch/auction-demo) (kombinatorische Auktionen, Multi-Agenten-Linie): dort bieten Fahrzeuge auf Bündel von Aufträgen, hier auf einzelne Aufträge, und die Aufgabe ist das klassische Zuordnungsproblem.

**Zwei Umschalter, deren Vergleich der Kern der Demo ist:** *Gebote* (nacheinander: ein Fahrzeug pro Schritt; gleichzeitig: alle Fahrzeuge ohne Auftrag in derselben Runde, wie es dezentral wirklich liefe) und *Schrittweite ε* (ε-Skalierung: mit großem ε anfangen und schrittweise verkleinern; fest klein: sofort exakt, aber ein langer **Preiskrieg**; oder fest grob: schnell, aber nur noch fast optimal).

**Einordnung in die Reihe (die Kanten des Graphen):** dieses Stück löst die Zentralität der Ungarischen Methode. Seine eigenen Schwächen sind die Ansatzpunkte der nächsten: es gibt nur zwei getrennte Seiten (**Blossom → Gewichteter Blossom**), Kosten statt Vorlieben (**Gale–Shapley**), alles ist vorab bekannt (**Online-Matching**).
```
greedy-matching-demo (Wurzel: eine gewählte Zuordnung bleibt)                     [gebaut]
  ├─ augmenting-path-demo (Verbesserungswege: Paare optimal, Kosten blind)        [gebaut]
  │    ├─ hopcroft-karp-demo (viele kürzeste Wege je Phase)                       [gebaut]
  │    ├─ hungarian-demo (Ungarische Methode: Paare zuerst, dann Kosten)           [gebaut]
  │    │    └─ auction-algorithm-demo (Auktionsalgorithmus: dezentral)             [dieses Stück]
  │    └─ blossom-demo (allgemeine Graphen: ungerade Kreise, Kontraktion)          [gebaut]
  │        └─ weighted-blossom-demo (Ungarisch + Blossom, Konvergenz)              [gebaut]
  ├─ Gale–Shapley → Stabile Mitbewohner, Krankenhaus-Zulassung, Top Trading Cycles, Nierentausch [gebaut]
  └─ online-matching-demo (Aufträge kommen nacheinander)                          [gebaut]
```

## Ergebnis (Zahlen aus den Tests)

Jede hier genannte Zahl ist in `tests/test_claims.py` über die 100 festen Karten (Seeds 100000–100099) belegt (20 Fahrzeuge, 20 Aufträge, Reichweite 40, wo nichts anderes steht). Gezählt werden **Gebote** und **durchsuchte Kanten** (Vorwärts: Kantenliste eines Fahrzeugs je Entscheidung, Rückwärts: Spalte eines Auftrags), nie Sekunden. Die Ungarische Methode zählt ihre durchsuchten Kanten im Dijkstra und braucht zusätzlich Heap-Operationen; die Zähler sind vergleichbar, nicht identisch in ihrer Bedeutung.

| Frage | Ergebnis |
|---|---|
| Ist das Ergebnis optimal? | ✅ Ja: alle vier Varianten (nacheinander/gleichzeitig × ε-Skalierung/fest, exaktes ε) treffen auf allen 100 Karten bei jeder Reichweite der Sweep-Reihe und bei Ballung 50 und 100 das Optimum der Ungarischen Methode; zusätzlich gegen scipy, networkx und Brute Force geprüft. Das Zertifikat (ε-Schlupf, Preise ≥ 0, unversorgte Aufträge bei 0, Lücke G < 1 Einheit) folgt aus dem Ergebnis allein und braucht die Ungarische Methode nicht. |
| Und der Aufwand gegen die Ungarische Methode? | ❌ Bei Reichweite 40 durchsucht die Auktion mit ε-Skalierung im Mittel 2 210 Kanten, die Ungarische Methode 1 698 (Faktor 1,3). Bei Reichweite 10 sind es 91 gegen 51; auf großen Karten (40 × 40, Reichweite 60) dreht es sich: 14 567 gegen 21 124 (0,69). Im Aufwand-Experiment (konstanter mittlerer Grad, 10 bis 320 Fahrzeuge) liegt die Skalierung bei 1,3 bis 1,6 mal so vielen Kanten, beide wachsen etwa quadratisch. |
| Preiskrieg bei fester Schrittweite | ❌ Mit ε = 1/(n+1) Minute braucht die Auktion im Median 428, im **Mittel 3 036** Gebote (Verteilung stark rechtsschief, Ausreißer über 40 000), mit ε-Skalierung im Mittel 209: 14,5-mal so viele, auf 69 von 100 Karten mehr. Mehr Fahrzeuge als Aufträge (20 × 10): 1 677 gegen 80, auf allen 100 Karten. |
| Wo die feste Schrittweite gewinnt | ✅ Bei knapper Reichweite (10): 9 Gebote und 13 durchsuchte Kanten gegen 19 und 91 mit Skalierung (und 51 bei der Ungarischen Methode). Bei mehr Aufträgen als Fahrzeugen (10 × 20): 16 gegen 56 Gebote. Der Preiskrieg tritt vor allem bei mittlerer Reichweite auf (Gipfel bei 40; bei 150 nur 1,9-mal so viele Gebote wie mit Skalierung). |
| Grobes ε | ⚠️ Spart Gebote (Mittel 701 / 465 / 251 / 152 / 92 bei ε = 1 / 2 / 5 / 10 / 20 Minuten gegen 3 036 exakt), ist aber nur noch fast optimal: optimal auf 75 / 39 / 9 / 2 / 0 von 100 Karten, größte Lücke 3 / 7 / 32 / 44 / 84 Minuten gegen die garantierte Schranke n·ε = 20 / 40 / 100 / 200 / 400. Dass keine Paare verloren gehen, ist gemessen (0 auf allen Karten), nicht bewiesen. |
| Gleichzeitig bieten | ⚠️ Weniger Runden (im Mittel 97) als Einzelgebote (209), dafür mehr Gebote insgesamt (268; 58 davon vergeblich, weil ein höheres Gebot auf denselben Auftrag gewinnt). |
| Strafe für Verzicht | ⚠️ M = mögliche Paare × größte Fahrzeit + 1 ist bewiesen sicher; M − 1 kostet auf der langen Kette ein Paar (6 statt 7). Ein größeres M verlängert den Preiskrieg der festen Schrittweite etwa im gleichen Verhältnis (2 096 → 4 071 → 8 025 Gebote bei ×1, ×2, ×4), die Skalierung kaum (202 → 233 → 296). |
| Die billigste Kante klaut (2 × 2) | ✅ 10 Minuten wie die Ungarische Methode (Greedy zahlt 18), in fünf Geboten; die Preise (71 und 68 in Drittelminuten) sind nur ein Nebenprodukt. |
| Lange Kette | ✅ 70 Minuten; 20 Gebote in 4 Phasen mit Skalierung, 7 mit fester Schrittweite. |
| Wie wächst der Aufwand? | ⚠️ Die feste Schrittweite wächst viel steiler: bei 80 Fahrzeugen über 4 Millionen durchsuchte Kanten gegen rund 32 000 mit Skalierung; sie wird deshalb im Experiment nur bis 80 Fahrzeuge gerechnet. |

## Was nicht funktioniert hat / widerlegte Vorab-Hypothesen

- **„ε-Skalierung mit behaltenen Preisen ist Standard und genügt.“** Falsch, sobald nicht alle Aufträge vergeben werden (ungleich viele Seiten, unmögliche Paare): unversorgte Aufträge behalten alte Preise, und die Schranke n·ε gilt nicht mehr. Ohne den **Preissenkungs-Schritt** am Ende jeder Phase ist das Ergebnis auf 42 von 100 Karten (20 × 20) und auf allen 100 Karten bei 20 × 10, 10 × 20 und Reichweite 10 falsch. Der Schritt ist deshalb Teil des Verfahrens und Teil der Erklärung in der App.
- **„Ein Skalierungsfaktor S = 1 genügt.“** Ohne die Skalierung der Kosten mit S = n + 1 ist die Lücke n·ε nicht kleiner als eine Kosteneinheit: falsch auf 25 (fest) bzw. 31 (Skalierung) von 100 Karten.
- **„Auch ohne Verzicht-Option kommt die Auktion zum Ziel.“** Nein: gibt es weniger mögliche Paare als Fahrzeuge, bieten Fahrzeuge um Aufträge, die nicht alle bekommen können, ohne Ende hoch. Mit Gebotsgrenze bricht die naive Auktion auf 35 von 100 Karten (20 × 20) und auf allen 100 bei mehr Fahrzeugen als Aufträgen ab.
- **„Fahrzeuge im Verzicht bleiben dort.“** Nur im reinen Vorwärtslauf; nach einer Preissenkung müssen Verzichter neu bewertet werden (sonst falsch auf einem Teil kleiner Zufallskarten, getestet).
- **„ε-Skalierung ist immer besser.“** Nur bei mittlerer Reichweite und mit Aufträgen knapp; bei knapper Reichweite und bei mehr Aufträgen als Fahrzeugen gewinnt die feste Schrittweite.
- **„Gleichzeitiges Bieten spart Arbeit.“** Es spart Runden, nicht Gebote.
- **Die Standardkarte (Seed 165) ist für die feste Schrittweite untypisch leicht** (288 Gebote gegen Median 428) und zeigt deshalb nur die Skalierung; der Preiskrieg-Preset nutzt eine Karte nahe dem Median.

## Was die Demo zeigt

- **Die Auktion in Aktion:** Schritt-Slider und ▶️ über alle Ereignisse (Phasenwechsel, Gebote, Verzichte, Preissenkungen), Umschalter für Gebotsart und Schrittweite. Die Karte zeigt Preise als Größe und Farbe der Aufträge, das aktuelle Gebot, das verdrängte Fahrzeug; daneben der Verlauf und ein Ereignisprotokoll. Im letzten Schritt der **Beweis** (Tabelle E1–E5 mit echten Zahlen).
- **Optimal – und was kostet das?** Ergebnis gegen die Ungarische Methode, Vergleichstabelle der vier Varianten, Verteilung über 100 feste Karten (Histogramm, Mittel und Median), Aufwandstabelle. Auf Abruf: Schrittweiten-Sweep, Reichweite-Sweep, Strafe M ×1/×2/×4, Aufwand-Experiment (10 bis 320 Fahrzeuge).
- **Feste Lehrbuchkarten** (2 × 2, lange Kette) und zufällige Karten; **Wo die Annahmen enden:** welches spätere Stück an welcher Schwäche ansetzt.

## Modell und Verfahren

- **Verzicht-Option:** jedes Fahrzeug darf auf einen privaten, nicht überbietbaren Wert −S·M verzichten, M = min(n, m)·c_max + 1. Damit ist "erst Paare, dann Kosten" exakt (Gesamtkosten = Paarkosten + M·Verzichte) und das Problem ein gewöhnliches Zuordnungsproblem.
- **Gebot:** Wert eines Auftrags −S·Kosten − Preis; bester Wert v₁, zweitbester (einschließlich Verzicht) v₂; Preis ← Preis + (v₁ − v₂) + ε, der bisherige Halter wird verdrängt; ist der Verzicht mindestens so gut wie v₁, verzichtet das Fahrzeug. Gleichstand: niedrigster Index.
- **ε-Plan:** Skalierung ε₀ = S·c_max, dann durch 5 bis ε = 1; fest: eine Phase (exakt = 1, oder 1 / 2 / 5 / 10 / 20 Minuten). Am Phasenende (nur bei Skalierung): **Preissenkung** unversorgter Aufträge auf max(0, max_i(a_ij − π_i − ε)); das straffe Fahrzeug übernimmt.
- **Gleichzeitig:** alle Fahrzeuge ohne Auftrag bieten auf den Preisen vom Rundenanfang, pro Auftrag gewinnt das höchste Gebot (Gleichstand: kleinster Fahrzeugindex), die übrigen sind vergeblich.
- **Beweis:** mit π_i = Wert des eigenen Auftrags und π*_i = bester Wert gilt A ≥ A* − Σ(π*_i − π_i) − Σ(Preise unversorgter Aufträge) ≥ A* − n·ε; sind alle Werte Vielfache von S und n·ε < S, ist A = A*.
- **Preiskrieg:** jedes Gebot hebt einen Preis um mindestens ε; streiten k Fahrzeuge um weniger als k Aufträge, steigt der Preis in ε-Schritten bis in die Nähe der Strafe – die Länge wächst mit M/ε.
- **Aufwand:** Gebote und durchsuchte Kanten, getrennt nach Vorwärts- und Rückwärtssuche; Runden bei gleichzeitigem Bieten; nie Sekunden.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `au_constants.py` | Regler-Grenzen, Presets und Hilfetexte, feste Seed-Mengen |
| `au_presets.py` | Permalink, Preset- und Zufalls-Seed-Logik (Standardmuster des Portfolios, kopiert und um Gebotsart und Schrittweite ergänzt) |
| `au_scenario.py` | Karten, eigener Zufallsgenerator, feste Lehrbuchkarten (aus den Vorgängerdemos kopiert) |
| `au_greedy.py`, `au_augment.py`, `au_hungarian.py` | Greedy-Regeln, Verbesserungswege und Ungarische Methode aus den Vorgängerdemos (kopiert, ohne Import; die Ungarische Methode ist Referenz und Vergleichsverfahren) |
| `au_algorithm.py` | **Neu:** Auktion (nacheinander / gleichzeitig, ε-Skalierung / fest), Preissenkung, Ereignisprotokoll und Wiedergabe, Zertifikat |
| `au_evaluation.py` | Einordnung, Verdict, Vergleichstabelle, Verteilung über viele Karten, Sweeps, Strafe |
| `au_visualization.py` | Plotly-Abbildungen (Achsen gesperrt für Touch-Geräte; Bögen bei Punkten auf einer Geraden) |
| `tests/` | Algorithmus (Handrechnung, Invarianten je Ereignis, scipy, networkx, Brute Force und Ungarische Methode als Gegenprobe, Zertifikat mit Negativtest, Negativkontrollen), Auswertung, Presets, belegte Zahlen, AppTest-Rauchtests (alle vier Kombinationen, auch beim Abspielen) |

Die Kopien der Vorgänger werden durch Tests bewacht: Zufallsgenerator-Vektor und die Seed-2-Karte der Vorgänger (20 Paare, 316 Minuten). Alle Daten sind synthetisch; die Laufzeit braucht nur numpy, pandas, plotly und streamlit (scipy und networkx sind reine Testorakel).

## Lokal starten

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\streamlit run app.py
```

## Tests ausführen

```bash
venv\Scripts\pip install -r requirements-dev.txt
venv\Scripts\python -m pytest tests -v
```

Die Logik rechnet ausschließlich mit ganzen Zahlen; die im Text genannten Anteile und Mittelwerte sind deshalb auf jeder Plattform identisch.
Die CI (`.github/workflows/tests.yml`) läuft auf Ubuntu mit Python 3.12, bei jedem Push und wöchentlich mit den jeweils neuesten Bibliotheksversionen.
