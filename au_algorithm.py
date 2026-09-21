"""Auktionsalgorithmus (Bertsekas) für die lexikografische Zuordnung: erst möglichst viele Paare, dann minimale Kosten.

Bieter sind die Fahrzeuge, Objekte die Aufträge. Jedes Fahrzeug kennt nur seine eigene Kostenzeile und die aktuellen Preise.
Es bietet auf den Auftrag mit dem besten Wert v = -S*Kosten - Preis, hebt dessen Preis um (bester - zweitbester Wert + eps) an
und verdrängt den bisherigen Halter. Preise steigen nur; eine Runde endet, wenn jedes Fahrzeug einen Auftrag oder seine
Verzicht-Option hat.

Verzicht-Option: jedes Fahrzeug darf auf einen privaten, nicht überbietbaren Wert W = -S*M verzichten (M = min(n, m)*Cmax + 1).
Damit ist "erst Paare, dann Kosten" exakt (Gesamt = Paarkosten + M * Verzichte) und der Preiskrieg um unmögliche Paare endet.
Kosten sind mit S = n + 1 skaliert: bei eps = 1 ist die Lücke zum Optimum kleiner als S, also gleich null (ganze Zahlen).

eps-Skalierung mit behaltenen Preisen ist bei ungleich vielen Seiten NUR MIT einem Preissenkungs-Schritt am Ende jeder Phase
korrekt (unversorgte Aufträge dürfen keinen alten Preis behalten); der Schritt ist hier `reverse`. Ohne ihn ist das Ergebnis
auf den meisten Karten falsch (Negativkontrolle in den Tests).

Aufwand in Geboten und durchsuchten Kanten (Vorwärts: Kantenliste des Fahrzeugs je Entscheidung, Rückwärts: Spalte eines
Auftrags), nie in Sekunden. Alles ganzzahlig, deterministisch (niedrigster Index bei Gleichstand).
"""

import heapq
from dataclasses import dataclass, field

EPS_FACTOR = 5          # eps-Skalierung: eps <- max(1, eps // 5)
NEG_INF = None


@dataclass(frozen=True)
class Result:
    pairs: tuple             # ((i, j), ...) nach Fahrzeug sortiert
    cost: int                # Summe der echten Anfahrtszeiten
    n_idle: int              # Fahrzeuge, die auf die Verzicht-Option gegangen sind
    prices: tuple            # Auftragspreise (in 1/S Minute)
    idle: tuple              # bool je Fahrzeug
    scale: int               # S
    penalty: int             # M (in Minuten)
    eps_schedule: tuple      # eps je Phase
    events: tuple            # Ereignisprotokoll (leer bei record=False)
    counters: dict
    bid_mode: str
    capped: bool = False     # Gebotsgrenze erreicht (nur bei der naiven Auktion ohne Verzicht-Option)
    a: tuple = field(default=(), repr=False)   # -S*Kosten, je Fahrzeug ein dict {j: wert}

    @property
    def count(self):
        return len(self.pairs)

    @property
    def eps_final(self):
        return self.eps_schedule[-1] if self.eps_schedule else 1

    @property
    def penalised(self):
        return self.cost + self.penalty * self.n_idle

    @property
    def bids(self):
        return self.counters["bids"]

    @property
    def scans_total(self):
        return self.counters["fwd_scans"] + self.counters["rev_scans"]


def eps_plan(scale, cmax, mode, eps_fixed=1, eps0=None):
    """eps je Phase. 'scaled': eps0 = S*Cmax, dann durch 5 bis 1; 'fixed': eine Phase mit eps_fixed (in 1/S Minute)."""
    if mode == "fixed":
        return (max(1, int(eps_fixed)),)
    eps = max(1, scale * cmax if eps0 is None else int(eps0))
    plan = [eps]
    while eps > 1:
        eps = max(1, eps // EPS_FACTOR)
        plan.append(eps)
    return tuple(plan)


def auction(sc, bid_mode="sequential", eps_mode="scaled", eps_fixed=1, *, reverse=None, reset_idle=True,
            penalty=None, scale=None, eps0=None, outside=True, bid_cap=None, record=True):
    """Löst die Zuordnung per Auktion.

    bid_mode: 'sequential' (ein unversorgtes Fahrzeug pro Schritt, Gauss-Seidel) oder 'simultaneous' (alle Unversorgten bieten
    in einer Runde auf den Preisen vom Rundenanfang, pro Auftrag gewinnt das höchste Gebot, Jacobi).
    eps_mode: 'scaled' (eps-Skalierung) oder 'fixed' (eine Phase mit eps_fixed).
    reverse: Preissenkungs-Schritt am Phasenende; Standard: an bei 'scaled', aus bei 'fixed' (dort nie nötig).
    Die übrigen Parameter sind Negativkontrollen (kein Preissenkungs-Schritt, Verzicht bleibt dauerhaft, anderes M oder S,
    keine Verzicht-Option mit Gebotsgrenze).
    """
    n, m = sc.n, sc.m
    feas = sc.feasible
    cost = sc.cost
    adj = [[j for j in range(m) if feas[i, j]] for i in range(n)]
    col = [[i for i in range(n) if feas[i, j]] for j in range(m)]
    cmax = max([int(cost[i, j]) for i in range(n) for j in adj[i]] or [1]) or 1
    S = (n + 1) if scale is None else int(scale)
    M = (min(n, m) * cmax + 1) if penalty is None else int(penalty)
    W = -S * M
    a = [{j: -S * int(cost[i, j]) for j in adj[i]} for i in range(n)]
    plan = eps_plan(S, cmax, eps_mode, eps_fixed, eps0)
    if reverse is None:
        reverse = eps_mode == "scaled"

    p = [0] * m
    assign = [-1] * n
    owner = [-1] * m
    idle = [False] * n
    ev = []
    cnt = dict(bids=0, wasted=0, idles=0, takeovers=0, fwd_scans=0, rev_scans=0, rounds=0, phases=0, rev_passes=0,
               bids_by_phase=[], max_price=0)
    capped = False

    def value(i):
        if assign[i] >= 0:
            return a[i][assign[i]] - p[assign[i]]
        return W if outside else 0

    def decide(i, eps):
        """Entscheidung von Fahrzeug i auf den aktuellen Preisen: ('idle',) oder ('bid', j, neuer_preis, v1, v2)."""
        cnt["fwd_scans"] += len(adj[i])
        j1, v1, v2 = -1, None, None
        for j in adj[i]:
            v = a[i][j] - p[j]
            if v1 is None or v > v1:
                v2, j1, v1 = v1, j, v
            elif v2 is None or v > v2:
                v2 = v
        if j1 < 0:
            return ("idle",)
        if outside:
            if W >= v1:
                return ("idle",)
            v2 = W if v2 is None or W > v2 else v2
        elif v2 is None:
            v2 = v1                       # naive Auktion: einziger Kandidat, Zuwachs nur eps (Preiskrieg ohne Ende)
        return ("bid", j1, p[j1] + (v1 - v2) + eps, v1, v2)

    def take(i, j, newp, v1, v2, won=True):
        cnt["bids"] += 1
        if not won:
            cnt["wasted"] += 1
            if record:
                ev.append(("bid", i, j, p[j], newp, v1, v2, -1, False))
            return
        old = owner[j]
        if record:
            ev.append(("bid", i, j, p[j], newp, v1, v2, old, True))
        p[j] = newp
        cnt["max_price"] = max(cnt["max_price"], newp)
        owner[j] = i
        assign[i] = j
        if old >= 0:
            assign[old] = -1
        return old

    for ph, eps in enumerate(plan):
        cnt["phases"] += 1
        assign = [-1] * n
        owner = [-1] * m
        if reset_idle:
            idle = [False] * n
        if record:
            ev.append(("phase", eps, ph))
        bids_before = cnt["bids"]
        # ---- Vorwärts: bieten -------------------------------------------------------------------------------------
        if bid_mode == "sequential":
            heap = [i for i in range(n) if not idle[i]]
            heapq.heapify(heap)
            while heap:
                if bid_cap is not None and cnt["bids"] >= bid_cap:
                    capped = True
                    break
                i = heapq.heappop(heap)
                d = decide(i, eps)
                if d[0] == "idle":
                    idle[i] = True
                    cnt["idles"] += 1
                    if record:
                        ev.append(("idle", i))
                else:
                    old = take(i, d[1], d[2], d[3], d[4])
                    if old >= 0:
                        heapq.heappush(heap, old)
        else:
            while True:
                todo = [i for i in range(n) if assign[i] < 0 and not idle[i]]
                if not todo or (bid_cap is not None and cnt["bids"] >= bid_cap):
                    capped = capped or bool(todo)
                    break
                cnt["rounds"] += 1
                if record:
                    ev.append(("round", cnt["rounds"]))
                best = {}                       # Auftrag -> (neuer_preis, Fahrzeug, v1, v2)
                losers = []
                for i in todo:
                    d = decide(i, eps)
                    if d[0] == "idle":
                        idle[i] = True
                        cnt["idles"] += 1
                        if record:
                            ev.append(("idle", i))
                        continue
                    j, newp = d[1], d[2]
                    if j not in best or newp > best[j][0]:
                        if j in best:
                            losers.append((best[j][1], j, best[j][0], best[j][2], best[j][3]))
                        best[j] = (newp, i, d[3], d[4])
                    else:
                        losers.append((i, j, newp, d[3], d[4]))
                for i, j, newp, v1, v2 in losers:
                    take(i, j, newp, v1, v2, won=False)
                for j in sorted(best):
                    newp, i, v1, v2 = best[j]
                    take(i, j, newp, v1, v2)
        cnt["bids_by_phase"].append(cnt["bids"] - bids_before)
        if capped:
            break
        # ---- Rückwärts: unversorgte Aufträge auf ihren tragbaren Preis senken --------------------------------------
        if reverse:
            changed = True
            while changed:
                changed = False
                cnt["rev_passes"] += 1
                for j in range(m):
                    if owner[j] >= 0 or p[j] <= 0:
                        continue
                    cnt["rev_scans"] += len(col[j])
                    lower, taker = 0, -1
                    for i in col[j]:
                        need = a[i][j] - value(i) - eps
                        if need > lower:
                            lower, taker = need, i
                    freed, was_idle = -1, False
                    if taker >= 0:
                        freed = assign[taker]
                        was_idle = idle[taker]
                        if freed >= 0:
                            owner[freed] = -1
                        idle[taker] = False
                        assign[taker] = j
                        owner[j] = taker
                        cnt["takeovers"] += 1
                    if record:
                        ev.append(("lower", j, p[j], lower, taker, freed, was_idle))
                    p[j] = lower
                    changed = True

    pairs = tuple((i, assign[i]) for i in range(n) if assign[i] >= 0)
    total = sum(int(cost[i, j]) for i, j in pairs)
    n_idle = n - len(pairs)
    return Result(pairs=pairs, cost=total, n_idle=n_idle, prices=tuple(p), idle=tuple(idle), scale=S, penalty=M,
                  eps_schedule=plan, events=tuple(ev), counters=cnt, bid_mode=bid_mode, capped=capped,
                  a=tuple(a))


# --- Protokoll abspielen ----------------------------------------------------------------------------------------------------

def initial_state(n, m):
    return dict(assign=[-1] * n, owner=[-1] * m, p=[0] * m, idle=[False] * n, eps=None, phase=-1, round=0)


def apply_event(st, e):
    kind = e[0]
    if kind == "phase":
        n, m = len(st["assign"]), len(st["owner"])
        st["assign"], st["owner"] = [-1] * n, [-1] * m
        st["idle"] = [False] * n if st.get("_reset_idle", True) else st["idle"]
        st["eps"], st["phase"] = e[1], e[2]
    elif kind == "round":
        st["round"] = e[1]
    elif kind == "idle":
        st["idle"][e[1]] = True
    elif kind == "bid":
        _, i, j, _p0, p1, _v1, _v2, old, won = e
        if won:
            st["p"][j] = p1
            st["owner"][j] = i
            st["assign"][i] = j
            if old >= 0:
                st["assign"][old] = -1
    elif kind == "lower":
        _, j, _p0, p1, taker, freed, _was = e
        st["p"][j] = p1
        if taker >= 0:
            if freed >= 0:
                st["owner"][freed] = -1
            st["idle"][taker] = False
            st["assign"][taker] = j
            st["owner"][j] = taker


def state_at(res, n, m, k, checkpoints=None):
    """Zustand nach den ersten k Ereignissen (spielt das Protokoll ab, ab dem nächsten Kontrollpunkt)."""
    st = initial_state(n, m)
    for e in res.events[:k]:
        apply_event(st, e)
    return st


def all_states(res, n, m):
    """Zustände nach 0, 1, ..., len(events) Ereignissen (Kopien; nur für kleine Läufe oder gezogene Schritte nutzen)."""
    st = initial_state(n, m)
    out = [_copy(st)]
    for e in res.events:
        apply_event(st, e)
        out.append(_copy(st))
    return out


def _copy(st):
    return dict(assign=list(st["assign"]), owner=list(st["owner"]), p=list(st["p"]), idle=list(st["idle"]), eps=st["eps"],
                phase=st["phase"], round=st["round"])


# --- Beweis -----------------------------------------------------------------------------------------------------------------

def certificate(sc, res):
    """Zertifikat aus dem Ergebnis allein (ohne Ungarische Methode).

    E1 eps-Schlupf: jedes Fahrzeug liegt höchstens eps unter seinem besten Wert (Aufträge und Verzicht).
    E2 Preise >= 0. E3 unversorgte Aufträge haben Preis 0. E4 jeder Auftrag hat höchstens ein Fahrzeug.
    E5 Lücke G = Summe der Fahrzeugschlupfe + Preise unversorgter Aufträge; G <= n*eps, und bei G < S ist die Zuordnung exakt
    (schwache Dualität; alle Werte sind Vielfache von S).
    """
    n, m = sc.n, sc.m
    S, eps = res.scale, res.eps_final
    W = -S * res.penalty
    p = res.prices
    assign = {i: j for i, j in res.pairs}
    used = [j for _, j in res.pairs]
    slack, e1 = [], True
    for i in range(n):
        a_i = res.a[i]
        best = max([W] + [a_i[j] - p[j] for j in a_i])
        own = (a_i[assign[i]] - p[assign[i]]) if i in assign else W
        slack.append(best - own)
        e1 = e1 and 0 <= best - own <= eps and (i in assign or res.idle[i])
    e2 = all(x >= 0 for x in p)
    e3 = all(p[j] == 0 for j in range(m) if j not in used)
    e4 = len(used) == len(set(used))
    g = sum(slack) + sum(p[j] for j in range(m) if j not in used)
    return dict(e1=e1, e2=e2, e3=e3, e4=e4, g=g, bound=n * eps, scale=S, eps=eps, e5=g <= n * eps,
                exact_guaranteed=g < S, gap_minutes_bound=g / S, max_slack=max(slack or [0]),
                all_ok=e1 and e2 and e3 and e4 and g <= n * eps)
