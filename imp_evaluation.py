"""Kennzahlen, Verdikt-Logik und Experimente über feste Ströme (reine Funktionen, kein Streamlit).

Ereignis A = das Netz hängt nicht zusammen (seltener Ausfall, Wahrscheinlichkeit P). Importance Sampling zieht jede Kante mit q' statt q als ausgefallen und gewichtet; die Varianz ist
exakt aus der Aufzählung berechenbar (CV^2). Diagnose: ESS, nahe dem Optimum etwa N / (1 + CV^2); Pareto-k̂ gilt nur bei stetigen Gewichten (Begleiter), bei den diskreten Gewichten des Netzes in dieser Demo nicht.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import imp_algorithm as A
import imp_constants as C
from imp_scenario import Netz, named_netz


def de(x: float, nd: int = 2) -> str:
    """Zahl im deutschen Format: Punkt als Tausendertrenner, Komma als Dezimalzeichen, echtes Minuszeichen."""
    return f"{x:,.{nd}f}".translate(str.maketrans(",.", ".,")).replace("-", "−")


def dn(x: float, nd: int = 2) -> str:
    """Wie `de`, aber „nicht bestimmbar“ für NaN und „unendlich“ für ∞."""
    if np.isnan(x):
        return "nicht bestimmbar"
    if np.isinf(x):
        return "unendlich"
    return de(x, nd)


def de_e(x: float) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma, z. B. 4,2e-10."""
    return f"{x:.1e}".replace(".", ",")


def dz(x: float) -> str:
    """Zahl für Größen über viele Zehnerpotenzen (Ziehungen, CV²): ab 10 Millionen wissenschaftlich, sonst mit Tausenderpunkt und passend vielen Stellen."""
    if not np.isfinite(x):
        return dn(x)
    if abs(x) >= 1e7:
        return de_e(x)
    if abs(x) >= 100:
        return de(x, 0)
    return de(x, 1 if abs(x) >= 10 else 2)


def kanten(n: int) -> str:
    return f"{n} Kante" if n == 1 else f"{n} Kanten"


@dataclass(frozen=True)
class Settings:
    netz: str = C.DEFAULT_NETZ
    q: float = C.DEFAULT_Q
    qp: float = C.DEFAULT_QP
    n: int = C.DEFAULT_N
    t: float = C.DEFAULT_T
    theta: float = C.DEFAULT_THETA
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    netz: Netz
    P: float                      # exakte Ausfallwahrscheinlichkeit
    cv2_plain: float              # exakt, einfaches Ziehen
    cv2_is: float                 # exakt bei der gewählten Kippung
    gain: float                   # cv2_plain / cv2_is
    qp_opt: float                 # optimale Kippung
    cv2_opt: float
    min_cut: int                  # kleinste Ausfallmenge (Kanten)
    n_plain: float                # Ziehungen für RELERR bei einfachem Ziehen
    n_is: float
    ess_exact: float              # N / (1 + CV^2) bei der gewählten Kippung
    med_rel: float                # gemessen: mediane relative Abweichung
    frac_gt50: float              # Anteil der Wiederholungen mit mehr als 50 % Fehler
    zero: float                   # Anteil der Wiederholungen mit Schätzung null
    cover: float                  # Überdeckung des Wald-Intervalls
    ess_meas: float               # Median der gemessenen ESS
    k_med: float                  # Median Pareto-k̂ (NaN wenn nicht bestimmbar)
    frac_k_bad: float             # Anteil der Wiederholungen mit k̂ über der Marke
    est: tuple[float, ...]


# --- Wahrheit -----------------------------------------------------------------------------------------------------------------------------

def netz_table(netz: Netz) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aufzählung aller Kantenmengen (höchstens C.LIVE_EDGE_LIMIT Kanten): X, Zahl haltender Kanten und die Zählung N_f der Ausfallmengen."""
    if netz.m > C.LIVE_EDGE_LIMIT:
        raise ValueError(f"{netz.m} Kanten: aufgezählt wird nur bis {C.LIVE_EDGE_LIMIT}")
    X, k = A.enumerate_x(netz.n, netz.edges)
    return X, k, A.failure_counts(X, k, netz.m)


def min_cut(Nf: np.ndarray) -> int:
    """Kleinste Zahl ausgefallener Kanten, die das Netz trennt (−1, wenn nichts trennt)."""
    nz = np.nonzero(Nf > 0)[0]
    return int(nz[0]) if len(nz) else -1


# --- Hauptanalyse -------------------------------------------------------------------------------------------------------------------------

def analyse(settings: Settings, netz: Netz | None = None, table=None) -> Analysis:
    """REPS Schätzungen aus je N Ziehungen mit der gewählten Kippung q'; exakte Kennzahlen aus der Aufzählung daneben."""
    netz = netz if netz is not None else named_netz(settings.netz)
    X, k, Nf = table if table is not None else netz_table(netz)
    base = A.exact_is(Nf, settings.q, settings.q)
    cur = A.exact_is(Nf, settings.q, settings.qp)
    qopt = A.optimal_qp(Nf, settings.q)
    opt = A.exact_is(Nf, settings.q, qopt) if np.isfinite(qopt) else {"cv2": float("nan")}
    key = (1, netz.n, netz.m, int(round(settings.q * 1e4)), int(round(settings.qp * 1e4)), settings.n, settings.seed)
    r = A.simulate_is(X, netz.m, settings.q, settings.qp, settings.n, key, C.REPS, base["P"])
    ks = r["k"]
    ok = ~np.isnan(ks)
    return Analysis(
        netz, base["P"], base["cv2"], cur["cv2"], base["cv2"] / cur["cv2"] if cur["cv2"] > 0 and np.isfinite(cur["cv2"]) else (0.0 if np.isinf(cur["cv2"]) else float("nan")),
        qopt, opt["cv2"], min_cut(Nf), A.draws_needed(base["cv2"]), A.draws_needed(cur["cv2"]), settings.n / (1.0 + cur["cv2"]) if np.isfinite(cur["cv2"]) else 0.0,
        float(np.nanmedian(r["rel"])) if base["P"] > 0 else float("nan"), float(np.mean(r["rel"] > 0.5)) if base["P"] > 0 else float("nan"), r["zero"], r["cover"], float(np.median(r["ess"])),
        float(np.median(ks[ok])) if ok.any() else float("nan"), float(np.mean(ks[ok] > C.K_UNRELIABLE)) if ok.any() else float("nan"), tuple(float(x) for x in r["est"]))


def verdict_main(a: Analysis, settings: Settings) -> tuple[str, str]:
    """Trägt die gewählte Hilfsverteilung, und was sagen ESS und k̂?"""
    if not (a.P > 0):
        return "info", "Das Netz fällt bei diesen Einstellungen nie aus: das Ereignis ist unmöglich, es gibt nichts zu schätzen."
    kopf = (f"Die Ausfallwahrscheinlichkeit ist {de_e(a.P)}; einfaches Ziehen hat CV² = {dz(a.cv2_plain)} und braucht für 10 % relativen Fehler rund {dz(a.n_plain)} Ziehungen. "
            f"Mit der Kippung q' = {de(settings.qp, 3)} ist CV² = {dz(a.cv2_is) if np.isfinite(a.cv2_is) else 'unendlich'} (Gewinn {dz(a.gain)}, {dz(a.n_is)} Ziehungen für 10 %); "
            f"die optimale Kippung wäre q' = {de(a.qp_opt, 3)} (kleinster Schnitt {kanten(a.min_cut)} von {a.netz.m}, also {de(a.min_cut / a.netz.m, 3)}).")
    mess = f" Gemessen mit N = {de(settings.n, 0)}: mediane Abweichung {de(a.med_rel * 100, 0)} %, {de(a.zero * 100, 0)} % der Läufe ohne einen Ausfall, Überdeckung {de(a.cover, 2)}; ESS {de(a.ess_meas, 1)} von {de(settings.n, 0)}."
    if a.gain < 1.0:
        return "warning", kopf + mess + " Die Kippung ist schlechter als einfaches Ziehen: die Gewichte sind so ungleich, dass wenige Ziehungen alles bestimmen."
    if abs(settings.qp - settings.q) < 1e-12:
        return "warning", kopf + mess + " Ohne Kippung sehen die meisten Läufe keinen oder kaum einen Ausfall."
    if np.isfinite(a.k_med) and a.k_med > C.K_UNRELIABLE:
        return "success", kopf + mess + f" Pareto-k̂ liegt im Median bei {de(a.k_med, 1)} und damit über {de(C.K_UNRELIABLE, 1)}, obwohl die Kippung gut ist: bei den diskreten Gewichten dieser Demo taugt k̂ nicht, ESS ist die Diagnose."
    return "success", kopf + mess


# --- Experiment 1: exakte Kurve gegen q' --------------------------------------------------------------------------------------------------------

def qp_curve(settings: Settings, netz: Netz, table) -> dict:
    """CV^2 gegen die Kippung q' (exakt, ohne Zufall) auf einem Raster, mit Optimum."""
    _, _, Nf = table
    grid = np.unique(np.concatenate([np.geomspace(max(settings.q / 10, 1e-4), 0.9, 90), [settings.q, settings.qp]]))
    rows = [{"qp": float(x), "cv2": A.exact_is(Nf, settings.q, float(x))["cv2"]} for x in grid]
    qopt = A.optimal_qp(Nf, settings.q)
    return {"rows": rows, "qp_opt": qopt, "cv2_opt": A.exact_is(Nf, settings.q, qopt)["cv2"] if np.isfinite(qopt) else float("nan"), "cv2_plain": A.exact_is(Nf, settings.q, settings.q)["cv2"],
            "min_cut": min_cut(Nf), "m": netz.m}


def verdict_qp_curve(exp: dict, settings: Settings) -> tuple[str, str]:
    if not np.isfinite(exp["qp_opt"]):
        return "info", "Das Ereignis ist unmöglich, es gibt keine Kurve."
    rows = {round(r["qp"], 9): r for r in exp["rows"]}
    far = max((r for r in exp["rows"] if r["qp"] <= 0.75), key=lambda r: r["qp"])
    return "info", (f"Optimum bei q' = {de(exp['qp_opt'], 3)} mit CV² = {dz(exp['cv2_opt'])} gegen {dz(exp['cv2_plain'])} beim einfachen Ziehen; der kleinste Schnitt hat {kanten(exp['min_cut'])} von {exp['m']} "
                    f"({de(exp['min_cut'] / exp['m'], 3)}). Zu schwach gekippt (q' kleiner als q) ist schlechter als einfaches Ziehen, bei q' = {de(far['qp'], 2)} ist CV² = {dz(far['cv2'])}: zu viel Kippung entartet (hier ab q' ≈ 0,4 schlimmer als gar keine).")


# --- Experiment 2: Seltenheit (exakt) -----------------------------------------------------------------------------------------------------------

def rarity(settings: Settings, netz: Netz, table) -> dict:
    """Für jedes q (Seltenheit): P, CV^2 einfach, optimales q', CV^2 dort, Gewinn und Ziehungen für 10 % (exakt)."""
    _, _, Nf = table
    rows = []
    for q in C.Q_CHOICES:
        base = A.exact_is(Nf, q, q)
        qo = A.optimal_qp(Nf, q)
        opt = A.exact_is(Nf, q, qo)
        rows.append({"q": q, "P": base["P"], "cv2_plain": base["cv2"], "qp_opt": qo, "cv2_opt": opt["cv2"], "gain": base["cv2"] / opt["cv2"], "n_plain": A.draws_needed(base["cv2"]), "n_opt": A.draws_needed(opt["cv2"])})
    return {"rows": rows, "min_cut": min_cut(Nf), "m": netz.m}


def verdict_rarity(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    a, b = rows[0], rows[-1]
    return "info", (f"Bei q = {de(a['q'], 3)} gewinnt die optimale Kippung nur das {dz(a['gain'])}-Fache, bei q = {de(b['q'], 3)} das {dz(b['gain'])}-Fache: "
                    f"einfaches Ziehen braucht dort {dz(b['n_plain'])} Ziehungen für 10 % Fehler, die Kippung {dz(b['n_opt'])}. Das optimale q' bleibt dabei zwischen {de(min(r['qp_opt'] for r in rows), 3)} und {de(max(r['qp_opt'] for r in rows), 3)} "
                    f"(kleinster Schnitt {kanten(exp['min_cut'])} von {exp['m']}): für kleine q gilt das, ohne dieses Wissen muss man q' lernen (Stück 8).")


# --- Experiment 3: Simulation gegen q' --------------------------------------------------------------------------------------------------------

def sweep(settings: Settings, netz: Netz, table) -> dict:
    """Simulation bei q' = q (einfach), dem optimalen q' und großen Kippungen: mediane Abweichung, Anteil Nullschätzungen, Überdeckung, ESS und Pareto-k̂."""
    X, k, Nf = table
    base = A.exact_is(Nf, settings.q, settings.q)
    qopt = A.optimal_qp(Nf, settings.q)
    rows = []
    if not np.isfinite(qopt):
        return {"rows": rows, "qp_opt": qopt, "P": 0.0}
    cands = [settings.q, qopt / 2, qopt, qopt * 2] + list(C.SWEEP_ABS)
    for qp in cands:
        key = (2, netz.n, netz.m, int(round(settings.q * 1e4)), int(round(qp * 1e4)), settings.n, settings.seed)
        r = A.simulate_is(X, netz.m, settings.q, float(qp), settings.n, key, C.SWEEP_REPS, base["P"])
        ks = r["k"]
        ok = ~np.isnan(ks)
        cv2 = A.exact_is(Nf, settings.q, float(qp))["cv2"]
        rows.append({"qp": float(qp), "cv2": cv2, "med_rel": float(np.median(r["rel"])), "frac_gt50": float(np.mean(r["rel"] > 0.5)), "zero": r["zero"], "cover": r["cover"],
                     "ess": float(np.median(r["ess"])), "ess_exact": settings.n / (1 + cv2), "k_med": float(np.median(ks[ok])) if ok.any() else float("nan"),
                     "frac_k_bad": float(np.mean(ks[ok] > C.K_UNRELIABLE)) if ok.any() else float("nan")})
    return {"rows": rows, "qp_opt": qopt, "P": base["P"]}


def verdict_sweep(exp: dict, settings: Settings) -> tuple[str, str]:
    rows = exp["rows"]
    if not rows:
        return "info", "Das Ereignis ist unmöglich, es gibt nichts zu messen."
    plain, opt = rows[0], rows[2]
    far = rows[-1]
    return "warning", (f"Einfaches Ziehen (q' = q): {de(plain['zero'] * 100, 0)} % der Läufe ohne Ausfall, Überdeckung {de(plain['cover'], 2)}. Mit dem optimalen q' = {de(opt['qp'], 3)}: mediane Abweichung {de(opt['med_rel'] * 100, 0)} %, "
                       f"Überdeckung {de(opt['cover'], 2)}, ESS {de(opt['ess'], 1)} (Formel {de(opt['ess_exact'], 1)}). Bei q' = {de(far['qp'], 2)}: Abweichung {de(far['med_rel'] * 100, 0)} %, Überdeckung {de(far['cover'], 2)}. "
                       f"Pareto-k̂ liegt im Netz auch beim optimalen q' bei {dn(opt['k_med'], 1)} und in {de(opt['frac_k_bad'] * 100, 0) if np.isfinite(opt['frac_k_bad']) else '–'} % der Läufe über {de(C.K_UNRELIABLE, 1)}: bei den diskreten Gewichten dieser Demo taugt es nicht als Warnung.")


# --- Experiment 4: Genauigkeit und Ziehungen (exakt) -----------------------------------------------------------------------------------------------

PRECISION = (0.5, 0.2, 0.1, 0.05)


def precision(settings: Settings, netz: Netz, table) -> dict:
    """Ziehungen für relativen Fehler 50 / 20 / 10 / 5 % bei einfachem Ziehen, der gewählten und der optimalen Kippung (exakt: CV^2 / Fehler^2), und die ESS, die dafür nötig ist (1 / Fehler^2)."""
    _, _, Nf = table
    base = A.exact_is(Nf, settings.q, settings.q)
    cur = A.exact_is(Nf, settings.q, settings.qp)
    qopt = A.optimal_qp(Nf, settings.q)
    opt = A.exact_is(Nf, settings.q, qopt) if np.isfinite(qopt) else {"cv2": float("nan")}
    rows = [{"relerr": r, "n_plain": A.draws_needed(base["cv2"], r), "n_cur": A.draws_needed(cur["cv2"], r), "n_opt": A.draws_needed(opt["cv2"], r), "ess_needed": 1.0 / r**2} for r in PRECISION]
    return {"rows": rows, "P": base["P"], "qp_opt": qopt}


def verdict_precision(exp: dict, settings: Settings) -> tuple[str, str]:
    if not (exp["P"] > 0):
        return "info", "Das Ereignis ist unmöglich, es gibt nichts zu messen."
    r = next(x for x in exp["rows"] if abs(x["relerr"] - 0.1) < 1e-9)
    return "info", (f"Für 10 % relativen Fehler braucht einfaches Ziehen {dz(r['n_plain'])} Ziehungen, die gewählte Kippung q' = {de(settings.qp, 3)} {dz(r['n_cur'])}, die optimale (q' = {de(exp['qp_opt'], 3)}) {dz(r['n_opt'])}. "
                    f"Jede Zeile heißt auch: man braucht eine effektive Stichprobengröße von etwa 1 / Fehler² = {dz(r['ess_needed'])} Gewichten, die wirklich zählen; mit N = {de(settings.n, 0)} Ziehungen ist die ESS höchstens {de(settings.n, 0)}.")


# --- Experiment 5: stetiger Begleiter -------------------------------------------------------------------------------------------------------------

def continuous(settings: Settings) -> dict:
    """Exponentialschwanz P(X > t): exakte CV^2 und exaktes k gegen theta, Simulation (k̂, Überdeckung, Fehler) bei CONT_THETAS."""
    t = settings.t
    rows = []
    for th in C.CONT_THETAS:
        r = A.cont_simulate(t, th, C.CONT_N, (3, int(round(t * 100)), int(round(th * 100)), settings.seed), C.CONT_REPS)
        ks = r["k"]
        ok = ~np.isnan(ks)
        rows.append({"theta": th, "cv2": A.cont_cv2(th, t), "k_exact": A.cont_exact_k(th), "k_med": float(np.median(ks[ok])) if ok.any() else float("nan"), "ess_frac": float(np.median(r["ess"])) / C.CONT_N,
                     "med_rel": float(np.median(r["rel"])), "frac_gt50": float(np.mean(r["rel"] > 0.5)), "cover": r["cover"], "hits": float(np.mean(ks[ok] > C.K_UNRELIABLE)) if ok.any() else float("nan")})
    return {"t": t, "P": float(np.exp(-t)), "rows": rows, "theta_opt": A.cont_optimal_theta(t), "cv2_opt": A.cont_cv2(A.cont_optimal_theta(t), t)}


def verdict_continuous(exp: dict) -> tuple[str, str]:
    rows = {r["theta"]: r for r in exp["rows"]}
    inf = rows[2.0]
    return "info", (f"P(X > {de(exp['t'], 1)}) = {de(exp['P'], 4)}; optimal wäre θ = {de(exp['theta_opt'], 2)} (CV² = {de(exp['cv2_opt'], 1)}). Bei θ = 2 ist die Varianz **unendlich**, aber mit N = {de(C.CONT_N, 0)} deckt das Intervall noch "
                    f"{de(inf['cover'], 2)} ab: das Intervall bemerkt es nicht. Pareto-k̂ liegt dort im Median bei {de(inf['k_med'], 2)} (exakt {de(inf['k_exact'], 2)}), an der Marke 0,5; bei θ = 3 ist die Überdeckung "
                    f"{de(rows[3.0]['cover'], 2)}.".replace("**", ""))


# --- Spielwiese ---------------------------------------------------------------------------------------------------------------------------

def custom_result(settings: Settings, netz: Netz) -> dict:
    """Ein eigenes Netz: Analyse mit der gewählten Kippung und die exakte Kurve."""
    table = netz_table(netz)
    a = analyse(settings, netz=netz, table=table)
    return {"analysis": a, "curve": qp_curve(settings, netz, table)}
