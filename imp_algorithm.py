"""Kern der Demo (nur numpy).

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):
  Zufallsstrom    stream_seed, uniforms                                SplitMix64, zählerbasiert
  Ziel            connected, enumerate_x, failure_counts               Ereignis A = das Netz hängt NICHT zusammen; N_f = Zahl der Ausfallmengen mit f ausgefallenen Kanten
  Exakt           exact_is, optimal_qp                                  zweites Moment, P, relatives Varianzquadrat CV^2 und optimales q' aus den N_f
  Schätzer        simulate_is                                           Importance Sampling mit der Kippung q': Schätzung, naiver Standardfehler, ESS, Pareto-k̂
  Diagnose        ess, gpdfit, pareto_k                                 effektive Stichprobengröße; Pareto-k̂ nach Zhang und Stephens (2009) in der Form von Vehtari et al.
  Stetiger Begleiter  cont_cv2, cont_exact_k, cont_simulate            P(X > t), X ~ Exp(1), Hilfsverteilung Exp(theta): geschlossene Formel, unendliche Varianz ab theta = 2
Zufall kommt nur über einen übergebenen Strom, nie aus einem globalen Generator.

Jede Kante fällt unabhängig mit Wahrscheinlichkeit q aus; Hilfsverteilung: mit q'. Gewicht einer Ziehung mit f Ausfällen: w(f) = (q/q')^f ((1-q)/(1-q'))^(m-f). Schätzer: Mittel von 1[A] w.
Exakt: M2 = Summe_f N_f [q^f (1-q)^(m-f)]^2 / [q'^f (1-q')^(m-f)] und CV^2 = M2/P^2 - 1 (relatives Varianzquadrat je Ziehung).
"""

from __future__ import annotations

import numpy as np

import imp_constants as C

GAMMA = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1


# --- Zufallsstrom -------------------------------------------------------------------------------------------------------------------------

def mix_int(z: int) -> int:
    """SplitMix64-Mischfunktion mit Python-Ganzzahlen."""
    z &= MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return (z ^ (z >> 31)) & MASK


def _mix_arr(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def stream_seed(*keys: int) -> int:
    """Seed eines Stroms aus einem Schlüsseltupel (Einstellung, Wiederholung, ...)."""
    s = 0x1234_5678_9ABC_DEF0
    for k in keys:
        s = mix_int(s ^ mix_int(int(k) + GAMMA))
    return s


def uniforms(seed: int, count: int) -> np.ndarray:
    """`count` gleichverteilte Zahlen in [0, 1) (53 Bit) aus dem Strom `seed`; Ausgabe i hängt nur von (seed, i) ab."""
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = _mix_arr(np.uint64(seed) + idx * np.uint64(GAMMA))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


# --- Ziel ---------------------------------------------------------------------------------------------------------------------------------

def connected(present: np.ndarray, n: int, edges) -> np.ndarray:
    """Zusammenhang je Zeile (`present` ist (B, m) bool): Kleinste-Marke-Ausbreitung, zusammenhängend genau dann, wenn alle Marken 0 sind."""
    B = present.shape[0]
    if n <= 1:
        return np.ones(B, dtype=bool)
    lab = np.tile(np.arange(n, dtype=np.int16), (B, 1))
    while True:
        before = int(lab.sum())
        for e, (u, v) in enumerate(edges):
            low = np.minimum(lab[:, u], lab[:, v])
            pe = present[:, e]
            lab[:, u] = np.where(pe, low, lab[:, u])
            lab[:, v] = np.where(pe, low, lab[:, v])
        if int(lab.sum()) == before:
            break
    return lab.max(axis=1) == 0


def enumerate_x(n: int, edges, chunk: int = 1 << 15) -> tuple[np.ndarray, np.ndarray]:
    """Alle 2^m Kantenmengen in Maskenreihenfolge (Kante e = Bit e): X (1 = zusammenhängend) und Zahl haltender Kanten k."""
    m = len(edges)
    shifts = np.arange(m, dtype=np.uint32)
    xs, ks = [], []
    for start in range(0, 1 << m, chunk):
        masks = np.arange(start, min(start + chunk, 1 << m), dtype=np.uint32)
        present = ((masks[:, None] >> shifts) & 1).astype(bool)
        xs.append(connected(present, n, edges).astype(np.float64))
        ks.append(present.sum(axis=1))
    return np.concatenate(xs), np.concatenate(ks)


def failure_counts(X: np.ndarray, k: np.ndarray, m: int) -> np.ndarray:
    """N_f (f = 0..m): Zahl der Kantenmengen mit f ausgefallenen Kanten, bei denen das Netz NICHT zusammenhängt."""
    return np.bincount((m - k)[X == 0], minlength=m + 1).astype(float)


# --- Exakte Rechnung ----------------------------------------------------------------------------------------------------------------------

def exact_is(Nf: np.ndarray, q: float, qp: float) -> dict:
    """Exakt: P = Wahrscheinlichkeit des Ausfalls, zweites Moment M2 und CV^2 = (M2 - P^2) / P^2 je Ziehung bei Hilfsverteilung q' (q' = q: einfaches Ziehen).
    Bei q' = 0 oder 1 ist das zweite Moment unendlich (die Hilfsverteilung deckt den Ausfall nicht ab); bei P = 0 sind CV^2 und M2 NaN."""
    m = len(Nf) - 1
    f = np.arange(m + 1)
    pq = q**f * (1.0 - q) ** (m - f)
    P = float(np.sum(Nf * pq))
    if P <= 0:
        return {"P": 0.0, "M2": float("nan"), "cv2": float("nan")}
    if qp <= 0 or qp >= 1:
        return {"P": P, "M2": float("inf"), "cv2": float("inf")}
    pqp = qp**f * (1.0 - qp) ** (m - f)
    M2 = float(np.sum(Nf * pq**2 / pqp))
    return {"P": P, "M2": M2, "cv2": (M2 - P**2) / P**2}


def optimal_qp(Nf: np.ndarray, q: float) -> float:
    """Das q' mit kleinstem CV^2 (Gittersuche in zwei Stufen); NaN, wenn der Ausfall unmöglich ist."""
    if exact_is(Nf, q, q)["P"] <= 0:
        return float("nan")
    grid = np.concatenate([np.geomspace(1e-4, 0.5, 600), np.linspace(0.5, 0.999, 300)])
    best = min(grid, key=lambda x: exact_is(Nf, q, float(x))["cv2"])
    fine = np.linspace(best * 0.7, min(best * 1.4, 0.999), 400)
    return float(min(fine, key=lambda x: exact_is(Nf, q, float(x))["cv2"]))


def draws_needed(cv2: float, relerr: float = C.RELERR) -> float:
    """Ziehungen, damit der relative Fehler (Standardabweichung durch Erwartungswert) relerr erreicht: CV^2 / relerr^2."""
    return cv2 / relerr**2


# --- Diagnose -----------------------------------------------------------------------------------------------------------------------------

def ess(weights: np.ndarray) -> float:
    """Effektive Stichprobengröße (Summe w)^2 / Summe w^2; 0 bei lauter Nullgewichten."""
    s = float(weights.sum())
    s2 = float(np.sum(weights**2))
    return s * s / s2 if s2 > 0 else 0.0


def gpdfit(x: np.ndarray) -> tuple[float, float]:
    """Verallgemeinerte Pareto-Verteilung nach Zhang und Stephens (2009), Form von Vehtari et al.: x aufsteigend sortierte Überschreitungen > 0.
    Rückgabe (k, sigma); k > 0 heißt schwerer Rand (k >= 1/2: unendliche Varianz). k wird zum Prior 1/2 hin regularisiert (Gewicht 10)."""
    n = len(x)
    prior = 3
    m = 30 + int(np.sqrt(n))
    bs = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    bs = bs / (prior * x[int(n / 4 + 0.5) - 1]) + 1.0 / x[-1]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ks = np.mean(np.log1p(-bs[:, None] * x[None, :]), axis=1)
        L = n * (np.log(-bs / ks) - ks - 1.0)
        w = 1.0 / np.sum(np.exp(L[None, :] - L[:, None]), axis=1)
    b = float(np.sum(bs * w))
    k = float(np.mean(np.log1p(-b * x)))
    sigma = -k / b
    return (n * k + 10 * 0.5) / (n + 10), sigma


def pareto_k(weights: np.ndarray) -> float:
    """k̂ aus den größten positiven Gewichten (PSIS: M = min(0,2 N, 3 sqrt N) Gewichte oberhalb des (M+1)-größten); NaN bei weniger als 10 positiven Gewichten oder lauter gleichen Gewichten."""
    w = np.sort(weights[weights > 0])
    N = len(weights)
    M = int(min(0.2 * N, 3 * np.sqrt(N)))
    if len(w) < 10 or M < 5:
        return float("nan")
    M = min(M, len(w) - 1)
    tail = w[-M:] - w[-M - 1]
    if tail.max() <= 0:
        return float("nan")                                  # lauter gleiche Gewichte: kein Rand
    tail = np.maximum(tail, 1e-12 * w.max())
    return gpdfit(np.sort(tail))[0]


# --- Schätzer im Netz ---------------------------------------------------------------------------------------------------------------------

def simulate_is(X: np.ndarray, m: int, q: float, qp: float, N: int, key: tuple, reps: int, truth: float) -> dict:
    """Je Wiederholung (fester Strom aus `key` und der Nummer) N Ziehungen mit Ausfallwahrscheinlichkeit q' je Kante; Schätzung = Mittel von 1[Ausfall] w, naiver Standardfehler,
    ESS und Pareto-k̂ der Gewichte; Überdeckung des Wald-Intervalls und Anteil der Wiederholungen ohne einen einzigen Ausfall."""
    est, se, es, ks = np.empty(reps), np.empty(reps), np.empty(reps), np.empty(reps)
    shifts = np.arange(m, dtype=np.int64)
    for rep in range(reps):
        u = uniforms(stream_seed(*key, rep), N * m).reshape(N, m)
        fail = u < qp
        mask = ((~fail).astype(np.int64) << shifts).sum(axis=1)                     # Bit e = Kante e hält
        f = fail.sum(axis=1)
        w = np.where(X[mask] == 0, (q / qp) ** f * ((1.0 - q) / (1.0 - qp)) ** (m - f), 0.0)
        est[rep] = w.mean()
        se[rep] = w.std(ddof=1) / np.sqrt(N)
        es[rep] = ess(w)
        ks[rep] = pareto_k(w)
    rel = np.abs(est / truth - 1.0) if truth > 0 else np.full(reps, np.nan)
    return {"est": est, "se": se, "ess": es, "k": ks, "rel": rel, "zero": float(np.mean(est == 0)), "cover": float(np.mean(np.abs(est - truth) <= C.Z95 * se))}


# --- Stetiger Begleiter -------------------------------------------------------------------------------------------------------------------

def cont_cv2(theta: float, t: float) -> float:
    """CV^2 = e^{theta t} / (theta (2 - theta)) - 1 für theta < 2, sonst unendlich (P(X > t) mit X ~ Exp(1), Hilfsverteilung Exp(theta))."""
    if theta <= 0 or theta >= 2:
        return float("inf")
    return float(np.exp(theta * t) / (theta * (2.0 - theta)) - 1.0)


def cont_exact_k(theta: float) -> float:
    """Formparameter k = (theta - 1) / theta des Gewichtsrandes für theta > 1 (Randindex theta / (theta - 1)); für theta <= 1 sind die Gewichte beschränkt (NaN)."""
    return (theta - 1.0) / theta if theta > 1 else float("nan")


def cont_optimal_theta(t: float) -> float:
    grid = np.linspace(0.02, 1.99, 4000)
    return float(grid[int(np.argmin([cont_cv2(float(x), t) for x in grid]))])


def cont_simulate(t: float, theta: float, N: int, key: tuple, reps: int) -> dict:
    """Je Wiederholung N Ziehungen x = -ln(1-u)/theta (Exp(theta)); Gewicht e^{-(1-theta) x} / theta auf x > t; Schätzung, Standardfehler, ESS, k̂, Überdeckung."""
    P = float(np.exp(-t))
    est, se, es, ks = np.empty(reps), np.empty(reps), np.empty(reps), np.empty(reps)
    for rep in range(reps):
        u = uniforms(stream_seed(*key, rep), N)
        x = -np.log1p(-u) / theta
        w = np.where(x > t, np.exp(-(1.0 - theta) * x) / theta, 0.0)
        est[rep], se[rep], es[rep], ks[rep] = w.mean(), w.std(ddof=1) / np.sqrt(N), ess(w), pareto_k(w)
    rel = np.abs(est / P - 1.0)
    return {"P": P, "est": est, "se": se, "ess": es, "k": ks, "rel": rel, "cover": float(np.mean(np.abs(est - P) <= C.Z95 * se))}
