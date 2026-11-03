"""Orakel-Tests: andere Rechenwege als die der Demo.

- Ausfallzählung gegen networkx auf allen Kantenmengen kleiner Netze.
- Zweites Moment durch DIREKTE Summation über alle 2^m Kantenmengen (Brüche) gegen die Formel über die N_f.
- Pareto-k̂ gegen scipy genpareto bei bekannten Formparametern; stetiges zweites Moment gegen numerische Integration (scipy quad).
- Simulation: Erwartungstreue, Varianz gegen die exakte, Anteil der Nullschätzungen gegen (1 - P)^N, ESS gegen N / (1 + CV^2).
"""

from fractions import Fraction

import numpy as np
import pytest

import imp_algorithm as A
import imp_evaluation as ev
from imp_scenario import grid_netz, hantel_netz

nx = pytest.importorskip("networkx")
scipy_stats = pytest.importorskip("scipy.stats")
scipy_integrate = pytest.importorskip("scipy.integrate")


def x_networkx(n, edges, mask):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(e for i, e in enumerate(edges) if (mask >> i) & 1)
    return nx.is_connected(g)


@pytest.mark.parametrize("netz", [hantel_netz(), grid_netz(3, 3)], ids=["hantel", "3x3"])
def test_ausfallzaehlung_gegen_networkx(netz):
    X, k, Nf = ev.netz_table(netz)
    soll = np.zeros(netz.m + 1)
    for mask in range(1 << netz.m):
        assert bool(X[mask]) == x_networkx(netz.n, netz.edges, mask), mask
        if not x_networkx(netz.n, netz.edges, mask):
            soll[netz.m - bin(mask).count("1")] += 1
    assert Nf.tolist() == soll.tolist()


@pytest.mark.parametrize("netz", [hantel_netz(), grid_netz(2, 3)], ids=["hantel", "2x3"])
@pytest.mark.parametrize("q,qp", [(Fraction(1, 10), Fraction(1, 5)), (Fraction(1, 100), Fraction(3, 20)), (Fraction(1, 2), Fraction(1, 2))])
def test_zweites_moment_gegen_summation_ueber_alle_kantenmengen(netz, q, qp):
    X, k, Nf = ev.netz_table(netz)
    m = netz.m
    P = Fraction(0)
    M2 = Fraction(0)
    for mask in range(1 << m):
        if X[mask] == 0:
            f = m - bin(mask).count("1")
            pq = q**f * (1 - q) ** (m - f)
            pqp = qp**f * (1 - qp) ** (m - f)
            P += pq
            M2 += pq * pq / pqp
    e = A.exact_is(Nf, float(q), float(qp))
    assert e["P"] == pytest.approx(float(P), rel=1e-12) and e["M2"] == pytest.approx(float(M2), rel=1e-12)
    assert e["cv2"] == pytest.approx(float((M2 - P * P) / (P * P)), rel=1e-9)


@pytest.mark.parametrize("xi", [0.2, 0.5, 1.0])
def test_pareto_k_gegen_bekannte_formparameter(xi):
    rng = np.random.default_rng(int(xi * 100))
    ks = [A.pareto_k(scipy_stats.genpareto.rvs(xi, size=20000, random_state=rng.integers(1 << 31))) for _ in range(30)]
    assert np.median(ks) == pytest.approx(xi, abs=0.08)


def test_pareto_k_gegen_scipy_maximum_likelihood():
    rng = np.random.default_rng(3)
    x = scipy_stats.genpareto.rvs(0.4, size=5000, random_state=7)
    c, _, _ = scipy_stats.genpareto.fit(x, floc=0)
    assert A.gpdfit(np.sort(x))[0] == pytest.approx(c, abs=0.1)
    assert rng is not None


@pytest.mark.parametrize("theta,t", [(1.2, 2.0), (1.5, 3.0), (0.5, 2.0)])
def test_stetiges_zweites_moment_gegen_numerische_integration(theta, t):
    f = lambda x: np.exp(-2.0 * x + theta * x) / theta                                    # w(x)^2 g(x) = f(x)^2 / g(x) mit f = e^-x, g = theta e^(-theta x), als Exponent geschrieben
    m2, _ = scipy_integrate.quad(f, t, np.inf)
    P = np.exp(-t)
    assert A.cont_cv2(theta, t) == pytest.approx((m2 - P**2) / P**2, rel=1e-6)


def test_stetiges_zweites_moment_divergiert_ab_theta_zwei():
    f = lambda x, th: np.exp(-2.0 * x + th * x) / th
    # Teilintegrale bis X wachsen unbeschränkt für theta >= 2 und konvergieren für theta < 2
    def partial(th, upper):
        return scipy_integrate.quad(f, 2.0, upper, args=(th,))[0]
    assert partial(2.5, 60) > 1e6 * partial(2.5, 20)                                      # wächst wie e^{0,5 U}
    assert partial(1.5, 60) == pytest.approx(partial(1.5, 200), rel=1e-6)
    assert A.cont_cv2(2.0, 2.0) == float("inf")


def test_simulation_erwartungstreu_und_varianz_gegen_exakt(hantel):
    netz, (X, k, Nf) = hantel
    q, qp, N, R = 0.01, 0.14, 200, 3000
    e = A.exact_is(Nf, q, qp)
    r = A.simulate_is(X, netz.m, q, qp, N, (11, 1), R, e["P"])
    z = (r["est"].mean() - e["P"]) / (r["est"].std(ddof=1) / np.sqrt(R))
    assert abs(z) < 4.0
    assert r["est"].var(ddof=1) == pytest.approx(e["P"] ** 2 * e["cv2"] / N, rel=0.12)
    # ESS im Mittel nahe N / (1 + CV^2)
    assert np.median(r["ess"]) == pytest.approx(N / (1 + e["cv2"]), rel=0.25)


def test_anteil_der_nullschaetzungen_gegen_exakte_formel(gitter34):
    netz, (X, k, Nf) = gitter34
    q, N, R = 0.01, 300, 2000
    P = A.exact_is(Nf, q, q)["P"]
    r = A.simulate_is(X, netz.m, q, q, N, (12, 1), R, P)
    assert r["zero"] == pytest.approx((1 - P) ** N, abs=4 * np.sqrt((1 - P) ** N * (1 - (1 - P) ** N) / R))


def test_zu_starke_kippung_ist_in_der_simulation_schlechter_als_einfaches_ziehen(gitter34):
    netz, (X, k, Nf) = gitter34
    q, N, R = 0.05, 500, 600
    P = A.exact_is(Nf, q, q)["P"]
    plain = A.simulate_is(X, netz.m, q, q, N, (13, 1), R, P)
    opt = A.simulate_is(X, netz.m, q, 0.136, N, (13, 2), R, P)
    far = A.simulate_is(X, netz.m, q, 0.6, N, (13, 3), R, P)
    assert opt["est"].var(ddof=1) < plain["est"].var(ddof=1) < far["est"].var(ddof=1)
