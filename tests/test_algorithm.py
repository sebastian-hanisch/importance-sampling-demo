"""Kern von Hand geprüft: Zufallsstrom, Ausfallzählung und zweites Moment am Pfad mit Brüchen, Gewichte, ESS, k̂-Randfälle, stetige Formeln."""

from fractions import Fraction

import numpy as np
import pytest

import imp_algorithm as A


def test_uniforms_gegen_die_ganzzahl_referenz():
    seed = A.stream_seed(3, 7)
    soll = [(A.mix_int((seed + i * A.GAMMA) & A.MASK) >> 11) * (1.0 / (1 << 53)) for i in range(1, 6)]
    assert A.uniforms(seed, 5).tolist() == soll


def test_ausgabe_i_haengt_nur_von_seed_und_i_ab():
    assert A.uniforms(5, 100)[:10].tolist() == A.uniforms(5, 10).tolist()


def test_ausfallzaehlung_am_pfad_von_hand(pfad):
    netz, (X, k, Nf) = pfad
    assert Nf.tolist() == [0.0, 2.0, 1.0] and int(X.sum()) == 1                         # nur "beide halten" hängt zusammen


def test_zweites_moment_am_pfad_mit_bruechen(pfad):
    netz, (X, k, Nf) = pfad
    q, qp = Fraction(1, 2), Fraction(3, 4)
    e = A.exact_is(Nf, float(q), float(qp))
    assert e["P"] == pytest.approx(0.75) and e["M2"] == pytest.approx(7 / 9)           # 2/3 + 1/9
    assert e["cv2"] == pytest.approx((7 / 9 - 9 / 16) / (9 / 16))
    plain = A.exact_is(Nf, 0.5, 0.5)
    assert plain["M2"] == pytest.approx(0.75) and plain["cv2"] == pytest.approx(1 / 3)  # alle Gewichte 1: CV^2 = (1 - P) / P


def test_gewichte_am_pfad_von_hand():
    q, qp, m = 0.5, 0.75, 2
    w = lambda f: (q / qp) ** f * ((1 - q) / (1 - qp)) ** (m - f)
    assert w(1) == pytest.approx(4 / 3) and w(2) == pytest.approx(4 / 9) and w(0) == pytest.approx(4.0)
    # Die gekippte Wahrscheinlichkeit mal Gewicht ist die Originalwahrscheinlichkeit: P'(f=1) w(1) = P(f=1)
    assert 2 * qp * (1 - qp) * w(1) == pytest.approx(2 * q * (1 - q))


def test_unendliches_zweites_moment_bei_q_prime_null_oder_eins(pfad):
    _, (X, k, Nf) = pfad
    assert A.exact_is(Nf, 0.1, 0.0)["cv2"] == float("inf") and A.exact_is(Nf, 0.1, 1.0)["cv2"] == float("inf")
    assert np.isnan(A.exact_is(np.zeros(3), 0.1, 0.1)["cv2"])                           # unmögliches Ereignis


def test_ziehungen_fuer_zielgenauigkeit():
    assert A.draws_needed(1.0) == pytest.approx(100.0) and A.draws_needed(4.0, 0.05) == pytest.approx(1600.0)


def test_ess_von_hand():
    assert A.ess(np.array([1.0, 1.0, 1.0, 0.0])) == pytest.approx(3.0)
    assert A.ess(np.array([2.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert A.ess(np.array([1.0, 3.0])) == pytest.approx(16 / 10)
    assert A.ess(np.zeros(5)) == 0.0


def test_pareto_k_randfaelle():
    assert np.isnan(A.pareto_k(np.ones(1000)))                                          # lauter gleiche Gewichte: kein Rand
    assert np.isnan(A.pareto_k(np.zeros(1000)))
    w = np.zeros(1000)
    w[:5] = np.arange(1, 6)
    assert np.isnan(A.pareto_k(w))                                                      # weniger als 10 positive Gewichte


def test_pareto_k_bei_exponentialgewichten_ist_klein():
    rng = np.random.default_rng(1)
    k = A.pareto_k(rng.exponential(1.0, 5000))
    assert -0.2 < k < 0.2                                                               # exponentieller Rand: Formparameter 0


def test_stetige_formeln_von_hand():
    t = 2.0
    assert A.cont_cv2(1.0, t) == pytest.approx(np.exp(t) - 1)                            # θ = 1: einfaches Ziehen, CV^2 = (1 - P) / P
    assert A.cont_cv2(2.0, t) == float("inf") and A.cont_cv2(2.5, t) == float("inf") and A.cont_cv2(0.0, t) == float("inf")
    assert A.cont_cv2(1.5, t) == pytest.approx(np.exp(3.0) / (1.5 * 0.5) - 1)
    assert A.cont_exact_k(2.0) == pytest.approx(0.5) and A.cont_exact_k(3.0) == pytest.approx(2 / 3) and np.isnan(A.cont_exact_k(0.8))
    th = A.cont_optimal_theta(t)
    assert th < 1.0 and A.cont_cv2(th, t) <= min(A.cont_cv2(x, t) for x in (0.2, 0.3, 0.5, 0.7, 1.0))
    assert th == pytest.approx(0.382, abs=0.01)


def test_stetige_ziehung_wie_von_hand():
    # x = -ln(1 - u) / θ; Gewicht auf x > t: e^{-(1-θ) x} / θ
    t, th = 1.0, 1.5
    u = A.uniforms(A.stream_seed(8, 0), 3)
    x = -np.log1p(-u) / th
    w = np.where(x > t, np.exp(-(1 - th) * x) / th, 0.0)
    r = A.cont_simulate(t, th, 3, (8,), 1)
    assert r["est"][0] == pytest.approx(w.mean())
