"""Eigenschaftstests (ORACLE-PLAYBOOK §5): Aussagen, die für jede zulässige Eingabe gelten müssen."""

from math import comb

import numpy as np
import pytest

import imp_algorithm as A
import imp_evaluation as ev


def test_gewichte_summieren_sich_unter_der_hilfsverteilung_zur_eins():
    # Summe über alle Konfigurationen von P'(x) * w(x) = Summe P(x) = 1 (Importance-Identität), für beliebige q, q'
    for m in (2, 7, 17):
        for q, qp in ((0.01, 0.12), (0.1, 0.5), (0.3, 0.05)):
            s = sum(comb(m, f) * qp**f * (1 - qp) ** (m - f) * (q / qp) ** f * ((1 - q) / (1 - qp)) ** (m - f) for f in range(m + 1))
            assert s == pytest.approx(1.0, abs=1e-12)


def test_einfaches_ziehen_hat_cv2_gleich_eins_minus_p_durch_p(gitter34):
    netz, (X, k, Nf) = gitter34
    for q in (0.1, 0.01, 0.001):
        e = A.exact_is(Nf, q, q)
        assert e["cv2"] == pytest.approx((1 - e["P"]) / e["P"], rel=1e-9)
        assert A.exact_is(Nf, q, q)["M2"] == pytest.approx(e["P"], rel=1e-12)           # alle Gewichte 1


def test_cv2_nie_negativ_und_nur_bei_sonderverteilung_null(gitter34):
    netz, (X, k, Nf) = gitter34
    for q in (0.1, 0.01):
        for qp in np.geomspace(0.002, 0.95, 40):
            assert A.exact_is(Nf, q, float(qp))["cv2"] > 0.0


def test_optimales_qp_liegt_zwischen_q_und_einer_haelfte_und_ist_nahe_schnitt_durch_kanten(gitter34, hantel):
    for (netz, (X, k, Nf)), erwartet in ((gitter34, 2 / 17), (hantel, 1 / 7)):
        for q in (0.01, 0.001):
            qo = A.optimal_qp(Nf, q)
            assert q < qo < 0.5 and qo == pytest.approx(erwartet, abs=0.04)
            assert A.exact_is(Nf, q, qo)["cv2"] <= min(A.exact_is(Nf, q, float(x))["cv2"] for x in np.linspace(0.02, 0.6, 30)) + 1e-9


def test_gewinn_waechst_mit_der_seltenheit(gitter34):
    netz, (X, k, Nf) = gitter34
    gains = []
    for q in (0.1, 0.05, 0.01, 0.005, 0.001):
        base = A.exact_is(Nf, q, q)["cv2"]
        gains.append(base / A.exact_is(Nf, q, A.optimal_qp(Nf, q))["cv2"])
    assert all(a < b for a, b in zip(gains, gains[1:]))


def test_zu_schwache_und_zu_starke_kippung_sind_schlechter_als_einfaches_ziehen(gitter34):
    netz, (X, k, Nf) = gitter34
    q = 0.01
    plain = A.exact_is(Nf, q, q)["cv2"]
    assert A.exact_is(Nf, q, 0.5)["cv2"] > plain and A.exact_is(Nf, q, 0.7)["cv2"] > 1000 * plain
    assert A.exact_is(Nf, q, 0.003)["cv2"] > plain                                      # q' kleiner als q
    assert A.exact_is(Nf, q, 0.12)["cv2"] < plain / 20


def test_ess_ist_nie_groesser_als_n_und_bei_einfachem_ziehen_die_zahl_der_ausfaelle(hantel):
    netz, (X, k, Nf) = hantel
    P = A.exact_is(Nf, 0.1, 0.1)["P"]
    r = A.simulate_is(X, netz.m, 0.1, 0.1, 200, (21, 1), 50, P)
    assert np.all(r["ess"] <= 200 + 1e-9)
    assert np.median(r["ess"]) == pytest.approx(200 * P, rel=0.2)                      # alle Gewichte 1: ESS = Zahl der Ausfälle
    assert np.all(np.isnan(r["k"]))                                                    # gleiche Gewichte: k̂ nicht bestimmbar


def test_pareto_k_im_netz_ist_beim_optimalen_qp_nicht_klein(gitter34):
    # Der Befund: bei diskreten Gewichten warnt k̂ auch dort, wo das Verfahren am besten arbeitet
    netz, (X, k, Nf) = gitter34
    q = 0.01
    P = A.exact_is(Nf, q, q)["P"]
    r = A.simulate_is(X, netz.m, q, 0.12, 1000, (22, 1), 120, P)
    assert np.nanmedian(r["k"]) > 0.7


def test_pareto_k_im_stetigen_begleiter_folgt_dem_exakten_wert():
    t = 2.0
    for th, k_exact in ((1.2, 1 / 6), (1.5, 1 / 3), (1.8, 4 / 9)):
        r = A.cont_simulate(t, th, 10000, (23, int(th * 100)), 120)
        assert np.nanmedian(r["k"]) == pytest.approx(k_exact, abs=0.08), th
    r2 = A.cont_simulate(t, 2.5, 10000, (23, 250), 120)
    assert np.nanmedian(r2["k"]) > 0.5 and A.cont_cv2(2.5, t) == float("inf")


def test_stetiger_begleiter_ueberdeckung_faellt_ab_theta_zwei():
    t = 2.0
    cov = {th: A.cont_simulate(t, th, 10000, (24, int(th * 100)), 300)["cover"] for th in (1.2, 2.0, 3.0)}
    assert cov[1.2] > 0.9 and cov[3.0] < 0.85 and cov[3.0] < cov[2.0] < cov[1.2] + 0.03


def test_gleiche_staerke_gleiche_schaetzung_und_anderer_strom_andere(hantel):
    netz, (X, k, Nf) = hantel
    P = A.exact_is(Nf, 0.05, 0.05)["P"]
    a = A.simulate_is(X, netz.m, 0.05, 0.2, 100, (25, 1), 10, P)
    b = A.simulate_is(X, netz.m, 0.05, 0.2, 100, (25, 1), 10, P)
    c = A.simulate_is(X, netz.m, 0.05, 0.2, 100, (25, 2), 10, P)
    assert a["est"].tolist() == b["est"].tolist() and a["est"].tolist() != c["est"].tolist()
    assert ev.min_cut(Nf) == 1
