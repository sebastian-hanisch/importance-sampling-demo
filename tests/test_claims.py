"""Jede Zahl der README, nachgerechnet über dieselben Auswertungsfunktionen wie in der App (DEMO-PLAYBOOK §4).

Exakte Zahlen (Aufzählung, Formeln) mit engen Bändern. Gemessene Zahlen stammen aus festen Strömen und sind reproduzierbar; sie stehen hier mit Bändern, die das Rauschen
(300 bis 400 Wiederholungen, bei wenigen Ausfällen mehr) und kleine Plattformunterschiede der Gleitkommasummen abdecken."""

import numpy as np
import pytest

import imp_algorithm as A
import imp_evaluation as ev
from imp_scenario import named_netz

STANDARD = ev.Settings("3x4", 0.01, 0.12, 1000)


@pytest.fixture(scope="module")
def g():
    netz = named_netz("3x4")
    return netz, ev.netz_table(netz)


@pytest.fixture(scope="module")
def h():
    netz = named_netz("hantel")
    return netz, ev.netz_table(netz)


def test_wie_selten_und_wie_viele_ziehungen_einfach(g):
    Nf = g[1][2]
    e = A.exact_is(Nf, 0.01, 0.01)
    assert e["P"] == pytest.approx(4.17e-4, rel=1e-3) and e["cv2"] == pytest.approx(2397, abs=1)
    assert A.draws_needed(e["cv2"]) == pytest.approx(239737, abs=100)
    assert A.draws_needed(A.exact_is(Nf, 0.001, 0.001)["cv2"]) == pytest.approx(2.49e7, rel=5e-3)


def test_gewinn_und_optimale_kippung_gegen_die_seltenheit(g):
    Nf = g[1][2]
    soll = {0.1: (2.1, 0.176), 0.05: (3.4, 0.137), 0.02: (10.1, 0.121), 0.01: (29.8, 0.119), 0.005: (101, 0.118), 0.002: (572, 0.118), 0.001: (2210, 0.118)}
    for q, (gain, qo) in soll.items():
        base = A.exact_is(Nf, q, q)["cv2"]
        qopt = A.optimal_qp(Nf, q)
        assert qopt == pytest.approx(qo, abs=1.5e-3), q
        assert base / A.exact_is(Nf, q, qopt)["cv2"] == pytest.approx(gain, rel=0.02), q
    assert A.draws_needed(A.exact_is(Nf, 0.01, A.optimal_qp(Nf, 0.01))["cv2"]) == pytest.approx(8057, abs=15)
    assert A.draws_needed(A.exact_is(Nf, 0.001, A.optimal_qp(Nf, 0.001))["cv2"]) == pytest.approx(11263, abs=30)


def test_optimale_kippung_ist_schnitt_durch_kanten(g, h):
    netz3 = named_netz("3x3")
    t3 = ev.netz_table(netz3)
    assert ev.min_cut(g[1][2]) == 2 and A.optimal_qp(g[1][2], 0.001) == pytest.approx(2 / 17, abs=2e-3)
    assert A.optimal_qp(t3[2], 0.001) == pytest.approx(2 / 12, abs=1e-3)
    assert ev.min_cut(h[1][2]) == 1 and A.optimal_qp(h[1][2], 0.001) == pytest.approx(1 / 7, abs=2e-3)
    assert A.exact_is(t3[2], 0.001, 0.001)["cv2"] / A.exact_is(t3[2], 0.001, A.optimal_qp(t3[2], 0.001))["cv2"] == pytest.approx(4681, rel=0.01)
    assert A.exact_is(h[1][2], 0.001, 0.001)["cv2"] / A.exact_is(h[1][2], 0.001, A.optimal_qp(h[1][2], 0.001))["cv2"] == pytest.approx(61.2, rel=0.01)


def test_falsche_kippung_cv2(g):
    Nf = g[1][2]
    soll = {0.003: 37104, 0.01: 2397, 0.12: 80.6, 0.3: 399, 0.5: 22351}
    for qp, cv2 in soll.items():
        assert A.exact_is(Nf, 0.01, qp)["cv2"] == pytest.approx(cv2, rel=2e-3), qp
    assert A.exact_is(Nf, 0.01, 0.7)["cv2"] == pytest.approx(2.42e7, rel=5e-3)


def test_einfaches_ziehen_bei_n_1000(g):
    netz, table = g
    a = ev.analyse(ev.Settings("3x4", 0.01, 0.01, 1000), netz, table)
    assert a.zero == pytest.approx(0.65, abs=0.07) and (1 - a.P) ** 1000 == pytest.approx(0.66, abs=0.01)
    assert a.cover == pytest.approx(0.35, abs=0.07) and a.med_rel == pytest.approx(1.0, abs=1e-9)


def test_standardfall_bei_n_1000(g):
    netz, table = g
    a = ev.analyse(STANDARD, netz, table)
    assert a.gain == pytest.approx(29.75, abs=0.05) and a.min_cut == 2
    assert a.med_rel == pytest.approx(0.19, abs=0.06) and a.zero == 0.0 and a.cover == pytest.approx(0.92, abs=0.05)
    assert a.ess_meas == pytest.approx(12.4, abs=2.0) and a.ess_exact == pytest.approx(12.26, abs=0.02)
    assert 0.0 < a.frac_gt50 < 0.15


def test_zu_stark_gekippt(g):
    netz, table = g
    a = ev.analyse(ev.Settings("3x4", 0.01, 0.5, 1000), netz, table)
    assert a.gain == pytest.approx(0.107, abs=0.005) and a.med_rel == pytest.approx(0.98, abs=0.05) and a.cover == pytest.approx(0.14, abs=0.06) and a.ess_meas == pytest.approx(2.9, abs=1.0)
    b = ev.analyse(ev.Settings("3x4", 0.01, 0.7, 1000), netz, table)
    assert b.cover < 0.08


def test_hantel_bei_n_1000(h):
    netz, table = h
    a = ev.analyse(ev.Settings("hantel", 0.01, 0.14, 1000), netz, table)
    assert a.cv2_plain == pytest.approx(93.4, abs=0.1) and a.cv2_is == pytest.approx(13.1, abs=0.1) and a.gain == pytest.approx(7.15, abs=0.05)
    assert a.med_rel == pytest.approx(0.08, abs=0.04) and a.cover == pytest.approx(0.94, abs=0.05) and a.ess_meas == pytest.approx(71.2, abs=8.0) and a.ess_exact == pytest.approx(71.1, abs=0.2)


def test_ziehungen_fuer_genauigkeit(g):
    netz, table = g
    ex = ev.precision(STANDARD, netz, table)
    rows = {r["relerr"]: r for r in ex["rows"]}
    assert rows[0.5]["n_opt"] == pytest.approx(322, abs=2) and rows[0.2]["n_opt"] == pytest.approx(2014, abs=8) and rows[0.1]["n_opt"] == pytest.approx(8057, abs=30) and rows[0.05]["n_opt"] == pytest.approx(32227, abs=120)
    assert [rows[r]["n_plain"] for r in (0.5, 0.2, 0.1, 0.05)] == pytest.approx([9589, 59934, 239737, 958946], rel=1e-3)


def test_pareto_k_im_netz_ist_beim_optimum_nicht_klein(g, h):
    a = ev.analyse(STANDARD, *g)
    assert a.k_med == pytest.approx(2.6, abs=0.8) and a.frac_k_bad > 0.9
    b = ev.analyse(ev.Settings("hantel", 0.01, 0.14, 1000), *h)
    assert b.k_med > 5.0 and b.frac_k_bad > 0.85
    c = ev.analyse(ev.Settings("3x4", 0.01, 0.01, 1000), *g)
    assert np.isnan(c.k_med)                                                               # einfaches Ziehen: alle Gewichte gleich


def test_stetiger_begleiter():
    ex = ev.continuous(ev.Settings(t=2.0))
    rows = {r["theta"]: r for r in ex["rows"]}
    assert ex["P"] == pytest.approx(0.1353, abs=1e-4) and ex["theta_opt"] == pytest.approx(0.382, abs=0.002) and ex["cv2_opt"] == pytest.approx(2.47, abs=0.01)
    assert [rows[t]["k_exact"] for t in (1.2, 1.5, 1.8, 2.0, 2.5, 3.0)] == pytest.approx([0.167, 0.333, 0.444, 0.5, 0.6, 0.667], abs=1e-3)
    assert [rows[t]["cv2"] for t in (1.2, 1.5, 1.8)] == pytest.approx([10.5, 25.8, 100.7], abs=0.1)
    assert all(rows[t]["cv2"] == float("inf") for t in (2.0, 2.5, 3.0))
    assert [rows[t]["k_med"] for t in (1.2, 1.5, 1.8, 2.0, 2.5, 3.0)] == pytest.approx([0.18, 0.34, 0.44, 0.50, 0.58, 0.60], abs=0.08)
    assert [rows[t]["cover"] for t in (1.2, 1.5, 1.8, 2.0, 2.5, 3.0)] == pytest.approx([0.94, 0.94, 0.93, 0.92, 0.85, 0.80], abs=0.06)
    assert rows[2.0]["cover"] > 0.85 and rows[3.0]["cover"] < rows[1.2]["cover"] - 0.05
