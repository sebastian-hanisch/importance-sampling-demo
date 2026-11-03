"""Randfälle: eigene Netze, unmögliche Ereignisse, q' an den Grenzen, kleinste Größen, Eingabefehler."""

import math

import numpy as np
import pytest

import imp_algorithm as A
import imp_constants as C
import imp_evaluation as ev
from imp_scenario import grid_netz, parse_edges


def _s(**kw):
    return ev.Settings(**kw)


def test_eingabefehler_der_spielwiese():
    with pytest.raises(ValueError, match="Schleife"):
        parse_edges("0-0")
    with pytest.raises(ValueError, match="Keine Kanten"):
        parse_edges("\n# nur Kommentar\n")
    with pytest.raises(ValueError, match="Doppelte"):
        parse_edges("0-1\n1-0")
    with pytest.raises(ValueError, match="erwartet zwei ganze Zahlen"):
        parse_edges("0-a")
    with pytest.raises(ValueError, match="höchstens"):
        parse_edges("\n".join(f"{i}-{i + 1}" for i in range(C.CUSTOM_EDGE_LIMIT + 1)))
    with pytest.raises(ValueError, match="Knoten"):
        parse_edges("\n".join(f"{2 * i}-{2 * i + 1}" for i in range(8)))


def test_pfad_hat_jeden_schnitt_von_einer_kante_und_exakte_wahrscheinlichkeit():
    netz = parse_edges("0-1\n1-2\n2-3")
    a = ev.analyse(_s(q=0.01, qp=0.3, n=300), netz=netz)
    assert a.P == pytest.approx(1 - 0.99**3) and a.min_cut == 1
    assert a.qp_opt == pytest.approx(1 / 3, abs=0.06)                                   # kleinster Schnitt 1 von 3 Kanten


def test_q_prime_an_den_reglergrenzen_laeuft():
    for qp in (C.QP_MIN, C.QP_MAX):
        a = ev.analyse(_s(qp=qp))
        assert a.P > 0 and len(a.est) == C.REPS


def test_alle_netze_und_alle_q_und_n_laufen_ohne_fehler():
    for netz_key in C.NETZE:
        for q in (C.Q_MIN, C.Q_MAX):
            for n in (C.N_MIN, C.N_MAX):
                a = ev.analyse(_s(netz=netz_key, q=q, n=n, qp=0.15))
                assert math.isfinite(a.P) and a.P > 0 and 0.0 <= a.zero <= 1.0


def test_ein_netz_dessen_ausfall_unmoeglich_ist_gibt_ein_ehrliches_verdikt():
    netz = grid_netz(2, 3)
    X = np.ones(1 << netz.m)                                                            # kein Ausfall in keiner Menge
    k = np.array([bin(x).count("1") for x in range(1 << netz.m)])
    Nf = A.failure_counts(X, k, netz.m)
    a = ev.analyse(_s(), netz=netz, table=(X, k, Nf))
    assert a.P == 0.0 and math.isnan(a.med_rel) and math.isnan(a.qp_opt)
    kind, text = ev.verdict_main(a, _s())
    assert kind == "info" and "nie aus" in text


def test_kleinstes_n_und_wenige_ausfaelle_geben_nicht_bestimmbar_statt_fehler():
    a = ev.analyse(_s(netz="3x4", q=0.001, qp=0.001, n=100))
    assert a.zero > 0.9 and math.isnan(a.k_med)
    _, text = ev.verdict_main(a, _s(netz="3x4", q=0.001, qp=0.001, n=100))
    assert "nan" not in text


def test_q_prime_gleich_null_oder_eins_ist_unendlich_in_der_exakten_rechnung(gitter34):
    netz, (X, k, Nf) = gitter34
    assert A.exact_is(Nf, 0.01, 0.0)["cv2"] == float("inf") and A.exact_is(Nf, 0.01, 1.0)["cv2"] == float("inf")


def test_aufzaehlung_ueber_dem_limit_wird_abgelehnt():
    with pytest.raises(ValueError, match="aufgezählt"):
        ev.netz_table(grid_netz(3, 5))


def test_stetiger_begleiter_an_den_grenzen():
    ex = ev.continuous(_s(t=C.T_MIN))
    assert len(ex["rows"]) == len(C.CONT_THETAS) and ex["P"] == pytest.approx(math.exp(-C.T_MIN))
    r = A.cont_simulate(C.T_MAX, 0.5, 200, (31,), 5)
    assert np.all(np.isfinite(r["est"]))
