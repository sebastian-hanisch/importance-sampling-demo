"""Verdikte (reine Funktionen in imp_evaluation.py) an handgesetzten Werten: Schwellen und jede Zahl im Text."""

import dataclasses

import imp_evaluation as ev
from imp_scenario import grid_netz

S = ev.Settings()


def _a(**kw) -> ev.Analysis:
    basis = ev.Analysis(
        netz=grid_netz(3, 4), P=4.17e-4, cv2_plain=2397.0, cv2_is=80.6, gain=29.7, qp_opt=0.119, cv2_opt=80.6, min_cut=2, n_plain=239700.0, n_is=8060.0, ess_exact=12.3,
        med_rel=0.2, frac_gt50=0.08, zero=0.0, cover=0.92, ess_meas=12.4, k_med=2.6, frac_k_bad=1.0, est=(0.0004,) * 3)
    return dataclasses.replace(basis, **kw)


def test_erfolg_mit_hinweis_auf_pareto_k_bei_diskreten_gewichten():
    kind, text = ev.verdict_main(_a(), S)
    assert kind == "success"
    for frag in ("4,2e-04", "CV² = 2.397", "rund 239.700 Ziehungen", "q' = 0,120", "CV² = 80,6", "Gewinn 29,7", "8.060 Ziehungen", "q' = 0,119", "kleinster Schnitt 2 Kanten von 17, also 0,118",
                 "mediane Abweichung 20 %", "0 % der Läufe ohne einen Ausfall", "Überdeckung 0,92", "ESS 12,4 von 1.000", "Pareto-k̂ liegt im Median bei 2,6", "ESS ist die Diagnose"):
        assert frag in text, frag


def test_erfolg_ohne_k_hinweis_wenn_k_klein_ist():
    kind, text = ev.verdict_main(_a(k_med=0.4), S)
    assert kind == "success" and "Pareto-k̂" not in text
    assert ev.verdict_main(_a(k_med=float("nan")), S)[0] == "success"


def test_warnung_ohne_kippung():
    kind, text = ev.verdict_main(_a(gain=1.0, zero=0.65, cover=0.35), ev.Settings(q=0.01, qp=0.01))
    assert kind == "warning" and "Ohne Kippung sehen die meisten Läufe keinen" in text and "65 % der Läufe ohne einen Ausfall" in text


def test_warnung_bei_zu_starker_kippung():
    kind, text = ev.verdict_main(_a(gain=0.107, cv2_is=22400.0, n_is=2240000.0), ev.Settings(q=0.01, qp=0.5))
    assert kind == "warning" and "schlechter als einfaches Ziehen" in text and "Gewinn 0,11" in text and "22.400" in text


def test_unendliches_cv2_wird_genannt():
    _, text = ev.verdict_main(_a(gain=0.0, cv2_is=float("inf"), n_is=float("inf")), ev.Settings(q=0.01, qp=0.0))
    assert "unendlich" in text


def test_nie_ein_ausfall():
    kind, text = ev.verdict_main(_a(P=0.0), S)
    assert kind == "info" and "nie aus" in text


def test_kurven_verdikt_zahlen():
    rows = [{"qp": 0.01, "cv2": 2400.0}, {"qp": 0.119, "cv2": 80.6}, {"qp": 0.7, "cv2": 5.16e7}]
    kind, text = ev.verdict_qp_curve({"rows": rows, "qp_opt": 0.119, "cv2_opt": 80.6, "cv2_plain": 2400.0, "min_cut": 2, "m": 17}, S)
    assert kind == "info" and "q' = 0,119 mit CV² = 80,6 gegen 2.400" in text and "kleinste Schnitt hat 2 Kanten von 17 (0,118)" in text and "q' = 0,70 ist CV² = 5,2e+07" in text
    assert "unmöglich" in ev.verdict_qp_curve({"rows": rows, "qp_opt": float("nan"), "cv2_opt": float("nan"), "cv2_plain": 1.0, "min_cut": -1, "m": 17}, S)[1]


def test_seltenheits_verdikt_zahlen():
    rows = [{"q": 0.1, "gain": 2.07, "n_plain": 1700.0, "n_opt": 823.0, "qp_opt": 0.176}, {"q": 0.001, "gain": 2210.0, "n_plain": 2.49e7, "n_opt": 11300.0, "qp_opt": 0.118}]
    kind, text = ev.verdict_rarity({"rows": rows, "min_cut": 2, "m": 17})
    assert kind == "info" and "q = 0,100 gewinnt die optimale Kippung nur das 2,07-Fache" in text and "q = 0,001 das 2.210-Fache" in text and "2,5e+07 Ziehungen" in text and "11.300" in text
    assert "zwischen 0,118 und 0,176" in text and "kleinster Schnitt 2 Kanten von 17" in text


def test_simulations_verdikt_zahlen():
    rows = [{"qp": 0.01, "zero": 0.65, "cover": 0.35, "med_rel": 1.0, "ess": 0.0, "ess_exact": 0.4, "k_med": float("nan"), "frac_k_bad": float("nan")},
            {"qp": 0.06, "zero": 0.0, "cover": 0.92, "med_rel": 0.23, "ess": 8.0, "ess_exact": 8.1, "k_med": 0.5, "frac_k_bad": 0.3},
            {"qp": 0.119, "zero": 0.0, "cover": 0.92, "med_rel": 0.2, "ess": 12.4, "ess_exact": 12.3, "k_med": 2.5, "frac_k_bad": 1.0},
            {"qp": 0.7, "zero": 0.0, "cover": 0.02, "med_rel": 1.0, "ess": 2.0, "ess_exact": 2.1, "k_med": 7.6, "frac_k_bad": 1.0}]
    kind, text = ev.verdict_sweep({"rows": rows, "qp_opt": 0.119, "P": 4e-4}, S)
    assert kind == "warning" and "65 % der Läufe ohne Ausfall, Überdeckung 0,35" in text and "q' = 0,119: mediane Abweichung 20 %" in text and "ESS 12,4 (Formel 12,3)" in text
    assert "q' = 0,70: Abweichung 100 %, Überdeckung 0,02" in text and "bei 2,5 und in 100 % der Läufe über 0,7" in text
    assert ev.verdict_sweep({"rows": [], "qp_opt": float("nan"), "P": 0.0}, S)[0] == "info"


def test_genauigkeits_verdikt_zahlen():
    rows = [{"relerr": 0.5, "n_plain": 9600.0, "n_cur": 322.0, "n_opt": 322.0, "ess_needed": 4.0}, {"relerr": 0.1, "n_plain": 239700.0, "n_cur": 8060.0, "n_opt": 8060.0, "ess_needed": 100.0}]
    kind, text = ev.verdict_precision({"rows": rows, "P": 4e-4, "qp_opt": 0.119}, S)
    assert kind == "info" and "10 % relativen Fehler braucht einfaches Ziehen 239.700" in text and "q' = 0,120" in text and "8.060" in text and "von etwa 1 / Fehler² = 100" in text
    assert "unmöglich" in ev.verdict_precision({"rows": rows, "P": 0.0, "qp_opt": float("nan")}, S)[1]


def test_stetiger_verdikt_zahlen():
    rows = [{"theta": 2.0, "cover": 0.92, "k_med": 0.50, "k_exact": 0.5}, {"theta": 3.0, "cover": 0.8, "k_med": 0.59, "k_exact": 0.667}]
    kind, text = ev.verdict_continuous({"t": 2.0, "P": 0.1353, "rows": rows, "theta_opt": 0.38, "cv2_opt": 2.5})
    assert kind == "info" and "P(X > 2,0) = 0,1353" in text and "θ = 0,38 (CV² = 2,5)" in text and "Intervall noch 0,92" in text and "im Median bei 0,50 (exakt 0,50)" in text and "bei θ = 3 ist die Überdeckung 0,80" in text


def test_formate():
    assert ev.de(1234.567, 1) == "1.234,6" and ev.de(-0.5, 2) == "−0,50" and ev.de_e(0.000042) == "4,2e-05" and ev.dn(float("nan")) == "nicht bestimmbar" and ev.dn(float("inf")) == "unendlich"
    assert ev.dz(2.4e8) == "2,4e+08" and ev.dz(239700) == "239.700" and ev.dz(29.74) == "29,7" and ev.dz(2.071) == "2,07" and ev.kanten(1) == "1 Kante" and ev.kanten(2) == "2 Kanten"
