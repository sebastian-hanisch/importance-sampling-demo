"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset rechnet und hat das erwartete Verdikt."""

import pytest

import imp_algorithm as A
import imp_constants as C
import imp_evaluation as ev
import imp_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_options():
    for name, preset in C.PRESETS.items():
        assert C.QP_MIN <= preset["qp"] <= C.QP_MAX and C.T_MIN <= preset["t"] <= C.T_MAX and C.THETA_MIN <= preset["theta"] <= C.THETA_MAX and 0 <= preset["seed"] <= C.SEED_MAX, name
        assert preset["q"] in C.Q_CHOICES and preset["n"] in C.N_CHOICES and preset["netz"] in C.NETZE, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_jedes_preset_unterscheidet_sich_vom_standard():
    standard = C.PRESETS["Standardfall (Voreinstellung)"]
    for name, preset in C.PRESETS.items():
        if name != "Standardfall (Voreinstellung)":
            assert preset != standard, name


def test_standard_preset_traegt():
    s = ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"])
    a = ev.analyse(s)
    assert a.gain > 20 and a.zero == 0.0 and a.cover > 0.85 and ev.verdict_main(a, s)[0] == "success"


def test_preset_einfaches_ziehen_sieht_nichts():
    s = ev.Settings(**C.PRESETS["Einfaches Ziehen sieht nichts"])
    a = ev.analyse(s)
    assert s.qp == s.q and a.zero > 0.5 and a.cover < 0.5 and ev.verdict_main(a, s)[0] == "warning"


def test_preset_zu_stark_gekippt_ist_schlechter_als_einfach():
    s = ev.Settings(**C.PRESETS["Zu stark gekippt"])
    a = ev.analyse(s)
    assert a.gain < 1.0 and a.cover < 0.3 and ev.verdict_main(a, s)[0] == "warning"


def test_preset_kritische_bruecke_hat_optimum_bei_einem_siebtel():
    s = ev.Settings(**C.PRESETS["Eine kritische Brücke"])
    a = ev.analyse(s)
    assert a.min_cut == 1 and abs(a.qp_opt - 1 / 7) < 0.02 and a.gain > 5 and ev.verdict_main(a, s)[0] == "success"


def test_preset_unendliche_varianz_setzt_theta_ueber_zwei():
    s = ev.Settings(**C.PRESETS["Unendliche Varianz (stetiger Begleiter)"])
    assert s.theta == 2.5 and A.cont_cv2(s.theta, s.t) == float("inf")
    assert A.cont_cv2(ev.Settings().theta, ev.Settings().t) < float("inf")


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_rechnet(name):
    a = ev.analyse(ev.Settings(**C.PRESETS[name]))
    assert 0.0 < a.P < 1.0
