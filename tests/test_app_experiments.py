"""AppTest der Experimente, der Spielwiese, des PDF-Exports und des Permalinks (Einzelheiten, die der allgemeine Rauchtest nicht trifft)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import imp_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(e.value for kind in (at.success, at.info, at.warning, at.error) for e in kind)


@pytest.mark.parametrize("key,fragment", [
    ("run_sweep", "Einfaches Ziehen (q' = q):"),
    ("run_cont", "Bei θ = 2 ist die Varianz"),
])
def test_experiment_auf_klick_zeigt_sein_verdikt(key, fragment):
    at = _app()
    assert fragment not in _texte(at)                       # vor dem Klick steht noch nichts da
    at.button(key=key).click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert fragment in _texte(at)


def test_exakte_experimente_stehen_ohne_klick_da():
    t = _texte(_app())
    assert "Zu schwach gekippt (q' kleiner als q) ist schlechter als einfaches Ziehen" in t
    assert "gewinnt die optimale Kippung nur das" in t
    assert "Für 10 % relativen Fehler braucht einfaches Ziehen" in t


def test_experiment_bleibt_nach_einer_aenderung_der_einstellungen_sichtbar():
    at = _app()
    at.button(key="run_sweep").click().run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="n_select").set_value(300).run(timeout=TIMEOUT)
    assert not at.exception
    assert "Einfaches Ziehen (q' = q):" in _texte(at)


def test_kernfrage_verdikt_wechselt_mit_dem_preset():
    at = _app()
    assert "Die Kippung ist" not in _texte(at) and "Pareto-k̂ liegt im Median" in _texte(at)
    at.sidebar.button(key="preset_Einfaches Ziehen sieht nichts").click().run(timeout=TIMEOUT)
    assert "Ohne Kippung sehen die meisten Läufe keinen" in _texte(at)
    at.sidebar.button(key="preset_Zu stark gekippt").click().run(timeout=TIMEOUT)
    assert "schlechter als einfaches Ziehen: die Gewichte sind so ungleich" in _texte(at)
    at.sidebar.button(key="preset_Eine kritische Brücke").click().run(timeout=TIMEOUT)
    assert "kleinster Schnitt 1 Kante von 7" in _texte(at)


def test_jedes_netz_und_jedes_q_laeuft_durch():
    at = _app()
    for netz in C.NETZE:
        at.sidebar.selectbox(key="netz_select").set_value(netz).run(timeout=TIMEOUT)
        assert not at.exception, (netz, [str(e) for e in at.exception])
    for q in (C.Q_MIN, C.Q_MAX):
        at.sidebar.select_slider(key="q_select").set_value(q).run(timeout=TIMEOUT)
        assert not at.exception, (q, [str(e) for e in at.exception])


def test_spielwiese_gueltiges_netz_und_fehlermeldungen():
    at = _app()
    assert not at.error
    at.text_area(key="eigene_kanten").set_value("0-0").run(timeout=TIMEOUT)
    assert not at.exception and "Schleife" in " ".join(e.value for e in at.error)
    at.text_area(key="eigene_kanten").set_value("").run(timeout=TIMEOUT)
    assert not at.exception and "Keine Kanten" in " ".join(e.value for e in at.error)


def test_spielwiese_pfad_laeuft_durch():
    at = _app()
    at.text_area(key="eigene_kanten").set_value("0-1\n1-2").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error


def test_pdf_wird_erst_nach_klick_erzeugt():
    at = _app()
    assert not at.get("download_button")
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("download_button")) == 1
    at.sidebar.select_slider(key="n_select").set_value(300).run(timeout=TIMEOUT)         # andere Einstellung: altes PDF nicht mehr anbieten
    assert not at.get("download_button")


def test_permalink_wird_begrenzt_und_eingerastet():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["qp"] = "5"
    at.query_params["q"] = "0.016"
    at.query_params["n"] = "400"
    at.query_params["g"] = "9x9"            # unbekanntes Netz: verworfen
    at.query_params["t"] = "9"
    at.query_params["th"] = "-1"
    at.query_params["seed"] = "-4"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.slider(key="qp_slider").value == C.QP_MAX
    assert at.sidebar.select_slider(key="q_select").value == 0.02                           # 0,016 liegt näher an 0,02 als an 0,01
    assert at.sidebar.select_slider(key="n_select").value == 300                            # 400 liegt näher an 300 als an 1000
    assert at.sidebar.selectbox(key="netz_select").value == C.DEFAULT_NETZ
    assert at.sidebar.slider(key="t_slider").value == C.T_MAX and at.sidebar.slider(key="theta_slider").value == C.THETA_MIN
    assert at.sidebar.number_input(key="seed_input").value == 0
