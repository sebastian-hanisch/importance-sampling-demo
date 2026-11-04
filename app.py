"""Importance Sampling: seltene Ausfälle schätzen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 7 der Linie "Monte-Carlo-Schätzung und Varianzreduktion": Seltene Ereignisse werden aus einer Hilfsverteilung gezogen und mit dem Dichtequotienten gewichtet. Der Gewinn ist
exakt aus der Aufzählung berechenbar und wächst mit der Seltenheit; die optimale Kippung ist für kleine q etwa der kleinste Schnitt, ohne Kenntnis der Struktur muss man sie lernen; zu viel Kippung entartet. Die
ehrliche Diagnose ist die effektive Stichprobengröße (ESS); Pareto-k̂ ist bei den diskreten Gewichten dieser Demo nicht aussagekräftig und wirkt im stetigen Begleiter. Vehikel ist der Ausfall eines Gitternetzes.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import imp_algorithm as A
import imp_constants as C
import imp_evaluation as ev
from imp_evaluation import Settings, de, de_e, dn, dz
from imp_layout import dunkles_design, kopf_css
from imp_pdf_export import build_pdf
from imp_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from imp_scenario import named_netz, parse_edges
from imp_visualization import build_continuous, build_estimates, build_netz, build_qp_curve, build_rarity, build_sweep

st.set_page_config(page_title="Importance Sampling: seltene Ausfälle schätzen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Alle Kantenmengen aufzählen …")
def _table(netz_key):
    return ev.netz_table(named_netz(netz_key))


@st.cache_data(show_spinner="Schätzungen wiederholen …")
def _analysis(settings, table):
    return ev.analyse(settings, table=table)


@st.cache_data(show_spinner=False)
def _curve(settings, table):
    return ev.qp_curve(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner=False)
def _rarity(settings, table):
    return ev.rarity(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner="Kippungen messen …")
def _sweep(settings, table):
    return ev.sweep(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner="Stetigen Begleiter messen …")
def _continuous(settings):
    return ev.continuous(settings)


@st.cache_data(show_spinner=False)
def _custom(text, settings):
    netz = parse_edges(text)
    return netz, ev.custom_result(settings, netz)


def _netz_label(key):
    netz = named_netz(key)
    return f"{netz.name} ({netz.m} Kanten)"


def _q_label(q):
    return f"{q:g}".replace(".", ",")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in imp_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎯 Importance Sampling: seltene Ausfälle schätzen")
st.markdown(
    """
Fällt ein Netz nur selten aus, sieht einfaches Ziehen kaum einen Ausfall: bei 1000 Ziehungen und einer Ausfallwahrscheinlichkeit von 4·10⁻⁴ zeigen zwei von drei Läufen keinen einzigen und melden null.
**Importance Sampling** zieht stattdessen aus einer **Hilfsverteilung**, in der jede Kante öfter ausfällt (q' statt q), und **gewichtet** jede Ziehung mit dem Dichtequotienten. Das senkt die Varianz bei seltenen Ausfällen um Zehnerpotenzen (2,1-fach bei q = 0,1, 2210-fach bei q = 0,001), **wenn** die Hilfsverteilung passt.
Der Gewinn ist hier **exakt berechenbar**, er wächst mit der Seltenheit, die beste Kippung ist für kleine q etwa der kleinste Schnitt (ohne Kenntnis der Struktur muss man sie lernen), und **zu viel Kippung (hier ab q' ≈ 0,4) ist schlimmer als gar keine**.
Als Diagnose taugt die **effektive Stichprobengröße (ESS)**; **Pareto-k̂** warnt bei den diskreten Gewichten dieser Demo nicht, nur bei stetigen Gewichten (unten ein stetiger Begleiter mit unendlicher Varianz).

Weiter unten: **🔬 fünf Experimente**, **🧪 Spielwiese**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Importance Sampling", expanded=True):
    st.markdown(
        """
1. **Ereignis:** Jede Kante fällt unabhängig mit Wahrscheinlichkeit q aus; gesucht ist die Wahrscheinlichkeit P, dass das Netz nicht mehr zusammenhängt.
2. **Hilfsverteilung:** Man zieht jede Kante mit Ausfallwahrscheinlichkeit q' (zum Beispiel 12 % statt 1 %). Ausfälle treten dann oft auf.
3. **Gewichte:** Eine Ziehung mit f Ausfällen bekommt das Gewicht w(f) = (q/q')^f · ((1−q)/(1−q'))^(m−f). Der Schätzer ist das Mittel von 1[Ausfall] · w und bleibt erwartungstreu.
4. **Exakte Varianz:** Das zweite Moment ist Σ_f N_f [q^f (1−q)^(m−f)]² / [q'^f (1−q')^(m−f)] (N_f = Zahl der Ausfallmengen mit f Ausfällen). Daraus folgen CV² (relatives Varianzquadrat je Ziehung), die nötige Zahl von Ziehungen und die beste Kippung.
5. **Diagnose:** ESS = (Σw)² / Σw², nahe dem Optimum etwa N / (1 + CV²); der relative Fehler ist etwa 1/√ESS, solange die Gewichte nicht entarten. Pareto-k̂ (Rand der Gewichte) über 0,5 heißt unendliche Varianz, über 0,7 nicht verlässlich; bei den diskreten Gewichten dieser Demo (zwei Netze, eigene Implementierung) ist es nicht aussagekräftig.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz und Ereignis**")
    netz_key = st.selectbox("Netz", options("netz_select"), format_func=_netz_label, key="netz_select",
                            help="Gitter (Knoten in Zeilen und Spalten) oder die Hantel: zwei Dreiecke, durch eine einzige Brücke verbunden. Ereignis: das Netz hängt nicht mehr zusammen.")
    q = st.select_slider("Ausfallwahrscheinlichkeit q je Kante", options("q_select")[::-1], key="q_select", format_func=_q_label,
                         help="Je kleiner q, desto seltener der Netzausfall und desto größer der Gewinn von Importance Sampling.")
    st.markdown("**Hilfsverteilung**")
    qp = st.slider("Kippung q' (Ausfall je Kante in der Hilfsverteilung)", *bounds("qp_slider"), key="qp_slider", step=0.005, format="%.3f",
                   help="Mit q' = q ist es einfaches Ziehen. Gut ist bei kleinen q etwa kleinster Schnitt durch Kantenzahl; zu große Werte entarten.")
    n = st.select_slider("Ziehungen N", options("n_select"), key="n_select", help="Ziehungen je Schätzung.")
    st.markdown("**Stetiger Begleiter (Experiment 5)**")
    t = st.slider("Schwelle t", *bounds("t_slider"), key="t_slider", step=0.5, help="Gesucht ist P(X > t) für X mit Exponentialverteilung der Rate 1.")
    theta = st.slider("Rate θ der Hilfsverteilung", *bounds("theta_slider"), key="theta_slider", step=0.05, help="Hilfsverteilung Exp(θ); ab θ = 2 ist die Varianz unendlich.")
    seed = st.number_input("Zufallsstrom", *bounds("seed_input"), key="seed_input", step=1, help="Welcher feste Strom von Zufallszahlen benutzt wird (SplitMix64).")
    st.button("🎲 Neuen Strom wählen", width="stretch", on_click=randomize_seed)

sync_query_params({"netz_select": netz_key, "q_select": float(q), "qp_slider": float(qp), "n_select": int(n), "t_slider": float(t), "theta_slider": float(theta), "seed_input": int(seed)})

settings = Settings(str(netz_key), float(q), round(float(qp), 3), int(n), float(t), float(theta), int(seed))
table = _table(settings.netz)
a = _analysis(settings, table)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if a.P > 0:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ausfallwahrscheinlichkeit P", de_e(a.P), help=f"Exakte Wahrscheinlichkeit, dass das {a.netz.name} bei q = {_q_label(settings.q)} je Kante nicht mehr zusammenhängt (Aufzählung aller Kantenmengen).")
    m2.metric("Gewinn gegen einfaches Ziehen", dz(a.gain), help="CV² einfach durch CV² bei der gewählten Kippung (exakt): so viel weniger Ziehungen braucht dasselbe Ergebnis.")
    m3.metric("Ziehungen für 10 % Fehler", dz(a.n_is), delta=f"einfach {dz(a.n_plain)}", delta_color="off", help="100 mal CV² (relativer Fehler 10 %), exakt.")
    m4.metric("ESS (gemessen)", de(a.ess_meas, 1), delta=f"Formel {de(a.ess_exact, 1)}", delta_color="off", help="Effektive Stichprobengröße der Gewichte, Median über die Wiederholungen; die Formel N / (1 + CV²) trifft sie nahe dem Optimum, bei zu starker Kippung liegt die Messung weit darüber.")
    st.caption(f"Optimale Kippung q' = {de(a.qp_opt, 3)} (kleinster Schnitt {ev.kanten(a.min_cut)} von {a.netz.m}: {de(a.min_cut / a.netz.m, 3)}). Gemessen mit N = {de(settings.n, 0)}: mediane Abweichung {de(a.med_rel * 100, 0)} %, "
               f"{de(a.zero * 100, 0)} % der Läufe ohne einen Ausfall, Überdeckung des 95-%-Intervalls {de(a.cover, 2)}, Pareto-k̂ im Median {dn(a.k_med, 1)}.")
    if any(x > 0 for x in a.est):
        st.plotly_chart(build_estimates(a, a.P), width="stretch", key="main_chart")
        st.caption("Kasten: Verteilung der Schätzungen über die Wiederholungen (logarithmische Achse; Läufe ohne Ausfall erscheinen nicht, ihr Anteil steht oben). Gestrichelt: die Wahrheit.")
    else:
        st.info("Alle Wiederholungen haben keinen einzigen Ausfall gesehen: jede Schätzung ist null.")
else:
    st.info("Bei diesen Einstellungen fällt das Netz nie aus.")

if st.button("📄 PDF der Schätzung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="importance-sampling.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie schätzt man seltene Ereignisse, und wann tragen die Gewichte nicht mehr?")
kind, text = ev.verdict_main(a, settings)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Welche Kippung?")
st.caption("Das **exakte** relative Varianzquadrat CV² je Ziehung gegen die Kippung q' (aus der Aufzählung aller Kantenmengen, kein Zufall), mit dem Optimum. Je kleiner CV², desto weniger Ziehungen für dieselbe Genauigkeit.")
ex1 = _curve(settings, table)
if ex1["qp_opt"] == ex1["qp_opt"]:
    st.plotly_chart(build_qp_curve(ex1, settings.qp), width="stretch", key="curve_chart")
kind, text = ev.verdict_qp_curve(ex1, settings)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 2: Wie viel gewinnt man, je seltener das Ereignis ist?")
st.caption("Für jedes q (Ausfall je Kante) die optimale Kippung und die Zahl der Ziehungen für 10 % relativen Fehler gegen einfaches Ziehen (exakt, kein Zufall).")
ex2 = _rarity(settings, table)
if all(r["gain"] == r["gain"] for r in ex2["rows"]):
    st.plotly_chart(build_rarity(ex2), width="stretch", key="rarity_chart")
    st.dataframe(
        {"q": [_q_label(r["q"]) for r in ex2["rows"]], "P": [de_e(r["P"]) for r in ex2["rows"]], "CV² einfach": [dz(r["cv2_plain"]) for r in ex2["rows"]], "optimales q'": [de(r["qp_opt"], 3) for r in ex2["rows"]],
         "CV² optimal": [dz(r["cv2_opt"]) for r in ex2["rows"]], "Gewinn": [dz(r["gain"]) for r in ex2["rows"]]},
        hide_index=True, width="stretch", key="rarity_table",
    )
    kind, text = ev.verdict_rarity(ex2)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 3: Simulation gegen die Kippung")
st.caption(f"Bei q' = q (einfach), dem halben, dem optimalen und dem doppelten q' und großen Kippungen: mediane Abweichung, Anteil der Läufe ohne Ausfall, Überdeckung des 95-%-Intervalls, ESS und Pareto-k̂; je {C.SWEEP_REPS} Wiederholungen mit N = Regler. Kostet einige Sekunden.")
if st.button("▶ Messung starten", key="run_sweep"):
    st.session_state["sweep_done"] = True
if st.session_state.get("sweep_done"):
    ex3 = _sweep(settings, table)
    if ex3["rows"]:
        st.plotly_chart(build_sweep(ex3), width="stretch", key="sweep_chart")
        st.dataframe(
            {"q'": [de(r["qp"], 3) for r in ex3["rows"]], "CV² (exakt)": [dz(r["cv2"]) for r in ex3["rows"]], "mediane Abweichung": [f"{de(r['med_rel'] * 100, 0)} %" for r in ex3["rows"]],
             "ohne Ausfall": [f"{de(r['zero'] * 100, 0)} %" for r in ex3["rows"]], "Überdeckung": [de(r["cover"], 2) for r in ex3["rows"]], "ESS gemessen": [de(r["ess"], 1) for r in ex3["rows"]],
             "ESS nach Formel": [de(r["ess_exact"], 1) for r in ex3["rows"]], "Pareto-k̂ (Median)": [dn(r["k_med"], 1) for r in ex3["rows"]]},
            hide_index=True, width="stretch", key="sweep_table",
        )
    kind, text = ev.verdict_sweep(ex3, settings)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 4: Wie viele Ziehungen für welche Genauigkeit?")
st.caption("Exakt (kein Zufall): Ziehungen für 50, 20, 10 und 5 % relativen Fehler bei einfachem Ziehen, der gewählten und der optimalen Kippung (CV² durch Fehler²). Die Wirksamkeit eines Verfahrens ist die ESS, die dafür nötig ist: 1 / Fehler² Gewichte, die wirklich zählen.")
ex4 = ev.precision(settings, named_netz(settings.netz), table)
if ex4["P"] > 0:
    st.dataframe(
        {"relativer Fehler": [f"{de(r['relerr'] * 100, 0)} %" for r in ex4["rows"]], "einfach": [dz(r["n_plain"]) for r in ex4["rows"]], "gewählte Kippung": [dz(r["n_cur"]) for r in ex4["rows"]],
         "optimale Kippung": [dz(r["n_opt"]) for r in ex4["rows"]], "nötige ESS": [dz(r["ess_needed"]) for r in ex4["rows"]]},
        hide_index=True, width="stretch", key="precision_table",
    )
kind, text = ev.verdict_precision(ex4, settings)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 5: Stetiger Begleiter mit unendlicher Varianz")
st.caption(f"P(X > t) für X mit Exponentialverteilung (Rate 1), Hilfsverteilung Exp(θ). Zweites Moment e^(−(2−θ)t) / (θ (2−θ)) für θ < 2, **unendlich** für θ ≥ 2; Pareto-k̂ des Gewichtsrandes ist exakt (θ−1)/θ. N = {C.CONT_N}, je {C.CONT_REPS} Wiederholungen. Schwelle t aus der Seitenleiste.")
if st.button("▶ Messung starten", key="run_cont"):
    st.session_state["cont_done"] = True
if st.session_state.get("cont_done"):
    ex5 = _continuous(settings)
    st.plotly_chart(build_continuous(ex5), width="stretch", key="cont_chart")
    st.dataframe(
        {"θ": [de(r["theta"], 2) for r in ex5["rows"]], "CV² (exakt)": [dz(r["cv2"]) for r in ex5["rows"]], "k exakt": [dn(r["k_exact"], 3) for r in ex5["rows"]], "k̂ (Median)": [dn(r["k_med"], 2) for r in ex5["rows"]],
         "mediane Abweichung": [f"{de(r['med_rel'] * 100, 1)} %" for r in ex5["rows"]], "Überdeckung": [de(r["cover"], 2) for r in ex5["rows"]]},
        hide_index=True, width="stretch", key="cont_table",
    )
    kind, text = ev.verdict_continuous(ex5)
    getattr(st, kind)(text)
st.caption(f"Reglerwert θ = {de(settings.theta, 2)} bei t = {de(settings.t, 1)}: CV² = {dz(A.cont_cv2(settings.theta, settings.t))}"
           + (" (unendliche Varianz)" if settings.theta >= 2 else "") + ".")

st.markdown("---")

# --- Spielwiese -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🧪 Spielwiese: ein eigenes Netz"):
    st.caption(
        f"Eine Kante je Zeile, zwei ganze Zahlen (»0-1«). Höchstens {C.CUSTOM_EDGE_LIMIT} Kanten und {C.CUSTOM_NODE_LIMIT} Knoten; q, q', N und Strom stammen aus der Seitenleiste. "
        "Es zeigt die exakten Kennzahlen und die Schätzung; Ereignis ist wieder: das Netz hängt nicht zusammen."
    )
    text_in = st.text_area("Kanten", "0-1\n1-2\n2-0\n3-4\n4-5\n5-3\n2-3", height=140, key="eigene_kanten")
    try:
        cnetz, cres = _custom(text_in, settings)
    except ValueError as err:
        st.error(str(err))
    else:
        ca = cres["analysis"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Ausfallwahrscheinlichkeit P", de_e(ca.P) if ca.P > 0 else "0")
        s2.metric("optimale Kippung q'", dn(ca.qp_opt, 3))
        s3.metric("Gewinn bei dem Regler q'", dz(ca.gain))
        st.plotly_chart(build_netz(cnetz), width="stretch", key="custom_net")
        kind, text = ev.verdict_main(ca, settings)
        getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Man kennt eine gute Hilfsverteilung** | Die beste Kippung ist für kleine q etwa kleinster Schnitt durch Kantenzahl; ohne Kenntnis der Struktur muss man sie lernen. Zu schwach gekippt bringt nichts, zu stark (hier ab q' ≈ 0,4) ist schlimmer als gar keine (Experimente 1 und 3). | Stück 8 (adaptives Importance Sampling lernt sie) |
| **Die Gewichte tragen** | Wenige Ziehungen bestimmen das Ergebnis; ESS sagt, wie viele effektiv zählen. | – |
| **Pareto-k̂ ist aussagekräftig** | Bei den diskreten Gewichten dieser Demo (Netz) nicht; bei stetigen schon (Experiment 5). | – |
| **Das Wald-Intervall gilt** | Bei kleiner ESS und schwerem Rand ist es zu schmal; bei unendlicher Varianz bemerkt es das mit N = 10 000 nicht. | – |
| **Eine globale Kippung genügt** | Sie ist die einparametrige Familie; je Kante verschieden gekippt wäre besser. | Stück 8 (Kreuzentropie) |
| **Importance Sampling statt Splitting** | Beide schätzen seltene Ereignisse, Splitting über eine Folge von Zwischenstufen. | `splitting-demo` (Warteschlangen 9) |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Alle Netze und Ausfälle sind **erfunden**; die Wahrheit ist die exakte Aufzählung. Die Kanten fallen unabhängig und mit gleicher Wahrscheinlichkeit aus. Die Zufallszahlen kommen aus festen Strömen (SplitMix64), nie aus einem einzelnen Seed.

**Grenzen.** Aufgezählt wird live bis 17 Kanten, Ereignis ist nur der Netzausfall. Es gibt **eine globale Kippung**. Pareto-k̂ ist eine eigene Implementierung nach Zhang und Stephens (2009) in der Form von Vehtari et al.; sie ist am scipy-Orakel mit bekannten Formparametern geprüft, nicht am Netz. Selbstnormiertes Importance Sampling und Mischungen von Hilfsverteilungen fehlen.

**Bekannte Probleme und Unsicherheiten.** Messungen mit wenigen Ausfällen (kleines N, kleines q) rauschen stark; ein Teil der Läufe sieht gar keinen Ausfall. Der Anteil der Läufe mit k̂ über 0,7 beim optimalen q' hängt vom Netz ab. Die Literaturangaben im Mathematik-Expander stammen aus dem Gedächtnis und sind nicht eingesehen. Die Oberfläche ist mit AppTest geprüft und im Browser angesehen, nicht mit echtem Mausklick auf jedem Gerät.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Gesucht ist $P = \mathbb P(A)$ für das seltene Ereignis $A$ = „das Netz hängt nicht zusammen“; jede der $m$ Kanten fällt unabhängig mit Wahrscheinlichkeit $q$ aus.

**Importance Sampling.** Hilfsverteilung: jede Kante fällt mit $q'$ aus. Eine Ziehung mit $f$ Ausfällen hat das Gewicht $w(f) = (q/q')^f\,((1-q)/(1-q'))^{m-f}$, und $\hat P = \frac1N\sum_i \mathbf 1[A_i]\,w(f_i)$ ist erwartungstreu.

**Exakte Varianz.** Mit $N_f$ = Zahl der Ausfallmengen mit $f$ ausgefallenen Kanten:
$$M_2 = \sum_f N_f\,\frac{\bigl[q^f(1-q)^{m-f}\bigr]^2}{q'^f(1-q')^{m-f}},\qquad \mathrm{CV}^2 = \frac{M_2 - P^2}{P^2}.$$
Für 10 % relativen Fehler braucht man $N \approx 100\,\mathrm{CV}^2$ Ziehungen. Einfaches Ziehen ist $q' = q$: $\mathrm{CV}^2 = (1-P)/P$.

**Effektive Stichprobengröße.** $\mathrm{ESS} = (\sum w_i)^2 / \sum w_i^2$; nahe dem Optimum etwa $N/(1+\mathrm{CV}^2)$ (ein Verhältnis von Erwartungswerten, bei schweren Rändern der Gewichte weicht die gemessene ESS stark ab). Der relative Fehler ist etwa $1/\sqrt{\mathrm{ESS}}$, solange die Gewichte nicht entarten.

**Pareto-k̂.** An den größten Gewichten wird eine verallgemeinerte Pareto-Verteilung mit Formparameter $k$ angepasst (Zhang und Stephens 2009). $k \ge 1/2$ bedeutet unendliche Varianz, $k > 0{,}7$ gilt nach Vehtari et al. als nicht verlässlich.

**Stetiger Begleiter.** $X \sim \mathrm{Exp}(1)$, gesucht $P(X > t) = e^{-t}$, Hilfsverteilung $\mathrm{Exp}(\theta)$, Gewicht $w(x) = e^{-(1-\theta)x}/\theta$ auf $x > t$. Zweites Moment $M_2 = e^{-(2-\theta)t}/(\theta(2-\theta))$ für $\theta < 2$, sonst unendlich; für $\theta > 1$ ist der Randindex der Gewichte $\theta/(\theta-1)$, also $k = (\theta-1)/\theta$, und die Varianz ist genau für $k < 1/2$ endlich.

**Wahrheit.** Die Aufzählung aller $2^m$ Kantenmengen liefert $P$ und alle $N_f$ exakt.

**Literatur (aus dem Gedächtnis, nicht eingesehen).** Glasserman, *Monte Carlo Methods in Financial Engineering* 2003 (Kapitel zu Importance Sampling); Owen, *Monte Carlo theory, methods and examples* (Kapitel zu Importance Sampling); Zhang und Stephens 2009, „A new and efficient estimation method for the generalized Pareto distribution“ (Technometrics); Vehtari, Simpson, Gelman, Yao und Gabry, „Pareto smoothed importance sampling“ (JMLR 2024).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
