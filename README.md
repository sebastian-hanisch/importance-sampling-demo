# Importance Sampling: seltene Ausfälle schätzen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-importance-sampling-demo.streamlit.app/)**

Siebtes Stück der Linie **„Monte-Carlo-Schätzung und Varianzreduktion“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Linie wird erst vollständig veröffentlicht). Nachfolger von [bedingtes-monte-carlo-demo](https://github.com/sebastian-hanisch/bedingtes-monte-carlo-demo) (Stück 6, das bei seltenen Ausfällen das naive Intervall repariert) und [monte-carlo-schaetzer-demo](https://github.com/sebastian-hanisch/monte-carlo-schaetzer-demo) (Stück 1, das zeigt, dass einfaches Ziehen seltene Ereignisse nicht sieht). Hier hebt das Stück die Annahme „gezogen wird aus der Verteilung des Problems“ auf: Gezogen wird aus einer **Hilfsverteilung**, die den Ausfall häufig macht, und **gewichtet**. Vehikel ist wie in Stück 1 bis 6 die Zuverlässigkeit eines Gitternetzes (Ereignis: das Netz hängt **nicht** zusammen), dazu die Hantel (zwei Dreiecke, eine Brücke) und ein **stetiger Begleiter** mit geschlossener Formel. Die Wahrheit ist exakt aufgezählt.

## Warum dieses Problem

Fällt ein Netz nur selten aus, sieht einfaches Ziehen kaum einen Ausfall. Importance Sampling kippt die Ausfallwahrscheinlichkeit jeder Kante von q auf q'. Die Fragen: **Wie viel gewinnt man, welche Kippung ist gut, und woran merkt man, dass die Gewichte nicht mehr tragen?** Die Antworten aus der exakten Rechnung sind scharf, und eine davon widerspricht der Gliederung: Pareto-k̂ ist bei den diskreten Gewichten dieser Demo keine brauchbare Warnung.

## Befunde und Korrekturen gegenüber dem Plan

Vor dem Bau wurden die Hypothesen der Gliederung an Messreihen geprüft (Vormessung im Website-Repo, `docs/planung/monte-carlo-planung/vormessung/`). Was sich geändert hat:

- **„Schlechte Wahl liefert unendliche Varianz, erkennbar an Pareto-k̂ und ESS“ trifft im Netz nicht zu.** Der Zustandsraum ist endlich, die Varianz ist für 0 < q' < 1 endlich (nur riesig). **Pareto-k̂ ist bei den diskreten Gewichten dieser Demo (zwei Netze, eigene Implementierung) nicht aussagekräftig**: Es liegt auch beim optimalen q' bei 2,6 (über der Marke 0,7), trotz endlicher Varianz und guter ESS. Die ESS ist nahe dem Optimum etwa N/(1 + CV²) (12,4 gegen 12,3) und die ehrlichere Größe; bei zu starker Kippung liegt die gemessene Median-ESS (2,9) weit über diesem Wert (0,04). Unendliche Varianz und k̂ zeigt der **stetige Begleiter**.
- **Das optimale q' ist für kleine q (q ≤ 0,02) etwa kleinster Schnitt durch Kantenzahl**; ohne Kenntnis der Struktur muss man q' lernen (Stück 8).
- **Zu viel Kippung (hier ab q' ≈ 0,4) ist schlimmer als gar keine** (q' = 0,5: CV² 22 000 gegen 2400).
- **Der Gewinn wächst mit der Seltenheit** (2-fach bei q = 0,1, 2200-fach bei q = 0,001).

## Modell

- **Ereignis:** A = das Netz hängt nicht zusammen. Jede Kante fällt unabhängig mit q aus; P = Wahrscheinlichkeit von A. Netze: Gitter 3×3, 3×4 und die Hantel (bis 17 Kanten, alle Kantenmengen live aufgezählt).
- **Importance Sampling:** Jede Kante fällt mit q' aus. Gewicht einer Ziehung mit f Ausfällen: w(f) = (q/q')^f ((1−q)/(1−q'))^(m−f); Schätzer: Mittel von 1[A]·w, erwartungstreu.
- **Exakt:** zweites Moment M₂ = Σ_f N_f [q^f(1−q)^(m−f)]² / [q'^f(1−q')^(m−f)] mit N_f = Zahl der Ausfallmengen mit f ausgefallenen Kanten; CV² = (M₂ − P²)/P²; Ziehungen für relativen Fehler ε: CV²/ε². Einfaches Ziehen ist q' = q.
- **Diagnose:** ESS = (Σw)²/Σw² (nahe dem Optimum etwa N/(1 + CV²)); Pareto-k̂ nach Zhang und Stephens (2009) in der Form von Vehtari et al.: eine verallgemeinerte Pareto-Verteilung an den größten Gewichten; k ≥ ½ heißt unendliche Varianz, k̂ über 0,7 gilt als nicht verlässlich.
- **Stetiger Begleiter:** P(X > t) für X ~ Exp(1), Hilfsverteilung Exp(θ), Gewicht e^{−(1−θ)x}/θ; M₂ = e^{−(2−θ)t}/(θ(2−θ)) für θ < 2, sonst unendlich; für θ > 1 ist der Randindex θ/(θ−1), also k = (θ−1)/θ. Mathematik im 📐-Expander der App.

## Methodik

Zufallszahlen aus festen SplitMix64-Strömen (nie ein einzelner Seed). Wahrheit, P und alle N_f aus der vollständigen Aufzählung. Je Messung 300 bis 400 Wiederholungen mit festen Strömen; der stetige Begleiter mit N = 10 000.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Standardfall: Gitter 3×4, q = 0,01 (P = 4,2·10⁻⁴), q' = 0,12, N = 1000.

| Frage | Befund |
|---|---|
| Wie selten ist der Ausfall, wie viel Ziehungen braucht einfaches Ziehen? (exakt) | P = **4,2·10⁻⁴**, CV² = **2397**, für 10 % relativen Fehler **239 737 Ziehungen** (bei q = 0,001: 24,9 Millionen) |
| Optimale Kippung und Gewinn, 3×4 (exakt), q = 0,1 / 0,05 / 0,02 / 0,01 / 0,005 / 0,002 / 0,001 | Gewinn **2,1 / 3,4 / 10,1 / 29,8 / 101 / 572 / 2210**; q' = 0,176 / 0,137 / 0,121 / **0,119** / 0,118 / 0,118 / 0,118; Ziehungen für 10 % bei q = 0,01: **8057** statt 239 737; bei q = 0,001: **11 263** statt 24,9 Millionen |
| Woher kommt das optimale q'? (kleine q) | kleinster Schnitt durch Kantenzahl: 3×4 **2/17 = 0,118**, 3×3 (q = 0,001) 2/12 = 0,167 (gefunden 0,1667), Hantel 1/7 = 0,143 (gefunden 0,1428); Gewinn bei q = 0,001: 4681 (3×3) und 61 (Hantel) |
| Falsche Kippung, 3×4, q = 0,01, CV² | q' = 0,003: **37 104**; q' = 0,01 (einfach): 2397; q' = 0,12: **80,6**; q' = 0,3: 399; q' = 0,5: **22 351**; q' = 0,7: **2,4·10⁷** |
| Einfaches Ziehen bei N = 1000 (400 Wdh.) | **65 %** der Läufe sehen keinen Ausfall (exakt (1−P)^N = 0,66), Überdeckung des Wald-Intervalls **0,35**, mediane Abweichung 100 % |
| Standardfall bei N = 1000 | mediane Abweichung **19 %**, keine Nullschätzung, Überdeckung **0,92**, ESS **12,4** (Formel 12,3), mehr als 50 % Fehler in 6 % der Läufe |
| Zu stark gekippt (q' = 0,5, N = 1000) | mediane Abweichung **98 %**, Überdeckung **0,14**, ESS 2,9; bei q' = 0,7 Überdeckung 0,01 bis 0,02 |
| Hantel, q = 0,01, q' = 0,14, N = 1000 | CV² 93,4 → **13,1** (Gewinn 7,2), mediane Abweichung **8 %**, Überdeckung 0,94, ESS **71,2** (Formel 71,1) |
| Ziehungen für 50 / 20 / 10 / 5 % Fehler, 3×4, q = 0,01, optimale Kippung (exakt) | **322 / 2014 / 8057 / 32 227**; einfach 9589 / 59 934 / 239 737 / 958 946 |
| **Pareto-k̂ im Netz (diskrete Gewichte, zwei Netze, eigene Implementierung)** | Median **2,6** beim optimalen q' (3×4) und in **100 %** der Läufe über 0,7; Hantel **15,9** (96 %); q' = 0,5: 7,3. Das Verfahren arbeitet dort am besten: k̂ warnt falsch (die Werte sind Eigenschaften dieser Implementierung auf Gewichten mit vielen Gleichständen, nicht gegen eine Referenzimplementierung geprüft). Beim einfachen Ziehen sind alle Gewichte gleich: k̂ nicht bestimmbar |
| **Stetiger Begleiter**, t = 2 (P = 0,135), N = 10 000, θ = 1,2 / 1,5 / 1,8 / 2,0 / 2,5 / 3,0 | exaktes k = (θ−1)/θ = 0,167 / 0,333 / 0,444 / **0,500** / 0,600 / 0,667; **k̂ (Median) 0,18 / 0,34 / 0,44 / 0,50 / 0,58 / 0,60**; CV² 10,5 / 25,8 / 100,7 / **unendlich** / unendlich / unendlich |
| dasselbe: Überdeckung des Wald-Intervalls | **0,94 / 0,94 / 0,93 / 0,92 / 0,85 / 0,80**: Bei θ = 2 ist die Varianz unendlich und das Intervall merkt es mit N = 10 000 nicht; k̂ über 0,5 warnt hier, knapp darunter (θ = 1,8) bleibt die Varianz endlich, aber sehr groß. Optimales θ = **0,382** mit CV² = 2,47 |

## Ehrliche Grenzen

- **Nur eine globale Kippung q'** für alle Kanten; eine Kippung je Kante lernt Stück 8.
- **Pareto-k̂ ist bei den diskreten Gewichten dieser Demo (zwei Netze) nicht aussagekräftig**, im stetigen Begleiter schon; eine eigene Implementierung nach Zhang und Stephens, am scipy-Orakel mit bekannten Formparametern geprüft (ξ = 0,2 / 0,5 / 1,0), nicht am Netz.
- **Nur das Ereignis „Netzausfall“** und Netze bis 17 Kanten; unabhängige, gleiche Ausfälle. Kein selbstnormiertes Importance Sampling, keine Mischungen von Hilfsverteilungen, kein Vergleich mit Splitting.
- **Die Überdeckung ist die des Wald-Intervalls** aus dem Standardfehler der gewichteten Werte; bei wenigen effektiven Ausfällen ist es zu schmal.
- **Messungen mit wenigen Ausfällen** (kleines N, kleines q) rauschen stark; die exakten Zahlen sind belastbar.
- **Literaturangaben im Mathematik-Expander** stammen aus dem Gedächtnis und sind nicht eingesehen.

## Verifikation

- **Kern** (`tests/test_algorithm.py`): Zufallsstrom gegen Ganzzahl-Referenz, Ausfallzählung und zweites Moment am Pfad aus zwei Kanten mit Brüchen (N₁ = 2, N₂ = 1; M₂ = 7/9 bei q = ½, q' = ¾), Gewichte, ESS, k̂-Randfälle, stetige Formeln.
- **Orakel** (`tests/test_oracle_imp.py`): Ausfallzählung gegen `networkx` auf allen Kantenmengen von Hantel und 3×3; **zweites Moment durch direkte Summation über alle 2^m Kantenmengen** (Brüche); **k̂ gegen `scipy.stats.genpareto`** bei bekanntem Formparameter und gegen die Maximum-Likelihood-Anpassung; **stetiges zweites Moment gegen numerische Integration** (`scipy.integrate.quad`) und das Divergieren ab θ = 2; Simulation erwartungstreu, Varianz und ESS gegen die exakten Werte, Anteil der Nullschätzungen gegen (1−P)^N.
- **Eigenschaften** (`tests/test_properties.py`): Importance-Identität (Gewichte mal gekippte Wahrscheinlichkeit summieren sich zu 1), CV² = (1−P)/P beim einfachen Ziehen, optimales q' zwischen q und ½ und nahe Schnitt durch Kanten, Gewinn wächst mit der Seltenheit, zu schwache und zu starke Kippung schlechter als einfach, ESS ≤ N, k̂ im Netz am Optimum groß, im stetigen Begleiter dem exakten Wert folgend.
- **Randfälle** (`tests/test_edge_cases.py`): Eingabefehler der Spielwiese, Pfad, q' an den Reglergrenzen, jedes Netz mit jedem q und N, unmögliches Ereignis, kein Ausfall in der Stichprobe, q' = 0 und 1.
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede genannte Zahl an von Hand gesetzten Werten, deutsches Zahlenformat.
- **Presets, Layout, Oberfläche:** Presets vollständig und jedes mit dem erwarteten Verdikt; Schnellstart oben, fester Kopf, aktives Preset hervorgehoben; AppTest für jeden Button, jeden Regler an Minimum und Maximum, die Experimente, die Spielwiese, den PDF-Export und den Permalink.
- **Laufzeit-Importe, Footer, Claims.**

Zahl der Tests: siehe `pytest --collect-only -q tests`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `imp_algorithm.py` | Kern: Strom, Aufzählung, exakte Varianz, Simulation, ESS, Pareto-k̂, stetiger Begleiter |
| `imp_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `imp_evaluation.py` | Kennzahlen, Verdikt-Logik, fünf Experimente über feste Ströme |
| `imp_layout.py` | Fester Kopf (Stylesheet) |
| `imp_pdf_export.py` | PDF der aktuellen Einstellung |
| `imp_presets.py` | Permalink (Begrenzen), Presets, Zufallsstrom-Knopf |
| `imp_scenario.py` | Gitter, Hantel und eigene Netze |
| `imp_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, networkx und scipy |
| `requirements.txt` | Laufzeit-Bibliotheken |
| `tests/` | Kern von Hand, Orakel, Eigenschaften, Randfälle, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

| Annahme | Verweis |
|---|---|
| Die Hilfsverteilung wird aus den Ziehungen gelernt | Stück 8 der Linie (adaptives Importance Sampling, Kreuzentropie) |
| Seltene Ereignisse über eine Folge von Zwischenstufen | [splitting-demo](https://github.com/sebastian-hanisch/splitting-demo) (Warteschlangen 9) |
| Einen Teil exakt ausrechnen statt kippen | [bedingtes-monte-carlo-demo](https://github.com/sebastian-hanisch/bedingtes-monte-carlo-demo) (Stück 6) |
| Selbstnormiertes Importance Sampling, Mischungen von Hilfsverteilungen | nicht gebaut |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Monte-Carlo-Schätzung: vom Schätzer zur Wahl](https://sebastianhanisch.net/konzepte-monte-carlo.html).
