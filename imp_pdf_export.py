"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import imp_constants as C
from imp_evaluation import Analysis, Settings, de, de_e, dn, dz, verdict_main

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "€": "EUR", "„": '"', "“": '"', "²": "^2", "³": "3", "×": "x", "√": "Wurzel", "≈": "~",
            "σ": "Sigma", "ρ": "rho", "α": "alpha", "β": "beta", "θ": "theta", "∞": "unendlich", "Σ": "Summe", "δ": "delta", "Δ": "Delta", "½": "1/2", "̂": "", "ϱ": "rho", "k̂": "k-Dach"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, a: Analysis) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Importance Sampling: seltene Ausfälle schätzen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(
        f"{a.netz.name}, {a.netz.n} Knoten, {a.netz.m} Kanten; jede Kante fällt mit Wahrscheinlichkeit q = {de(settings.q, 3)} aus; Ereignis: das Netz hängt nicht zusammen. "
        f"Hilfsverteilung q' = {de(settings.qp, 3)}, {de(settings.n, 0)} Ziehungen je Schätzung, {C.REPS} Wiederholungen, Strom {settings.seed}. Alles simuliert, keine Messdaten."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    widths = (70, 60)
    pdf.cell(widths[0], 6, _s("Kennzahl"), border=1)
    pdf.cell(widths[1], 6, _s("Wert"), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for label, val in (("Ausfallwahrscheinlichkeit P (exakt)", de_e(a.P)), ("CV^2 einfaches Ziehen (exakt)", dz(a.cv2_plain)), ("CV^2 bei der Kippung (exakt)", dz(a.cv2_is)),
                       ("Gewinn", dz(a.gain)), ("optimale Kippung q'", dn(a.qp_opt, 3)), ("Ziehungen für 10 % (einfach / gekippt)", f"{dz(a.n_plain)} / {dz(a.n_is)}"),
                       ("mediane Abweichung (gemessen)", f"{de(a.med_rel * 100, 0)} %"), ("Läufe ohne Ausfall", f"{de(a.zero * 100, 0)} %"), ("Überdeckung des 95-%-Intervalls", de(a.cover, 2)),
                       ("ESS gemessen / Formel", f"{de(a.ess_meas, 1)} / {de(a.ess_exact, 1)}"), ("Pareto-k-Dach (Median)", dn(a.k_med, 2))):
        pdf.cell(widths[0], 6, _s(label), border=1)
        pdf.cell(widths[1], 6, _s(val), border=1)
        pdf.ln()
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(verdict_main(a, settings)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: importance-sampling-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
