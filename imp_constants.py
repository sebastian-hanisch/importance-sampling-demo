"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

NETZE = {"3x3": ("gitter", 3, 3), "3x4": ("gitter", 3, 4), "hantel": ("hantel", 0, 0)}
DEFAULT_NETZ = "3x4"
LIVE_EDGE_LIMIT = 17          # Aufzählung aller Kantenmengen live bis hierher (2^17 = 131 072, rund 1 s); eigenes Netz ebenso
CUSTOM_EDGE_LIMIT = 17
CUSTOM_NODE_LIMIT = 14

Q_CHOICES = (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)    # Ausfallwahrscheinlichkeit je Kante (Seltenheit)
Q_MIN, Q_MAX, DEFAULT_Q = Q_CHOICES[-1], Q_CHOICES[0], 0.01
QP_MIN, QP_MAX, DEFAULT_QP = 0.005, 0.8, 0.12                # Hilfsverteilung: jede Kante fällt mit q' aus
N_CHOICES = (100, 300, 1000, 3000)                           # Ziehungen je Schätzung
N_MIN, N_MAX, DEFAULT_N = N_CHOICES[0], N_CHOICES[-1], 1000
SEED_MAX, DEFAULT_SEED = 999999, 35
REPS = 400                                # Wiederholungen je Schätzung (feste Ströme)

# Stetiger Begleiter: P(X > t) mit X ~ Exp(1), Hilfsverteilung Exp(theta)
T_MIN, T_MAX, DEFAULT_T = 1.0, 4.0, 2.0
THETA_MIN, THETA_MAX, DEFAULT_THETA = 0.2, 3.0, 1.5
CONT_N = 10_000
CONT_REPS = 400
CONT_THETAS = (0.5, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0)

# Experimente
RELERR = 0.10                             # Zielgenauigkeit: 10 % relativer Fehler (Zahl der Ziehungen = CV^2 / RELERR^2)
SWEEP_REPS = 300
SWEEP_FACTORS = (1.0, 2.0, 5.0, 10.0)     # Vielfache von q für die Simulationsreihe (plus optimales q' und große Werte)
SWEEP_ABS = (0.3, 0.5, 0.7)

Z95 = 1.959963984540054
K_UNRELIABLE = 0.7                        # Pareto-k̂ über dieser Marke gilt (PSIS, Vehtari et al.) als nicht verlässlich
K_FINITE_VARIANCE = 0.5                   # Formparameter 1/2: Grenze zur unendlichen Varianz

COLOR_PLAIN = "#4c78a8"
COLOR_IS = "#2F6B65"
COLOR_BAD = "#C0392B"
COLOR_TRUTH = "#222222"
COLOR_OTHER = "rgba(150,150,150,0.55)"
COLOR_ACCENT = "#E08E0B"

PRESETS = {
    "Standardfall (Voreinstellung)": {"netz": DEFAULT_NETZ, "q": DEFAULT_Q, "qp": DEFAULT_QP, "n": DEFAULT_N, "t": DEFAULT_T, "theta": DEFAULT_THETA, "seed": DEFAULT_SEED},
    "Einfaches Ziehen sieht nichts": {"netz": DEFAULT_NETZ, "q": DEFAULT_Q, "qp": DEFAULT_Q, "n": DEFAULT_N, "t": DEFAULT_T, "theta": DEFAULT_THETA, "seed": DEFAULT_SEED},
    "Zu stark gekippt": {"netz": DEFAULT_NETZ, "q": DEFAULT_Q, "qp": 0.5, "n": DEFAULT_N, "t": DEFAULT_T, "theta": DEFAULT_THETA, "seed": DEFAULT_SEED},
    "Eine kritische Brücke": {"netz": "hantel", "q": DEFAULT_Q, "qp": 0.14, "n": DEFAULT_N, "t": DEFAULT_T, "theta": DEFAULT_THETA, "seed": DEFAULT_SEED},
    "Unendliche Varianz (stetiger Begleiter)": {"netz": DEFAULT_NETZ, "q": DEFAULT_Q, "qp": DEFAULT_QP, "n": DEFAULT_N, "t": DEFAULT_T, "theta": 2.5, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Gitter 3×4, jede Kante fällt mit 1 % aus, Hilfsverteilung mit 12 % (kleinster Schnitt 2 Kanten von 17): der seltene Ausfall wird sicher geschätzt.",
    "Einfaches Ziehen sieht nichts": "Dieselbe Seltenheit ohne Hilfsverteilung (q' = q): bei 1000 Ziehungen sehen zwei von drei Läufen keinen einzigen Ausfall und melden null.",
    "Zu stark gekippt": "Hilfsverteilung mit 50 %: fast jede Ziehung ist ein Ausfall, aber die Gewichte sind so ungleich, dass das Ergebnis schlechter ist als ohne Kippung.",
    "Eine kritische Brücke": "Hantel (zwei Dreiecke, eine Brücke, kleinster Schnitt 1 Kante von 7): das optimale q' liegt bei 1/7.",
    "Unendliche Varianz (stetiger Begleiter)": "Der Exponentialschwanz mit zu leichter Hilfsverteilung (θ = 2,5): die Varianz ist unendlich, Pareto-k̂ liegt über 0,5.",
}
