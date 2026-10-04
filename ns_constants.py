"""Konstanten der Novelty-Search-Demo: Labyrinth und Roboter wie in Stück 1 (deception-maze-demo), Suchvarianten, Studien-Achsen, Regler, Voreinstellungen.
Längen in Zellen (Zellraster N × N, Zelle (i, j) belegt den Bereich [i, i + 1) × [j, j + 1))."""


def fmt_int(n):
    """Ganzzahl mit Punkt als Tausendertrenner (10000 -> 10.000)."""
    return f"{n:,}".replace(",", ".")


def fmt_pct(x, digits=0):
    """Anteil als Prozent mit Leerzeichen (0.086 -> "9 %", mit digits=1 "8.6 %")."""
    return f"{x:.{digits}%}".replace("%", " %")


N = 24                                       # Kantenlänge des Labyrinths in Zellen (Rand ist Wand)
START = (3.5, 3.5)                           # Mitte der Startzelle (3, 3)
GOAL = (3.5, 20.5)                           # Mitte der Zielzelle (3, 20)
SUCCESS_RADIUS = 2.0                         # Erfolg: Endposition höchstens so weit vom Ziel
START_HEADING = 0.7853981633974483           # Anfangsrichtung π/4
STEPS = 140                                  # Länge der Steuerfolge
TURN_MAX = 0.6                               # Drehung je Schritt (rad), Gen ∈ [−1, 1]
SPEED_MAX = 0.7                              # Fahrstrecke je Schritt (Zellen), Gen ∈ [0, 1]

POP = 100                                    # Individuen je Generation
ELITE = 2
TOURNAMENT = 3
MUT_SIGMA = 0.25                             # Standardabweichung der Mutation je verändertem Gen
MUT_RATE = 0.15                              # Anteil der Gene, die je Kind mutieren

LEVELS = (0, 1, 2, 3, 4)
LEVEL_LABELS = {0: "0: freies Feld", 1: "1: Öffnung links", 2: "2: Öffnung in der Mitte", 3: "3: Öffnung rechts", 4: "4: Schlangenlinie"}
GAP = 3                                      # Breite einer Öffnung in der Wand (Zellen)

# Suchvarianten: Auswahl nach Fitness (Bezug, wie Stück 1) oder nach Novelty mit drei Archiv-Regeln
FITNESS = "fitness"
VARIANTS = ("fitness", "novelty", "archive_top", "archive_threshold")
VARIANT_LABELS = {"fitness": "Fitness (Abstand zum Ziel)", "novelty": "Novelty, nur Population", "archive_top": "Novelty, Archiv der Neuartigsten",
                  "archive_threshold": "Novelty, Archiv mit Schwelle"}
NOVELTY_VARIANTS = ("novelty", "archive_top", "archive_threshold")   # im Live-Lauf wählbar; die Fitness läuft immer daneben
ARCHIVE_TOP_SHARE = 0.05                     # Archiv der Neuartigsten: Anteil der Population je Generation (5 von 100)
THRESHOLD_START = 1.0                        # Schwelle ρ: Novelty (Zellen), ab der ein Individuum ins Archiv kommt
THRESHOLD_MAX_ADDED = 5                      # mehr als so viele Kandidaten je Generation: Schwelle × 1.1, es kommen nur die neuartigsten 5 ins Archiv
THRESHOLD_UP, THRESHOLD_DOWN = 1.1, 0.9      # Schwelle steigt bei Andrang und fällt nach 3 Generationen ohne Neuzugang
THRESHOLD_QUIET = 3
K_DEFAULT = 15                               # Nachbarn der Novelty (k)

STUDY_SEEDS = 20                             # Läufe je Zelle der Studie
STUDY_GENS = 400                             # Generationen je Lauf (bei POP = 100: 40 000 Evaluationen)
STUDY_BUDGET = POP * STUDY_GENS
LONG_GENS = 4000                             # verlängerter Lauf auf Stufe 4 (zehnfaches Budget: 400 000 Evaluationen)
LONG_BUDGET = POP * LONG_GENS
LONG_VARIANTS = ("fitness", "novelty", "archive_top")   # das Schwellen-Archiv wächst bei 400 000 Evaluationen auf Tausende Punkte; hier nicht mitgerechnet
LONG_STEP = 10                              # Verlauf des verlängerten Laufs wird alle 10 Generationen gespeichert
SWEEP_LEVELS = (3, 4)                        # Stufen des Einstellungs-Rasters (nur Variante „Novelty, nur Population“)
SWEEP_KS = (5, 15, 30)
SWEEP_POPS = (50, 100, 200)                  # Populationsgrößen bei gleichem Gesamtbudget

GENS_MIN, GENS_MAX, GENS_STEP, DEFAULT_GENS = 10, 200, 10, 100      # Regler: Generationen des Live-Laufs
K_MIN, K_MAX, K_STEP = 1, 30, 1
DEFAULT_LEVEL = 3
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_VARIANT = "novelty"

PRESET_ORDER = ("Falle: Fitness gegen Novelty", "Freies Feld", "Schlangenlinie", "Mit Archiv")


def _preset(level=DEFAULT_LEVEL, gens=DEFAULT_GENS, variant=DEFAULT_VARIANT, k=K_DEFAULT):
    return {"level": level, "gens": gens, "seed": DEFAULT_SEED, "variant": variant, "k": k}


PRESETS = {
    "Falle: Fitness gegen Novelty": _preset(),
    "Freies Feld": _preset(level=0),
    "Schlangenlinie": _preset(level=4, gens=200),
    "Mit Archiv": _preset(variant="archive_top"),
}
# Zahlen aus der vorgerechneten Studie (20 Läufe je Zelle); tests/test_claims.py rechnet jede nach
PRESET_HELP = {
    "Falle: Fitness gegen Novelty": "Stufe 3 (Falle 9 Zellen vor dem Ziel): Novelty ohne Archiv erreicht das Ziel in 20 von 20 Läufen (Median 988.5 Evaluationen), die Fitness in 11 von 20; Novelty besucht in jedem Lauf alle 465 freien Zellen, die Fitness im Mittel 343.9.",
    "Freies Feld": "Stufe 0, keine Täuschung: beide Suchen lösen 20 von 20 Läufen, im Median nach 63.5 Evaluationen; Novelty hat hier keinen Vorteil, besucht aber alle 484 Zellen (die Fitness im Mittel 327.15).",
    "Schlangenlinie": "Stufe 4 (kürzester Weg 77 Zellen): keine der Varianten erreicht das Ziel (0 von 20); der kleinste wahre Restweg liegt bei Novelty im Mittel bei 26.25 Zellen, bei der Fitness bei 34.75.",
    "Mit Archiv": "Stufe 3 mit Archiv der Neuartigsten (je Generation die 5 neuartigsten Endpositionen): ebenfalls 20 von 20 Läufen, Median 1 195 gegen 988.5 ohne Archiv; das Archiv bringt hier keinen Gewinn.",
}
