"""Auswertung: Täuschungsmaße je Stufe (exakt), Lösbarkeitsnachweis, Live-Lauf (Fitness neben einer Novelty-Variante), Lesen und Aggregieren der vorgerechneten Studie.
Die Studie (Erfolgsquoten über Seeds, besuchte Zellen, Einstellungs-Raster, verlängerter Lauf) wird beim Bau gerechnet (`generate_precomputed.py`) und hier nur gelesen."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import ns_constants as C
import ns_maze as M
import ns_robot as R
import ns_search as S

PRECOMPUTED_PATH = Path(__file__).resolve().parent / "precomputed_sweep.json"


def level_metrics(level):
    """Exakte Täuschungsmaße der Stufe: kürzester Weg (Zellen), Luftlinie Start–Ziel, Umwegfaktor, Fitness-Distanz-Korrelation, lokale Minima (Zahl), Abstand des tiefsten Minimums zum Ziel, freie Zellen."""
    g = M.make_maze(level)
    path = M.shortest_path(g)
    return {"path": len(path) - 1, "straight": float(np.hypot(C.START[0] - C.GOAL[0], C.START[1] - C.GOAL[1])), "detour": M.detour_ratio(g), "fdc": M.fitness_distance_correlation(g),
            "minima": len(M.local_minima(g)), "trap": M.trap_distance(g), "free": int((~g).sum())}


def solvability(level):
    """Lösbarkeitsnachweis: eine aus dem kürzesten Pfad abgeleitete Steuerfolge fährt in diesem Labyrinth ins Ziel. Gibt (Abstand am Ende, benötigte Schritte) zurück."""
    g = M.make_maze(level)
    genome = R.controller_from_path(M.shortest_path(g))
    end = R.simulate(genome[None], g)
    return float(R.distance_to_goal(end)[0]), int((np.abs(genome).sum(axis=1) > 0).sum())


def live_run(level, gens, seed, variant, k):
    """Ein Lauf mit der Variante `variant` auf der gewählten Stufe, samt bester Fahrt (Positionen je Schritt)."""
    g = M.make_maze(level)
    res = S.run_search(g, seed, variant, gens=gens, k=k)
    _, track = R.simulate(res.best_genome[None], g, trajectories=True)
    return res, track[0]


def live_pair(level, gens, seed, variant, k):
    """Derselbe Seed, dieselbe Stufe: der Fitness-Lauf (Bezug) und der Lauf der gewählten Novelty-Variante. Gibt ((Lauf, Fahrt) der Fitness, (Lauf, Fahrt) der Novelty) zurück."""
    return live_run(level, gens, seed, C.FITNESS, k), live_run(level, gens, seed, variant, k)


@lru_cache(maxsize=1)
def load_precomputed():
    return json.loads(PRECOMPUTED_PATH.read_text(encoding="utf-8"))


def success_count(runs, budget=None):
    """Zahl der Läufe, die den Zielkreis erreichten (höchstens `budget` Evaluationen, wenn angegeben)."""
    return sum(1 for r in runs if r is not None and (budget is None or r <= budget))


def success_curve(runs, budgets):
    """Anteil erfolgreicher Läufe je Evaluationsbudget."""
    return [success_count(runs, b) / len(runs) for b in budgets]


def median_solved(runs):
    """Median der Evaluationen bis zum Erfolg über die erfolgreichen Läufe (None, wenn kein Lauf erfolgreich war)."""
    ok = [r for r in runs if r is not None]
    return float(np.median(ok)) if ok else None


def level_info(pre, level):
    return pre["levels"][str(level)]["metrics"]


def study_cell(pre, level, variant):
    return pre["levels"][str(level)]["variants"][variant]


def sweep_cell(pre, level, k, pop):
    return next(x for x in pre["sweep"] if x["level"] == level and x["k"] == k and x["pop"] == pop)


def long_cell(pre, variant):
    return pre["long"]["variants"][variant]
