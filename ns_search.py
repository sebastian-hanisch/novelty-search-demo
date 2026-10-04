"""Suche mit Fitness oder mit Novelty: derselbe Genetische Algorithmus wie in Stück 1 (Turnier 3, die zwei Besten bleiben, Mutation ohne Kreuzung), nur der Wert, nach dem ausgewählt wird, wechselt.

* **Fitness:** −Abstand der Endposition zum Ziel (Bezug, identisch zu Stück 1).
* **Novelty** (Lehman und Stanley 2008, 2011): mittlerer Abstand der Endposition zu den `k` nächsten Endpositionen in der Referenzmenge aus Population und Archiv. Das Ziel kommt in der Auswahl nicht vor.
  Das Ziel dient nur dazu, den Lauf zu beenden und zu messen (Erfolg: eine Endposition höchstens 2 Zellen vom Ziel).
* **Archiv-Regeln:** `novelty` ohne Archiv (Referenz nur die Population); `archive_top`: je Generation kommen die neuartigsten 5 % der Population ins Archiv; `archive_threshold`: alle Individuen mit Novelty über der Schwelle ρ kommen
  ins Archiv (höchstens 5 je Generation, die neuartigsten; bei Andrang ρ × 1.1, nach 3 Generationen ohne Neuzugang ρ × 0.9).

Zufall nur über `numpy.random.default_rng(seed)` (PCG64, plattformstabil)."""

from dataclasses import dataclass, field

import numpy as np

import ns_constants as C
import ns_maze as M
import ns_robot as R


@dataclass
class SearchResult:
    best_distance: list                  # kleinster Abstand zum Ziel je Generation
    visited_cells: list                  # Zahl der bisher besuchten Zellen (Endpositionen) nach jeder Generation
    best_progress: list                  # kleinster wahrer Restweg (Zellen) aller bisher besuchten Zellen nach jeder Generation
    first_solved: object                 # Zahl der Evaluationen bis zum ersten Individuum im Zielkreis (None: nie)
    end_positions: list = field(default_factory=list)    # Endpositionen (pop, 2) je Generation
    archive: object = None               # Archiv (m, 2) am Ende des Laufs (leer ohne Archiv)
    visited_grid: object = None          # bool-Raster der besuchten Zellen
    best_genome: object = None           # Steuerfolge mit dem kleinsten Abstand über den ganzen Lauf
    evaluations: int = 0


def next_generation(rng, genomes, score, sigma=C.MUT_SIGMA, rate=C.MUT_RATE):
    """Neue Population: die ELITE höchsten `score` bleiben unverändert, der Rest sind Mutationen von Turnier-Siegern (Turniergröße TOURNAMENT). Größer ist besser."""
    pop = len(genomes)
    order = np.argsort(-score, kind="stable")
    nxt = [genomes[i].copy() for i in order[:C.ELITE]]
    while len(nxt) < pop:
        cand = rng.integers(0, pop, C.TOURNAMENT)
        parent = cand[np.argmax(score[cand])]
        child = genomes[parent] + rng.normal(0.0, sigma, genomes[parent].shape) * (rng.random(genomes[parent].shape) < rate)
        nxt.append(R.clip_genomes(child))
    return np.array(nxt)


def novelty_scores(points, archive, k):
    """Novelty jedes Punkts: mittlerer Abstand zu den `k` nächsten anderen Punkten der Referenzmenge aus `points` und `archive`. Der Punkt selbst zählt nicht mit (Abstand 0 zu sich selbst wird übersprungen);
    ein anderer Punkt am selben Ort zählt mit Abstand 0. Hat die Referenzmenge weniger als k + 1 Punkte, werden alle anderen benutzt."""
    ref = np.vstack([archive, points]) if len(archive) else points
    if len(ref) < 2 or k < 1:
        raise ValueError("Novelty braucht mindestens zwei Referenzpunkte und k >= 1")
    d = np.linalg.norm(points[:, None, :] - ref[None, :, :], axis=2)
    m = min(k + 1, d.shape[1])
    d = np.partition(d, m - 1, axis=1)[:, :m]            # die m kleinsten Abstände je Zeile (statt die ganze Zeile zu sortieren; gleiches Ergebnis)
    d.sort(axis=1)
    return d[:, 1:k + 1].mean(axis=1)


def update_archive(variant, archive, points, score, rho, quiet, pop):
    """Archiv nach einer Generation. Gibt (Archiv, Schwelle, Zahl der Generationen ohne Neuzugang) zurück."""
    if variant == "archive_top":
        take = np.argsort(-score, kind="stable")[:max(1, int(round(pop * C.ARCHIVE_TOP_SHARE)))]
        return np.vstack([archive, points[take]]), rho, quiet
    if variant == "archive_threshold":
        cand = np.flatnonzero(score > rho)
        if len(cand) > C.THRESHOLD_MAX_ADDED:
            rho *= C.THRESHOLD_UP
            cand = cand[np.argsort(-score[cand], kind="stable")[:C.THRESHOLD_MAX_ADDED]]
        quiet = 0 if len(cand) else quiet + 1
        if quiet >= C.THRESHOLD_QUIET:
            rho *= C.THRESHOLD_DOWN
            quiet = 0
        return np.vstack([archive, points[cand]]), rho, quiet
    return archive, rho, quiet


def run_search(grid, seed, variant=C.FITNESS, gens=C.STUDY_GENS, pop=C.POP, k=C.K_DEFAULT, sigma=C.MUT_SIGMA, rate=C.MUT_RATE, keep_positions=True, stop_when_solved=False):
    """Ein Lauf der Suche mit der Auswahlvariante `variant` (siehe C.VARIANTS). Zählt Evaluationen (pop je Generation) bis die erste Endposition im Zielkreis liegt, und verfolgt, wie viele Zellen die Suche
    besucht hat und wie nah dem Ziel (entlang des wahren Wegs) sie gekommen ist."""
    if variant not in C.VARIANTS:
        raise ValueError(f"unbekannte Variante: {variant}")
    rng = np.random.default_rng(seed)
    genomes = R.random_genomes(rng, pop)
    true_path = M.bfs_distances(grid, M.cell_of(C.GOAL))
    visited = np.zeros(grid.shape, bool)
    archive = np.zeros((0, 2))
    rho, quiet = C.THRESHOLD_START, 0
    res = SearchResult(best_distance=[], visited_cells=[], best_progress=[], first_solved=None)
    best_d, best_p = np.inf, np.inf
    for gen in range(gens):
        end = R.simulate(genomes, grid)
        dist = R.distance_to_goal(end)
        res.evaluations += pop
        if res.first_solved is None and dist.min() < C.SUCCESS_RADIUS:
            res.first_solved = gen * pop + int(np.argmax(dist < C.SUCCESS_RADIUS)) + 1
        idx = np.clip(end.astype(int), 0, grid.shape[0] - 1)
        visited[idx[:, 0], idx[:, 1]] = True
        best_p = min(best_p, float(true_path[idx[:, 0], idx[:, 1]].min()))
        res.best_distance.append(float(dist.min()))
        res.visited_cells.append(int(visited.sum()))
        res.best_progress.append(best_p)
        if keep_positions:
            res.end_positions.append(end.copy())
        if dist.min() < best_d:
            best_d = float(dist.min())
            res.best_genome = genomes[int(np.argmin(dist))].copy()
        if stop_when_solved and res.first_solved is not None:
            break
        if variant == C.FITNESS:
            score = -dist
        else:
            score = novelty_scores(end, archive, k)
            archive, rho, quiet = update_archive(variant, archive, end, score, rho, quiet, pop)
        genomes = next_generation(rng, genomes, score, sigma, rate)
    res.archive = archive
    res.visited_grid = visited
    return res
