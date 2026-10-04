"""Rechnet die Studie vor (Build-Zeit, nicht in der App): `python generate_precomputed.py [Prozesse]` schreibt `precomputed_sweep.json`.

  levels  je Stufe 0 bis 4 und je Variante (Fitness, Novelty ohne Archiv, mit Archiv der Neuartigsten, mit Schwellen-Archiv): 20 Läufe à 400 Generationen × 100 Individuen = 40 000 Evaluationen:
          Evaluationen bis zum ersten Erfolg (None = nie), besuchte Zellen und kleinster wahrer Restweg am Ende, mittlerer Verlauf beider Größen; dazu die exakten Täuschungsmaße der Stufe
  sweep   Stufen 3 und 4, Variante „Novelty, nur Population“, gleiches Gesamtbudget (40 000 Evaluationen): k = 5 / 15 / 30 × Populationsgröße 50 / 100 / 200, je 20 Läufe
  long    Stufe 4 mit zehnfachem Budget (4 000 Generationen = 400 000 Evaluationen): Fitness, Novelty ohne Archiv, Novelty mit Archiv der Neuartigsten, je 20 Läufe"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import ns_constants as C
import ns_evaluation as E
import ns_maze as M
import ns_search as S


def _level_task(args):
    level, variant, seed = args
    res = S.run_search(M.make_maze(level), seed, variant, gens=C.STUDY_GENS, keep_positions=False)
    return level, variant, seed, res.first_solved, res.visited_cells, res.best_progress


def _sweep_task(args):
    level, k, pop, seed = args
    res = S.run_search(M.make_maze(level), seed, "novelty", gens=C.STUDY_BUDGET // pop, pop=pop, k=k, keep_positions=False)
    return level, k, pop, seed, res.first_solved, res.best_progress[-1], res.visited_cells[-1]


def _long_task(args):
    variant, seed = args
    res = S.run_search(M.make_maze(4), seed, variant, gens=C.LONG_GENS, keep_positions=False)
    return variant, seed, res.first_solved, res.visited_cells, res.best_progress


def _curve(rows, index, step=1):
    """Mittelwert über die Läufe, je `step` Generationen (am Ende jedes Schritts) auf 3 Stellen gerundet."""
    arr = np.mean([r[index] for r in rows], axis=0)
    return [round(float(x), 3) for x in arr[step - 1::step]]


def _by_seed(rows):
    return sorted(rows, key=lambda r: r[2])


def main(workers):
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        level_res = list(ex.map(_level_task, [(lv, v, s) for lv in C.LEVELS for v in C.VARIANTS for s in range(C.STUDY_SEEDS)], chunksize=1))
        print(f"Stufen fertig nach {time.time() - t0:.0f} s", flush=True)
        sweep_res = list(ex.map(_sweep_task, [(lv, k, p, s) for lv in C.SWEEP_LEVELS for k in C.SWEEP_KS for p in C.SWEEP_POPS for s in range(C.STUDY_SEEDS)], chunksize=1))
        print(f"Raster fertig nach {time.time() - t0:.0f} s", flush=True)
        long_res = list(ex.map(_long_task, [(v, s) for v in C.LONG_VARIANTS for s in range(C.STUDY_SEEDS)], chunksize=1))
    levels = {}
    for lv in C.LEVELS:
        variants = {}
        for v in C.VARIANTS:
            rows = _by_seed([r for r in level_res if r[0] == lv and r[1] == v])
            variants[v] = {"runs": [r[3] for r in rows], "cells": [r[4][-1] for r in rows], "progress": [r[5][-1] for r in rows], "mean_visited": _curve(rows, 4), "mean_progress": _curve(rows, 5)}
        levels[str(lv)] = {"metrics": E.level_metrics(lv), "variants": variants}
    sweep = []
    for lv in C.SWEEP_LEVELS:
        for k in C.SWEEP_KS:
            for p in C.SWEEP_POPS:
                rows = sorted([r for r in sweep_res if r[:3] == (lv, k, p)], key=lambda r: r[3])
                sweep.append({"level": lv, "k": k, "pop": p, "runs": [r[4] for r in rows], "progress": [r[5] for r in rows], "cells": [r[6] for r in rows]})
    long = {}
    for v in C.LONG_VARIANTS:
        rows = sorted([r for r in long_res if r[0] == v], key=lambda r: r[1])
        long[v] = {"runs": [r[2] for r in rows], "cells": [r[3][-1] for r in rows], "progress": [r[4][-1] for r in rows],
                   "mean_visited": [round(float(x), 3) for x in np.mean([r[3] for r in rows], axis=0)[C.LONG_STEP - 1::C.LONG_STEP]],
                   "mean_progress": [round(float(x), 3) for x in np.mean([r[4] for r in rows], axis=0)[C.LONG_STEP - 1::C.LONG_STEP]]}
    data = {"seeds": C.STUDY_SEEDS, "budget": C.STUDY_BUDGET, "levels": levels, "sweep": sweep, "long": {"budget": C.LONG_BUDGET, "variants": long}}
    E.PRECOMPUTED_PATH.write_text(json.dumps(data), encoding="utf-8")
    print(f"fertig in {time.time() - t0:.0f} s -> {E.PRECOMPUTED_PATH}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else min(14, os.cpu_count() or 1))
