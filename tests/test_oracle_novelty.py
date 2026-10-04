"""Orakel-Tests für die Novelty: unabhängige Gegenproben für den k-Nächste-Nachbarn-Kern und die besuchten Zellen.

* Novelty gegen SciPys `cKDTree` (ein KD-Baum statt der Abstandsmatrix mit Teilsortierung) mit und ohne Archiv, für mehrere k, mit doppelten Punkten.
* Eine skalare Neuberechnung nur aus der Definition (zwei Schleifen, `math.hypot`, `sorted`).
* Die besuchten Zellen eines Laufs gegen eine Zählung über eine Menge von Zellen-Tupeln aus den gespeicherten Endpositionen."""

import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

import ns_constants as C
import ns_maze as M
import ns_search as S


def kdtree_novelty(points, archive, k):
    ref = np.vstack([archive, points]) if len(archive) else points
    dist, _ = cKDTree(ref).query(points, k=min(k + 1, len(ref)))
    return dist[:, 1:].mean(axis=1)


def scalar_novelty(points, archive, k):
    """Nur aus der Definition: Abstände zu allen anderen Punkten (Population ohne sich selbst, plus Archiv), die k kleinsten mitteln."""
    out = []
    for i, p in enumerate(points):
        others = [math.hypot(p[0] - q[0], p[1] - q[1]) for j, q in enumerate(points) if j != i]
        others += [math.hypot(p[0] - q[0], p[1] - q[1]) for q in archive]
        near = sorted(others)[:k]
        out.append(sum(near) / len(near))
    return np.array(out)


@pytest.mark.parametrize("k", [1, 3, 15, 30])
@pytest.mark.parametrize("archive_size", [0, 7, 120])
def test_novelty_agrees_with_a_kd_tree_and_with_the_scalar_definition(k, archive_size):
    rng = np.random.default_rng(100 + k + archive_size)
    points = rng.uniform(0, 24, (40, 2))
    archive = rng.uniform(0, 24, (archive_size, 2))
    mine = S.novelty_scores(points, archive, k)
    assert mine == pytest.approx(kdtree_novelty(points, archive, k), abs=1e-12)
    assert mine == pytest.approx(scalar_novelty(points, archive, k), abs=1e-12)


def test_novelty_with_many_duplicate_points_agrees_too():
    """Gestrandete Roboter enden auf demselben Punkt; solche Gleichstände sind der Normalfall an Wänden."""
    rng = np.random.default_rng(7)
    base = rng.uniform(0, 24, (6, 2))
    points = base[rng.integers(0, 6, 50)]
    for k in (1, 5, 15):
        assert S.novelty_scores(points, np.zeros((0, 2)), k) == pytest.approx(kdtree_novelty(points, np.zeros((0, 2)), k), abs=1e-12)
        assert S.novelty_scores(points, np.zeros((0, 2)), k) == pytest.approx(scalar_novelty(points, [], k), abs=1e-12)


def test_the_most_novel_point_of_a_cluster_with_one_outlier_is_the_outlier():
    pts = np.vstack([np.random.default_rng(3).uniform(10, 11, (30, 2)), [[20.0, 2.0]]])
    assert int(np.argmax(S.novelty_scores(pts, np.zeros((0, 2)), 5))) == 30


def test_visited_cells_agree_with_a_set_of_cell_tuples_over_the_stored_end_positions():
    grid = M.make_maze(3)
    res = S.run_search(grid, 5, "novelty", gens=30, keep_positions=True)
    seen = set()
    counts = []
    for end in res.end_positions:
        seen.update((int(x), int(y)) for x, y in end)
        counts.append(len(seen))
    assert res.visited_cells == counts and res.visited_cells[-1] == int(res.visited_grid.sum())
    assert all(not grid[c] for c in seen) and len(seen) <= int((~grid).sum())


def test_best_progress_agrees_with_a_scalar_search_of_the_visited_cells():
    """Kleinster wahrer Restweg über alle besuchten Zellen, mit der Breitensuche von SciPy statt der eigenen."""
    from test_oracle_maze import scipy_distances
    grid = M.make_maze(4)
    res = S.run_search(grid, 2, "fitness", gens=30, keep_positions=True)
    ref = scipy_distances(grid, M.cell_of(C.GOAL))
    best, series = np.inf, []
    for end in res.end_positions:
        best = min(best, min(ref[int(x), int(y)] for x, y in end))
        series.append(best)
    assert res.best_progress == pytest.approx(series, abs=0)
