"""Suche: Novelty von Hand, Archiv-Regeln von Hand, Auswahl nach Novelty statt Fitness, Reproduzierbarkeit, Zählungen (Evaluationen, besuchte Zellen, Restweg)."""

import numpy as np
import pytest

import ns_constants as C
import ns_maze as M
import ns_robot as R
import ns_search as S


# ---- Novelty von Hand ------------------------------------------------------------------------------------------------------------------------------

LINE = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [10.0, 0.0]])


def test_novelty_with_one_neighbour_is_the_distance_to_the_nearest_other_point():
    assert S.novelty_scores(LINE, np.zeros((0, 2)), 1) == pytest.approx([1.0, 1.0, 2.0, 7.0])


def test_novelty_with_two_neighbours_is_the_mean_of_the_two_nearest_distances():
    # Punkt 0: Abstände 1 und 3 -> 2; Punkt 1: 1 und 2 -> 1.5; Punkt 2: 2 und 3 -> 2.5; Punkt 3: 7 und 9 -> 8
    assert S.novelty_scores(LINE, np.zeros((0, 2)), 2) == pytest.approx([2.0, 1.5, 2.5, 8.0])


def test_the_point_itself_does_not_count_but_a_duplicate_does():
    pts = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 0.0]])
    assert S.novelty_scores(pts, np.zeros((0, 2)), 1) == pytest.approx([0.0, 0.0, 5.0])


def test_an_archive_point_at_the_same_place_removes_the_novelty():
    """Der Punkt bei 10 war der neuartigste (7); steht ein Archivpunkt am selben Ort, ist seine Novelty 0."""
    archive = np.array([[10.0, 0.0]])
    out = S.novelty_scores(LINE, archive, 1)
    assert out == pytest.approx([1.0, 1.0, 2.0, 0.0])


def test_k_larger_than_the_reference_uses_all_other_points():
    assert S.novelty_scores(LINE, np.zeros((0, 2)), 30) == pytest.approx([(1 + 3 + 10) / 3, (1 + 2 + 9) / 3, (3 + 2 + 7) / 3, (10 + 9 + 7) / 3])


def test_novelty_needs_at_least_two_reference_points_and_k_one():
    with pytest.raises(ValueError):
        S.novelty_scores(np.array([[1.0, 1.0]]), np.zeros((0, 2)), 1)
    with pytest.raises(ValueError):
        S.novelty_scores(LINE, np.zeros((0, 2)), 0)


# ---- Archiv-Regeln von Hand --------------------------------------------------------------------------------------------------------------------------

def _points(n):
    return np.column_stack([np.arange(n, dtype=float), np.zeros(n)])


def test_no_archive_variant_never_changes_the_archive():
    pts = _points(100)
    arch, rho, quiet = S.update_archive("novelty", np.zeros((0, 2)), pts, np.arange(100.0), 1.0, 0, 100)
    assert len(arch) == 0 and rho == 1.0 and quiet == 0


def test_top_archive_takes_the_most_novel_five_percent():
    pts = _points(100)
    score = np.arange(100.0)                                  # Punkt i hat Novelty i: die neuartigsten sind 95 bis 99
    arch, _, _ = S.update_archive("archive_top", np.zeros((0, 2)), pts, score, 1.0, 0, 100)
    assert sorted(arch[:, 0]) == [95.0, 96.0, 97.0, 98.0, 99.0]
    arch2, _, _ = S.update_archive("archive_top", arch, pts, score, 1.0, 0, 100)
    assert len(arch2) == 10
    small, _, _ = S.update_archive("archive_top", np.zeros((0, 2)), _points(10), np.arange(10.0), 1.0, 0, 10)
    assert len(small) == 1 and small[0, 0] == 9.0              # mindestens ein Individuum


def test_threshold_archive_with_few_candidates_takes_all_of_them_and_keeps_rho():
    pts = _points(10)
    score = np.array([0, 0, 0, 0, 0, 0, 0, 1.5, 2.0, 3.0])
    arch, rho, quiet = S.update_archive("archive_threshold", np.zeros((0, 2)), pts, score, 1.0, 0, 10)
    assert sorted(arch[:, 0]) == [7.0, 8.0, 9.0] and rho == 1.0 and quiet == 0


def test_threshold_archive_with_a_crowd_takes_the_five_most_novel_and_raises_rho():
    pts = _points(10)
    score = np.arange(10.0) + 5                                # alle 10 über der Schwelle 1
    arch, rho, quiet = S.update_archive("archive_threshold", np.zeros((0, 2)), pts, score, 1.0, 0, 10)
    assert sorted(arch[:, 0]) == [5.0, 6.0, 7.0, 8.0, 9.0] and rho == pytest.approx(1.1) and quiet == 0


def test_threshold_archive_lowers_rho_after_three_quiet_generations():
    pts, score = _points(5), np.zeros(5)
    arch = np.zeros((0, 2))
    rho, quiet = 1.0, 0
    for expected_quiet, expected_rho in ((1, 1.0), (2, 1.0), (0, 0.9)):
        arch, rho, quiet = S.update_archive("archive_threshold", arch, pts, score, rho, quiet, 5)
        assert (quiet, rho) == (expected_quiet, pytest.approx(expected_rho)) and len(arch) == 0


def test_a_quiet_generation_resets_when_something_is_added():
    arch, rho, quiet = S.update_archive("archive_threshold", np.zeros((0, 2)), _points(3), np.array([0.0, 0.0, 2.0]), 1.0, 2, 3)
    assert quiet == 0 and len(arch) == 1


# ---- Auswahl und GA-Einheiten --------------------------------------------------------------------------------------------------------------------------

def test_next_generation_keeps_the_elite_unchanged_and_keeps_the_size():
    genomes = R.random_genomes(np.random.default_rng(1), 10)
    score = np.array([0.0, 5.0, 1.0, 9.0, 2.0, 3.0, 4.0, 8.0, 7.0, 6.0])                # beste: Index 3 (9), 7 (8)
    nxt = S.next_generation(np.random.default_rng(2), genomes, score)
    assert nxt.shape == genomes.shape and (nxt[0] == genomes[3]).all() and (nxt[1] == genomes[7]).all()


def test_without_mutation_every_child_is_a_copy_of_a_parent():
    genomes = R.random_genomes(np.random.default_rng(1), 8)
    nxt = S.next_generation(np.random.default_rng(2), genomes, np.arange(8, dtype=float), rate=0.0)
    for child in nxt:
        assert any((child == parent).all() for parent in genomes)


def test_the_tournament_prefers_higher_scores():
    genomes = R.random_genomes(np.random.default_rng(5), 60)
    nxt = S.next_generation(np.random.default_rng(6), genomes, np.arange(60, dtype=float), rate=0.0)
    parents = [next(i for i in range(60) if (child == genomes[i]).all()) for child in nxt[C.ELITE:]]
    assert len(parents) == 58 and np.mean(parents) > 36


def test_children_respect_the_gene_bounds():
    genomes = R.random_genomes(np.random.default_rng(7), 20)
    nxt = S.next_generation(np.random.default_rng(8), genomes, np.arange(20, dtype=float), sigma=5.0, rate=1.0)
    assert nxt[..., 0].min() >= -1 and nxt[..., 0].max() <= 1 and nxt[..., 1].min() >= 0 and nxt[..., 1].max() <= 1


# ---- Läufe ---------------------------------------------------------------------------------------------------------------------------------------------

def test_unknown_variant_is_refused():
    with pytest.raises(ValueError):
        S.run_search(M.make_maze(0), 1, "zufall", gens=2)


@pytest.mark.parametrize("variant", C.VARIANTS)
def test_runs_are_reproducible_and_depend_on_the_seed(variant):
    grid = M.make_maze(2)
    a, b, c = (S.run_search(grid, s, variant, gens=12, keep_positions=False) for s in (4, 4, 5))
    assert a.best_distance == b.best_distance and a.first_solved == b.first_solved and a.visited_cells == b.visited_cells and a.best_distance != c.best_distance


@pytest.mark.parametrize("variant", C.VARIANTS)
def test_counts_and_series_have_the_right_shape(variant):
    res = S.run_search(M.make_maze(3), 3, variant, gens=25, keep_positions=True)
    assert res.evaluations == 25 * C.POP and len(res.best_distance) == len(res.visited_cells) == len(res.best_progress) == len(res.end_positions) == 25
    assert all(b >= a for a, b in zip(res.visited_cells, res.visited_cells[1:])) and res.visited_cells[-1] == int(res.visited_grid.sum())
    assert all(b <= a for a, b in zip(res.best_progress, res.best_progress[1:])) and res.end_positions[0].shape == (C.POP, 2)


def test_the_fitness_variant_never_gets_worse_because_the_elite_survives():
    res = S.run_search(M.make_maze(3), 3, "fitness", gens=40, keep_positions=False)
    assert all(b <= a + 1e-12 for a, b in zip(res.best_distance, res.best_distance[1:]))


def test_the_archive_grows_by_the_rule_of_each_variant():
    grid = M.make_maze(3)
    assert len(S.run_search(grid, 1, "fitness", gens=20, keep_positions=False).archive) == 0
    assert len(S.run_search(grid, 1, "novelty", gens=20, keep_positions=False).archive) == 0
    assert len(S.run_search(grid, 1, "archive_top", gens=20, keep_positions=False).archive) == 20 * 5
    thr = S.run_search(grid, 1, "archive_threshold", gens=20, keep_positions=False)
    assert 0 < len(thr.archive) <= 20 * C.THRESHOLD_MAX_ADDED


def test_the_goal_is_not_used_by_the_novelty_selection():
    """Dasselbe Labyrinth mit einem anderen Ziel: die Novelty-Läufe sind bis auf das Zählen des Erfolgs gleich; die Fitness-Läufe nicht."""
    grid = M.make_maze(3)
    base = S.run_search(grid, 2, "novelty", gens=30, keep_positions=True)
    old_goal = C.GOAL
    try:
        C.GOAL = (20.5, 20.5)
        moved = S.run_search(grid, 2, "novelty", gens=30, keep_positions=True)
        moved_fit = S.run_search(grid, 2, "fitness", gens=30, keep_positions=True)
    finally:
        C.GOAL = old_goal
    same_fit = S.run_search(grid, 2, "fitness", gens=30, keep_positions=True)
    assert all((a == b).all() for a, b in zip(base.end_positions, moved.end_positions))
    assert any((a != b).any() for a, b in zip(same_fit.end_positions, moved_fit.end_positions))


def test_first_solved_counts_evaluations_by_hand_on_the_free_field():
    """Erste Generation mit einem Individuum im Zielkreis: Zahl der Evaluationen = (Generation − 1) · pop + Index des ersten Treffers + 1."""
    grid = M.make_maze(0)
    res = S.run_search(grid, 3, "fitness", gens=30, keep_positions=True)
    for gen, end in enumerate(res.end_positions):
        hit = np.flatnonzero(R.distance_to_goal(end) < C.SUCCESS_RADIUS)
        if len(hit):
            assert res.first_solved == gen * C.POP + int(hit[0]) + 1
            break
    else:
        assert res.first_solved is None


def test_stop_when_solved_ends_the_run_early():
    res = S.run_search(M.make_maze(0), 3, "fitness", gens=300, keep_positions=False, stop_when_solved=True)
    assert res.first_solved is not None and len(res.best_distance) < 300 and len(res.best_distance) == (res.first_solved - 1) // C.POP + 1
