"""Abbildungen: gesperrte Achsen, Spuren, Beschriftungen, Reihenfolge und Farben der Varianten."""

import numpy as np

import ns_constants as C
import ns_evaluation as E
import ns_maze as M
import ns_visualization as V

PRE = E.load_precomputed()


def _locked(fig):
    return all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())


def test_all_charts_lock_their_axes():
    grid = M.make_maze(3)
    (fit, ft), (nov, nt) = E.live_pair(3, 10, 1, "archive_top", 15)
    figs = [V.build_run_figure(grid, fit.end_positions[-1], ft, fit.visited_grid), V.build_run_figure(grid, nov.end_positions[-1], nt, nov.visited_grid, nov.archive),
            V.build_progress_chart(fit.best_distance, nov.best_distance, "archive_top", 9.0), V.build_progress_chart(fit.best_distance, nov.best_distance, "novelty", None),
            V.build_coverage_chart(fit.visited_cells, nov.visited_cells, "novelty", 465), V.build_success_chart(PRE), V.build_long_chart(PRE),
            *[V.build_budget_chart(PRE, lv) for lv in C.LEVELS], *[V.build_study_coverage_chart(PRE, lv) for lv in C.LEVELS], *[V.build_sweep_chart(PRE, lv) for lv in C.SWEEP_LEVELS]]
    assert all(_locked(f) for f in figs)


def test_run_figure_shows_the_archive_only_when_there_is_one():
    grid = M.make_maze(3)
    (fit, ft), (nov, nt) = E.live_pair(3, 10, 1, "archive_top", 15)
    without = V.build_run_figure(grid, fit.end_positions[-1], ft, fit.visited_grid, fit.archive)
    with_archive = V.build_run_figure(grid, nov.end_positions[-1], nt, nov.visited_grid, nov.archive)
    assert not any((t.name or "").startswith("Archiv") for t in without.data)
    arch = next(t for t in with_archive.data if (t.name or "").startswith("Archiv"))
    assert len(arch.x) == len(nov.archive) == 50 and arch.name == "Archiv (50 Punkte)"


def test_run_figure_marks_walls_and_visited_cells_with_different_values():
    grid = M.make_maze(3)
    (fit, ft), _ = E.live_pair(3, 10, 1, "novelty", 15)
    z = np.array(V.build_run_figure(grid, fit.end_positions[-1], ft, fit.visited_grid).data[0].z).T
    assert (z[grid] == 2.0).all() and (z[fit.visited_grid & ~grid] == 1.0).all() and np.isnan(z[~grid & ~fit.visited_grid]).all()


def test_success_chart_has_one_bar_series_per_variant_in_the_variant_colors():
    fig = V.build_success_chart(PRE)
    assert [t.name for t in fig.data] == [C.VARIANT_LABELS[v] for v in C.VARIANTS] and [t.marker.color for t in fig.data] == [V.VARIANT_COLORS[v] for v in C.VARIANTS]
    assert all(len(t.x) == len(C.LEVELS) for t in fig.data)


def test_budget_chart_curves_are_monotone_and_end_at_the_success_share():
    for lv in C.LEVELS:
        fig = V.build_budget_chart(PRE, lv)
        for t, v in zip(fig.data, C.VARIANTS):
            ys = list(t.y)
            assert all(b >= a for a, b in zip(ys, ys[1:])) and ys[-1] == E.success_count(E.study_cell(PRE, lv, v)["runs"]) / PRE["seeds"]


def test_long_chart_uses_the_evaluation_axis_and_marks_the_study_budget():
    fig = V.build_long_chart(PRE)
    assert [t.name for t in fig.data] == [C.VARIANT_LABELS[v] for v in C.LONG_VARIANTS]
    assert fig.data[0].x[0] == C.LONG_STEP * C.POP and fig.data[0].x[-1] == C.LONG_BUDGET
    assert any(s.x0 == PRE["budget"] for s in fig.layout.shapes)


def test_sweep_chart_counts_the_successes_of_every_cell():
    fig = V.build_sweep_chart(PRE, 3)
    z = np.array(fig.data[0].z)
    assert z.shape == (len(C.SWEEP_POPS), len(C.SWEEP_KS))
    assert z[1, 1] == E.success_count(E.sweep_cell(PRE, 3, 15, 100)["runs"])
