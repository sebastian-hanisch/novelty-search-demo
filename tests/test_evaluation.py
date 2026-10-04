"""Auswertung: Täuschungsmaße je Stufe, Lösbarkeit, Live-Paar, Hilfsfunktionen der Studie, Vollständigkeit und Stimmigkeit der vorgerechneten Datei (inklusive Neurechnung einzelner Läufe)."""

import numpy as np
import pytest

import ns_constants as C
import ns_evaluation as E
import ns_maze as M
import ns_search as S

PRE = E.load_precomputed()


def test_level_metrics_by_hand():
    m0, m3 = E.level_metrics(0), E.level_metrics(3)
    assert m0["path"] == 17 and m0["straight"] == pytest.approx(17.0) and m0["detour"] == pytest.approx(1.0) and m0["minima"] == 0 and m0["trap"] is None and m0["free"] == 22 * 22
    assert m3["path"] == 49 and m3["detour"] == pytest.approx(49 / 17) and m3["minima"] == 1 and m3["trap"] == pytest.approx(9.0) and m3["free"] == 22 * 22 - 19


@pytest.mark.parametrize("level", list(C.LEVELS))
def test_every_level_is_solvable_within_the_step_budget(level):
    dist, steps = E.solvability(level)
    assert dist < 0.5 and steps <= C.STEPS


def test_live_pair_runs_the_fitness_and_the_chosen_variant_from_the_same_start():
    (fit, ft), (nov, nt) = E.live_pair(2, 12, 5, "novelty", 15)
    assert len(fit.best_distance) == len(nov.best_distance) == 12 and ft.shape == nt.shape == (C.STEPS + 1, 2) and tuple(ft[0]) == tuple(nt[0]) == C.START
    assert (fit.end_positions[0] == nov.end_positions[0]).all()                                       # gleiche Anfangspopulation
    assert np.hypot(*(nt[-1] - np.array(C.GOAL))) == pytest.approx(min(nov.best_distance)) and np.hypot(*(ft[-1] - np.array(C.GOAL))) == pytest.approx(min(fit.best_distance))


def test_success_helpers_by_hand():
    runs = [100, None, 500, 50, None]
    assert E.success_count(runs) == 3 and E.success_count(runs, 100) == 2 and E.success_count(runs, 10) == 0
    assert E.success_curve(runs, [0, 60, 100, 1000]) == [0.0, 0.2, 0.4, 0.6]
    assert E.median_solved(runs) == 100.0 and E.median_solved([None, None]) is None and E.median_solved([10, 20]) == 15.0


def test_the_study_covers_every_level_and_every_variant():
    assert set(PRE["levels"]) == {str(lv) for lv in C.LEVELS} and PRE["seeds"] == C.STUDY_SEEDS and PRE["budget"] == C.STUDY_BUDGET
    for lv in C.LEVELS:
        assert set(PRE["levels"][str(lv)]["variants"]) == set(C.VARIANTS)
        for v in C.VARIANTS:
            c = E.study_cell(PRE, lv, v)
            assert len(c["runs"]) == len(c["cells"]) == len(c["progress"]) == C.STUDY_SEEDS and len(c["mean_visited"]) == len(c["mean_progress"]) == C.STUDY_GENS
            assert all(r is None or 1 <= r <= C.STUDY_BUDGET for r in c["runs"]) and max(c["cells"]) <= E.level_info(PRE, lv)["free"]


def test_the_stored_level_metrics_equal_the_exact_ones():
    for lv in C.LEVELS:
        stored, fresh = E.level_info(PRE, lv), E.level_metrics(lv)
        for key in ("path", "straight", "detour", "fdc", "minima", "free"):
            assert stored[key] == pytest.approx(fresh[key])
        assert stored["trap"] == (pytest.approx(fresh["trap"]) if fresh["trap"] is not None else None)


@pytest.mark.parametrize("level,variant,seed", [(0, "fitness", 3), (2, "novelty", 4), (3, "archive_top", 0), (3, "archive_threshold", 7)])
def test_a_study_run_is_reproduced_by_a_fresh_run_with_the_same_seed(level, variant, seed):
    """Gespeicherte Zahl der Evaluationen bis zum Erfolg, besuchte Zellen und Restweg stimmen mit einem frischen Lauf überein."""
    res = S.run_search(M.make_maze(level), seed, variant, gens=C.STUDY_GENS, keep_positions=False)
    c = E.study_cell(PRE, level, variant)
    assert res.first_solved == c["runs"][seed] and res.visited_cells[-1] == c["cells"][seed] and res.best_progress[-1] == c["progress"][seed]


def test_the_mean_curves_start_within_the_population_and_never_get_worse():
    """Besuchte Zellen: nach der ersten Generation höchstens so viele wie Individuen, dann nie weniger; Restweg nie größer (beides sind laufende Bestwerte)."""
    c = E.study_cell(PRE, 3, "novelty")
    assert 1 <= c["mean_visited"][0] <= C.POP and all(b >= a for a, b in zip(c["mean_visited"], c["mean_visited"][1:])) and all(b <= a + 1e-9 for a, b in zip(c["mean_progress"], c["mean_progress"][1:]))


def test_the_sweep_uses_the_same_total_budget_for_every_population_size():
    assert all(C.STUDY_BUDGET % p == 0 for p in C.SWEEP_POPS) and {C.STUDY_BUDGET // p for p in C.SWEEP_POPS} == {800, 400, 200}
    assert len(PRE["sweep"]) == len(C.SWEEP_LEVELS) * len(C.SWEEP_KS) * len(C.SWEEP_POPS) == 18
    for x in PRE["sweep"]:
        assert len(x["runs"]) == len(x["progress"]) == len(x["cells"]) == C.STUDY_SEEDS and E.sweep_cell(PRE, x["level"], x["k"], x["pop"]) is x
        assert all(r is None or r <= C.STUDY_BUDGET for r in x["runs"])


def test_the_default_setting_in_the_sweep_equals_the_level_study():
    """Standardeinstellung (k = 15, 100 Individuen) im Raster: dieselben Läufe wie die Variante „nur Population“ in der Stufenstudie."""
    for lv in C.SWEEP_LEVELS:
        assert E.sweep_cell(PRE, lv, C.K_DEFAULT, C.POP)["runs"] == E.study_cell(PRE, lv, "novelty")["runs"]
        assert E.sweep_cell(PRE, lv, C.K_DEFAULT, C.POP)["progress"] == E.study_cell(PRE, lv, "novelty")["progress"]


def test_the_long_runs_cover_the_listed_variants_and_start_like_the_study():
    long = PRE["long"]
    assert long["budget"] == C.LONG_BUDGET and set(long["variants"]) == set(C.LONG_VARIANTS)
    for v in C.LONG_VARIANTS:
        c = E.long_cell(PRE, v)
        assert len(c["runs"]) == C.STUDY_SEEDS and len(c["mean_progress"]) == len(c["mean_visited"]) == C.LONG_GENS // C.LONG_STEP
        idx = C.STUDY_GENS // C.LONG_STEP - 1                                                          # Generation 400: dasselbe wie das Ende der Studie
        study = E.study_cell(PRE, 4, v)
        assert c["mean_progress"][idx] == pytest.approx(study["mean_progress"][-1], abs=1e-3) and c["mean_visited"][idx] == pytest.approx(study["mean_visited"][-1], abs=1e-3)


def test_a_long_run_is_reproduced_by_a_fresh_run_for_one_seed():
    """Fitness, Stufe 4, Seed 1 mit 4 000 Generationen: gleiche Endwerte wie gespeichert (rund 40 s)."""
    res = S.run_search(M.make_maze(4), 1, "fitness", gens=C.LONG_GENS, keep_positions=False)
    c = E.long_cell(PRE, "fitness")
    assert res.first_solved == c["runs"][1] and res.visited_cells[-1] == c["cells"][1] and res.best_progress[-1] == c["progress"][1]
