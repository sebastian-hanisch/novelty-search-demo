"""JEDE Zahl aus README und App-Texten wird hier nachgerechnet: Täuschungsmaße exakt aus dem Labyrinth, Studienzahlen aus der vorgerechneten Datei (20 Läufe je Zelle, 40 000 Evaluationen, Seeds 0 bis 19; verlängerter Lauf
mit 400 000 Evaluationen). Die Datei ist fest; ändern sich die Zahlen nach einer neuen Rechnung, müssen README und Hilfetexte nachgezogen werden. Längen in Zellen, Aufwand in Evaluationen."""

import numpy as np
import pytest

import ns_constants as C
import ns_evaluation as E
import ns_maze as M
import ns_search as S

PRE = E.load_precomputed()
V = C.VARIANTS


def cell(level, variant):
    return E.study_cell(PRE, level, variant)


def runs(level, variant):
    return cell(level, variant)["runs"]


def mean(level, variant, key):
    return float(np.mean(cell(level, variant)[key]))


def test_exact_deception_measures_of_the_levels_as_in_the_first_piece():
    """README: kürzester Weg 17 / 17 / 33 / 49 / 77 Zellen, Umwegfaktor 1.00 / 1.00 / 1.94 / 2.88 / 4.53, Korrelation 0.97 / 0.97 / 0.89 / 0.69 / 0.68, Fallen 0 / 0 / 1 / 1 / 2, freie Zellen 484 / 465 / 465 / 465 / 427."""
    m = [E.level_metrics(lv) for lv in C.LEVELS]
    assert [x["path"] for x in m] == [17, 17, 33, 49, 77] and [round(x["detour"], 2) for x in m] == [1.00, 1.00, 1.94, 2.88, 4.53]
    assert [round(x["fdc"], 2) for x in m] == [0.97, 0.97, 0.89, 0.69, 0.68] and [x["minima"] for x in m] == [0, 0, 1, 1, 2] and [x["free"] for x in m] == [484, 465, 465, 465, 427]


def test_the_fitness_reference_equals_the_first_piece():
    """README: die Fitness-Suche löst 20 / 20 / 20 / 11 / 0 von 20 Läufen, Median 63.5 / 126 / 190.5 / 1 267 Evaluationen (Stück 1: 64 / 126 / 190 / 1 267, dort auf ganze Zahlen gerundet); schlimmster Lauf auf Stufe 3: 3 355."""
    assert [E.success_count(runs(lv, "fitness")) for lv in C.LEVELS] == [20, 20, 20, 11, 0]
    assert [E.median_solved(runs(lv, "fitness")) for lv in C.LEVELS[:4]] == [63.5, 126.0, 190.5, 1267.0]
    assert max(r for r in runs(3, "fitness") if r is not None) == 3355


def test_novelty_without_archive_solves_the_trap_in_every_run():
    """README: Novelty ohne Archiv löst 20 / 20 / 20 / 20 / 0 von 20 Läufen, Median 63.5 / 146 / 207.5 / 988.5; auf Stufe 3 der schlimmste Lauf 3 373, der beste 77 (wie bei der Fitness)."""
    assert [E.success_count(runs(lv, "novelty")) for lv in C.LEVELS] == [20, 20, 20, 20, 0]
    assert [E.median_solved(runs(lv, "novelty")) for lv in C.LEVELS[:4]] == [63.5, 146.0, 207.5, 988.5]
    ok = [r for r in runs(3, "novelty") if r is not None]
    assert max(ok) == 3373 and min(ok) == 77 and min(r for r in runs(3, "fitness") if r is not None) == 77


def test_novelty_is_not_faster_where_the_fitness_does_not_deceive():
    """README: auf Stufe 1 und 2 braucht Novelty im Median mehr (146 gegen 126, 207.5 gegen 190.5), auf Stufe 0 gleich viel (63.5); der schlimmste Lauf auf Stufe 1 / 2: 307 / 447 gegen 252 / 875."""
    assert E.median_solved(runs(1, "novelty")) > E.median_solved(runs(1, "fitness")) and E.median_solved(runs(2, "novelty")) > E.median_solved(runs(2, "fitness"))
    assert E.median_solved(runs(0, "novelty")) == E.median_solved(runs(0, "fitness")) == 63.5
    assert [max(runs(lv, v)) for lv, v in ((1, "novelty"), (2, "novelty"), (1, "fitness"), (2, "fitness"))] == [307, 447, 252, 875]


def test_the_medians_on_level_three_compare_different_samples():
    """README (Vorbehalt): der Median der Fitness (1 267) gilt nur für die 11 erfolgreichen Läufe, der von Novelty (988.5) für alle 20; die schnellsten Läufe sind bei beiden 77."""
    assert E.success_count(runs(3, "fitness")) == 11 and E.success_count(runs(3, "novelty")) == 20
    assert sum(r is None for r in runs(3, "fitness")) == 9


def test_the_archive_changes_nothing_on_the_success_rate():
    """README: beide Archiv-Varianten lösen Stufe 0 bis 3 in 20 von 20 Läufen, Median 63.5 / 146 / 207.5 / 1 195 (beide gleich); schlimmster Lauf auf Stufe 3: 4 006 (Neuartigste), 3 764 (Schwelle), 3 373 (ohne Archiv)."""
    for v in ("archive_top", "archive_threshold"):
        assert [E.success_count(runs(lv, v)) for lv in C.LEVELS] == [20, 20, 20, 20, 0]
        assert [E.median_solved(runs(lv, v)) for lv in C.LEVELS[:4]] == [63.5, 146.0, 207.5, 1195.0]
    assert max(runs(3, "archive_top")) == 4006 and max(runs(3, "archive_threshold")) == 3764


def test_the_two_archive_rules_start_identically():
    """README: bis der Schwellenwert greift, nehmen beide Regeln dieselben 5 neuartigsten Individuen je Generation auf; Stufe 0 bis 2 ergeben deshalb für beide dieselben Mediane und dieselbe Abdeckung am Anfang."""
    for lv in (0, 1, 2):
        assert E.median_solved(runs(lv, "archive_top")) == E.median_solved(runs(lv, "archive_threshold"))
    assert cell(3, "archive_top")["mean_visited"][0] == cell(3, "archive_threshold")["mean_visited"][0] == cell(3, "novelty")["mean_visited"][0]
    grid = M.make_maze(3)
    a = S.run_search(grid, 0, "archive_top", gens=5, keep_positions=False)
    b = S.run_search(grid, 0, "archive_threshold", gens=5, keep_positions=False)
    assert (a.archive == b.archive).all() and len(a.archive) == 25


def test_novelty_visits_every_free_cell_in_every_run_on_levels_zero_to_three():
    """README: auf Stufe 0 bis 3 besuchen alle 60 Novelty-Läufe (je Variante) alle freien Zellen (484 / 465 / 465 / 465); die Fitness im Mittel 327.15 / 376.95 / 422.45 / 343.9, im schlechtesten Lauf auf Stufe 3 nur 191."""
    for lv in (0, 1, 2, 3):
        free = E.level_info(PRE, lv)["free"]
        for v in C.NOVELTY_VARIANTS:
            assert cell(lv, v)["cells"] == [free] * C.STUDY_SEEDS
    assert [round(mean(lv, "fitness", "cells"), 2) for lv in (0, 1, 2, 3)] == [327.15, 376.95, 422.45, 343.9]
    assert min(cell(3, "fitness")["cells"]) == 191


def test_novelty_spreads_faster_than_the_fitness_in_the_first_generations():
    """README: nach 50 Generationen auf Stufe 3 sind im Mittel 445.5 von 465 Zellen besucht (Novelty ohne Archiv) gegen 272.6 (Fitness); nach 100 Generationen 463.15 gegen 306.95."""
    assert cell(3, "novelty")["mean_visited"][49] == pytest.approx(445.5, abs=1e-6) and cell(3, "fitness")["mean_visited"][49] == pytest.approx(272.6, abs=1e-6)
    assert cell(3, "novelty")["mean_visited"][99] == pytest.approx(463.15, abs=1e-6) and cell(3, "fitness")["mean_visited"][99] == pytest.approx(306.95, abs=1e-6)


def test_level_three_fitness_runs_stay_in_front_of_the_wall():
    """README: der kleinste wahre Restweg der Fitness-Läufe auf Stufe 3 liegt im Mittel bei 5.5 Zellen (schlechtester Lauf 17), bei Novelty bei 0."""
    assert round(mean(3, "fitness", "progress"), 2) == 5.55 and max(cell(3, "fitness")["progress"]) == 17 and mean(3, "novelty", "progress") == 0


def test_level_four_is_solved_by_no_variant_but_novelty_comes_closer():
    """README: Stufe 4, 40 000 Evaluationen: 0 / 0 / 0 / 0 von 20 Läufen; besuchte Zellen 223.8 / 264.15 / 257.45 / 255.35 von 427; kleinster wahrer Restweg 34.75 / 26.25 / 29.95 / 30.3 Zellen (bester Lauf 22 / 22 / 26 / 26)."""
    assert [E.success_count(runs(4, v)) for v in V] == [0, 0, 0, 0]
    assert [round(mean(4, v, "cells"), 2) for v in V] == [223.8, 264.15, 257.45, 255.35]
    assert [round(mean(4, v, "progress"), 2) for v in V] == [34.75, 26.25, 29.95, 30.3]
    assert [min(cell(4, v)["progress"]) for v in V] == [22, 22, 26, 26]
    assert E.level_info(PRE, 4)["path"] == 77 and E.level_info(PRE, 4)["free"] == 427


def test_settings_grid_on_level_three():
    """README: alle 9 Einstellungen (k = 5 / 15 / 30 × 50 / 100 / 200 Individuen) lösen Stufe 3 in 20 von 20 Läufen; Median 718.5 / 1 186.5 / 1 175 (k = 5), 659 / 988.5 / 1 308.5 (k = 15), 710 / 763.5 / 1 271.5 (k = 30)."""
    grid = [[E.success_count(E.sweep_cell(PRE, 3, k, p)["runs"]) for p in C.SWEEP_POPS] for k in C.SWEEP_KS]
    assert grid == [[20, 20, 20]] * 3
    meds = [[E.median_solved(E.sweep_cell(PRE, 3, k, p)["runs"]) for p in C.SWEEP_POPS] for k in C.SWEEP_KS]
    assert meds == [[718.5, 1186.5, 1175.0], [659.0, 988.5, 1308.5], [710.0, 763.5, 1271.5]]


def test_settings_grid_on_level_four_has_no_success_at_all():
    """README: Stufe 4: 0 Erfolge in allen 9 Einstellungen, also 0 von 180 Läufen; kleinster wahrer Restweg im Mittel zwischen 25.9 und 27.6 Zellen."""
    cells = [x for x in PRE["sweep"] if x["level"] == 4]
    assert len(cells) == 9 and all(E.success_count(x["runs"]) == 0 for x in cells) and sum(len(x["runs"]) for x in cells) == 180
    means = [float(np.mean(x["progress"])) for x in cells]
    assert round(min(means), 1) == 25.9 and round(max(means), 1) == 27.6


def test_ten_times_the_budget_on_level_four():
    """README: mit 400 000 Evaluationen 0 von 60 Läufen (Fitness, Novelty ohne Archiv, Archiv der Neuartigsten); mittlerer Restweg nach 40 000 / 400 000: 34.8 / 32.3 (Fitness), 26.3 / 22.4 (Novelty), 30.0 / 23.8 (Archiv);
    bester Lauf am Ende 20 / 16 / 21; besuchte Zellen 238.8 / 288.2 / 286.4 von 427."""
    idx40 = C.STUDY_BUDGET // (C.LONG_STEP * C.POP) - 1
    assert sum(E.success_count(E.long_cell(PRE, v)["runs"]) for v in C.LONG_VARIANTS) == 0 and len(C.LONG_VARIANTS) * C.STUDY_SEEDS == 60
    at40 = [round(E.long_cell(PRE, v)["mean_progress"][idx40], 2) for v in C.LONG_VARIANTS]
    end = [round(float(np.mean(E.long_cell(PRE, v)["progress"])), 2) for v in C.LONG_VARIANTS]
    assert at40 == [34.75, 26.25, 29.95] and end == [32.3, 22.4, 23.75]
    assert [min(E.long_cell(PRE, v)["progress"]) for v in C.LONG_VARIANTS] == [20, 16, 21]
    assert [round(float(np.mean(E.long_cell(PRE, v)["cells"])), 1) for v in C.LONG_VARIANTS] == [238.8, 288.2, 286.4]


def test_the_long_runs_flatten_out():
    """README: der mittlere Restweg von Novelty fällt von 26.3 (40 000) über 24.8 (100 000) und 23.9 (200 000) auf 22.4 (400 000); die Fitness von 34.8 über 33.7 und 33.0 auf 32.3."""
    n, f = E.long_cell(PRE, "novelty")["mean_progress"], E.long_cell(PRE, "fitness")["mean_progress"]
    assert [round(n[i], 2) for i in (99, 199, 399)] == [24.75, 23.85, 22.4] and [round(f[i], 2) for i in (99, 199, 399)] == [33.65, 32.95, 32.3]


def test_preset_help_quotes_the_study():
    """PRESET_HELP: Falle (20 gegen 11 von 20, Median 988.5, 465 Zellen gegen 343.9), Freies Feld (20 von 20, Median 63.5, 484 gegen 327.15), Schlangenlinie (0 von 20, Restweg 26.25 gegen 34.75 von 77),
    Mit Archiv (20 von 20, Median 1 195 gegen 988.5)."""
    assert E.success_count(runs(3, "novelty")) == 20 and E.success_count(runs(3, "fitness")) == 11 and E.median_solved(runs(3, "novelty")) == 988.5
    assert cell(3, "novelty")["cells"][0] == 465 and round(mean(3, "fitness", "cells"), 2) == 343.9
    assert E.success_count(runs(0, "fitness")) == E.success_count(runs(0, "novelty")) == 20 and cell(0, "novelty")["cells"][0] == 484 and round(mean(0, "fitness", "cells"), 2) == 327.15
    assert E.success_count(runs(4, "novelty")) == 0 and round(mean(4, "novelty", "progress"), 2) == 26.25 and round(mean(4, "fitness", "progress"), 2) == 34.75
    assert E.success_count(runs(3, "archive_top")) == 20 and E.median_solved(runs(3, "archive_top")) == 1195.0
    assert C.STUDY_BUDGET == 40_000
    for name, text in C.PRESET_HELP.items():
        assert text and name in C.PRESETS
