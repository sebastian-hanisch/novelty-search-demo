"""AppTest-Rauchtests: Voreinstellung, jedes Preset, alle Stufen und Varianten, Randwerte, lokale Regler, Würfel-Knopf, Permalink-Grenzen, Abschnitte, Footer."""

import random
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ns_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_the_reference_values():
    at = _run()
    _ok(at)
    assert _metric(at, "Kürzester Weg") == "49 Zellen" and _metric(at, "Fitness-Distanz-Korrelation") == "0.69" and _metric(at, "Fallen (lokale Minima)") == "1" and _metric(at, "Freie Zellen") == "465"
    assert _metric(at, "Fitness: Ergebnis") in ("Ziel erreicht", "Ziel verfehlt") and _metric(at, "Novelty: Ergebnis") in ("Ziel erreicht", "Ziel verfehlt")
    assert len(at.get("plotly_chart")) == 9


def test_all_sections_are_present():
    at = _run()
    assert [s.value for s in at.subheader] == ["📐 Ein Lauf: Fitness gegen Novelty", "🔬 Wie oft gelingt es? Die Studie über 20 Läufe", "🔬 Wie viel hängt an k und der Population?",
                                               "🔬 Hilft mehr Budget auf Stufe 4?", "🚧 Wo die Annahmen enden"]
    assert any(m.value.startswith("## 🔗 Wie täuschend ist diese Stufe?") for m in at.markdown)
    assert any("Diese Demo ist Teil des Portfolios" in c.value for c in at.caption)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert (at.session_state["level_select"], at.session_state["gens_slider"], at.session_state["variant_select"], at.session_state["k_slider"]) == (p["level"], p["gens"], p["variant"], p["k"])
    assert at.metric


@pytest.mark.parametrize("level", list(C.LEVELS))
def test_every_level_runs(level):
    at = _run(level_select=level, gens_slider=10)
    _ok(at)
    assert _metric(at, "Fitness: besuchte Zellen").endswith(f"von {at.metric[3].value}")


@pytest.mark.parametrize("variant", C.NOVELTY_VARIANTS)
def test_every_novelty_variant_runs(variant):
    at = _run(variant_select=variant, gens_slider=20)
    _ok(at)
    assert len(at.get("plotly_chart")) == 9 and _metric(at, "Novelty: besuchte Zellen")


def test_free_field_both_searches_reach_the_goal_and_the_text_says_that_novelty_has_no_advantage():
    at = _run(level_select=0)
    _ok(at)
    assert _metric(at, "Fitness: Ergebnis") == "Ziel erreicht" and _metric(at, "Novelty: Ergebnis") == "Ziel erreicht" and any("keinen Vorteil" in i.value for i in at.info)


def test_the_serpentine_fails_for_both_searches():
    at = _run(level_select=4, gens_slider=100)
    _ok(at)
    assert _metric(at, "Fallen (lokale Minima)") == "2" and _metric(at, "Kürzester Weg") == "77 Zellen"
    assert _metric(at, "Fitness: Ergebnis") == "Ziel verfehlt" and _metric(at, "Novelty: Ergebnis") == "Ziel verfehlt"


def test_the_solvability_line_names_the_steps_used():
    at = _run(level_select=4)
    assert any("136 von 140 Schritten benutzt" in c.value for c in at.caption)


@pytest.mark.parametrize("kw", [dict(gens_slider=10), dict(gens_slider=200), dict(k_slider=1), dict(k_slider=30), dict(level_select=3, gens_slider=200, seed_input=0),
                                dict(level_select=1, gens_slider=10, seed_input=999999, k_slider=1, variant_select="archive_threshold")])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_the_study_table_follows_the_level_regulator():
    at = _run()
    table = next(m.value for m in at.markdown if m.value.startswith("| Variante | Erfolg |"))
    assert table.count("\n| ") == len(C.VARIANTS)
    four = _run(study_level_select=4)
    _ok(four)
    assert next(m.value for m in four.markdown if m.value.startswith("| Variante | Erfolg |")).count("0 von 20") == len(C.VARIANTS)


def test_the_sweep_regulator_switches_the_level():
    at = _run(sweep_level_select=4)
    _ok(at)
    assert any("keiner" in i.value and "0 Erfolge in 180 Läufen" in i.value for i in at.info)


def test_dice_button_changes_the_seed_and_the_run(monkeypatch):
    """Der Würfel zieht sonst einen unseeded Zufalls-Seed; deshalb ist der gewürfelte Seed im Test fest."""
    monkeypatch.setattr(random, "randint", lambda a, b: 508145)
    at = _run(level_select=2)
    old_seed = at.session_state["seed_input"]
    old = (_metric(at, "Fitness: Evaluationen bis zum Erfolg"), _metric(at, "Novelty: Evaluationen bis zum Erfolg"))
    next(b for b in at.button if b.label == "🎲 Neuen Lauf würfeln").click().run()
    _ok(at)
    assert at.session_state["seed_input"] == 508145 != old_seed
    assert (_metric(at, "Fitness: Evaluationen bis zum Erfolg"), _metric(at, "Novelty: Evaluationen bis zum Erfolg")) != old


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["level"] = "9"
    at.query_params["gens"] = "104"
    at.query_params["k"] = "99"
    at.query_params["variant"] = "archive_threshold"
    at.run()
    _ok(at)
    assert at.session_state["level_select"] == 4 and at.session_state["gens_slider"] == 100 and at.session_state["k_slider"] == 30 and at.session_state["variant_select"] == "archive_threshold"


def test_permalink_ignores_garbage():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["level"] = "viele"
    at.query_params["seed"] = "nan"
    at.query_params["variant"] = "fitness"
    at.query_params["k"] = "0"
    at.run()
    _ok(at)
    assert at.session_state["level_select"] == C.DEFAULT_LEVEL and at.session_state["seed_input"] == C.DEFAULT_SEED and at.session_state["variant_select"] == C.DEFAULT_VARIANT
    assert at.session_state["k_slider"] == C.K_MIN                                            # 0 liegt unter der Grenze und wird auf 1 gehoben


def test_no_sentence_wide_comma_replacement_in_the_app_source():
    """Regressionsschutz: `.replace(",", ".")` auf einem ganzen (verketteten) Satz macht aus Kommas im Fließtext Punkte; Tausender nur über `fmt_int`."""
    source = Path(APP).read_text(encoding="utf-8")
    assert '.replace(",", ".")' not in source
