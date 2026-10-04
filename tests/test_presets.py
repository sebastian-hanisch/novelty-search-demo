"""Presets: Vollständigkeit, gültige Werte, Permalink-Konstanten, Formatierer."""

import pytest

import ns_constants as C
import ns_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER) and len(C.PRESETS) == 4
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for preset in C.PRESETS.values():
        assert preset["level"] in C.LEVELS and C.GENS_MIN <= preset["gens"] <= C.GENS_MAX and P.snap_gens(preset["gens"]) == preset["gens"]
        assert preset["variant"] in C.NOVELTY_VARIANTS and C.K_MIN <= preset["k"] <= C.K_MAX
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(preset[key])


def test_preset_names_state_the_values_they_set():
    assert C.PRESETS["Freies Feld"]["level"] == 0 and C.PRESETS["Falle: Fitness gegen Novelty"]["level"] == 3
    assert C.PRESETS["Schlangenlinie"]["level"] == 4 and C.PRESETS["Mit Archiv"]["variant"] == "archive_top"


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Falle: Fitness gegen Novelty"]
    assert (p["level"], p["gens"], p["seed"], p["variant"], p["k"]) == (C.DEFAULT_LEVEL, C.DEFAULT_GENS, C.DEFAULT_SEED, C.DEFAULT_VARIANT, C.K_DEFAULT)


def test_bounds_and_url_params():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("gens_slider") == (10, 200) and P.bounds("level_select") == (0, 4) and P.bounds("k_slider") == (1, 30)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS) == 5


@pytest.mark.parametrize("value,expected", [(0, 10), (14, 10), (16, 20), (104, 100), (106, 110), (999, 200)])
def test_generations_snap_to_the_step_from_the_lower_bound(value, expected):
    assert P.snap_gens(value) == expected


@pytest.mark.parametrize("value,expected", [(0, 0), (1, 1), (3, 3), (4, 4), (2, 2)])
def test_levels_are_taken_over_unchanged(value, expected):
    assert P.snap_to_option("level_select", value) == expected


@pytest.mark.parametrize("value", C.NOVELTY_VARIANTS)
def test_variant_caster_accepts_every_novelty_variant(value):
    assert P.SETTING_SPECS["variant_select"].caster(value) == value


@pytest.mark.parametrize("value", ["fitness", "zufall", "", 3, None, ["novelty"]])
def test_variant_caster_refuses_everything_else(value):
    """Die Fitness ist im Live-Lauf kein wählbarer Wert (sie läuft immer daneben)."""
    with pytest.raises(ValueError):
        P.SETTING_SPECS["variant_select"].caster(value)


def test_the_variant_lists_are_consistent():
    assert C.VARIANTS[0] == C.FITNESS and C.NOVELTY_VARIANTS == C.VARIANTS[1:] and set(C.VARIANT_LABELS) == set(C.VARIANTS) and set(C.LONG_VARIANTS) <= set(C.VARIANTS)


def test_formatters():
    assert C.fmt_int(40000) == "40.000" and C.fmt_pct(0.55) == "55 %" and C.fmt_pct(0.0898, 1) == "9.0 %"
