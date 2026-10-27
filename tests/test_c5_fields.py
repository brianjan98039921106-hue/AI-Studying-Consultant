"""C5 all-fields extension (W-63, W-67): verification (L2) and the B-C2-8 probe."""

import numpy as np
import pytest

from aisc.contracts import Program, StudentProfile
from aisc.dimensions import NEED_KEYS
from aisc.discovery import shortlist
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.evaluate import c2_pipeline, evaluate_lists_per_student
from aisc.simulation.fields import (
    FAR_PROTOTYPES,
    HIDDEN_BASELINE,
    HIDDEN_KEYS,
    MACRO_GROUPS,
    TRANSITIONS,
    all_fields_catalog,
    macro_group,
    pick_backgrounds,
)
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.truth import best_fit_id, relevant_ids, utilities


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    biz = synthetic_catalog(rng, n_per_family=3)
    cat, hidden = all_fields_catalog(rng, biz, n_per_family=3)
    return cat, hidden


def test_catalog_has_far_fields_and_hidden_vectors(world):
    cat, hidden = world
    assert len(cat) == 12 * 3 + len(FAR_PROTOTYPES) * 3
    assert set(hidden) == {p.program_id for p in cat}
    assert all(h.shape == (len(HIDDEN_KEYS),) and np.all((h >= 0) & (h <= 1))
               for h in hidden.values())  # fmt: skip
    biz = [hidden[p.program_id] for p in cat if not p.program_id.startswith("far-")]
    assert np.mean(biz) < 0.1
    dent = [hidden[p.program_id] for p in cat if p.cip == "51.0401"]
    assert np.mean([h[HIDDEN_KEYS.index("h_health")] for h in dent]) > 0.7
    # components only ever see the 15 measured dimensions
    assert all(set(p.attributes) == set(NEED_KEYS) for p in cat)


def test_macro_groups_partition_families():
    seen = [f for fams in MACRO_GROUPS.values() for f in fams]
    assert len(seen) == len(set(seen))
    assert macro_group("51") == "health_life" and macro_group("52") == "business_data"
    with pytest.raises(ValueError):
        macro_group("99")


@pytest.mark.parametrize("transition", TRANSITIONS)
def test_pick_backgrounds_respects_transition(transition):
    fams = sorted(set().union(*MACRO_GROUPS.values()))
    rng = np.random.default_rng(3)
    for natural in ("52", "51", "23", "14"):
        for _ in range(20):
            bg = pick_backgrounds(transition, natural, fams, rng)
            g = macro_group(natural)
            if transition == "continuing":
                assert bg == (natural,)
            elif transition == "adjacent":
                assert len(bg) == 1 and macro_group(bg[0]) == g
            else:
                assert all(macro_group(b) != g for b in bg)
                assert len(bg) == (2 if transition == "multi_degree" else 1)
    with pytest.raises(ValueError):
        pick_backgrounds("teleport", "52", fams, rng)


def test_v1_population_is_unchanged_by_the_extension(world):
    cat = synthetic_catalog(np.random.default_rng(0), n_per_family=3)
    pop = generate_population(cat, SimConfig(n_students=20), np.random.default_rng(1))
    assert all(s.transition == "" and s.hidden_needs is None for s in pop)
    assert {s.background_stratum for s in pop} <= {"continuing", "switching"}


@pytest.mark.parametrize("needs_model", ["archetype", "continuous"])
def test_all_fields_population(world, needs_model):
    cat, hidden = world
    cfg = SimConfig(n_students=300, needs_model=needs_model, field_model="all_fields")
    with pytest.raises(ValueError, match="hidden"):
        generate_population(cat, cfg, np.random.default_rng(2))
    pop = generate_population(cat, cfg, np.random.default_rng(2), hidden=hidden)
    assert {s.transition for s in pop} == set(TRANSITIONS)
    for s in pop:
        assert s.hidden_needs is not None and s.hidden_needs.shape == (len(HIDDEN_KEYS),)
        assert set(s.stated_needs) == set(NEED_KEYS)  # hidden domains cannot be stated
        assert s.background_stratum == s.transition
        assert s.background_family == s.background_families[-1]
        if s.transition == "multi_degree":
            assert len(s.background_families) == 2
    far_needs = [s for s in pop if s.natural_family in {"51", "26", "23", "13", "22"}]
    assert len(far_needs) > 30  # far-field students exist in both needs models


def test_config_validation():
    with pytest.raises(ValueError):
        SimConfig(field_model="galaxy")
    with pytest.raises(ValueError):
        SimConfig(transition_mix={"teleport": 1.0})
    with pytest.raises(ValueError):
        SimConfig(transition_mix={"far": -1.0, "continuing": 2.0})


def test_truth_with_hidden_domains(world):
    cat, hidden = world
    w = np.full(len(NEED_KEYS), 0.5)
    wh = np.full(len(HIDDEN_KEYS), HIDDEN_BASELINE)
    u_measured = utilities(w, cat)
    u_full = utilities(w, cat, hidden=hidden, w_hidden=wh)
    far = [i for i, p in enumerate(cat) if p.program_id.startswith("far-")]
    biz = [i for i in range(len(cat)) if i not in far]
    delta = u_full - u_measured
    assert delta[far].mean() < delta[biz].mean()  # hidden mismatch lowers far-field fit
    with pytest.raises(ValueError, match="both"):
        utilities(w, cat, hidden=hidden)
    assert relevant_ids(w, cat, hidden=hidden, w_hidden=wh)
    assert best_fit_id(w, cat, set(), hidden=hidden, w_hidden=wh) is not None


def test_hidden_mismatch_rate_only_with_hidden(world):
    cat, hidden = world
    cfg = SimConfig(n_students=40, field_model="all_fields")
    pop = generate_population(cat, cfg, np.random.default_rng(5), hidden=hidden)
    with_h = evaluate_lists_per_student(pop, cat, {"c2": c2_pipeline()}, k=5, hidden=hidden)["c2"]
    assert len(with_h["hidden_mismatch_rate"]) == len(pop)
    assert all(0 <= v <= 1 for v in with_h["hidden_mismatch_rate"])
    without = evaluate_lists_per_student(pop, cat, {"c2": c2_pipeline()}, k=5)["c2"]
    assert "hidden_mismatch_rate" not in without


@pytest.mark.probe
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="B-C2-8: a far-field program that matches on measured dimensions outranks the true fit",
)
def test_probe_far_field_trap_is_avoided():
    """A quantitative researcher (no health or life-science interest) should get the
    statistics program first, not a biology program that happens to sit closer on the
    15 measured dimensions but is wrong on the hidden domains."""
    w = np.array([0.2, 0.9, 0.2, 0.1, 0.1, 0.5, 0.9, 0.6, 0.2, 0.3, 0.2, 0.0, 0.2, 0.9, 0.0])
    stats_x = np.clip(w + np.r_[[0.1] * 3, [0.0] * 12], 0, 1)
    progs = [
        Program("stats", "S", "MS Statistics", "27.0501",
                dict(zip(NEED_KEYS, map(float, stats_x), strict=True)), synthetic=True),
        Program("bio", "B", "MS Biology", "26.0101",
                dict(zip(NEED_KEYS, map(float, w), strict=True)), synthetic=True),
    ]  # fmt: skip
    hidden = {"stats": np.full(6, HIDDEN_BASELINE),
              "bio": np.array([.4, .95, HIDDEN_BASELINE, HIDDEN_BASELINE, HIDDEN_BASELINE,
                               HIDDEN_BASELINE])}  # fmt: skip
    truth_best = best_fit_id(w, progs, set(), hidden=hidden, w_hidden=np.full(6, HIDDEN_BASELINE))
    assert truth_best == "stats"  # sanity: the full truth prefers statistics
    profile = StudentProfile("s", dict(zip(NEED_KEYS, map(float, w), strict=True)))
    assert shortlist(profile, progs, n=1) == [truth_best]
