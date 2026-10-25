"""C5.1 generator verification (L2): the simulator must behave as specified
before it can be used to judge anything else."""

import numpy as np
import pytest

from aisc.dimensions import NEED_KEYS
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.population import (
    ARCHETYPES,
    SimConfig,
    familiarity_probs,
    generate_population,
)
from aisc.simulation.truth import best_fit_id, relevant_ids, true_utility


@pytest.fixture(scope="module")
def sim_catalog():
    return synthetic_catalog(np.random.default_rng(0), n_per_family=10)


def _pop(sim_catalog, seed=1, **kw):
    return generate_population(sim_catalog, SimConfig(**kw), np.random.default_rng(seed))


# --- catalogue -------------------------------------------------------------------


def test_catalog_shape_and_ranges(sim_catalog):
    assert len(sim_catalog) == 120
    assert len({p.program_id for p in sim_catalog}) == 120
    for p in sim_catalog:
        assert p.synthetic and p.attribute_source == "simulated"
        assert 0 <= p.prestige <= 1
        assert np.all((p.vector >= 0) & (p.vector <= 1))


def test_catalog_is_reproducible():
    a = synthetic_catalog(np.random.default_rng(5), n_per_family=3)
    b = synthetic_catalog(np.random.default_rng(5), n_per_family=3)
    assert [p.attributes for p in a] == [p.attributes for p in b]


def test_within_family_sd_controls_spread():
    tight = synthetic_catalog(np.random.default_rng(1), n_per_family=30, within_sd=0.02)
    loose = synthetic_catalog(np.random.default_rng(1), n_per_family=30, within_sd=0.20)
    fam = lambda cat: np.array([p.vector for p in cat if p.program_id.startswith("sim-00-")])  # noqa: E731
    assert fam(tight).std(axis=0).mean() < fam(loose).std(axis=0).mean()


def test_prestige_is_right_skewed(sim_catalog):
    prestige = np.array([p.prestige for p in sim_catalog])
    assert np.mean(prestige) < 0.5 and np.median(prestige) < 0.5


# --- truth functions ---------------------------------------------------------------


def test_truth_functions():
    w = np.zeros(len(NEED_KEYS))
    x = np.zeros(len(NEED_KEYS))
    w[0], x[0] = 0.2, 1.0
    assert true_utility(w, x, "cosine") == pytest.approx(1.0)
    assert true_utility(w, x, "distance") == pytest.approx(1 - 0.8 / np.sqrt(len(NEED_KEYS)))
    assert true_utility(x, x, "distance") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        true_utility(w, x, "nope")


def test_relevant_and_best_fit(sim_catalog):
    w = sim_catalog[0].vector
    rel = relevant_ids(w, sim_catalog, top_frac=0.1)
    assert len(rel) == 12 and sim_catalog[0].program_id in rel
    assert best_fit_id(w, sim_catalog, exclude=set()) == sim_catalog[0].program_id
    assert best_fit_id(w, sim_catalog, exclude={sim_catalog[0].program_id}) != (
        sim_catalog[0].program_id
    )


# --- population ----------------------------------------------------------------------


def test_population_reproducible_and_valid(sim_catalog):
    a, b = _pop(sim_catalog, seed=3, n_students=50), _pop(sim_catalog, seed=3, n_students=50)
    assert [s.anchors for s in a] == [s.anchors for s in b]
    for s in a:
        assert s.archetype in ARCHETYPES
        assert np.all((s.true_needs >= 0) & (s.true_needs <= 1))
        assert all(0 <= v <= 1 for v in s.stated_needs.values())
        assert len(s.anchors) == len(set(s.anchors)) <= 3


def test_anchors_are_always_familiar(sim_catalog):
    for s in _pop(sim_catalog, n_students=200):
        assert set(s.anchors) <= s.familiar


def test_observed_profile_hides_truth(sim_catalog):
    s = _pop(sim_catalog, n_students=1)[0]
    hidden = s.observed_profile()
    assert hidden.familiar == [] and hidden.needs == s.stated_needs
    assert s.observed_profile(include_familiar=True).familiar == sorted(s.familiar)


def test_familiarity_rises_with_prestige_and_background(sim_catalog):
    cfg = SimConfig()
    probs = familiarity_probs(sim_catalog, "52", cfg)
    prestige = np.array([p.prestige for p in sim_catalog])
    same = np.array([p.cip_family == "52" for p in sim_catalog])
    assert np.corrcoef(prestige[~same], probs[~same])[0, 1] > 0.5
    assert probs[same].mean() > probs[~same].mean()


def test_prestige_weight_shifts_anchor_prestige(sim_catalog):
    """Knob for B-C1-2: high π must produce more prestigious, less fitting anchors."""
    by_id = {p.program_id: p for p in sim_catalog}

    def mean_anchor(pop, attr):
        vals = []
        for s in pop:
            for a in s.anchors:
                vals.append(
                    by_id[a].prestige
                    if attr == "prestige"
                    else true_utility(s.true_needs, by_id[a].vector)
                )
        return float(np.mean(vals))

    low = _pop(sim_catalog, n_students=300, prestige_weight=0.0)
    high = _pop(sim_catalog, n_students=300, prestige_weight=0.9)
    assert mean_anchor(high, "prestige") > mean_anchor(low, "prestige") + 0.05
    assert mean_anchor(high, "fit") < mean_anchor(low, "fit")


def test_background_match_probability(sim_catalog):
    from aisc.simulation.population import NATURAL_FAMILY

    pop = _pop(sim_catalog, n_students=600, background_match_prob=1.0)
    assert all(s.background_family == NATURAL_FAMILY[s.archetype] for s in pop)


def test_response_bias_shifts_stated_needs(sim_catalog):
    pop = _pop(sim_catalog, n_students=800, noise_sd=0.05, acquiescence=0.0)
    q = NEED_KEYS.index("quant_intensity")
    shift = np.mean([s.stated_needs["quant_intensity"] - s.true_needs[q] for s in pop])
    assert 0.05 < shift < 0.12  # desirability +0.10, reduced by clipping at 1


@pytest.mark.parametrize(
    "bad", [{"prestige_weight": 1.5}, {"background_match_prob": -0.1}, {"desirability": {"x": 1}}]
)
def test_config_validation(bad):
    with pytest.raises(ValueError):
        SimConfig(**bad)


# --- W-13 continuous needs model (C5.5) ------------------------------------------------


def test_hexagon_correlation_structure():
    from aisc.simulation.population import riasec_hexagon_corr

    c = riasec_hexagon_corr()
    assert c[0, 1] == c[0, 5] == pytest.approx(0.3)  # R-I, R-C adjacent
    assert c[0, 2] == pytest.approx(0.0)  # R-A alternate
    assert c[0, 3] == pytest.approx(-0.2)  # R-S opposite
    with pytest.raises(ValueError):
        riasec_hexagon_corr(adjacent=0.9, opposite=0.9)  # not positive definite


def test_continuous_model_reproduces_hexagon_ordering(sim_catalog):
    """Adjacent RIASEC types must correlate more than opposite ones (A-C5-6)."""
    from aisc.dimensions import RIASEC_KEYS

    pop = _pop(sim_catalog, n_students=1500, needs_model="continuous")
    w = np.array([s.true_needs for s in pop])
    idx = [NEED_KEYS.index(k) for k in RIASEC_KEYS]
    corr = np.corrcoef(w[:, idx].T)
    adjacent = np.mean([corr[i, (i + 1) % 6] for i in range(6)])
    opposite = np.mean([corr[i, (i + 3) % 6] for i in range(6)])
    assert adjacent > 0.15 and opposite < -0.05


def test_continuous_model_has_no_archetypes_and_valid_range(sim_catalog):
    pop = _pop(sim_catalog, n_students=300, needs_model="continuous")
    assert {s.archetype for s in pop} == {"continuous"}
    w = np.array([s.true_needs for s in pop])
    assert np.all((w > 0) & (w < 1))
    assert 0.3 < w.mean() < 0.5


def test_continuous_model_is_not_clustered(sim_catalog):
    """Archetype needs collapse onto ~7 centres; continuous needs must not."""
    arche = np.array([s.true_needs for s in _pop(sim_catalog, n_students=400)])
    cont = np.array(
        [s.true_needs for s in _pop(sim_catalog, n_students=400, needs_model="continuous")]
    )

    def nearest_neighbour_spread(w):
        d = np.linalg.norm(w[:, None, :] - w[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return d.min(axis=1).mean() / d[np.isfinite(d)].mean()

    assert nearest_neighbour_spread(cont) > nearest_neighbour_spread(arche)


def test_natural_family_follows_strongest_need():
    from aisc.simulation.population import natural_family_from_needs

    w = np.full(len(NEED_KEYS), 0.1)
    w[NEED_KEYS.index("domain_design")] = 0.9
    assert natural_family_from_needs(w) == "50"
    w[NEED_KEYS.index("quant_intensity")] = 0.95
    assert natural_family_from_needs(w) == "27"


def test_unknown_needs_model_rejected():
    with pytest.raises(ValueError):
        SimConfig(needs_model="nope")
