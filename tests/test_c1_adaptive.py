"""C1.3 adaptive questionnaire (L2) and calibration probe (L4)."""

import numpy as np
import pytest

from aisc.contracts import Program, StudentProfile
from aisc.dimensions import NEED_KEYS
from aisc.discovery import relevance, relevance_matrix
from aisc.profiling import AdaptiveSession, Item, complete_neutral, default_item_bank

PRIOR = complete_neutral(StudentProfile("s", {}))


def _toy_catalog(varying: str) -> list[Program]:
    """10 programs identical except on one dimension."""
    progs = []
    for i in range(10):
        attrs = {k: 0.5 for k in NEED_KEYS}
        attrs[varying] = i / 9
        progs.append(Program(f"p{i}", "S", "MS", "27.0501", attrs, synthetic=True))
    return progs


def test_item_bank_size_and_validation():
    assert len(default_item_bank()) == 45
    assert len(default_item_bank(1)) == 15
    with pytest.raises(ValueError):
        default_item_bank(0)


def test_session_validation(catalog):
    with pytest.raises(ValueError, match="cover every dimension"):
        AdaptiveSession(StudentProfile("s", {"quant_intensity": 0.5}), catalog)
    with pytest.raises(ValueError):
        AdaptiveSession(PRIOR, catalog, strategy="psychic")
    with pytest.raises(ValueError):
        AdaptiveSession(PRIOR, catalog, item_noise_sd=0)


def test_record_is_precision_weighted_update(catalog):
    s = AdaptiveSession(PRIOR, catalog, item_noise_sd=0.3)  # prior sd 0.3 -> equal weights
    item = Item("quant_intensity#0", "quant_intensity")
    s.record(item, 0.9)
    d = NEED_KEYS.index("quant_intensity")
    assert s.mean[d] == pytest.approx((0.5 + 0.9) / 2)
    assert s.var[d] == pytest.approx(0.09 / 2)
    with pytest.raises(ValueError):
        s.record(item, 0.5)  # already asked
    with pytest.raises(ValueError):
        s.record(Item("quant_intensity#1", "quant_intensity"), 1.5)


def test_variance_strategy_covers_every_dimension_first(catalog):
    s = AdaptiveSession(PRIOR, catalog, strategy="variance")
    dims = []
    for _ in range(15):
        item = s.next_item()
        dims.append(item.dimension)
        s.record(item, 0.5)
    assert sorted(dims) == sorted(NEED_KEYS)


def test_session_exhausts_bank(catalog):
    s = AdaptiveSession(PRIOR, catalog, strategy="random", rng=np.random.default_rng(3))
    for _ in range(45):
        s.record(s.next_item(), 0.5)
    assert s.done() and s.next_item() is None
    assert all(v < 0.3**2 for v in s.var)


@pytest.mark.parametrize("strategy", ["discriminative", "voi"])
def test_ranking_aware_strategies_target_the_dimension_that_reorders(strategy):
    """Only domain_finance differs between programs: that is the only question
    whose answer can change the ranking."""
    cat = _toy_catalog("domain_finance")
    s = AdaptiveSession(PRIOR, cat, strategy=strategy, top_k=3, top_m=10)
    assert s.next_item().dimension == "domain_finance"


def test_voi_ignores_dimensions_that_cannot_change_top_k():
    cat = _toy_catalog("domain_finance")
    s = AdaptiveSession(PRIOR, cat, strategy="voi", top_k=3)
    for item in list(s.remaining["domain_finance"]):
        s.record(item, 0.9)
    # domain_finance exhausted; remaining dims are identical across programs -> VOI 0;
    # ties broken by variance, all equal -> first available dimension
    assert s.next_item().dimension == NEED_KEYS[0]


def test_profile_reports_posterior(catalog):
    s = AdaptiveSession(PRIOR, catalog)
    s.record(s.next_item(), 0.8)
    p = s.profile()
    assert set(p.needs) == set(NEED_KEYS)
    assert min(p.uncertainty.values()) < 0.3 == pytest.approx(max(p.uncertainty.values()))


def test_relevance_matrix_matches_scalar_relevance(catalog):
    """Guard: C1.3 optimises for C2's scorer only if these stay identical (W-26)."""
    X = np.array([p.vector for p in catalog])
    w = np.random.default_rng(0).random(len(NEED_KEYS))
    assert relevance_matrix(w, X) == pytest.approx([relevance(w, x) for x in X])
    assert relevance_matrix(np.zeros(len(NEED_KEYS)), X) == pytest.approx(np.zeros(len(X)))


def test_pipeline_questionnaire_respects_budget(catalog):
    from aisc.pipeline import prepare_profile, run_questionnaire

    asked = []

    def answer(item):
        asked.append(item)
        return 0.7

    profile = StudentProfile("s", {}, anchors=["syn-01"])
    out = run_questionnaire(profile, catalog, answer, budget=6)
    assert len(asked) == 6 and len({i.item_id for i in asked}) == 6
    assert set(out.needs) == set(NEED_KEYS)
    zero = run_questionnaire(profile, catalog, answer, budget=0)
    assert zero.needs == pytest.approx(prepare_profile(profile, catalog).needs)
    with pytest.raises(ValueError):
        run_questionnaire(profile, catalog, answer, budget=-1)


# --- calibration ------------------------------------------------------------------------


def _coverage(
    true_noise: float,
    n_students: int = 150,
    *,
    assumed: float = 0.25,
    needs_model: str = "continuous",
    noise_model: str = "fixed",
    edge_only: bool = False,
) -> float:
    """Share of dimensions whose 90% posterior interval contains the true need,
    after all 45 items. `edge_only` restricts to true needs < 0.15 or > 0.85."""
    from aisc.simulation.catalog_gen import synthetic_catalog
    from aisc.simulation.population import SimConfig, generate_population
    from aisc.simulation.respondent import make_item_responder

    cat = synthetic_catalog(np.random.default_rng(0), n_per_family=2)
    pop = generate_population(
        cat, SimConfig(n_students=n_students, needs_model=needs_model), np.random.default_rng(4)
    )
    hits = total = 0
    for i, st in enumerate(pop):
        answer = make_item_responder(st, np.random.default_rng([2, i]), noise_sd=true_noise)
        s = AdaptiveSession(
            PRIOR, cat, strategy="variance", item_noise_sd=assumed, noise_model=noise_model
        )
        while not s.done():
            item = s.next_item()
            s.record(item, answer(item))
        lo, hi = s.interval(0.90)
        keep = (st.true_needs < 0.15) | (st.true_needs > 0.85) if edge_only else slice(None)
        hits += int(np.sum(((lo <= st.true_needs) & (st.true_needs <= hi))[keep]))
        total += int(np.size(st.true_needs[keep]))
    return hits / total


def test_posterior_is_calibrated_when_noise_is_correctly_specified():
    assert 0.85 <= _coverage(true_noise=0.25) <= 0.96


@pytest.mark.probe
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="B-C1-5: with underestimated item noise the posterior is overconfident",
)
def test_probe_posterior_calibrated_under_misspecified_noise():
    """True item noise 0.40, assumed 0.25: 90% intervals should still cover >= 85%."""
    assert _coverage(true_noise=0.40) >= 0.85


@pytest.mark.probe
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="B-C1-16: Gaussian likelihood ignores censoring at the scale ends",
)
def test_probe_edge_dimensions_calibrated_under_censoring():
    """Noise correctly assumed (0.40); dimensions with true need near 0 or 1 should
    still have >= 85% coverage of their 90% intervals (archetype students)."""
    cov = _coverage(0.40, assumed=0.40, needs_model="archetype", edge_only=True)
    assert cov >= 0.85
