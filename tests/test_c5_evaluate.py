"""C5.3 runner verification (L2): the harness must score known recommenders correctly."""

import numpy as np
import pytest

from aisc.contracts import StudentProfile
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.evaluate import (
    c2_pipeline,
    evaluate_lists,
    evaluate_profiles,
    popular_baseline,
    similar_baseline,
)
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.truth import utilities


@pytest.fixture(scope="module")
def setup():
    cat = synthetic_catalog(np.random.default_rng(0), n_per_family=5)
    pop = generate_population(cat, SimConfig(n_students=40), np.random.default_rng(2))
    return cat, pop


def test_popular_baseline_orders_by_prestige(setup):
    cat, _ = setup
    top = max(cat, key=lambda p: p.prestige)
    ids = popular_baseline(StudentProfile("x", {}), cat, 3)
    assert ids[0] == top.program_id
    assert top.program_id not in popular_baseline(StudentProfile("x", {}, anchors=[ids[0]]), cat, 3)


def test_similar_baseline_prefers_anchor_label(setup):
    cat, _ = setup
    anchor = cat[0]
    ids = similar_baseline(StudentProfile("x", {}, anchors=[anchor.program_id]), cat, 4)
    assert anchor.program_id not in ids
    assert all(next(p for p in cat if p.program_id == i).cip_4 == anchor.cip_4 for i in ids)


def test_oracle_recommender_scores_perfectly(setup):
    """A recommender that sees true needs must get relevant_rate = 1 and keep the best fit."""
    cat, pop = setup
    truth_by_student = {s.student_id: s.true_needs for s in pop}

    def oracle(profile, catalog, k):
        u = utilities(truth_by_student[profile.student_id], catalog)
        order = [catalog[i].program_id for i in np.argsort(-u)]
        return [pid for pid in order if pid not in set(profile.anchors)][:k]

    k = 3  # 10% of 60 programs = 6 relevant; anchors can occupy up to 3 of them
    res = evaluate_lists(pop, cat, {"oracle": oracle}, k=k)["oracle"]
    assert res["relevant_rate"][0] == pytest.approx(1.0)
    assert res["best_fit_retained"][0] == pytest.approx(1.0)


def test_evaluate_lists_reports_relevance_with_novelty(setup):
    cat, pop = setup
    res = evaluate_lists(pop, cat, {"c2": c2_pipeline(), "pop": popular_baseline}, k=5)
    for summary in res.values():
        assert {"relevant_rate", "novel_relevant_rate", "best_fit_retained"} <= set(summary)
        assert summary["novel_relevant_rate"][0] <= summary["relevant_rate"][0] + 1e-12
        assert 0 <= summary["coverage"][0] <= 1


def test_familiarity_flag_is_per_recommender(setup):
    cat, pop = setup
    res = evaluate_lists(
        pop,
        cat,
        {"blind": c2_pipeline(), "informed": c2_pipeline()},
        k=5,
        include_familiar={"informed": True},
    )
    assert res["informed"]["unfamiliar_rate"][0] >= res["blind"]["unfamiliar_rate"][0]


def test_evaluate_profiles_keys(setup):
    cat, pop = setup
    res = evaluate_profiles(pop[:10], cat)
    assert set(res) == {"stated_only", "anchor_blend"}
    assert all(0 <= v["rmse"][0] <= 1 for v in res.values())
