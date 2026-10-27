"""Regression tests for defects found by the 2026-10-28 code audit (B-C1-25,
B-C2-15..17, B-C4-7, B-C5-10..12); they started as xfail probes."""

import math

import numpy as np

from aisc.contracts import StudentProfile, to_vector
from aisc.dimensions import LAYERS, NEED_KEYS


def _raises(fn, exc=ValueError) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def test_nan_scores_are_rejected():
    assert _raises(lambda: to_vector({"riasec_r": float("nan")}))


def test_non_positive_uncertainty_is_rejected():
    assert _raises(lambda: StudentProfile("s", {"riasec_r": 0.5}, {"riasec_r": 0.0}))


def test_layer_balanced_rejects_layered_base_up_front():
    from aisc.discovery import layer_balanced, layered

    assert _raises(lambda: layer_balanced({lay: 1.0 for lay in LAYERS}, base=layered(0.5)))


def test_worst_case_ignores_nan_regardless_of_order():
    from aisc.simulation.strata import maximin, worst_case

    a = worst_case({"m": {"a": math.nan, "b": 0.1}})["m"]
    b = worst_case({"m": {"b": 0.1, "a": math.nan}})["m"]
    assert a == b == 0.1
    assert maximin({"m1": {"a": math.nan, "b": 0.1}, "m2": {"a": 0.2, "b": 0.3}}) == "m2"


def test_dp_accepts_whole_number_float_costs():
    from aisc.portfolio.solvers import dp_varied_cost

    ok = not _raises(lambda: dp_varied_cost([0.5, 0.5], [1, 1], [1.0, 2.0], 3), TypeError)
    assert ok and dp_varied_cost([0.5, 0.5], [1, 1], [1.0, 2.0], 3) == dp_varied_cost(
        [0.5, 0.5], [1, 1], [1, 2], 3
    )


def test_corrected_estimate_does_not_report_naive_betas():
    from aisc.discovery.layer_weights import _pair_rows, estimate_layer_weights_simex

    rng = np.random.default_rng(0)
    w = rng.random((40, len(NEED_KEYS)))
    X = rng.random((10, len(NEED_KEYS)))
    pairs = np.array([(r, a, (a + 1) % 10) for r in range(40) for a in range(5)])
    G = _pair_rows(w, X, pairs, "v1")
    y = (rng.random(len(G)) < 1 / (1 + np.exp(-(G @ np.full(4, 30.0))))).astype(float)
    est = estimate_layer_weights_simex(w, X, pairs, y, 0.1, "v1", n_sim=2, n_boot=0, rng=rng)
    beta = est.corrected.beta
    assert not beta or all(
        math.isclose(beta[k] / sum(beta.values()), est.corrected.shares[k]) for k in LAYERS
    )


def test_all_fields_evaluation_requires_hidden():
    from aisc.simulation.catalog_gen import synthetic_catalog
    from aisc.simulation.evaluate import evaluate_lists_per_student, popular_baseline
    from aisc.simulation.fields import all_fields_catalog
    from aisc.simulation.population import SimConfig, generate_population

    rng = np.random.default_rng(1)
    src, hidden = all_fields_catalog(rng, synthetic_catalog(rng, n_per_family=2), 2)
    pop = generate_population(src, SimConfig(n_students=5, field_model="all_fields"), rng,
                              hidden=hidden)  # fmt: skip
    assert _raises(lambda: evaluate_lists_per_student(pop, src, {"pop": popular_baseline}))


def test_adjacent_fallback_is_relabelled():
    from aisc.simulation import fields

    assert hasattr(fields, "realised_transition")
    assert fields.realised_transition("adjacent", "f1", ("f1",)) == "continuing"
