"""Regression tests for defects found by the 2026-10-28 code audit (B-C1-25,
B-C2-15..17, B-C4-7, B-C5-10..12); they started as xfail probes."""

import math

import numpy as np
import pytest

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


# --- third audit (2026-10-28): C4.2 / C2 diversity --------------------------------


def test_cip_family_handles_unpadded_codes():
    from dataclasses import replace

    from aisc.discovery.sectioned import cip_family
    from aisc.simulation.catalog_gen import synthetic_catalog

    p = synthetic_catalog(np.random.default_rng(0), n_per_family=1)[0]
    assert cip_family(replace(p, cip="9.0101")) == "09.01"
    assert cip_family(replace(p, cip="52.1301")) == "52.13"


def test_recommend_portfolio_validates_rho_with_draws():
    import aisc.portfolio as pf

    p, u = [0.3, 0.6], [1.0, 0.5]
    d = pf.admission_draws(p, 0.5, 100, np.random.default_rng(0))
    assert _raises(lambda: pf.recommend_portfolio(p, u, [1, 1], 1, rho=-0.5, draws=d))


# --- fifth audit (2026-10-28): the default recommendation path ---------------------


def _probe(reason):
    """Probe marker: strict xfail that only accepts an AssertionError."""

    def mark(fn):
        return pytest.mark.probe(pytest.mark.xfail(strict=True, raises=AssertionError,
                                                   reason=reason)(fn))  # fmt: skip

    return mark


def _synthetic_catalog():
    from pathlib import Path

    from aisc.catalog import load_programs

    return load_programs(Path(__file__).resolve().parents[1] / "data/synthetic/programs.jsonl")


def test_blend_rejects_unknown_keys_and_out_of_range():
    from aisc.profiling import blend_profile_v1

    by_id = {p.program_id: p for p in _synthetic_catalog()}
    typo = StudentProfile("s", {"quant_intensty": 0.9}, anchors=["syn-01"])
    high = StudentProfile("s", {"quant_intensity": 1.7}, anchors=["syn-01"])
    assert _raises(lambda: blend_profile_v1(typo, by_id))
    assert _raises(lambda: blend_profile_v1(high, by_id))


def test_answered_dimensions_are_more_certain_than_unanswered():
    from aisc.pipeline import prepare_profile

    cat = _synthetic_catalog()
    answered = {"quant_intensity": 0.8, "domain_finance": 0.7}
    for anchors in ([], ["syn-01"]):
        full = prepare_profile(StudentProfile("s", answered, anchors=anchors), cat)
        sd_ans = max(full.uncertainty.get(k, float("inf")) for k in answered)
        sd_un = min(v for k, v in full.uncertainty.items() if k not in answered)
        assert sd_ans < sd_un


def test_questionnaire_tracks_only_recommendable_programmes():
    from aisc.discovery.constraints import passes_constraints
    from aisc.pipeline import c2_scores, prepare_profile, run_questionnaire
    from aisc.profiling import AdaptiveSession

    cat = _synthetic_catalog()
    prof = StudentProfile("s", {}, anchors=["syn-01", "syn-12"],
                          hard_constraints={"max_tuition_usd": 60000})  # fmt: skip

    def answer(item):
        return 0.7

    got = run_questionnaire(prof, cat, answer, budget=6, rng=np.random.default_rng(0))
    pool = [p for p in cat if p.program_id not in prof.anchors
            and passes_constraints(p, prof.hard_constraints)]  # fmt: skip
    s = AdaptiveSession(prior=prepare_profile(prof, cat), catalog=pool, score_fn=c2_scores,
                        rng=np.random.default_rng(0))  # fmt: skip
    for _ in range(6):
        item = s.next_item()
        if item is None:
            break
        s.record(item, answer(item))
    assert got.needs == s.profile().needs


def test_duplicate_anchors_count_once():
    from aisc.profiling import blend_profile_v1

    by_id = {p.program_id: p for p in _synthetic_catalog()}
    once = blend_profile_v1(StudentProfile("s", {}, anchors=["syn-01"]), by_id)
    twice = blend_profile_v1(StudentProfile("s", {}, anchors=["syn-01", "syn-01"]), by_id)
    assert once.needs == twice.needs and once.uncertainty == twice.uncertainty


def test_questionnaire_checks_catalogue():
    from dataclasses import replace

    from aisc.pipeline import run_questionnaire

    cat = _synthetic_catalog()
    prof = StudentProfile("s", {"quant_intensity": 0.8})

    def message(fn):
        try:
            fn()
        except ValueError as e:
            return str(e)
        return ""

    assert "empty" in message(lambda: run_questionnaire(prof, [], lambda i: 0.5, budget=2))
    v2 = [replace(cat[0], dims="v2", attributes={})]
    assert "dimension" in message(lambda: run_questionnaire(prof, v2, lambda i: 0.5, budget=2))


def test_ceec_rejects_nan():
    from aisc.profiling.ceec import riasec_prior_from_ceec

    scores = dict.fromkeys("RIASEC", 50)
    assert _raises(lambda: riasec_prior_from_ceec(scores, float("nan")))
    assert _raises(lambda: riasec_prior_from_ceec(scores, 1.0, differentiation=float("nan")))


def test_constraints_reject_unknown_keys_and_types():
    from aisc.discovery.constraints import passes_constraints

    p = _synthetic_catalog()[0]
    for bad in ({"max_duration_months": 12}, {"max_tuition_usd": "50000"},
                {"max_tuition_usd": float("nan")}, {"max_tuition_usd": True},
                {"stem_opt": "yes"}):  # fmt: skip
        assert _raises(lambda b=bad: passes_constraints(p, b))


def test_duplicate_programme_ids_are_rejected():
    from dataclasses import replace

    from aisc.discovery import shortlist
    from aisc.profiling import complete_neutral

    cat = _synthetic_catalog()
    dup = [*cat, replace(cat[2])]
    prof = complete_neutral(StudentProfile("s", {"quant_intensity": 0.8}))
    assert _raises(lambda: shortlist(prof, dup, n=30))
