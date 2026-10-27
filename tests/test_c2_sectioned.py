"""C2.3 v1 sectioned discovery and the system pipeline (L2)."""

import pytest

from aisc.contracts import StudentProfile
from aisc.discovery import discover, relevance, shortlist
from aisc.pipeline import interactive_discovery
from aisc.profiling import complete_neutral

NEEDS = {"quant_intensity": 0.8, "domain_finance": 0.9, "riasec_e": 0.6}


def full(**kwargs) -> StudentProfile:
    """Explicitly completed profile: C2 refuses incomplete ones (B-C1-14)."""
    return complete_neutral(StudentProfile("s", NEEDS, **kwargs))


def test_shortlist_is_fit_ranked_and_excludes_anchors(catalog):
    profile = full(anchors=["syn-01"])
    ids = shortlist(profile, catalog, n=5)
    assert len(ids) == 5 and "syn-01" not in ids
    by_id = {p.program_id: p for p in catalog}
    rels = [relevance(profile.vector, by_id[i].vector) for i in ids]
    assert rels == sorted(rels, reverse=True)


def test_discover_partitions_by_familiarity(catalog):
    profile = full(anchors=["syn-01"], familiar=["syn-02", "syn-12"])
    res = discover(profile, catalog, k_new=4, k_known=3)
    new = [r.program.program_id for r in res.discoveries]
    known = [r.program.program_id for r in res.known_good]
    assert set(known) <= {"syn-02", "syn-12"} and known
    assert not set(new) & {"syn-01", "syn-02", "syn-12"}
    assert len(new) == 4
    assert all(r.novelty == 1.0 for r in res.discoveries)
    assert all(r.novelty == 0.0 for r in res.known_good)
    for section in (res.discoveries, res.known_good):
        scores = [r.relevance for r in section]
        assert scores == sorted(scores, reverse=True)
        # contributions are the scorer's exact additive explanation (B-C2-6)
        assert all(sum(r.contributions.values()) == pytest.approx(r.relevance) for r in section)


def test_discover_respects_constraints_and_floor(catalog):
    profile = full(hard_constraints={"spring_intake": True})
    res = discover(profile, catalog, k_new=12, min_relevance=0.6)
    assert all(r.program.spring_intake is not False for r in res.discoveries)
    assert all(r.relevance >= 0.6 for r in res.discoveries)


def test_discover_validates_sizes(catalog):
    with pytest.raises(ValueError):
        discover(full(), catalog, k_new=-1)


def test_pipeline_asks_shortlist_once_and_uses_answers(catalog):
    calls = []

    def ask(ids):
        calls.append(list(ids))
        return {i: ("heard" if i == ids[0] else "unknown") for i in ids}

    profile = StudentProfile("s", NEEDS, anchors=["syn-01"])
    res = interactive_discovery(profile, catalog, ask, n_shortlist=6, k_new=3, k_known=2)
    assert len(calls) == 1 and len(calls[0]) == 6
    top = calls[0][0]
    assert top in [r.program.program_id for r in res.known_good]
    assert top not in [r.program.program_id for r in res.discoveries]
    assert res.asked == calls[0]


def test_pipeline_rejects_answers_to_unasked_programs(catalog):
    profile = StudentProfile("s", NEEDS)
    with pytest.raises(ValueError):
        interactive_discovery(profile, catalog, lambda ids: {"not-asked": "heard"}, n_shortlist=3)


def test_pipeline_completes_unanswered_dimensions(catalog):
    """B-C1-14: an unanswered dimension must not be treated as 0 ("no interest")."""
    from aisc.dimensions import NEED_KEYS
    from aisc.pipeline import prepare_profile

    partial = StudentProfile("s", {"quant_intensity": 0.9}, {"quant_intensity": 0.2})
    out = prepare_profile(partial, catalog)
    assert set(out.needs) == set(NEED_KEYS)
    assert out.needs["quant_intensity"] == pytest.approx(0.9)  # answered: unchanged
    assert all(v > 0 for k, v in out.needs.items() if k != "quant_intensity")


def test_pipeline_leaves_fully_answered_profiles_unchanged(catalog):
    from aisc.dimensions import NEED_KEYS
    from aisc.pipeline import prepare_profile

    full = StudentProfile("s", {k: 0.3 for k in NEED_KEYS}, anchors=["syn-01"])
    assert prepare_profile(full, catalog).needs == pytest.approx(full.needs)


@pytest.mark.parametrize("entry", ["shortlist", "discover", "recommend"])
def test_c2_refuses_incomplete_profiles(catalog, entry):
    """B-C1-14 enforced at the C2 boundary, not only inside the pipeline."""
    from aisc.discovery import recommend

    fn = {"shortlist": shortlist, "discover": discover, "recommend": recommend}[entry]
    with pytest.raises(ValueError, match="unanswered dimensions"):
        fn(StudentProfile("s", NEEDS), catalog)
    fn(full(), catalog)  # completed profile is accepted


def test_missing_dimensions_reported():
    from aisc.dimensions import NEED_KEYS

    p = StudentProfile("s", {"quant_intensity": 0.5})
    assert len(p.missing_dimensions) == len(NEED_KEYS) - 1
    assert complete_neutral(p).missing_dimensions == []


def test_explicit_scorer_changes_ranking_and_explanations(catalog):
    from aisc.discovery import COSINE, layered

    profile = full(anchors=["syn-01"])
    a = discover(profile, catalog, k_new=12, scorer=COSINE)
    b = discover(profile, catalog, k_new=12, scorer=layered(2.0))
    for res in (a, b):
        assert all(sum(r.contributions.values()) == pytest.approx(r.relevance)
                   for r in res.discoveries)  # fmt: skip
    assert [r.relevance for r in a.discoveries] != [r.relevance for r in b.discoveries]
    assert shortlist(profile, catalog, n=5, scorer=COSINE) == [
        r.program.program_id for r in a.discoveries[:5]
    ]


def test_discoveries_can_be_diversified_by_cip_family():
    """B-C2-9 (was a probe). 30 programmes per family and a student sitting on one
    programme: the pure-fit discoveries come from at most two families (first written
    as "one family"; the nearest neighbouring family also enters, so the premise was
    relaxed). With family diversity (C2.3, A-C2-15) a strong enough penalty must bring
    in other families, and diversity 0 must reproduce the pure-fit list exactly."""
    import inspect

    import numpy as np

    from aisc.simulation.catalog_gen import synthetic_catalog

    assert "diversity" in inspect.signature(discover).parameters
    assert "diversity" in inspect.signature(shortlist).parameters
    cat = synthetic_catalog(np.random.default_rng(0), n_per_family=30)
    profile = StudentProfile("p", dict(cat[0].attributes))
    plain = discover(profile, cat, k_new=10)
    fams = {r.program.cip[:5] for r in plain.discoveries}
    assert len(fams) <= 2
    same = discover(profile, cat, k_new=10, diversity=0.0)
    assert [r.program.program_id for r in same.discoveries] == [
        r.program.program_id for r in plain.discoveries
    ]
    spread = discover(profile, cat, k_new=10, diversity=1.0)
    assert len({r.program.cip[:5] for r in spread.discoveries}) >= 5
    assert shortlist(profile, cat, n=10, diversity=0.0) == shortlist(profile, cat, n=10)


def test_diversity_validates_and_keeps_exact_explanations():
    import numpy as np

    from aisc.simulation.catalog_gen import synthetic_catalog

    cat = synthetic_catalog(np.random.default_rng(1), n_per_family=5)
    profile = StudentProfile("p", dict(cat[3].attributes))
    with pytest.raises(ValueError):
        discover(profile, cat, diversity=-0.1)
    with pytest.raises(ValueError):
        shortlist(profile, cat, diversity=-0.1)
    res = discover(profile, cat, k_new=8, diversity=0.05)
    for r in res.discoveries:
        assert sum(r.contributions.values()) == pytest.approx(r.relevance)
