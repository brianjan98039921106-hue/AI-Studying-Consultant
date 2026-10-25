"""C2.3 v1 sectioned discovery and the system pipeline (L2)."""

import pytest

from aisc.contracts import StudentProfile
from aisc.discovery import discover, relevance, shortlist
from aisc.pipeline import interactive_discovery

NEEDS = {"quant_intensity": 0.8, "domain_finance": 0.9, "riasec_e": 0.6}


def test_shortlist_is_fit_ranked_and_excludes_anchors(catalog):
    profile = StudentProfile("s", NEEDS, anchors=["syn-01"])
    ids = shortlist(profile, catalog, n=5)
    assert len(ids) == 5 and "syn-01" not in ids
    by_id = {p.program_id: p for p in catalog}
    rels = [relevance(profile.vector, by_id[i].vector) for i in ids]
    assert rels == sorted(rels, reverse=True)


def test_discover_partitions_by_familiarity(catalog):
    profile = StudentProfile("s", NEEDS, anchors=["syn-01"], familiar=["syn-02", "syn-12"])
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
        assert all(sum(r.contributions.values()) == pytest.approx(1.0) for r in section)


def test_discover_respects_constraints_and_floor(catalog):
    profile = StudentProfile("s", NEEDS, hard_constraints={"spring_intake": True})
    res = discover(profile, catalog, k_new=12, min_relevance=0.6)
    assert all(r.program.spring_intake is not False for r in res.discoveries)
    assert all(r.relevance >= 0.6 for r in res.discoveries)


def test_discover_validates_sizes(catalog):
    with pytest.raises(ValueError):
        discover(StudentProfile("s", NEEDS), catalog, k_new=-1)


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
