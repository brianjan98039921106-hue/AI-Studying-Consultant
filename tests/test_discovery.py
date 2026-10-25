"""Behavioural tests for profiling (L1) and discovery (L2) on synthetic data."""

from pathlib import Path

import pytest

from aisc.discovery import label_similarity, novelty, recommend, relevance
from aisc.needs import NEED_KEYS
from aisc.profiling import blend_profile, needs_from_anchors
from aisc.schema import StudentProfile, load_programs

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "programs.jsonl"


@pytest.fixture(scope="module")
def catalog():
    return load_programs(DATA)


@pytest.fixture(scope="module")
def by_id(catalog):
    return {p.program_id: p for p in catalog}


def test_catalog_loads_and_is_marked_synthetic(catalog):
    assert len(catalog) == 12
    assert all(p.synthetic for p in catalog)


def test_label_similarity_levels(by_id):
    assert label_similarity(by_id["syn-03"], by_id["syn-11"]) == 1.0  # both 14.35
    assert label_similarity(by_id["syn-01"], by_id["syn-04"]) == 0.5  # both 52.xx
    assert label_similarity(by_id["syn-01"], by_id["syn-05"]) == 0.0


def test_novelty_without_anchors_is_one(by_id):
    assert novelty(by_id["syn-01"], []) == 1.0


def test_relevance_is_cosine():
    import numpy as np

    v = np.ones(len(NEED_KEYS))
    assert relevance(v, v) == pytest.approx(1.0)
    assert relevance(v, np.zeros(len(NEED_KEYS))) == 0.0


def test_finance_student_discovers_non_business_programs(by_id, catalog):
    """A student anchored on MS Finance should be shown quantitative programs
    outside CIP 52 (e.g. Financial Mathematics, Operations Research)."""
    student = StudentProfile(
        student_id="t1",
        needs={"quant": 0.8, "finance": 0.9, "business": 0.6, "programming": 0.5},
        anchors=["syn-01"],
    )
    recs = recommend(blend_profile(student, by_id), catalog, k=5, novelty_weight=0.4)
    ids = [r.program.program_id for r in recs]
    assert "syn-01" not in ids  # anchors are never re-recommended
    top3_families = {r.program.cip_family for r in recs[:3]}
    assert top3_families - {"52"}, "top results should include a non-business family"
    assert {"syn-02", "syn-03"} & set(ids)


def test_pure_fit_ignores_novelty(by_id, catalog):
    student = StudentProfile(student_id="t2", needs=dict(by_id["syn-04"].attributes))
    recs = recommend(student, catalog, k=1, novelty_weight=0.0, diversity_weight=0.0)
    assert recs[0].program.program_id == "syn-04"


def test_hard_constraints_filter_spring_intake(catalog):
    student = StudentProfile(
        student_id="t3",
        needs={"quant": 0.8, "programming": 0.6},
        hard_constraints={"spring_intake": True},
    )
    recs = recommend(student, catalog, k=12)
    assert recs and all(r.program.spring_intake is not False for r in recs)


def test_anchor_blend_moves_toward_anchors(by_id):
    student = StudentProfile(student_id="t4", needs={"design": 0.0}, anchors=["syn-08"])
    blended = blend_profile(student, by_id)
    implied = needs_from_anchors([by_id["syn-08"]])
    design_idx = NEED_KEYS.index("design")
    assert 0.0 < blended.needs["design"] <= implied[design_idx]
    assert all(blended.uncertainty[k] < 0.5 for k in NEED_KEYS)


def test_invalid_need_key_rejected():
    with pytest.raises(ValueError):
        _ = StudentProfile(student_id="bad", needs={"not_a_need": 0.5}).vector
