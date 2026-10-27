"""C6 requirement gaps (D9): verification (L2) and the needs/abilities guard.

Requirement texts and URLs below are FICTIONAL fixtures.
"""

from pathlib import Path

import pytest

from aisc.contracts import (
    REQUIREMENT_KINDS,
    Degree,
    RequirementItem,
    StudentRecord,
)
from aisc.requirements import compute_gap, compute_gaps, load_taxonomy

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "data" / "reference" / "requirement_taxonomy.json"
URL, DATE = "https://example.edu/fixture-requirements", "2026-10-27"


def _req(pid, kind, category, level="required", **kw):
    return RequirementItem(
        pid, kind, category, level, f"Fixture text for {category}", URL, DATE, **kw
    )


def test_requirement_item_validation():
    with pytest.raises(ValueError, match="kind"):
        _req("p", "vibes", None)
    with pytest.raises(ValueError, match="level"):
        _req("p", "course", "calculus_1", level="nice to have")
    with pytest.raises(ValueError, match="quote"):
        RequirementItem("p", "course", "calculus_1", "required", "  ", URL, DATE)
    with pytest.raises(ValueError, match="source_url"):
        RequirementItem("p", "course", "calculus_1", "required", "x", "", DATE)


def test_gap_classifies_items():
    items = [
        _req("p", "course", "calculus_1"),
        _req("p", "course", "linear_algebra"),
        _req("p", "course", "programming_intro", level="recommended"),
        _req("p", "test", "gre"),
        _req("p", "other", None),
    ]
    rec = StudentRecord("s", completed={"calculus_1": "微積分（一）"},
                        planned={"linear_algebra": "2027-06"})  # fmt: skip
    gap = compute_gap("p", items, rec, load_taxonomy(TAXONOMY))
    assert [i.category for i in gap.met] == ["calculus_1"]
    assert [i.category for i in gap.planned] == ["linear_algebra"]
    assert [i.category for i in gap.missing] == ["programming_intro", "gre"]
    assert [i.category for i in gap.missing_required] == ["gre"]
    assert len(gap.unknown) == 1 and not gap.none_published
    assert gap.before_target == ()  # exploring: no deadline is assumed (D9)


def test_before_target_only_when_student_sets_a_date():
    items = [
        _req("p", "course", "organic_chemistry"),
        _req("p", "course", "biochemistry", completable_after_admission=True),
        _req("p", "test", "gre", level="recommended"),
    ]
    exploring = compute_gap("p", items, StudentRecord("s"))
    applying = compute_gap("p", items, StudentRecord("s", target_apply_by="2027-12-01"))
    assert exploring.before_target == ()
    assert [i.category for i in applying.before_target] == ["organic_chemistry"]
    # the gap itself is identical in both modes; only the deadline view differs
    assert exploring.missing == applying.missing


def test_every_program_gets_a_gap_and_none_published_is_flagged():
    items = [_req("a", "course", "statistics")]
    gaps = compute_gaps(["a", "b"], items, StudentRecord("s"))
    assert list(gaps) == ["a", "b"]
    assert gaps["b"].none_published and gaps["b"].missing == ()
    with pytest.raises(ValueError, match="not in the list"):
        compute_gaps(["b"], items, StudentRecord("s"))
    with pytest.raises(ValueError, match="another program"):
        compute_gap("b", items, StudentRecord("s"))


def test_career_changer_still_sees_the_program_with_its_gaps():
    """Materials -> dentistry -> design: a design MFA asking for a portfolio and
    drawing foundations is still returned, with exactly what is missing."""
    rec = StudentRecord(
        "s",
        degrees=[
            Degree("bachelor", "Materials Science"),
            Degree("professional", "Dentistry (DDS)"),
        ],
        completed={"general_chemistry": "普通化學", "organic_chemistry": "有機化學",
                   "bachelor_degree": "材料系學士", "english_proficiency": "TOEFL 100"},
    )  # fmt: skip
    items = [
        _req("mfa", "portfolio", "design_portfolio"),
        _req("mfa", "course", "design_foundations", level="recommended"),
        _req("mfa", "credential", "bachelor_degree"),
        _req("mfa", "language", "english_proficiency"),
    ]
    gap = compute_gaps(["mfa"], items, rec, load_taxonomy(TAXONOMY))["mfa"]
    assert {i.category for i in gap.met} == {"bachelor_degree", "english_proficiency"}
    assert {i.category for i in gap.missing} == {"design_portfolio", "design_foundations"}


def test_taxonomy_checks(tmp_path):
    tax = load_taxonomy(TAXONOMY)
    assert set(tax) <= set(REQUIREMENT_KINDS)
    assert "design_portfolio" in tax["portfolio"] and "dat" in tax["test"]
    with pytest.raises(ValueError, match="not a 'test'"):
        compute_gap("p", [_req("p", "test", "calculus_1")], StudentRecord("s"), tax)
    with pytest.raises(ValueError, match="unknown categories"):
        compute_gap("p", [], StudentRecord("s", completed={"astrology": "x"}), tax)
    dup = tmp_path / "t.json"
    dup.write_text('{"categories": {"course": {"x": ""}, "test": {"x": ""}}}')
    with pytest.raises(ValueError, match="appears under"):
        load_taxonomy(dup)


def test_c2_never_reads_student_abilities():
    """D9 guard: discovery (C2) must not depend on StudentRecord or requirement gaps,
    so an unmet requirement can never drop or demote a recommendation."""
    banned = ("StudentRecord", "RequirementGap", "RequirementItem", "aisc.requirements")
    offenders = [
        f"{path.name}: {word}"
        for path in (ROOT / "src" / "aisc" / "discovery").glob("*.py")
        for word in banned
        if word in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders
