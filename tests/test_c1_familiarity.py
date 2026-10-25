"""C1.4 familiarity elicitation (L2)."""

import pytest

from aisc.contracts import StudentProfile
from aisc.profiling import STRICT_KNOWN, apply_familiarity, known_ids

REPORT = {"a": "heard", "b": "can_describe", "c": "unknown"}


def test_default_policy_counts_heard_and_describe_as_known():
    assert known_ids(REPORT) == {"a", "b"}


def test_strict_policy_counts_only_describe():
    assert known_ids(REPORT, STRICT_KNOWN) == {"b"}


def test_apply_familiarity_updates_and_latest_answer_wins():
    profile = StudentProfile("s", {"quant_intensity": 0.5}, anchors=["x"], familiar=["c", "z"])
    out = apply_familiarity(profile, REPORT)
    assert out.familiar == ["a", "b", "z"]  # c removed (now "unknown"), z kept (not asked)
    assert out.anchors == ["x"] and out.needs == profile.needs
    assert profile.familiar == ["c", "z"]  # input not mutated


@pytest.mark.parametrize("bad", [{"a": "maybe"}, {"a": ""}])
def test_unknown_level_rejected(bad):
    with pytest.raises(ValueError):
        known_ids(bad)


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        known_ids(REPORT, frozenset({"sort_of"}))
