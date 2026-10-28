"""W-77 real programme catalogue v0: IPEDS school x CIP joined with C0.2 profiles."""

import numpy as np
import pytest

from aisc.catalog import load_programs
from aisc.catalog.real import (
    build_real_catalog_v0,
    load_real_catalog_v0,
    unmeasured_keys,
    write_jsonl,
)
from aisc.contracts import StudentProfile
from aisc.dimensions import dims_keys
from aisc.discovery.scorers import DEFAULT_SCORER_V2
from aisc.pipeline import interactive_discovery, prepare_profile
from aisc.report import render_profile_report

KEYS = dims_keys("v2")
GAP = unmeasured_keys()
MEASURED = [k for k in KEYS if k not in GAP]


@pytest.fixture(scope="module")
def real():
    return load_real_catalog_v0()


def _prof(v):
    return {k: v for k in MEASURED}


def _row(unitid, cip, level="7", awards=20, name="U"):
    return {"unitid": unitid, "institution": name, "cip": cip, "award_level": level,
            "awards": str(awards)}  # fmt: skip


def test_builder_filters_levels_small_rows_and_uses_fallback():
    rows = [
        _row("1", "11.0701"),
        _row("2", "11.0701", awards=3),  # below min_awards
        _row("3", "11.0701", level="17"),  # research doctorate -> academic table
        _row("4", "52.1399"),  # no profile -> fallback
        _row("5", "24.0101"),  # no profile, no fallback -> skipped
        _row("6", "11.0701", level="6"),  # level not requested
    ]
    profiles = {
        "nonacademic": {"11.0701": _prof(0.2)},
        "academic": {"11.0701": _prof(0.9)},
        "fallback": {"52.1399": _prof(0.4)},
    }
    cat = build_real_catalog_v0(
        rows, profiles, {"11.0701": "Computer Science"}, "https://x", "2026-10-26",
        award_levels=("7", "17"), min_awards=10,
    )  # fmt: skip
    ids = {p.program_id: p for p in cat.programs}
    assert set(ids) == {"ipeds-1-11.0701-7", "ipeds-3-11.0701-17", "ipeds-4-52.1399-7"}
    assert cat.n_rows_considered == 5 and cat.n_below_min_awards == 1
    assert cat.skipped_no_profile == ("24.0101",) and cat.n_fallback == 1
    assert ids["ipeds-1-11.0701-7"].attributes["riasec_i"] == 0.2
    assert ids["ipeds-3-11.0701-17"].attributes["riasec_i"] == 0.9  # A-C0-10
    assert ids["ipeds-4-52.1399-7"].attribute_source == "occupation_fallback"
    p = ids["ipeds-1-11.0701-7"]
    assert p.degree_name == "Computer Science (Master's)" and not p.synthetic
    assert p.unmeasured == GAP and all(p.attributes[k] == 0.5 for k in GAP)


def test_builder_validates_arguments():
    with pytest.raises(ValueError):
        build_real_catalog_v0([], {"nonacademic": {}, "academic": {}}, {}, "u", "d", placeholder=2)
    with pytest.raises(ValueError):
        build_real_catalog_v0([], {"nonacademic": {}, "academic": {}}, {}, "u", "d", min_awards=0)
    with pytest.raises(ValueError):
        build_real_catalog_v0([], {"nonacademic": {}, "academic": {}}, {}, "u", "d",
                              award_levels=("9",))  # fmt: skip


def test_real_catalog_unmeasured_layers(real):
    """B-C0-18: every programme is real, sourced and v2; intensity and orientation are
    placeholders listed in `unmeasured`; one CIP gives one vector (B-C0-1)."""
    progs = real.programs
    assert len(progs) > 10_000 and real.skipped_no_profile  # a few CIPs have no profile
    assert len({p.program_id for p in progs}) == len(progs)
    assert all(not p.synthetic and p.source_url and p.retrieved_at for p in progs)
    assert all(p.dims == "v2" and p.unmeasured == GAP for p in progs)
    assert set(GAP) == {"quant_intensity", "programming_intensity", "research_orientation",
                        "entrepreneurship"}  # fmt: skip
    by_cip = {}
    for p in progs[:3000]:
        by_cip.setdefault(p.cip, p.vector)
        assert np.array_equal(by_cip[p.cip], p.vector)


def test_placeholder_is_rank_neutral_under_v2_default(real):
    """A-C0-15: a placeholder shared by the catalogue shifts every score equally."""
    other = load_real_catalog_v0(placeholder=0.2)
    X1 = np.array([p.vector for p in real.programs[:4000]])
    X2 = np.array([p.vector for p in other.programs[:4000]])
    rng = np.random.default_rng(0)
    for _ in range(5):
        w = rng.uniform(0, 1, len(KEYS))
        diff = DEFAULT_SCORER_V2.score_matrix(w, X1) - DEFAULT_SCORER_V2.score_matrix(w, X2)
        assert np.ptp(diff) < 1e-12


def test_jsonl_round_trip_keeps_unmeasured(real, tmp_path):
    path = tmp_path / "real.jsonl"
    write_jsonl(real.programs[:50], path)
    back = load_programs(path)
    assert [p.program_id for p in back] == [p.program_id for p in real.programs[:50]]
    assert back[0].unmeasured == GAP and back[0] == real.programs[0]


def test_discovery_and_report_run_on_real_catalog(real):
    catalog = real.programs
    student = StudentProfile("real-demo", {"riasec_i": 0.8, "bi_mathematics_statistics": 0.9,
                                           "bi_finance": 0.7, "quant_intensity": 0.9},
                             dims="v2")  # fmt: skip
    result = interactive_discovery(student, catalog, lambda ids: {i: "unknown" for i in ids},
                                   n_shortlist=10, k_new=5, k_known=0)  # fmt: skip
    assert len(result.discoveries) == 5
    assert all(not r.program.synthetic for r in result.discoveries)
    md = render_profile_report(student, prepare_profile(student, catalog), result)
    assert "尚未測量" in md and "合成的示範資料" not in md
    table = md.split("## 2.")[1].split("## 4.")[0]
    for k in ("數理強度", "程式強度", "研究取向", "創業取向"):
        assert f"{k}（" not in table  # placeholders are never cited as reasons
