"""Need dimensions v2 (W-16c): registry, contracts, scorers, priors (L2)."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from aisc.contracts import Program, StudentProfile, require_same_dims
from aisc.dimensions import (
    BASIC_INTEREST_KEYS,
    LAYERS,
    NEED_KEYS,
    NEED_KEYS_V2,
    dims_for_length,
    dims_keys,
    keys_in_layer,
    migrate_needs_v1_to_v2,
)
from aisc.discovery import (
    COSINE,
    DEFAULT_SCORER,
    DEFAULT_SCORER_V2,
    SQDIST,
    blend,
    default_scorer,
    layer_balanced,
    layered,
    shortlist,
)
from aisc.profiling import complete_neutral
from aisc.profiling.hierarchy import (
    AREA_GROUPS,
    BASE_RATE,
    GROUP_LABELS,
    area_prior,
    areas_in_group,
)

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"


def _v2(values=None, fill=0.3):
    vals = {k: fill for k in NEED_KEYS_V2}
    vals.update(values or {})
    return vals


# --- registry ------------------------------------------------------------------------


def test_v2_registry_shape():
    assert len(NEED_KEYS_V2) == len(set(NEED_KEYS_V2)) == 51
    assert [len(keys_in_layer(lay, "v2")) for lay in LAYERS] == [6, 2, 41, 2]
    assert keys_in_layer("interest", "v2") == keys_in_layer("interest")
    assert dims_for_length(15) == "v1" and dims_for_length(51) == "v2"
    assert dims_keys("v1") == NEED_KEYS
    with pytest.raises(ValueError):
        dims_keys("v3")
    with pytest.raises(ValueError):
        dims_for_length(16)


def test_v2_areas_match_the_official_map():
    with open(REF / "basic_interest_map.csv", encoding="utf-8") as fh:
        official = [r["key"] for r in csv.DictReader(fh)]
    assert list(BASIC_INTEREST_KEYS) == official


def test_migrate_v1_to_v2():
    v1 = {"riasec_i": 0.8, "domain_finance": 0.9, "domain_tech_product": 0.4,
          "domain_operations": 0.6, "research_orientation": 0.2}  # fmt: skip
    v2 = migrate_needs_v1_to_v2(v1)
    assert v2["riasec_i"] == 0.8 and v2["research_orientation"] == 0.2
    assert v2["bi_finance"] == 0.9 and v2["bi_engineering"] == 0.4
    assert v2["bi_management_administration"] == 0.6
    assert not any(k.startswith("domain_") for k in v2)
    assert set(v2) <= set(NEED_KEYS_V2)
    with pytest.raises(ValueError):
        migrate_needs_v1_to_v2({"bi_finance": 0.5})


# --- contracts -----------------------------------------------------------------------


def test_contracts_carry_the_dimension_set():
    p = Program("p", "S", "MS", "27.0501", _v2(), synthetic=True, dims="v2")
    assert p.vector.shape == (51,)
    with pytest.raises(ValueError):
        _ = Program("q", "S", "MS", "27.0501", {"bi_finance": 0.5}, synthetic=True).vector
    with pytest.raises(ValueError):
        Program("q", "S", "MS", "27.0501", {}, synthetic=True, dims="v9")
    s = StudentProfile("s", _v2(), dims="v2", group_scores={1: 0.8})
    assert s.vector.shape == (51,) and not s.missing_dimensions
    with pytest.raises(ValueError, match="only for the v2"):
        StudentProfile("s", {}, group_scores={1: 0.5})
    v1_prog = Program("v1", "S", "MS", "27.0501", {"riasec_i": 0.5}, synthetic=True)
    with pytest.raises(ValueError, match="migrate"):
        require_same_dims(s, [p, v1_prog])


# --- scorers -------------------------------------------------------------------------

UNEVEN = layer_balanced({"interest": 2, "intensity": 1, "domain": 1, "orientation": 1})
SCORERS = [COSINE, SQDIST, blend(0.25), layered(1.0), DEFAULT_SCORER_V2, UNEVEN]


@pytest.mark.parametrize("scorer", SCORERS, ids=[s.name for s in SCORERS])
@pytest.mark.parametrize("n", [15, 51])
def test_explanations_stay_exact_in_both_dimension_sets(scorer, n):
    rng = np.random.default_rng(n)
    w, X = rng.random(n), rng.random((5, n))
    m = scorer.score_matrix(w, X)
    for i, x in enumerate(X):
        expl = scorer.explain(w, x)
        assert list(expl) == list(dims_keys(dims_for_length(n)))
        assert sum(expl.values()) == pytest.approx(m[i])
        assert scorer.score(w, x) == pytest.approx(m[i])


def test_layer_balanced_is_the_weighted_sum_of_layer_scores():
    rng = np.random.default_rng(0)
    w, x = rng.random(51), rng.random(51)
    s = layer_balanced({"interest": 1, "intensity": 1, "domain": 2, "orientation": 0})
    keys = dims_keys("v2")
    expected = 0.0
    for lay, wt in (("interest", 0.25), ("intensity", 0.25), ("domain", 0.5)):
        idx = [keys.index(k) for k in keys_in_layer(lay, "v2")]
        expected += wt * blend(0.25).score(w[idx], x[idx])
    assert s.score(w, x) == pytest.approx(expected)
    with pytest.raises(ValueError):
        layer_balanced({"interest": 0, "intensity": 0, "domain": 0, "orientation": 0})
    assert default_scorer(15) is DEFAULT_SCORER and default_scorer(51) is DEFAULT_SCORER_V2


def test_domain_layer_cannot_drown_the_other_layers():
    """Unbalanced blend on 51 dims lets 41 domain dims dominate; the balanced one does not."""
    keys = dims_keys("v2")
    dom = [keys.index(k) for k in keys_in_layer("domain", "v2")]
    w = np.full(51, 0.1)
    w[[keys.index("riasec_i"), keys.index("quant_intensity")]] = 0.9
    match_rest = w.copy()  # matches interest/intensity/orientation, domain slightly off
    match_rest[dom] = 0.2
    match_domain = np.full(51, 0.5)  # domain matches exactly, everything else off
    match_domain[dom] = 0.1
    X = np.vstack([match_rest, match_domain])
    assert DEFAULT_SCORER_V2.score_matrix(w, X)[0] > DEFAULT_SCORER_V2.score_matrix(w, X)[1]


# --- hierarchy and priors ------------------------------------------------------------


def test_hierarchy_constants_match_reference_files():
    spec = json.loads((REF / "interest_area_clusters.json").read_text())
    primary = {a["key"]: c["id"] for c in spec["clusters"] for a in c["areas"]}
    assert {k: g[0] for k, g in AREA_GROUPS.items()} == primary
    assert set(GROUP_LABELS) == {c["id"] for c in spec["clusters"]}
    with open(REF / "basic_interest_map.csv", encoding="utf-8") as fh:
        base = {r["key"]: float(r["occupation_mean"]) for r in csv.DictReader(fh)}
    assert pytest.approx(base, abs=1e-4) == BASE_RATE


def test_area_prior_and_cross_links():
    assert "bi_law" in areas_in_group(3) and "bi_law" in areas_in_group(9)
    assert area_prior("bi_law", {3: 0.7, 9: 0.2}) == (0.7, 0.25)
    assert area_prior("bi_music", {1: 0.9}) == (BASE_RATE["bi_music"], 0.30)
    with pytest.raises(ValueError):
        area_prior("domain_finance", {})
    with pytest.raises(ValueError):
        areas_in_group(13)


def test_v2_completion_never_fills_areas_with_neutral_values():
    """B-C1-24 guard: 41 areas filled with 0.5 would claim interest in everything."""
    prof = StudentProfile("s", {"riasec_i": 0.8, "bi_finance": 0.9}, dims="v2",
                          group_scores={6: 0.75})  # fmt: skip
    done = complete_neutral(prof)
    assert not done.missing_dimensions and done.dims == "v2" and done.group_scores == {6: 0.75}
    assert done.needs["bi_finance"] == 0.9  # answered: kept
    assert done.needs["bi_life_science"] == 0.75  # group 6 rated
    assert done.needs["bi_music"] == pytest.approx(BASE_RATE["bi_music"])  # nothing rated
    assert done.needs["riasec_a"] == 0.5  # non-domain layers: neutral as in v1
    areas = [done.needs[k] for k in BASIC_INTEREST_KEYS if k not in ("bi_finance",)]
    assert np.mean(areas) < 0.3
    with pytest.raises(ValueError):
        complete_neutral(StudentProfile("s", {}, dims="v2", group_scores={13: 0.5}))


def test_v2_discovery_end_to_end_and_mixing_is_rejected():
    rng = np.random.default_rng(1)
    progs = [Program(f"p{i}", "S", "MS", "27.0501",
                     dict(zip(NEED_KEYS_V2, map(float, rng.random(51)), strict=True)),
                     synthetic=True, dims="v2") for i in range(8)]  # fmt: skip
    prof = complete_neutral(StudentProfile("s", {"bi_finance": 0.9}, dims="v2"))
    assert len(shortlist(prof, progs, n=3)) == 3
    v1prof = complete_neutral(StudentProfile("s", {}))
    with pytest.raises(ValueError, match="dimension set"):
        shortlist(v1prof, progs, n=3)
