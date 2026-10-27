"""C5 v1/v2 comparison world (W-16c): verification (L2)."""

import numpy as np
import pytest

from aisc.contracts import StudentProfile
from aisc.dimensions import BASIC_INTEREST_KEYS, NEED_KEYS, NEED_KEYS_V2
from aisc.profiling import apply_familiarity
from aisc.profiling.hierarchy import GROUP_LABELS, areas_in_group
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.fields import HIDDEN_KEYS, all_fields_catalog
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.world_v2 import (
    AREA_LOW,
    N_RESIDUAL,
    group_answer,
    to_v1,
    to_v2,
    two_stage_profile,
    v2_catalog,
    v2_students,
)


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    src, hidden = all_fields_catalog(rng, synthetic_catalog(rng, n_per_family=2), 2)
    v2, v1, res = v2_catalog(src, hidden, rng)
    pop = generate_population(src, SimConfig(n_students=40, field_model="all_fields"),
                              rng, hidden=hidden)  # fmt: skip
    return src, hidden, v2, v1, res, v2_students(pop, rng)


def test_conversions(world):
    v1 = {k: 0.5 for k in NEED_KEYS}
    v1["domain_finance"] = 0.9
    h = np.full(len(HIDDEN_KEYS), 0.05)
    h[HIDDEN_KEYS.index("h_health")] = 0.95
    v2 = to_v2(v1, h)
    assert set(v2) == set(NEED_KEYS_V2)
    assert v2["bi_finance"] == 0.9 and v2["bi_medical_science"] == 0.95
    assert v2["bi_music"] == AREA_LOW
    back = to_v1(v2)
    assert set(back) == set(NEED_KEYS) and back["domain_finance"] == 0.9
    assert back["domain_tech_product"] == pytest.approx(0.5)  # mean(IT 0.5, engineering 0.5)
    with pytest.raises(ValueError):
        to_v2({"riasec_i": 0.5}, None)


def test_catalog_views_share_ids(world):
    src, _, v2, v1, res, _ = world
    assert [p.program_id for p in v2] == [p.program_id for p in v1] == [p.program_id for p in src]
    assert all(p.dims == "v2" and p.vector.shape == (51,) for p in v2)
    assert all(p.dims == "v1" and p.vector.shape == (15,) for p in v1)
    assert all(r.shape == (N_RESIDUAL,) for r in res.values())


def test_two_stage_profile(world):
    *_, students = world
    st, w2, _ = students[0]
    true = dict(zip(NEED_KEYS_V2, map(float, w2), strict=True))
    for m in (1, 2, 3):
        prof, items = two_stage_profile(st, w2, m, np.random.default_rng(m), 0.1, 0.0)
        asked = [k for k in prof.needs if k in BASIC_INTEREST_KEYS]
        assert items == 10 + len(GROUP_LABELS) + len(asked)
        assert prof.dims == "v2" and set(prof.group_scores) == set(GROUP_LABELS)
        top = sorted(prof.group_scores, key=lambda g: -prof.group_scores[g])[:m]
        assert set(asked) == {a for g in top for a in areas_in_group(g)}
    assert group_answer(true, 6) == max(true[k] for k in ("bi_life_science", "bi_medical_science",
                                                          "bi_health_care_service"))  # fmt: skip


def test_profile_updates_keep_the_dimension_set():
    prof = StudentProfile("s", {"bi_law": 0.8}, dims="v2", group_scores={9: 0.7})
    upd = apply_familiarity(prof, {"p1": "heard"})
    assert upd.dims == "v2" and upd.group_scores == {9: 0.7} and upd.familiar == ["p1"]
    upd.group_scores[9] = 0.1
    assert prof.group_scores == {9: 0.7}  # copies, not shared
