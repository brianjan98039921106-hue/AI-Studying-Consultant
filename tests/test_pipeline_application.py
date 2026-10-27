"""W-62 application mode: plan_applications composes C2 output, C3 estimates and C4."""

import copy

import pytest

from aisc import portfolio as pf
from aisc.contracts import AdmitEstimate, StudentProfile
from aisc.pipeline import interactive_discovery, plan_applications

NEEDS = {"quant_intensity": 0.8, "domain_finance": 0.9, "riasec_e": 0.6}


@pytest.fixture
def recs(catalog):
    res = interactive_discovery(
        StudentProfile("s", NEEDS), catalog, lambda ids: {i: "unknown" for i in ids}, k_new=6
    )
    return res.discoveries


def _assumed(recs, base=0.3):
    return {
        r.program.program_id: AdmitEstimate(
            r.program.program_id, base + 0.05 * j, base + 0.05 * j - 0.1,
            base + 0.05 * j + 0.1, "assumed",
        )
        for j, r in enumerate(recs)
    }  # fmt: skip


def test_point_plan_is_the_c41_optimum_and_labels_assumptions(recs):
    admit = _assumed(recs)
    before = copy.deepcopy(recs)
    plan = plan_applications(recs, admit, budget=3)
    assert recs == before  # D9: planning never changes the recommendations
    p = [admit[r.program.program_id].mean for r in recs]
    u = [100 * r.relevance for r in recs]
    chosen, value = pf.dp_varied_cost(p, u, [1] * len(recs), 3)
    assert plan.result.chosen == sorted(chosen)
    assert plan.result.expected_utility == pytest.approx(value)
    assert plan.assumed and "assumed" in plan.result.method
    assert len(plan.chosen_ids) == 3
    v = plan.value_if_correlated
    assert v[0.0] == pytest.approx(value) and v[0.8] < v[0.0]


def test_interval_criteria_and_validation(recs):
    admit = _assumed(recs)
    low = [admit[r.program.program_id].low for r in recs]
    u = [100 * r.relevance for r in recs]
    wc = plan_applications(recs, admit, budget=2, criterion="worst_case")
    assert wc.result.chosen == pf.worst_case_portfolio(low, u, [1] * len(recs), 2)
    mr = plan_applications(recs, admit, budget=2, criterion="minimax_regret")
    assert len(mr.result.chosen) <= 2
    with pytest.raises(ValueError):
        plan_applications(recs, admit, budget=2, criterion="optimistic")
    partial = dict(list(admit.items())[1:])
    with pytest.raises(ValueError):
        plan_applications(recs, partial, budget=2)
    wrong = dict(admit)
    k = next(iter(wrong))
    wrong[k] = AdmitEstimate("other", 0.3, 0.2, 0.4, "assumed")
    with pytest.raises(ValueError):
        plan_applications(recs, wrong, budget=2)


def test_custom_costs_and_utilities(recs):
    admit = _assumed(recs)
    ids = [r.program.program_id for r in recs]
    cost = {i: (2 if j == 0 else 1) for j, i in enumerate(ids)}
    util = {i: float(10 - j) for j, i in enumerate(ids)}
    plan = plan_applications(recs, admit, budget=2, cost=cost, utility=util, rhos=())
    assert sum(cost[ids[j]] for j in plan.result.chosen) <= 2
    assert plan.value_if_correlated == {}
