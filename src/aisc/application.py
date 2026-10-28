"""Module B — application planning (C3 -> C4), deferred (ROADMAP 模組 B).

    plan_applications (application mode only, W-62, D9):
        recommendations + C3 AdmitEstimates -> C4.1 / C4.3 choice, with the C4.2
        correlated-admission value of the chosen portfolio as a sensitivity check

Module A (aisc.pipeline and the discovery components) never imports this module;
it consumes module A's output (`Recommendation`) and nothing flows back (D9).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from aisc import portfolio as _pf
from aisc.contracts import AdmitEstimate, PortfolioResult, Recommendation

PLAN_CRITERIA = ("point", "worst_case", "minimax_regret")


@dataclass(frozen=True)
class ApplicationPlan:
    """C4 choice over recommended programmes. `result.chosen` indexes `program_ids`;
    `value_if_correlated` maps rho -> expected utility of the chosen portfolio when
    admissions are correlated (C4.2; F109: independence overstates it); `assumed` is
    True when any admission probability was an assumption rather than a C3 estimate."""

    program_ids: list[str]
    result: PortfolioResult
    value_if_correlated: dict[float, float]
    assumed: bool

    @property
    def chosen_ids(self) -> list[str]:
        return [self.program_ids[j] for j in self.result.chosen]


def plan_applications(
    recommendations: Sequence[Recommendation],
    admit: Mapping[str, AdmitEstimate],
    budget: int,
    cost: Mapping[str, int] | None = None,
    utility: Mapping[str, float] | None = None,
    criterion: str = "point",
    rhos: Sequence[float] = (0.0, 0.4, 0.8),
    n_sim: int = 20_000,
    rng: np.random.Generator | None = None,
) -> ApplicationPlan:
    """Choose which recommended programmes to apply to (application mode only, D9:
    nothing here feeds back into the recommendations).

    Utility defaults to 100 x relevance; cost defaults to 1 per application. The
    default criterion is the C4.1 point estimate (F110, F120-F122, F134);
    "worst_case" and "minimax_regret" use the AdmitEstimate intervals (C4.3)."""
    if criterion not in PLAN_CRITERIA:
        raise ValueError(f"criterion must be one of {PLAN_CRITERIA}")
    ids = [r.program.program_id for r in recommendations]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate programmes in recommendations")
    missing = [i for i in ids if i not in admit]
    if missing:
        raise ValueError(f"no AdmitEstimate for {missing}")
    est = [admit[i] for i in ids]
    if any(e.program_id != i for e, i in zip(est, ids, strict=True)):
        raise ValueError("AdmitEstimate.program_id does not match its key")
    u = [float(utility[i]) if utility is not None else 100 * r.relevance
         for i, r in zip(ids, recommendations, strict=True)]  # fmt: skip
    c = [int(cost[i]) if cost is not None else 1 for i in ids]
    p, low, high = [e.mean for e in est], [e.low for e in est], [e.high for e in est]
    rng = rng if rng is not None else np.random.default_rng(0)
    if criterion == "point":
        chosen, _ = _pf.dp_varied_cost(p, u, c, budget)
    elif criterion == "worst_case":
        chosen = _pf.worst_case_portfolio(low, u, c, budget)
    else:
        chosen = _pf.minimax_regret_portfolio(low, high, p, u, c, budget, rng=rng)
    chosen = sorted(chosen)
    assumed = any(e.method == "assumed" for e in est)
    method = f"C4 {criterion}" + (" (assumed P(admit))" if assumed else "")
    result = PortfolioResult(chosen, _pf.portfolio_value(p, u, chosen), method)
    corr = {}
    for rho in rhos:
        if rho == 0:
            corr[rho] = result.expected_utility
        else:
            draws = _pf.admission_draws(p, rho, n_sim, rng)
            corr[rho] = _pf.portfolio_value_mc(u, draws, chosen)
    return ApplicationPlan(ids, result, corr, assumed)
