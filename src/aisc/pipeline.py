"""System layer: composes components through their public APIs only.

    run_questionnaire:
        C1.2 v1 completion (prior) -> C1.3 adaptive questionnaire, scored with C2.1
    interactive_discovery:
        C1.2 v1 profile completion -> C2 shortlist -> C1.4 familiarity check (ask)
        -> C2.3 v1 discover
    plan_applications (application mode only, W-62, D9):
        recommendations + C3 AdmitEstimates -> C4.1 / C4.3 choice, with the C4.2
        correlated-admission value of the chosen portfolio as a sensitivity check

`ask` is whatever collects the student's answers: a UI in production, a
simulated respondent in C5. It receives program_ids and returns
{program_id: "unknown" | "heard" | "can_describe"}.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from aisc import portfolio as _pf
from aisc.contracts import (
    AdmitEstimate,
    DiscoveryResult,
    PortfolioResult,
    Program,
    Recommendation,
    StudentProfile,
)
from aisc.discovery import Scorer, discover, shortlist
from aisc.discovery import scorers as _scorers
from aisc.profiling import AdaptiveSession, Item, apply_familiarity, blend_profile_v1

AskFn = Callable[[list[str]], dict[str, str]]
AnswerFn = Callable[[Item], float]


def c2_scores(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Default C2.1 relevance for every catalogue row: what C1.3 optimises for."""
    return _scorers.default_scorer(len(w)).score_matrix(w, X)


def prepare_profile(
    profile: StudentProfile, catalog: list[Program], area_model: str = "hierarchy"
) -> StudentProfile:
    """Complete unanswered dimensions before ranking (B-C1-14).

    Without this, `StudentProfile.vector` treats an unanswered dimension as 0
    ("no interest") instead of "unknown". Uses C1.2 v1 with its confirmed
    defaults (results/synthetic-l3-c1-v1-findings.md): anchors act only as a
    prior for unanswered dimensions; answered dimensions are left unchanged.
    """
    return blend_profile_v1(profile, {p.program_id: p for p in catalog}, area_model=area_model)


def run_questionnaire(
    profile: StudentProfile,
    catalog: list[Program],
    answer: AnswerFn,
    budget: int,
    strategy: str = "voi",
    item_noise_sd: float = 0.25,
    rng: np.random.Generator | None = None,
    scorer: Scorer | None = None,
    noise_model: str = "fixed",
    noise_prior_df: float = 4.0,
    noise_prior_center: str = "scale",
) -> StudentProfile:
    """Ask up to `budget` items chosen by C1.3 and return the posterior profile."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    session = AdaptiveSession(
        prior=prepare_profile(profile, catalog),
        catalog=catalog,
        strategy=strategy,
        item_noise_sd=item_noise_sd,
        score_fn=scorer.score_matrix if scorer is not None else c2_scores,
        rng=rng if rng is not None else np.random.default_rng(0),
        noise_model=noise_model,
        noise_prior_df=noise_prior_df,
        noise_prior_center=noise_prior_center,
    )
    for _ in range(budget):
        item = session.next_item()
        if item is None:
            break
        session.record(item, answer(item))
    return session.profile()


def interactive_discovery(
    profile: StudentProfile,
    catalog: list[Program],
    ask: AskFn,
    n_shortlist: int = 25,
    k_new: int = 10,
    k_known: int = 5,
    min_relevance: float = 0.0,
    scorer: Scorer | None = None,
    diversity: float = 0.0,
) -> DiscoveryResult:
    profile = prepare_profile(profile, catalog)
    candidates = shortlist(
        profile,
        catalog,
        n=n_shortlist,
        min_relevance=min_relevance,
        scorer=scorer,
        diversity=diversity,
    )
    report = ask(candidates) if candidates else {}
    unexpected = set(report) - set(candidates)
    if unexpected:
        raise ValueError(f"ask() answered programs that were not asked: {sorted(unexpected)}")
    updated = apply_familiarity(profile, report)
    return discover(
        updated,
        catalog,
        k_new=k_new,
        k_known=k_known,
        min_relevance=min_relevance,
        asked=candidates,
        scorer=scorer,
        diversity=diversity,
    )


# --- application mode (W-62) ---------------------------------------------------------

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
