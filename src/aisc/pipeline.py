"""System layer: composes components through their public APIs only.

    run_questionnaire:
        C1.2 v1 completion (prior) -> C1.3 adaptive questionnaire, scored with C2.1
    interactive_discovery:
        C1.2 v1 profile completion -> C2 shortlist -> C1.4 familiarity check (ask)
        -> C2.3 v1 discover

Module A (discovery: C0, C1, C2, C6) only. Application planning (C3, C4) lives in
`aisc.application` (module B, deferred) and must not be imported from here; see
tests/test_module_boundary.py.

`ask` is whatever collects the student's answers: a UI in production, a
simulated respondent in C5. It receives program_ids and returns
{program_id: "unknown" | "heard" | "can_describe"}.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aisc.contracts import (
    DiscoveryResult,
    Program,
    StudentProfile,
    require_same_dims,
)
from aisc.discovery import Scorer, discover, shortlist
from aisc.discovery import scorers as _scorers
from aisc.discovery.constraints import passes_constraints, validate_constraints
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
    """Ask up to `budget` items chosen by C1.3 and return the posterior profile.

    The anchor prior uses the whole catalogue, but C1.3 chooses questions only for
    programmes that can be recommended: anchors and programmes the hard constraints
    remove are left out (B-C1-28)."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if not catalog:
        raise ValueError("the catalogue is empty")  # B-C1-30
    require_same_dims(profile, catalog)
    validate_constraints(profile.hard_constraints)
    anchors = set(profile.anchors)
    hard = profile.hard_constraints
    pool = [p for p in catalog if p.program_id not in anchors and passes_constraints(p, hard)]
    if not pool:
        raise ValueError("no recommendable programmes left after anchors and hard constraints")
    session = AdaptiveSession(
        prior=prepare_profile(profile, catalog),
        catalog=pool,
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


_MOVED = {"plan_applications", "ApplicationPlan", "PLAN_CRITERIA"}


def __getattr__(name: str):
    """Backward compatibility: application mode moved to `aisc.application` (v0.3.20).
    Imported lazily so that importing this module never loads C3/C4."""
    if name in _MOVED:
        import warnings

        from aisc import application

        warnings.warn(
            f"aisc.pipeline.{name} moved to aisc.application.{name}",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(application, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
