"""System layer: composes components through their public APIs only.

    run_questionnaire:
        C1.2 v1 completion (prior) -> C1.3 adaptive questionnaire, scored with C2.1
    interactive_discovery:
        C1.2 v1 profile completion -> C2 shortlist -> C1.4 familiarity check (ask)
        -> C2.3 v1 discover

`ask` is whatever collects the student's answers: a UI in production, a
simulated respondent in C5. It receives program_ids and returns
{program_id: "unknown" | "heard" | "can_describe"}.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aisc.contracts import DiscoveryResult, Program, StudentProfile
from aisc.discovery import Scorer, discover, shortlist
from aisc.discovery import scorers as _scorers
from aisc.profiling import AdaptiveSession, Item, apply_familiarity, blend_profile_v1

AskFn = Callable[[list[str]], dict[str, str]]
AnswerFn = Callable[[Item], float]


def c2_scores(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Default C2.1 relevance for every catalogue row: what C1.3 optimises for."""
    return _scorers.DEFAULT_SCORER.score_matrix(w, X)


def prepare_profile(profile: StudentProfile, catalog: list[Program]) -> StudentProfile:
    """Complete unanswered dimensions before ranking (B-C1-14).

    Without this, `StudentProfile.vector` treats an unanswered dimension as 0
    ("no interest") instead of "unknown". Uses C1.2 v1 with its confirmed
    defaults (results/synthetic-l3-c1-v1-findings.md): anchors act only as a
    prior for unanswered dimensions; answered dimensions are left unchanged.
    """
    return blend_profile_v1(profile, {p.program_id: p for p in catalog})


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
) -> DiscoveryResult:
    profile = prepare_profile(profile, catalog)
    candidates = shortlist(
        profile, catalog, n=n_shortlist, min_relevance=min_relevance, scorer=scorer
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
    )
