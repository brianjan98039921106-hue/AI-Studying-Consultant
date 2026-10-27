"""C5.3 Evaluation runner (minimal; part of W-12).

Runs components under test against a synthetic population and scores them
with C5.2 metrics against ground truth. Components are called only through
their public APIs and `contracts` types.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aisc.contracts import DiscoveryResult, Program, StudentProfile
from aisc.discovery import label_similarity, recommend, relevance
from aisc.pipeline import AskFn, interactive_discovery
from aisc.profiling import blend_profile
from aisc.simulation import metrics as M
from aisc.simulation.population import SyntheticStudent
from aisc.simulation.respondent import make_ask
from aisc.simulation.truth import best_fit_id, relevant_ids, utilities

# A recommender takes an observed profile and returns program_ids.
Recommender = Callable[[StudentProfile, list[Program], int], list[str]]
# An interactive recommender (attribute `interactive = True`) also receives an
# `ask` function for the C1.4 familiarity check and returns a DiscoveryResult.
InteractiveRecommender = Callable[[StudentProfile, list[Program], int, AskFn], DiscoveryResult]


# --- baselines ------------------------------------------------------------------------


def popular_baseline(profile: StudentProfile, catalog: list[Program], k: int) -> list[str]:
    """(a) Most prestigious programs the student has not anchored on."""
    pool = [p for p in catalog if p.program_id not in set(profile.anchors)]
    pool.sort(key=lambda p: (-(p.prestige or 0.0), p.program_id))
    return [p.program_id for p in pool[:k]]


def similar_baseline(profile: StudentProfile, catalog: list[Program], k: int) -> list[str]:
    """(b) Programs most like the anchors: same label first, then attribute similarity."""
    by_id = {p.program_id: p for p in catalog}
    anchors = [by_id[a] for a in profile.anchors if a in by_id]
    if not anchors:
        target = profile.vector
        pool = sorted(catalog, key=lambda p: -relevance(target, p.vector))
        return [p.program_id for p in pool[:k]]
    target = np.mean([a.vector for a in anchors], axis=0)
    anchor_ids = {a.program_id for a in anchors}
    pool = [p for p in catalog if p.program_id not in anchor_ids]
    pool.sort(
        key=lambda p: (
            -max(label_similarity(p, a) for a in anchors),
            -relevance(target, p.vector),
        )
    )
    return [p.program_id for p in pool[:k]]


def c2_pipeline(
    novelty_weight: float = 0.3, diversity_weight: float = 0.3, blend: bool = True
) -> Recommender:
    """Current pipeline: C1.2 anchor blend (optional) -> C2 recommend."""

    def run(profile: StudentProfile, catalog: list[Program], k: int) -> list[str]:
        if blend:
            profile = blend_profile(profile, {p.program_id: p for p in catalog})
        recs = recommend(
            profile,
            catalog,
            k=k,
            novelty_weight=novelty_weight,
            diversity_weight=diversity_weight,
        )
        return [r.program.program_id for r in recs]

    return run


def c2_v1_pipeline(
    n_shortlist: int = 25, k_known: int = 5, diversity: float = 0.0
) -> InteractiveRecommender:
    """C2.3 v1 via the system pipeline: shortlist -> familiarity check -> sections.
    `diversity`: CIP-family diversity of shortlist and discoveries (B-C2-9)."""

    def run(profile: StudentProfile, catalog: list[Program], k: int, ask: AskFn):
        return interactive_discovery(
            profile,
            catalog,
            ask,
            n_shortlist=n_shortlist,
            k_new=k,
            k_known=k_known,
            diversity=diversity,
        )

    run.interactive = True  # type: ignore[attr-defined]
    return run


# --- evaluation -----------------------------------------------------------------------


def _list_rows(
    population: list[SyntheticStudent],
    catalog: list[Program],
    name: str,
    rec,
    k: int,
    truth: str,
    top_frac: float,
    fam_flag: bool,
    miss: float,
    false_alarm: float,
    seed: int,
    hidden: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, list[float]], list[list[str]]]:
    """Per-student list metrics for one recommender (one value per student, in order).

    With `hidden` (all-fields model, W-67) relevance and best fit use the full truth
    including hidden domains, and `hidden_mismatch_rate` reports the share of the
    list that is relevant on the measured dimensions but not on the full truth
    (B-C2-8)."""
    rows: dict[str, list[float]] = {
        "relevant_rate": [],
        "novel_relevant_rate": [],
        "unfamiliar_rate": [],
        "best_fit_retained": [],
        "outside_shortlist_rate": [],
    }
    if hidden is None and any(s.hidden_needs is not None for s in population):
        raise ValueError(  # B-C5-11
            "these students have hidden domains (all-fields model); pass `hidden` so that "
            "relevance uses the full truth"
        )
    if hidden is not None:
        rows["hidden_mismatch_rate"] = []
    interactive = getattr(rec, "interactive", False)
    lists: list[list[str]] = []
    for idx, s in enumerate(population):
        profile = s.observed_profile(include_familiar=fam_flag)
        if interactive:
            ask = make_ask(s, np.random.default_rng([seed, idx]), miss, false_alarm)
            result = rec(profile, catalog, k, ask)
            ids = [r.program.program_id for r in result.discoveries]
            shown = result.all_ids
            asked = set(result.asked)
            rows["outside_shortlist_rate"].append(
                float("nan") if not ids else sum(i not in asked for i in ids) / len(ids)
            )
        else:
            ids = rec(profile, catalog, k)
            shown = ids
            rows["outside_shortlist_rate"].append(float("nan"))
        lists.append(ids)
        wh = s.hidden_needs if hidden is not None else None
        rel = relevant_ids(s.true_needs, catalog, truth, top_frac, hidden, wh)
        for key, val in M.list_quality(ids, rel, set(s.familiar)).items():
            rows[key].append(val)
        best = best_fit_id(s.true_needs, catalog, set(s.anchors), truth, hidden, wh)
        rows["best_fit_retained"].append(M.best_fit_retained(shown, best))
        if hidden is not None:
            rel_measured = relevant_ids(s.true_needs, catalog, truth, top_frac)
            rows["hidden_mismatch_rate"].append(
                float("nan") if not ids
                else sum(i in rel_measured and i not in rel for i in ids) / len(ids)
            )  # fmt: skip
    return rows, lists


def _fam_settings(name, include_familiar, familiarity_noise):
    fam_flag = (
        include_familiar.get(name, False)
        if isinstance(include_familiar, dict)
        else include_familiar
    )
    miss, false_alarm = (familiarity_noise or {}).get(name, (0.0, 0.0))
    return fam_flag, miss, false_alarm


def evaluate_lists(
    population: list[SyntheticStudent],
    catalog: list[Program],
    recommenders: dict[str, Recommender],
    k: int = 10,
    truth: str = "distance",
    top_frac: float = 0.1,
    include_familiar: bool | dict[str, bool] = False,
    familiarity_noise: dict[str, tuple[float, float]] | None = None,
    seed: int = 7,
    hidden: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Mean ± SE of list metrics per recommender.

    `include_familiar` may be a dict to give only some recommenders the
    student's true familiarity set (an oracle C1.4).

    Interactive recommenders get a simulated respondent instead; its
    (miss_rate, false_alarm_rate) come from `familiarity_noise[name]`
    (default perfect answers). For them, list metrics are computed on the
    discoveries section, `best_fit_retained` on both sections, and
    `outside_shortlist_rate` reports discoveries that were never asked (B-C2-7).
    `hidden`: the catalogue's hidden domains for all-fields populations (W-67).
    """
    by_id = {p.program_id: p for p in catalog}
    catalog_prestige = [p.prestige or 0.0 for p in catalog]
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for name, rec in recommenders.items():
        fam_flag, miss, false_alarm = _fam_settings(name, include_familiar, familiarity_noise)
        rows, lists = _list_rows(population, catalog, name, rec, k, truth, top_frac,
                                 fam_flag, miss, false_alarm, seed, hidden)  # fmt: skip
        summary = {key: M.mean_se(vals) for key, vals in rows.items()}
        summary["coverage"] = (M.catalog_coverage(lists, len(catalog)), float("nan"))
        rec_prestige = [by_id[i].prestige or 0.0 for lst in lists for i in lst]
        summary["popularity_kl"] = (M.popularity_kl(rec_prestige, catalog_prestige), float("nan"))
        out[name] = summary
    return out


def evaluate_lists_per_student(
    population: list[SyntheticStudent],
    catalog: list[Program],
    recommenders: dict[str, Recommender],
    k: int = 10,
    truth: str = "distance",
    top_frac: float = 0.1,
    include_familiar: bool | dict[str, bool] = False,
    familiarity_noise: dict[str, tuple[float, float]] | None = None,
    seed: int = 7,
    hidden: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Same metrics as `evaluate_lists`, one value per student (for paired and
    stratified analysis, W-68). Same respondent seeds, so results line up."""
    out = {}
    for name, rec in recommenders.items():
        fam_flag, miss, false_alarm = _fam_settings(name, include_familiar, familiarity_noise)
        out[name], _ = _list_rows(population, catalog, name, rec, k, truth, top_frac,
                                  fam_flag, miss, false_alarm, seed, hidden)  # fmt: skip
    return out


def evaluate_profiles(
    population: list[SyntheticStudent],
    catalog: list[Program],
    truth: str = "distance",
) -> dict[str, dict[str, tuple[float, float]]]:
    """V-C1-1 (need recovery RMSE) and V-C1-2 (ranking agreement, Kendall τ-b)
    for stated-only vs C1.2 anchor-blended profiles."""
    by_id = {p.program_id: p for p in catalog}
    rows: dict[str, dict[str, list[float]]] = {
        "stated_only": {"rmse": [], "kendall_tau": []},
        "anchor_blend": {"rmse": [], "kendall_tau": []},
    }
    for s in population:
        true_u = utilities(s.true_needs, catalog, truth)
        observed = s.observed_profile()
        for name, prof in (
            ("stated_only", observed),
            ("anchor_blend", blend_profile(observed, by_id)),
        ):
            est = prof.vector
            rows[name]["rmse"].append(M.rmse(est, s.true_needs))
            rows[name]["kendall_tau"].append(
                M.kendall_tau_b(utilities(est, catalog, truth), true_u)
            )
    return {
        name: {key: M.mean_se(vals) for key, vals in metrics.items()}
        for name, metrics in rows.items()
    }


# --- C1 method comparison with partial elicitation (W-23) --------------------------------

ProfileMethod = Callable[[StudentProfile, dict[str, Program]], StudentProfile]


def mask_profile(
    profile: StudentProfile, coverage: float, rng: np.random.Generator
) -> StudentProfile:
    """Keep a random `coverage` share of answered dimensions (at least one).

    Simulates a student who has answered only some questions: cold start, or
    dimensions skipped by the adaptive questionnaire (C1.3).
    """
    if not 0 < coverage <= 1:
        raise ValueError("coverage must lie in (0, 1]")
    keys = list(profile.needs)
    n_keep = max(1, round(coverage * len(keys)))
    keep = set(rng.choice(len(keys), size=n_keep, replace=False).tolist())
    kept = [k for i, k in enumerate(keys) if i in keep]
    return StudentProfile(
        student_id=profile.student_id,
        needs={k: profile.needs[k] for k in kept},
        uncertainty={k: profile.uncertainty[k] for k in kept if k in profile.uncertainty},
        anchors=list(profile.anchors),
        familiar=list(profile.familiar),
        hard_constraints=dict(profile.hard_constraints),
    )


def evaluate_profile_methods(
    population: list[SyntheticStudent],
    catalog: list[Program],
    methods: dict[str, ProfileMethod],
    baseline: str,
    truth: str = "distance",
    coverage: float = 1.0,
    seed: int = 11,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Per method: τ, RMSE, and the *paired* τ difference vs `baseline` (mean ± SE).

    Every method sees the same masked profile for a given student, so the
    paired difference removes between-student variance.
    """
    if baseline not in methods:
        raise ValueError("baseline must be one of the methods")
    by_id = {p.program_id: p for p in catalog}
    taus: dict[str, list[float]] = {m: [] for m in methods}
    rmses: dict[str, list[float]] = {m: [] for m in methods}
    for idx, s in enumerate(population):
        observed = mask_profile(s.observed_profile(), coverage, np.random.default_rng([seed, idx]))
        true_u = utilities(s.true_needs, catalog, truth)
        for name, method in methods.items():
            est = method(observed, by_id).vector
            taus[name].append(M.kendall_tau_b(utilities(est, catalog, truth), true_u))
            rmses[name].append(M.rmse(est, s.true_needs))
    base = np.array(taus[baseline])
    return {
        name: {
            "kendall_tau": M.mean_se(taus[name]),
            "rmse": M.mean_se(rmses[name]),
            "tau_diff_vs_baseline": M.mean_se(list(np.array(taus[name]) - base)),
        }
        for name in methods
    }
