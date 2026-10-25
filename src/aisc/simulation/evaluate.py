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


def c2_v1_pipeline(n_shortlist: int = 25, k_known: int = 5) -> InteractiveRecommender:
    """C2.3 v1 via the system pipeline: shortlist -> familiarity check -> sections."""

    def run(profile: StudentProfile, catalog: list[Program], k: int, ask: AskFn):
        return interactive_discovery(
            profile, catalog, ask, n_shortlist=n_shortlist, k_new=k, k_known=k_known
        )

    run.interactive = True  # type: ignore[attr-defined]
    return run


# --- evaluation -----------------------------------------------------------------------


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
) -> dict[str, dict[str, tuple[float, float]]]:
    """Mean ± SE of list metrics per recommender.

    `include_familiar` may be a dict to give only some recommenders the
    student's true familiarity set (an oracle C1.4).

    Interactive recommenders get a simulated respondent instead; its
    (miss_rate, false_alarm_rate) come from `familiarity_noise[name]`
    (default perfect answers). For them, list metrics are computed on the
    discoveries section, `best_fit_retained` on both sections, and
    `outside_shortlist_rate` reports discoveries that were never asked (B-C2-7).
    """
    noise = familiarity_noise or {}
    by_id = {p.program_id: p for p in catalog}
    catalog_prestige = [p.prestige or 0.0 for p in catalog]
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for name, rec in recommenders.items():
        fam_flag = (
            include_familiar.get(name, False)
            if isinstance(include_familiar, dict)
            else include_familiar
        )
        rows: dict[str, list[float]] = {
            "relevant_rate": [],
            "novel_relevant_rate": [],
            "unfamiliar_rate": [],
            "best_fit_retained": [],
            "outside_shortlist_rate": [],
        }
        interactive = getattr(rec, "interactive", False)
        miss, false_alarm = noise.get(name, (0.0, 0.0))
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
            rel = relevant_ids(s.true_needs, catalog, truth, top_frac)
            for key, val in M.list_quality(ids, rel, set(s.familiar)).items():
                rows[key].append(val)
            best = best_fit_id(s.true_needs, catalog, set(s.anchors), truth)
            rows["best_fit_retained"].append(M.best_fit_retained(shown, best))
        summary = {key: M.mean_se(vals) for key, vals in rows.items()}
        summary["coverage"] = (M.catalog_coverage(lists, len(catalog)), float("nan"))
        rec_prestige = [by_id[i].prestige or 0.0 for lst in lists for i in lst]
        summary["popularity_kl"] = (M.popularity_kl(rec_prestige, catalog_prestige), float("nan"))
        out[name] = summary
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
