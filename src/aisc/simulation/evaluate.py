"""C5.3 Evaluation runner (minimal; part of W-12).

Runs components under test against a synthetic population and scores them
with C5.2 metrics against ground truth. Components are called only through
their public APIs and `contracts` types.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aisc.contracts import Program, StudentProfile
from aisc.discovery import label_similarity, recommend, relevance
from aisc.profiling import blend_profile
from aisc.simulation import metrics as M
from aisc.simulation.population import SyntheticStudent
from aisc.simulation.truth import best_fit_id, relevant_ids, utilities

# A recommender takes an observed profile and returns program_ids.
Recommender = Callable[[StudentProfile, list[Program], int], list[str]]


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


# --- evaluation -----------------------------------------------------------------------


def evaluate_lists(
    population: list[SyntheticStudent],
    catalog: list[Program],
    recommenders: dict[str, Recommender],
    k: int = 10,
    truth: str = "distance",
    top_frac: float = 0.1,
    include_familiar: bool | dict[str, bool] = False,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Mean ± SE of list metrics per recommender.

    `include_familiar` may be a dict to give only some recommenders the
    student's true familiarity set (simulating C1.4).
    """
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
        }
        lists: list[list[str]] = []
        for s in population:
            profile = s.observed_profile(include_familiar=fam_flag)
            ids = rec(profile, catalog, k)
            lists.append(ids)
            rel = relevant_ids(s.true_needs, catalog, truth, top_frac)
            for key, val in M.list_quality(ids, rel, set(s.familiar)).items():
                rows[key].append(val)
            best = best_fit_id(s.true_needs, catalog, set(s.anchors), truth)
            rows["best_fit_retained"].append(M.best_fit_retained(ids, best))
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
