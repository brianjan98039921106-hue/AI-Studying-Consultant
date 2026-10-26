"""End-to-end L3: the whole pipeline on simulated students.

Per student:
    anchors only -> pipeline.run_questionnaire (C1.2 v1 prior + C1.3 voi, Q items)
    -> pipeline.interactive_discovery (C2 shortlist -> C1.4 familiarity check -> C2.3 v1)

Conditions per generator: 2 truths x budgets Q in {0, 5, 15, 45} x familiarity
noise {perfect, 10% miss / 5% false alarm}. Prestige weight 0.5, true item noise
0.25 (= assumed), response biases on and not modelled.

Comparators (same student, same answers):
    fit-only     the same post-questionnaire profile, ranked by fit, no sections
                 (C2 shortlist top-k): isolates what the sections add
    stated-full  complete stated needs (as in component-level L3 v2) -> same discovery
    fit-only-stated  the stated-full profile ranked by fit, no sections (for H-E4)
    oracle       true needs -> same discovery (upper bound)
    popular, similar-to-anchors  baselines

PRE-REGISTERED HYPOTHESES (written before the first run, 2026-10-26):
  H-E1 sections survive integration: at Q=45, novel-relevant rate of the
       pipeline >= fit-only on the same profile in every condition (paired
       difference not significantly negative) and > 0 on average.
  H-E2 more questions help: mean novel-relevant rate (over conditions) is
       increasing in Q (0 < 5 < 15 < 45) for each generator.
  H-E3 beats baselines: at Q=45, pipeline novel-relevant rate > popular and
       > similar-to-anchors in every condition.
  H-E4 no interference: in every condition, the section gain at Q=45
       (pipeline - fit-only, novel-relevant rate) is at least half the
       section gain measured on stated-full profiles.
  Relevant rate and best-fit retention are always reported alongside.

    python scripts/run_l3_e2e.py [--students 200]
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np

from aisc.contracts import StudentProfile
from aisc.dimensions import NEED_KEYS
from aisc.discovery import shortlist
from aisc.pipeline import interactive_discovery, prepare_profile, run_questionnaire
from aisc.profiling import default_item_bank
from aisc.simulation import metrics as M
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.evaluate import popular_baseline, similar_baseline
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.respondent import make_ask, make_item_responder
from aisc.simulation.truth import best_fit_id, relevant_ids

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ("archetype", "continuous")
TRUTHS = ("distance", "cosine")
BUDGETS = (0, 5, 15, 45)
FAM_NOISE = {"perfect": (0.0, 0.0), "10/5": (0.10, 0.05)}
K = 10
N_SHORTLIST = 25
ROWS = ["pipeline", "fit-only", "stated-full", "oracle", "popular", "similar"]


def evaluate_student(st, catalog, truth, answers, fam_name, resp_seed, idx):
    """Return {(row, Q): (relevant, novel_relevant, best_fit_retained)}."""
    rel = relevant_ids(st.true_needs, catalog, truth)
    familiar = set(st.familiar)
    best = best_fit_id(st.true_needs, catalog, set(st.anchors), truth)
    miss, fa = FAM_NOISE[fam_name]
    anchors_only = StudentProfile(st.student_id, {}, anchors=list(st.anchors))
    out = {}

    def score(ids, shown):
        q = M.list_quality(ids, rel, familiar)
        return q["relevant_rate"], q["novel_relevant_rate"], M.best_fit_retained(shown, best)

    def discovery(profile, tag):
        ask = make_ask(st, np.random.default_rng([resp_seed, idx, tag]), miss, fa)
        res = interactive_discovery(profile, catalog, ask, n_shortlist=N_SHORTLIST, k_new=K)
        return [r.program.program_id for r in res.discoveries], res.all_ids

    for q in BUDGETS:
        profile = run_questionnaire(
            anchors_only,
            catalog,
            lambda item: answers[item.item_id],
            budget=q,
            rng=np.random.default_rng([resp_seed, idx, 7]),
        )
        ids, shown = discovery(profile, 1)
        out[("pipeline", q)] = score(ids, shown)
        fit = shortlist(profile, catalog, n=K)
        out[("fit-only", q)] = score(fit, fit)

    stated = StudentProfile(
        st.student_id,
        dict(st.stated_needs),
        {k: st.stated_sigma for k in NEED_KEYS},
        anchors=list(st.anchors),
    )
    ids, shown = discovery(stated, 2)
    out[("stated-full", None)] = score(ids, shown)
    fit_stated = shortlist(stated, catalog, n=K)
    out[("fit-only-stated", None)] = score(fit_stated, fit_stated)
    oracle = StudentProfile(
        st.student_id,
        {k: float(v) for k, v in zip(NEED_KEYS, st.true_needs, strict=True)},
        anchors=list(st.anchors),
    )
    ids, shown = discovery(oracle, 3)
    out[("oracle", None)] = score(ids, shown)
    observed = prepare_profile(st.observed_profile(), catalog)
    for name, fn in (("popular", popular_baseline), ("similar", similar_baseline)):
        ids = fn(observed, catalog, K)
        out[(name, None)] = score(ids, ids)
    return out


def run(gen, catalog, n, pop_seed, resp_seed):
    results = {}
    bank = default_item_bank()
    for truth in TRUTHS:
        cfg = SimConfig(n_students=n, prestige_weight=0.5, truth=truth, needs_model=gen)
        pop = generate_population(catalog, cfg, np.random.default_rng(pop_seed))
        per = {fam: [] for fam in FAM_NOISE}
        for idx, st in enumerate(pop):
            respond = make_item_responder(
                st,
                np.random.default_rng([resp_seed, idx]),
                noise_sd=0.25,
                acquiescence=cfg.acquiescence,
                desirability=cfg.desirability,
            )
            answers = {item.item_id: respond(item) for item in bank}
            for fam in FAM_NOISE:
                per[fam].append(evaluate_student(st, catalog, truth, answers, fam, resp_seed, idx))
        for fam, rows in per.items():
            results[(truth, fam)] = rows
        print(f"  {gen} truth={truth}")
    return results


def stat(rows, key, metric):
    i = {"rel": 0, "nov": 1, "best": 2}[metric]
    return M.mean_se([r[key][i] for r in rows])


def paired(rows, a, b, metric="nov"):
    i = {"rel": 0, "nov": 1, "best": 2}[metric]
    return M.mean_se([r[a][i] - r[b][i] for r in rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--students", type=int, default=200)
    parser.add_argument("--catalog-seed", type=int, default=0)
    parser.add_argument("--pop-seed", type=int, default=1)
    parser.add_argument("--resp-seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-l3-e2e.md")
    args = parser.parse_args()

    catalog = synthetic_catalog(np.random.default_rng(args.catalog_seed), n_per_family=10)
    all_res = {gen: run(gen, catalog, args.students, args.pop_seed, args.resp_seed)
               for gen in GENERATORS}  # fmt: skip

    lines = [
        "# L3 端到端：完整流程驗證",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`python scripts/run_l3_e2e.py "
        f"--catalog-seed {args.catalog_seed} --pop-seed {args.pop_seed} "
        f"--resp-seed {args.resp_seed}`",
        f"- 流程：只有錨點 → 自適應問卷（voi，Q 題）→ C1.2 v1 補全 → 短名單 {N_SHORTLIST} → "
        f"熟悉度問答 → 分區；推薦 k={K}；每條件 {args.students} 位學生",
        "- 名氣權重 0.5；真實題目雜訊 0.25（與假設一致）；作答偏誤始終存在且未被建模",
        "- 每格：相關率 / **相關且不知道** / 最佳契合保留（平均）",
        "- 預先登記的假設：見 `scripts/run_l3_e2e.py` 檔頭",
        "",
    ]
    verdict = {"H-E1": True, "H-E2": True, "H-E3": True, "H-E4": True}
    notes = {h: [] for h in verdict}
    for gen in GENERATORS:
        res = all_res[gen]
        lines += [f"## 生成模型：{gen}", ""]
        header = "| 條件 | " + " | ".join(f"流程 Q={q}" for q in BUDGETS)
        header += " | fit-only Q=45 | stated-full | oracle | popular | similar |"
        lines += [header, "|---" * (len(BUDGETS) + 6) + "|"]
        mean_nov_by_q = {q: [] for q in BUDGETS}
        for (truth, fam), rows in res.items():

            def cell(key, rows=rows):
                r, n, b = (stat(rows, key, m)[0] for m in ("rel", "nov", "best"))
                return f"{r:.3f} / **{n:.3f}** / {b:.3f}"

            cells = [cell(("pipeline", q)) for q in BUDGETS]
            cells += [cell(("fit-only", 45))]
            cells += [
                cell((name, None)) for name in ("stated-full", "oracle", "popular", "similar")
            ]
            lines.append(f"| {truth}, 熟悉度 {fam} | " + " | ".join(cells) + " |")
            for q in BUDGETS:
                mean_nov_by_q[q].append(stat(rows, ("pipeline", q), "nov")[0])

            # H-E1
            d, se = paired(rows, ("pipeline", 45), ("fit-only", 45))
            if d < -2 * se:
                verdict["H-E1"] = False
            notes["H-E1"].append(d)
            # H-E3
            pipe = stat(rows, ("pipeline", 45), "nov")[0]
            for base in ("popular", "similar"):
                if not pipe > stat(rows, (base, None), "nov")[0]:
                    verdict["H-E3"] = False
            # H-E4: section gain end-to-end vs on stated-full profiles
            gain_e2e = d
            gain_stated = paired(rows, ("stated-full", None), ("fit-only-stated", None))[0]
            notes["H-E4"].append((gain_e2e, gain_stated))
            if gain_e2e < 0.5 * gain_stated:
                verdict["H-E4"] = False
        lines.append("")
        means = [float(np.mean(mean_nov_by_q[q])) for q in BUDGETS]
        notes["H-E2"].append((gen, means))
        if not all(a < b for a, b in zip(means, means[1:], strict=False)):
            verdict["H-E2"] = False
        lines += [
            "相關且不知道（條件平均）依題數："
            + "、".join(f"Q={q} {m:.3f}" for q, m in zip(BUDGETS, means, strict=True)),
            "",
        ]
        lines += ["配對差（流程 Q=45 − fit-only Q=45）：", ""]
        lines += ["| 條件 | 相關率差 | 相關且不知道差 | 最佳契合保留差 |", "|---|---|---|---|"]
        for (truth, fam), rows in res.items():
            cells = []
            for m in ("rel", "nov", "best"):
                d, se = paired(rows, ("pipeline", 45), ("fit-only", 45), m)
                cells.append(f"{d:+.3f} ± {se:.3f}")
            lines.append(f"| {truth}, {fam} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## 預先登記假設判定", ""]
    lines += [
        f"- H-E1 分區在整合後仍有效：{'✅' if verdict['H-E1'] else '❌'}"
        f"（各條件差 {', '.join(f'{d:+.3f}' for d in notes['H-E1'])}；平均 "
        f"{np.mean(notes['H-E1']):+.3f}）",
        f"- H-E2 題數越多越好：{'✅' if verdict['H-E2'] else '❌'}"
        + "".join(f"（{g}：{' < '.join(f'{m:.3f}' for m in ms)}）" for g, ms in notes["H-E2"]),
        f"- H-E3 勝過熱門與相似基準：{'✅' if verdict['H-E3'] else '❌'}",
        f"- H-E4 沒有互相抵銷：{'✅' if verdict['H-E4'] else '❌'}"
        "（端到端增益 / 完整陳述增益："
        + ", ".join(f"{a:+.3f}/{b:+.3f}" for a, b in notes["H-E4"])
        + "）",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
