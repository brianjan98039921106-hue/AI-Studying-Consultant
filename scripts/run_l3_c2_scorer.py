"""L3 for C2.1 scorer candidates (W-26).

Per student: complete stated-needs profile -> pipeline.interactive_discovery with
each scorer (shortlist 25, familiarity check with 10% miss / 5% false alarm,
sections). Familiarity answers use common random numbers: one uniform draw per
(student, program), shared by every scorer.

Conditions per generator: 2 truths x stated noise {0.15, 0.30} = 4; prestige
weight 0.5; 300 students per condition.

PRE-REGISTERED RULES (written before the first run, 2026-10-26):
  Primary metric: novel-relevant rate of the discoveries section.
  Guard metric: best-fit retention (both sections).
  Comparisons are paired against COSINE (v0); "significantly worse" = < -2 SE.

  1. SELECT on ARCHETYPE. Eligible: never significantly worse than cosine on the
     primary or the guard metric in any condition. Pick the eligible candidate with
     the largest MAXIMIN score = min over the two truths of the mean paired primary
     difference over that truth's conditions (A-C2-8). Ties within 0.001 -> the
     earlier candidate in CANDIDATES. If none is eligible, cosine stays.
  2. VALIDATE on CONTINUOUS: chosen candidate never significantly worse than cosine
     on primary or guard metric in any condition.
  3. CONFIRM on fresh seeds (--catalog-seed 400 --pop-seed 401 --resp-seed 402):
     same criterion as step 2 on BOTH generators.
  Oracle profiles (true needs) are reported only: under the distance truth any
  intensity-aware scorer improves by construction (B-C5-1), so they are not evidence.

    python scripts/run_l3_c2_scorer.py [--students 300]
    python scripts/run_l3_c2_scorer.py --catalog-seed 400 --pop-seed 401 --resp-seed 402 \\
        --out results/synthetic-l3-c2-scorer-confirm.md
"""

from __future__ import annotations

import argparse
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

from aisc.contracts import StudentProfile
from aisc.dimensions import NEED_KEYS
from aisc.discovery import COSINE, SQDIST, blend, layered
from aisc.pipeline import interactive_discovery
from aisc.simulation import metrics as M
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.truth import best_fit_id, relevant_ids

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ("archetype", "continuous")
TRUTHS = ("distance", "cosine")
NOISES = (0.15, 0.30)
MISS, FALSE_ALARM = 0.10, 0.05
K = 10
CANDIDATES = [layered(0.5), layered(1.0), layered(2.0), blend(0.25), blend(0.5), SQDIST]
SCORERS = [COSINE, *CANDIDATES]


def run_condition(catalog, gen, truth, noise, n, pop_seed, resp_seed, profile_kind):
    cfg = SimConfig(n_students=n, prestige_weight=0.5, truth=truth, noise_sd=noise, needs_model=gen)
    pop = generate_population(catalog, cfg, np.random.default_rng(pop_seed))
    rows = {s.name: {"nov": [], "rel": [], "best": []} for s in SCORERS}
    for idx, st in enumerate(pop):
        u = dict(
            zip(
                [p.program_id for p in catalog],
                np.random.default_rng([resp_seed, idx]).random(len(catalog)),
                strict=True,
            )
        )

        def ask(ids, st=st, u=u):
            return {
                pid: (
                    "heard"
                    if (pid in st.familiar and u[pid] >= MISS)
                    or (pid not in st.familiar and u[pid] < FALSE_ALARM)
                    else "unknown"
                )
                for pid in ids
            }

        if profile_kind == "stated":
            needs, sd = dict(st.stated_needs), st.stated_sigma
        else:
            needs, sd = {k: float(v) for k, v in zip(NEED_KEYS, st.true_needs, strict=True)}, 0.05
        profile = StudentProfile(
            st.student_id, needs, {k: sd for k in NEED_KEYS}, anchors=list(st.anchors)
        )
        rel = relevant_ids(st.true_needs, catalog, truth)
        best = best_fit_id(st.true_needs, catalog, set(st.anchors), truth)
        for scorer in SCORERS:
            res = interactive_discovery(profile, catalog, ask, k_new=K, scorer=scorer)
            ids = [r.program.program_id for r in res.discoveries]
            q = M.list_quality(ids, rel, set(st.familiar))
            rows[scorer.name]["nov"].append(q["novel_relevant_rate"])
            rows[scorer.name]["rel"].append(q["relevant_rate"])
            rows[scorer.name]["best"].append(M.best_fit_retained(res.all_ids, best))
    return rows


def paired(rows, name, metric):
    a, b = np.array(rows[name][metric]), np.array(rows[COSINE.name][metric])
    return M.mean_se(list(a - b))


def worse(rows, name) -> bool:
    for metric in ("nov", "best"):
        d, se = paired(rows, name, metric)
        if d < -2 * se:
            return True
    return False


def maximin(res, name) -> float:
    per_truth = []
    for truth in TRUTHS:
        per_truth.append(
            np.mean([paired(rows, name, "nov")[0] for (t, _), rows in res.items() if t == truth])
        )
    return float(min(per_truth))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--students", type=int, default=300)
    parser.add_argument("--catalog-seed", type=int, default=0)
    parser.add_argument("--pop-seed", type=int, default=1)
    parser.add_argument("--resp-seed", type=int, default=19)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-l3-c2-scorer.md")
    args = parser.parse_args()

    catalog = synthetic_catalog(np.random.default_rng(args.catalog_seed), n_per_family=10)
    results = {kind: {gen: {} for gen in GENERATORS} for kind in ("stated", "oracle")}
    for kind, gen, truth, noise in product(("stated", "oracle"), GENERATORS, TRUTHS, NOISES):
        if kind == "oracle" and noise != NOISES[0]:
            continue  # oracle ignores stated noise
        results[kind][gen][(truth, noise)] = run_condition(
            catalog, gen, truth, noise, args.students, args.pop_seed, args.resp_seed, kind
        )
        print(f"  {kind} {gen} truth={truth} noise={noise}")

    st = results["stated"]
    eligible = [
        c.name for c in CANDIDATES if not any(worse(r, c.name) for r in st["archetype"].values())
    ]
    chosen = None
    if eligible:
        best_val = max(maximin(st["archetype"], n) for n in eligible)
        chosen = next(n for n in eligible if maximin(st["archetype"], n) >= best_val - 0.001)

    lines = [
        "# L3：C2.1 計分器候選（W-26）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`python scripts/run_l3_c2_scorer.py "
        f"--catalog-seed {args.catalog_seed} --pop-seed {args.pop_seed} "
        f"--resp-seed {args.resp_seed}`",
        f"- 每條件 {args.students} 位學生；完整陳述需求 → 短名單 25 → 熟悉度（漏報 10%／誤報 5%，"
        "共同隨機數）→ 分區；k=10；名氣權重 0.5",
        "- 配對比較對象：COSINE（v0）。每格：平均配對差 ± SE；▲／▼ = 顯著較好／較差（> 2 SE）",
        "- 主要指標：相關且不知道；守門指標：最佳契合保留（兩區合計）；相關率只報告",
        "- 預先登記規則：見 `scripts/run_l3_c2_scorer.py` 檔頭",
        "",
    ]

    def fmt(d, se):
        mark = "▲" if d > 2 * se else ("▼" if d < -2 * se else "")
        return f"{d:+.3f}±{se:.3f}{mark}"

    for kind in ("stated", "oracle"):
        title = (
            "陳述需求（判定依據）"
            if kind == "stated"
            else "真實需求（只報告；distance 真相下為循環比較）"
        )
        lines += [f"## {title}", ""]
        for gen in GENERATORS:
            res = results[kind][gen]
            lines += [f"### 生成模型：{gen}", ""]
            conds = list(res)
            header = "| 計分器 | 指標 | " + " | ".join(f"{t}／雜訊 {n}" for t, n in conds) + " |"
            lines += [header, "|---|---|" + "---|" * len(conds)]
            base = COSINE.name
            for metric, label in (("nov", "相關且不知道"), ("rel", "相關率"), ("best", "最佳契合")):
                cells = [f"{np.mean(res[c][base][metric]):.3f}" for c in conds]
                lines.append(f"| cosine（基準值） | {label} | " + " | ".join(cells) + " |")
            for c in CANDIDATES:
                for metric, label in (("nov", "相關且不知道"), ("best", "最佳契合")):
                    cells = [fmt(*paired(res[cond], c.name, metric)) for cond in conds]
                    lines.append(f"| {c.name} | {label} | " + " | ".join(cells) + " |")
            lines.append("")
            if kind == "stated":
                lines += [
                    "| 計分器 | 有任一條件顯著較差 | maximin（兩真相中較差者的平均主要指標差） |",
                    "|---|---|---|",
                ]
                for c in CANDIDATES:
                    w = any(worse(r, c.name) for r in res.values())
                    lines.append(
                        f"| {c.name} | {'是' if w else '否'} | {maximin(res, c.name):+.4f} |"
                    )
                lines.append("")

    lines += ["## 挑選與驗證（預先登記）", ""]
    if chosen is None:
        lines += ["**原型模型上沒有候選符合條件，預設維持 cosine。**", ""]
    else:
        ok_cont = not any(worse(r, chosen) for r in st["continuous"].values())
        lines += [
            f"- 原型模型合格候選：{', '.join(eligible)}",
            f"- 選出（maximin）：**{chosen}**（{maximin(st['archetype'], chosen):+.4f}）",
            f"- 連續模型保留驗證：{'✅ 通過' if ok_cont else '❌ 未通過'}"
            f"（maximin {maximin(st['continuous'], chosen):+.4f}）",
            "",
        ]
        both = all(not any(worse(r, chosen) for r in st[g].values()) for g in GENERATORS)
        lines += [f"- 兩個生成模型都無顯著較差（新種子確認時的判定）：{'✅' if both else '❌'}", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
