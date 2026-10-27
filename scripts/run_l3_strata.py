"""Stratified L3 (W-68, D12): does C2 v1 hold for students who switch fields?

Per student: complete stated needs -> C2 v1 sections (shortlist 25 -> C1.4
familiarity check -> discoveries) versus fit-only (top-k by DEFAULT_SCORER, no
sections) on the same profile. Strata: 'continuing' (undergraduate background =
the family the true needs point to) and 'switching' (any other background).
Conditions per generator: 2 truths x familiarity noise {perfect, 10% miss /
5% false alarm} = 4; 400 students per condition; prestige weight 0.5.

PRE-REGISTERED (written before the first run, 2026-10-27):
  H-S1 (primary, D12 check of F11): in every condition and in every supported
       stratum (n >= 30), the paired difference sections - fit-only in
       novel-relevant rate is not significantly negative (> -2 SE).
  Report only: per-stratum relevant rate, novel-relevant rate and best-fit
       retention for both methods; stratum sizes; the worst stratum per metric.
  Replication: rerun with --catalog-seed 600 --pop-seed 601 --seed 602; H-S1
       must hold in both runs to be reported as confirmed.

    python scripts/run_l3_strata.py
    python scripts/run_l3_strata.py --catalog-seed 600 --pop-seed 601 --seed 602 \\
        --out results/synthetic-l3-strata-confirm.md
"""

from __future__ import annotations

import argparse
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

from aisc.discovery import shortlist
from aisc.profiling import complete_neutral
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.evaluate import c2_v1_pipeline, evaluate_lists_per_student
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.strata import ALL, format_cell, paired_stratified, stratified, worst_stratum

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ("archetype", "continuous")
TRUTHS = ("distance", "cosine")
FAM_NOISE = {"perfect": (0.0, 0.0), "10/5": (0.10, 0.05)}
METRICS = ("novel_relevant_rate", "relevant_rate", "best_fit_retained")
K = 10


def fit_only(profile, catalog, k):
    return shortlist(complete_neutral(profile), catalog, n=k)


def run_condition(catalog, gen, truth, noise, n, pop_seed, seed):
    cfg = SimConfig(n_students=n, prestige_weight=0.5, truth=truth, needs_model=gen)
    pop = generate_population(catalog, cfg, np.random.default_rng(pop_seed))
    labels = [st.background_stratum for st in pop]
    recs = {"sections": c2_v1_pipeline(), "fit-only": fit_only}
    per = evaluate_lists_per_student(
        pop, catalog, recs, k=K, truth=truth,
        familiarity_noise={"sections": FAM_NOISE[noise]}, seed=seed,
    )  # fmt: skip
    levels = {m: {r: stratified(per[r][m], labels) for r in recs} for m in METRICS}
    diffs = {m: paired_stratified(per["sections"][m], per["fit-only"][m], labels) for m in METRICS}
    return levels, diffs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=400)
    ap.add_argument("--catalog-seed", type=int, default=0)
    ap.add_argument("--pop-seed", type=int, default=41)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-l3-strata.md")
    args = ap.parse_args()
    catalog = synthetic_catalog(np.random.default_rng(args.catalog_seed), n_per_family=10)

    res = {}
    for gen, truth, noise in product(GENERATORS, TRUTHS, FAM_NOISE):
        res[(gen, truth, noise)] = run_condition(
            catalog, gen, truth, noise, args.students, args.pop_seed, args.seed
        )
        print(f"  {gen} {truth} {noise}", flush=True)

    fails = [
        (cond, s)
        for cond, (_, diffs) in res.items()
        for s, st in diffs["novel_relevant_rate"].items()
        if s != ALL and st.significantly_below()
    ]
    cmd = (f"python scripts/run_l3_strata.py --catalog-seed {args.catalog_seed} "
           f"--pop-seed {args.pop_seed} --seed {args.seed}")  # fmt: skip
    L = [
        "# L3 分層：C2 v1 對轉換領域的學生是否成立（W-68，D12）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`{cmd}`",
        f"- 每條件 {args.students} 位學生；完整陳述需求；"
        "分區（短名單 25 → 熟悉度 → 發現區）vs 只依契合",
        "- 分層：continuing（大學背景 = 需求指向的族群）／switching（其他）；"
        "方括號 = 人數 < 30，不納入判定",
        "- 預先登記：見 `scripts/run_l3_strata.py` 檔頭",
        "",
        f"## 判定：H-S1 {'✅ 成立' if not fails else '❌ 不成立'}",
        "",
    ]
    if fails:
        L += [f"- 顯著較差：{c} 的 {s}" for c, s in fails] + [""]
    L += ["## 分區 − 只依契合（配對差）", ""]
    for m in METRICS:
        L += [
            f"### {m}",
            "",
            "| 條件 | all | continuing | switching | 最差層 |",
            "|---|---|---|---|---|",
        ]
        for cond, (_, diffs) in res.items():
            d = diffs[m]
            cells = [format_cell(d[s], signed=True) if s in d else "—"
                     for s in (ALL, "continuing", "switching")]  # fmt: skip
            L.append(f"| {' / '.join(cond)} | " + " | ".join(cells) + f" | {worst_stratum(d)} |")
        L.append("")
    L += ["## 各方法的水準（只報告）", ""]
    for m in METRICS:
        L += [
            f"### {m}",
            "",
            "| 條件 | 方法 | all | continuing | switching |",
            "|---|---|---|---|---|",
        ]
        for cond, (levels, _) in res.items():
            for r, st in levels[m].items():
                cells = [format_cell(st[s]) if s in st else "—"
                         for s in (ALL, "continuing", "switching")]  # fmt: skip
                L.append(f"| {' / '.join(cond)} | {r} | " + " | ".join(cells) + " |")
        L.append("")
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"H-S1 {'holds' if not fails else 'FAILS'} -> {args.out}")


if __name__ == "__main__":
    main()
