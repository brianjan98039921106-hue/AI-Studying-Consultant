"""L3 selection of the v2 scorer (W-16c, B-C2-10, B-C5-8).

World: the v1/v2 comparison world (simulation/world_v2.py, A-C5-12): all-fields
catalogue (180 programmes) and population, truth = 51 v2 dims + 3 residual dims.
Every student answers all 51 v2 items (v2-full) once per noise level; the SAME
stated profile is scored by every candidate, so candidates differ only in the scorer.
v1 baseline: 15 items (v1 projection), DEFAULT_SCORER. Full pipeline for everyone
(complete -> shortlist 25 -> familiarity check 10% miss / 5% false alarm -> sections).
Familiarity answers use common random numbers: one uniform per (student, programme).

Conditions per generator: truth kind {distance, cosine} x answer noise {0.15, 0.30};
every list is scored under BOTH truth layer weightings {dim, layer} (A-C5-13), giving
8 cells per generator. 400 students per condition.

Candidates (order = tie-break order):
    L25  layer-balanced, equal layer weights (current DEFAULT_SCORER_V2; reference R)
    U25  unlayered blend(0.25)  (the C2 v1 scorer)
    U0   unlayered cosine
    U50  unlayered blend(0.5)
    L50  layer-balanced, domain 0.50, other three layers 1/6 each
    L75  layer-balanced, domain 0.75, other three layers 1/12 each

PRE-REGISTERED (written and committed before the first run, 2026-10-27):
  Primary metric: novel-relevant rate of the discoveries. Guard: best-fit retention.
  "Significantly worse" = paired difference < -2 SE.
  1. SELECT on ARCHETYPE (seeds 1100/1101/1102). Eligible: never significantly worse
     than R on the primary or the guard metric ('all' row) in any of the 8 cells.
     Maximin score = minimum, over the 4 (truth kind x weighting) cells and over the
     need-group strata supported (n >= 30) at both noise levels, of the paired
     primary difference vs v1 averaged over the 2 noise levels (D12, A-C2-8,
     A-C2-9). Choose the eligible candidate with the highest score; ties within
     0.001 -> earlier candidate.
  2. VALIDATE on CONTINUOUS (same seeds): the chosen candidate is never
     significantly worse than R on primary or guard in any of the 8 cells.
  3. CONFIRM on fresh seeds (--catalog-seed 1200 --pop-seed 1201 --seed 1202):
     criterion 2 on BOTH generators.
  Outcome: if 1-3 hold and the chosen candidate is not R, it becomes
  DEFAULT_SCORER_V2; otherwise R stays and B-C2-10 stays open.
  NOT decided here: whether v2 may become the default dimension set. That needs a
  separate pre-registered rerun of scripts/run_l3_dims_v2.py with the chosen scorer
  under both truth weightings on fresh seeds.
  Report only: differences vs v1 by need group and transition, relevant rate.
  Disclosure: these candidates were motivated by the exploratory diagnostic (F74-F76,
  seeds 4242-4244); none of the seeds above has been used before. A 40-student smoke
  run on seeds 5/6/7 was made only to check that the script runs; its output was
  seen, so those seeds are not used.

    python scripts/run_l3_v2_scorer.py
    python scripts/run_l3_v2_scorer.py --catalog-seed 1200 --pop-seed 1201 --seed 1202 \\
        --out results/synthetic-l3-v2-scorer-confirm.md
"""

from __future__ import annotations

import argparse
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

from aisc.contracts import StudentProfile
from aisc.dimensions import NEED_KEYS, NEED_KEYS_V2
from aisc.discovery import DEFAULT_SCORER, DEFAULT_SCORER_V2, blend, layer_balanced
from aisc.pipeline import interactive_discovery
from aisc.simulation import metrics as M
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.fields import all_fields_catalog, macro_group
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.strata import ALL, format_cell, paired_stratified, stratified
from aisc.simulation.world_v2 import (
    TRUTH_WEIGHTINGS,
    best_fit_weighted,
    relevant_ids_weighted,
    stated,
    to_v1,
    v2_catalog,
    v2_students,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ("archetype", "continuous")
KINDS = ("distance", "cosine")
NOISES = (0.15, 0.30)
BIAS = 0.05
MISS, FALSE_ALARM = 0.10, 0.05
CANDIDATES = {
    "L25": DEFAULT_SCORER_V2,
    "U25": blend(0.25),
    "U0": blend(0.0),
    "U50": blend(0.5),
    "L50": layer_balanced({"interest": 1 / 6, "intensity": 1 / 6, "domain": 0.5,
                           "orientation": 1 / 6}),
    "L75": layer_balanced({"interest": 1 / 12, "intensity": 1 / 12, "domain": 0.75,
                           "orientation": 1 / 12}),
}  # fmt: skip
REF = "L25"
METHODS = ("v1", *CANDIDATES)
METRICS = ("nov", "rel", "best")


def run_condition(world, gen, kind, noise, n, pop_seed, seed):
    src, hidden, cat_v2, cat_v1, residual = world
    cfg = SimConfig(n_students=n, prestige_weight=0.5, truth=kind, needs_model=gen,
                    field_model="all_fields")  # fmt: skip
    pop = generate_population(src, cfg, np.random.default_rng(pop_seed), hidden=hidden)
    students = v2_students(pop, np.random.default_rng([pop_seed, 1]))
    ids_all = [p.program_id for p in cat_v2]
    rows = {wt: {m: {k: [] for k in METRICS} for m in METHODS} for wt in TRUTH_WEIGHTINGS}
    groups, trans = [], []
    for idx, (st, w2, r) in enumerate(students):
        groups.append(macro_group(st.natural_family))
        trans.append(st.transition)
        true2 = dict(zip(NEED_KEYS_V2, map(float, w2), strict=True))
        w1 = np.array([to_v1(true2)[k] for k in NEED_KEYS])
        rel = {wt: relevant_ids_weighted(w2, cat_v2, kind, residual, r, wt)
               for wt in TRUTH_WEIGHTINGS}  # fmt: skip
        best = {wt: best_fit_weighted(w2, cat_v2, set(st.anchors), kind, residual, r, wt)
                for wt in TRUTH_WEIGHTINGS}  # fmt: skip
        u = dict(zip(ids_all, np.random.default_rng([seed, idx, 7]).random(len(ids_all)),
                     strict=True))  # fmt: skip

        def ask(pids, st=st, u=u):
            return {pid: "heard" if (pid in st.familiar and u[pid] >= MISS)
                    or (pid not in st.familiar and u[pid] < FALSE_ALARM) else "unknown"
                    for pid in pids}  # fmt: skip

        needs1 = stated(w1, NEED_KEYS, np.random.default_rng([seed, idx, 1]), noise, BIAS)
        needs2 = stated(w2, NEED_KEYS_V2, np.random.default_rng([seed, idx, 2]), noise, BIAS)
        for m in METHODS:
            if m == "v1":
                prof = StudentProfile(st.student_id, needs1, {k: 0.3 for k in needs1},
                                      anchors=list(st.anchors))  # fmt: skip
                res = interactive_discovery(prof, cat_v1, ask, n_shortlist=25, k_new=10,
                                            k_known=5, scorer=DEFAULT_SCORER)  # fmt: skip
            else:
                prof = StudentProfile(st.student_id, dict(needs2), {k: 0.3 for k in needs2},
                                      anchors=list(st.anchors), dims="v2")  # fmt: skip
                res = interactive_discovery(prof, cat_v2, ask, n_shortlist=25, k_new=10,
                                            k_known=5, scorer=CANDIDATES[m])  # fmt: skip
            disc = [x.program.program_id for x in res.discoveries]
            for wt in TRUTH_WEIGHTINGS:
                q = M.list_quality(disc, rel[wt], set(st.familiar))
                rows[wt][m]["nov"].append(q["novel_relevant_rate"])
                rows[wt][m]["rel"].append(q["relevant_rate"])
                rows[wt][m]["best"].append(M.best_fit_retained(res.all_ids, best[wt]))
    return rows, groups, trans


def cells(gen_res):
    """(kind, weighting, noise) -> (rows for that weighting, groups, trans)."""
    return {(k, wt, nz): (rows[wt], g, t)
            for (k, nz), (rows, g, t) in gen_res.items() for wt in TRUTH_WEIGHTINGS}  # fmt: skip


def never_worse_than_ref(gen_res, cand):
    if cand == REF:
        return True, []
    fails = []
    for key, (rows, g, _) in cells(gen_res).items():
        for metric in ("nov", "best"):
            d = paired_stratified(rows[cand][metric], rows[REF][metric], g)[ALL]
            if d.significantly_below():
                fails.append((*key, metric))
    return not fails, fails


def maximin_score(gen_res, cand):
    """Min over (kind, weighting) and supported need groups of the noise-averaged
    paired primary difference vs v1. Returns (score, where)."""
    vals = []
    for kind, wt in product(KINDS, TRUTH_WEIGHTINGS):
        per_noise = []
        for nz in NOISES:
            rows_wt, g, _ = gen_res[(kind, nz)]
            rows = rows_wt[wt]
            per_noise.append(paired_stratified(rows[cand]["nov"], rows["v1"]["nov"], g))
        for stratum in per_noise[0]:
            if all(stratum in p and p[stratum].supported for p in per_noise):
                vals.append((float(np.mean([p[stratum].mean for p in per_noise])),
                             f"{kind}/{wt}/{stratum}"))  # fmt: skip
    return min(vals) if vals else (float("-inf"), "無足夠人數的層")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=400)
    ap.add_argument("--catalog-seed", type=int, default=1100)
    ap.add_argument("--pop-seed", type=int, default=1101)
    ap.add_argument("--seed", type=int, default=1102)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-l3-v2-scorer.md")
    ap.add_argument("--confirm", choices=list(CANDIDATES),
                    help="step 3: check this (run-1) choice instead of selecting")  # fmt: skip
    args = ap.parse_args()
    rng = np.random.default_rng(args.catalog_seed)
    src, hidden = all_fields_catalog(rng, synthetic_catalog(rng, n_per_family=10), 10)
    cat_v2, cat_v1, residual = v2_catalog(src, hidden, rng)
    world = (src, hidden, cat_v2, cat_v1, residual)

    res: dict[str, dict] = {g: {} for g in GENERATORS}
    for gen, kind, nz in product(GENERATORS, KINDS, NOISES):
        res[gen][(kind, nz)] = run_condition(world, gen, kind, nz, args.students,
                                             args.pop_seed, args.seed)  # fmt: skip
        print(f"  {gen} {kind} noise={nz}", flush=True)

    arch = res["archetype"]
    elig = {c: never_worse_than_ref(arch, c) for c in CANDIDATES}
    scores = {c: maximin_score(arch, c) for c in CANDIDATES}
    ok = [c for c in CANDIDATES if elig[c][0]]
    top = max(scores[c][0] for c in ok)
    chosen = args.confirm or next(c for c in ok if scores[c][0] >= top - 0.001)
    val_arch = never_worse_than_ref(arch, chosen)
    val_cont = never_worse_than_ref(res["continuous"], chosen)
    cont_scores = {c: maximin_score(res["continuous"], c) for c in CANDIDATES}

    cmd = (f"python scripts/run_l3_v2_scorer.py --catalog-seed {args.catalog_seed} "
           f"--pop-seed {args.pop_seed} --seed {args.seed}"
           + (f" --confirm {args.confirm}" if args.confirm else ""))  # fmt: skip
    yn = {True: "✅", False: "❌"}
    L = [
        "# L3：v2 計分器選擇（W-16c，B-C2-10）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`{cmd}`",
        f"- 目錄 {len(cat_v2)} 個學程；每條件 {args.students} 位學生；v2 學生 51 題全答",
        "- 每個生成模型 8 格 = 真相函數 2 × 作答雜訊 2（0.15、0.30）× 真相層權重 2（dim、layer）",
        f"- 參考 R = {REF}（現行 DEFAULT_SCORER_V2）；基準 = v1（15 題、DEFAULT_SCORER）",
        "- 方括號 = 人數 < 30；預先登記見 `scripts/run_l3_v2_scorer.py` 檔頭",
        "",
        "## 判定",
        "",
        "| 候選 | 原型：不劣於 R | 原型 maximin（vs v1） | 最差處 | 連續：不劣於 R "
        "| 連續 maximin（僅報告） |",
        "|---|---|---|---|---|---|",
    ]
    for c in CANDIDATES:
        L.append(f"| {c} | {yn[elig[c][0]]} | {scores[c][0]:+.3f} | {scores[c][1]} | "
                 f"{yn[never_worse_than_ref(res['continuous'], c)[0]]} | "
                 f"{cont_scores[c][0]:+.3f}（{cont_scores[c][1]}） |")  # fmt: skip
    L += [
        "",
        (
            f"- 3 確認：檢查第一次選出的 **{chosen}**（新種子，兩種生成模型都須不劣於 R）"
            if args.confirm
            else f"- 1 選擇（原型）：**{chosen}**"
        ),
        f"- 連續模型不劣於 R：{yn[val_cont[0]]}"
        + (f"（未通過：{val_cont[1]}）" if not val_cont[0] else ""),
        f"- 原型模型不劣於 R：{yn[val_arch[0]]}"
        + (f"（未通過：{val_arch[1]}）" if not val_arch[0] else ""),
        "",
    ]
    for gen in GENERATORS:
        for (kind, wt, nz), (rows, g, t) in cells(res[gen]).items():
            L += [f"## {gen} / {kind} / {wt} / 雜訊 {nz}", "",
                  "| 方法 | 相關且不知道 | 相關率 | 最佳契合保留 | 相關且不知道 − v1 | − R "
                  "| 最佳契合 − R |",
                  "|---|---|---|---|---|---|---|"]  # fmt: skip
            for m in METHODS:
                lv = {k: stratified(rows[m][k], g)[ALL] for k in METRICS}
                d1 = paired_stratified(rows[m]["nov"], rows["v1"]["nov"], g)[ALL]
                dr = paired_stratified(rows[m]["nov"], rows[REF]["nov"], g)[ALL]
                db = paired_stratified(rows[m]["best"], rows[REF]["best"], g)[ALL]
                L.append(f"| {m} | {format_cell(lv['nov'])} | {format_cell(lv['rel'])} | "
                         f"{format_cell(lv['best'])} | {format_cell(d1, signed=True)} | "
                         f"{format_cell(dr, signed=True)} | "
                         f"{format_cell(db, signed=True)} |")  # fmt: skip
            by_g = {m: paired_stratified(rows[m]["nov"], rows["v1"]["nov"], g) for m in CANDIDATES}
            by_t = {m: paired_stratified(rows[m]["nov"], rows["v1"]["nov"], t) for m in CANDIDATES}
            for title, tab in (("需求大領域群", by_g), ("轉換類型", by_t)):
                names = [s for s in tab[REF] if s != ALL]
                L += ["", f"相關且不知道 − v1，依{title}：", "",
                      "| 候選 | " + " | ".join(names) + " |",
                      "|---|" + "---|" * len(names)]  # fmt: skip
                for m in CANDIDATES:
                    row = " | ".join(format_cell(tab[m][s], signed=True) for s in names)
                    L.append(f"| {m} | {row} |")
            L.append("")
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"chosen={chosen} arch_ok={val_arch[0]} cont_ok={val_cont[0]} wrote {args.out}")


if __name__ == "__main__":
    main()
