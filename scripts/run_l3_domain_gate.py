"""L3 for the hybrid "v1 relevance + v2 domain gate" (W-16c, B-C2-8, B-C2-11).

World: the v1/v2 comparison world (simulation/world_v2.py, A-C5-12); truth = 51 v2
dims + 3 residual dims. Full pipeline for everyone (complete -> shortlist 25 ->
familiarity check 10% miss / 5% false alarm -> sections); familiarity answers use
common random numbers (one uniform per student and programme).

Methods:
    v1         15 v1 items answered directly, v1 catalogue view, DEFAULT_SCORER
    gate-off   51 v2 items; domain_gated(lam=0) = v1 scoring of the v1 projection of
               the v2 answers (ablation: separates "more items" from "the gate")
    G(tau,lam) 51 v2 items; domain_gated(tau, lam): v1 scoring minus
               lam * max(0, tau - a), a = programme-content weighted area interest
Candidates in tie-break order (least intervention first):
    tau in {0.2, 0.3, 0.4} x lam in {0.5, 2.0}
Every v2 method scores the SAME stated v2 answers.

Conditions per generator: truth kind {distance, cosine} x answer noise {0.15, 0.30};
each list scored under both truth layer weightings {dim, layer} (A-C5-13): 8 cells
per generator, 400 students per condition.
Metrics: nov = novel-relevant rate of the discoveries; best = best-fit retention;
mis = domain-mismatch rate = share of discoveries relevant under the v1-view truth
(15 dims + residual, unweighted) but not under the full truth of that weighting
(the B-C2-8 proxy used in F70); rel reported.

PRE-REGISTERED (written and committed before the first run, 2026-10-27):
  "Significantly" = paired difference beyond 2 SE.
  Eligible (per generator): in each of the 8 cells, nov and best ('all' row) not
    significantly below v1; and no supported (n >= 30) need-group or transition
    stratum has a nov difference vs v1 that is both significant and below -0.05
    (D12 with a practical margin; transitions watch B-C2-11).
  H-G1: mis significantly below v1 in all 8 cells.
  1. SELECT on ARCHETYPE (seeds 1300/1301/1302): among eligible candidates, the
     highest score = min over the 4 (kind x weighting) cells of the noise-averaged
     mis reduction (v1 - candidate). Ties within 0.001 -> earlier candidate. No
     eligible candidate -> no gate.
  2. VALIDATE on CONTINUOUS (same seeds): the chosen candidate is eligible and
     satisfies H-G1 on continuous, and satisfies H-G1 on archetype.
  3. CONFIRM on fresh seeds (--catalog-seed 1400 --pop-seed 1401 --seed 1402
     --confirm <choice>): eligibility and H-G1 on BOTH generators.
  Outcome: if 1-3 hold, the gate is documented as a v2 option for far-field
  protection. It does NOT change the default dimension set (v1 needs 15 items, the
  gate 51); a two-stage version and any default change need their own
  pre-registered run.
  Report only: gate-off vs v1, levels, strata.
  Disclosure: none of the seeds above has been used. A 40-student smoke run on
  seeds 8/9/10 checks only that the script runs; those seeds are not used.

    python scripts/run_l3_domain_gate.py
    python scripts/run_l3_domain_gate.py --catalog-seed 1400 --pop-seed 1401 \\
        --seed 1402 --confirm "G(0.3,2)" --out results/synthetic-l3-domain-gate-confirm.md
"""

from __future__ import annotations

import argparse
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

from aisc.contracts import StudentProfile
from aisc.dimensions import NEED_KEYS, NEED_KEYS_V2
from aisc.discovery import DEFAULT_SCORER, domain_gated
from aisc.pipeline import interactive_discovery
from aisc.simulation import metrics as M
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.fields import all_fields_catalog, macro_group
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.strata import ALL, format_cell, paired_stratified, stratified
from aisc.simulation.truth import relevant_ids
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
MARGIN = 0.05
CANDIDATES = {f"G({t:g},{lam:g})": domain_gated(t, lam)
              for t in (0.2, 0.3, 0.4) for lam in (0.5, 2.0)}  # fmt: skip
SCORERS = {"gate-off": domain_gated(0.3, 0.0), **CANDIDATES}
METHODS = ("v1", *SCORERS)
METRICS = ("nov", "rel", "best", "mis")


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
        rel_v1view = relevant_ids(w1, cat_v1, kind, 0.1, residual, r)
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
                                            k_known=5, scorer=SCORERS[m])  # fmt: skip
            disc = [x.program.program_id for x in res.discoveries]
            for wt in TRUTH_WEIGHTINGS:
                q = M.list_quality(disc, rel[wt], set(st.familiar))
                rows[wt][m]["nov"].append(q["novel_relevant_rate"])
                rows[wt][m]["rel"].append(q["relevant_rate"])
                rows[wt][m]["best"].append(M.best_fit_retained(res.all_ids, best[wt]))
                rows[wt][m]["mis"].append(
                    float("nan") if not disc
                    else sum(i in rel_v1view and i not in rel[wt] for i in disc) / len(disc)
                )  # fmt: skip
    return rows, groups, trans


def cells(gen_res):
    """(kind, weighting, noise) -> (rows for that weighting, groups, trans)."""
    return {(k, wt, nz): (rows[wt], g, t)
            for (k, nz), (rows, g, t) in gen_res.items() for wt in TRUTH_WEIGHTINGS}  # fmt: skip


def eligible(gen_res, cand):
    fails = []
    for key, (rows, g, t) in cells(gen_res).items():
        for metric in ("nov", "best"):
            if paired_stratified(rows[cand][metric], rows["v1"][metric], g)[ALL] \
                    .significantly_below():  # fmt: skip
                fails.append((*key, metric, ALL))
        for labels in (g, t):
            for s, d in paired_stratified(rows[cand]["nov"], rows["v1"]["nov"], labels).items():
                if s != ALL and d.significantly_below() and d.mean < -MARGIN:
                    fails.append((*key, "nov", s))
    return not fails, fails


def h_g1(gen_res, cand):
    fails = [key for key, (rows, g, _) in cells(gen_res).items()
             if not paired_stratified(rows[cand]["mis"], rows["v1"]["mis"], g)[ALL]
             .significantly_below()]  # fmt: skip
    return not fails, fails


def mis_score(gen_res, cand):
    """Min over (kind, weighting) of the noise-averaged mis reduction v1 - cand."""
    vals = []
    for kind, wt in product(KINDS, TRUTH_WEIGHTINGS):
        red = [paired_stratified(gen_res[(kind, nz)][0][wt]["v1"]["mis"],
                                 gen_res[(kind, nz)][0][wt][cand]["mis"],
                                 gen_res[(kind, nz)][1])[ALL].mean for nz in NOISES]  # fmt: skip
        vals.append((float(np.mean(red)), f"{kind}/{wt}"))
    return min(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=400)
    ap.add_argument("--catalog-seed", type=int, default=1300)
    ap.add_argument("--pop-seed", type=int, default=1301)
    ap.add_argument("--seed", type=int, default=1302)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "synthetic-l3-domain-gate.md")  # fmt: skip
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

    arch, cont = res["archetype"], res["continuous"]
    elig = {c: eligible(arch, c) for c in CANDIDATES}
    scores = {c: mis_score(arch, c) for c in CANDIDATES}
    ok = [c for c in CANDIDATES if elig[c][0]]
    if args.confirm:
        chosen = args.confirm
    elif ok:
        top = max(scores[c][0] for c in ok)
        chosen = next(c for c in ok if scores[c][0] >= top - 0.001)
    else:
        chosen = None
    checks = {}
    if chosen:
        checks = {
            "原型：資格": eligible(arch, chosen),
            "原型：H-G1": h_g1(arch, chosen),
            "連續：資格": eligible(cont, chosen),
            "連續：H-G1": h_g1(cont, chosen),
        }
    passed = bool(chosen) and all(v[0] for v in checks.values())

    cmd = (f"python scripts/run_l3_domain_gate.py --catalog-seed {args.catalog_seed} "
           f"--pop-seed {args.pop_seed} --seed {args.seed}"
           + (f' --confirm "{args.confirm}"' if args.confirm else ""))  # fmt: skip
    yn = {True: "✅", False: "❌"}
    L = [
        "# L3：v1 計分 ＋ v2 領域把關（W-16c 混合設計，B-C2-8）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`{cmd}`",
        f"- 目錄 {len(cat_v2)} 個學程；每條件 {args.students} 位學生；"
        "v1 直接答 15 題，其他方法答 51 題（同一份作答）",
        "- 每個生成模型 8 格 = 真相函數 2 × 作答雜訊 2（0.15、0.30）× 真相層權重 2（dim、layer）",
        "- mis = 領域不符率（v1 視角相關、完整真相不相關）；方括號 = 人數 < 30",
        "- 預先登記見 `scripts/run_l3_domain_gate.py` 檔頭",
        "",
        "## 判定",
        "",
        "| 候選 | 原型：資格 | 原型：mis 降幅 maximin | 最差處 | 連續：資格 | 連續：H-G1 |",
        "|---|---|---|---|---|---|",
    ]
    for c in CANDIDATES:
        L.append(f"| {c} | {yn[elig[c][0]]} | {scores[c][0]:+.4f} | {scores[c][1]} | "
                 f"{yn[eligible(cont, c)[0]]} | {yn[h_g1(cont, c)[0]]} |")  # fmt: skip
    L += ["", ("- 3 確認：檢查 " if args.confirm else "- 1 選擇（原型）：")
          + (f"**{chosen}**" if chosen else "**無合格候選 → 不加把關**")]  # fmt: skip
    for name, (good, fails) in checks.items():
        L.append(f"- {name}：{yn[good]}" + (f"（未通過：{fails}）" if not good else ""))
    L += [f"- 整體：{yn[passed]}", ""]
    for gen in GENERATORS:
        for (kind, wt, nz), (rows, g, t) in cells(res[gen]).items():
            L += [f"## {gen} / {kind} / {wt} / 雜訊 {nz}", "",
                  "| 方法 | 相關且不知道 | 相關率 | 最佳契合保留 | mis | 相關且不知道 − v1 "
                  "| 最佳契合 − v1 | mis − v1 |",
                  "|---|---|---|---|---|---|---|---|"]  # fmt: skip
            for m in METHODS:
                lv = {k: stratified(rows[m][k], g)[ALL] for k in METRICS}
                d = {k: paired_stratified(rows[m][k], rows["v1"][k], g)[ALL]
                     for k in ("nov", "best", "mis")}  # fmt: skip
                L.append(f"| {m} | {format_cell(lv['nov'])} | {format_cell(lv['rel'])} | "
                         f"{format_cell(lv['best'])} | {format_cell(lv['mis'])} | "
                         f"{format_cell(d['nov'], signed=True)} | "
                         f"{format_cell(d['best'], signed=True)} | "
                         f"{format_cell(d['mis'], signed=True)} |")  # fmt: skip
            for title, labels in (("需求大領域群", g), ("轉換類型", t)):
                tab = {m: paired_stratified(rows[m]["nov"], rows["v1"]["nov"], labels)
                       for m in SCORERS}  # fmt: skip
                names = [s for s in tab["gate-off"] if s != ALL]
                L += ["", f"相關且不知道 − v1，依{title}：", "",
                      "| 方法 | " + " | ".join(names) + " |",
                      "|---|" + "---|" * len(names)]  # fmt: skip
                for m in SCORERS:
                    row = " | ".join(format_cell(tab[m][s], signed=True) for s in names)
                    L.append(f"| {m} | {row} |")
            L.append("")
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"chosen={chosen} passed={passed} wrote {args.out}")


if __name__ == "__main__":
    main()
