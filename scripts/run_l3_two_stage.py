"""L3 pre-registered choice of the two-stage switching parameters m / q (W-66, B-C1-21).

DEFINITIONS (aisc.profiling.hierarchy.areas_to_ask, docs/dimensions-v2.md §2, A-C1-45)
  Stage 1 always asks the 10 non-domain items (RIASEC 6, intensity 2, orientation 2)
  and one item per area group (12 groups). Stage 2 order = the areas of the
  highest-rated group (areas_in_group order, cross-linked areas once), then of the
  next group, ... (ties -> lower group id).
    m  = number of top groups whose areas are all asked in stage 2 (q unset).
         The item count varies by student (groups have 1-10 areas).
    q  = fixed stage-2 budget: the first q areas of that order over all groups (m
         unset). q may cut a group partway, in registry order (A-C1-45).
  Unasked areas are completed by C1.2 (anchors + area prior) with either completion
  model: "hierarchy" (default prior, dimensions-v2 §3) or "correlated" (opt-in O*NET
  conditional prior, A-C1-25/26). Each completion model gets its own selection.

WORLD (simulation/world_v2.py, A-C5-12, A-C1-47): all-fields catalogue (180
programmes, 18 families x 10) and population, default transition mix (continuing 0.4,
adjacent 0.25, far 0.25, multi_degree 0.10; W-63 switchers with anchors from the old
field); truth = 51 v2 dims + 3 residual dims. 500 students per condition.
Answers: noise sd 0.15, acquiescence 0.05. Common random numbers: each student's 51
item answers and 12 group answers are drawn once and every questionnaire takes its
subset; familiarity answers use one uniform per (student, programme), 10% misses,
5% false alarms. Full pipeline (complete -> shortlist 25 -> familiarity -> sections).

Profiles per student (15):
    FULL               all 51 v2 items (reference; nothing to complete)
    {H,C}-q0           stage 1 only (no area item)
    {H,C}-q4/q8/q12    fixed stage-2 budget q
    {H,C}-m1/m2/m3     all areas of the top m groups
  H = hierarchy completion, C = correlated completion.
Cells per generator: truth kind {distance, cosine} x truth layer weighting {dim, layer}
(A-C5-13, B-C5-8) x v2 scorer {L25 = DEFAULT_SCORER_V2, U25 = unlayered blend(0.25)}
= 8. The scorer is unresolved (B-C2-10, W-76), so m / q must not depend on it.

METRICS (all paired per student against FULL, same cell)
  tau  V-C1-2: Kendall tau-b between the scorer's ranking of the whole catalogue from
       the completed profile and the true (weighted) utility ranking. PRIMARY.
  nov  novel-relevant rate of the 10 discoveries; rel = relevant rate of the same list
       (never reported apart, CLAUDE.md); best = best-fit retention (report only).
  items  items asked; saving = 1 - items / 51 (V-C1-3 analogue: saving >= 0.30).
  Strata (D12): transition {continuing, adjacent, far, multi_degree} and need group
  (macro group of the natural family); a stratum counts only with n >= 30.

PRE-REGISTERED (written and committed before the first real run, 2026-10-29)
  Non-inferiority (Piaggio et al. 2012, A-C1-46): margins d_tau = 0.02, d_list = 0.03.
  A candidate is ELIGIBLE on a generator if, in every one of its 8 cells:
    (i)   tau, 'all' row:          mean(dtau) - 2 SE >= -0.02;
    (ii)  tau, every supported transition and need-group stratum: mean(dtau) >= -0.02
          (worst stratum, D12);
    (iii) nov and rel, 'all' row and every supported stratum: NOT significantly below
          the margin, i.e. NOT mean(d) < -0.03 - 2 SE.
  1. SELECT on ARCHETYPE (seeds 6600/6601/6602), separately for H and for C: among
     the eligible candidates, the one with the fewest mean items (A-C1-48); ties within
     0.5 items -> higher worst-stratum mean dtau over the 8 cells; then grid order
     (q0, q4, q8, q12, m1, m2, m3). None eligible -> no choice for that family.
  2. VALIDATE on CONTINUOUS (same seeds): the chosen candidate is eligible there.
  3. CONFIRM on fresh seeds (--catalog-seed 6700 --pop-seed 6701 --seed 6702,
     --confirm-h / --confirm-c = the run-1 choices): eligible on BOTH generators.
  No fallback: if 2 or 3 fails, nothing is chosen for that family; another candidate
  can only be tried under a new pre-registration on new seeds.
  Report only (not a criterion): the reverse direction (choice on continuous,
  checked on archetype); V-C1-3 saving vs the 0.30 threshold; absolute tau vs the
  V-C1-2 threshold 0.7; best-fit retention; items by stratum.
  OUTCOME: if 1-3 hold for H, its m or q becomes the default of areas_to_ask
  (provisional: synthetic only, W-41 re-checks with student data); if 1-3 hold for
  C, it becomes the documented setting for area_model="correlated" (that completion
  model stays opt-in). v1 stays the default dimension set (W-16c, unaffected).
  Disclosure: before this was written, a 20-student timing check (seeds 3/4/5, m=2)
  printed only run times and transition counts. A smoke run of this script on seeds
  6690/6691/6692 (40 students) checks that it runs; its output is seen, so those
  seeds are not used again.

    PYTHONPATH=src python scripts/run_l3_two_stage.py
    PYTHONPATH=src python scripts/run_l3_two_stage.py --catalog-seed 6700 \\
        --pop-seed 6701 --seed 6702 --confirm-h <H choice> --confirm-c <C choice> \\
        --out results/synthetic-l3-two-stage-confirm.md
"""

from __future__ import annotations

import argparse
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

from aisc.contracts import StudentProfile
from aisc.dimensions import BASIC_INTEREST_KEYS, NEED_KEYS_V2
from aisc.discovery import DEFAULT_SCORER_V2, blend
from aisc.pipeline import interactive_discovery, prepare_profile
from aisc.profiling.hierarchy import GROUP_LABELS, areas_to_ask
from aisc.simulation import metrics as M
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.fields import all_fields_catalog, macro_group
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.strata import ALL, format_cell, paired_stratified, stratified
from aisc.simulation.world_v2 import (
    TRUTH_WEIGHTINGS,
    best_fit_weighted,
    group_answer,
    v2_catalog,
    v2_students,
    weighted_utilities,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ("archetype", "continuous")
KINDS = ("distance", "cosine")
SCORERS = {"L25": DEFAULT_SCORER_V2, "U25": blend(0.25)}
NOISE, BIAS = 0.15, 0.05
MISS, FALSE_ALARM = 0.10, 0.05
D_TAU, D_LIST = 0.02, 0.03
TIE_ITEMS = 0.5
SWITCHES = {"q0": (None, 0), "q4": (None, 4), "q8": (None, 8), "q12": (None, 12),
            "m1": (1, None), "m2": (2, None), "m3": (3, None)}  # fmt: skip
FAMILIES = {"H": "hierarchy", "C": "correlated"}
CANDS = {f: [f"{f}-{s}" for s in SWITCHES] for f in FAMILIES}
REF = "FULL"
PROFILES = (REF, *CANDS["H"], *CANDS["C"])
LIST_METRICS = ("nov", "rel", "best")
STRATA = ("trans", "group")
OTHER_KEYS = tuple(k for k in NEED_KEYS_V2 if k not in BASIC_INTEREST_KEYS)
GROUPS = tuple(sorted(GROUP_LABELS))


def tau_signs(v: np.ndarray, iu) -> np.ndarray:
    return np.sign(v[:, None] - v[None, :])[iu]


def tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau-b from pair signs; equals metrics.kendall_tau_b (checked in smoke)."""
    prod = a * b
    con, dis = int(np.sum(prod > 0)), int(np.sum(prod < 0))
    tx, ty = int(np.sum((a == 0) & (b != 0))), int(np.sum((b == 0) & (a != 0)))
    den = np.sqrt(float(con + dis + tx) * float(con + dis + ty))
    return float("nan") if den == 0 else (con - dis) / den


def questionnaires(st, w2, rng):
    """{profile name: (StudentProfile before completion, items asked, completion model)}."""
    ans = np.clip(w2 + BIAS + rng.normal(0, NOISE, len(w2)), 0, 1)
    full = {k: float(v) for k, v in zip(NEED_KEYS_V2, ans, strict=True)}
    true = dict(zip(NEED_KEYS_V2, map(float, w2), strict=True))
    g_true = np.array([group_answer(true, g) for g in GROUPS])
    g_ans = np.clip(g_true + BIAS + rng.normal(0, NOISE, len(GROUPS)), 0, 1)
    gs = {g: float(v) for g, v in zip(GROUPS, g_ans, strict=True)}

    def prof(needs, group_scores):
        return StudentProfile(st.student_id, needs, {k: 0.3 for k in needs},
                              anchors=list(st.anchors), dims="v2",
                              group_scores=group_scores)  # fmt: skip

    out = {REF: (prof(full, {}), len(NEED_KEYS_V2), "hierarchy")}
    for fam, model in FAMILIES.items():
        for s, (m, q) in SWITCHES.items():
            asked = areas_to_ask(gs, m=m, q=q)
            needs = {k: full[k] for k in OTHER_KEYS} | {a: full[a] for a in asked}
            n_items = len(OTHER_KEYS) + len(GROUPS) + len(asked)
            out[f"{fam}-{s}"] = (prof(needs, gs), n_items, model)
    return out


def run_condition(world, gen, kind, n, pop_seed, seed, check_tau=False):
    src, hidden, cat2, residual = world
    cfg = SimConfig(n_students=n, prestige_weight=0.5, truth=kind, needs_model=gen,
                    field_model="all_fields")  # fmt: skip
    pop = generate_population(src, cfg, np.random.default_rng(pop_seed), hidden=hidden)
    students = v2_students(pop, np.random.default_rng([pop_seed, 1]))
    X = np.array([p.vector for p in cat2])
    iu = np.triu_indices(len(cat2), 1)
    ids_all = [p.program_id for p in cat2]
    # rows[(weighting, scorer)][profile][metric] -> list over students
    rows = {(wt, sc): {p: {k: [] for k in ("tau", *LIST_METRICS)} for p in PROFILES}
            for wt in TRUTH_WEIGHTINGS for sc in SCORERS}  # fmt: skip
    items = {p: [] for p in PROFILES}
    labels = {"trans": [], "group": []}
    for idx, (st, w2, r) in enumerate(students):
        labels["trans"].append(st.transition)
        labels["group"].append(macro_group(st.natural_family))
        u = {wt: weighted_utilities(w2, cat2, kind, residual, r, wt) for wt in TRUTH_WEIGHTINGS}
        u_sign = {wt: tau_signs(u[wt], iu) for wt in TRUTH_WEIGHTINGS}
        n_rel = max(1, int(np.ceil(0.1 * len(cat2))))
        rel = {wt: {ids_all[i] for i in np.argsort(-u[wt], kind="stable")[:n_rel]}
               for wt in TRUTH_WEIGHTINGS}  # fmt: skip
        best = {wt: best_fit_weighted(w2, cat2, set(st.anchors), kind, residual, r, wt)
                for wt in TRUTH_WEIGHTINGS}  # fmt: skip
        fam_u = dict(zip(ids_all, np.random.default_rng([seed, idx, 7]).random(len(ids_all)),
                         strict=True))  # fmt: skip

        def ask(pids, st=st, fu=fam_u):
            return {pid: "heard" if (pid in st.familiar and fu[pid] >= MISS)
                    or (pid not in st.familiar and fu[pid] < FALSE_ALARM) else "unknown"
                    for pid in pids}  # fmt: skip

        for name, (raw, n_items, model) in questionnaires(
            st, w2, np.random.default_rng([seed, idx, 1])
        ).items():
            items[name].append(float(n_items))
            done = prepare_profile(raw, cat2, area_model=model)
            for sc_name, scorer in SCORERS.items():
                s_sign = tau_signs(scorer.score_matrix(done.vector, X), iu)
                res = interactive_discovery(done, cat2, ask, n_shortlist=25, k_new=10,
                                            k_known=5, scorer=scorer)  # fmt: skip
                disc = [x.program.program_id for x in res.discoveries]
                for wt in TRUTH_WEIGHTINGS:
                    t = tau_b(s_sign, u_sign[wt])
                    if check_tau and idx < 3:
                        ref = M.kendall_tau_b(scorer.score_matrix(done.vector, X), u[wt])
                        assert abs(t - ref) < 1e-12, (t, ref)
                    q = M.list_quality(disc, rel[wt], set(st.familiar))
                    row = rows[(wt, sc_name)][name]
                    row["tau"].append(t)
                    row["nov"].append(q["novel_relevant_rate"])
                    row["rel"].append(q["relevant_rate"])
                    row["best"].append(M.best_fit_retained(res.all_ids, best[wt]))
            if check_tau and idx < 3:  # completion inside interactive_discovery is a no-op
                assert np.allclose(prepare_profile(done, cat2).vector, done.vector)
    return rows, items, labels


def cells(gen_res):
    """(kind, weighting, scorer) -> (rows, labels)."""
    return {(kind, wt, sc): (rows[(wt, sc)], labels)
            for kind, (rows, _, labels) in gen_res.items()
            for wt in TRUTH_WEIGHTINGS for sc in SCORERS}  # fmt: skip


def diffs(rows, labels, cand, metric):
    """{stratum family: {stratum: StratumStat}} of the paired difference cand - FULL."""
    return {f: paired_stratified(rows[cand][metric], rows[REF][metric], labels[f])
            for f in STRATA}  # fmt: skip


def check(gen_res, cand):
    """(eligible, failures, worst-stratum mean dtau over cells)."""
    fails, worst = [], float("inf")
    for key, (rows, labels) in cells(gen_res).items():
        tag = "/".join(key)
        dt = diffs(rows, labels, cand, "tau")
        a = dt["trans"][ALL]
        if a.mean - 2 * a.se < -D_TAU:
            fails.append(f"{tag} tau all {a.mean:+.3f}±{a.se:.3f}")
        for f in STRATA:
            for s, stat in dt[f].items():
                if s != ALL and stat.supported:
                    worst = min(worst, stat.mean)
                    if stat.mean < -D_TAU:
                        fails.append(f"{tag} tau {s} {stat.mean:+.3f}")
        for metric in ("nov", "rel"):
            for f, tab in diffs(rows, labels, cand, metric).items():
                for s, stat in tab.items():
                    if (s != ALL or f == "trans") and stat.significantly_below(-D_LIST):
                        fails.append(f"{tag} {metric} {s} {stat.mean:+.3f}±{stat.se:.3f}")
    return not fails, fails, worst


def mean_items(gen_res, cand):
    return float(np.mean([np.mean(items[cand]) for _, items, _ in gen_res.values()]))


def select(gen_res, fam):
    info = {c: check(gen_res, c) for c in CANDS[fam]}
    ok = [c for c in CANDS[fam] if info[c][0]]
    if not ok:
        return None, info
    fewest = min(mean_items(gen_res, c) for c in ok)
    near = [c for c in ok if mean_items(gen_res, c) <= fewest + TIE_ITEMS]
    top = max(info[c][2] for c in near)
    return next(c for c in near if info[c][2] >= top), info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=500)
    ap.add_argument("--catalog-seed", type=int, default=6600)
    ap.add_argument("--pop-seed", type=int, default=6601)
    ap.add_argument("--seed", type=int, default=6602)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-l3-two-stage.md")
    ap.add_argument("--confirm-h", choices=CANDS["H"], help="step 3: run-1 choice for H")
    ap.add_argument("--confirm-c", choices=CANDS["C"], help="step 3: run-1 choice for C")
    ap.add_argument("--check-tau", action="store_true", help="smoke: check tau_b and no-op")
    args = ap.parse_args()
    confirming = args.confirm_h is not None or args.confirm_c is not None
    rng = np.random.default_rng(args.catalog_seed)
    src, hidden = all_fields_catalog(rng, synthetic_catalog(rng, n_per_family=10), 10)
    cat2, _, residual = v2_catalog(src, hidden, rng)
    world = (src, hidden, cat2, residual)

    res: dict[str, dict] = {g: {} for g in GENERATORS}
    for gen, kind in product(GENERATORS, KINDS):
        res[gen][kind] = run_condition(world, gen, kind, args.students, args.pop_seed,
                                       args.seed, args.check_tau)  # fmt: skip
        print(f"  {gen} {kind}", flush=True)

    arch, cont = res["archetype"], res["continuous"]
    info = {g: {c: check(res[g], c) for fam in FAMILIES for c in CANDS[fam]} for g in GENERATORS}
    chosen, reverse = {}, {}
    for fam in FAMILIES:
        chosen[fam] = select(arch, fam)[0]
        reverse[fam] = select(cont, fam)[0]
    if confirming:
        chosen = {"H": args.confirm_h, "C": args.confirm_c}

    cmd = (f"PYTHONPATH=src python scripts/run_l3_two_stage.py --students {args.students} "
           f"--catalog-seed {args.catalog_seed} --pop-seed {args.pop_seed} --seed {args.seed}"
           + (f" --confirm-h {args.confirm_h}" if args.confirm_h else "")
           + (f" --confirm-c {args.confirm_c}" if args.confirm_c else ""))  # fmt: skip
    yn = {True: "✅", False: "❌"}
    full_items = len(NEED_KEYS_V2)
    L = [
        "# L3：兩階段提問的切換參數 m／q（W-66，B-C1-21）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`{cmd}`",
        f"- 目錄 {len(cat2)} 個學程；每條件 {args.students} 位學生；作答雜訊 {NOISE}、"
        f"默許偏誤 {BIAS}；**合成資料**",
        "- 每個生成模型 8 格 = 真相函數 2 × 真相層權重 2（dim、layer）× v2 計分器 2"
        "（L25 = DEFAULT_SCORER_V2、U25 = 不分層 blend(0.25)）",
        f"- 比較基準 FULL = 51 題全答；非劣性界限 τ {D_TAU}、清單指標 {D_LIST}；"
        "方括號 = 人數 < 30（不列入判定）",
        "- 預先登記見 `scripts/run_l3_two_stage.py` 檔頭",
        "",
        "## 判定",
        "",
        "| 候選 | 平均題數（原型／連續） | 省題 | 原型合格 | 原型最差層 Δτ | 原型未過數 "
        "| 連續合格 | 連續最差層 Δτ | 連續未過數 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for fam in FAMILIES:
        for c in CANDS[fam]:
            ia, ic = mean_items(arch, c), mean_items(cont, c)
            a, b = info["archetype"][c], info["continuous"][c]
            L.append(f"| {c} | {ia:.1f}／{ic:.1f} | {1 - (ia + ic) / 2 / full_items:.1%} | "
                     f"{yn[a[0]]} | {a[2]:+.3f} | {len(a[1])} | {yn[b[0]]} | {b[2]:+.3f} | "
                     f"{len(b[1])} |")  # fmt: skip
    L.append("")
    for fam, model in FAMILIES.items():
        c = chosen[fam]
        L.append(f"### {fam}（{model} 補全）")
        L.append("")
        if c is None:
            L += ["- 1 選擇（原型）：**無合格候選**", ""]
        else:
            step = "3 確認（新種子）：檢查第一次選出的" if confirming else "1 選擇（原型）："
            ok_a, ok_c = info["archetype"][c][0], info["continuous"][c][0]
            which = "步驟 3" if confirming else "步驟 2 驗證"
            L += [f"- {step} **{c}**",
                  f"- 原型合格：{yn[ok_a]}；連續合格（{which}）：{yn[ok_c]}"]  # fmt: skip
        L.append(f"- 僅報告：反方向（連續挑選）會選 **{reverse[fam]}**")
        L.append("")
    L += ["## 未通過明細（每個候選最多列 12 項）", ""]
    for g in GENERATORS:
        for fam in FAMILIES:
            for c in CANDS[fam]:
                f = info[g][c][1]
                if f:
                    L.append(f"- {g} {c}（{len(f)} 項）：" + "；".join(f[:12]))
    L.append("")
    L += ["## 題數依轉換類型（平均）", "",
          "| 生成 | 候選 | " + " | ".join(("continuing", "adjacent", "far", "multi_degree")) + " |",
          "|---|---|---|---|---|---|"]  # fmt: skip
    for g in GENERATORS:
        for c in PROFILES:
            vals, labs = [], []
            for _, items, labels in res[g].values():
                vals += items[c]
                labs += labels["trans"]
            st = stratified(vals, labs)
            cells_txt = " | ".join(
                f"{st[t].mean:.1f} (n={st[t].n})" if t in st else "—"
                for t in ("continuing", "adjacent", "far", "multi_degree"))  # fmt: skip
            L.append(f"| {g} | {c} | {cells_txt} |")
    L.append("")
    for g in GENERATORS:
        for (kind, wt, sc), (rows, labels) in cells(res[g]).items():
            L += [f"## {g} / {kind} / {wt} / {sc}", "",
                  "| 方法 | τ | Δτ | 相關且不知道 | 相關率 | 最佳契合保留 | Δ相關且不知道 "
                  "| Δ相關率 | Δ最佳契合 |",
                  "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
            for p in PROFILES:
                lv = {k: stratified(rows[p][k], labels["trans"])[ALL]
                      for k in ("tau", *LIST_METRICS)}  # fmt: skip
                d = {k: diffs(rows, labels, p, k)["trans"][ALL] for k in ("tau", *LIST_METRICS)}
                L.append(f"| {p} | {format_cell(lv['tau'])} | {format_cell(d['tau'], signed=True)}"
                         f" | {format_cell(lv['nov'])} | {format_cell(lv['rel'])} | "
                         f"{format_cell(lv['best'])} | {format_cell(d['nov'], signed=True)} | "
                         f"{format_cell(d['rel'], signed=True)} | "
                         f"{format_cell(d['best'], signed=True)} |")  # fmt: skip
            for metric, title in (("tau", "Δτ"), ("nov", "Δ相關且不知道"), ("rel", "Δ相關率")):
                for f, fname in (("trans", "轉換類型"), ("group", "需求大領域群")):
                    tab = {p: diffs(rows, labels, p, metric)[f] for p in PROFILES[1:]}
                    names = [s for s in tab[PROFILES[1]] if s != ALL]
                    L += ["", f"{title}（− FULL），依{fname}：", "",
                          "| 候選 | " + " | ".join(names) + " |",
                          "|---|" + "---|" * len(names)]  # fmt: skip
                    for p in PROFILES[1:]:
                        L.append(f"| {p} | "
                                 + " | ".join(format_cell(tab[p][s], signed=True) for s in names)
                                 + " |")  # fmt: skip
            L.append("")
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    ok = {g: {k: info[g][v][0] for k, v in chosen.items() if v} for g in GENERATORS}
    print(f"chosen={chosen} reverse={reverse} arch_ok={ok['archetype']} "
          f"cont_ok={ok['continuous']} wrote {args.out}")  # fmt: skip


if __name__ == "__main__":
    main()
