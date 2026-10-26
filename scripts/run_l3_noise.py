"""L3 for C1.3 v2 item-noise models (B-C1-5, B-C1-16).

PRE-REGISTERED RULES (written before the first L3 run, 2026-10-26):
  Disclosure: exploratory runs (catalog seed 0, n_per_family=2, population seed 4,
  150 students) informed adding the "censored" model and reporting HPD intervals
  for it. Those runs are not evidence; every run of this script uses other seeds.

  Candidates: estimated(df0) and censored(df0) for df0 in {2, 4, 8}.
  Reference: fixed (v1, assumed item noise 0.25).

  Part A - calibration (primary). Neutral prior (complete_neutral), strategy
  "variance", all 45 items, response biases OFF. True item noise per condition:
  0.15, 0.25, 0.40, and "mixed" (per student U(0.15, 0.40)). Metric: share of
  dimensions whose stated 90% interval covers the true need (Gaussian models:
  mean +/- 1.645 sd; censored: highest-density interval).
  Coverage passes when it lies in [0.85, 0.95] in every noise condition.
  Reported only: biases ON; noise varying by dimension within a student
  (violates A-C1-15); coverage of edge dimensions (true need < 0.15 or > 0.85).

  AMENDMENT (2026-10-26, before any output of this script was seen; run 1 was
  stopped with no results written): exploratory diagnostics on the exploratory
  seeds showed that under the neutral prior N(0.5, 0.3^2), dimensions with true
  need near 0 or 1 are under-covered even with the true noise and the censored
  likelihood (archetype, noise 0.40: edge coverage 0.747), and that this goes away
  with a wider prior (sd 0.5: 0.89). That is prior misspecification (B-C1-17), not
  item noise. To isolate the noise model, the PRIMARY Part A prior is now
  population-matched: per-dimension mean and sd of the true needs of an
  independent draw of 2000 students from the same generator (population seed +
  1000), as in simulation-based calibration (Talts et al. 2018). The neutral
  prior is reported only. All other rules are unchanged.

  Part B - ranking guard. Pipeline prior (C1.2 v1 from anchors only), strategy
  "voi", DEFAULT_SCORER, biases ON, 2 truths x true noise {0.25, 0.40}. Metric:
  Kendall tau between C2's ranking of the posterior mean and the true-utility
  ranking at Q=15 and Q=45, paired per student against "fixed". The guard passes
  when the paired difference is never < -2 SE.

  1. SELECT on ARCHETYPE: eligible = passes Part A and Part B; choose the
     candidate with the smallest max |coverage - 0.90| over the four Part A
     conditions. No eligible candidate -> the default stays "fixed".
  2. VALIDATE the chosen candidate on CONTINUOUS (held out): Part A and Part B.
  3. CONFIRM on fresh seeds (--catalog-seed 300 --pop-seed 301 --resp-seed 302):
     re-run steps 1-2; the same candidate must be chosen (or another candidate
     that also passes both generators, reported as such) and pass both.

  RUN 1 OUTCOME (seeds 0/31/32): no candidate eligible -> default stays "fixed".
  censored(df0) passed calibration on both generators but failed the ranking
  guard on archetype at Q=15 (B-C1-18); estimated(df0) failed calibration.

  RUN 2 (pre-registered 2026-10-26, after run 1, before any run-2 output;
  --protocol run2). New candidates: censored_within(df0, rms) for df0 in
  {4, 8, 16}: sigma learned only from within-dimension repeats, sigma prior with
  root-mean-square 0.25. Reported comparators: fixed, censored(4) from run 1.
  Exploratory diagnostics behind these candidates used the exploratory seeds
  only. Rules 1-2 unchanged (Part A with the population-matched prior, Part B
  guard vs fixed). Because run-1 held-out results were seen, run 2 uses fresh
  seeds for selection and validation (--catalog-seed 400 --pop-seed 401
  --resp-seed 402) and rule 3 confirms on further fresh seeds
  (--catalog-seed 500 --pop-seed 501 --resp-seed 502).

    python scripts/run_l3_noise.py
    python scripts/run_l3_noise.py --protocol run2 --catalog-seed 400 --pop-seed 401 \
        --resp-seed 402 --out results/synthetic-l3-noise-run2.md
    python scripts/run_l3_noise.py --protocol run2 --catalog-seed 500 --pop-seed 501 \
        --resp-seed 502 --out results/synthetic-l3-noise-run2-confirm.md
    python scripts/run_l3_noise.py --catalog-seed 300 --pop-seed 301 --resp-seed 302 \\
        --out results/synthetic-l3-noise-confirm.md
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

from aisc.contracts import StudentProfile
from aisc.dimensions import NEED_KEYS
from aisc.discovery import DEFAULT_SCORER
from aisc.pipeline import prepare_profile
from aisc.profiling import AdaptiveSession, complete_neutral, default_item_bank
from aisc.simulation import metrics as M
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.truth import utilities

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ("archetype", "continuous")
NOISE_CONDS = ("0.15", "0.25", "0.40", "mixed")
REPORT_CONDS = ("per-dim",)
TRUTHS = ("distance", "cosine")
GUARD_NOISES = (0.25, 0.40)
GUARD_Q = (15, 45)
ASSUMED = 0.25
RUN1_MODELS = [("fixed", 4.0, "scale")] + [
    (m, d, "scale") for m in ("estimated", "censored") for d in (2.0, 4.0, 8.0)
]
RUN2_MODELS = [("fixed", 4.0, "scale"), ("censored", 4.0, "scale")] + [
    ("censored_within", d, "rms") for d in (4.0, 8.0, 16.0)
]
PROTOCOLS = {  # (models, candidates = indices eligible for selection)
    "run1": (RUN1_MODELS, list(range(1, len(RUN1_MODELS)))),
    "run2": (RUN2_MODELS, [2, 3, 4]),
}
LO, HI = 0.85, 0.95


def name(model) -> str:
    m, d, c = model
    if m == "fixed":
        return m
    return f"{m}({d:g}{', rms' if c == 'rms' else ''})"


def responder(st, rng, noise, acq=0.0, desir=None):
    """noise: scalar, or one sd per dimension (per-dim condition)."""
    sd = np.broadcast_to(np.asarray(noise, float), (len(NEED_KEYS),))
    shift = np.array([(desir or {}).get(k, 0.0) for k in NEED_KEYS]) + acq

    def answer(item):
        d = NEED_KEYS.index(item.dimension)
        return float(np.clip(st.true_needs[d] + shift[d] + rng.normal(0.0, sd[d]), 0.0, 1.0))

    return answer


def noise_for(cond, rng):
    if cond == "mixed":
        return rng.uniform(0.15, 0.40)
    if cond == "per-dim":
        return rng.uniform(0.15, 0.40, size=len(NEED_KEYS))
    return float(cond)


def population_prior(catalog, gen, pop_seed) -> StudentProfile:
    """Moment-matched normal prior from an independent population draw (amendment)."""
    cfg = SimConfig(n_students=2000, needs_model=gen)
    W = np.array([st.true_needs for st in generate_population(
        catalog, cfg, np.random.default_rng(pop_seed + 1000))])  # fmt: skip
    return StudentProfile(
        "population",
        dict(zip(NEED_KEYS, map(float, W.mean(0)), strict=True)),
        uncertainty=dict(zip(NEED_KEYS, map(float, W.std(0)), strict=True)),
    )


def calibration(catalog, gen, cond, n, seeds, biases, prior_kind="population", models=()):
    cfg = SimConfig(n_students=n, needs_model=gen)
    pop = generate_population(catalog, cfg, np.random.default_rng(seeds[0]))
    bank = default_item_bank()
    if prior_kind == "population":
        prior = population_prior(catalog, gen, seeds[0])
    else:
        prior = complete_neutral(StudentProfile("s", {}))
    cov = {name(m): [] for m in models}
    edge = {name(m): [] for m in models}
    sig = {name(m): [] for m in models}
    for idx, st in enumerate(pop):
        rng = np.random.default_rng([seeds[1], idx])
        ans = responder(
            st,
            rng,
            noise_for(cond, rng),
            cfg.acquiescence if biases else 0.0,
            cfg.desirability if biases else None,
        )
        answers = {it.item_id: ans(it) for it in bank}  # common random numbers
        e = (st.true_needs < 0.15) | (st.true_needs > 0.85)
        for m in models:
            s = AdaptiveSession(
                prior, catalog, strategy="variance", item_noise_sd=ASSUMED,
                noise_model=m[0], noise_prior_df=m[1], noise_prior_center=m[2],
            )  # fmt: skip
            while not s.done():
                it = s.next_item()
                s.record(it, answers[it.item_id])
            lo, hi = s.interval(0.90)
            hit = (lo <= st.true_needs) & (st.true_needs <= hi)
            cov[name(m)].append(hit.mean())
            edge[name(m)] += list(hit[e])
            sig[name(m)].append(s.noise_sd)
    return {
        k: {"cov": M.mean_se(cov[k]), "edge": M.mean_se(edge[k]), "sigma": float(np.mean(sig[k]))}
        for k in cov
    }


def guard(catalog, gen, truth, noise, n, seeds, models=()):
    cfg = SimConfig(n_students=n, prestige_weight=0.5, truth=truth, needs_model=gen)
    pop = generate_population(catalog, cfg, np.random.default_rng(seeds[0]))
    X = np.array([p.vector for p in catalog])
    bank = default_item_bank()
    taus = {name(m): {q: [] for q in GUARD_Q} for m in models}
    for idx, st in enumerate(pop):
        ans = responder(
            st, np.random.default_rng([seeds[1], idx]), noise, cfg.acquiescence, cfg.desirability
        )
        answers = {it.item_id: ans(it) for it in bank}
        prior = prepare_profile(
            StudentProfile(st.student_id, {}, anchors=list(st.anchors)), catalog
        )
        true_u = utilities(st.true_needs, catalog, truth)
        for m in models:
            s = AdaptiveSession(
                prior, catalog, strategy="voi", item_noise_sd=ASSUMED,
                score_fn=DEFAULT_SCORER.score_matrix, noise_model=m[0], noise_prior_df=m[1],
                noise_prior_center=m[2], rng=np.random.default_rng([seeds[1], idx, 7]),
            )  # fmt: skip
            for q in range(1, max(GUARD_Q) + 1):
                it = s.next_item()
                s.record(it, answers[it.item_id])
                if q in GUARD_Q:
                    tau = M.kendall_tau_b(DEFAULT_SCORER.score_matrix(s.mean, X), true_u)
                    taus[name(m)][q].append(tau)
    out = {}
    for k in taus:
        out[k] = {}
        for q in GUARD_Q:
            diff = np.array(taus[k][q]) - np.array(taus["fixed"][q])
            out[k][q] = {"tau": M.mean_se(taus[k][q]), "diff": M.mean_se(list(diff))}
    return out


def passes_cal(cal, k) -> bool:
    return all(LO <= cal[c][k]["cov"][0] <= HI for c in NOISE_CONDS)


def passes_guard(g, k) -> bool:
    return all(
        not (cell[k][q]["diff"][0] < -2 * cell[k][q]["diff"][1])
        for cell in g.values()
        for q in GUARD_Q
    )


def worst_gap(cal, k) -> float:
    return max(abs(cal[c][k]["cov"][0] - 0.90) for c in NOISE_CONDS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=300)
    ap.add_argument("--guard-students", type=int, default=200)
    ap.add_argument("--catalog-seed", type=int, default=0)
    ap.add_argument("--pop-seed", type=int, default=31)
    ap.add_argument("--resp-seed", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="run1")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-l3-noise.md")
    args = ap.parse_args()
    catalog = synthetic_catalog(np.random.default_rng(args.catalog_seed), n_per_family=10)
    seeds = (args.pop_seed, args.resp_seed)
    models, cand_idx = PROTOCOLS[args.protocol]
    jobs = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for gen in GENERATORS:
            for c in NOISE_CONDS + REPORT_CONDS:
                jobs[("cal", gen, c)] = ex.submit(
                    calibration, catalog, gen, c, args.students, seeds, False, "population", models
                )
            for c in NOISE_CONDS:
                jobs[("calb", gen, c)] = ex.submit(
                    calibration, catalog, gen, c, args.students, seeds, True, "population", models
                )
            for c in NOISE_CONDS:
                jobs[("caln", gen, c)] = ex.submit(
                    calibration, catalog, gen, c, args.students, seeds, False, "neutral", models
                )
            for t, z in product(TRUTHS, GUARD_NOISES):
                jobs[("gd", gen, (t, z))] = ex.submit(
                    guard, catalog, gen, t, z, args.guard_students, seeds, models
                )
        res = {key: f.result() for key, f in jobs.items()}
    cal = {g: {k[2]: v for k, v in res.items() if k[0] == "cal" and k[1] == g} for g in GENERATORS}
    cal_b = {g: {k[2]: v for k, v in res.items() if k[0] == "calb" and k[1] == g}
             for g in GENERATORS}  # fmt: skip
    cal_n = {g: {k[2]: v for k, v in res.items() if k[0] == "caln" and k[1] == g}
             for g in GENERATORS}  # fmt: skip
    gd = {g: {k[2]: v for k, v in res.items() if k[0] == "gd" and k[1] == g} for g in GENERATORS}

    cands = [name(models[i]) for i in cand_idx]
    ok = {
        g: {k: (passes_cal(cal[g], k), passes_guard(gd[g], k)) for k in cands} for g in GENERATORS
    }
    eligible = [k for k in cands if all(ok["archetype"][k])]
    chosen = min(eligible, key=lambda k: worst_gap(cal["archetype"], k)) if eligible else None
    validated = chosen is not None and all(ok["continuous"][chosen])

    cmd = (
        f"python scripts/run_l3_noise.py --catalog-seed {args.catalog_seed} "
        f"--pop-seed {args.pop_seed} --resp-seed {args.resp_seed} --protocol {args.protocol}"
    )
    L = [
        f"# L3：C1.3 v2 題目雜訊模型（B-C1-5、B-C1-16、B-C1-18）· {args.protocol}",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`{cmd}`",
        f"- 校準：每條件 {args.students} 位學生、母體配適先驗（修正案）、最大變異選題、45 題全答；"
        f"守門：每條件 {args.guard_students} 位學生、錨點先驗、voi 選題、預設計分器、作答偏誤開啟",
        "- 預先登記的規則：見 `scripts/run_l3_noise.py` 檔頭",
        "",
        "## 判定",
        "",
        "| 候選 | 原型：校準 | 原型：守門 | 原型最大偏離 | 連續：校準 | 連續：守門 |",
        "|---|---|---|---|---|---|",
    ]
    yn = {True: "✅", False: "❌"}
    for k in cands:
        a, c = ok["archetype"][k], ok["continuous"][k]
        L.append(
            f"| {k} | {yn[a[0]]} | {yn[a[1]]} | {worst_gap(cal['archetype'], k):.3f} "
            f"| {yn[c[0]]} | {yn[c[1]]} |"
        )
    L += [
        "",
        f"- 原型模型合格：{', '.join(eligible) or '無'}",
        f"- 選出：**{chosen or '無（維持 fixed）'}**",
        f"- 連續模型保留驗證：{'✅ 通過' if validated else '❌ 未通過' if chosen else '—'}",
        "",
    ]
    for gen in GENERATORS:
        role = "挑選" if gen == "archetype" else "保留驗證"
        L += [f"## 生成模型：{gen}（{role}）", ""]
        for title, table, conds in [
            (
                "校準：90% 區間涵蓋率（母體配適先驗、作答偏誤關閉；主要判定）",
                cal[gen],
                NOISE_CONDS + REPORT_CONDS,
            ),
            ("校準：作答偏誤開啟（只報告）", cal_b[gen], NOISE_CONDS),
            ("校準：中性先驗 N(0.5, 0.3²)（只報告；B-C1-17）", cal_n[gen], NOISE_CONDS),
        ]:
            L += [f"### {title}", "", "格式：涵蓋率 ± SE · 端點維度涵蓋率 · 平均 σ̂", ""]
            L += ["| 模型 | " + " | ".join(f"雜訊 {c}" for c in conds) + " |"]
            L += ["|---" * (len(conds) + 1) + "|"]
            for m in models:
                k = name(m)
                cells = [
                    f"{table[c][k]['cov'][0]:.3f}±{table[c][k]['cov'][1]:.3f} · "
                    f"{table[c][k]['edge'][0]:.3f} · {table[c][k]['sigma']:.3f}"
                    for c in conds
                ]
                L.append(f"| {k} | " + " | ".join(cells) + " |")
            L.append("")
        L += ["### 守門：排序 τ（與 fixed 配對差 ± SE；* = 顯著較差）", ""]
        cols = [(t, z, q) for (t, z) in gd[gen] for q in GUARD_Q]
        L += ["| 模型 | " + " | ".join(f"{t} σ={z} Q={q}" for t, z, q in cols) + " |"]
        L += ["|---" * (len(cols) + 1) + "|"]
        for m in models:
            k = name(m)
            cells = []
            for t, z, q in cols:
                cell = gd[gen][(t, z)][k][q]
                if k == "fixed":
                    cells.append(f"τ={cell['tau'][0]:.3f}")
                else:
                    d, se = cell["diff"]
                    star = "*" if d < -2 * se else ""
                    cells.append(f"{d:+.3f}±{se:.3f}{star}")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.append("")
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"chosen={chosen} validated={validated} -> {args.out}")


if __name__ == "__main__":
    main()
