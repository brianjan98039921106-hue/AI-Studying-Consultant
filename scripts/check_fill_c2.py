"""B-C1-14 check: does completing unanswered dimensions help C2's *cosine* ranking?

The C1 experiment (run_l3_c1.py) ranks estimated needs with the truth function.
This script instead ranks with C2's actual scorer (cosine relevance) and compares
three ways of handling unanswered dimensions: zero-fill (StudentProfile.vector's
behaviour before the fix), neutral 0.5, and C1.2 v1. Prints a Markdown table.

    python scripts/check_fill_c2.py > results/synthetic-l3-fill-check.md
"""

import numpy as np

from aisc.discovery import relevance
from aisc.profiling import blend_profile_v1, complete_neutral
from aisc.simulation import metrics as M
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.evaluate import mask_profile
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.truth import utilities

cat = synthetic_catalog(np.random.default_rng(0), n_per_family=10)
by_id = {p.program_id: p for p in cat}
methods = {
    "zero-fill (current bug)": lambda p: p,
    "neutral 0.5": complete_neutral,
    "v1 (confirmed)": lambda p: blend_profile_v1(p, by_id),
}
print("# B-C1-14：未作答維度的處理方式 × C2 cosine 排序\n")
print("每格：平均 τ（相對 zero-fill 的配對差 ± SE）；200 位學生；作答比例 = 已答維度占比\n")
print("| 生成模型 | 真相 | 作答比例 | " + " | ".join(methods) + " |")
print("|---|---|---|" + "---|" * len(methods))
for gen in ("archetype", "continuous"):
    for truth in ("distance", "cosine"):
        cfg = SimConfig(n_students=200, needs_model=gen, truth=truth)
        pop = generate_population(cat, cfg, np.random.default_rng(5))
        for cov in (1.0, 0.5, 0.25):
            taus = {m: [] for m in methods}
            for i, s in enumerate(pop):
                obs = mask_profile(s.observed_profile(), cov, np.random.default_rng([9, i]))
                true_u = utilities(s.true_needs, cat, truth)
                for m, f in methods.items():
                    v = f(obs).vector
                    c2 = [relevance(v, p.vector) for p in cat]  # C2's actual scorer
                    taus[m].append(M.kendall_tau_b(c2, true_u))
            base = np.array(taus["zero-fill (current bug)"])
            cells = []
            for m in methods:
                d, se = M.mean_se(list(np.array(taus[m]) - base))
                cells.append(f"{np.mean(taus[m]):.3f}（{d:+.3f} ± {se:.3f}）")
            print(f"| {gen} | {truth} | {cov} | " + " | ".join(cells) + " |")
