"""Descriptive (report-only, exploratory) comparison for B-C5-3: is the synthetic
catalogue more separable than real programme profiles?

Only the 6 RIASEC dimensions exist on both sides.
  synthetic  data/synthetic/programs.jsonl and the C5 generator's default catalogue
             (synthetic_catalog, 10 per family, seed 0); groups = prototype family
  real       all-fields non-academic CIP profiles (occupation_profiles_all_nonacademic);
             one unit per CIP (programmes within a CIP are identical, B-C0-1);
             groups = 4-digit CIP group, and 2-digit series
Statistics: pooled within-group SD, SD of group means (between), their ratio, and the
share of units whose nearest neighbour (Euclidean, RIASEC) is in the same group.
No decision depends on this; it informs the simulator's within-family noise (B-C5-2,
B-C5-3).

    python scripts/describe_catalog_spread.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from aisc.simulation.catalog_gen import synthetic_catalog

ROOT = Path(__file__).resolve().parents[1]
RIASEC = ["riasec_r", "riasec_i", "riasec_a", "riasec_s", "riasec_e", "riasec_c"]


def stats(X: np.ndarray, groups: list[str]) -> dict[str, float]:
    by = defaultdict(list)
    for x, g in zip(X, groups, strict=True):
        by[g].append(x)
    multi = {g: np.array(v) for g, v in by.items() if len(v) >= 2}
    within = np.sqrt(np.mean(np.concatenate([(v - v.mean(0)) ** 2 for v in multi.values()])))
    means = np.array([v.mean(0) for v in multi.values()])
    between = float(np.sqrt(np.mean((means - means.mean(0)) ** 2)))
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    g = np.array(groups, dtype=object)
    same = float(np.mean(g[nn] == g))
    return {"n": len(X), "groups": len(multi), "within": float(within), "between": between,
            "ratio": between / float(within), "nn_same": same}  # fmt: skip


def main() -> None:
    rows = []
    cat = synthetic_catalog(np.random.default_rng(0), n_per_family=10)
    X = np.array([[p.attributes[k] for k in RIASEC] for p in cat])
    rows.append(("合成（C5 產生器，每族群 10 個）", stats(X, [p.program_id[:6] for p in cat])))
    with open(
        ROOT / "data/reference/occupation_profiles_all_nonacademic.csv", encoding="utf-8"
    ) as fh:
        real = [r for r in csv.DictReader(fh) if r[RIASEC[0]] != ""]
    Xr = np.array([[float(r[k]) for k in RIASEC] for r in real])
    rows.append(("真實 CIP（4 碼群組）", stats(Xr, [r["cip"][:5] for r in real])))
    rows.append(("真實 CIP（2 碼系列）", stats(Xr, [r["cip"][:2] for r in real])))
    L = ["# 合成目錄與真實剖面的分散程度（B-C5-3，描述性、探索性）", "",
         f"- 產生日期：{date.today().isoformat()} · "
         "指令：`python scripts/describe_catalog_spread.py`",
         "- 只比較 RIASEC 6 維；沒有預先登記的決策，只供模擬器參數參考（B-C5-2）", "",
         "| 資料 | 單位數 | 群數（≥ 2 個單位） | 群內 SD | 群平均的 SD | 比值 | 最近鄰同群比例 |",
         "|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, st in rows:
        L.append(f"| {name} | {st['n']} | {st['groups']} | {st['within']:.3f} | "
                 f"{st['between']:.3f} | {st['ratio']:.2f} | {st['nn_same']:.2f} |")  # fmt: skip
    (ROOT / "results" / "catalog-spread.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    for name, st in rows:
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in st.items()})


if __name__ == "__main__":
    main()
