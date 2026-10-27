"""Confirmation of the pool-integrated anchor on fresh seeds (F137, F138; B-C3-8;
THEORY "C3.2 v0.1 候選"; A-C3-6).

F138 proposed, after seeing results, to judge interval coverage by the project's own
L5 band (V-C3-3: 85-95%) instead of the distance to 0.90. Per the tuning rule, that
proposal must be confirmed on new seeds before use. Worlds, settings and evaluation
exactly as scripts/run_l3_admission_bias.py, fresh seed 2031, 30 replicates per
setting, official rates used by every fit. Compared:
    M   current default (tau model averaging, mean anchor)
    PM  tau model averaging + anchor="pool"

PRE-REGISTERED (written and committed before the first run, 2026-10-28):
  PM becomes the default if
    (a) in no setting is PM's RMSE above M's by more than 2 SE (paired), and in at
        least one setting of each generator it is below by more than 2 SE; and
    (b) PM's 90% coverage lies in [0.85, 0.95] in every standard-generator setting,
        and the number of the 8 settings with coverage in [0.85, 0.95] is larger for
        PM than for M.
  Otherwise M stays the default and anchor="pool" stays an option.
  Report only: mean error (bias), coverage, the B-C3-8 probe afterwards.
  Disclosure: the candidate and the band criterion were chosen after seeing
  synthetic-l3-admission-v01.md (seed 2030); this run uses seed 2031 only.
  Smoke-tested with 1 replicate (exit status only).

    python scripts/run_l3_admission_pool_confirm.py
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

from aisc.feasibility import fit_admission_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_l3_admission_bias import DATA, GENS, QS, evaluate, world  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 2031
N_REP = 30
CANDS = {"M": {"tau_average": True}, "PM": {"tau_average": True, "anchor": "pool"}}
BAND = (0.85, 0.95)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=N_REP)
    out = ROOT / "results" / "synthetic-l3-admission-pool-confirm.md"
    ap.add_argument("--out", type=Path, default=out)
    args = ap.parse_args()
    res = {}
    for gen in GENS:
        for q in QS:
            for data in DATA:
                rows = {c: [] for c in CANDS}
                for r in range(args.reps):
                    rng = np.random.default_rng([SEED, GENS.index(gen), QS.index(q),
                                                 DATA.index(data), r])  # fmt: skip
                    recs, rates, tests, fams = world(gen, q, data, rng)
                    for c, kw in CANDS.items():
                        m = fit_admission_model(recs, families=fams, official_rates=rates, **kw)
                        rows[c].append(evaluate(m, tests))
                res[(gen, q, data)] = rows
                print(gen, q, data, flush=True)

    def paired(a, b):
        d = np.asarray(a) - np.asarray(b)
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))

    def rm(r, c):
        return [x[0] for x in r[c]]

    def cov(r, c):
        return float(np.mean([x[2] for x in r[c]]))

    def in_band(v):
        return BAND[0] <= v <= BAND[1]

    diffs = {k: paired(rm(r, "PM"), rm(r, "M")) for k, r in res.items()}
    a_worse = any(d > 2 * se for d, se in diffs.values())
    a_better = all(any(d < -2 * se for (g, _, _), (d, se) in diffs.items() if g == gen)
                   for gen in GENS)  # fmt: skip
    b_std = all(in_band(cov(r, "PM")) for (g, _, _), r in res.items() if g == "standard")
    n_band = {c: sum(in_band(cov(r, c)) for r in res.values()) for c in CANDS}
    adopted = (not a_worse) and a_better and b_std and n_band["PM"] > n_band["M"]

    def ms(v):
        v = np.asarray(v)
        return f"{v.mean():.3f}±{v.std(ddof=1) / math.sqrt(len(v)):.3f}"

    L = [
        "# 母體積分錨點的新種子確認（F137–F138）",
        "",
        f"- 產生日期：{date.today().isoformat()} · "
        "指令：`python scripts/run_l3_admission_pool_confirm.py`",
        f"- V-C3-1 的世界與設定，新種子 {SEED}，每個設定 {args.reps} 個世界；全部使用官方錄取率",
        "- 預先登記見腳本檔頭；全部是合成資料",
        "",
        f"## 判定：(a) RMSE 不變差 {'✅' if not a_worse else '❌'}、每種生成至少一格較好 "
        f"{'✅' if a_better else '❌'}；(b) standard 都在 85–95% {'✅' if b_std else '❌'}、"
        f"在範圍內的設定數 PM {n_band['PM']} vs M {n_band['M']} → "
        f"**{'PM 成為預設' if adopted else 'M 維持預設'}**",
        "",
        "| 生成 | q₁ / q₀ | 資料 | RMSE M | RMSE PM | PM − M | 涵蓋 M | 涵蓋 PM "
        "| 平均誤差 M | 平均誤差 PM |",
        "|---" * 10 + "|",
    ]
    for (gen, q, data), r in res.items():
        d = diffs[(gen, q, data)]
        L.append(f"| {gen} | {q[0]} / {q[1]} | {data} | {ms(rm(r, 'M'))} | {ms(rm(r, 'PM'))} | "
                 f"{d[0]:+.4f}±{d[1]:.4f} | {cov(r, 'M'):.3f} | {cov(r, 'PM'):.3f} | "
                 f"{np.mean([x[1] for x in r['M']]):+.3f} | "
                 f"{np.mean([x[1] for x in r['PM']]):+.3f} |")  # fmt: skip
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"adopted={adopted} a_worse={a_worse} a_better={a_better} b_std={b_std} "
          f"n_band={n_band} wrote {args.out}")  # fmt: skip


if __name__ == "__main__":
    main()
