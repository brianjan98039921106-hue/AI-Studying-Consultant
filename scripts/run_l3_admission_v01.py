"""C3.2 v0 vs v0.1 candidates on the V-C3-1 worlds, fresh seeds (F130, B-C3-8, B-C3-9;
THEORY "C3.2 v0.1 候選"; A-C3-6).

Worlds, settings and evaluation exactly as scripts/run_l3_admission_bias.py (both
generators x 2 reporting strengths x 2 data sources), but with a fresh seed (2030) and
30 replicates. All fits use official rates (the corrected model). Candidates:
    v0     anchor="mean", plug-in tau (current default)
    P      anchor="pool"
    M      tau_average=True
    PM     anchor="pool", tau_average=True
Per replicate: RMSE of the posterior median vs the true P, mean error, 90% coverage.

PRE-REGISTERED (written and committed before the first run, 2026-10-28):
  A candidate X QUALIFIES if
    (a) RMSE: in no setting is X's RMSE above v0's by more than 2 SE (paired), and in
        at least one setting of each generator it is below by more than 2 SE; and
    (b) coverage: |coverage - 0.90| is smaller than v0's in at least 6 of the 8 settings.
  Among qualifying candidates the one with the lowest mean RMSE over the 8 settings
  becomes the C3.2 default (v0.1); v0 stays available as anchor="mean",
  tau_average=False. None qualifies -> v0 stays the default.
  Report only: the B-C3-8 and B-C3-9 probes are re-run by the test suite afterwards.
  Disclosure: before this run the options were only exercised by unit tests and a
  regression check showing v0 unchanged; smoke-tested with 1 replicate (exit status).

    python scripts/run_l3_admission_v01.py
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
SEED = 2030
N_REP = 30
CANDS = {
    "v0": {},
    "P": {"anchor": "pool"},
    "M": {"tau_average": True},
    "PM": {"anchor": "pool", "tau_average": True},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=N_REP)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-l3-admission-v01.md")
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

    def rmse(rows, c):
        return [x[0] for x in rows[c]]

    def cov(rows, c):
        return float(np.mean([x[2] for x in rows[c]]))

    qual = {}
    for c in CANDS:
        if c == "v0":
            continue
        worse = any(paired(rmse(r, c), rmse(r, "v0"))[0] > 2 * paired(rmse(r, c), rmse(r, "v0"))[1]
                    for r in res.values())  # fmt: skip
        better = all(
            any((lambda d: d[0] < -2 * d[1])(paired(rmse(r, c), rmse(r, "v0")))
                for (g, _, _), r in res.items() if g == gen)
            for gen in GENS
        )  # fmt: skip
        closer = sum(abs(cov(r, c) - 0.9) < abs(cov(r, "v0") - 0.9) for r in res.values())
        qual[c] = (not worse) and better and closer >= 6
    mean_rmse = {c: float(np.mean([np.mean(rmse(r, c)) for r in res.values()])) for c in CANDS}
    ok = [c for c in qual if qual[c]]
    chosen = min(ok, key=lambda c: mean_rmse[c]) if ok else "v0"

    def ms(v):
        v = np.asarray(v)
        return f"{v.mean():.3f}±{v.std(ddof=1) / math.sqrt(len(v)):.3f}"

    L = [
        "# C3.2 v0 與 v0.1 候選（母體積分錨點、τ 模型平均）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`python scripts/run_l3_admission_v01.py`",
        f"- V-C3-1 的世界與設定，新種子 {SEED}，每個設定 {args.reps} 個世界；全部使用官方錄取率",
        "- 預先登記見腳本檔頭；全部是合成資料",
        "",
        "## 判定",
        "",
        "- "
        + "；".join(f"{c}：{'✅' if qual[c] else '❌'}" for c in qual)
        + f" → 預設 **{chosen}**",
        "",
        "| 生成 | q₁ / q₀ | 資料 | "
        + " | ".join(f"RMSE {c}" for c in CANDS)
        + " | "
        + " | ".join(f"{c} − v0" for c in CANDS if c != "v0")
        + " | "
        + " | ".join(f"涵蓋 {c}" for c in CANDS)
        + " | "
        + " | ".join(f"平均誤差 {c}" for c in CANDS)
        + " |",
        "|---" * (3 + 4 + 3 + 4 + 4) + "|",
    ]
    for (gen, q, data), r in res.items():
        diffs = [paired(rmse(r, c), rmse(r, "v0")) for c in CANDS if c != "v0"]
        L.append(
            f"| {gen} | {q[0]} / {q[1]} | {data} | "
            + " | ".join(ms(rmse(r, c)) for c in CANDS)
            + " | "
            + " | ".join(f"{d[0]:+.4f}±{d[1]:.4f}" for d in diffs)
            + " | "
            + " | ".join(f"{cov(r, c):.2f}" for c in CANDS)
            + " | "
            + " | ".join(f"{np.mean([x[1] for x in r[c]]):+.3f}" for c in CANDS)
            + " |"
        )
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"qualifies={qual} chosen={chosen} wrote {args.out}")


if __name__ == "__main__":
    main()
