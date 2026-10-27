"""Real-data check of series-mean fallback profiles for unmapped CIP codes (B-C0-9;
THEORY "無對照代碼的系列後備剖面"; A-C0-14).

Units: CIPs of the all-fields reference (rollout v3) whose non-academic occupation
profile exists for both the 6 RIASEC types (occupation_profiles_all_nonacademic.csv)
and the 41 interest areas (occupation_basic_interests_all_nonacademic.csv).
Leave-one-out: each CIP's profile is predicted, without itself, by
    F  catalog.occupation.series_fallback: mean of its 4-digit group, else 2-digit series
    G  global mean of all other CIPs (reference)
Metrics per CIP: RMSE over the 6 RIASEC values, RMSE over the 41 areas, top-5 area
recall (|top5(pred) & top5(true)| / 5).

PRE-REGISTERED (written and committed before the first run, 2026-10-28):
  ADOPT F for the CIPs without a profile (written as flagged, low-confidence fallback
  columns) if, over all CIPs with a prediction, the paired RMSE difference F - G is
  below -2 SE for BOTH the RIASEC and the area profile, and F's top-5 recall is not
  below G's by more than 2 SE. Otherwise B-C0-9 stays as is.
  Report only: the same metrics restricted to "other" codes (XX.XX99) that have a
  profile, as the closest analogue of the unmapped codes (A-C0-14); group vs series
  level counts.
  Disclosure: no fallback was computed on these data before this commit; the files
  are the committed reference tables (built from official IPEDS/CIP/O*NET files).
  The script was not smoke-run before the commit (it can only read the real tables);
  a crash fix after the commit would not change these rules.

    python scripts/check_series_fallback.py
"""

from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path

import numpy as np

from aisc.catalog.occupation import series_fallback
from aisc.dimensions import BASIC_INTEREST_KEYS

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
RIASEC = ["riasec_r", "riasec_i", "riasec_a", "riasec_s", "riasec_e", "riasec_c"]


def _read(name, keys):
    out = {}
    with open(REF / name, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if all(r[k] != "" for k in keys):
                out[r["cip"]] = {k: float(r[k]) for k in keys}
    return out


def top5(v):
    return set(np.argsort(-np.asarray(v), kind="stable")[:5].tolist())


def main() -> None:
    ria = _read("occupation_profiles_all_nonacademic.csv", RIASEC)
    are = _read("occupation_basic_interests_all_nonacademic.csv", list(BASIC_INTEREST_KEYS))
    cips = sorted(set(ria) & set(are))
    prof = {c: {**ria[c], **are[c]} for c in cips}
    keys = RIASEC + list(BASIC_INTEREST_KEYS)
    rows = {"F": {"r": [], "a": [], "t": []}, "G": {"r": [], "a": [], "t": []}}
    other, levels = [], {"group": 0, "series": 0}
    for c in cips:
        rest = {k: v for k, v in prof.items() if k != c}
        fb = series_fallback(rest, [c])
        if c not in fb:
            continue
        pf, level = fb[c]
        levels[level] += 1
        g = {k: float(np.mean([p[k] for p in rest.values()])) for k in keys}
        true = prof[c]
        for name, pred in (("F", pf), ("G", g)):
            rows[name]["r"].append(math.sqrt(np.mean([(pred[k] - true[k]) ** 2 for k in RIASEC])))
            rows[name]["a"].append(math.sqrt(np.mean([(pred[k] - true[k]) ** 2
                                                      for k in BASIC_INTEREST_KEYS])))  # fmt: skip
            hit = top5([pred[k] for k in BASIC_INTEREST_KEYS]) & top5(
                [true[k] for k in BASIC_INTEREST_KEYS]
            )
            rows[name]["t"].append(len(hit) / 5)
        other.append(c.endswith("99"))

    def paired(a, b, sel=None):
        d = np.asarray(a) - np.asarray(b)
        if sel is not None:
            d = d[np.asarray(sel)]
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))), len(d)

    dr, da, dt = (paired(rows["F"][m], rows["G"][m]) for m in ("r", "a", "t"))
    adopt = dr[0] < -2 * dr[1] and da[0] < -2 * da[1] and not dt[0] < -2 * dt[1]
    missing = sorted(set(ria) ^ set(are))
    L = [
        "# 無對照 CIP 的系列後備剖面：留一法檢驗（真實資料，B-C0-9）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`python scripts/check_series_fallback.py`",
        f"- 有剖面的 CIP：{len(cips)}；有後備預測的：{len(other)}（4 碼群組 {levels['group']}、"
        f"2 碼系列 {levels['series']}）；只有 RIASEC 或只有領域剖面的 CIP：{len(missing)}",
        "- 預先登記見腳本檔頭；F = 系列後備、G = 全體平均",
        "",
        f"## 判定：{'✅ 採用' if adopt else '❌ 不採用'}",
        "",
        "| 範圍 | n | RMSE RIASEC F | G | F − G | RMSE 領域 F | G | F − G "
        "| 前5 F | 前5 G | F − G |",
        "|---" * 11 + "|",
    ]
    for label, sel in (("全部", None), ("「其他」代碼 XX.XX99", other)):

        def m(v, sel=sel):
            v = np.asarray(v)
            return float(np.mean(v if sel is None else v[np.asarray(sel)]))

        r, a, t = (paired(rows["F"][k], rows["G"][k], sel) for k in ("r", "a", "t"))
        L.append(f"| {label} | {r[2]} | {m(rows['F']['r']):.4f} | {m(rows['G']['r']):.4f} | "
                 f"{r[0]:+.4f}±{r[1]:.4f} | {m(rows['F']['a']):.4f} | {m(rows['G']['a']):.4f} | "
                 f"{a[0]:+.4f}±{a[1]:.4f} | {m(rows['F']['t']):.3f} | {m(rows['G']['t']):.3f} | "
                 f"{t[0]:+.3f}±{t[1]:.3f} |")  # fmt: skip
    (ROOT / "results" / "series-fallback.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"adopt={adopt} n={len(other)} levels={levels}")


if __name__ == "__main__":
    main()
