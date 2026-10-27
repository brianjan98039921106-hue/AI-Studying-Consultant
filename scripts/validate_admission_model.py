"""Fit the C3.2 admission model on real (private) outcome files and check calibration
with a time split (C3.1-C3.3; V-C3-2, V-C3-3, V-C3-4; docs/data-intake.md).

    python scripts/validate_admission_model.py --records data/private/admission_records.csv \
        --rates data/private/official_rates.csv [--out data/private/admission_check.md]

Steps
  1. load and validate both files (aisc.feasibility.intake); print the intake report
  2. fit on all records (current defaults) and list warnings (convergence, conflicts)
  3. time split, when records carry years and the latest year has >= 20 complete
     records: fit on the earlier years (official rates of their latest year, if given)
     and evaluate on the latest year's complete records: Brier skill vs the observed
     rate, Cox calibration intercept / slope, reliability table, per applicant group
The output goes to data/private/ by default because it describes private data.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from aisc.feasibility import (
    brier,
    brier_skill,
    by_group,
    calibration_slope,
    fit_admission_model,
    reliability_table,
)
from aisc.feasibility.intake import load_admission_records, load_official_rates

ROOT = Path(__file__).resolve().parents[1]
ON_ZH = {"complete records": "完整紀錄", "all records": "全部紀錄"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--rates", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "private" / "admission_check.md")
    ap.add_argument("--min-test", type=int, default=20)
    args = ap.parse_args()
    data = load_admission_records(args.records)
    rep = data.report
    L = ["# 錄取模型：資料載入與時間切分檢查（私人資料，勿提交）", "",
         f"- 紀錄 {rep.n_rows} 筆（完整 {rep.by_source['complete']}、自陳 "
         f"{rep.by_source['self_report']}）；學程 {len(data.families)} 個；"
         f"特徵 {', '.join(rep.features) or '無'}（以{ON_ZH[rep.standardized_on]}標準化）",
         f"- 缺漏：{rep.missing}；申請者群組：{list(rep.groups) or '未提供'}"]  # fmt: skip
    L += [f"- ⚠️ {w}" for w in rep.warnings]
    years = sorted({r.year for r in data.records if r.year is not None})
    latest = years[-1] if years else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rates = load_official_rates(args.rates, latest) if args.rates else None
        model = fit_admission_model(data.records, families=data.families,
                                    official_rates=rates, types=data.types)  # fmt: skip
    L += ["", "## 全部資料的擬合", "",
          f"- 方法 {model.method}；收斂 {'是' if model.converged else '否'}；"
          f"自陳平移 {model.shift:+.2f}；年度 {list(model.years)} → "
          f"預測 {model.reference_year}"]  # fmt: skip
    L += [f"- ⚠️ {w.message}" for w in caught]
    test = [i for i, r in enumerate(data.records)
            if latest is not None and r.year == latest and r.source == "complete"]  # fmt: skip
    L += ["", "## 時間切分（以較早年度預測最新年度的完整紀錄）", ""]
    if len(years) < 2 or len(test) < args.min_test:
        L.append(f"- 無法進行：需要至少兩個年度，且最新年度至少 {args.min_test} 筆完整紀錄"
                 f"（目前 {len(years)} 個年度、{len(test)} 筆）")  # fmt: skip
    else:
        train = [r for r in data.records if r.year != latest]
        train_rates = load_official_rates(args.rates, years[-2]) if args.rates else None
        train_rates = {p: v for p, v in (train_rates or {}).items()
                       if p in {r.program_id for r in train}} or None  # fmt: skip
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = fit_admission_model(train, families=data.families, official_rates=train_rates,
                                    types=data.types)  # fmt: skip
        pred, y, grp = [], [], []
        for i in test:
            r = data.records[i]
            ptype = data.types.get(r.program_id) if data.types else None
            est = m.predict(r.program_id, r.features, family=data.families[r.program_id],
                            ptype=ptype)  # fmt: skip
            pred.append(est.mean)
            y.append(r.admitted)
            grp.append(data.groups[i] or "（未填）")
        a, b = calibration_slope(pred, y)
        L += [f"- 測試筆數 {len(y)}；實際錄取率 {sum(y) / len(y):.3f}；平均預測 "
              f"{sum(pred) / len(pred):.3f}",
              f"- Brier {brier(pred, y):.4f}；Brier skill（相對於常數預測）"
              f"{brier_skill(pred, y):+.3f}（V-C3-2：須 > 0）",
              f"- 校準截距 {a:+.2f}、斜率 {b:.2f}（理想 0 與 1；斜率 < 1 表示過度自信）", "",
              "| 預測區間 | 筆數 | 平均預測 | 實際錄取率 |", "|---|---|---|---|"]  # fmt: skip
        L += [f"| {lo:.1f}–{hi:.1f} | {n} | {mp:.3f} | {ob:.3f} |"
              for lo, hi, n, mp, ob in reliability_table(pred, y, bins=5)]  # fmt: skip
        if len(set(grp)) > 1:
            L += ["", "| 群組 | Brier |", "|---|---|"]
            L += [f"| {g} | {v:.4f} |" for g, v in by_group(brier, pred, y, grp).items()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
