"""Fit the C3.2 admission model on real (private) outcome files and check calibration
with a time split (C3.1-C3.3; V-C3-2, V-C3-3, V-C3-4; docs/data-intake.md).

    python scripts/validate_admission_model.py --records data/private/admission_records.csv \
        --rates data/private/official_rates.csv [--out data/private/admission_check.md]

Steps
  1. load and validate both files (aisc.feasibility.intake); print the intake report
  2. fit on all records (current defaults) and list warnings (convergence, conflicts)
  3. time split, when records carry years and the latest year has >= 20 complete
     records: fit on the earlier years (features standardised with their own statistics,
     official rates of the last training year, the same rule as the full fit) and
     evaluate on the latest year's complete records: Brier skill vs the observed rate,
     Cox calibration intercept / slope, reliability table, per applicant group. The
     report states how many programmes are anchored in each fit and skips (and counts)
     test records the training fit cannot predict (B-C3-14, B-C3-15).
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
ON_ZH = {
    "complete records": "完整紀錄",
    "all records": "全部紀錄",
    "given statistics": "指定統計量",
}


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
    n_test = sum(r.year == latest and r.source == "complete" for r in data.records)
    L += ["", "## 時間切分（以較早年度預測最新年度的完整紀錄）", ""]
    if len(years) < 2 or n_test < args.min_test:
        L.append(f"- 無法進行：需要至少兩個年度，且最新年度至少 {args.min_test} 筆完整紀錄"
                 f"（目前 {len(years)} 個年度、{n_test} 筆）")  # fmt: skip
    else:
        train = load_admission_records(args.records, years=set(years[:-1]))
        test = load_admission_records(args.records, years={latest},
                                      standardize_with=train.report)  # fmt: skip
        train_rates = load_official_rates(args.rates, years[-2]) if args.rates else {}
        train_rates = {p: v for p, v in train_rates.items() if p in train.families}
        full_rates = {p: v for p, v in (rates or {}).items() if p in data.families}
        if args.rates:
            L.append(f"- 官方錄取率錨點：全部資料 {len(full_rates)} 個學程（{latest} 年）；"
                     f"時間切分訓練 {len(train_rates)} 個學程（{years[-2]} 年）")  # fmt: skip
            if len(train_rates) < len(full_rates):
                L.append("- ⚠️ 訓練擬合的錨點比全部資料的擬合少：檢查結果代表錨點較少的模型"
                         "（請補上較早年度的官方錄取率）")  # fmt: skip
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = fit_admission_model(train.records, families=train.families,
                                    official_rates=train_rates or None,
                                    types=train.types)  # fmt: skip
        pred, y, grp, skipped = [], [], [], 0
        for i, r in enumerate(test.records):
            if r.source != "complete":
                continue
            ptype = test.types.get(r.program_id) if test.types else None
            try:
                est = m.predict(r.program_id, r.features, family=test.families[r.program_id],
                                ptype=ptype)  # fmt: skip
            except ValueError:
                skipped += 1
                continue
            pred.append(est.mean)
            y.append(r.admitted)
            grp.append(test.groups[i] or "（未填）")
        if skipped:
            L.append(f"- 略過 {skipped} 筆：訓練年度沒有的族群或學程類型，無法預測")
        if len(y) < args.min_test:
            L.append(f"- 可預測的測試紀錄只有 {len(y)} 筆（< {args.min_test}），不計算指標")
        else:
            try:
                skill = f"{brier_skill(pred, y):+.3f}"
                a, b = calibration_slope(pred, y)
                cal = f"{a:+.2f}、斜率 {b:.2f}"
            except ValueError:
                skill = cal = "無法計算（測試年度的結果全部相同）"
            L += [f"- 測試筆數 {len(y)}；實際錄取率 {sum(y) / len(y):.3f}；平均預測 "
                  f"{sum(pred) / len(pred):.3f}",
                  f"- Brier {brier(pred, y):.4f}；Brier skill（相對於常數預測）{skill}"
                  "（V-C3-2：須 > 0）",
                  f"- 校準截距 {cal}（理想 0 與 1；斜率 < 1 表示過度自信）", "",
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
