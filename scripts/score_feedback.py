"""Score trial feedback from the self-service page (W-82; docs/evaluation-plan.md §3;
V-C2-10, V-C1-10, V-C2-8). Feedback is personal data: keep it in data/private/.

Input: files or folders of pasted messages. Each feedback block is
    AISC-FEEDBACK v1
    {...json, schema "aisc-feedback/1"...}
    END
anywhere in a file (surrounding chat text is ignored); identical blocks count once.

Per student (fixed in docs/evaluation-plan.md §3.3 before any trial):
    NR@10         "new" section, rank <= 10, rated 4-5, not marked "already considering"
    R@15          share of rated recommendations rated 4-5 (with how many were rated)
    far FP        "new" section, rank <= 10, rated 1 (V-C2-8)
    explore       recommendations marked "want to explore"
    profile fit   1-5 (V-C1-10)
    burden        minutes, items answered
Targets (§3.4, medians): NR@10 >= 3, profile fit >= 4, far FP <= 2. With few students
these are descriptive. The diagnosis follows the §3.4 table.

    python scripts/score_feedback.py data/private/feedback/ [--out data/private/feedback_summary.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"AISC-FEEDBACK v1\s*\n\s*(\{.*?\})\s*\n\s*END", re.S)
SCHEMA = "aisc-feedback/1"
TARGETS = {"nr10": 3, "fit": 4, "far_fp": 2}
BG_ZH = {
    "same_field": "延續",
    "adjacent": "相近",
    "distant": "很不同",
    "multi_degree": "多學位",
    "": "未填",
}


def extract(paths: list[Path]) -> list[dict]:
    seen, out = set(), []
    files = [f for p in paths for f in (sorted(p.rglob("*")) if p.is_dir() else [p]) if f.is_file()]
    for f in files:
        for m in BLOCK.finditer(f.read_text(encoding="utf-8", errors="replace")):
            raw = m.group(1)
            key = hashlib.sha256(raw.encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{f}: a feedback block is not valid JSON ({exc})") from exc
            if obj.get("schema") != SCHEMA:
                raise SystemExit(f"{f}: unknown feedback schema {obj.get('schema')!r}")
            out.append(obj)
    return out


def score(fb: dict) -> dict:
    recs = fb["recs"]
    new10 = [r for r in recs if r["section"] == "new" and r["rank"] <= 10]
    rated = [r for r in recs if r["rating"] is not None]
    return {
        "nr10": sum(1 for r in new10 if (r["rating"] or 0) >= 4 and not r["already_considering"]),
        "r15": (sum(r["rating"] >= 4 for r in rated) / len(rated)) if rated else None,
        "rated": len(rated),
        "n_recs": len(recs),
        "far_fp": sum(1 for r in new10 if r["rating"] == 1),
        "explore": sum(1 for r in recs if r["want_to_explore"]),
        "fit": fb["profile_fit"],
        "minutes": fb["duration_sec"] / 60,
        "answered": fb["answered"],
        "total_items": fb["total_items"],
        "background": fb.get("background", ""),
    }


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def diagnose(med: dict) -> str:
    """docs/evaluation-plan.md §3.4."""
    fit, nr, explore = med["fit"], med["nr10"], med["explore"]
    if fit is not None and fit < TARGETS["fit"]:
        return "剖面吻合度低：問題多半在問卷（C1），先看「看不懂的題目」"
    if nr is not None and nr < TARGETS["nr10"]:
        return "剖面吻合度可以但 NR@10 低：問題多半在學程資料或計分器（C0、C2），看被評低分的學門"
    if nr is not None and explore is not None and explore < 1:
        return "NR@10 達標但很少人想進一步查：問題多半在結果頁的呈現"
    return "三個目標都達到（人數少，只能當描述）"


def summary(fbs: list[dict]) -> str:
    rows = [score(f) for f in fbs]
    med = {
        k: _median([r[k] for r in rows])
        for k in ("nr10", "r15", "far_fp", "explore", "fit", "minutes")
    }
    rng = {k: (min(v), max(v)) if (v := [r[k] for r in rows if r[k] is not None]) else None
           for k in ("nr10", "far_fp", "fit")}  # fmt: skip

    def f(x, fmt="{:.1f}"):
        return "—" if x is None else fmt.format(x)

    ok = {True: "✅", False: "❌", None: "—"}
    L = ["# 試用回饋彙總（私人資料，勿提交）", "",
         f"- 回饋份數 {len(rows)}；中位數與範圍（人數少，只當描述）", "",
         "| 指標 | 中位數 | 範圍 | 目標 |", "|---|---|---|---|",
         f"| NR@10（適合又沒想過） | {f(med['nr10'])} | {rng['nr10']} | ≥ 3 "
         f"{ok[None if med['nr10'] is None else med['nr10'] >= TARGETS['nr10']]} |",
         f"| 剖面吻合度 | {f(med['fit'])} | {rng['fit']} | ≥ 4 "
         f"{ok[None if med['fit'] is None else med['fit'] >= TARGETS['fit']]} |",
         f"| 遠領域誤推（前 10 評 1 分） | {f(med['far_fp'])} | {rng['far_fp']} | ≤ 2 "
         f"{ok[None if med['far_fp'] is None else med['far_fp'] <= TARGETS['far_fp']]} |",
         f"| R@15（評 4–5 分的比例） | {f(med['r15'], '{:.0%}')} | — | 回報 |",
         f"| 想進一步查 | {f(med['explore'])} | — | 回報 |",
         f"| 作答時間（分鐘） | {f(med['minutes'])} | — | 回報 |", "",
         f"**診斷**：{diagnose(med)}", "",
         "| # | 背景 | NR@10 | R@15 | 評分數 | 遠領域誤推 | 想查 | 吻合度 | 分鐘 | 作答 |",
         "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for i, r in enumerate(rows, 1):
        L.append(f"| {i} | {BG_ZH.get(r['background'], r['background'])} | {r['nr10']} | "
                 f"{f(r['r15'], '{:.0%}')} | {r['rated']}/{r['n_recs']} | {r['far_fp']} | "
                 f"{r['explore']} | {f(r['fit'], '{}')} | {r['minutes']:.0f} | "
                 f"{r['answered']}/{r['total_items']} |")  # fmt: skip
    low = {}
    for fb in fbs:
        for rec in fb["recs"]:
            if rec["rating"] in (1, 2):
                low.setdefault(f"{rec['name']}（{rec['cip']}）", []).append(rec["rating"])
    if low:
        ranked = sorted(low.items(), key=lambda kv: -len(kv[1]))
        L += ["", "評 1–2 分的學門（看 C0、C2）：" + "、".join(f"{k} ×{len(v)}" for k, v in ranked)]
    unclear = [fb["unclear_items"].strip() for fb in fbs if fb.get("unclear_items", "").strip()]
    if unclear:
        L += ["", "看不懂的題目（W-41）：", *[f"- {u}" for u in unclear]]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "private" / "feedback_summary.md")
    args = ap.parse_args()
    fbs = extract(args.paths)
    if not fbs:
        raise SystemExit("no feedback blocks found (look for lines 'AISC-FEEDBACK v1' ... 'END')")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(summary(fbs), encoding="utf-8")
    print(f"{len(fbs)} feedback; wrote {args.out}")


if __name__ == "__main__":
    main()
