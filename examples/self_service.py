"""Student self-service run in the terminal (W-52; the web interface is W-53).

    python examples/self_service.py                 # answer in the terminal
    python examples/self_service.py --answers a.json   # scripted (tests, demos)

Flow: privacy notice and consent (docs/privacy-notice.md) -> 73 questions
(skippable) -> familiarity check on the shortlisted degree types -> two-section
recommendations on the real catalogue v0 (IPEDS master's programmes, experimental,
docs/USAGE.md §4.11) -> Markdown report written under data/private/ (never
committed). No name or contact details are asked for.

--answers JSON: {"consent": true, "items": {item_id: 1..5 or null},
                 "familiar": {degree name: "unknown" | "heard" | "can_describe"}}
Degrees missing from "familiar" count as "unknown".
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aisc.catalog.real import load_real_catalog_v0  # noqa: E402
from aisc.pipeline import (  # noqa: E402
    interactive_discovery,
    merge_identical_programs,
    prepare_profile,
)
from aisc.profiling.questionnaire import Questionnaire, load_questions  # noqa: E402
from aisc.report import render_profile_report  # noqa: E402

NOTICE = """本工具依你的作答產生研究所學程推薦。
- 不需要姓名、身分證字號、電話或 e-mail。
- 作答只用來計算你的推薦與報告，報告存在這台電腦的 data/private/，不會上傳。
- 每一題都可以按 Enter 跳過；跳過的部分用一般預設值補上，推薦會比較不精準。
完整說明：docs/privacy-notice.md"""
FAMILIAR = {"0": "unknown", "1": "heard", "2": "can_describe"}


def ask_terminal(q, n, total) -> int | None:
    print(f"\n[{n}/{total}] {q.stem_zh}")
    for ex in q.examples_zh:
        print(f"    · {ex}")
    for i, lab in enumerate(q.labels_zh, start=1):
        print(f"  {i}. {lab}")
    while True:
        s = input("你的答案（1–5，Enter 跳過）：").strip()
        if s == "":
            return None
        if s in {"1", "2", "3", "4", "5"}:
            return int(s)
        print("請輸入 1 到 5，或直接按 Enter 跳過。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, help="scripted answers (JSON)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--k-new", type=int, default=10)
    ap.add_argument("--k-known", type=int, default=5)
    args = ap.parse_args()
    script = json.loads(args.answers.read_text(encoding="utf-8")) if args.answers else None

    print(NOTICE)
    if script is not None:
        consent = bool(script.get("consent"))
    else:
        consent = input("\n同意並開始？（y／n）：").strip().lower() == "y"
    if not consent:
        print("沒有同意，不會記錄任何作答。")
        return

    qs = Questionnaire(load_questions(ROOT / "data" / "items" / "item_bank_v2.json"))
    total = len(qs.questions)
    while qs.next_page():
        for q in qs.next_page():
            if script is not None:
                v = script["items"].get(q.item_id)
            else:
                v = ask_terminal(q, qs.progress()[0] + 1, total)
            qs.answer(q.item_id, v)
    student = qs.profile(f"self-{date.today().isoformat()}")
    print(f"\n完成：{total - len(qs.skipped())} 題作答、{len(qs.skipped())} 題跳過。")

    real = load_real_catalog_v0(ROOT / "data" / "reference").programs
    catalog, members = merge_identical_programs(real)  # one entry per degree type (B-C2-30)
    names = {p.program_id: p.degree_name for p in catalog}

    def ask(program_ids: list[str]) -> dict[str, str]:
        degrees = sorted({names[i] for i in program_ids})
        if script is not None:
            got = {d: script.get("familiar", {}).get(d, "unknown") for d in degrees}
        else:
            print("\n下面這些學位你聽過嗎？（0 沒聽過、1 聽過、2 能說明它在學什麼）")
            got = {}
            for d in degrees:
                s = ""
                while s not in FAMILIAR:
                    s = input(f"  {d}：").strip() or "0"
                got[d] = FAMILIAR[s]
        return {i: got[names[i]] for i in program_ids}

    result = interactive_discovery(student, catalog, ask, k_new=args.k_new, k_known=args.k_known)
    report = render_profile_report(student, prepare_profile(student, catalog), result)
    out = args.out or ROOT / "data" / "private" / f"{student.student_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print("\n== 你可能沒想過 ==")
    for r in result.discoveries:
        schools = [m.school for m in members[r.program.program_id]]
        more = f" 等 {len(schools)} 所" if len(schools) > 3 else ""
        print(f"  {r.program.degree_name}｜{'、'.join(schools[:3])}{more}")
    print(f"\n完整報告：{out}")


if __name__ == "__main__":
    main()
