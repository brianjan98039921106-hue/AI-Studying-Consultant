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
import secrets
import sys
from datetime import datetime
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
DELETE_NOTE = "這份報告只存在這台電腦；不需要時直接刪除這個檔案。共用電腦請看完就刪除。"


def fail(msg: str) -> None:
    print(f"錯誤：{msg}", file=sys.stderr)
    sys.exit(2)


def check_script(script: dict, item_ids: set[str]) -> None:
    """Scripted answers must name known items with 1-5 integers or null (B-C1-41)."""
    items = script.get("items")
    if not isinstance(items, dict):
        fail("answers 檔需要 items 物件")
    unknown = sorted(set(items) - item_ids)
    if unknown:
        fail(f"不認得的題號 {unknown[:5]}")
    bad = {k: v for k, v in items.items()
           if v is not None and (isinstance(v, bool) or v not in (1, 2, 3, 4, 5))}  # fmt: skip
    if bad:
        fail(f"答案必須是 1-5 的整數或 null：{dict(list(bad.items())[:5])}")
    fam = script.get("familiar", {})
    if not isinstance(fam, dict) or set(fam.values()) - set(FAMILIAR.values()):
        fail(f"familiar 的值必須是 {sorted(FAMILIAR.values())}")


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
    sys.stdout.reconfigure(errors="replace")  # terminals that cannot show Chinese
    try:
        run()
    except (EOFError, KeyboardInterrupt):
        print("\n已中止，沒有儲存任何資料。")
        sys.exit(1)


def run() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, help="scripted answers (JSON)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--k-new", type=int, default=10)
    ap.add_argument("--k-known", type=int, default=5)
    args = ap.parse_args()
    script = json.loads(args.answers.read_text(encoding="utf-8")) if args.answers else None

    print(NOTICE)
    if script is not None:
        consent = script.get("consent") is True  # B-C1-41: only an explicit true
    else:
        consent = input("\n同意並開始？（y／n）：").strip().lower() == "y"
    if not consent:
        print("沒有同意，不會記錄任何作答。")
        return

    qs = Questionnaire(load_questions(ROOT / "data" / "items" / "item_bank_v2.json"))
    total = len(qs.questions)
    if script is not None:
        check_script(script, {q.item_id for q in qs.questions})
    while qs.next_page():
        for q in qs.next_page():
            if script is not None:
                v = script["items"].get(q.item_id)
            else:
                v = ask_terminal(q, qs.progress()[0] + 1, total)
            qs.answer(q.item_id, v)
    answered = total - len(qs.skipped())
    print(f"\n完成：{answered} 題作答、{len(qs.skipped())} 題跳過。")
    if not qs.needs():
        print("沒有任何作答，無法產生推薦；沒有儲存任何資料。")
        return
    if answered < total / 2:
        print("提醒：跳過的題目超過一半，推薦主要來自預設值，參考價值有限。")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    student = qs.profile(f"self-{stamp}-{secrets.token_hex(3)}")

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
    print(f"\n完整報告：{out}\n{DELETE_NOTE}")


if __name__ == "__main__":
    main()
