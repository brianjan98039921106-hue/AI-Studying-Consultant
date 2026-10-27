"""Build the draft v2 item bank (W-41, docs/item-bank-v2.md) from its sources.

Inputs (all in data/items/):
    onet_illustrative_activities_zh.csv  O*NET 31.0 "Interests Illustrative Activities"
                                         (4 activities per RIASEC type and per specific
                                         interest area) with Traditional Chinese wording
    core_items_zh.json                   intensity and orientation items (rubric §3)
Output: data/items/item_bank_v2.json (deterministic; rerun after editing a source).

Layout (73 items, 51 dimensions + 12 group items):
    RIASEC        2 items per type, 2 activities each (alpha computable, V-C1-5)
    intensity     2 items per dimension (behaviourally anchored)
    orientation   2 items per dimension (agreement)
    groups        1 item per area group (12, stage 1 of dimensions-v2 §2): one example
                  activity from every primary area of the group, at least 3 examples
    areas         1 item per specific interest area (41, stage 2): all 4 activities
    retest reliability of single-item areas comes from the W-41 retest (V-C1-6)

    python scripts/build_item_bank_v2.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from aisc.dimensions import BASIC_INTEREST_KEYS, NEED_DIMENSIONS_V2
from aisc.profiling.hierarchy import AREA_GROUPS, GROUP_LABELS

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "data" / "items"
SOURCE_URL = "https://www.onetcenter.org/dl_files/database/db_31_0_json/interests_illustrative_activities.json"
RIASEC = {"Realistic": "riasec_r", "Investigative": "riasec_i", "Artistic": "riasec_a",
          "Social": "riasec_s", "Enterprising": "riasec_e", "Conventional": "riasec_c"}  # fmt: skip
LIKERT = {"type": "likert5", "labels_zh": ["完全不想", "不太想", "普通", "有點想", "非常想"],
          "to_unit": "(answer - 1) / 4"}  # fmt: skip
MIN_GROUP_EXAMPLES = 3
# official O*NET 31.0 interests_illustrative_activities.csv the activities were checked against
VERIFIED_SHA256 = "f2602bb57b1dff64fa91b198685ddf66b6dfdbaa3e83d3c3c606e7a5615ff82c"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ITEMS / "item_bank_v2.json")
    args = ap.parse_args()
    with open(ITEMS / "onet_illustrative_activities_zh.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    labels = {d.key: (d.label_zh, d.label_en) for d in NEED_DIMENSIONS_V2}
    area_key = {en: k for k, (_, en) in labels.items() if k in BASIC_INTEREST_KEYS}
    acts: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = RIASEC.get(r["element_name"]) or area_key.get(r["element_name"])
        if key is None:
            raise SystemExit(f"unknown O*NET element {r['element_name']!r}")
        acts[key].append(r)
    missing = [k for k in [*RIASEC.values(), *BASIC_INTEREST_KEYS] if len(acts[k]) != 4]
    if missing:
        raise SystemExit(f"need exactly 4 activities for: {missing}")

    stem = "以下這些活動，你有多想做？（不考慮你現在會不會做）"
    items = []
    for key in RIASEC.values():
        for i, part in enumerate((acts[key][:2], acts[key][2:]), 1):
            items.append({"item_id": f"{key}#{i}", "dimension": key, "stage": "core",
                          "stem_zh": stem, "examples_zh": [a["activity_zh"] for a in part],
                          "examples_en": [a["activity_en"] for a in part],
                          "source": [a["element_id"] for a in part], "scale": LIKERT})  # fmt: skip
    core = json.loads((ITEMS / "core_items_zh.json").read_text(encoding="utf-8"))
    for it in core["items"]:
        items.append({**it, "stage": "core", "source": ["docs/rubric.md §3"]})
    for g, (zh, en) in GROUP_LABELS.items():
        members = [k for k in BASIC_INTEREST_KEYS if AREA_GROUPS[k][0] == g]
        # one example per primary area (every area is represented), then further
        # activities round-robin until the item shows at least MIN_GROUP_EXAMPLES
        ex = [acts[k][0] for k in members]
        j = 1
        while len(ex) < MIN_GROUP_EXAMPLES:
            ex += [acts[k][j] for k in members][: MIN_GROUP_EXAMPLES - len(ex)]
            j += 1
        items.append({"item_id": f"group_{g}", "dimension": None, "group": g, "stage": "group",
                      "label_zh": zh, "label_en": en,
                      "stem_zh": "整體而言，這一類的活動你有多想做？",
                      "examples_zh": [a["activity_zh"] for a in ex],
                      "examples_en": [a["activity_en"] for a in ex],
                      "source": [a["element_id"] for a in ex], "scale": LIKERT})  # fmt: skip
    for key in BASIC_INTEREST_KEYS:
        items.append({"item_id": f"{key}#1", "dimension": key, "stage": "area",
                      "groups": list(AREA_GROUPS[key]), "label_zh": labels[key][0],
                      "stem_zh": stem, "examples_zh": [a["activity_zh"] for a in acts[key]],
                      "examples_en": [a["activity_en"] for a in acts[key]],
                      "source": [a["element_id"] for a in acts[key]], "scale": LIKERT})  # fmt: skip
    bank = {
        "version": "v2-draft-0",
        "status": "draft: needs back-translation and cognitive interviews (W-41, D11)",
        "sources": {
            "onet_illustrative_activities": {
                "url": SOURCE_URL, "release": "O*NET 31.0", "retrieved_at": "2026-10-28",
                "method": "transcribed via web fetch, then verified row by row against the "
                          "official CSV (188 rows, identical text and order)",
                "verified_file": "interests_illustrative_activities.csv (O*NET 31.0 CSV "
                                 "release, provided by the owner 2026-10-28)",
                "verified_sha256": VERIFIED_SHA256,
            },
            "core_items": "data/items/core_items_zh.json (docs/rubric.md §3)",
        },
        "n_items": len(items),
        "items": items,
    }  # fmt: skip
    args.out.write_text(json.dumps(bank, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()
