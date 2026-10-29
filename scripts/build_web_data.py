"""Build web/data.json for the student self-service page (W-53, D14).

Everything the page needs, exported from the Python model so the JavaScript engine
(web/engine.js) only re-implements arithmetic, never data (A-C2-60):
    items       the v2 item bank (data/items/item_bank_v2.json)
    dims        v2 dimension keys in vector order, layer and Chinese label
    groups      the 12 interest-area groups, area -> groups, base rates (C1.2 priors)
    scorer      the v2 default scorer's layer weights and blend parameters
    catalog     real catalogue v0 merged into degree types (pipeline.merge_identical_programs):
                id, degree name, CIP, attribute vector, unmeasured dims, school indices
    schools     school names referenced by the catalog
    flow        n_shortlist, k_new, k_known and the familiarity levels counted as known

    python scripts/build_web_data.py            # write web/data.json
    python scripts/build_web_data.py --check    # exit 1 if web/data.json is stale
    python scripts/build_web_data.py --bundle web/dist/self-service.html
        # one self-contained page (engine and data inlined) for hosting without a server
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aisc.catalog.real import load_real_catalog_v0  # noqa: E402
from aisc.dimensions import DIMENSION_SETS, LAYERS, NEED_KEYS_V2  # noqa: E402
from aisc.discovery.scorers import DEFAULT_SCORER_V2  # noqa: E402
from aisc.pipeline import merge_identical_programs  # noqa: E402
from aisc.profiling.anchors import NEUTRAL_VALUE  # noqa: E402
from aisc.profiling.familiarity import DEFAULT_KNOWN, LEVELS  # noqa: E402
from aisc.profiling.hierarchy import AREA_GROUPS, BASE_RATE, GROUP_LABELS  # noqa: E402

OUT = ROOT / "web" / "data.json"
FLOW = {"n_shortlist": 25, "k_new": 10, "k_known": 5}  # pipeline.interactive_discovery


def build() -> dict:
    bank = json.loads((ROOT / "data" / "items" / "item_bank_v2.json").read_text("utf-8"))
    items = [{"id": it["item_id"], "stage": it["stage"], "dimension": it.get("dimension"),
              "group": it.get("group"), "stem": it["stem_zh"],
              "examples": it.get("examples_zh") or [], "labels": it["scale"]["labels_zh"]}
             for it in bank["items"]]  # fmt: skip
    dims = [{"key": d.key, "layer": d.layer, "label": d.label_zh} for d in DIMENSION_SETS["v2"]]
    if [d["key"] for d in dims] != list(NEED_KEYS_V2):
        raise SystemExit("dimension order mismatch")
    base = DEFAULT_SCORER_V2.base
    if base.intensity_gamma:
        raise SystemExit("the web engine supports base scorers without an intensity penalty")
    scorer = {"name": DEFAULT_SCORER_V2.name, "layers": list(LAYERS),
              "weights": dict(DEFAULT_SCORER_V2.weights), "cos_weight": base.cos_weight,
              "sq_weight": base.sq_weight}  # fmt: skip
    real = load_real_catalog_v0(ROOT / "data" / "reference")
    types, members = merge_identical_programs(real.programs)
    names = sorted({m.school for ms in members.values() for m in ms})
    index = {n: i for i, n in enumerate(names)}
    catalog = [{"id": p.program_id, "name": p.degree_name, "cip": p.cip,
                "vector": [float(v) for v in p.vector], "unmeasured": list(p.unmeasured),
                "schools": [index[m.school] for m in members[p.program_id]]}
               for p in types]  # fmt: skip
    first = real.programs[0]
    return {
        "version": 1,
        "items": items,
        "dims": dims,
        "groups": {str(g): label[0] for g, label in sorted(GROUP_LABELS.items())},
        "area_groups": {k: list(v) for k, v in AREA_GROUPS.items()},
        "base_rate": {k: float(v) for k, v in BASE_RATE.items()},
        "neutral": NEUTRAL_VALUE,
        "scorer": scorer,
        "catalog": catalog,
        "schools": names,
        "catalog_source": {"name": "IPEDS 2025 master's x O*NET (real catalogue v0)",
                           "source_url": first.source_url, "retrieved_at": first.retrieved_at,
                           "programmes": len(real.programs), "degree_types": len(types)},
        "flow": {**FLOW, "levels": list(LEVELS), "known": sorted(DEFAULT_KNOWN)},
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bundle", type=Path, default=None)
    args = ap.parse_args()
    text = json.dumps(build(), ensure_ascii=False, separators=(",", ":")) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text("utf-8") != text:
            raise SystemExit("web/data.json is stale: run python scripts/build_web_data.py")
        print("web/data.json is up to date")
        return
    if args.bundle:
        bundle(text, args.bundle)
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


def bundle(data_text: str, out: Path) -> None:
    """Inline engine.js and the data into index.html (no relative loads)."""
    page = (ROOT / "web" / "index.html").read_text("utf-8")
    engine = (ROOT / "web" / "engine.js").read_text("utf-8")
    tag = '<script src="engine.js"></script>'
    if page.count(tag) != 1:
        raise SystemExit("index.html must load engine.js exactly once")
    safe = data_text.strip().replace("</", "<\\/")
    page = page.replace(
        tag, f"<script>\n{engine}</script>\n<script>window.AISC_DATA = {safe};</script>"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
