"""W-80 persona benchmark with blind pooled judging (docs/evaluation-plan.md §2,
docs/persona-benchmark.md; V-C2-9; A-C2-61). Synthetic personas only.

Each persona (data/benchmark/personas.json) is turned into 73 questionnaire answers and
run through three methods on real catalogue v0 merged into degree types:
    MODEL  pipeline.interactive_discovery (default scorer and completion), every
           shortlisted type reported unknown, so its top 10 is the fit order
    B1     popularity: most schools first, answers ignored
    B2     interest only: cosine on the six RIASEC dimensions of the completed profile
Ties go to the lower id.

Judging (TREC-style pooling): the pool of a persona is the union of the three top-10
lists plus 5 random degree types (decoys), shuffled; nothing tells the rater which
method produced which type, and the pool file stores no sources (they are recomputed
when scoring). In the judging page (web/judge.html) a rater first names 2-3 degree
types that must appear (must-include, locked before the pool is shown), then grades
every pooled type 0 (does not fit), 1 (acceptable), 2 (fits well). Judgments are pasted
back as an "AISC-JUDGMENTS v1" block and saved as data/benchmark/judgments_<rater>.txt.

PRE-REGISTERED (v2, 2026-10-31; replaces v1 of 2026-10-29 before any judgment or run:
v1 asked raters to list expected CIP prefixes from all 863 types; v2 grades a blind pool
instead, adds must-include recall and decoys, and replaces "0 exclude-labelled types"
with the mean number of grade-0 types):
  Reference grades: the main rater's (brian). PASS if, over the personas judged,
    mean nDCG@10(MODEL) - mean nDCG@10(B1) >= 0.20,
    mean nDCG@10(MODEL) - mean nDCG@10(B2) >= 0,
    Hit@10(MODEL) (>= 1 grade-2 type in the top 10) for >= 90% of personas,
    mean number of grade-0 types in MODEL's top 10 <= 2,
    mean must-include recall@10(MODEL) >= 0.5 (share of a persona's must-include types
    that appear in MODEL's top 10).
  nDCG@10 uses gains 2^g - 1 and the ideal order of the persona's judged pool; an
  unjudged type in a top 10 (possible after a model change) counts 0 and the judged
  coverage is reported.
  Report only: P@10 (grade >= 1), CIP 4-digit series in the top 10, means by background
  (D12), the worst background, decoy mean grade (rater leniency; >= 1 is flagged), and
  on the personas both raters judged: quadratic weighted kappa and exact agreement of
  the grades, Jaccard of the must-include sets, and the PASS checks recomputed with the
  second rater's grades.
  REGRESSION (--check, CI): fails if mean nDCG@10(MODEL) falls by more than 0.05 from
  results/persona-benchmark.json or judged coverage of MODEL's top 10 drops below 90%
  (then add the new types to the pool and judge them).
  Guards: --build-pool refuses to overwrite a pool that has judgments; scoring refuses
  judgments that are not committed (--allow-uncommitted is for tests and writes nothing).

    python scripts/run_persona_benchmark.py --build-pool        # data/benchmark/pool.json
    python scripts/run_persona_benchmark.py --build-page OUT    # judging page (one file)
    python scripts/run_persona_benchmark.py --score [--second consultant]
    python scripts/run_persona_benchmark.py --check
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aisc.catalog.real import load_real_catalog_v0  # noqa: E402
from aisc.dimensions import DIMENSION_SETS, NEED_KEYS_V2, keys_in_layer  # noqa: E402
from aisc.pipeline import (  # noqa: E402
    interactive_discovery,
    merge_identical_programs,
    prepare_profile,
)
from aisc.profiling.hierarchy import AREA_GROUPS  # noqa: E402
from aisc.profiling.questionnaire import Questionnaire, load_questions  # noqa: E402

BENCH = ROOT / "data" / "benchmark"
POOL = BENCH / "pool.json"
OUT_MD = ROOT / "results" / "persona-benchmark.md"
OUT_JSON = ROOT / "results" / "persona-benchmark.json"
K, DECOYS, SEED = 10, 5, 2080
RULES = {"vs_b1": 0.20, "vs_b2": 0.0, "hit_rate": 0.90, "grade0": 2.0, "mi_recall": 0.5,
         "regression": 0.05, "coverage": 0.90}  # fmt: skip
BACKGROUNDS = ("same_field", "adjacent", "distant", "multi_degree")
BLOCK = re.compile(r"AISC-JUDGMENTS v1\s*\n\s*(\{.*?\})\s*\n\s*END", re.S)
SCHEMA = "aisc-judgments/1"
METHODS = ("MODEL", "B1", "B2")


# --- personas and methods -------------------------------------------------------------


def load_personas(path: Path = BENCH / "personas.json") -> list[dict]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for p in spec["personas"]:
        bad = [k for k in [*p["core"], *p["areas"]] if k not in NEED_KEYS_V2]
        if bad:
            raise SystemExit(f"{p['id']}: unknown dimensions {bad}")
        if p["background"] not in BACKGROUNDS:
            raise SystemExit(f"{p['id']}: background must be one of {BACKGROUNDS}")
        out.append({**p, "core_default": spec["core_default"]})
    return out


def persona_answers(p: dict, questions) -> dict[str, int]:
    """73 answers from a persona's levels (personas.json "note")."""
    area_level = {q.dimension: p["areas"].get(q.dimension, 1) for q in questions
                  if q.stage == "area"}  # fmt: skip
    out = {}
    for q in questions:
        if q.stage == "core":
            out[q.item_id] = p["core"].get(q.dimension, p["core_default"])
        elif q.stage == "area":
            out[q.item_id] = area_level[q.dimension]
        else:
            levels = [v for k, v in area_level.items() if AREA_GROUPS[k][0] == q.group]
            out[q.item_id] = max(levels, default=1)
    return out


def catalog():
    real = load_real_catalog_v0(ROOT / "data" / "reference").programs
    types, members = merge_identical_programs(real)
    return types, {t.program_id: len(members[t.program_id]) for t in types}


def _profile(p, questions):
    qs = Questionnaire(questions)
    for item_id, v in persona_answers(p, questions).items():
        qs.answer(item_id, v)
    return qs.profile(p["id"])


def top_model(profile, types):
    res = interactive_discovery(profile, types, lambda ids: dict.fromkeys(ids, "unknown"),
                                k_new=K, k_known=0)  # fmt: skip
    return [r.program.program_id for r in res.discoveries]


def top_popularity(types, n_schools):
    order = sorted(types, key=lambda t: (-n_schools[t.program_id], t.program_id))
    return [t.program_id for t in order[:K]]


def top_interest_only(profile, types):
    w = prepare_profile(profile, types).vector
    idx = [NEED_KEYS_V2.index(k) for k in keys_in_layer("interest", "v2")]
    wv = w[idx]
    X = np.array([t.vector[idx] for t in types])
    s = X @ wv / (np.linalg.norm(X, axis=1) * np.linalg.norm(wv) + 1e-12)
    order = sorted(range(len(types)), key=lambda i: (-s[i], types[i].program_id))
    return [types[i].program_id for i in order[:K]]


def method_tops(personas) -> dict[str, dict[str, list[str]]]:
    questions = load_questions(ROOT / "data" / "items" / "item_bank_v2.json")
    types, n_schools = catalog()
    b1 = top_popularity(types, n_schools)
    out = {}
    for p in personas:
        prof = _profile(p, questions)
        out[p["id"]] = {"MODEL": top_model(prof, types), "B1": b1,
                        "B2": top_interest_only(prof, types)}  # fmt: skip
    return out


# --- pool and judging page ------------------------------------------------------------


def build_pool(force: bool = False) -> dict:
    if POOL.exists() and any(BENCH.glob("judgments_*.txt")) and not force:
        raise SystemExit("judgments exist: extend the pool instead of rebuilding it")
    personas = load_personas()
    tops = method_tops(personas)
    types, _ = catalog()
    ids = sorted(t.program_id for t in types)
    rng = np.random.default_rng(SEED)
    pool = {}
    for p in personas:
        union = sorted({i for m in METHODS for i in tops[p["id"]][m]})
        rest = [i for i in ids if i not in set(union)]
        decoys = sorted(rng.choice(rest, size=DECOYS, replace=False).tolist())
        items = union + decoys
        pool[p["id"]] = [items[i] for i in rng.permutation(len(items))]
    spec = {"version": 1, "created": date.today().isoformat(), "seed": SEED,
            "note": "Shuffled pooled degree types per persona; sources are not stored.",
            "personas": pool}  # fmt: skip
    POOL.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    sizes = [len(v) for v in pool.values()]
    print(f"wrote {POOL} ({len(pool)} personas, {min(sizes)}-{max(sizes)} types each)")
    return spec


def page_data() -> dict:
    """Everything the judging page shows: personas, pools and the degree-type list."""
    labels = {d.key: d.label_zh for d in DIMENSION_SETS["v2"]}
    families = {}
    with open(ROOT / "data" / "reference" / "cip2020_all.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            families[row["family"]] = row["family_title"]
    types, n_schools = catalog()
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    personas = []
    for p in load_personas():
        core = sorted(p["core"].items(), key=lambda kv: -kv[1])
        hi = [f"{labels[k]} {v}" for k, v in core if v >= 4]
        areas = [f"{labels[k]} {v}" for k, v in sorted(p["areas"].items(), key=lambda kv: -kv[1])]
        personas.append({"id": p["id"], "background": p["background"],
                         "description": p["description"], "high": hi, "areas": areas})  # fmt: skip
    info = {t.program_id: {"name": t.degree_name, "cip": t.cip,
                           "family": families.get(t.cip[:2], ""),
                           "schools": n_schools[t.program_id]} for t in types}  # fmt: skip
    return {"pool_version": pool["version"], "personas": personas, "pool": pool["personas"],
            "types": info, "consultant": ["P02", "P06", "P09", "P13"]}  # fmt: skip


def build_page(out: Path, fragment: bool = True) -> None:
    page = (ROOT / "web" / "judge.html").read_text(encoding="utf-8")
    data = json.dumps(page_data(), ensure_ascii=False, separators=(",", ":"))
    blob = data.replace("<", "\\u003c")
    tag = "<script>window.JUDGE_DATA = null;</script>"
    if page.count(tag) != 1:
        raise SystemExit("judge.html must contain the data placeholder exactly once")
    page = page.replace(tag, f"<script>window.JUDGE_DATA = {blob};</script>")
    if not fragment:
        page = ('<!doctype html>\n<html lang="zh-Hant-TW">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                "</head>\n<body>\n" + page + "\n</body>\n</html>\n")  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


# --- judgments and scoring ------------------------------------------------------------


def load_judgments(path: Path) -> dict:
    """{persona_id: {"must": [type ids], "grades": {type id: 0|1|2}}} from pasted text."""
    found = BLOCK.findall(path.read_text(encoding="utf-8", errors="replace"))
    if not found:
        raise SystemExit(f"{path.name}: no AISC-JUDGMENTS v1 block")
    out: dict = {}
    for raw in found:  # later blocks override earlier ones per persona
        obj = json.loads(raw)
        if obj.get("schema") != SCHEMA:
            raise SystemExit(f"{path.name}: unknown schema {obj.get('schema')!r}")
        for pid, j in obj["personas"].items():
            grades = {k: int(v) for k, v in j["grades"].items()}
            if set(grades.values()) - {0, 1, 2}:
                raise SystemExit(f"{path.name}: {pid} has grades outside 0-2")
            out[pid] = {"must": list(j.get("must_include", [])), "grades": grades}
    return out


def require_committed(paths: list[Path]) -> None:
    for p in paths:
        rel = str(p.relative_to(ROOT))
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT,
                                 capture_output=True).returncode == 0  # fmt: skip
        clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel],
                               cwd=ROOT).returncode == 0  # fmt: skip
        if not (tracked and clean):
            raise SystemExit(f"{rel} must be committed before scoring (A-C2-61)")


def metrics(top: list[str], judged: dict[str, int], must: list[str], series) -> dict:
    gains = [judged.get(i, 0) for i in top]
    dcg = sum((2**g - 1) / math.log2(r + 2) for r, g in enumerate(gains))
    ideal = sorted(judged.values(), reverse=True)[:K]
    idcg = sum((2**g - 1) / math.log2(r + 2) for r, g in enumerate(ideal))
    return {
        "ndcg": dcg / idcg if idcg > 0 else float("nan"),
        "hit": any(g == 2 for g in gains),
        "precision": sum(g >= 1 for g in gains) / K,
        "grade0": sum(1 for i in top if judged.get(i) == 0),
        "coverage": sum(i in judged for i in top) / K,
        "mi_recall": (sum(m in top for m in must) / len(must)) if must else float("nan"),
        "series": len({series[i] for i in top}),
    }


def kappa_quadratic(a: list[int], b: list[int], k: int = 3) -> float:
    obs = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        obs[x, y] += 1
    if obs.sum() < 2:
        return float("nan")
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    den = (w * exp).sum()
    return float(1 - (w * obs).sum() / den) if den > 0 else float("nan")


def score(judgments: dict, tops: dict, personas: list[dict], series: dict) -> dict:
    rows = {}
    for p in personas:
        if p["id"] not in judgments:
            continue
        j = judgments[p["id"]]
        row = {"background": p["background"]}
        for m in METHODS:
            row[m] = metrics(tops[p["id"]][m], j["grades"], j["must"], series)
        rows[p["id"]] = row
    return rows


def summarise(rows: dict, decoys: float | None = None) -> dict:
    vals = list(rows.values())

    def mean(m, key):
        xs = [r[m][key] for r in vals if not math.isnan(r[m][key])]
        return float(np.mean(xs)) if xs else float("nan")

    keys = ("ndcg", "hit", "precision", "grade0", "coverage", "mi_recall")
    s = {m: {k: mean(m, k) for k in keys} for m in METHODS}
    model, b1, b2 = s["MODEL"], s["B1"], s["B2"]
    s["checks"] = {
        "vs_b1": model["ndcg"] - b1["ndcg"] >= RULES["vs_b1"],
        "vs_b2": model["ndcg"] - b2["ndcg"] >= RULES["vs_b2"],
        "hit_rate": model["hit"] >= RULES["hit_rate"],
        "grade0": model["grade0"] <= RULES["grade0"],
        "mi_recall": model["mi_recall"] >= RULES["mi_recall"],
    }
    s["pass"] = all(s["checks"].values())
    s["by_background"] = {}
    for b in BACKGROUNDS:
        xs = [r["MODEL"]["ndcg"] for r in vals if r["background"] == b]
        if xs:
            s["by_background"][b] = float(np.mean(xs))
    s["decoy_mean"] = decoys
    return s


def decoy_mean(judgments: dict, tops: dict) -> float | None:
    gs = []
    for pid, j in judgments.items():
        listed = {i for m in METHODS for i in tops.get(pid, {}).get(m, [])}
        gs += [g for i, g in j["grades"].items() if i not in listed]
    return float(np.mean(gs)) if gs else None


def agreement(main: dict, second: dict) -> dict:
    out = {}
    for pid in sorted(set(main) & set(second)):
        ga, gb = main[pid]["grades"], second[pid]["grades"]
        both = sorted(set(ga) & set(gb))
        a, b = [ga[i] for i in both], [gb[i] for i in both]
        ma, mb = set(main[pid]["must"]), set(second[pid]["must"])
        same = [x == y for x, y in zip(a, b, strict=True)]
        exact = float(np.mean(same)) if same else float("nan")
        jac = len(ma & mb) / len(ma | mb) if ma | mb else float("nan")
        out[pid] = {"n": len(both), "kappa": kappa_quadratic(a, b), "exact": exact,
                    "must_jaccard": jac}  # fmt: skip
    return out


def run_score(rater: str, second: str | None, allow_uncommitted: bool, bench: Path) -> dict:
    paths = [bench / f"judgments_{rater}.txt"]
    if second:
        paths.append(bench / f"judgments_{second}.txt")
    if not allow_uncommitted:
        require_committed(paths)
    main = load_judgments(paths[0])
    personas = [p for p in load_personas(bench / "personas.json") if p["id"] in main]
    if not personas:
        raise SystemExit(f"{paths[0].name} judges no persona")
    tops = method_tops(personas)
    types, _ = catalog()
    series = {t.program_id: t.cip[:5] for t in types}
    rows = score(main, tops, personas, series)
    res = {"rater": rater, "second": second, "rows": rows,
           "summary": summarise(rows, decoy_mean(main, tops))}  # fmt: skip
    if second:
        other = load_judgments(paths[1])
        res["agreement"] = agreement(main, other)
        rows2 = score(other, tops, [p for p in personas if p["id"] in other], series)
        res["second_summary"] = summarise(rows2, decoy_mean(other, tops))
    return res


def report(res: dict) -> str:
    s = res["summary"]
    ok = {True: "✅", False: "❌"}
    m, b1, b2, chk = s["MODEL"], s["B1"], s["B2"], s["checks"]
    L = [
        "# 人物設定基準（W-80，V-C2-9）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 評分者：{res['rater']}"
        f"（{len(res['rows'])} 個人物）· 不公開來源的評分池；預先登記見腳本檔頭",
        "- 合成人物；學程為真實目錄 v0（強度與取向未測量）",
        "",
        f"## 判定：{'**通過**' if s['pass'] else '**未通過**'}",
        "",
        "| 判準 | 結果 |",
        "|---|---|",
        f"| nDCG@10 比熱門（B1）高 ≥ 0.20 | {m['ndcg'] - b1['ndcg']:+.3f} {ok[chk['vs_b1']]} |",
        f"| nDCG@10 不低於只看興趣（B2） | {m['ndcg'] - b2['ndcg']:+.3f} {ok[chk['vs_b2']]} |",
        f"| Hit@10 ≥ 90% | {m['hit']:.0%} {ok[chk['hit_rate']]} |",
        f"| 前 10 名評 0 分的平均個數 ≤ 2 | {m['grade0']:.1f} {ok[chk['grade0']]} |",
        f"| 一定要出現的學門進前 10 名 ≥ 50% | {m['mi_recall']:.0%} {ok[chk['mi_recall']]} |",
        "",
        "| 方法 | nDCG@10 | Hit@10 | P@10 | 評 0 分個數 | 已評分比例 | 一定要出現的命中 |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, name in (("MODEL", "模型"), ("B1", "B1 熱門"), ("B2", "B2 只看興趣")):
        x = s[key]
        L.append(
            f"| {name} | {x['ndcg']:.3f} | {x['hit']:.0%} | {x['precision']:.2f} | "
            f"{x['grade0']:.1f} | {x['coverage']:.0%} | {x['mi_recall']:.0%} |"
        )
    L += [
        "",
        "| 人物 | 背景 | nDCG@10 | Hit | P@10 | 評 0 分 | 系列數 | B1 | B2 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for pid, r in res["rows"].items():
        x = r["MODEL"]
        L.append(
            f"| {pid} | {r['background']} | {x['ndcg']:.3f} | {ok[x['hit']]} | "
            f"{x['precision']:.1f} | {x['grade0']} | {x['series']} | "
            f"{r['B1']['ndcg']:.3f} | {r['B2']['ndcg']:.3f} |"
        )
    by = s["by_background"]
    if by:
        worst = min(by, key=by.get)
        parts = "、".join(f"{b} {v:.3f}" for b, v in by.items())
        L += ["", f"各背景平均 nDCG@10（D12，只報告）：{parts}；最差：{worst}"]
    if s["decoy_mean"] is not None:
        flag = "（偏寬鬆，請注意）" if s["decoy_mean"] >= 1 else ""
        L += [f"隨機學門的平均評分（評分寬嚴，只報告）：{s['decoy_mean']:.2f}{flag}"]
    if res.get("agreement"):
        L += [
            "",
            f"## 評分者一致性（{res['rater']} vs {res['second']}，只報告）",
            "",
            "| 人物 | 共同評分數 | 加權 κ | 完全一致 | 一定要出現的重疊 |",
            "|---|---|---|---|---|",
        ]
        for pid, a in res["agreement"].items():
            L.append(
                f"| {pid} | {a['n']} | {a['kappa']:.2f} | {a['exact']:.0%} | "
                f"{a['must_jaccard']:.2f} |"
            )
        checks = res["second_summary"]["checks"]
        parts = "、".join(f"{k} {ok[v]}" for k, v in checks.items())
        L += ["", f"用 {res['second']} 的評分重算判準（只報告）：{parts}"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-pool", action="store_true")
    ap.add_argument("--build-page", type=Path, default=None)
    ap.add_argument("--standalone", action="store_true", help="page as a full HTML document")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--rater", default="brian")
    ap.add_argument("--second", default=None)
    ap.add_argument("--allow-uncommitted", action="store_true", help="tests only; writes nothing")
    ap.add_argument("--bench-dir", type=Path, default=BENCH, help="tests only")
    ap.add_argument("--json-out", type=Path, default=None, help="tests only")
    args = ap.parse_args()
    if args.bench_dir != BENCH and not args.allow_uncommitted:
        raise SystemExit("--bench-dir is for tests (with --allow-uncommitted)")
    if args.build_pool:
        build_pool()
    elif args.build_page:
        build_page(args.build_page, fragment=not args.standalone)
    elif args.check:
        if not OUT_JSON.exists():
            print("no stored benchmark yet; nothing to check")
            return
        old = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        res = run_score(old["rater"], None, False, BENCH)
        new, prev = res["summary"]["MODEL"], old["summary"]["MODEL"]
        drop = prev["ndcg"] - new["ndcg"]
        if drop > RULES["regression"] or new["coverage"] < RULES["coverage"]:
            raise SystemExit(
                f"benchmark regression: nDCG@10 drop {drop:+.3f}, "
                f"judged coverage {new['coverage']:.0%}"
            )
        print(f"benchmark ok: nDCG@10 {new['ndcg']:.3f} (stored {prev['ndcg']:.3f})")
    elif args.score:
        res = run_score(args.rater, args.second, args.allow_uncommitted, args.bench_dir)
        if args.allow_uncommitted:
            if args.json_out:
                args.json_out.write_text(json.dumps(res), encoding="utf-8")
            print(f"pass={res['summary']['pass']} (test run, nothing written to results/)")
            return
        OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        OUT_MD.write_text(report(res), encoding="utf-8")
        print(f"pass={res['summary']['pass']} wrote {OUT_MD}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
