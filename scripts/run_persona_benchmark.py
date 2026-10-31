"""W-80 persona benchmark (docs/evaluation-plan.md §2, docs/persona-benchmark.md;
V-C2-9; A-C2-61). Synthetic personas only.

Each persona (data/benchmark/personas.json) is turned into 73 questionnaire answers,
run through the default pipeline on real catalogue v0 merged into degree types, and the
"you may not have thought of" top 10 is scored against expectations written BEFORE the
first run by a rater (data/benchmark/expectations_<rater>.csv):
    persona_id, cip_prefix, label        label in core (2), acceptable (1), exclude
A degree type takes the label of its longest matching CIP prefix ("27", "27.05",
"27.0501"); unlabelled types score 0. Familiarity is "unknown" for every shortlisted
type, so the top 10 is the fit order.

Methods: MODEL (pipeline.interactive_discovery, default scorer and completion);
B1 popularity (most schools first, answers ignored); B2 interest only (cosine on the six
RIASEC dimensions of the completed profile). Ties go to the lower id.

PRE-REGISTERED (docs/evaluation-plan.md §2.4; fixed 2026-10-29, before any run):
  PASS if, over the personas the main rater labelled,
    mean nDCG@10(MODEL) - mean nDCG@10(B1) >= 0.20,
    mean nDCG@10(MODEL) - mean nDCG@10(B2) >= 0,
    Hit@10(MODEL) (>= 1 core type in the top 10) for >= 90% of personas,
    exclude-labelled types in MODEL's top 10: 0 in total.
  Report only: P@10, CIP 4-digit series covered, means by background (D12), the worst
  background, rater agreement on the personas both raters labelled (quadratic weighted
  kappa over the labelled degree types, Jaccard of the core sets).
  REGRESSION (--check, CI): fails if mean nDCG@10(MODEL) falls by more than 0.05 from
  results/persona-benchmark.json or a new exclude-labelled type enters a top 10.
  Guard: the expectations files must be committed and unchanged before a run
  (--allow-uncommitted is for tests only and writes nothing).

    python scripts/run_persona_benchmark.py --rater brian [--second consultant]
    python scripts/run_persona_benchmark.py --check
    python scripts/run_persona_benchmark.py --write-reference   # degree_types.csv for raters
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aisc.catalog.real import load_real_catalog_v0  # noqa: E402
from aisc.dimensions import NEED_KEYS_V2, keys_in_layer  # noqa: E402
from aisc.pipeline import (  # noqa: E402
    interactive_discovery,
    merge_identical_programs,
    prepare_profile,  # noqa: E402
)
from aisc.profiling.hierarchy import AREA_GROUPS  # noqa: E402
from aisc.profiling.questionnaire import Questionnaire, load_questions  # noqa: E402

BENCH = ROOT / "data" / "benchmark"
OUT_MD = ROOT / "results" / "persona-benchmark.md"
OUT_JSON = ROOT / "results" / "persona-benchmark.json"
GAIN = {"core": 2, "acceptable": 1, "exclude": 0}
K = 10
RULES = {"vs_b1": 0.20, "vs_b2": 0.0, "hit_rate": 0.90, "violations": 0, "regression": 0.05}
BACKGROUNDS = ("same_field", "adjacent", "distant", "multi_degree")


# --- inputs ---------------------------------------------------------------------------


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


def load_expectations(path: Path) -> dict[str, list[tuple[str, str]]]:
    """{persona_id: [(cip_prefix, label), ...]} from a rater's CSV ('#' lines ignored)."""
    lines = [ln for ln in path.read_text(encoding="utf-8-sig").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]  # fmt: skip
    out: dict[str, list[tuple[str, str]]] = {}
    for n, row in enumerate(csv.DictReader(lines), start=2):
        pid, prefix, label = (row.get(c, "").strip() for c in ("persona_id", "cip_prefix", "label"))
        if not (pid and prefix and label):
            continue
        if label not in GAIN:
            raise SystemExit(f"{path.name}:{n}: label must be core, acceptable or exclude")
        out.setdefault(pid, []).append((prefix, label))
    return out


def label_of(cip: str, labels: list[tuple[str, str]]) -> str | None:
    best = None
    for prefix, label in labels:
        if cip.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, label)
    return best[1] if best else None


def require_committed(paths: list[Path]) -> None:
    """Pre-registration guard: expectations must be committed and unchanged."""
    for p in paths:
        rel = str(p.relative_to(ROOT))
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT,
                                 capture_output=True).returncode == 0  # fmt: skip
        clean = (
            subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
        )
        if not (tracked and clean):
            raise SystemExit(f"{rel} must be committed before the benchmark runs (A-C2-61)")


# --- methods and metrics --------------------------------------------------------------


def catalog():
    types, members = merge_identical_programs(
        load_real_catalog_v0(ROOT / "data" / "reference").programs
    )
    return types, {t.program_id: len(members[t.program_id]) for t in types}


def top_model(answers, questions, types, pid):
    qs = Questionnaire(questions)
    for item_id, v in answers.items():
        qs.answer(item_id, v)
    res = interactive_discovery(qs.profile(pid), types, lambda ids: dict.fromkeys(ids, "unknown"),
                                k_new=K, k_known=0)  # fmt: skip
    return [r.program for r in res.discoveries], qs.profile(pid)


def top_popularity(types, n_schools):
    return sorted(types, key=lambda t: (-n_schools[t.program_id], t.program_id))[:K]


def top_interest_only(profile, types):
    w = prepare_profile(profile, types).vector
    idx = [NEED_KEYS_V2.index(k) for k in keys_in_layer("interest", "v2")]
    wv = w[idx]
    X = np.array([t.vector[idx] for t in types])
    s = X @ wv / (np.linalg.norm(X, axis=1) * np.linalg.norm(wv) + 1e-12)
    order = sorted(range(len(types)), key=lambda i: (-s[i], types[i].program_id))
    return [types[i] for i in order[:K]]


def metrics(top, labels, types) -> dict:
    gains = [GAIN.get(label_of(t.cip, labels) or "", 0) for t in top]
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((GAIN.get(label_of(t.cip, labels) or "", 0) for t in types), reverse=True)[:K]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return {
        "ndcg": dcg / idcg if idcg > 0 else float("nan"),
        "hit": any(g == 2 for g in gains),
        "precision": sum(g > 0 for g in gains) / K,
        "violations": [t.cip for t in top if label_of(t.cip, labels) == "exclude"],
        "series": len({t.cip[:5] for t in top}),
        "top": [t.program_id for t in top],
    }


def agreement(a: list[tuple[str, str]], b: list[tuple[str, str]], types) -> dict:
    """Quadratic weighted kappa over degree types either rater labelled, on the ordinal
    scale exclude < none < acceptable < core; Jaccard of the core sets."""
    score = {"exclude": 0, None: 1, "acceptable": 2, "core": 3}
    pairs = [(score[label_of(t.cip, a)], score[label_of(t.cip, b)]) for t in types
             if label_of(t.cip, a) or label_of(t.cip, b)]  # fmt: skip
    if len(pairs) < 2:
        return {"kappa": float("nan"), "core_jaccard": float("nan"), "n": len(pairs)}
    k = 4
    obs = np.zeros((k, k))
    for x, y in pairs:
        obs[x, y] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    kappa = 1 - (w * obs).sum() / (w * exp).sum() if (w * exp).sum() > 0 else float("nan")
    ca = {t.program_id for t in types if label_of(t.cip, a) == "core"}
    cb = {t.program_id for t in types if label_of(t.cip, b) == "core"}
    jac = len(ca & cb) / len(ca | cb) if ca | cb else float("nan")
    return {"kappa": float(kappa), "core_jaccard": jac, "n": len(pairs)}


# --- run ------------------------------------------------------------------------------


def run(rater: str, second: str | None, allow_uncommitted: bool, bench: Path = BENCH) -> dict:
    exp_path = bench / f"expectations_{rater}.csv"
    paths = [exp_path] + ([bench / f"expectations_{second}.csv"] if second else [])
    if not allow_uncommitted:
        require_committed(paths)
    expectations = load_expectations(exp_path)
    personas = [p for p in load_personas(bench / "personas.json") if p["id"] in expectations]
    if not personas:
        raise SystemExit(
            f"{exp_path.name} has no labelled personas yet (docs/persona-benchmark.md)"
        )
    questions = load_questions(ROOT / "data" / "items" / "item_bank_v2.json")
    types, n_schools = catalog()
    rows = {}
    for p in personas:
        labels = expectations[p["id"]]
        model_top, profile = top_model(persona_answers(p, questions), questions, types, p["id"])
        rows[p["id"]] = {
            "background": p["background"],
            "MODEL": metrics(model_top, labels, types),
            "B1": metrics(top_popularity(types, n_schools), labels, types),
            "B2": metrics(top_interest_only(profile, types), labels, types),
        }
    agree = {}
    if second:
        other = load_expectations(bench / f"expectations_{second}.csv")
        for pid in sorted(set(other) & set(expectations)):
            agree[pid] = agreement(expectations[pid], other[pid], types)
    return {"rater": rater, "second": second, "rows": rows, "agreement": agree}


def summarise(res: dict) -> dict:
    rows = list(res["rows"].values())

    def mean(method, key):
        vals = [r[method][key] for r in rows if not math.isnan(r[method]["ndcg"])]
        return float(np.mean(vals)) if vals else float("nan")

    s = {}
    for m in ("MODEL", "B1", "B2"):
        s[m] = {
            "ndcg": mean(m, "ndcg"),
            "hit_rate": mean(m, "hit"),
            "precision": mean(m, "precision"),
            "violations": sum(len(r[m]["violations"]) for r in rows),
        }
    s["by_background"] = {}
    for b in BACKGROUNDS:
        vals = [r["MODEL"]["ndcg"] for r in rows if r["background"] == b]
        if vals:
            s["by_background"][b] = float(np.mean(vals))
    model, b1, b2 = s["MODEL"], s["B1"], s["B2"]
    s["checks"] = {
        "vs_b1": model["ndcg"] - b1["ndcg"] >= RULES["vs_b1"],
        "vs_b2": model["ndcg"] - b2["ndcg"] >= RULES["vs_b2"],
        "hit_rate": model["hit_rate"] >= RULES["hit_rate"],
        "violations": model["violations"] <= RULES["violations"],
    }
    s["pass"] = all(s["checks"].values())
    return s


def report(res: dict, s: dict) -> str:
    ok = {True: "✅", False: "❌"}
    model, b1, b2, chk = s["MODEL"], s["B1"], s["B2"], s["checks"]
    L = [
        "# 人物設定基準（W-80，V-C2-9）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 預期答案：`expectations_{res['rater']}.csv`"
        f"（{len(res['rows'])} 個人物）· 預先登記見 `scripts/run_persona_benchmark.py` 檔頭",
        "- 合成人物；學程為真實目錄 v0（強度與取向未測量）",
        "",
        f"## 判定：{'**通過**' if s['pass'] else '**未通過**'}",
        "",
        "| 判準 | 結果 |",
        "|---|---|",
        f"| nDCG@10 比熱門（B1）高 ≥ 0.20 | {model['ndcg'] - b1['ndcg']:+.3f} {ok[chk['vs_b1']]} |",
        f"| nDCG@10 不低於只看興趣（B2） | {model['ndcg'] - b2['ndcg']:+.3f} {ok[chk['vs_b2']]} |",
        f"| Hit@10 ≥ 90% | {model['hit_rate']:.0%} {ok[chk['hit_rate']]} |",
        f"| 違規 0 | {model['violations']} {ok[chk['violations']]} |",
        "",
        "| 方法 | 平均 nDCG@10 | Hit@10 | P@10 | 違規 |",
        "|---|---|---|---|---|",
    ]
    for m, name in (("MODEL", "模型"), ("B1", "B1 熱門"), ("B2", "B2 只看興趣")):
        x = s[m]
        L.append(
            f"| {name} | {x['ndcg']:.3f} | {x['hit_rate']:.0%} | {x['precision']:.2f} | "
            f"{x['violations']} |"
        )
    L += [
        "",
        "| 人物 | 背景 | nDCG@10 | Hit | P@10 | 違規 | 學門系列數 | B1 | B2 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for pid, r in res["rows"].items():
        m = r["MODEL"]
        viol = ", ".join(m["violations"]) or "0"
        L.append(
            f"| {pid} | {r['background']} | {m['ndcg']:.3f} | {ok[m['hit']]} | "
            f"{m['precision']:.1f} | {viol} | {m['series']} | {r['B1']['ndcg']:.3f} | "
            f"{r['B2']['ndcg']:.3f} |"
        )
    by = s["by_background"]
    worst = min(by, key=by.get)
    parts = "、".join(f"{b} {v:.3f}" for b, v in by.items())
    L += ["", f"各背景平均 nDCG@10（D12，只報告）：{parts}；最差：{worst}"]
    if res["agreement"]:
        L += [
            "",
            f"## 評分者一致性（{res['rater']} vs {res['second']}，只報告）",
            "",
            "| 人物 | 加權 κ | 核心集合 Jaccard | 標記的學位類型數 |",
            "|---|---|---|---|",
        ]
        for pid, a in res["agreement"].items():
            L.append(f"| {pid} | {a['kappa']:.2f} | {a['core_jaccard']:.2f} | {a['n']} |")
    return "\n".join(L) + "\n"


def write_reference() -> None:
    """data/benchmark/degree_types.csv: every degree type a rater can label (no model output)."""
    families = {}
    with open(ROOT / "data" / "reference" / "cip2020_all.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            families[row["family"]] = row["family_title"]
    types, n_schools = catalog()
    path = BENCH / "degree_types.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cip", "series", "family", "family_title", "degree_name", "n_schools"])
        for t in sorted(types, key=lambda t: t.cip):
            fam = t.cip[:2]
            w.writerow([t.cip, t.cip[:5], fam, families.get(fam, ""), t.degree_name,
                        n_schools[t.program_id]])  # fmt: skip
    print(f"wrote {path} ({len(types)} degree types)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rater", default="brian")
    ap.add_argument("--second", default=None)
    ap.add_argument("--check", action="store_true", help="regression check against results")
    ap.add_argument("--allow-uncommitted", action="store_true", help="tests only; writes nothing")
    ap.add_argument("--json-out", type=Path, default=None, help="tests only")
    ap.add_argument("--bench-dir", type=Path, default=BENCH, help="tests only")
    ap.add_argument("--write-reference", action="store_true")
    args = ap.parse_args()
    if args.write_reference:
        write_reference()
        return
    if args.check:
        if not OUT_JSON.exists():
            print("no stored benchmark yet; nothing to check")
            return
        old = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        res = run(old["rater"], None, allow_uncommitted=False)
        s = summarise(res)
        drop = old["summary"]["MODEL"]["ndcg"] - s["MODEL"]["ndcg"]
        new_v = {pid: set(r["MODEL"]["violations"]) - set(old["rows"].get(pid, {}).get("MODEL", {})
                 .get("violations", [])) for pid, r in res["rows"].items()}  # fmt: skip
        new_v = {k: v for k, v in new_v.items() if v}
        if drop > RULES["regression"] or new_v:
            raise SystemExit(
                f"benchmark regression: nDCG@10 drop {drop:+.3f}, new violations {new_v}"
            )
        stored = old["summary"]["MODEL"]["ndcg"]
        print(f"benchmark ok: nDCG@10 {s['MODEL']['ndcg']:.3f} (stored {stored:.3f})")
        return
    if args.bench_dir != BENCH and not args.allow_uncommitted:
        raise SystemExit("--bench-dir is for tests (with --allow-uncommitted)")
    res = run(args.rater, args.second, args.allow_uncommitted, args.bench_dir)
    s = summarise(res)
    if args.allow_uncommitted:
        if args.json_out:
            args.json_out.write_text(json.dumps({**res, "summary": s}), encoding="utf-8")
        print(f"pass={s['pass']} (test run, nothing written to results/)")
        return
    OUT_JSON.write_text(
        json.dumps({**res, "summary": s}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    OUT_MD.write_text(report(res, s), encoding="utf-8")
    print(f"pass={s['pass']} wrote {OUT_MD}")


if __name__ == "__main__":
    main()
