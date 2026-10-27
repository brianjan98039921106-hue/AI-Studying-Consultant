"""Pre-registered analysis of the W-41 field study (docs/study-protocol.md §4; THEORY
"實測分析計畫"; A-C1-25, A-C1-27, A-C1-29). Reads only de-identified files.

FILES (default folder data/private/study/; headers in data/templates/study_*.csv)
  s1.csv     respondent_id, stratum, [synthetic], one column per item_id of
             data/items/item_bank_v2.json (73 items), answers 1-5, blank = not answered
  s2.csv     respondent_id, [synthetic], same item columns (retest, 1-2 weeks later)
  pairs.csv  respondent_id, session (s1/s2), pair_id, chosen (stimulus id), confidence 1-4
  stratum is one of same_field / adjacent / distant / multi_degree (protocol §2).
  Answers are scored (a - 1) / 4; a dimension score is the mean of its items.

PRE-REGISTERED (written and committed before any study data exist, 2026-10-28).
Binding only with >= 200 S1 and >= 60 S2 respondents (protocol §2); with fewer, every
verdict is printed as "pilot, not binding".
  1 Internal consistency (V-C1-5): Cronbach's alpha of each multi-item dimension (6
    RIASEC, 2 intensity, 2 orientation) on S1. PASS if alpha >= 0.70; reported with a
    respondent bootstrap 95% CI (2000 resamples, seed 2026); "uncertain" when the CI
    contains 0.70.
  2 Retest reliability (V-C1-6): Pearson r of each dimension score (51 dimensions and 12
    group items) between S1 and S2. PASS if r >= 0.70, bootstrap CI as in 1.
    Systematic change (A-C1-29): items whose mean S2 - S1 difference has |t| > 3 are
    listed.
  3 Answer noise: per item sigma = SD(S1 - S2) / sqrt(2) on the unit scale; pooled per
    layer as sqrt(mean of item variances). Reported: share of items with sigma > 0.30
    (F89). The pooled area + group sigma feeds analysis 7.
  4 Dimension overlap (V-C1-7): S1 correlations between the 51 dimension scores; PASS if
    no pair has |r| >= 0.80 (listed otherwise).
  5 Choice consistency: agreement of the chosen stimulus in pairs shown in both sessions
    (same respondent and pair_id). If the Wilson 95% CI excludes 0.75 (the simulation
    assumption), the W-76 parameter recovery is re-run with the observed rate.
  6 Area completion (re-test of F105-F108 on students): S1 respondents with all 41 area
    and 12 group items. 10-fold CV over respondents (seed 2026). Each respondent answers
    the 12 groups, then the first q in {4, 8, 12} areas of areas_to_ask (the shipped
    two-stage order). Unanswered areas are predicted by
        H    shipped hierarchy prior (area_prior)
        G2   shipped correlated completion (correlated_area_priors, answer_sd 0.15)
        G2s  conditional normal from the training folds' answers (Ledoit-Wolf over 41
             areas + 12 groups; noise is already in that covariance, jitter 1e-6)
    Target: the respondent's own S1 answers on the unanswered areas (their noise is
    independent of every predictor, so paired MSE differences are unbiased).
    Primary metric RMSE over the unanswered areas; guard top-5 recall of the completed
    profile against the full S1 answers, with ties broken at random in expectation.
    Rule as F105: a candidate QUALIFIES if in all three q its paired RMSE difference vs H
    is below -2 SE and its recall is not lower than H's by more than 2 SE.
    G2 qualifies -> correlated completion is recommended as the default.
    Only G2s qualifies -> rebuild the area model from student answers (provisional until
    confirmed on the next cohort). Neither -> H stays.
  7 Denoising answered areas (re-test of F117-F119): S2 respondents with complete areas
    and groups in both sessions, same folds and q. TS-G2 = answered areas raw, the rest
    from the shipped correlated model; TS-G2D = posterior mean of all 41 areas under the
    shipped joint model with noise variance sigma^2 on every observed answer, sigma
    pooled over area and group items from the training folds' S1 - S2 differences;
    clipped to [0, 1]. Target: S2 answers on all 41 areas (A-C1-29).
    PRIMARY top-5 recall (fixed before the study, F118); guard RMSE.
    TS-G2D QUALIFIES if (a) the per-respondent mean over q of its recall difference vs
    TS-G2 is above +2 SE, (b) in no q is that difference below -2 SE and (c) in no q is
    its RMSE higher by more than 2 SE. Qualifies -> denoising becomes an option of the
    correlated completion (default only if analysis 6 also recommends G2).
    Report only: sigma x 0.7 (protocol §4 risk), and F115: RMSE of TS-G2 with q = 4 vs H
    with q = 12 (answered raw, rest from H), both against S2.
  8 Strata (D12), report only: the paired differences of 6 and 7 by stratum.
NOT IN THIS SCRIPT: measurement invariance / DIF (V-C1-9, W-73: own pre-registration),
layer shares (W-76: method still open, protocol §8), familiarity reporting (needs a
ground-truth design).
  Disclosure: dry-run only on fake data from --make-fake (exit status and layout).

    python scripts/analyze_w41_study.py --data data/private/study
    python scripts/analyze_w41_study.py --make-fake <folder>   # synthetic files
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

from aisc.dimensions import BASIC_INTEREST_KEYS
from aisc.profiling.hierarchy import (
    AREA_GROUPS,
    GROUP_LABELS,
    area_prior,
    areas_to_ask,
    correlated_area_priors,
)

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "items" / "item_bank_v2.json"
MODEL = ROOT / "src" / "aisc" / "profiling" / "interest_area_model.json"
STRATA = ("same_field", "adjacent", "distant", "multi_degree")
QS = (4, 8, 12)
FOLDS, SEED, BOOT = 10, 2026, 2000
MIN_S1, MIN_S2 = 200, 60
GROUPS = sorted(GROUP_LABELS)
D = len(BASIC_INTEREST_KEYS)


# --- item bank and files ---------------------------------------------------------------


def item_bank() -> list[dict]:
    return json.loads(BANK.read_text(encoding="utf-8"))["items"]


def dim_items() -> dict[str, list[str]]:
    """{dimension or group_g: [item_id, ...]} in bank order."""
    out: dict[str, list[str]] = defaultdict(list)
    for it in item_bank():
        key = it["dimension"] if it["dimension"] else f"group_{it['group']}"
        out[key].append(it["item_id"])
    return dict(out)


def read_answers(path: Path, items: list[str], need_stratum: bool) -> dict[str, dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    miss = {"respondent_id", *items} - set(rows[0])
    if need_stratum:
        miss |= {"stratum"} - set(rows[0])
    if miss:
        raise SystemExit(f"{path}: missing columns {sorted(miss)[:5]}...")
    out = {}
    for n, r in enumerate(rows, start=2):
        rid = r["respondent_id"].strip()
        if not rid or rid in out:
            raise SystemExit(f"{path}:{n}: respondent_id must be present and unique")
        ans = {}
        for it in items:
            v = (r[it] or "").strip()
            if v == "":
                continue
            if v not in {"1", "2", "3", "4", "5"}:
                raise SystemExit(f"{path}:{n}: {it} = {v!r} is not 1-5")
            ans[it] = (int(v) - 1) / 4
        st = (r.get("stratum") or "").strip()
        if need_stratum and st not in STRATA:
            raise SystemExit(f"{path}:{n}: stratum {st!r} not in {STRATA}")
        syn = (r.get("synthetic") or "").strip().lower() == "true"
        out[rid] = {"ans": ans, "stratum": st, "synthetic": syn}
    return out


def read_pairs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for n, r in enumerate(rows, start=2):
        if r.get("session") not in ("s1", "s2") or not (r.get("chosen") or "").strip():
            raise SystemExit(f"{path}:{n}: session must be s1/s2 and chosen non-empty")
    return rows


# --- statistics ------------------------------------------------------------------------


def cronbach_alpha(X: np.ndarray) -> float:
    k = X.shape[1]
    tot = X.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / tot)) if tot > 0 else math.nan


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def boot_ci(fn, n: int, rng: np.random.Generator) -> tuple[float, float]:
    vals = []
    for _ in range(BOOT):
        v = fn(rng.integers(0, n, n))
        if not math.isnan(v):
            vals.append(v)
    if len(vals) < BOOT // 2:
        return math.nan, math.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    z, p = 1.959964, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def top5_probs(v: np.ndarray) -> np.ndarray:
    """P(each entry is among the 5 highest) when ties are broken uniformly at random."""
    r = np.round(v, 9)
    t = np.sort(r)[::-1][4]
    above, eq = r > t, r == t
    p = above.astype(float)
    p[eq] = (5 - above.sum()) / eq.sum()
    return p


def top5_recall(pred: np.ndarray, target: np.ndarray) -> float:
    """Expected overlap of the two top-5 sets / 5, ties broken at random independently."""
    return float(top5_probs(pred) @ top5_probs(target) / 5)


def paired(a, b) -> tuple[float, float]:
    d = np.asarray(a) - np.asarray(b)
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def cond_mean(mu, S, obs, y, s2):
    Soo = S[np.ix_(obs, obs)] + np.eye(len(obs)) * s2
    return mu + S[:, obs] @ np.linalg.solve(Soo, y - mu[obs])


# --- analyses --------------------------------------------------------------------------


def area_vectors(resp: dict, dims: dict[str, list[str]]):
    """41 area answers and {group: answer} or None when any is missing."""
    a = resp["ans"]
    items = [dims[k][0] for k in BASIC_INTEREST_KEYS]
    gitems = [dims[f"group_{g}"][0] for g in GROUPS]
    if any(i not in a for i in items + gitems):
        return None
    return np.array([a[i] for i in items]), {g: a[dims[f"group_{g}"][0]] for g in GROUPS}


def completion(s1: dict, dims) -> dict:
    ids = [r for r in s1 if area_vectors(s1[r], dims) is not None]
    V = {r: area_vectors(s1[r], dims) for r in ids}
    fold = np.random.default_rng(SEED).permutation(np.arange(len(ids)) % FOLDS)
    res = {q: {m: {"rmse": [], "rec": [], "st": []} for m in ("H", "G2", "G2s")} for q in QS}
    gidx = D + np.arange(len(GROUPS))
    for f in range(FOLDS):
        tr = [ids[i] for i in np.flatnonzero(fold != f)]
        Z = np.array([np.concatenate([V[r][0], [V[r][1][g] for g in GROUPS]]) for r in tr])
        mu, S = Z.mean(0), LedoitWolf().fit(Z).covariance_
        for i in np.flatnonzero(fold == f):
            rid = ids[i]
            y, gs = V[rid]
            gv = np.array([gs[g] for g in GROUPS])
            for q in QS:
                asked = [BASIC_INTEREST_KEYS.index(k) for k in areas_to_ask(gs, q=q)]
                un = [j for j in range(D) if j not in asked]
                rated = {BASIC_INTEREST_KEYS[j]: float(y[j]) for j in asked}
                pri = correlated_area_priors(rated, gs)
                obs = np.concatenate([asked, gidx]).astype(int)
                g2s = np.clip(cond_mean(mu, S, obs, np.concatenate([y[asked], gv]), 1e-6), 0, 1)
                preds = {
                    "H": [area_prior(BASIC_INTEREST_KEYS[j], gs)[0] for j in un],
                    "G2": [pri[BASIC_INTEREST_KEYS[j]][0] for j in un],
                    "G2s": [g2s[j] for j in un],
                }
                for m, p in preds.items():
                    full = y.copy()
                    full[un] = p
                    r = res[q][m]
                    r["rmse"].append(math.sqrt(float(np.mean((np.array(p) - y[un]) ** 2))))
                    r["rec"].append(top5_recall(full, y))
                    r["st"].append(s1[rid]["stratum"])
    res["n"] = len(ids)
    return res


def denoise(s1: dict, s2: dict, dims, sigma_items: list[str]) -> dict:
    ids = [r for r in s1 if r in s2 and area_vectors(s1[r], dims) and area_vectors(s2[r], dims)]
    fold = np.random.default_rng(SEED).permutation(np.arange(len(ids)) % FOLDS)
    spec = json.loads(MODEL.read_text(encoding="utf-8"))
    mu, S = np.array(spec["mean"]), np.array(spec["covariance"])
    gidx = D + np.arange(len(GROUPS))
    keys = ("G2", "G2D", "G2D07", "H12")
    res = {q: {m: {"rmse": [], "rec": [], "st": []} for m in keys} for q in QS}
    for f in range(FOLDS):
        tr = [ids[i] for i in np.flatnonzero(fold != f)]
        d = [s1[r]["ans"][i] - s2[r]["ans"][i] for r in tr for i in sigma_items]
        sig = float(np.std(d, ddof=1) / math.sqrt(2)) if len(d) > 1 else 0.15
        for i in np.flatnonzero(fold == f):
            rid = ids[i]
            y, gs = area_vectors(s1[rid], dims)
            t = area_vectors(s2[rid], dims)[0]
            gv = np.array([gs[g] for g in GROUPS])
            for q in QS:
                asked = [BASIC_INTEREST_KEYS.index(k) for k in areas_to_ask(gs, q=q)]
                un = [j for j in range(D) if j not in asked]
                rated = {BASIC_INTEREST_KEYS[j]: float(y[j]) for j in asked}
                pri = correlated_area_priors(rated, gs)
                raw = y.copy()
                raw[un] = [pri[BASIC_INTEREST_KEYS[j]][0] for j in un]
                obs = np.concatenate([asked, gidx]).astype(int)
                yo = np.concatenate([y[asked], gv])
                out = {"G2": raw}
                for name, s in (("G2D", sig), ("G2D07", 0.7 * sig)):
                    out[name] = np.clip(cond_mean(mu, S, obs, yo, max(s * s, 1e-6))[:D], 0, 1)
                if q == QS[-1]:
                    h = y.copy()
                    h[un] = [area_prior(BASIC_INTEREST_KEYS[j], gs)[0] for j in un]
                    out["H12"] = h
                for m, v in out.items():
                    r = res[q][m]
                    r["rmse"].append(math.sqrt(float(np.mean((v - t) ** 2))))
                    r["rec"].append(top5_recall(v, t))
                    r["st"].append(s1[rid]["stratum"])
    res["n"] = len(ids)
    return res


# --- report ----------------------------------------------------------------------------


def layer_name(key: str) -> str:
    if key.startswith("group_"):
        return "群"
    if key.startswith("bi_"):
        return "領域"
    return "興趣" if key.startswith("riasec") else "強度／取向"


def f3(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def run(data: Path, out: Path) -> None:
    dims = dim_items()
    items = [it["item_id"] for it in item_bank()]
    s1 = read_answers(data / "s1.csv", items, need_stratum=True)
    s2 = (
        read_answers(data / "s2.csv", items, need_stratum=False)
        if (data / "s2.csv").exists()
        else {}
    )
    unknown = set(s2) - set(s1)
    if unknown:
        raise SystemExit(f"s2.csv: {len(unknown)} respondents not in s1.csv")
    pairs = read_pairs(data / "pairs.csv")
    synthetic = any(r["synthetic"] for r in [*s1.values(), *s2.values()])
    both = [r for r in s1 if r in s2]
    binding = len(s1) >= MIN_S1 and len(both) >= MIN_S2
    tag = "" if binding else "（前導，不具約束力）"
    rng = np.random.default_rng(SEED)
    L = [
        "# W-41 實測分析（預先登記：scripts/analyze_w41_study.py）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 資料夾：`{data}`",
        f"- S1 {len(s1)} 人、S2 {len(both)} 人；各層 S1："
        + "、".join(f"{s} {sum(r['stratum'] == s for r in s1.values())}" for s in STRATA),
        f"- 判定{'具約束力' if binding else f'**不具約束力**（需 S1 ≥ {MIN_S1}、S2 ≥ {MIN_S2}）'}",
    ]
    if synthetic:
        L.append("- ⚠️ **含合成資料（synthetic = true）：不是真實學生的結果**")

    def scores(resp, key):
        v = [resp["ans"][i] for i in dims[key] if i in resp["ans"]]
        return float(np.mean(v)) if len(v) == len(dims[key]) else None

    # 1 alpha
    L += ["", f"## 1. 內部一致性 α（V-C1-5，≥ 0.70）{tag}", "",
          "| 維度 | n | α | 95% CI | 判定 |", "|---|---|---|---|---|"]  # fmt: skip
    for key, its in dims.items():
        if len(its) < 2:
            continue
        X = np.array([[r["ans"][i] for i in its] for r in s1.values()
                      if all(i in r["ans"] for i in its)])  # fmt: skip
        a = cronbach_alpha(X) if len(X) > 2 else math.nan
        lo, hi = boot_ci(lambda ix, X=X: cronbach_alpha(X[ix]), len(X), rng) if len(X) > 2 \
            else (math.nan, math.nan)  # fmt: skip
        v = "—" if math.isnan(a) else ("✅" if a >= 0.7 else "❌")
        if v != "—" and not math.isnan(lo) and lo < 0.7 <= hi:
            v += " 不確定"
        L.append(f"| {key} | {len(X)} | {f3(a)} | [{f3(lo)}, {f3(hi)}] | {v} |")

    # 2 retest
    L += ["", f"## 2. 重測信度（V-C1-6，≥ 0.70）{tag}", ""]
    if not both:
        L.append("沒有 S2 資料。")
    else:
        L += [
            "| 層 | 維度數 | 通過 | 未通過 | 不確定 | 最低 r（維度） |",
            "|---|---|---|---|---|---|",
        ]
        layer_of = {k: layer_name(k) for k in dims}
        per_layer = defaultdict(list)
        for key in dims:
            pr = [(scores(s1[r], key), scores(s2[r], key)) for r in both]
            pr = np.array([p for p in pr if None not in p])
            if len(pr) < 3:
                continue
            rr = pearson(pr[:, 0], pr[:, 1])
            lo, hi = boot_ci(lambda ix, pr=pr: pearson(pr[ix, 0], pr[ix, 1]), len(pr), rng)
            per_layer[layer_of[key]].append((key, rr, lo, hi))
        for lay, v in per_layer.items():
            ok = sum(not math.isnan(r) and r >= 0.7 for _, r, _, _ in v)
            unc = sum(not math.isnan(lo) and lo < 0.7 <= hi for _, _, lo, hi in v)
            worst = min(v, key=lambda x: -2 if math.isnan(x[1]) else x[1])
            L.append(f"| {lay} | {len(v)} | {ok} | {len(v) - ok} | {unc} | "
                     f"{f3(worst[1])}（{worst[0]}） |")  # fmt: skip
        drift = []
        for it in items:
            d = np.array([s2[r]["ans"][it] - s1[r]["ans"][it] for r in both
                          if it in s1[r]["ans"] and it in s2[r]["ans"]])  # fmt: skip
            if len(d) > 2 and d.std(ddof=1) > 0:
                t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
                if abs(t) > 3:
                    drift.append(f"{it}（{d.mean():+.3f}，t = {t:.1f}）")
        L += ["", "系統性變化（|t| > 3，A-C1-29）：" + ("、".join(drift) if drift else "無")]

    # 3 sigma
    sigma_items = [dims[k][0] for k in (*BASIC_INTEREST_KEYS, *(f"group_{g}" for g in GROUPS))]
    if both:
        L += ["", "## 3. 每題作答誤差 σ（單位量尺）", "",
              "| 層 | 題數 | 合併 σ | σ > 0.30 的題數 | 最大 σ（題目） |",
              "|---|---|---|---|---|"]  # fmt: skip
        by_layer = defaultdict(list)
        for it in items:
            d = np.array([s1[r]["ans"][it] - s2[r]["ans"][it] for r in both
                          if it in s1[r]["ans"] and it in s2[r]["ans"]])  # fmt: skip
            if len(d) > 2:
                key = it.split("#")[0]
                by_layer[layer_name(key)].append((it, float(d.std(ddof=1) / math.sqrt(2))))
        for lay, v in by_layer.items():
            pooled = math.sqrt(float(np.mean([s * s for _, s in v])))
            mx = max(v, key=lambda x: x[1])
            L.append(f"| {lay} | {len(v)} | {f3(pooled)} | {sum(s > 0.3 for _, s in v)} | "
                     f"{f3(mx[1])}（{mx[0]}） |")  # fmt: skip

    # 4 overlap
    keys51 = [k for k in dims if not k.startswith("group_")]
    M = np.array([[scores(r, k) for k in keys51] for r in s1.values()
                  if all(scores(r, k) is not None for k in keys51)], dtype=float)  # fmt: skip
    L += ["", f"## 4. 維度重疊（V-C1-7，任兩維度 |r| < 0.80）{tag}", ""]
    if len(M) > 3:
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(M.T)
        hi_pairs = [f"{keys51[i]}–{keys51[j]}（{C[i, j]:.2f}）" for i in range(len(keys51))
                    for j in range(i + 1, len(keys51)) if abs(C[i, j]) >= 0.8]  # fmt: skip
        L.append(f"n = {len(M)}；{'✅ 沒有' if not hi_pairs else '❌ ' + '、'.join(hi_pairs)}")
    else:
        L.append("答完全部 51 維的人數不足。")

    # 5 choice consistency
    L += ["", "## 5. 選擇一致性（模擬假設 0.75）", ""]
    first = {(p["respondent_id"], p["pair_id"]): p["chosen"] for p in pairs if p["session"] == "s1"}
    rep = [(first[k], p["chosen"]) for p in pairs if p["session"] == "s2"
           for k in [(p["respondent_id"], p["pair_id"])] if k in first]  # fmt: skip
    if rep:
        k = sum(a == b for a, b in rep)
        lo, hi = wilson(k, len(rep))
        rerun = not lo <= 0.75 <= hi
        act = "**以觀察到的一致率重跑 W-76 參數還原**" if rerun else "與 0.75 相容"
        L.append(f"{len(rep)} 組重複：一致率 {k / len(rep):.3f}，"
                 f"95% CI [{lo:.3f}, {hi:.3f}] → {act}{tag}")  # fmt: skip
    else:
        L.append("沒有重複的成對比較。")

    def verdict_rows(res, base, cands, keys_q=QS):
        rows = []
        for q in keys_q:
            for c in cands:
                dr = paired(res[q][c]["rmse"], res[q][base]["rmse"])
                dc = paired(res[q][c]["rec"], res[q][base]["rec"])
                rows.append((q, c, dr, dc))
        return rows

    def strata_lines(res, base, cands, metric):
        out = ["", f"各層（D12，只報告；{metric} 差，候選 − {base}）：", "",
               "| q | 候選 | " + " | ".join(STRATA) + " |",
               "|---" * (2 + len(STRATA)) + "|"]  # fmt: skip
        for q in QS:
            for c in cands:
                cells = []
                for s in STRATA:
                    ix = [i for i, x in enumerate(res[q][c]["st"]) if x == s]
                    if len(ix) < 3:
                        cells.append("—")
                        continue
                    m, se = paired([res[q][c][metric][i] for i in ix],
                                   [res[q][base][metric][i] for i in ix])  # fmt: skip
                    cells.append(f"{m:+.4f}±{se:.4f}")
                out.append(f"| {q} | {c} | " + " | ".join(cells) + " |")
        return out

    # 6 completion
    L += ["", f"## 6. 領域補全（重驗 F105–F108）{tag}", ""]
    comp = completion(s1, dims)
    if comp["n"] < FOLDS * 2:
        L.append(f"答完全部領域與群的人數不足（{comp['n']}）。")
    else:
        rows = verdict_rows(comp, "H", ("G2", "G2s"))
        qual = {c: all(dr[0] < -2 * dr[1] and not dc[0] < -2 * dc[1]
                       for q, cc, dr, dc in rows if cc == c) for c in ("G2", "G2s")}  # fmt: skip
        dec = ("**建議相關補全（G2）成為預設**" if qual["G2"] else
               "**建議以學生資料重建領域模型（待下一批確認）**" if qual["G2s"]
               else "維持 H")  # fmt: skip
        mark = {c: "✅" if v else "❌" for c, v in qual.items()}
        L += [f"n = {comp['n']}；G2 {mark['G2']}、G2s {mark['G2s']} → {dec}", "",
              "| q | 候選 | RMSE（候選） | RMSE（H） | 差 | 前 5 名（候選） | （H） | 差 |",
              "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for q, c, dr, dc in rows:
            L.append(f"| {q} | {c} | {np.mean(comp[q][c]['rmse']):.4f} | "
                     f"{np.mean(comp[q]['H']['rmse']):.4f} | {dr[0]:+.4f}±{dr[1]:.4f} | "
                     f"{np.mean(comp[q][c]['rec']):.3f} | {np.mean(comp[q]['H']['rec']):.3f} | "
                     f"{dc[0]:+.4f}±{dc[1]:.4f} |")  # fmt: skip
        L += strata_lines(comp, "H", ("G2", "G2s"), "rmse")

    # 7 denoising
    L += ["", f"## 7. 已答領域去雜訊（重驗 F117–F119；主要指標前 5 名）{tag}", ""]
    dn = denoise(s1, s2, dims, sigma_items) if both else {"n": 0}
    if dn["n"] < FOLDS * 2:
        L.append(f"兩次都答完領域與群的人數不足（{dn['n']}）。")
    else:
        per = np.mean([np.array(dn[q]["G2D"]["rec"]) - np.array(dn[q]["G2"]["rec"]) for q in QS],
                      axis=0)  # fmt: skip
        m, se = float(per.mean()), float(per.std(ddof=1) / math.sqrt(len(per)))
        rows = verdict_rows(dn, "G2", ("G2D", "G2D07"))
        main_rows = [r for r in rows if r[1] == "G2D"]
        a = m > 2 * se
        b = all(not dc[0] < -2 * dc[1] for _, _, _, dc in main_rows)
        c = all(not dr[0] > 2 * dr[1] for _, _, dr, _ in main_rows)
        ok = a and b and c
        L += [f"n = {dn['n']}；(a) 前 5 名平均差 {m:+.4f}±{se:.4f} {'✅' if a else '❌'}；"
              f"(b) {'✅' if b else '❌'}；(c) RMSE 守門 {'✅' if c else '❌'} → "
              f"{'**去雜訊成為相關補全的選項**' if ok else '不新增'}", "",
              "| q | 方法 | 前 5 名（方法） | （TS-G2） | 差 | RMSE（方法） | （TS-G2） | 差 |",
              "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for q, cand, dr, dc in rows:
            name = "TS-G2D" if cand == "G2D" else "TS-G2D（σ × 0.7，只報告）"
            L.append(f"| {q} | {name} | {np.mean(dn[q][cand]['rec']):.3f} | "
                     f"{np.mean(dn[q]['G2']['rec']):.3f} | {dc[0]:+.4f}±{dc[1]:.4f} | "
                     f"{np.mean(dn[q][cand]['rmse']):.4f} | {np.mean(dn[q]['G2']['rmse']):.4f} | "
                     f"{dr[0]:+.4f}±{dr[1]:.4f} |")  # fmt: skip
        f115 = paired(dn[4]["G2"]["rmse"], dn[12]["H12"]["rmse"])
        L += ["", f"F115（只報告）：TS-G2 q = 4 的 RMSE − H q = 12 = {f115[0]:+.4f}±{f115[1]:.4f}"
              "（負值＝4 題相關補全較準）"]  # fmt: skip
        L += strata_lines(dn, "G2", ("G2D",), "rec")

    L += ["", "## 未包含", "", "測量恆等性與 DIF（V-C1-9，W-73 另行預先登記）、層份量（W-76，"
          "方法待定）、熟悉度回報（需要真值設計）。"]  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")


# --- fake data (dry runs and tests only) ---------------------------------------------


def make_fake(folder: Path, n: int, n_retest: int, seed: int, noise: float = 0.15) -> None:
    """Synthetic study files (every row synthetic = true): true areas from the shipped
    joint model, groups = max of their primary areas, other dimensions uniform; answers
    = true + N(0, noise), rounded to the 5-point scale; S2 repeats with fresh noise;
    20 pairs per respondent, 4 repeated in S2 with agreement probability 0.75."""
    rng = np.random.default_rng(seed)
    spec = json.loads(MODEL.read_text(encoding="utf-8"))
    mu, S = np.array(spec["mean"])[:D], np.array(spec["covariance"])[:D, :D]
    dims = dim_items()
    items = [it["item_id"] for it in item_bank()]
    prim = {g: [j for j, k in enumerate(BASIC_INTEREST_KEYS) if AREA_GROUPS[k][0] == g]
            for g in GROUPS}  # fmt: skip
    truth = {}
    for i in range(n):
        a = np.clip(rng.multivariate_normal(mu, S), 0, 1)
        t = {k: a[j] for j, k in enumerate(BASIC_INTEREST_KEYS)}
        t |= {f"group_{g}": a[prim[g]].max() for g in GROUPS}
        t |= {k: rng.random() for k in dims if k not in t}
        truth[f"F{i:04d}"] = t
    item_dim = {it: k for k, its in dims.items() for it in its}

    def answers(t):
        return {it: int(round(float(np.clip(t[item_dim[it]] + rng.normal(0, noise), 0, 1)) * 4)) + 1
                for it in items}  # fmt: skip

    folder.mkdir(parents=True, exist_ok=True)
    ids = list(truth)
    with open(folder / "s1.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["respondent_id", "stratum", "synthetic", *items])
        for i, rid in enumerate(ids):
            a = answers(truth[rid])
            w.writerow([rid, STRATA[i % 4], "true", *(a[it] for it in items)])
    retest = ids[:n_retest]
    with open(folder / "s2.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["respondent_id", "synthetic", *items])
        for rid in retest:
            a = answers(truth[rid])
            w.writerow([rid, "true", *(a[it] for it in items)])
    with open(folder / "pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["respondent_id", "session", "pair_id", "chosen", "confidence"])
        for rid in ids:
            for p in range(20):
                a, b = f"S{2 * p + 1:02d}", f"S{2 * p + 2:02d}"
                c = a if rng.random() < 0.5 else b
                w.writerow([rid, "s1", f"P{p:02d}", c, int(rng.integers(1, 5))])
                if rid in retest and p < 4:
                    c2 = c if rng.random() < 0.75 else (b if c == a else a)
                    w.writerow([rid, "s2", f"P{p:02d}", c2, int(rng.integers(1, 5))])
    print(f"wrote synthetic study files to {folder}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "private" / "study")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--make-fake", type=Path, default=None, help="write synthetic files")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-retest", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.make_fake:
        make_fake(args.make_fake, args.n, args.n_retest, args.seed)
        return
    run(args.data, args.out or args.data / "analysis.md")


if __name__ == "__main__":
    main()
