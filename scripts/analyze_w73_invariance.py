"""Pre-registered measurement-invariance and DIF analysis of the W-41 field study (W-73,
V-C1-9, D11; THEORY "測量恆等性與 DIF（W-73）"; A-C1-40..44, B-C1-40..45). Reads only the
de-identified S1 file of the W-41 study; no study data existed when v1 or v2 was committed.

FILES (default folder data/private/study/; header in data/templates/study_s1.csv)
  s1.csv  respondent_id, stratum, [synthetic], one column per item_id of
          data/items/item_bank_v2.json (73 items), answers 1-5, blank = not answered.
  stratum is one of same_field / adjacent / distant / multi_degree (protocol §2, D12).
  Reference group: same_field; focal groups: the other three.
  Optional, written by the generated R script (see CFA below):
  w73_cfa_fit.csv     model, step, status, index, cfi, rmsea, chisq, df
                      (status "ok", "not_converged" or "error: <message>"; index
                      "robust" or "scaled", whichever lavaan returned, robust first)
  w73_cfa_params.csv  lavaan parameterEstimates() of the configural models (+ model)

PRE-REGISTERED v2 (v1 committed 2026-10-29; v2 the same day after an independent review
of the v1 code and synthetic dry-run output, still before any study data existed;
changes listed at the end).
ITEM TIERS (fixed by the item bank, not by the data)
  Tier 1  the 10 two-item dimensions (6 RIASEC, 2 intensity, 2 orientation; 20 items).
          Matching criterion = raw sum of the dimension's two items (studied item
          included, Holland & Thayer 1988). With two items and this criterion the two
          items' Mantel statistics are identical and their SMDs are exact negatives
          (given the sum, the two conditional means add up to the sum; F160). |SMD|/SD is
          NOT mirrored (each item has its own SD), so the two items can land in different
          ETS classes; the larger one does not localise DIF. Tier 1 DIF is therefore a
          DIMENSION-level result; which item is biased is not identified (B-C1-40).
  Tier 2  the 53 single-item dimensions (12 area groups, 41 areas). Matching criterion
          = raw sum of the RIASEC items of the type(s) the area links to in O*NET
          (data/reference/basic_interest_map.csv); a group uses the union of the types of
          its primary areas. A Tier 2 flag cannot separate item bias from a genuine
          difference in that specific interest or from a short, unreliable anchor under
          group mean differences (A-C1-43, B-C1-41, B-C1-43): Tier 2 flags only send the
          item to review, they never change or remove it.
  Each item uses the respondents with the studied item and the full matching criterion.
DIF METHODS (S1 only; both run on every item)
  MH  polytomous Mantel test (Mantel 1963; Zwick, Donoghue & Grima 1993), each focal
      group vs same_field, stratified on every value of the matching criterion; effect
      size SMD (Dorans & Schmitt) divided by the SD of the item in reference + focal.
      Class after Zwick (2012): AA if p_BH >= .05 or |SMD|/SD <= .17; CC if p_BH < .05
      and |SMD|/SD > .25; BB otherwise. DEVIATION: ETS uses the unadjusted p < .05; this
      plan uses Benjamini-Hochberg adjusted p (60 Tier 1 tests), which makes BB/CC
      harder to reach. Item MH class = worst over the three focal groups.
  LR  proportional-odds logistic regression (Zumbo 1999; Swaminathan & Rogers 1990):
      M1 item ~ match; M2 + stratum dummies; M3 + stratum x match (match standardised).
      Test M3 vs M1 (likelihood ratio, df = 2 x number of focal groups present, 6).
      Effect: Nagelkerke Delta R2 (M3 - M1). Class with the Jodoin & Gierl (2001)
      cutoffs: C if p_BH < .05 and Delta R2 >= .070; B if p_BH < .05 and .035 <= Delta
      R2 < .070; A otherwise. EXTRAPOLATION: those cutoffs were derived for dichotomous
      items, two groups and a 2-df test; here they are applied to 5-point items, four
      strata and 6 df, where Delta R2 is further diluted (F162). Report only: M2 vs M1
      (uniform), M3 vs M2 (non-uniform), Zumbo-Thomas class (.13 / .26). A fit that
      does not converge (or |beta| > 15, separation) is "not computable" (class A, noted).
  MULTIPLE TESTING: Benjamini-Hochberg at q = .05 within each method x tier family
      (LR Tier 1: 20 tests; MH Tier 1: 60; LR Tier 2: 53; MH Tier 2: 159).
  ITEM CLASS: C if LR = C or MH = CC; else B if LR = B or MH = BB; else A.
  DIMENSION CLASS (Tier 1): the worse of its two items; flagged focal groups = those
      with MH CC/BB in either item.
  DETECTION: a dimension (or Tier 2 item) is "detected" when its class is B or C
      (C = large DIF, B = moderate DIF).
  LOCALISATION AID (Tier 1 RIASEC only, report only): for B or C dimensions, MH of each
      item matched on the sum of the area items linked to that type; the item with the
      larger |SMD|/SD for a flagged focal group is listed first as the candidate for
      content review. It is not a test (the anchor is a different construct, A-C1-43).
MULTI-GROUP ORDINAL CFA (Tier 1 only; external tool, pre-registered choice)
  Tool: R lavaan (Rosseel 2012) 0.6-19 + semTools 0.5-6 measEq.syntax; estimator WLSMV,
  parameterization theta, ID.fac std.lv, ID.cat Wu.Estabrook.2016. This script writes the
  data (w73_cfa_items.csv) and the R script (w73_cfa.R); it does not fit the model
  (no numpy implementation of ordinal WLSMV; no new Python dependency).
  Models: "riasec" 6 correlated factors x 2 items; "core" 4 correlated factors x 2 items
  (quant, programming, research, entrepreneurship); every item loads on its own factor.
  If any dimension of a model is DIF class C, a second model "<name>_reduced" without
  those dimensions is also fitted; the unflagged dimensions are judged on it.
  Respondents: complete on all 20 Tier 1 items. Categories: a category with zero
  responses in any stratum is merged with its neighbour toward the midpoint (3), the
  same way in every stratum, until all are non-empty (lavaan needs the same categories).
  Steps (group.equal), after Wu & Estabrook (2016) and Svetina, Rutkowski & Rutkowski
  (2020): configural (none); thresholds ("thresholds"; REPORT ONLY: with >= 3
  categories and this identification it is equivalent to the configural model and
  tests nothing); metric ("thresholds", "loadings"); scalar ("thresholds", "loadings",
  "intercepts").
  Fit indices: robust CFI / RMSEA (cfi.robust, rmsea.robust) when lavaan returns them,
  otherwise the scaled ones; the index used is recorded per row.
  Configural acceptable: status ok, CFI >= .95 and RMSEA <= .08.
  A step HOLDS (PRIMARY: Chen 2007 small-sample rule, N <= 300) if Delta CFI (previous -
  this) <= .005 AND Delta RMSEA (this - previous) <= .010; metric is compared with
  configural, scalar with metric. Report only: the Cheung & Rensvold (2002) / Chen
  large-sample rule (.010 / .015). CAVEAT: both rules were derived for continuous
  indicators and normal-theory ML, not ordinal WLSMV (Sass, Schmitt & Marsh 2014); they
  are used as conventions, not calibrated cutoffs (A-C1-44).
  PARTIAL INVARIANCE: only for factors with >= 3 items, freeing the C items' thresholds
  and loadings while >= 2 items stay constrained. With the current two-item factors it
  is never run (freeing a two-item factor's items leaves it unidentified); the reduced
  model above is used instead.
  GRM calibration (report only): from the configural theta-parameterised estimates,
  normal-ogive a = lambda, b_k = tau_k / lambda (Takane & de Leeuw 1987; Kamata & Bauer
  2008); logistic a = 1.702 lambda.
BINDING
  DIF verdicts bind only if the respondents used meet protocol §2 (>= 200 people and
  >= 40 per stratum). CFA verdicts bind only if every stratum has >= 100 complete
  respondents (project rule: below that a group has fewer people than free parameters,
  B-C1-42). Otherwise: "前導，不具約束力".
  POWER RULE: a dimension without DIF is read as "invariant" only where the --power
  reference run gives power >= .80 to DETECT (class B or C) a large uniform DIF
  (delta = 1.0) at the smallest stratum size used; N80 = the smallest tested n per
  stratum with that power (constant N80 below, from results/w73-dif-power.md, F163);
  below it the verdict is "not detected (low power)". v2 reference run (seed 2026, 200
  replications): detection power 25.5% / 54.5% / 89.0% / 99.0% at 40 / 100 / 200 / 400
  per stratum, so N80 = 200 (CC alone: 11.5% / 38.5% / 35.5% / 32.0%, because the
  effective |SMD|/SD under the simulated mean differences is 0.217, below .25).
  --impact-check (added after the v1 reference run; exploratory, report only): Tier 2
  false-flag rate for two sizes of stratum mean differences, 20 datasets per cell (F164).
  --asymmetry (added in v2 after review; report only): |SMD|/SD gap between the two
  items of a dimension (F160).
DECISIONS
  Tier 1 dimension C -> NOT COMPARABLE; both items REVIEW in cognitive interviews (the
      aid lists the candidate first). Dimension B -> REVIEW; verdict "comparable, moderate
      DIF flagged". An item is REVISED only when the interview finds a content reason
      (rewrite with other O*NET activities of the same element, or a new item); a
      revised item is re-tested in the next cohort and REMOVED if its dimension is C
      again. A -> KEEP. No item is revised or removed on these data alone: with two
      items per dimension the data cannot say which one (B-C1-40).
  Tier 2 item C or B -> REVIEW (single item); A -> KEEP (not tested for invariance).
  Dimension comparability across strata (Tier 1; general rule for k items):
      COMPARABLE        no C item and (CFA scalar holds for the model it is judged on)
      COMPARABLE (DIF only)  no C item, CFA not run or not binding
      PARTIAL           k >= 3, C items localised and >= 2 items not C, partial CFA holds
                        or not run: latent means only; observed-score means are NOT
                        comparable (Steinmetz 2013). Unreachable with two-item factors.
      NOT COMPARABLE    dimension C; configural fails; or metric / scalar fails for the
                        model the dimension is judged on
      B dimensions carry "moderate DIF flagged"; a COMPARABLE verdict below N80 per
      stratum is printed "not detected (low power)".
  Tier 2 dimensions: always "not tested (single item)"; flagged items listed.
NOT IN THIS SCRIPT: invariance by source field (s1.csv has no field column, B-C1-44);
  invariance over time (S1 vs S2).
v1 -> v2 (independent review of the v1 code and synthetic dry-run output; no study
  data): worst() consumed its generator so no dimension was ever B (B-C1-45, fixed);
  detection = class B or C and the power rule uses it (v1: C only); ETS-with-BH and the
  Jodoin-Gierl cutoffs labelled as deviation / extrapolation; CFA steps renamed and
  fixed (v1 called thresholds + loadings "scalar" and decided on the uninformative
  thresholds step; scalar now adds intercepts); primary fit rule is Chen's small-sample
  rule on both Delta CFI and Delta RMSEA (v1: .010 / .015, and its small-sample variant
  changed CFI only); robust fit indices preferred; v1's partial step freed whole
  two-item factors (unidentified) and is replaced by the reduced model; the R script
  records errors and non-convergence explicitly, omits group.partial when unused and
  writes only the thresholds an item has after collapsing; the power run adds n = 400
  for delta = 1.0 and population effect sizes under the simulated mean differences and
  for the 4-stratum LR fit.

    python scripts/analyze_w73_invariance.py --data data/private/study
    python scripts/analyze_w73_invariance.py --make-fake <folder>   # synthetic files
    python scripts/analyze_w73_invariance.py --power                # operating characteristics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2

from aisc.dimensions import BASIC_INTEREST_KEYS
from aisc.profiling.hierarchy import AREA_GROUPS, GROUP_LABELS

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "items" / "item_bank_v2.json"
BI_MAP = ROOT / "data" / "reference" / "basic_interest_map.csv"
POWER_OUT = ROOT / "results" / "w73-dif-power.md"
STRATA = ("same_field", "adjacent", "distant", "multi_degree")
REF, FOCALS = 0, (1, 2, 3)
SEED, Q = 2026, 0.05
MIN_DIF = (200, 40)  # protocol §2
MIN_CFA = 100  # per stratum, project rule (B-C1-42)
ETS = (0.17, 0.25)
JG = (0.035, 0.070)
ZT = (0.13, 0.26)
CFI_MIN, RMSEA_MAX = 0.95, 0.08
CHEN_SMALL = (0.005, 0.010)  # primary step rule (Delta CFI, Delta RMSEA), Chen 2007
LARGE = (0.010, 0.015)  # report only: Cheung & Rensvold 2002 / Chen large-sample
N80 = 200  # v2 reference --power run: smallest tested n with detection power >= .80 (F163)
RIASEC_NAME = {"Realistic": "r", "Investigative": "i", "Artistic": "a", "Social": "s",
               "Enterprising": "e", "Conventional": "c"}  # fmt: skip
CORE_MODELS = {
    "riasec": [f"riasec_{t}" for t in "riasec"],
    "core": ["quant_intensity", "programming_intensity", "research_orientation",
             "entrepreneurship"],
}  # fmt: skip
PILOT = "（前導，不具約束力）"
NAN_MH = {"chi2": math.nan, "p": math.nan, "smd": math.nan, "es": math.nan}


# --- item bank and tiers ---------------------------------------------------------------


def item_bank() -> list[dict]:
    return json.loads(BANK.read_text(encoding="utf-8"))["items"]


def dim_items() -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for it in item_bank():
        out[it["dimension"] or f"group_{it['group']}"].append(it["item_id"])
    return dict(out)


def area_types() -> dict[str, list[str]]:
    with open(BI_MAP, encoding="utf-8") as fh:
        return {r["key"]: [RIASEC_NAME[t] for t in r["riasec"].split()] for r in csv.DictReader(fh)}


def matching_items() -> dict[str, list[str]]:
    """item -> items whose raw sum is its matching criterion (pre-registered tiers)."""
    dims, types = dim_items(), area_types()
    out = {}
    for d, its in dims.items():
        if len(its) > 1:  # Tier 1: own dimension, studied item included
            for it in its:
                out[it] = list(its)
            continue
        if d.startswith("group_"):
            g = int(d.split("_")[1])
            ts = sorted({t for k in BASIC_INTEREST_KEYS if AREA_GROUPS[k][0] == g
                         for t in types[k]})  # fmt: skip
        else:
            ts = types[d]
        out[its[0]] = [i for t in ts for i in dims[f"riasec_{t}"]]
    return out


def aid_anchor() -> dict[str, list[str]]:
    """RIASEC dimension -> area items linked to that type (localisation aid)."""
    dims, types = dim_items(), area_types()
    return {f"riasec_{t}": [dims[k][0] for k in BASIC_INTEREST_KEYS if t in types[k]]
            for t in "riasec"}  # fmt: skip


def tiers() -> tuple[list[str], list[str]]:
    dims = dim_items()
    t1 = [i for its in dims.values() if len(its) > 1 for i in its]
    t2 = [its[0] for its in dims.values() if len(its) == 1]
    return t1, t2


# --- files -----------------------------------------------------------------------------


def read_s1(path: Path, items: list[str]) -> tuple[np.ndarray, np.ndarray, bool]:
    """Raw answers (n x items; 0 = not answered), stratum codes, synthetic flag."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    miss = {"respondent_id", "stratum", *items} - set(rows[0])
    if miss:
        raise SystemExit(f"{path}: missing columns {sorted(miss)[:5]}...")
    Y = np.zeros((len(rows), len(items)), dtype=int)
    st = np.zeros(len(rows), dtype=int)
    ids, syn = set(), False
    for n, r in enumerate(rows):
        rid = r["respondent_id"].strip()
        if not rid or rid in ids:
            raise SystemExit(f"{path}:{n + 2}: respondent_id must be present and unique")
        ids.add(rid)
        s = (r["stratum"] or "").strip()
        if s not in STRATA:
            raise SystemExit(f"{path}:{n + 2}: stratum {s!r} not in {STRATA}")
        st[n] = STRATA.index(s)
        syn |= (r.get("synthetic") or "").strip().lower() == "true"
        for j, it in enumerate(items):
            v = (r[it] or "").strip()
            if v == "":
                continue
            if v not in {"1", "2", "3", "4", "5"}:
                raise SystemExit(f"{path}:{n + 2}: {it} = {v!r} is not 1-5")
            Y[n, j] = int(v)
    return Y, st, syn


# --- statistics ------------------------------------------------------------------------


def bh(p: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (NaN stays NaN)."""
    p = np.asarray(p, dtype=float)
    ok = np.flatnonzero(~np.isnan(p))
    out = np.full(len(p), np.nan)
    if len(ok):
        order = ok[np.argsort(p[ok])]
        m = len(ok)
        adj = p[order] * m / np.arange(1, m + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out[order] = np.minimum(adj, 1.0)
    return out.tolist()


def mantel(y: np.ndarray, focal: np.ndarray, match: np.ndarray) -> dict:
    """Polytomous Mantel test (1 df) and standardised mean difference (focal minus
    reference) stratified on each value of the matching score. y are raw scores."""
    F = E = V = 0.0
    smd_num, n_f = 0.0, 0
    for k in np.unique(match):
        s = match == k
        yk, fk = y[s], focal[s]
        nk, nf = len(yk), int(fk.sum())
        nr = nk - nf
        if nk < 2 or nf == 0 or nr == 0:
            continue
        sy, syy = float(yk.sum()), float((yk.astype(float) ** 2).sum())
        F += float(yk[fk].sum())
        E += nf * sy / nk
        V += nr * nf * (nk * syy - sy * sy) / (nk * nk * (nk - 1))
        smd_num += nf * (yk[fk].mean() - yk[~fk].mean())
        n_f += nf
    sd = float(y.std(ddof=1)) if len(y) > 1 else 0.0
    if V <= 0 or n_f == 0 or sd == 0:
        return dict(NAN_MH)
    stat = (F - E) ** 2 / V
    smd = float(smd_num / n_f)
    return {"chi2": stat, "p": float(chi2.sf(stat, 1)), "smd": smd, "es": smd / sd}


def po_loglik(y: np.ndarray, X: np.ndarray, K: int):
    """Negative log-likelihood and gradient of the proportional-odds model
    P(Y <= k) = expit(theta_k - X beta), thresholds theta_0 < ... < theta_{K-2}
    parameterised as theta_0 = a, increments exp(d)."""
    p_ = X.shape[1]

    def f(par):
        a, d, beta = par[0], par[1 : K - 1], par[K - 1 :]
        theta = a + np.concatenate([[0.0], np.cumsum(np.exp(d))])
        eta = X @ beta if p_ else np.zeros(len(y))
        ext = np.concatenate([[-np.inf], theta, [np.inf]])
        up, lo = ext[y + 1] - eta, ext[y] - eta
        Fu, Fl = expit(up), expit(lo)
        fu = np.where(np.isfinite(up), Fu * (1 - Fu), 0.0)
        fl = np.where(np.isfinite(lo), Fl * (1 - Fl), 0.0)
        pr = np.maximum(Fu - Fl, 1e-300)
        gth = np.zeros(K - 1)
        top, bot = y < K - 1, y > 0
        np.add.at(gth, y[top], fu[top] / pr[top])
        np.add.at(gth, y[bot] - 1, -fl[bot] / pr[bot])
        gb = -(X * ((fu - fl) / pr)[:, None]).sum(axis=0) if p_ else np.zeros(0)
        gd = np.exp(d) * np.cumsum(gth[::-1])[::-1][1:]
        return -float(np.log(pr).sum()), -np.concatenate([[gth.sum()], gd, gb])

    return f


def po_fit(y: np.ndarray, X: np.ndarray) -> tuple[float, bool, np.ndarray]:
    """Maximum-likelihood proportional-odds fit. y = category codes 0..K-1, every code
    present. Returns (log-likelihood, converged, beta)."""
    K = int(y.max()) + 1
    cum = np.cumsum(np.bincount(y, minlength=K))[:-1] / len(y)
    th = np.log(cum / (1 - cum))
    par0 = np.concatenate([[th[0]], np.log(np.maximum(np.diff(th), 1e-3)), np.zeros(X.shape[1])])
    res = minimize(po_loglik(y, X, K), par0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 1000, "gtol": 1e-7})  # fmt: skip
    beta = res.x[K - 1 :]
    ok = bool(res.success or np.abs(res.jac).max() < 1e-3) and bool(np.all(np.abs(beta) < 15))
    return -float(res.fun), ok, beta


def lr_dif(y: np.ndarray, match: np.ndarray, st: np.ndarray) -> dict:
    """Ordinal logistic-regression DIF with stratum dummies (reference = REF)."""
    levels = np.unique(y)
    groups = [g for g in FOCALS if np.any(st == g)]
    if len(levels) < 2 or not groups or not np.any(st == REF):
        return {"ok": False}
    yc = np.searchsorted(levels, y)
    n = len(yc)
    cnt = np.bincount(yc)
    ll0 = float((cnt * np.log(cnt / n)).sum())
    sd = match.std()
    t = (match - match.mean()) / (sd if sd > 0 else 1.0)
    D = np.column_stack([(st == g).astype(float) for g in groups])
    fits = [po_fit(yc, X) for X in (t[:, None], np.column_stack([t, D]),
                                    np.column_stack([t, D, D * t[:, None]]))]  # fmt: skip
    (l1, o1, _), (l2, o2, _), (l3, o3, _) = fits
    if not (o1 and o2 and o3):
        return {"ok": False}

    def r2n(ll):
        mx = 1 - math.exp(2 * ll0 / n)
        return (1 - math.exp(2 * (ll0 - ll) / n)) / mx if mx > 0 else math.nan

    df, s31 = 2 * len(groups), max(2 * (l3 - l1), 0.0)
    return {"ok": True, "chi2": s31, "df": df, "p": float(chi2.sf(s31, df)),
            "dr2": r2n(l3) - r2n(l1),
            "p_uni": float(chi2.sf(max(2 * (l2 - l1), 0.0), len(groups))),
            "p_non": float(chi2.sf(max(2 * (l3 - l2), 0.0), len(groups)))}  # fmt: skip


def ets_class(p_adj: float, es: float) -> str:
    if math.isnan(p_adj) or math.isnan(es) or p_adj >= Q or abs(es) <= ETS[0]:
        return "AA"
    return "CC" if abs(es) > ETS[1] else "BB"


def lr_class(p_adj: float, dr2: float, cut=JG) -> str:
    if math.isnan(p_adj) or math.isnan(dr2) or p_adj >= Q or dr2 < cut[0]:
        return "A"
    return "C" if dr2 >= cut[1] else "B"


def _mh_all(y, s, m) -> dict:
    out = {}
    for g in FOCALS:
        sel = (s == REF) | (s == g)
        ok = np.any(s == g) and np.any(s == REF)
        out[g] = mantel(y[sel], s[sel] == g, m[sel]) if ok else dict(NAN_MH)
    return out


def dif_family(Y: np.ndarray, st: np.ndarray, cols: dict[str, int], items: list[str],
               match: dict[str, list[str]], lr: bool = True) -> dict[str, dict]:  # fmt: skip
    """Run MH and LR on each item of one family, BH within method, classify."""
    res = {}
    for it in items:
        j, mj = cols[it], [cols[m] for m in match[it]]
        use = (Y[:, j] > 0) & np.all(Y[:, mj] > 0, axis=1)
        y, m, s = Y[use, j], Y[use][:, mj].sum(axis=1), st[use]
        res[it] = {"n": int(use.sum()), "n_st": [int((s == g).sum()) for g in range(4)],
                   "mh": _mh_all(y, s, m),
                   "lr": lr_dif(y, m, s) if lr else {"ok": False}}  # fmt: skip
    p_lr = bh([res[it]["lr"].get("p", math.nan) for it in items])
    p_mh = bh([res[it]["mh"][g]["p"] for it in items for g in FOCALS])
    for i, it in enumerate(items):
        r = res[it]
        r["lr"]["p_adj"] = p_lr[i]
        r["lr_class"] = lr_class(p_lr[i], r["lr"].get("dr2", math.nan))
        r["zt_class"] = lr_class(p_lr[i], r["lr"].get("dr2", math.nan), ZT)
        for k, g in enumerate(FOCALS):
            r["mh"][g]["p_adj"] = p_mh[3 * i + k]
            r["mh"][g]["cls"] = ets_class(p_mh[3 * i + k], r["mh"][g]["es"])
        mh_cls = [r["mh"][g]["cls"] for g in FOCALS]
        r["mh_class"] = "CC" if "CC" in mh_cls else ("BB" if "BB" in mh_cls else "AA")
        r["cls"] = ("C" if r["lr_class"] == "C" or r["mh_class"] == "CC"
                    else "B" if r["lr_class"] == "B" or r["mh_class"] == "BB" else "A")  # fmt: skip
    return res


def worst(classes) -> str:
    classes = list(classes)  # a generator would be exhausted by the first test (B-C1-45)
    return "C" if "C" in classes else ("B" if "B" in classes else "A")


def dimension_dif(dif: dict, dims: dict[str, list[str]]) -> dict[str, dict]:
    """Tier 1 dimension class (worse item) and the focal groups flagged in either item."""
    out = {}
    for k, its in dims.items():
        if len(its) < 2:
            continue
        focal = sorted({g for i in its for g in FOCALS if dif[i]["mh"][g]["cls"] != "AA"})
        out[k] = {"cls": worst(dif[i]["cls"] for i in its), "focal": focal}
    return out


def localisation_aid(Y, st, cols, dims, ddif) -> dict[str, list[tuple[str, int, float]]]:
    """Report-only: for flagged RIASEC dimensions, MH of each item on the linked-area
    anchor; per flagged focal group, items ordered by |SMD|/SD (largest first)."""
    anchor, out = aid_anchor(), {}
    for k, v in ddif.items():
        if k not in anchor or v["cls"] == "A":
            continue
        rows = []
        for g in v["focal"]:
            vals = []
            for it in dims[k]:
                mj = [cols[a] for a in anchor[k]]
                use = (
                    (Y[:, cols[it]] > 0) & np.all(Y[:, mj] > 0, axis=1) & ((st == REF) | (st == g))
                )
                r = mantel(Y[use, cols[it]], st[use] == g, Y[use][:, mj].sum(axis=1))
                vals.append((it, g, r["es"]))
            rows += sorted(vals, key=lambda x: -abs(x[2]) if not math.isnan(x[2]) else 0)
        out[k] = rows
    return out


# --- CFA export, decisions and GRM conversion ----------------------------------------


def collapse_categories(x: np.ndarray, st: np.ndarray) -> np.ndarray:
    """Merge categories empty in any stratum toward the midpoint (3), identically in
    all strata; recode to 1..K. x are raw 1-5 answers."""
    m = {c: c for c in range(1, 6)}
    groups = np.unique(st)
    while True:
        codes = np.array([m[v] for v in x])
        present = sorted(set(codes.tolist()))
        empty = [c for c in present if any(not np.any((codes == c) & (st == g)) for g in groups)]
        if not empty or len(present) == 1:
            break
        c = max(empty, key=lambda c: (abs(c - 3), c))
        if c == 3:  # midpoint itself empty somewhere: merge into the nearer, larger neighbour
            to = min(
                (d for d in present if d != 3), key=lambda d: (abs(d - 3), -np.sum(codes == d))
            )
        else:
            to = next(
                d for d in (present if c < 3 else present[::-1]) if (d > c if c < 3 else d < c)
            )
        for k, v in m.items():
            if v == c:
                m[k] = to
    codes = np.array([m[v] for v in x])
    return np.searchsorted(sorted(set(codes.tolist())), codes) + 1


def rname(item: str) -> str:
    return item.replace("#", "_")


def cfa_models(ddif: dict) -> dict[str, list[str]]:
    """Models to fit: the two pre-registered ones, plus '<name>_reduced' without the
    DIF-C dimensions when at least two factors remain (one two-item factor alone is not
    identified)."""
    out = {}
    for name, keys in CORE_MODELS.items():
        out[name] = list(keys)
        keep = [k for k in keys if ddif[k]["cls"] != "C"]
        if len(keep) < len(keys) and len(keep) >= 2:
            out[f"{name}_reduced"] = keep
    return out


def cfa_partial(dif: dict, ddif: dict, dims: dict, models: dict) -> dict[str, list[str]]:
    """Items freed in the partial step: C items of factors with >= 3 items, only while
    >= 2 items of the factor stay constrained. Empty for two-item factors."""
    out = {}
    for name, keys in models.items():
        part = []
        for k in keys:
            its = dims[k]
            cs = [i for i in its if dif[i]["cls"] == "C"]
            if len(its) >= 3 and cs and len(its) - len(cs) >= 2:
                part += cs
        out[name] = [rname(i) for i in part]
    return out


def r_script(dims: dict, models: dict[str, list[str]], partial: dict[str, list[str]],
             ncat: dict[str, int]) -> str:  # fmt: skip
    """R script for the ordinal multi-group CFA. ncat = categories per item after
    collapsing (an item has ncat - 1 thresholds)."""

    def model(keys):
        return "\\n".join(f"F_{k} =~ " + " + ".join(rname(i) for i in dims[k]) for k in keys)

    lines = [
        "# Generated by scripts/analyze_w73_invariance.py (W-73 pre-registration v2).",
        "# Do not edit. Pre-registered tool versions: lavaan 0.6-19, semTools 0.5-6.",
        "library(lavaan); library(semTools)",
        'd <- read.csv("w73_cfa_items.csv")',
        "fit_rows <- list(); par_rows <- list()",
        "measures <- function(fit) {",
        '  fm <- tryCatch(fitMeasures(fit, c("cfi.robust", "rmsea.robust", "chisq.scaled",',
        '    "df.scaled")), error = function(e) rep(NA, 4))',
        '  if (!any(is.na(fm[1:2]))) return(list(index = "robust", fm = fm))',
        '  fm <- fitMeasures(fit, c("cfi.scaled", "rmsea.scaled", "chisq.scaled", "df.scaled"))',
        '  list(index = "scaled", fm = fm)',
        "}",
        "run_step <- function(name, step, m, its, eq, part) {",
        "  args <- list(configural.model = m, data = d, ordered = its,",
        '    parameterization = "theta", ID.fac = "std.lv", ID.cat = "Wu.Estabrook.2016",',
        '    group = "stratum")',
        "  if (length(eq)) args$group.equal <- eq",
        "  if (length(part)) args$group.partial <- part",
        '  status <- "ok"',
        "  fit <- tryCatch({",
        "    syn <- do.call(measEq.syntax, args)",
        '    cfa(as.character(syn), data = d, ordered = its, group = "stratum",',
        '      estimator = "WLSMV", parameterization = "theta")',
        '  }, error = function(e) { status <<- paste("error:", conditionMessage(e)); NULL })',
        '  if (!is.null(fit) && !lavInspect(fit, "converged")) status <- "not_converged"',
        "  index <- NA; fm <- rep(NA, 4)",
        '  if (status == "ok") { r <- measures(fit); index <- r$index; fm <- r$fm }',
        "  fit_rows[[length(fit_rows) + 1]] <<- data.frame(model = name, step = step,",
        "    status = status, index = index, cfi = fm[1], rmsea = fm[2], chisq = fm[3],",
        "    df = fm[4])",
        '  if (status == "ok" && step == "configural") {',
        "    pe <- parameterEstimates(fit); pe$model <- name",
        "    par_rows[[length(par_rows) + 1]] <<- pe",
        "  }",
        "}",
        'steps <- list(configural = character(0), thresholds = "thresholds",',
        '  metric = c("thresholds", "loadings"),',
        '  scalar = c("thresholds", "loadings", "intercepts"))',
    ]
    for name, keys in models.items():
        its = [i for k in keys for i in dims[k]]
        part = []
        for i in partial.get(name, []):
            fac = next(k for k in keys if any(rname(x) == i for x in dims[k]))
            orig = next(x for x in dims[fac] if rname(x) == i)
            part += [f"F_{fac} =~ {i}"] + [f"{i} | t{t}" for t in range(1, ncat[orig])]
        lines += [
            f'm <- "{model(keys)}"',
            "its <- c(" + ", ".join(f'"{rname(i)}"' for i in its) + ")",
            "for (s in names(steps)) run_step(" + f'"{name}"' + ", s, m, its, steps[[s]], NULL)",
        ]
        if part:
            lines += [
                "part <- c(" + ", ".join(f'"{p}"' for p in part) + ")",
                f'run_step("{name}", "partial", m, its, steps$scalar, part)',
            ]
    lines += [
        'write.csv(do.call(rbind, fit_rows), "w73_cfa_fit.csv", row.names = FALSE)',
        "if (length(par_rows)) write.csv(do.call(rbind, par_rows), "
        '"w73_cfa_params.csv", row.names = FALSE)',
    ]
    return "\n".join(lines) + "\n"


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def read_fit(path: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            status = (r.get("status") or "").strip()
            out[r["model"]][r["step"]] = {
                "ok": status == "ok",
                "status": status,
                "index": (r.get("index") or "").strip(),
                "cfi": _num(r["cfi"]),
                "rmsea": _num(r["rmsea"]),
            }
    return dict(out)


def step_holds(prev: dict, cur: dict, rule: tuple[float, float] = CHEN_SMALL) -> bool:
    if not (prev["ok"] and cur["ok"]):
        return False
    return prev["cfi"] - cur["cfi"] <= rule[0] and cur["rmsea"] - prev["rmsea"] <= rule[1]


def cfa_verdict(steps: dict[str, dict]) -> dict:
    """Configural acceptable? metric / scalar / partial hold (primary small-sample rule;
    *_large = report-only large-sample rule)? None = not run."""
    c = steps.get("configural")
    if c is None:
        return {"configural": None}
    conf = c["ok"] and c["cfi"] >= CFI_MIN and c["rmsea"] <= RMSEA_MAX
    out = {"configural": conf, "metric": None, "scalar": None, "partial": None,
           "status": c.get("status", ""), "index": c.get("index", "")}  # fmt: skip
    t = steps.get("thresholds")
    if t and t["ok"] and c["ok"]:
        out["thresholds_dcfi"] = c["cfi"] - t["cfi"]
    if not conf:
        return out
    mt, sc, pa = steps.get("metric"), steps.get("scalar"), steps.get("partial")
    if mt:
        out["metric"] = step_holds(c, mt)
        out["metric_large"] = step_holds(c, mt, LARGE)
    if mt and sc:
        out["scalar"] = bool(out["metric"]) and step_holds(mt, sc)
        out["scalar_large"] = bool(out.get("metric_large")) and step_holds(mt, sc, LARGE)
    if mt and pa:
        out["partial"] = bool(out["metric"]) and step_holds(mt, pa)
    return out


def grm_from_lavaan(rows: list[dict]) -> dict[tuple[str, str], tuple[float, list[float]]]:
    """(item, group) -> (normal-ogive a, [b_k]) from theta-parameterised estimates."""
    lam, tau = {}, defaultdict(list)
    for r in rows:
        if r["op"] == "=~":
            lam[(r["rhs"], r.get("group", "1"))] = float(r["est"])
        elif r["op"] == "|":
            tau[(r["lhs"], r.get("group", "1"))].append(
                (int(r["rhs"].lstrip("t")), float(r["est"]))
            )
    return {k: (a, [t / a for _, t in sorted(tau[k])]) for k, a in lam.items() if a != 0}


def dimension_verdict(k_items: int, n_c: int, localised: bool, cfa: dict | None,
                      underpowered: bool, moderate: bool = False) -> str:  # fmt: skip
    """Comparability of one dimension across strata (header DECISIONS)."""
    if cfa is not None and cfa.get("configural") is False:
        return "不可比較（形態恆等性未通過）"
    if n_c:
        if k_items >= 3 and localised and k_items - n_c >= 2:
            if cfa is not None and cfa.get("partial") is False:
                return "不可比較（部分純量未通過）"
            return "部分可比較（只比潛在平均；觀察分數平均不可比）"
        return "不可比較（維度層 DIF）" if k_items == 2 else "不可比較（DIF 無法定位）"
    if cfa is not None and cfa.get("metric") is False:
        return "不可比較（CFA 量尺未通過）"
    if cfa is not None and cfa.get("scalar") is False:
        return "不可比較（CFA 純量未通過）"
    base = "可比較" if cfa is not None and cfa.get("scalar") else \
        "可比較（只依 DIF；CFA 未執行或不具約束力）"  # fmt: skip
    verdict = "未偵測到差異（檢定力不足）" if underpowered else base
    return verdict + ("；標記中度 DIF（B），送審查" if moderate else "")


def item_decision(r: dict, tier: int, dim_cls: str | None = None) -> str:
    if tier == 2:
        return "審查（單題）" if r["cls"] in "BC" else "保留（未經恆等性檢驗）"
    if dim_cls == "C":
        return "審查（維度層 DIF，B-C1-40）"
    if dim_cls == "B":
        return "審查（維度層中度 DIF）"
    return "保留"


# --- report ----------------------------------------------------------------------------


def f3(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def dif_rows(dif: dict, items: list[str], decisions: dict) -> list[str]:
    L = ["| 題目 | n（各層） | LR χ²(df) | p_BH | ΔR²_N | LR | ZT | MH SMD/SD（adj／dist／multi）"
         " | MH | 類別 | 決定 |", "|---" * 11 + "|"]  # fmt: skip
    for it in items:
        r = dif[it]
        lr = r["lr"]
        es = "／".join(f"{r['mh'][g]['es']:+.2f}" if not math.isnan(r["mh"][g]["es"]) else "—"
                      for g in FOCALS)  # fmt: skip
        chi = f"{lr['chi2']:.1f}({lr['df']})" if lr.get("ok") else "無法計算"
        L.append(f"| {it} | {r['n']}（{'/'.join(map(str, r['n_st']))}） | {chi} | "
                 f"{f3(lr.get('p_adj', math.nan))} | {f3(lr.get('dr2', math.nan))} | "
                 f"{r['lr_class']} | {r['zt_class']} | {es} | {r['mh_class']} | "
                 f"**{r['cls']}** | {decisions[it]} |")  # fmt: skip
    return L


def binding_dif(n_st: list[int]) -> bool:
    return sum(n_st) >= MIN_DIF[0] and min(n_st) >= MIN_DIF[1]


def run(data: Path, out: Path) -> None:
    bank = [it["item_id"] for it in item_bank()]
    dims, match = dim_items(), matching_items()
    t1, t2 = tiers()
    Y, st, syn = read_s1(data / "s1.csv", bank)
    cols = {it: j for j, it in enumerate(bank)}
    dif1, dif2 = dif_family(Y, st, cols, t1, match), dif_family(Y, st, cols, t2, match)
    ddif = dimension_dif(dif1, dims)
    aid = localisation_aid(Y, st, cols, dims, ddif)
    n_used = [min(dif1[i]["n_st"][g] for i in t1) for g in range(4)]
    n_used2 = [min(dif2[i]["n_st"][g] for i in t2) for g in range(4)]
    bind1, bind2 = binding_dif(n_used), binding_dif(n_used2)
    underpowered = min(n_used) < N80
    n80_txt = "模擬的人數都未達檢定力 .80" if math.isinf(N80) else f"{N80} 人"

    comp = np.all(Y[:, [cols[i] for i in t1]] > 0, axis=1)
    n_cfa = [int((comp & (st == g)).sum()) for g in range(4)]
    bind_cfa = min(n_cfa) >= MIN_CFA
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.parent / "w73_cfa_items.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stratum", *(rname(i) for i in t1)])
        cc = {i: collapse_categories(Y[comp, cols[i]], st[comp]) for i in t1} if comp.any() else {}
        for r, s in enumerate(st[comp]):
            w.writerow([STRATA[s], *(int(cc[i][r]) for i in t1)])
    ncat = {i: int(cc[i].max()) if i in cc else 5 for i in t1}
    models = cfa_models(ddif)
    rtext = r_script(dims, models, cfa_partial(dif1, ddif, dims, models), ncat)
    (out.parent / "w73_cfa.R").write_text(rtext, "utf-8")
    fitp = data / "w73_cfa_fit.csv"
    cfa = {m: cfa_verdict(s) for m, s in read_fit(fitp).items()} if fitp.exists() else {}

    item_dim = {i: k for k, its in dims.items() for i in its}
    dec = {it: item_decision(dif1[it], 1, ddif[item_dim[it]]["cls"]) for it in t1}
    dec |= {it: item_decision(dif2[it], 2) for it in t2}

    L = ["# W-73 測量恆等性與 DIF（預先登記：scripts/analyze_w73_invariance.py）", "",
         f"- 產生日期：{date.today().isoformat()} · 資料夾：`{data}`",
         f"- S1 {len(st)} 人；各層：" + "、".join(f"{s} {int((st == g).sum())}"
                                             for g, s in enumerate(STRATA)),
         f"- DIF 約束力：使用人數 ≥ {MIN_DIF[0]} 且每層 ≥ {MIN_DIF[1]}（protocol §2）；"
         f"CFA 約束力：每層完整作答 ≥ {MIN_CFA}；「沒有 DIF」解讀為等值需每層 ≥ N80（{n80_txt}）"
         "（檢定力規則）"]  # fmt: skip
    if syn:
        L.append("- ⚠️ **含合成資料（synthetic = true）：不是真實學生的結果**")
    tag1 = "" if bind1 else PILOT
    L += ["", f"## 1. 第一層 DIF（10 個兩題維度；維度層判定）{tag1}", "",
          f"每層最少使用人數 {min(n_used)}" + (f"（< N80，{n80_txt}：沒有 DIF 只代表「未偵測到」）"
                                         if underpowered else "")
          + "。兩題維度以兩題總分配對時，兩題的 MH 統計量相同、SMD 正負相反（F160）："
          "判定是維度層級，資料無法指出是哪一題。", ""]  # fmt: skip
    L += dif_rows(dif1, t1, dec)
    L += ["", "| 維度 | 類別 | 標記的焦點層 | 定位輔助（只報告；連結領域為錨，"
          "|SMD|/SD 大者在前） |", "|---|---|---|---|"]  # fmt: skip
    for k, v in ddif.items():
        a = "；".join(f"{STRATA[g]}: {i} {e:+.2f}" for i, g, e in aid.get(k, [])) or "—"
        L.append(f"| {k} | **{v['cls']}** | {'、'.join(STRATA[g] for g in v['focal']) or '—'} "
                 f"| {a} |")  # fmt: skip
    L += ["", f"## 2. 第二層 DIF（53 個單題維度；只送審查）{'' if bind2 else PILOT}", ""]
    L += dif_rows(dif2, t2, dec)
    L += ["", f"## 3. 多群組次序 CFA（第一層；R lavaan）{'' if bind_cfa else PILOT}", "",
          f"完整作答人數（各層）：{'/'.join(map(str, n_cfa))}；已輸出 `w73_cfa_items.csv` 與 "
          "`w73_cfa.R`（在輸出資料夾執行 `Rscript w73_cfa.R`，再重跑本腳本）。", ""]  # fmt: skip
    if cfa:
        L += ["主要判準：Chen (2007) 小樣本規則 ΔCFI ≤ .005 且 ΔRMSEA ≤ .010（原為連續 ML 所訂，"
              "A-C1-44）；大樣本規則 .010／.015 只報告；閾值步驟與形態模型等價，只報告 ΔCFI。", "",
              "| 模型 | 狀態 | 指標 | 形態 | 量尺 | 純量 | 部分純量 | 大樣本規則（量尺／純量） "
              "| 閾值步驟 ΔCFI |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip

        def mk(v):
            return "—" if v is None else ("✅" if v else "❌")

        for m, v in cfa.items():
            L.append(f"| {m} | {v.get('status', '')} | {v.get('index', '')} | "
                     f"{mk(v['configural'])} | {mk(v.get('metric'))} | {mk(v.get('scalar'))} | "
                     f"{mk(v.get('partial'))} | {mk(v.get('metric_large'))}／"
                     f"{mk(v.get('scalar_large'))} | {f3(v.get('thresholds_dcfi'))} |")  # fmt: skip
        parp = data / "w73_cfa_params.csv"
        if parp.exists():
            with open(parp, encoding="utf-8-sig", newline="") as fh:
                grm = grm_from_lavaan(list(csv.DictReader(fh)))
            L += ["", "GRM 參數（只報告；形態模型，常態肩形 a = λ、b = τ／λ）：", "",
                  "| 題目 | 組 | a | b |", "|---|---|---|---|"]  # fmt: skip
            for (i, g), (a, b) in sorted(grm.items()):
                L.append(f"| {i} | {g} | {a:.2f} | {', '.join(f'{x:.2f}' for x in b)} |")
    else:
        L.append("尚未提供 `w73_cfa_fit.csv`：可比性只依 DIF。")
    L += ["", f"## 4. 各維度跨背景層的可比性{tag1}", "", "| 維度 | 判定 |", "|---|---|"]
    for k, v in ddif.items():
        model = next(m for m, ks in CORE_MODELS.items() if k in ks)
        judged = f"{model}_reduced" if f"{model}_reduced" in models else model
        c = cfa.get(judged) if cfa and bind_cfa else None
        n_c = sum(dif1[i]["cls"] == "C" for i in dims[k])
        verdict = dimension_verdict(len(dims[k]), n_c, False, c, underpowered, v["cls"] == "B")
        L.append(f"| {k} | {verdict} |")
    flagged2 = [i for i in t2 if dif2[i]["cls"] in "BC"]
    L += [
        "",
        "第二層（單題）：一律「未經恆等性檢驗」；送審查："
        + ("、".join(f"{i}（{dif2[i]['cls']}）" for i in flagged2) if flagged2 else "無"),
    ]
    L += ["", "## 未包含", "", "依原科系分組的 DIF（s1.csv 沒有科系欄位，B-C1-44）、"
          "跨時間（S1 vs S2）恆等性。"]  # fmt: skip
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")


# --- synthetic data (dry runs, tests, power) ------------------------------------------


def grm_sample(theta: np.ndarray, a: float, b: np.ndarray, rng) -> np.ndarray:
    """Logistic graded response model: P(Y >= k+2) = expit(a (theta - b_k)); 1..5."""
    p = expit(a * (theta[:, None] - b[None, :]))
    return 1 + (rng.random(len(theta))[:, None] < p).sum(axis=1)


def item_params(n_items: int, rng) -> list[tuple[float, np.ndarray]]:
    base = np.array([-1.5, -0.5, 0.5, 1.5])
    return [(float(rng.uniform(1.2, 2.2)), base + rng.uniform(-0.5, 0.5)) for _ in range(n_items)]


def answers_with_dif(theta, st, a, b, dif: dict | None, rng) -> np.ndarray:
    """dif = {"group": g, "shift": delta (threshold shift), "a_mult": m} for the focal."""
    y = grm_sample(theta, a, b, rng)
    if dif:
        f = st == dif["group"]
        y[f] = grm_sample(theta[f], a * dif.get("a_mult", 1.0), b + dif.get("shift", 0.0), rng)
    return y


PLANTED = {  # default planted DIF of --make-fake (synthetic only)
    "riasec_i#1": {"group": 2, "shift": 1.0},
    "quant_intensity#2": {"group": 1, "shift": -1.0},
    "bi_law#1": {"group": 3, "shift": 1.0},
}


def latents(st: np.ndarray, impact, rng) -> dict[str, np.ndarray]:
    """Synthetic latent traits: 6 RIASEC + 4 core N(impact[stratum], 1); each area =
    0.7 x mean of its linked types + sqrt(0.51) noise; each group = its primary areas'
    mean, rescaled to SD 1."""
    n, types = len(st), area_types()
    mu = np.array(impact)[st]
    lat = {k: mu + rng.normal(size=n) for k in [*(f"riasec_{t}" for t in "riasec"),
                                                *CORE_MODELS["core"]]}  # fmt: skip
    for k in BASIC_INTEREST_KEYS:
        lt = np.mean([lat[f"riasec_{t}"] for t in types[k]], axis=0)
        lat[k] = 0.7 * lt + math.sqrt(1 - 0.49) * rng.normal(size=n)
    for g in sorted(GROUP_LABELS):
        v = np.mean([lat[k] for k in BASIC_INTEREST_KEYS if AREA_GROUPS[k][0] == g], axis=0)
        lat[f"group_{g}"] = v / v.std()
    return lat


def make_fake(folder: Path, n_per: int, seed: int, planted: dict | None = None,
              impact: tuple = (0.0, -0.3, -0.6, -0.3), missing: float = 0.02) -> None:  # fmt: skip
    """Synthetic S1 file (every row synthetic = true) from a graded response model on
    latents(); DIF as in `planted` (default PLANTED), also written to planted_dif.json."""
    planted = PLANTED if planted is None else planted
    rng = np.random.default_rng(seed)
    bank = [it["item_id"] for it in item_bank()]
    dims = dim_items()
    st = np.repeat(np.arange(4), n_per)
    lat = latents(st, impact, rng)
    item_dim = {i: k for k, its in dims.items() for i in its}
    pars = dict(zip(bank, item_params(len(bank), rng), strict=True))
    Y = np.column_stack([answers_with_dif(lat[item_dim[i]], st, *pars[i], planted.get(i), rng)
                         for i in bank])  # fmt: skip
    Y[rng.random(Y.shape) < missing] = 0
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "s1.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["respondent_id", "stratum", "synthetic", *bank])
        for r in range(len(st)):
            w.writerow([f"F{r:04d}", STRATA[st[r]], "true", *(v if v else "" for v in Y[r])])
    (folder / "planted_dif.json").write_text(json.dumps(planted, indent=1), "utf-8")
    print(f"wrote synthetic S1 file to {folder}")


# --- operating characteristics (--power) ----------------------------------------------
# Pre-registered with the rules above (before any study data): all 73 items simulated
# from latents() with the PLANTED-free generator, item parameters fixed by seed 2026;
# DIF planted only in STUDIED for the distant stratum; equal n per stratum.

POWER_N = (40, 100, 200)
POWER_REPS = 200
POWER_IMPACT = (0.0, -0.3, -0.6, -0.3)
STUDIED = "riasec_i#1"
SCENARIOS = {  # name: (impact, dif on STUDIED for the distant stratum)
    "N0 無 DIF、無平均差": ((0.0, 0.0, 0.0, 0.0), None),
    "N1 無 DIF、有平均差": (POWER_IMPACT, None),
    "U5 均勻 DIF δ = 0.5": (POWER_IMPACT, {"group": 2, "shift": 0.5}),
    "U10 均勻 DIF δ = 1.0": (POWER_IMPACT, {"group": 2, "shift": 1.0}),
    "NU 非均勻 DIF a × 0.5": (POWER_IMPACT, {"group": 2, "a_mult": 0.5}),
}
EXTRA_N = {"N1 無 DIF、有平均差": (400,), "U10 均勻 DIF δ = 1.0": (400,)}  # v2: trend in n


def scenario_ns(name: str, ns=POWER_N) -> tuple[int, ...]:
    return tuple(ns) + EXTRA_N.get(name, ())


def sim_items(n_per: int, impact, dif, pars, rng) -> tuple[np.ndarray, np.ndarray]:
    bank = [it["item_id"] for it in item_bank()]
    item_dim = {i: k for k, its in dim_items().items() for i in its}
    st = np.repeat(np.arange(4), n_per)
    lat = latents(st, impact, rng)
    Y = np.column_stack([answers_with_dif(lat[item_dim[i]], st, *pars[i],
                                          dif if i == STUDIED else None, rng)
                         for i in bank])  # fmt: skip
    return Y, st


def _power_cell(args) -> tuple:
    name, n, reps = args
    impact, dif = SCENARIOS[name]
    bank = [it["item_id"] for it in item_bank()]
    dims, match = dim_items(), matching_items()
    t1, t2 = tiers()
    cols = {i: j for j, i in enumerate(bank)}
    pars = dict(zip(bank, item_params(len(bank), np.random.default_rng(SEED)), strict=True))
    sdim = STUDIED.split("#")[0]
    other_dims = [k for k, its in dims.items() if len(its) > 1 and k != sdim]
    rng = np.random.default_rng([SEED, n, list(SCENARIOS).index(name)])
    do_t2 = dif is None and n <= max(POWER_N)
    c = defaultdict(float)
    for _ in range(reps):
        Y, st = sim_items(n, impact, dif, pars, rng)
        d1 = dif_family(Y, st, cols, t1, match)
        dd = dimension_dif(d1, dims)
        c["fp_dim_c"] += sum(dd[k]["cls"] == "C" for k in other_dims)
        c["fp_dim_bc"] += sum(dd[k]["cls"] != "A" for k in other_dims)
        c["any_bc"] += any(dd[k]["cls"] != "A" for k in other_dims)
        c["fp_lr"] += sum(d1[i]["lr_class"] == "C" for k in other_dims for i in dims[k])
        c["fp_mh"] += sum(d1[i]["mh_class"] == "CC" for k in other_dims for i in dims[k])
        c["any_c"] += any(dd[k]["cls"] == "C" for k in other_dims)
        c["pw_c"] += dd[sdim]["cls"] == "C"
        c["pw_bc"] += dd[sdim]["cls"] != "A"
        c["pw_lr"] += any(d1[i]["lr_class"] == "C" for i in dims[sdim])
        c["pw_mh"] += any(d1[i]["mh_class"] == "CC" for i in dims[sdim])
        if dif is not None and dd[sdim]["cls"] != "A" and 2 in dd[sdim]["focal"]:
            aid = localisation_aid(Y, st, cols, dims, {sdim: {"cls": dd[sdim]["cls"],
                                                              "focal": [2]}})  # fmt: skip
            c["aid_n"] += 1
            c["aid_ok"] += aid[sdim][0][0] == STUDIED
        if do_t2:
            d2 = dif_family(Y, st, cols, t2, match)
            c["t2_c"] += sum(d2[i]["cls"] == "C" for i in t2)
            c["t2_bc"] += sum(d2[i]["cls"] != "A" for i in t2)
    k = len(other_dims)
    out = {"fp_dim_c": c["fp_dim_c"] / (reps * k), "fp_dim_bc": c["fp_dim_bc"] / (reps * k),
           "fp_lr": c["fp_lr"] / (reps * 2 * k), "fp_mh": c["fp_mh"] / (reps * 2 * k),
           "any_c": c["any_c"] / reps, "any_bc": c["any_bc"] / reps,
           "pw_c": c["pw_c"] / reps, "pw_bc": c["pw_bc"] / reps,
           "pw_lr": c["pw_lr"] / reps, "pw_mh": c["pw_mh"] / reps,
           "aid": c["aid_ok"] / c["aid_n"] if c["aid_n"] else math.nan,
           "aid_n": c["aid_n"]}  # fmt: skip
    if do_t2:
        out |= {"t2_c": c["t2_c"] / (reps * len(t2)), "t2_bc": c["t2_bc"] / (reps * len(t2))}
    print(name, n, out, flush=True)
    return (name, n), out


def population_effects() -> dict[str, tuple[float, float, float, float]]:
    """Large-sample (20,000 per stratum) effect sizes of the planted DIF:
    (|SMD|/SD and Delta R2_N, no mean differences, same_field vs distant only;
     |SMD|/SD distant vs same_field and Delta R2_N of the 4-stratum rule (df 6), both
     under the simulated mean differences POWER_IMPACT)."""
    bank = [it["item_id"] for it in item_bank()]
    cols, match = {i: j for j, i in enumerate(bank)}, matching_items()
    pars = dict(zip(bank, item_params(len(bank), np.random.default_rng(SEED)), strict=True))
    pop, rng = {}, np.random.default_rng(SEED + 1)
    j, mj = cols[STUDIED], [cols[m] for m in match[STUDIED]]
    for name, (impact, dif) in SCENARIOS.items():
        if dif is None:
            continue
        Y, st = sim_items(20000, (0, 0, 0, 0), dif, pars, rng)
        sel = (st == REF) | (st == 2)
        y, m = Y[sel, j], Y[sel][:, mj].sum(axis=1)
        es0, dr0 = mantel(y, st[sel] == 2, m)["es"], lr_dif(y, m, st[sel])["dr2"]
        Y, st = sim_items(20000, impact, dif, pars, rng)
        y, m = Y[:, j], Y[:, mj].sum(axis=1)
        es1 = mantel(y[sel], st[sel] == 2, m[sel])["es"]
        pop[name] = (es0, dr0, es1, lr_dif(y, m, st)["dr2"])
    return pop


def power(out: Path, reps: int = POWER_REPS, ns=POWER_N, workers: int = 2) -> dict:
    cells = [(name, n, reps) for name in SCENARIOS for n in scenario_ns(name, ns)]
    with Pool(workers) as pool:
        res = dict(pool.map(_power_cell, cells))
    write_power(out, res, population_effects(), reps, ns)
    return res


def write_power(out: Path, res: dict, pop: dict, reps: int, ns) -> None:
    def pct(x):
        return "—" if isinstance(x, float) and math.isnan(x) else f"{100 * x:.1f}%"

    L = ["# W-73 DIF 規則的操作特性（**合成資料試跑，不是研究結果**）", "",
         f"> 產生：`python scripts/analyze_w73_invariance.py --power`（種子 {SEED}，每格 {reps} "
         "次）。規則是腳本檔頭預先登記的版本；本檔只用合成的等級反應模型資料，回答「在計畫的"
         "每層人數下，規則能不能抓到 DIF、會不會誤報」。結論見 [FINDINGS](FINDINGS.md) F160–F166。"
         "v2（審查後）：修正 B-C1-45 並加入 n = 400 與有平均差時的母體效果量。",
         f"> 設定：73 題全部模擬，四層等人數；題目 a ~ U(1.2, 2.2)、閾值 (−1.5, −0.5, 0.5, 1.5) "
         f"+ U(−0.5, 0.5)；平均差情境各層潛在平均 {POWER_IMPACT}（same_field、adjacent、"
         "distant、multi_degree）；領域潛在特質 = 0.7 × 連結類型 + 雜訊。DIF 只放在 "
         f"{STUDIED}（distant 層）。", "",
         "## 1. 第一層誤報（被研究維度以外的 9 個維度）", "",
         "| 情境 | 每層 n | 維度判 C | 維度判 B 或 C | 題目 LR = C | 題目 MH = CC "
         "| 每次分析至少一個維度 C | 至少一個維度 B 或 C |",
         "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name in SCENARIOS:
        for n in scenario_ns(name, ns):
            r = res[(name, n)]
            L.append(f"| {name} | {n} | {pct(r['fp_dim_c'])} | {pct(r['fp_dim_bc'])} | "
                     f"{pct(r['fp_lr'])} | {pct(r['fp_mh'])} | {pct(r['any_c'])} | "
                     f"{pct(r['any_bc'])} |")  # fmt: skip
    L += ["", f"## 2. 第一層檢定力（{STUDIED.split('#')[0]} 維度）", "",
          "| 情境 | 每層 n | 維度 C | 維度 B 或 C（偵測） | 只看 LR = C | 只看 MH = CC "
          "| 定位輔助指對（次數） |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for name in SCENARIOS:
        if name not in pop:
            continue
        for n in scenario_ns(name, ns):
            r = res[(name, n)]
            L.append(f"| {name} | {n} | {pct(r['pw_c'])} | {pct(r['pw_bc'])} | "
                     f"{pct(r['pw_lr'])} | {pct(r['pw_mh'])} | "
                     f"{pct(r['aid'])}（{int(r['aid_n'])}） |")  # fmt: skip
    se = math.sqrt(0.25 / reps)
    L += ["", f"蒙地卡羅標準誤：比例 p 的 SE = √(p(1−p)/{reps})，p = 0.5 時最大 "
          f"{100 * se:.1f} 個百分點。", "",
          "母體效果量（每層 20,000 人，被研究題）：", "",
          "| 情境 | \\|SMD\\|/SD 無平均差（same_field vs distant） | ΔR²_N 兩組 "
          "| \\|SMD\\|/SD 有平均差（distant） | ΔR²_N 四層規則（df 6，有平均差） |",
          "|---|---|---|---|---|"]  # fmt: skip
    for name, (es0, dr0, es1, dr1) in pop.items():
        L.append(f"| {name} | {abs(es0):.3f} | {dr0:.4f} | {abs(es1):.3f} | {dr1:.4f} |")
    L += ["",
          "## 3. 第二層誤報（53 個單題，沒有任何 DIF）", "",
          "| 情境 | 每層 n | 題目判 C | 題目判 B 或 C |", "|---|---|---|---|"]  # fmt: skip
    for name in SCENARIOS:
        for n in scenario_ns(name, ns):
            r = res[(name, n)]
            if "t2_c" in r:
                L.append(f"| {name} | {n} | {pct(r['t2_c'])} | {pct(r['t2_bc'])} |")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")


IMPACT_CHECK = {"×1": POWER_IMPACT, "×5/3": (0.0, -0.5, -1.0, -0.5)}
IMPACT_REPS = 20


def impact_check(out: Path, reps: int = IMPACT_REPS, ns=POWER_N) -> dict:
    """Added after the reference run (exploratory, report only): Tier 2 items without
    DIF, MH class only (LR never reached C in the reference run), for two sizes of the
    stratum mean differences. Appends section 4 to the power file."""
    bank = [it["item_id"] for it in item_bank()]
    cols, match = {i: j for j, i in enumerate(bank)}, matching_items()
    _, t2 = tiers()
    pars = dict(zip(bank, item_params(len(bank), np.random.default_rng(SEED)), strict=True))
    res = {}
    for k, (label, impact) in enumerate(IMPACT_CHECK.items()):
        for n in ns:
            rng = np.random.default_rng([SEED, n, 100 + k])
            flags = 0
            for _ in range(reps):
                Y, st = sim_items(n, impact, None, pars, rng)
                d = dif_family(Y, st, cols, t2, match, lr=False)
                flags += sum(d[i]["cls"] == "C" for i in t2)
            res[(label, n)] = flags / (reps * len(t2))
            print(label, n, res[(label, n)], flush=True)
    L = ["", "## 4. 第二層誤報與層間平均差的大小（看過第 1–3 節後追加；探索性，只報告）", "",
         f"MH 分類（CC）；每格 {reps} 份資料 × 53 題；沒有任何 DIF。", "",
         "| 平均差 | 各層潛在平均 | 每層 n | 題目判 CC |", "|---|---|---|---|"]  # fmt: skip
    for label, impact in IMPACT_CHECK.items():
        for n in ns:
            L.append(f"| {label} | {impact} | {n} | {100 * res[(label, n)]:.1f}% |")
    with open(out, "a", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"appended to {out}")
    return res


def asymmetry_check(out: Path, reps: int = IMPACT_REPS, n: int = 200) -> dict:
    """Added in v2 after review (report only): the two items of a dimension share the
    Mantel statistic and opposite SMDs, but |SMD|/SD differs because each item has its
    own SD. Appends section 5 to the power file."""
    bank = [it["item_id"] for it in item_bank()]
    cols, match, dims = {i: j for j, i in enumerate(bank)}, matching_items(), dim_items()
    t1, _ = tiers()
    pars = dict(zip(bank, item_params(len(bank), np.random.default_rng(SEED)), strict=True))
    rng = np.random.default_rng([SEED, n, 200])
    gaps, differ, total = [], 0, 0
    for _ in range(reps):
        Y, st = sim_items(n, POWER_IMPACT, {"group": 2, "shift": 1.0}, pars, rng)
        d = dif_family(Y, st, cols, t1, match, lr=False)
        for its in dims.values():
            if len(its) != 2:
                continue
            for g in FOCALS:
                a, b = (d[i]["mh"][g] for i in its)
                gaps.append(abs(abs(a["es"]) - abs(b["es"])))
                differ += a["cls"] != b["cls"]
                total += 1
    res = {"max": max(gaps), "mean": float(np.mean(gaps)), "differ": differ, "total": total}
    L = ["", "## 5. 兩題維度的 \\|SMD\\|/SD 不對稱（v2 審查後追加；只報告）", "",
         f"每層 {n} 人、U10 情境、{reps} 份資料；每個兩題維度 × 每個焦點層比較兩題的 "
         "\\|SMD\\|/SD。", "",
         "| 比較數 | 兩題 \\|SMD\\|/SD 差的最大值 | 平均 | 兩題落在不同 ETS 類別 |",
         "|---|---|---|---|",
         f"| {total} | {res['max']:.3f} | {res['mean']:.4f} | {differ} |"]  # fmt: skip
    with open(out, "a", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(res, f"appended to {out}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "private" / "study")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--make-fake", type=Path, default=None, help="write a synthetic s1.csv")
    ap.add_argument("--n-per", type=int, default=50, help="fake respondents per stratum")
    ap.add_argument("--no-dif", action="store_true", help="fake data without planted DIF")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--power", action="store_true", help="operating characteristics")
    ap.add_argument("--reps", type=int, default=POWER_REPS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--impact-check", action="store_true", help="Tier 2 FPR vs mean gaps")
    ap.add_argument("--asymmetry", action="store_true", help="two-item |SMD|/SD asymmetry")
    args = ap.parse_args()
    if args.asymmetry:
        asymmetry_check(args.out or POWER_OUT)
        return
    if args.impact_check:
        impact_check(args.out or POWER_OUT)
        return
    if args.power:
        power(args.out or POWER_OUT, reps=args.reps, workers=args.workers)
        return
    if args.make_fake:
        make_fake(args.make_fake, args.n_per, args.seed, {} if args.no_dif else None)
        return
    run(args.data, args.out or args.data / "w73_analysis.md")


if __name__ == "__main__":
    main()
