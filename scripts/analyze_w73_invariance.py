"""Pre-registered measurement-invariance and DIF analysis of the W-41 field study
(W-73, V-C1-9; THEORY "恆等性與 DIF 的實作與判準"; A-C1-19, A-C1-30, A-C1-31).

Reads the same de-identified S1 file as scripts/analyze_w41_study.py
(data/private/study/s1.csv; strata same_field / adjacent / distant / multi_degree).
Answers are scored (a - 1) / 4.

PRE-REGISTERED (written and committed before any study data exist, 2026-10-29).
BINDING as in analyze_w41_study.py: >= 200 respondents used and >= 40 per stratum
(protocol §2); otherwise every verdict is printed as "pilot, not binding".

A. Invariance (V-C1-9). Model: the 10 multi-item dimensions (6 RIASEC, 2 intensity,
   2 orientation; 2 items each) as 10 correlated factors over their 20 items; S1
   respondents with all 20 items. Multi-group normal-theory ML on each stratum's mean
   vector and covariance (divisor n); the first item of each factor has loading 1.
     configural  loadings, intercepts, factor covariances and uniquenesses free per
                 group; factor means 0
     metric      loadings equal across groups
     scalar      loadings and intercepts equal; factor means free except group 1
   chi2 = sum_g n_g F_g; CFI against the independence model (variances and means free
   per group); RMSEA = sqrt(max(chi2 - df, 0) / (df N)) sqrt(G).
   Rules:
     - the configural model must converge with CFI >= 0.90; otherwise the factor
       structure is not supported and V-C1-9 is "not evaluable";
     - V-C1-9 PASS (metric invariance, as written in VALIDATION) if
       CFI(configural) - CFI(metric) <= 0.01;
     - scalar invariance (needed to compare score levels across backgrounds, D11)
       holds if CFI(metric) - CFI(scalar) <= 0.01.
   Groups: the four strata. If a stratum has fewer than 40 usable respondents, or the
   four-group configural model does not converge, the same rules are applied to two
   groups, near (same_field + adjacent) vs far (distant + multi_degree), labelled so.
   Report only: delta RMSEA (<= 0.015, Chen 2007); per dimension, the delta CFI when
   only that dimension's loading (metric step) or intercepts (scalar step, its factor
   mean freed) are constrained; item skewness and stratum means (A-C1-30).
B. DIF (A-C1-31). Proportional-odds logistic regression per item: M1 item ~ match;
   M2 + stratum (3 dummies); M3 + stratum x match. Likelihood-ratio test M3 vs M1
   (df 6); effect size Nagelkerke R2(M3) - R2(M1). An item SHOWS DIF if its
   Benjamini-Hochberg adjusted p (within its tier) < 0.05 AND delta R2 >= 0.035
   (moderate; large >= 0.070). Uniform (M2 vs M1) vs non-uniform (M3 vs M2) is
   reported for flagged items.
     binding tier    the 20 items of 2-item dimensions; match = the dimension's total
                     score (both items), standardised
     screening tier  the 41 area items (match = the area's primary group item) and the
                     12 group items (match = mean of the 12 group items); flags go to
                     cognitive interviews, never straight to removal (A-C1-31: real
                     background differences in specific areas look like DIF here)
   Decision (binding tier): an item showing DIF is rewritten or removed (V-C1-9, D11)
   after the cognitive interview confirms a wording or context cause.
NOT IN THIS SCRIPT: graded response model calibration (W-73, later); robust (scaled)
chi-square; partial-invariance model search (only the per-dimension report above).
  Disclosure: checked only on synthetic data (tests and --make-fake dry runs); the ML
  engine was cross-checked once against semopy on synthetic data (see the PR).

    python scripts/analyze_w73_invariance.py --data data/private/study
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_w41_study import (  # noqa: E402
    GROUPS,
    STRATA,
    binding,
    dim_items,
    item_bank,
    make_fake,
    read_answers,
)

from aisc.dimensions import BASIC_INTEREST_KEYS  # noqa: E402
from aisc.profiling.hierarchy import AREA_GROUPS  # noqa: E402

MIN_GROUP = 40
CFI_CONFIGURAL, DELTA_CFI, DELTA_RMSEA = 0.90, 0.01, 0.015
FDR, R2_MODERATE, R2_LARGE = 0.05, 0.035, 0.070
NEAR_FAR = {"same_field": "near", "adjacent": "near", "distant": "far", "multi_degree": "far"}


# --- multi-group CFA (normal-theory ML) ----------------------------------------------


class Fit(NamedTuple):
    chi2: float
    df: int
    cfi: float
    rmsea: float
    converged: bool
    x: np.ndarray
    boundary: bool = False  # a uniqueness at its lower bound (Heywood case)


class MultiGroupCFA:
    """Simple-structure CFA: item i loads on factor fac[i]; markers[k] is factor k's
    loading-1 item. Sharing across groups is set per non-marker loading (lam_shared),
    per intercept (tau_shared) and per factor mean (kappa_free: free in groups 2..G)."""

    def __init__(self, means, covs, ns, fac, tau_equivalent=False):
        self.m = [np.asarray(v, float) for v in means]
        self.S = [np.asarray(c, float) for c in covs]
        self.n = np.asarray(ns, float)
        self.N = float(self.n.sum())
        self.G, self.p = len(self.m), len(fac)
        self.fac = np.asarray(fac)
        self.q = int(self.fac.max()) + 1
        self.markers = [int(np.flatnonzero(self.fac == k)[0]) for k in range(self.q)]
        if tau_equivalent:  # every loading fixed at 1 (exploration only, not the plan)
            self.markers = list(range(self.p))
        self.free_items = [i for i in range(self.p) if i not in self.markers]
        self.tril = np.tril_indices(self.q)
        self.logdetS = [np.linalg.slogdet(S)[1] for S in self.S]
        if any(np.linalg.slogdet(S)[0] <= 0 for S in self.S):
            raise ValueError("a group covariance matrix is not positive definite")

    # parameter layout -----------------------------------------------------------------
    def layout(self, lam_shared, tau_shared, kappa_free):
        idx, k = {}, 0

        def take(key, size):
            nonlocal k
            idx[key] = np.arange(k, k + size)
            k += size

        for j, i in enumerate(self.free_items):
            if lam_shared[j]:
                take(("lam", i, None), 1)
            else:
                for g in range(self.G):
                    take(("lam", i, g), 1)
        for g in range(self.G):
            take(("chol", g), len(self.tril[0]))
            take(("logth", g), self.p)
        for i in range(self.p):
            if tau_shared[i]:
                take(("tau", i, None), 1)
            else:
                for g in range(self.G):
                    take(("tau", i, g), 1)
        for f in range(self.q):
            if kappa_free[f]:
                for g in range(1, self.G):
                    take(("kap", f, g), 1)
        return idx, k

    def _unpack(self, x, idx, g):
        L = np.zeros((self.p, self.q))
        lam_pos = {}
        for i in range(self.p):
            if i in self.markers:
                L[i, self.fac[i]] = 1.0
                continue
            key = ("lam", i, None) if ("lam", i, None) in idx else ("lam", i, g)
            L[i, self.fac[i]] = x[idx[key][0]]
            lam_pos[i] = idx[key][0]
        C = np.zeros((self.q, self.q))
        C[self.tril] = x[idx[("chol", g)]]
        th = np.exp(x[idx[("logth", g)]])
        tau, tau_pos = np.empty(self.p), np.empty(self.p, int)
        for i in range(self.p):
            key = ("tau", i, None) if ("tau", i, None) in idx else ("tau", i, g)
            tau_pos[i] = idx[key][0]
            tau[i] = x[tau_pos[i]]
        kap, kap_pos = np.zeros(self.q), {}
        for f in range(self.q):
            if ("kap", f, g) in idx:
                kap_pos[f] = idx[("kap", f, g)][0]
                kap[f] = x[kap_pos[f]]
        return L, C, th, tau, kap, lam_pos, tau_pos, kap_pos

    def objective(self, x, idx):
        """(sum_g n_g F_g / N, gradient)."""
        total, grad = 0.0, np.zeros_like(x)
        for g in range(self.G):
            L, C, th, tau, kap, lam_pos, tau_pos, kap_pos = self._unpack(x, idx, g)
            Phi = C @ C.T
            Sig = L @ Phi @ L.T + np.diag(th)
            sign, logdet = np.linalg.slogdet(Sig)
            if sign <= 0:
                return 1e10, np.zeros_like(x)
            Si = np.linalg.inv(Sig)
            d = self.m[g] - (tau + L @ kap)
            Sd = self.S[g] + np.outer(d, d)
            F = logdet + np.trace(Sd @ Si) - self.logdetS[g] - self.p
            w = self.n[g] / self.N
            total += w * F
            Gm = Si - Si @ Sd @ Si
            gmu = -2 * Si @ d
            dL = 2 * Gm @ L @ Phi + np.outer(gmu, kap)
            for i, pos in lam_pos.items():
                grad[pos] += w * dL[i, self.fac[i]]
            M = L.T @ Gm @ L
            grad[idx[("chol", g)]] += w * (2 * M @ C)[self.tril]
            grad[idx[("logth", g)]] += w * np.diag(Gm) * th
            np.add.at(grad, tau_pos, w * gmu)
            gk = L.T @ gmu
            for f, pos in kap_pos.items():
                grad[pos] += w * gk[f]
        return total, grad

    def start(self, idx, k):
        x = np.zeros(k)
        for key, pos in idx.items():
            if key[0] == "lam":
                x[pos] = 1.0
            elif key[0] == "chol":
                g = key[1]
                v = np.sqrt(0.5 * np.mean(np.diag(self.S[g])))
                C = np.eye(self.q) * v
                x[pos] = C[self.tril]
            elif key[0] == "logth":
                x[pos] = np.log(0.5 * np.diag(self.S[key[1]]))
            elif key[0] == "tau":
                i, g = key[1], key[2]
                x[pos] = np.mean([m[i] for m in self.m]) if g is None else self.m[g][i]
        return x

    def bounds(self, idx, k):
        """Uniquenesses between 1e-3 and 2 times the item variance (no Heywood cases
        beyond that; hitting the lower bound is reported); loadings in [-10, 10]."""
        lo, hi = np.full(k, -np.inf), np.full(k, np.inf)
        for key, pos in idx.items():
            if key[0] == "logth":
                v = np.diag(self.S[key[1]])
                lo[pos], hi[pos] = np.log(1e-3 * v), np.log(2 * v)
            elif key[0] == "lam":
                lo[pos], hi[pos] = -10, 10
        return lo, hi

    def fit(self, lam_shared, tau_shared, kappa_free, x0=None, x0_idx=None) -> Fit:
        """Best of the default start and (if given) a warm start from a less constrained
        model; L-BFGS-B with analytic gradients."""
        idx, k = self.layout(lam_shared, tau_shared, kappa_free)
        lo, hi = self.bounds(idx, k)
        starts = [self.start(idx, k)]
        if x0 is not None:
            x = starts[0].copy()
            for key, pos in idx.items():
                if key in x0_idx:
                    x[pos] = x0[x0_idx[key]]
                elif key[0] in ("lam", "tau") and key[2] is None:
                    x[pos] = np.mean([x0[x0_idx[(key[0], key[1], g)]] for g in range(self.G)])
            starts.append(np.clip(x, lo, hi))
        best = None
        for x in starts:
            res = optimize.minimize(self.objective, x, args=(idx,), jac=True,
                                    method="L-BFGS-B", bounds=list(zip(lo, hi, strict=True)),
                                    options={"maxiter": 20_000, "gtol": 1e-10,
                                             "ftol": 1e-15})  # fmt: skip
            if best is None or res.fun < best.fun:
                best = res
        res = best
        at_bound = np.isclose(res.x, lo) | np.isclose(res.x, hi)
        gnorm = float(np.abs(np.where(at_bound, 0.0, res.jac)).max())
        th_keys = [pos for key, pos in idx.items() if key[0] == "logth"]
        boundary = bool(at_bound[np.concatenate(th_keys)].any())
        chi2 = float(self.N * res.fun)
        moments = self.G * (self.p * (self.p + 1) // 2 + self.p)
        df = moments - k
        chi0, df0 = self.baseline()
        num = max(chi2 - df, 0.0)
        cfi = 1 - num / max(chi0 - df0, chi2 - df, 1e-12)
        rmsea = math.sqrt(num / (df * self.N)) * math.sqrt(self.G) if df > 0 else math.nan
        self._last_idx = idx
        return Fit(chi2, df, float(cfi), rmsea, gnorm < 1e-5, res.x, boundary)

    def baseline(self) -> tuple[float, int]:
        chi0 = sum(n * (np.log(np.diag(S)).sum() - ld)
                   for n, S, ld in zip(self.n, self.S, self.logdetS, strict=True))  # fmt: skip
        moments = self.G * (self.p * (self.p + 1) // 2 + self.p)
        return float(chi0), moments - self.G * 2 * self.p


def invariance(cfa: MultiGroupCFA) -> dict:
    nf, p, q = len(cfa.free_items), cfa.p, cfa.q
    F, T = [False] * nf, [False] * p
    out = {}
    conf = cfa.fit(F, T, [False] * q)
    ci = cfa._last_idx
    met = cfa.fit([True] * nf, T, [False] * q, conf.x, ci)
    mi = cfa._last_idx
    sca = cfa.fit([True] * nf, [True] * p, [True] * q, met.x, mi)
    out.update(configural=conf, metric=met, scalar=sca)
    per = {}
    for f in range(q):
        lam = [cfa.fac[i] == f for i in cfa.free_items]
        m1 = cfa.fit(lam, T, [False] * q, conf.x, ci)
        tau = [cfa.fac[i] == f for i in range(p)]
        kap = [g == f for g in range(q)]
        s1 = cfa.fit([True] * nf, tau, kap, met.x, mi)
        per[f] = (conf.cfi - m1.cfi, met.cfi - s1.cfi, m1.converged and s1.converged)
    out["per_dim"] = per
    return out


# --- ordinal logistic DIF --------------------------------------------------------------


def ordinal_loglik(y: np.ndarray, X: np.ndarray) -> float:
    """Max log-likelihood of a proportional-odds model P(y <= k) = sigmoid(a_k - X b)
    (y coded 0..K-1, every category observed); X may have zero columns."""
    K = int(y.max()) + 1
    n, d = X.shape
    cum = np.cumsum(np.bincount(y, minlength=K))[:-1] / n
    a0 = np.log(cum / (1 - cum))
    x0 = np.concatenate([[a0[0]], np.log(np.diff(a0).clip(1e-3)), np.zeros(d)])

    def nll(par):
        a = np.cumsum(np.concatenate([[par[0]], np.exp(par[1 : K - 1])]))
        eta = X @ par[K - 1 :] if d else np.zeros(n)
        cdf = 1 / (1 + np.exp(-(a[None, :] - eta[:, None])))
        cdf = np.hstack([np.zeros((n, 1)), cdf, np.ones((n, 1))])
        pr = cdf[np.arange(n), y + 1] - cdf[np.arange(n), y]
        return -float(np.sum(np.log(np.clip(pr, 1e-300, None))))

    res = optimize.minimize(nll, x0, method="BFGS", options={"gtol": 1e-7, "maxiter": 2000})
    return -float(res.fun)


def nagelkerke(ll: float, ll0: float, n: int) -> float:
    return (1 - math.exp(2 * (ll0 - ll) / n)) / (1 - math.exp(2 * ll0 / n))


def dif_item(y_raw: np.ndarray, match: np.ndarray, groups: np.ndarray) -> dict | None:
    """Zumbo's three nested models for one item; None if fewer than 2 categories."""
    cats = np.unique(y_raw)
    if len(cats) < 2:
        return None
    y = np.searchsorted(cats, y_raw)
    z = (match - match.mean()) / (match.std() or 1.0)
    levels = sorted(set(groups))
    D = np.column_stack([(groups == g).astype(float) for g in levels[1:]])
    n = len(y)
    ll0 = ordinal_loglik(y, np.empty((n, 0)))
    ll1 = ordinal_loglik(y, z[:, None])
    ll2 = ordinal_loglik(y, np.column_stack([z, D]))
    ll3 = ordinal_loglik(y, np.column_stack([z, D, D * z[:, None]]))
    k = len(levels) - 1
    r1, r2, r3 = (nagelkerke(v, ll0, n) for v in (ll1, ll2, ll3))
    return {"p": float(stats.chi2.sf(2 * (ll3 - ll1), 2 * k)), "dr2": r3 - r1,
            "p_uniform": float(stats.chi2.sf(2 * (ll2 - ll1), k)),
            "p_nonuniform": float(stats.chi2.sf(2 * (ll3 - ll2), k))}  # fmt: skip


def bh_adjust(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = p[order] * len(p) / np.arange(1, len(p) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1].clip(max=1)
    out = np.empty_like(adj)
    out[order] = adj
    return out.tolist()


def dif_tier(items, answers, match_of, groups):
    rows = []
    for it in items:
        r = dif_item(answers[it], match_of[it], groups)
        if r is not None:
            rows.append((it, r))
    adj = bh_adjust([r["p"] for _, r in rows]) if rows else []
    out = []
    for (it, r), pa in zip(rows, adj, strict=True):
        flag = pa < FDR and r["dr2"] >= R2_MODERATE
        size = "大" if r["dr2"] >= R2_LARGE else "中" if r["dr2"] >= R2_MODERATE else "小"
        out.append((it, r, pa, flag, size))
    return out


# --- report ----------------------------------------------------------------------------


def run(data: Path, out: Path) -> None:
    dims = dim_items()
    items = [it["item_id"] for it in item_bank()]
    s1 = read_answers(data / "s1.csv", items, need_stratum=True)
    multi = [k for k, v in dims.items() if len(v) >= 2]
    cfa_items = [i for k in multi for i in dims[k]]
    fac = [multi.index(k) for k in multi for _ in dims[k]]
    use = [r for r in s1 if all(i in s1[r]["ans"] for i in cfa_items)]
    synthetic = any(s1[r]["synthetic"] for r in s1)
    PILOT = "（前導，不具約束力）"
    tag = "" if binding(use, s1, "s1") else PILOT
    counts = {s: sum(s1[r]["stratum"] == s for r in use) for s in STRATA}
    L = ["# W-73 測量恆等性與 DIF（預先登記：scripts/analyze_w73_invariance.py）", "",
         f"- 產生日期：{date.today().isoformat()} · 資料夾：`{data}`",
         f"- 答完 20 題（10 個兩題維度）的 S1 人數 {len(use)}；各層："
         + "、".join(f"{s} {counts[s]}" for s in STRATA)]  # fmt: skip
    if synthetic:
        L.append("- ⚠️ **含合成資料（synthetic = true）：不是真實學生的結果**")

    def grouped(label_of):
        labels = sorted({label_of[s1[r]["stratum"]] for r in use})
        X = {lab: np.array([[s1[r]["ans"][i] for i in cfa_items] for r in use
                            if label_of[s1[r]["stratum"]] == lab]) for lab in labels}  # fmt: skip
        return labels, X

    # A. invariance
    L += ["", f"## A. 測量恆等性（V-C1-9）{tag}", ""]
    plan = "四層"
    labels, X = grouped({s: s for s in STRATA})
    res = None
    if all(len(X.get(s, [])) >= MIN_GROUP for s in STRATA):
        try:
            cfa = MultiGroupCFA([v.mean(0) for v in X.values()],
                                [np.cov(v.T, ddof=0) for v in X.values()],
                                [len(v) for v in X.values()], fac)  # fmt: skip
            res = invariance(cfa)
            if not res["configural"].converged:
                res = None
        except (ValueError, np.linalg.LinAlgError):
            res = None
    if res is None:
        plan = "兩組（近：same_field＋adjacent；遠：distant＋multi_degree）"
        labels, X = grouped(NEAR_FAR)
        if len(X) == 2 and all(len(v) >= MIN_GROUP for v in X.values()):
            try:
                cfa = MultiGroupCFA([v.mean(0) for v in X.values()],
                                    [np.cov(v.T, ddof=0) for v in X.values()],
                                    [len(v) for v in X.values()], fac)  # fmt: skip
                res = invariance(cfa)
            except (ValueError, np.linalg.LinAlgError):
                res = None
    if res is None:
        L.append("無法估計（各組人數不足或共變異數矩陣不正定）。")
    else:
        c, m, s = res["configural"], res["metric"], res["scalar"]
        ok_conf = c.converged and c.cfi >= CFI_CONFIGURAL
        d_met, d_sca = c.cfi - m.cfi, m.cfi - s.cfi
        if not ok_conf:
            verdict = "— 無法評估：形態模型不收斂或 CFI < 0.90（因素結構不成立）"
        else:
            verdict = (f"量尺恆等 {'✅' if d_met <= DELTA_CFI else '❌'}（ΔCFI {d_met:.4f}）；"
                       f"截距恆等 {'✅' if d_sca <= DELTA_CFI else '❌'}"
                       f"（ΔCFI {d_sca:.4f}）")  # fmt: skip
        L += [f"分組：{plan}（{'、'.join(f'{k} {len(v)}' for k, v in X.items())}）", "",
              f"**V-C1-9：{verdict}**", "",
              "| 模型 | χ² | df | CFI | RMSEA | 收斂 |", "|---|---|---|---|---|---|"]  # fmt: skip
        for name, f in (("形態", c), ("量尺", m), ("截距", s)):
            L.append(f"| {name} | {f.chi2:.1f} | {f.df} | {f.cfi:.4f} | {f.rmsea:.4f} | "
                     f"{'是' if f.converged else '否'} |")  # fmt: skip
        L += ["", f"ΔRMSEA（只報告，≤ {DELTA_RMSEA}）：量尺 {m.rmsea - c.rmsea:+.4f}、"
              f"截距 {s.rmsea - m.rmsea:+.4f}", "",
              "各維度（只報告；只限制該維度時的 ΔCFI）：", "",
              "| 維度 | 量尺 ΔCFI | 截距 ΔCFI | 收斂 |", "|---|---|---|---|"]  # fmt: skip
        for f_, (dm, ds, conv) in res["per_dim"].items():
            L.append(f"| {multi[f_]} | {dm:+.4f} | {ds:+.4f} | {'是' if conv else '否'} |")
    A = np.array([[s1[r]["ans"][i] for i in cfa_items] for r in use]) if use else None
    if A is not None and len(A) > 2:
        sk = stats.skew(A, axis=0)
        worst = int(np.argmax(np.abs(sk)))
        L += ["", f"偏態（A-C1-30，只報告）：|偏態| > 1 的題數 {int((np.abs(sk) > 1).sum())}；"
              f"最大 {sk[worst]:+.2f}（{cfa_items[worst]}）"]  # fmt: skip

    # B. DIF
    L += ["", f"## B. 題目差異功能 DIF{tag}", "",
          f"判準：BH 校正 p < {FDR} 且 Nagelkerke ΔR² ≥ {R2_MODERATE}"
          f"（≥ {R2_LARGE} 為大）"]  # fmt: skip
    resp = [r for r in s1 if all(i in s1[r]["ans"] for i in items)]
    groups = np.array([s1[r]["stratum"] for r in resp])
    ans = {i: np.array([round(s1[r]["ans"][i] * 4) + 1 for r in resp]) for i in items}
    if len(resp) < 2 * MIN_GROUP or len(set(groups)) < 2:
        L.append(f"答完全部題目的人數不足（{len(resp)}）。")
    else:
        match = {}
        for k in multi:
            tot = sum(ans[i] for i in dims[k]).astype(float)
            for i in dims[k]:
                match[i] = tot
        gmean = np.mean([ans[dims[f"group_{g}"][0]] for g in GROUPS], axis=0)
        for k in BASIC_INTEREST_KEYS:
            match[dims[k][0]] = ans[dims[f"group_{AREA_GROUPS[k][0]}"][0]].astype(float)
        for g in GROUPS:
            match[dims[f"group_{g}"][0]] = gmean
        area_items = [dims[k][0] for k in BASIC_INTEREST_KEYS]
        group_items = [dims[f"group_{g}"][0] for g in GROUPS]
        for title, tier_items in (("有約束力：兩題維度的 20 題", cfa_items),
                                  ("篩檢：41 個領域題與 12 個群題（只送認知訪談）",
                                   area_items + group_items)):  # fmt: skip
            rows = dif_tier(tier_items, ans, match, groups)
            flagged = [r for r in rows if r[3]]
            L += ["", f"### {title}", "",
                  f"檢定 {len(rows)} 題；顯示 DIF {len(flagged)} 題"
                  + ("" if not flagged else "：")]  # fmt: skip
            if flagged:
                L += ["", "| 題目 | ΔR² | 程度 | 校正 p | 一致型 p | 非一致型 p |",
                      "|---|---|---|---|---|---|"]  # fmt: skip
                for it, r, pa, _, size in flagged:
                    L.append(f"| {it} | {r['dr2']:.3f} | {size} | {pa:.4f} | "
                             f"{r['p_uniform']:.4f} | {r['p_nonuniform']:.4f} |")  # fmt: skip
            big = sorted(rows, key=lambda x: -x[1]["dr2"])[:3]
            L.append("ΔR² 最大的 3 題（只報告）：" + "、".join(
                f"{it}（{r['dr2']:.3f}，p {pa:.3f}）" for it, r, pa, _, _ in big))  # fmt: skip
    L += ["", "## 未包含", "", "等級反應模型校準（W-73 後續）、穩健 χ²、部分恆等的模型搜尋。"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--data", type=Path, default=root / "data" / "private" / "study")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--make-fake", type=Path, default=None, help="write synthetic files")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.make_fake:
        make_fake(args.make_fake, args.n, min(60, args.n), args.seed)
        return
    run(args.data, args.out or args.data / "invariance.md")


if __name__ == "__main__":
    main()
