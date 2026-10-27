"""C3.2 v0: hierarchical logistic admission model with a self-report correction
(W-44; THEORY "C3.2 v0 估計方式", "自陳資料的選擇偏誤校正"; A-C3-3 … A-C3-5).

    logit P(admit_i) = a_p + beta . x_i + b * [record i is self-reported]
    a_p ~ N(mu_f, tau^2)    programme p shrinks towards its CIP family f
    mu_f ~ N(0, 2^2), beta ~ N(0, 1), b ~ N(0, 2^2)
    optional anchor per programme: (logit r_p - a_p)^2 / (2 kappa^2), r_p = official rate

Posterior mode by Newton's method; intervals by the Laplace approximation (inverse
Hessian). Predictions are for the applicant population, i.e. with b = 0. b is only
estimated when something identifies it (complete records next to self-reports, or
official-rate anchors); otherwise it is fixed at 0 and the model is labelled
uncorrected. Features must be standardised against the applicant pool (A-C3-5).

v0.1 options (THEORY "C3.2 v0.1 候選"; A-C3-6):
    tau_average       average over TAU_GRID with Laplace marginal-likelihood weights
                      instead of picking one tau (B-C3-9). DEFAULT since the
                      pre-registered comparison F136; tau_average=False is v0.
    anchor="pool"     anchor r_p to the mean of sigmoid(a_p + beta.x) over the
                      programme's records, admitted self-reports weighted by e^(-b)
                      (Gauss-Newton term, backtracking line search). DEFAULT since
                      the fresh-seed confirmation F139 (v0.2); anchor="mean" is v0.1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from aisc.contracts import AdmitEstimate

SOURCES = ("self_report", "complete")
FAMILY_SD, BETA_SD, SHIFT_SD, TYPE_SD, YEAR_SD = 2.0, 1.0, 2.0, 1.0, 1.0
TAU_GRID = (0.25, 0.5, 0.75, 1.0, 1.5)


@dataclass(frozen=True)
class AdmissionRecord:
    program_id: str
    family: str
    features: tuple[float, ...]
    admitted: int
    source: str = "complete"
    year: int | None = None  # application year (B-C3-3); all records or none

    def __post_init__(self) -> None:
        if self.admitted not in (0, 1):
            raise ValueError("admitted must be 0 or 1")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        if not all(math.isfinite(v) for v in self.features):
            raise ValueError("features must be finite")


@dataclass(frozen=True)
class AdmissionModel:
    programs: tuple[str, ...]
    families: tuple[str, ...]
    program_family: dict[str, str]
    n_features: int
    theta: np.ndarray  # [beta (k), mu_f (F), eta_t (T), a_p (P), delta_y (Y), b]
    cov: np.ndarray  # Laplace covariance
    tau: float
    shift_estimated: bool
    method: str
    mixture: tuple = ()  # ((weight, theta, cov, tau), ...) when tau is averaged
    ptypes: tuple[str, ...] = ()  # programme types (W-74); empty = no type level
    program_type: dict = field(default_factory=dict)
    years: tuple[int, ...] = ()  # non-reference years with an effect (B-C3-3)
    reference_year: int | None = None  # predictions are for this (latest) year

    @property
    def shift(self) -> float:
        """Estimated logit shift of self-reported records (log q1 / q0 under A-C3-3)."""
        if self.mixture:
            return float(sum(w * th[-1] for w, th, _, _ in self.mixture))
        return float(self.theta[-1])

    def _z(self, program_id: str, features: Sequence[float], family: str | None, ptype: str | None):
        k, nf, nt = self.n_features, len(self.families), len(self.ptypes)
        npg, ny = len(self.programs), len(self.years)
        if len(features) != k:
            raise ValueError(f"expected {k} features")
        z = np.zeros(k + nf + nt + npg + ny + 1)  # delta_y = 0: the reference year
        z[:k] = features
        extra = 0.0
        if program_id in self.program_family:
            z[k + nf + nt + self.programs.index(program_id)] = 1.0
        else:
            if family not in self.families:
                raise ValueError(f"unknown programme {program_id!r}: pass a known family")
            z[k + self.families.index(family)] = 1.0
            if self.ptypes:
                if ptype not in self.ptypes:
                    raise ValueError(f"unknown programme {program_id!r}: pass a known ptype")
                z[k + nf + self.ptypes.index(ptype)] = 1.0
            extra = self.tau**2  # a new programme varies around its family (+ type) mean
        return z, extra

    def predict(
        self,
        program_id: str,
        features: Sequence[float],
        level: float = 0.9,
        family: str | None = None,
        ptype: str | None = None,
    ) -> AdmitEstimate:
        """P(admit) for an applicant from the population (b = 0): posterior median and
        a central `level` interval under the Laplace approximation."""
        if not 0 < level < 1:
            raise ValueError("level must lie in (0, 1)")
        z, extra = self._z(program_id, features, family, ptype)
        if self.mixture:  # moment-matched mixture over tau (B-C3-9)
            comps = [(w, float(z @ th), float(z @ cv @ z) + (t**2 if extra else 0.0))
                     for w, th, cv, t in self.mixture]  # fmt: skip
            eta = sum(w * e for w, e, _ in comps)
            var = sum(w * (v + e**2) for w, e, v in comps) - eta**2
        else:
            eta = float(z @ self.theta)
            var = float(z @ self.cov @ z) + extra
        sd = math.sqrt(max(var, 0.0))
        q = NormalDist().inv_cdf(0.5 + level / 2)

        def sig(v: float) -> float:
            return 1 / (1 + math.exp(-v))

        return AdmitEstimate(program_id, sig(eta), sig(eta - q * sd), sig(eta + q * sd),
                             self.method)  # fmt: skip


def _design(records, programs, families, k, nt=0, years=()):
    nf, npg = len(families), len(programs)
    Z = np.zeros((len(records), k + nf + nt + npg + len(years) + 1))
    y = np.zeros(len(records))
    for i, r in enumerate(records):
        Z[i, :k] = r.features
        Z[i, k + nf + nt + programs.index(r.program_id)] = 1.0
        if r.year in years:  # older year: its own level shift (B-C3-3)
            Z[i, k + nf + nt + npg + years.index(r.year)] = 1.0
        Z[i, -1] = 1.0 if r.source == "self_report" else 0.0
        y[i] = r.admitted
    return Z, y


def _prior(programs, families, fam_of, k, tau, anchors, kappa, estimate_shift,
           ptypes=(), type_of=None, ny=0):  # fmt: skip
    nf, npg, nt = len(families), len(programs), len(ptypes)
    d = k + nf + nt + npg + ny + 1
    Q, c = np.zeros((d, d)), np.zeros(d)
    Q[np.arange(k), np.arange(k)] = 1 / BETA_SD**2
    for j in range(nf):
        Q[k + j, k + j] += 1 / FAMILY_SD**2
    for j in range(nt):
        Q[k + nf + j, k + nf + j] += 1 / TYPE_SD**2
    for i, pid in enumerate(programs):
        a, m = k + nf + nt + i, k + families.index(fam_of[pid])
        # (a - mu_f - eta_t)^2 / tau^2 ; eta_t only with a type level (W-74)
        idx = [a, m] + ([k + nf + ptypes.index(type_of[pid])] if nt else [])
        sign = [1.0, -1.0] + ([-1.0] if nt else [])
        for x, sx in zip(idx, sign, strict=True):
            for y, sy in zip(idx, sign, strict=True):
                Q[x, y] += sx * sy / tau**2
        if pid in anchors:
            Q[a, a] += 1 / kappa**2
            c[a] += anchors[pid] / kappa**2
    for j in range(ny):
        Q[k + nf + nt + npg + j, k + nf + nt + npg + j] += 1 / YEAR_SD**2
    Q[-1, -1] = 1 / SHIFT_SD**2 if estimate_shift else 1e12  # pinned at 0 when not identified
    return Q, c


def _objective(theta, Z, y, Q, c, pool, k, kappa, use_b):
    """Negative log posterior, gradient and (Gauss-Newton) Hessian."""
    eta = Z @ theta
    p = 1 / (1 + np.exp(-eta))
    f = float(np.sum(np.logaddexp(0, eta) - y * eta)) + 0.5 * theta @ Q @ theta - c @ theta
    g = Z.T @ (p - y) + Q @ theta - c
    H = Z.T @ (Z * (p * (1 - p))[:, None]) + Q
    b = theta[-1] if use_b else 0.0
    for ai, rows, lr in pool:  # pool-integrated anchors (v0.1)
        X, yy, ss = Z[rows, :k], y[rows], Z[rows, -1]
        w = np.exp(-b * yy * ss)
        s0 = 1 / (1 + np.exp(-(theta[ai] + X @ theta[:k])))
        d0 = s0 * (1 - s0)
        W = w.sum()
        m = float(np.clip((w * s0).sum() / W, 1e-9, 1 - 1e-9))
        rho = math.log(m / (1 - m)) - lr
        dm = np.zeros_like(theta)
        dm[ai] = (w * d0).sum() / W
        dm[:k] = (w * d0) @ X / W
        if use_b:
            dm[-1] = (-(w * yy * ss * s0).sum() + m * (w * yy * ss).sum()) / W
        J = dm / (m * (1 - m))
        f += rho**2 / (2 * kappa**2)
        g = g + J * rho / kappa**2
        H = H + np.outer(J, J) / kappa**2
    return f, g, H


def _fit(Z, y, Q, c, pool=(), k=0, kappa=1.0, use_b=False, max_iter=200):
    theta = np.zeros(Z.shape[1])
    f, g, H = _objective(theta, Z, y, Q, c, pool, k, kappa, use_b)
    for _ in range(max_iter):
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:
            cand = theta - t * step
            f_new, g_new, H_new = _objective(cand, Z, y, Q, c, pool, k, kappa, use_b)
            if f_new <= f - 1e-4 * t * float(g @ step) or t < 1e-8:
                break
            t /= 2
        theta, f, g, H = cand, f_new, g_new, H_new
        if np.max(np.abs(t * step)) < 1e-10:
            break
    return theta, H, f


def fit_admission_model(
    records: Sequence[AdmissionRecord],
    families: Mapping[str, str] | None = None,
    official_rates: Mapping[str, float] | None = None,
    tau: float | None = None,
    kappa: float = 0.3,
    anchor: str = "pool",
    tau_average: bool | None = None,
    types: Mapping[str, str] | None = None,
) -> AdmissionModel:
    """Fit C3.2 v0. `families` maps programme -> CIP family (default: from records);
    `official_rates` anchors programme intercepts (A-C3-5); `tau` None averages over
    TAU_GRID with Laplace marginal-likelihood weights (default since F136; pass
    tau_average=False for the v0 reference, which picks the single best tau).
    `anchor` "pool" (default since F139) anchors official rates over the programme's
    reweighted records; "mean" is the v0 / v0.1 reference (module docstring).
    `types` maps every programme to an admission-process type (W-74, A-C3-7); it adds
    a type effect shared across families. None = no type level (unchanged model).
    Records with `year`: every older year gets a common logit shift and predictions
    are for the latest year (B-C3-3, A-C3-8); official rates should be that year's."""
    if anchor not in ("mean", "pool"):
        raise ValueError('anchor must be "mean" or "pool"')
    if tau_average and tau is not None:
        raise ValueError("tau_average needs tau=None")
    if tau_average is None:
        tau_average = tau is None
    if not records:
        raise ValueError("no records")
    k = len(records[0].features)
    if any(len(r.features) != k for r in records):
        raise ValueError("all records need the same number of features")
    fam_of = dict(families or {})
    for r in records:
        if fam_of.setdefault(r.program_id, r.family) != r.family:
            raise ValueError(f"programme {r.program_id} has two families")
    programs = tuple(sorted({r.program_id for r in records}))
    fams = tuple(sorted({fam_of[p] for p in programs}))
    type_of = dict(types or {})
    if types is not None:
        untyped = [p for p in programs if p not in type_of]
        if untyped:
            raise ValueError(f"no type for programmes {untyped}")
    ptypes = tuple(sorted({type_of[p] for p in programs})) if types is not None else ()
    nt = len(ptypes)
    yrs = {r.year for r in records}
    if None in yrs and len(yrs) > 1:
        raise ValueError("give a year for every record or for none")
    ref_year = max(yrs) if None not in yrs else None
    years = tuple(sorted(y for y in yrs if y is not None and y != ref_year))
    anchors = {}
    for pid, rate in (official_rates or {}).items():
        if pid in programs:
            if not 0 < rate < 1:
                raise ValueError("official rates must lie in (0, 1)")
            anchors[pid] = math.log(rate / (1 - rate))
    srcs = {r.source for r in records}
    has_self = "self_report" in srcs
    estimate_shift = has_self and ("complete" in srcs or bool(anchors))
    Z, y = _design(records, list(programs), list(fams), k, nt, years)
    nf = len(fams)
    pool, mean_anchors = [], anchors
    if anchor == "pool":
        mean_anchors = {}
        for pid, lr in anchors.items():
            rows = np.array([i for i, r in enumerate(records) if r.program_id == pid])
            pool.append((k + nf + nt + programs.index(pid), rows, lr))

    def run(t):
        Q, c = _prior(list(programs), list(fams), fam_of, k, t, mean_anchors, kappa,
                      estimate_shift, list(ptypes), type_of, len(years))  # fmt: skip
        theta, H, f = _fit(Z, y, Q, c, pool, k, kappa, estimate_shift)
        _, logdet = np.linalg.slogdet(H)
        log_ml = -(f + len(programs) * math.log(t)) - 0.5 * logdet
        cov = np.linalg.inv(H)
        if not estimate_shift:
            cov[-1, :] = cov[:, -1] = 0.0
        return theta, cov, log_ml

    mixture: tuple = ()
    if tau is None:
        fits = {t: run(t) for t in TAU_GRID}
        tau = max(fits, key=lambda t: fits[t][2])
        theta, cov, _ = fits[tau]
        if tau_average:
            top = max(v[2] for v in fits.values())
            raw = {t: math.exp(v[2] - top) for t, v in fits.items()}
            tot = sum(raw.values())
            mixture = tuple((raw[t] / tot, fits[t][0], fits[t][1], t) for t in TAU_GRID)
    else:
        if tau <= 0:
            raise ValueError("tau must be positive")
        theta, cov, _ = run(tau)
    version = "v0.2" if anchor == "pool" else ("v0.1" if tau_average else "v0")
    if not has_self:
        method = f"hierarchical_{version}"
    elif estimate_shift:
        method = f"hierarchical_{version}_corrected"
    else:
        method = f"hierarchical_{version}_uncorrected"
    return AdmissionModel(programs, fams, {p: fam_of[p] for p in programs}, k, theta, cov,
                          float(tau), estimate_shift, method, mixture, ptypes,
                          {p: type_of[p] for p in programs} if nt else {},
                          years, ref_year)  # fmt: skip
