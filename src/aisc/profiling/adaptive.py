"""C1.3 Adaptive questionnaire v1 (W-25).

Model (A-C1-12, A-C1-13):
- one item measures one dimension: response y = w_d + noise, noise sd known;
- independent Gaussian posterior per dimension, updated by precision weighting;
- prior: any StudentProfile (normally the C1.2 v1 completion, so anchors and a
  CEEC prior can seed it).

Item-selection strategies (A-C1-14 for the last two):
- "random"          uniform over dimensions with items left (baseline)
- "variance"        largest posterior variance (uncertainty sampling)
- "discriminative"  posterior variance x spread of that dimension among the
                    programs currently ranked near the top: uncertainty that
                    can actually reorder the recommendation list
- "voi"             myopic value of information: simulate possible answers and
                    pick the item expected to change the current top-k most
                    (Chajewska, Koller & Parr 2000)

The scorer is injected (`score_fn(w, X) -> scores`) so that C1 never imports
C2; the pipeline passes C2's relevance.

Item noise (C1.3 v2):
- "fixed"      v1 reference: item noise sd is assumed (default 0.25). Overconfident
               when the real noise is larger (B-C1-5).
- "estimated"  repeated items on the same dimension differ only by noise, so the
               pooled within-dimension sum of squares estimates the noise variance,
               shrunk toward the assumed value by an inverse-gamma prior worth
               `noise_prior_df` pseudo degrees of freedom:
                   sigma2_hat = (df0 * s0^2 + SS) / (df0 + df)
               After every answer all dimension posteriors are recomputed from the
               prior with sigma_hat (A-C1-15, A-C1-16).
- "censored"   responses are clipped to [0, 1], so answers at 0 or 1 are censored
               observations (Tobit likelihood, B-C1-16). Needs live on a grid over
               [0, 1], sigma on a log grid with the same scaled-inverse-chi^2 prior;
               the joint posterior is computed exactly on the grid and sigma is
               integrated out (A-C1-17). Look-ahead (voi) uses a normal
               approximation of the grid posterior. Sigma is learned from every
               answer's distance to the prior, so prior misfit is read as noise
               (B-C1-18).
- "censored_within"  as "censored", but sigma is learned only from repeated
               answers on the same dimension: each dimension contributes the
               conditional likelihood p(y_2..n | y_1, sigma) under a flat need prior,
               which is exactly uninformative with one answer (A-C1-18). The need
               posterior given sigma still uses the real prior.

Grid models, sigma prior: scaled inverse chi^2 with `noise_prior_df` and scale
`item_noise_sd` ("scale", run 1) or with root-mean-square `item_noise_sd` ("rms").
With "scale" and small df the prior's RMS is well above item_noise_sd
(df=4: 1.41x; df=2: unbounded), so early posteriors are wider than the v1
reference and lean harder on the prior (B-C1-18).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from aisc.contracts import Program, StudentProfile, require_same_dims
from aisc.dimensions import NEED_KEYS

STRATEGIES: tuple[str, ...] = ("random", "variance", "discriminative", "voi")
NOISE_MODELS: tuple[str, ...] = ("fixed", "estimated", "censored", "censored_within")
GRID_MODELS: tuple[str, ...] = ("censored", "censored_within")
_W_GRID = np.linspace(0.0, 1.0, 201)
_S_GRID = np.exp(np.linspace(np.log(0.03), np.log(1.0), 48))


def _log_norm_cdf(z: np.ndarray) -> np.ndarray:
    return np.log(np.vectorize(lambda t: 0.5 * math.erfc(-t / math.sqrt(2.0)))(z))


# Censored-observation log likelihoods on the (sigma, w) grid; fixed, so precomputed.
_LL_AT_0 = _log_norm_cdf(-_W_GRID[None, :] / _S_GRID[:, None])
_LL_AT_1 = _log_norm_cdf(-(1.0 - _W_GRID[None, :]) / _S_GRID[:, None])


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


ScoreFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def cosine_scores(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Default scorer: cosine of w with each row of X (matches C2.1 v0)."""
    norms = np.linalg.norm(X, axis=1) * np.linalg.norm(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = X @ w / norms
    return np.nan_to_num(s)


@dataclass(frozen=True)
class Item:
    item_id: str
    dimension: str


def default_item_bank(items_per_dimension: int = 3) -> list[Item]:
    """Abstract bank: `items_per_dimension` items per dimension (B-C1-15).

    Real item wording (scenario items, docs/rubric.md §3) replaces these in W-41.
    """
    if items_per_dimension < 1:
        raise ValueError("items_per_dimension must be >= 1")
    return [Item(f"{d}#{i}", d) for d in NEED_KEYS for i in range(items_per_dimension)]


@dataclass
class AdaptiveSession:
    prior: StudentProfile
    catalog: list[Program]
    strategy: str = "voi"
    item_noise_sd: float = 0.25
    score_fn: ScoreFn = cosine_scores
    bank: list[Item] = field(default_factory=default_item_bank)
    top_k: int = 10
    top_m: int = 20
    voi_samples: int = 8
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    prior_sd: float = 0.3
    noise_model: str = "fixed"
    noise_prior_df: float = 4.0
    noise_prior_center: str = "scale"

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if self.item_noise_sd <= 0:
            raise ValueError("item_noise_sd must be positive")
        if self.noise_model not in NOISE_MODELS:
            raise ValueError(f"noise_model must be one of {NOISE_MODELS}")
        if self.noise_prior_df <= 0:
            raise ValueError("noise_prior_df must be positive")
        if self.noise_prior_center not in ("scale", "rms"):
            raise ValueError("noise_prior_center must be 'scale' or 'rms'")
        if self.noise_prior_center == "rms" and self.noise_prior_df <= 2:
            raise ValueError("noise_prior_center='rms' needs noise_prior_df > 2")
        if self.prior.dims != "v1":
            raise NotImplementedError(
                "the adaptive questionnaire supports v1 profiles only; the v2 two-stage "
                "questionnaire is W-66"
            )
        if not self.catalog:
            raise ValueError("the catalogue is empty")  # B-C1-30
        require_same_dims(self.prior, self.catalog)
        missing = set(NEED_KEYS) - set(self.prior.needs)
        if missing:
            raise ValueError(
                f"prior must cover every dimension (complete it first); missing {sorted(missing)}"
            )
        self.mean = np.array([self.prior.needs[k] for k in NEED_KEYS], dtype=float)
        self.var = np.array(
            [self.prior.uncertainty.get(k, self.prior_sd) ** 2 for k in NEED_KEYS], dtype=float
        )
        self.X = np.array([p.vector for p in self.catalog])
        self.remaining: dict[str, list[Item]] = {k: [] for k in NEED_KEYS}
        for item in self.bank:
            self.remaining[item.dimension].append(item)
        self.asked: list[tuple[Item, float]] = []
        self._prior_mean = self.mean.copy()
        self._prior_var = self.var.copy()
        self._responses: dict[int, list[float]] = {i: [] for i in range(len(NEED_KEYS))}
        self.noise_sd = self.item_noise_sd  # current estimate (constant when "fixed")
        if self.noise_model in GRID_MODELS:
            self._init_grid()

    # --- censored (grid) model -------------------------------------------------
    def _init_grid(self) -> None:
        w = _W_GRID[None, :]
        self._log_prior_w = -0.5 * (w - self._prior_mean[:, None]) ** 2 / self._prior_var[:, None]
        s2, nu0, s02 = _S_GRID**2, self.noise_prior_df, self.item_noise_sd**2
        if self.noise_prior_center == "rms":
            # E[sigma^2] = nu0 s0^2 / (nu0 - 2): make the prior root-mean-square item_noise_sd
            s02 = s02 * (nu0 - 2) / nu0
        # scaled inverse chi^2 density on sigma^2, times the Jacobians d(sigma^2)/d(log sigma)
        self._log_prior_s = -(nu0 / 2 + 1) * np.log(s2) - nu0 * s02 / (2 * s2) + np.log(s2)
        self._loglik = np.zeros((len(NEED_KEYS), len(_S_GRID), len(_W_GRID)))
        self._first_ll = np.zeros_like(self._loglik)
        self._post_w = None
        self._grid_update()

    @staticmethod
    def _censored_loglik(y: float) -> np.ndarray:
        if y <= 0.0:
            return _LL_AT_0
        if y >= 1.0:
            return _LL_AT_1
        w, sd = _W_GRID[None, :], _S_GRID[:, None]
        return -0.5 * ((y - w) / sd) ** 2 - np.log(sd) - 0.5 * math.log(2 * math.pi)

    def _grid_update(self) -> None:
        logsumexp = _logsumexp
        lj = self._log_prior_w[:, None, :] + self._loglik  # (dims, sigma, w)
        lm = logsumexp(lj, axis=2)  # (dims, sigma): per-dimension marginal likelihood
        if self.noise_model == "censored":
            log_ps = self._log_prior_s + lm.sum(axis=0)
        else:  # censored_within: conditional likelihood of repeats, flat need prior
            rep = [i for i, r in self._responses.items() if len(r) > 1]
            log_ps = self._log_prior_s.copy()
            if rep:
                full = logsumexp(self._loglik[rep], axis=2)
                first = logsumexp(self._first_ll[rep], axis=2)
                log_ps = log_ps + (full - first).sum(axis=0)
        ps = np.exp(log_ps - logsumexp(log_ps, axis=0))  # posterior over sigma
        cond = np.exp(lj - lm[:, :, None])  # p(w | sigma, y) per dimension
        self._post_w = np.einsum("s,dsw->dw", ps, cond)
        self.mean = self._post_w @ _W_GRID
        self.var = np.maximum(self._post_w @ _W_GRID**2 - self.mean**2, 1e-12)
        self.noise_sd = float(np.sqrt(ps @ _S_GRID**2))
        self._post_s = ps

    def interval(self, level: float = 0.90, kind: str = "hpd") -> tuple[np.ndarray, np.ndarray]:
        """Posterior interval per dimension.

        Gaussian models: mean +/- z * sd (`kind` is ignored). Censored model:
        "hpd" (default) = highest-density interval on the grid, which can include
        the scale end when the posterior piles up there; "equal" = equal-tailed,
        which by construction excludes a true value lying exactly on 0 or 1.
        """
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1)")
        if kind not in ("hpd", "equal"):
            raise ValueError("kind must be 'hpd' or 'equal'")
        if self.noise_model not in GRID_MODELS:
            z = NormalDist().inv_cdf(0.5 + level / 2)
            sd = np.sqrt(self.var)
            return self.mean - z * sd, self.mean + z * sd
        if kind == "hpd":
            lo, hi = np.empty(len(NEED_KEYS)), np.empty(len(NEED_KEYS))
            for i, pw in enumerate(self._post_w):
                order = np.argsort(-pw)
                n = int(np.searchsorted(np.cumsum(pw[order]), level * pw.sum())) + 1
                sel = _W_GRID[order[:n]]
                lo[i], hi[i] = sel.min(), sel.max()
            return lo, hi
        cdf = np.cumsum(self._post_w, axis=1)
        a = (1 - level) / 2
        lo = np.array([np.interp(a, c, _W_GRID) for c in cdf])
        hi = np.array([np.interp(1 - a, c, _W_GRID) for c in cdf])
        return lo, hi

    def noise_estimate(self) -> tuple[float, float]:
        """(sigma_hat, within-dimension degrees of freedom used)."""
        ss = sum(float(np.sum((np.array(r) - np.mean(r)) ** 2))
                 for r in self._responses.values() if len(r) > 1)  # fmt: skip
        df = sum(len(r) - 1 for r in self._responses.values() if len(r) > 1)
        s2 = (self.noise_prior_df * self.item_noise_sd**2 + ss) / (self.noise_prior_df + df)
        return float(np.sqrt(s2)), float(df)

    # --- state ---------------------------------------------------------------------

    def _available(self) -> list[int]:
        return [i for i, k in enumerate(NEED_KEYS) if self.remaining[k]]

    def _top(self, w: np.ndarray, k: int) -> np.ndarray:
        return np.argsort(-self.score_fn(w, self.X), kind="stable")[:k]

    def done(self) -> bool:
        return not self._available()

    # --- selection -----------------------------------------------------------------

    def next_item(self) -> Item | None:
        avail = self._available()
        if not avail:
            return None
        if self.strategy == "random":
            d = int(self.rng.choice(avail))
        elif self.strategy == "variance":
            d = max(avail, key=lambda i: (self.var[i], -i))
        elif self.strategy == "discriminative":
            spread = self.X[self._top(self.mean, self.top_m)].var(axis=0)
            d = max(avail, key=lambda i: (self.var[i] * spread[i], self.var[i], -i))
        else:
            d = self._voi_choice(avail)
        return self.remaining[NEED_KEYS[d]][0]

    def _voi_choice(self, avail: list[int]) -> int:
        base = set(self._top(self.mean, self.top_k).tolist())
        z = self.rng.standard_normal(self.voi_samples)  # common random numbers across items
        best, best_val = avail[0], -1.0
        for d in avail:
            pred_sd = np.sqrt(self.var[d] + self.noise_sd**2)
            change = 0.0
            for zs in z:
                y = self.mean[d] + pred_sd * zs
                w = self.mean.copy()
                w[d] = self._updated_mean(d, y)
                change += 1.0 - len(base & set(self._top(w, self.top_k).tolist())) / self.top_k
            val = change / len(z)
            if val > best_val + 1e-12 or (
                abs(val - best_val) <= 1e-12 and self.var[d] > self.var[best]
            ):
                best, best_val = d, val
        return best

    # --- update --------------------------------------------------------------------

    def _updated_mean(self, d: int, y: float) -> float:
        w_prior, w_obs = 1.0 / self.var[d], 1.0 / self.noise_sd**2
        return (self.mean[d] * w_prior + y * w_obs) / (w_prior + w_obs)

    def record(self, item: Item, response: float) -> None:
        if item not in self.remaining[item.dimension]:
            raise ValueError(f"item {item.item_id} is not available")
        if not 0 <= response <= 1:
            raise ValueError("responses must lie in [0, 1]")
        d = NEED_KEYS.index(item.dimension)
        self._responses[d].append(response)
        if self.noise_model == "fixed":
            # v1 reference path: incremental conjugate update with the assumed noise.
            self.mean[d] = self._updated_mean(d, response)
            self.var[d] = 1.0 / (1.0 / self.var[d] + 1.0 / self.noise_sd**2)
        elif self.noise_model == "estimated":
            self.noise_sd, _ = self.noise_estimate()
            self._recompute()
        else:
            ll = self._censored_loglik(response)
            if len(self._responses[d]) == 1:
                self._first_ll[d] = ll
            self._loglik[d] += ll
            self._grid_update()
        self.remaining[item.dimension].remove(item)
        self.asked.append((item, response))

    def _recompute(self) -> None:
        """All dimension posteriors from the prior and every response, with noise_sd."""
        w_obs = 1.0 / self.noise_sd**2
        for i, resp in self._responses.items():
            w_prior = 1.0 / self._prior_var[i]
            n = len(resp)
            total = w_prior + n * w_obs
            self.mean[i] = (self._prior_mean[i] * w_prior + w_obs * sum(resp)) / total
            self.var[i] = 1.0 / total

    def profile(self) -> StudentProfile:
        return StudentProfile(
            student_id=self.prior.student_id,
            needs={k: float(np.clip(m, 0, 1)) for k, m in zip(NEED_KEYS, self.mean, strict=True)},
            uncertainty={k: float(np.sqrt(v)) for k, v in zip(NEED_KEYS, self.var, strict=True)},
            anchors=list(self.prior.anchors),
            familiar=list(self.prior.familiar),
            hard_constraints=dict(self.prior.hard_constraints),
        )
