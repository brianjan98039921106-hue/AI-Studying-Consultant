"""C2.1 calibration: estimate how much each need layer matters to fit (W-76).

docs/layer-weights.md. Fit utility U(s, p) = -sum_l beta_l * D_l(s, p), with D_l the
mean squared difference over the dimensions of layer l (A-C2-11). Students compare two
programmes and pick the better fit; the pairwise conditional logit (Bradley & Terry
1952; McFadden 1974) gives

    P(choose A) = sigmoid( sum_l beta_l * (D_l(B) - D_l(A)) ),   beta_l >= 0

estimated by maximum likelihood (concave in beta; projected Newton on beta >= 0).
Layer shares pi_l = beta_l / sum(beta). Uncertainty: cluster bootstrap over
respondents. Needs (StudentProfile) only, never abilities (D9).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from aisc.contracts import Program, StudentProfile, require_same_dims
from aisc.dimensions import LAYERS, dims_keys, keys_in_layer


def _layer_index(dims: str) -> list[np.ndarray]:
    keys = dims_keys(dims)
    return [np.array([keys.index(k) for k in keys_in_layer(lay, dims)]) for lay in LAYERS]


def layer_differences(w: np.ndarray, X: np.ndarray, dims: str) -> np.ndarray:
    """(n, 4) mean squared difference per layer between needs w and each row of X."""
    X = np.atleast_2d(X)
    sq = (X - w) ** 2
    return np.column_stack([sq[:, idx].mean(axis=1) for idx in _layer_index(dims)])


def pair_features(profile: StudentProfile, a: Program, b: Program) -> np.ndarray:
    """Features of one comparison: D(B) - D(A) per layer (positive favours A)."""
    require_same_dims(profile, [a, b])
    d = layer_differences(profile.vector, np.vstack([a.vector, b.vector]), profile.dims)
    return d[1] - d[0]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _loglik(beta: np.ndarray, F: np.ndarray, y: np.ndarray) -> float:
    eta = F @ beta
    return float(np.sum(y * -np.logaddexp(0.0, -eta) + (1 - y) * -np.logaddexp(0.0, eta)))


def fit_layer_weights(F: np.ndarray, y: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """MLE of beta >= 0 (no intercept: a pair with no differences is a coin flip)."""
    F = np.asarray(F, float)
    y = np.asarray(y, float)
    if F.ndim != 2 or F.shape[0] != y.shape[0]:
        raise ValueError("F must be (n_pairs, n_layers) and y (n_pairs,)")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y must be 0/1 (1 = chose A)")
    k = F.shape[1]
    beta = np.ones(k)
    ll = _loglik(beta, F, y)
    for _ in range(max_iter):
        p = _sigmoid(F @ beta)
        g = F.T @ (y - p)
        H = -(F * (p * (1 - p))[:, None]).T @ F - 1e-9 * np.eye(k)
        free = (beta > 0) | (g > 0)  # active set: bounds with an outward gradient stay fixed
        if not free.any() or np.max(np.abs(g[free])) < 1e-8:
            break
        step = np.zeros(k)
        step[free] = -np.linalg.solve(H[np.ix_(free, free)], g[free])
        t = 1.0
        while True:
            cand = np.maximum(beta + t * step, 0.0)
            new = _loglik(cand, F, y)
            if new >= ll - 1e-12 or t < 1e-8:
                break
            t /= 2
        if abs(new - ll) < 1e-10:
            beta, ll = cand, new
            break
        beta, ll = cand, new
    return beta


@dataclass(frozen=True)
class LayerWeightEstimate:
    shares: dict[str, float]  # pi_l, sums to 1
    beta: dict[str, float]
    ci: dict[str, tuple[float, float]]  # 95% cluster-bootstrap interval of each share
    n_respondents: int
    n_pairs: int


def _shares(beta: np.ndarray) -> np.ndarray:
    s = beta.sum()
    return np.full_like(beta, np.nan) if s <= 0 else beta / s


def estimate_layer_weights(
    F: np.ndarray,
    y: np.ndarray,
    respondents: Sequence[str],
    n_boot: int = 200,
    rng: np.random.Generator | None = None,
) -> LayerWeightEstimate:
    """Layer shares with a 95% bootstrap interval that resamples respondents."""
    F = np.asarray(F, float)
    y = np.asarray(y, float)
    resp = np.asarray(respondents, dtype=object)
    if len(resp) != len(y):
        raise ValueError("one respondent id per pair")
    beta = fit_layer_weights(F, y)
    ids = sorted(set(resp))
    rows = {r: np.flatnonzero(resp == r) for r in ids}
    rng = rng or np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(len(ids), len(ids), replace=True)
        sel = np.concatenate([rows[ids[i]] for i in pick])
        boot.append(_shares(fit_layer_weights(F[sel], y[sel])))
    boot = np.array(boot)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0) if n_boot else (beta * np.nan,) * 2
    share = _shares(beta)
    return LayerWeightEstimate(
        shares=dict(zip(LAYERS, map(float, share), strict=True)),
        beta=dict(zip(LAYERS, map(float, beta), strict=True)),
        ci={lay: (float(a), float(b)) for lay, a, b in zip(LAYERS, lo, hi, strict=True)},
        n_respondents=len(ids),
        n_pairs=len(y),
    )


SIMEX_LAMBDAS = (0.5, 1.0, 1.5, 2.0)


@dataclass(frozen=True)
class SimexEstimate:
    corrected: LayerWeightEstimate  # SIMEX shares and interval (see `interval`)
    naive: LayerWeightEstimate


def _layer_matrix(dims: str) -> np.ndarray:
    """(d, 4) matrix that averages squared differences within each layer."""
    idx = _layer_index(dims)
    M = np.zeros((sum(len(i) for i in idx), len(LAYERS)))
    for j, i in enumerate(idx):
        M[i, j] = 1.0 / len(i)
    return M


def _pair_rows(stated: np.ndarray, X: np.ndarray, pairs: np.ndarray, dims: str) -> np.ndarray:
    """Feature rows D(B) - D(A) for (respondent, a, b) index triples."""
    S = stated[pairs[:, 0]]
    diff = (X[pairs[:, 2]] - S) ** 2 - (X[pairs[:, 1]] - S) ** 2
    return diff @ _layer_matrix(dims)


def _simex_shares(stated, X, pairs, y, sigma, dims, lambdas, n_sim, rng) -> np.ndarray:
    """Quadratic SIMEX extrapolation of the layer shares to lambda = -1."""
    lam_pts = [0.0]
    share_pts = [_shares(fit_layer_weights(_pair_rows(stated, X, pairs, dims), y))]
    for lam in lambdas:
        sims = []
        for _ in range(n_sim):
            noise = np.sqrt(lam) * sigma * rng.standard_normal(stated.shape)
            noisy = np.clip(stated + noise, 0, 1)
            sims.append(_shares(fit_layer_weights(_pair_rows(noisy, X, pairs, dims), y)))
        lam_pts.append(lam)
        share_pts.append(np.nanmean(sims, axis=0))
    coef = np.polyfit(np.array(lam_pts), np.array(share_pts), 2)  # (3, n_layers)
    ext = np.maximum(coef[0] - coef[1] + coef[2], 0.0)  # value at lambda = -1
    return ext / ext.sum() if ext.sum() > 0 else np.full(len(LAYERS), np.nan)


INTERVALS = ("shift", "bootstrap")


def estimate_layer_weights_simex(
    stated: np.ndarray,
    X: np.ndarray,
    pairs: np.ndarray,
    y: np.ndarray,
    sigma: float | np.ndarray,
    dims: str,
    lambdas: Sequence[float] = SIMEX_LAMBDAS,
    n_sim: int = 20,
    n_boot: int = 200,
    rng: np.random.Generator | None = None,
    interval: str = "shift",
    n_sim_boot: int = 10,
) -> SimexEstimate:
    """SIMEX-corrected layer shares (Cook & Stefanski 1994; B-C2-12).

    stated: (n_respondents, d) answered needs on [0, 1]; X: (n_programmes, d);
    pairs: (n_pairs, 3) integer rows (respondent, a, b); y: 1 if a was chosen;
    sigma: answer error SD (scalar or per dimension), e.g. from a retest (W-41).
    For each lambda, extra noise with variance lambda * sigma^2 is added to the
    answers (clipped to [0, 1] like real answers), the shares are refitted n_sim
    times and averaged; a quadratic in lambda through (0, naive) and these points
    is extrapolated to lambda = -1 (no error). Negative shares are set to 0 and the
    rest renormalised.

    interval:
      "shift"      naive cluster-bootstrap interval shifted by the correction
                   (F88: undercovers in one of six recovery settings)
      "bootstrap"  respondent bootstrap around the WHOLE SIMEX procedure
                   (n_sim_boot simulations per lambda inside each resample)
    """
    stated = np.asarray(stated, float)
    X = np.asarray(X, float)
    pairs = np.asarray(pairs, int)
    y = np.asarray(y, float)
    if pairs.ndim != 2 or pairs.shape[1] != 3 or len(pairs) != len(y):
        raise ValueError("pairs must be (n_pairs, 3) rows of (respondent, a, b)")
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be non-negative")
    if interval not in INTERVALS:
        raise ValueError(f"interval must be one of {INTERVALS}")
    rng = rng or np.random.default_rng(0)
    F0 = _pair_rows(stated, X, pairs, dims)
    resp = [str(r) for r in pairs[:, 0]]
    naive = estimate_layer_weights(
        F0, y, resp, n_boot=n_boot if interval == "shift" else 0, rng=rng
    )
    ext = _simex_shares(stated, X, pairs, y, sigma, dims, lambdas, n_sim, rng)
    if interval == "shift":
        shift = ext - np.array([naive.shares[lay] for lay in LAYERS])
        lo = np.array([naive.ci[lay][0] for lay in LAYERS]) + shift
        hi = np.array([naive.ci[lay][1] for lay in LAYERS]) + shift
    else:
        ids = np.unique(pairs[:, 0])
        rows = {r: np.flatnonzero(pairs[:, 0] == r) for r in ids}
        boot = []
        for _ in range(n_boot):
            draw = rng.choice(ids, len(ids), replace=True)
            sel = np.concatenate([rows[r] for r in draw])
            new_resp = np.repeat(np.arange(len(draw)), [len(rows[r]) for r in draw])
            pb = np.column_stack([new_resp, pairs[sel, 1], pairs[sel, 2]])
            boot.append(_simex_shares(stated[draw], X, pb, y[sel], sigma, dims, lambdas,
                                      n_sim_boot, rng))  # fmt: skip
        lo, hi = np.nanpercentile(np.array(boot), [2.5, 97.5], axis=0)
    corrected = LayerWeightEstimate(
        shares=dict(zip(LAYERS, map(float, ext), strict=True)),
        beta=naive.beta,
        ci={lay: (float(a), float(b)) for lay, a, b in zip(LAYERS, lo, hi, strict=True)},
        n_respondents=naive.n_respondents,
        n_pairs=naive.n_pairs,
    )
    return SimexEstimate(corrected=corrected, naive=naive)
