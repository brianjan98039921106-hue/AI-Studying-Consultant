"""C0.4 v2: fuse the occupational prior with course evidence by hierarchical shrinkage
(THEORY "融合 v2"; A-C0-12, A-C0-13; B-C0-1, B-C0-3, B-C0-9).

    x_pd ~ N(m_cd, tau_d^2)          programme around its CIP's occupational profile
    e_pd ~ N(x_pd, sigma_pd^2)       course-extraction evidence (C0.3)
    posterior mean = (m / tau^2 + e / sigma^2) / (1 / tau^2 + 1 / sigma^2)

Dimensions without evidence keep the prior. tau_d^2 is estimated across programmes by
the method of moments (parametric empirical Bayes). A dimension is flagged for human
review when the standardised disagreement |e - m| / sqrt(tau^2 + sigma^2) exceeds z.
Nothing here reads real course evidence yet (W-21); the C0.2 occupational profile
remains the catalogue's attribute source until then.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FusedProfile:
    mean: dict[str, float]
    sd: dict[str, float]
    flags: tuple[str, ...]  # dimensions whose evidence disagrees with the prior


def _as_map(v: float | Mapping[str, float], keys) -> dict[str, float]:
    if isinstance(v, Mapping):
        return {k: float(v[k]) for k in keys}
    return {k: float(v) for k in keys}


def fuse_profile(
    prior: Mapping[str, float],
    evidence: Mapping[str, float],
    tau: float | Mapping[str, float],
    sigma: float | Mapping[str, float],
    flag_z: float = 2.0,
) -> FusedProfile:
    """Posterior of one programme's attributes. `evidence` may cover only some of the
    prior's dimensions; extra evidence keys are an error. tau = 0 returns the prior;
    sigma = 0 returns the evidence."""
    extra = set(evidence) - set(prior)
    if extra:
        raise ValueError(f"evidence for dimensions without a prior: {sorted(extra)}")
    keys = list(prior)
    t = _as_map(tau, keys)
    s = _as_map(sigma, [k for k in keys if k in evidence])
    if any(v < 0 or math.isnan(v) for v in [*t.values(), *s.values()]):
        raise ValueError("tau and sigma must be non-negative")
    mean, sd, flags = {}, {}, []
    for k in keys:
        m = float(prior[k])
        if k not in evidence:
            mean[k], sd[k] = m, t[k]
            continue
        e, t2, s2 = float(evidence[k]), t[k] ** 2, s[k] ** 2
        if t2 == 0 and s2 == 0:
            raise ValueError(f"{k}: tau and sigma cannot both be 0")
        if s2 == 0:
            mean[k], sd[k] = e, 0.0
        elif t2 == 0:
            mean[k], sd[k] = m, 0.0
        else:
            prec = 1 / t2 + 1 / s2
            mean[k], sd[k] = (m / t2 + e / s2) / prec, math.sqrt(1 / prec)
        if abs(e - m) > flag_z * math.sqrt(t2 + s2):
            flags.append(k)
    return FusedProfile(mean, sd, tuple(flags))


def estimate_tau(
    priors: Sequence[Mapping[str, float]],
    evidences: Sequence[Mapping[str, float]],
    sigma: float | Mapping[str, float],
) -> dict[str, float]:
    """Method-of-moments tau_d over programmes: tau^2 = max(0, mean (e - m)^2 - sigma^2),
    using the programmes that have evidence on d. sigma is one value per dimension
    (or one for all). Dimensions with fewer than 2 programmes are omitted."""
    if len(priors) != len(evidences):
        raise ValueError("priors and evidences must be paired")
    sq: dict[str, list[float]] = {}
    for m, e in zip(priors, evidences, strict=True):
        for k, v in e.items():
            if k not in m:
                raise ValueError(f"evidence for a dimension without a prior: {k}")
            sq.setdefault(k, []).append((float(v) - float(m[k])) ** 2)
    sig = _as_map(sigma, list(sq))
    return {k: math.sqrt(max(0.0, sum(v) / len(v) - sig[k] ** 2)) for k, v in sq.items()
            if len(v) >= 2}  # fmt: skip
