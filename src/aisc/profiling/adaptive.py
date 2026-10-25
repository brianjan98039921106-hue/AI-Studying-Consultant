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
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from aisc.contracts import Program, StudentProfile
from aisc.dimensions import NEED_KEYS

STRATEGIES: tuple[str, ...] = ("random", "variance", "discriminative", "voi")
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

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if self.item_noise_sd <= 0:
            raise ValueError("item_noise_sd must be positive")
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
            pred_sd = np.sqrt(self.var[d] + self.item_noise_sd**2)
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
        w_prior, w_obs = 1.0 / self.var[d], 1.0 / self.item_noise_sd**2
        return (self.mean[d] * w_prior + y * w_obs) / (w_prior + w_obs)

    def record(self, item: Item, response: float) -> None:
        if item not in self.remaining[item.dimension]:
            raise ValueError(f"item {item.item_id} is not available")
        if not 0 <= response <= 1:
            raise ValueError("responses must lie in [0, 1]")
        d = NEED_KEYS.index(item.dimension)
        self.mean[d] = self._updated_mean(d, response)
        self.var[d] = 1.0 / (1.0 / self.var[d] + 1.0 / self.item_noise_sd**2)
        self.remaining[item.dimension].remove(item)
        self.asked.append((item, response))

    def profile(self) -> StudentProfile:
        return StudentProfile(
            student_id=self.prior.student_id,
            needs={k: float(np.clip(m, 0, 1)) for k, m in zip(NEED_KEYS, self.mean, strict=True)},
            uncertainty={k: float(np.sqrt(v)) for k, v in zip(NEED_KEYS, self.var, strict=True)},
            anchors=list(self.prior.anchors),
            familiar=list(self.prior.familiar),
            hard_constraints=dict(self.prior.hard_constraints),
        )
