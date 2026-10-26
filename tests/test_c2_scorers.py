"""C2.1 scorers (W-26): verification (L2)."""

import numpy as np
import pytest

from aisc.dimensions import NEED_KEYS
from aisc.discovery import COSINE, SQDIST, blend, layered, relevance, relevance_matrix

CANDIDATES = [COSINE, layered(0.5), layered(1.0), layered(2.0), blend(0.25), blend(0.5), SQDIST]
IDS = [s.name for s in CANDIDATES]


@pytest.mark.parametrize("scorer", CANDIDATES, ids=IDS)
@pytest.mark.parametrize("seed", range(10))
def test_explanation_is_exact_and_matrix_matches_scalar(scorer, seed):
    """B-C2-6: contributions must sum to the score for every scorer."""
    rng = np.random.default_rng(seed)
    w, X = rng.random(len(NEED_KEYS)), rng.random((6, len(NEED_KEYS)))
    m = scorer.score_matrix(w, X)
    for i, x in enumerate(X):
        assert scorer.score(w, x) == pytest.approx(m[i])
        assert sum(scorer.explain(w, x).values()) == pytest.approx(m[i])


def test_cosine_scorer_equals_v0_relevance():
    rng = np.random.default_rng(0)
    w, X = rng.random(len(NEED_KEYS)), rng.random((8, len(NEED_KEYS)))
    assert COSINE.score_matrix(w, X) == pytest.approx(relevance_matrix(w, X))
    assert COSINE.score(w, X[0]) == pytest.approx(relevance(w, X[0]))
    assert COSINE.score(np.zeros(len(NEED_KEYS)), X[0]) == 0.0


def test_sqdist_ranks_like_distance_truth():
    """Documents the circularity (B-C5-1): sqdist and the distance truth agree on order."""
    from aisc.simulation.truth import true_utility

    rng = np.random.default_rng(1)
    w, X = rng.random(len(NEED_KEYS)), rng.random((30, len(NEED_KEYS)))
    s = SQDIST.score_matrix(w, X)
    t = np.array([true_utility(w, x, "distance") for x in X])
    assert list(np.argsort(-s)) == list(np.argsort(-t))


def test_layered_penalises_only_the_intensity_layer():
    w = np.full(len(NEED_KEYS), 0.5)
    x_domain = w.copy()
    x_domain[NEED_KEYS.index("domain_finance")] = 0.9
    x_quant = w.copy()
    x_quant[NEED_KEYS.index("quant_intensity")] = 0.9
    s = layered(1.0)
    # domain change affects only the cosine term; intensity change adds a penalty too
    assert s.score(w, x_domain) == pytest.approx(COSINE.score(w, x_domain))
    assert s.score(w, x_quant) == pytest.approx(COSINE.score(w, x_quant) - 0.16 / 2)
    expl = s.explain(w, x_quant)
    assert expl["quant_intensity"] < COSINE.explain(w, x_quant)["quant_intensity"]


def _matched_beats_other(scorer, w, X) -> bool:
    s = scorer.score_matrix(w, X)
    return bool(s[0] > s[1])


@pytest.mark.parametrize("scorer", CANDIDATES[1:], ids=IDS[1:])
def test_every_v1_candidate_passes_scale_free_b_c2_3(scorer):
    """Every non-cosine candidate fixes the scale-free probe cases."""
    q = NEED_KEYS.index("quant_intensity")
    for s_lvl, o_lvl in [(0.2, 1.0), (1.0, 0.2), (0.5, 1.0)]:
        w, a, b = (np.zeros(len(NEED_KEYS)) for _ in range(3))
        w[q], a[q], b[q] = s_lvl, s_lvl, o_lvl
        assert _matched_beats_other(scorer, w, np.vstack([a, b]))
    w = np.full(len(NEED_KEYS), 0.3)
    assert _matched_beats_other(scorer, w, np.vstack([w, np.full(len(NEED_KEYS), 0.9)]))


@pytest.mark.parametrize("bad", [lambda: layered(-1), lambda: blend(1.5), lambda: blend(-0.1)])
def test_invalid_parameters_rejected(bad):
    with pytest.raises(ValueError):
        bad()
