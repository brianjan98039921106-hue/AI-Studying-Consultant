"""C2 — Discovery.

Sub-components (docs/components/C2-discovery.md):
    C2.1 relevance scorer   (scoring.relevance)        — v0 cosine (reference)
    C2.1 scorers            (scorers.py)               — W-26 candidates + DEFAULT_SCORER
    C2.2 novelty scorer     (scoring.novelty)          — v0 CIP label distance
    C2.3 re-ranker          (rerank.recommend, MMR)    — v0, rejected at L3 (reference)
    C2.3 v1 sectioned       (sectioned.discover)       — v1: fit-first, two sections
    C2.4 explainer          (contributions)            — v0 per-dimension shares
    constraints             (constraints.passes)       — done
"""

from aisc.discovery.constraints import passes_constraints
from aisc.discovery.rerank import recommend
from aisc.discovery.scorers import (
    COSINE,
    DEFAULT_SCORER,
    DEFAULT_SCORER_V2,
    SQDIST,
    DomainGated,
    LayerBalanced,
    Scorer,
    blend,
    default_scorer,
    domain_gated,
    layer_balanced,
    layered,
)
from aisc.discovery.scoring import (
    contributions,
    label_similarity,
    novelty,
    relevance,
    relevance_matrix,
)
from aisc.discovery.sectioned import discover, shortlist

__all__ = [
    "COSINE",
    "DEFAULT_SCORER",
    "DEFAULT_SCORER_V2",
    "LayerBalanced",
    "DomainGated",
    "domain_gated",
    "default_scorer",
    "layer_balanced",
    "SQDIST",
    "Scorer",
    "blend",
    "layered",
    "contributions",
    "discover",
    "label_similarity",
    "novelty",
    "passes_constraints",
    "recommend",
    "relevance",
    "relevance_matrix",
    "shortlist",
]
