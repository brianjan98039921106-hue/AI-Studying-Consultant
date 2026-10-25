"""C2 — Discovery.

Sub-components (docs/components/C2-discovery.md):
    C2.1 relevance scorer   (scoring.relevance)        — v0 cosine
    C2.2 novelty scorer     (scoring.novelty)          — v0 CIP label distance
    C2.3 re-ranker          (rerank.recommend, MMR)    — v0 done
    C2.4 explainer          (contributions)            — v0 per-dimension shares
    constraints             (constraints.passes)       — done
"""

from aisc.discovery.constraints import passes_constraints
from aisc.discovery.rerank import recommend
from aisc.discovery.scoring import contributions, label_similarity, novelty, relevance

__all__ = [
    "contributions",
    "label_similarity",
    "novelty",
    "passes_constraints",
    "recommend",
    "relevance",
]
