"""C1 — Student profiling.

Sub-components (docs/components/C1-profiling.md):
    C1.1 measurement model (dimensions, item bank, reliability)   — planned
    C1.2 anchor decomposition (+ prestige de-biasing)             — v0 done (no de-biasing)
    C1.3 adaptive elicitation (value-of-information item choice)  — planned
    C1.4 familiarity elicitation (what the student already knows) — planned
"""

from aisc.profiling.anchors import blend_profile, needs_from_anchors

__all__ = ["blend_profile", "needs_from_anchors"]
