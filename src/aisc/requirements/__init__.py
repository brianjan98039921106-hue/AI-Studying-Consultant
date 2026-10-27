"""C6 requirement gaps (D9): what a recommended program asks for that the student
has not done yet. Shown next to recommendations; never filters or reranks them."""

from aisc.requirements.gap import compute_gap, compute_gaps, load_taxonomy

__all__ = ["compute_gap", "compute_gaps", "load_taxonomy"]
