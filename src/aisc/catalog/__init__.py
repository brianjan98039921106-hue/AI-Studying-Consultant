"""C0 — Program knowledge base.

Sub-components (docs/components/C0-catalog.md):
    C0.1 taxonomy backbone (CIP 2020 + IPEDS award level 7)      — planned
    C0.2 occupation bridge (CIP -> SOC -> O*NET interests)        — planned
    C0.3 curriculum extraction (program pages, with provenance)   — planned
    C0.4 fusion & QA (agreement between C0.2 and C0.3)            — fusion.py, awaits C0.3
    loader: JSON Lines I/O and integrity checks                   — done
"""

from aisc.catalog.fusion import FusedProfile, estimate_tau, fuse_profile
from aisc.catalog.loader import load_programs

__all__ = ["FusedProfile", "estimate_tau", "fuse_profile", "load_programs"]
