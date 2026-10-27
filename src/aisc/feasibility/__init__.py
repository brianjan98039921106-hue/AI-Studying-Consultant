"""C3 — Admission feasibility. Produces `AdmitEstimate` (mean + interval).

Sub-components (docs/components/C3-feasibility.md):
    C3.1 outcome data + selection-bias audit   — planned
    C3.2 hierarchical admit model (v0: MAP + Laplace, self-report shift) — hierarchical.py
    C3.3 calibration + interval coverage       — planned

Until C3 exists, callers must supply AdmitEstimate values explicitly and
label them as assumptions; nothing downstream may hard-code probabilities.
"""

from aisc.feasibility.hierarchical import (
    AdmissionModel,
    AdmissionRecord,
    fit_admission_model,
)

__all__ = ["AdmissionModel", "AdmissionRecord", "fit_admission_model"]
