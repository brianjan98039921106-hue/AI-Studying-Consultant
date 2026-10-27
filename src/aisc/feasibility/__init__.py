"""C3 — Admission feasibility. Produces `AdmitEstimate` (mean + interval).

Sub-components (docs/components/C3-feasibility.md):
    C3.1 outcome data + selection-bias audit   — planned
    C3.2 hierarchical admit model (v0.2 default; v0, v0.1 kept)  — hierarchical.py
    C3.3 calibration checks on held-out outcomes — calibration.py

C3.2 is validated only on synthetic data (F128-F140). Until real outcomes exist,
probabilities given without a fitted model must be AdmitEstimate(method="assumed");
nothing downstream may hard-code probabilities.
"""

from aisc.feasibility.calibration import (
    brier,
    brier_skill,
    by_group,
    calibration_slope,
    reliability_table,
)
from aisc.feasibility.hierarchical import (
    AdmissionModel,
    AdmissionRecord,
    fit_admission_model,
)

__all__ = [
    "AdmissionModel",
    "AdmissionRecord",
    "brier",
    "brier_skill",
    "by_group",
    "calibration_slope",
    "fit_admission_model",
    "reliability_table",
]
