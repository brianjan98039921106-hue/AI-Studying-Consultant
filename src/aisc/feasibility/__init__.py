"""C3 — Admission feasibility. Produces `AdmitEstimate` (mean + interval).

Sub-components (docs/components/C3-feasibility.md):
    C3.1 outcome-data intake (CSV templates, PII checks) — intake.py
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
from aisc.feasibility.intake import (
    IntakeData,
    IntakeReport,
    fit_from_files,
    load_admission_records,
    load_official_rates,
)

__all__ = [
    "AdmissionModel",
    "AdmissionRecord",
    "IntakeData",
    "IntakeReport",
    "brier",
    "brier_skill",
    "by_group",
    "calibration_slope",
    "fit_admission_model",
    "fit_from_files",
    "load_admission_records",
    "load_official_rates",
    "reliability_table",
]
