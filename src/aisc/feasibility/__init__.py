"""C3 — Admission feasibility. Produces `AdmitEstimate` (mean + interval).

Sub-components (docs/components/C3-feasibility.md):
    C3.1 outcome data + selection-bias audit   — planned
    C3.2 hierarchical Bayesian admit model     — planned
    C3.3 calibration + interval coverage       — planned

Until C3 exists, callers must supply AdmitEstimate values explicitly and
label them as assumptions; nothing downstream may hard-code probabilities.
"""
