"""C5 — Simulation & evaluation harness (cross-cutting).

Every other component is validated here against *known ground truth*
before any real student sees it (validation level L3 in docs/VALIDATION.md).

Sub-components (docs/components/C5-simulation.md):
    C5.1 synthetic student generator: true needs w*, familiarity set,
         anchor-choice model mixing fit and prestige, biased response model
    C5.2 metric library: recovery error, Recall@k, novel-relevant rate,
         coverage, popularity bias, calibration
    C5.3 ablation runner: swap one component, hold others fixed
    C5.4 blind-spot probes: one scenario per B-item in docs/REGISTER.md
"""
