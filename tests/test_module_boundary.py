"""Scope boundary (ARCHITECTURE §2.1, D13): the core (needs profiling and
recommendation: C0, C1, C2, C6, the pipeline and the report) must not depend on the
future extension (application planning: C3, C4, aisc.application).

Checked in a fresh interpreter so that modules imported by other tests do not hide a
dependency.
"""

import subprocess
import sys
import warnings

CORE = [
    "aisc.contracts",
    "aisc.dimensions",
    "aisc.catalog",
    "aisc.profiling",
    "aisc.discovery",
    "aisc.requirements",
    "aisc.pipeline",
    "aisc.report",
]
EXTENSION = ("aisc.feasibility", "aisc.portfolio", "aisc.application")


def test_core_does_not_load_extension():
    code = (
        "import importlib, sys\n"
        f"for m in {CORE!r}: importlib.import_module(m)\n"
        f"bad = sorted(m for m in sys.modules if m.startswith({EXTENSION!r}))\n"
        "print('\\n'.join(bad))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "", f"core loads the extension: {out}"


def test_old_pipeline_path_still_works_with_warning():
    import aisc.pipeline as pipeline
    from aisc import application

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn = pipeline.plan_applications
    assert fn is application.plan_applications
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
