# Notes for AI assistants working in this repo

## What this is
A recommender that profiles Taiwanese grad-school applicants by *needs* (not program names) and surfaces US programs they would not have found, then optimises the application portfolio. Read `docs/ARCHITECTURE.md` first, then the spec for the component you are touching in `docs/components/`.

## Commands
- Test: `pytest -rx` (probes show as xfail)
- Lint / format: `ruff check . && ruff format .`
- Demo: `python examples/quickstart.py`

## Component rules
- Components (C0–C6) exchange only the types in `src/aisc/contracts.py`. Do not import another component's internals.
- Before writing code for a new method: add it to `docs/THEORY.md` with its source and the assumptions it imposes, and register those in `docs/REGISTER.md` (Gate G0).
- New blind spot found → register a B item → write a probe test (`@pytest.mark.probe`, `xfail(strict=True, raises=AssertionError)`) → only then fix it. `raises=AssertionError` is mandatory: without it a probe broken by e.g. a renamed key would still "xfail" and hide the breakage.
- Fixing a blind spot = remove the `xfail`, update the register status and `docs/VALIDATION.md` §5 in the same commit.
- Keep reference implementations (e.g. C4.1 exact solvers, C2 v0 scorers) when adding v1; ablations compare against them.
- Never report a novelty gain without the matching relevance number.
- D9: abilities (`StudentRecord`, requirement gaps) never filter or rerank recommendations; C2 must not read them.
- Status lives in one file per kind (see CONTRIBUTING.md "Single sources of truth"); new findings get a row in `results/FINDINGS.md`. Verify every number written into a findings file against the raw results before committing.
- Simulation (L3) conclusions must hold under both truth functions and both generator families (`needs_model`).
- Tuning parameters in simulation: write the selection rule into the script first (pre-registration), select on one generator, validate on the other, then confirm on fresh seeds. A candidate chosen after seeing held-out results must be re-confirmed on new seeds before use.

## Data rules
- Never commit real student data. `data/private/` is gitignored for that.
- Never present synthetic data as real. Synthetic records carry `"synthetic": true` and fictional school names.
- Real program facts need `source_url` and `retrieved_at` (the loader enforces this). Do not fill attributes from memory.
- Admission probabilities without C3 must be passed as `AdmitEstimate(method="assumed")`.

## Style
- Docs in Traditional Chinese with English technical terms; code and docstrings in English.
- Conventional Commits. Tick finished work items in `docs/ROADMAP.md` in the same commit.
