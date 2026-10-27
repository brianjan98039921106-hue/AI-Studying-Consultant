# Contributing

## Setup
```bash
pip install -e ".[dev]"
pytest && ruff check . && ruff format --check .
```

## Code
- Python ≥ 3.11, type hints on public functions, docstrings explain *why*.
- `ruff check` and `ruff format` must pass (CI enforces).
- Every algorithm change ships with a test. Optimisation code is tested against brute force on small instances.
- Keep `numpy` as the only runtime dependency until a clear need arises.

## Component workflow
Each change targets one component (C0–C5) and one work item (W-xx in `docs/ROADMAP.md`).

1. **G0** — spec in `docs/components/`, theory source in `docs/THEORY.md`, assumptions in `docs/REGISTER.md`.
2. Implement behind the contracts in `src/aisc/contracts.py`; keep the previous version as a reference.
3. Run the validation ladder (`docs/VALIDATION.md`); record results in the PR template.
4. New weakness found → register a B item → add a probe test (`@pytest.mark.probe`, `xfail(strict=True, raises=AssertionError)`).
5. Weakness fixed → probe passes → remove `xfail` and update the register in the same commit.
6. User-facing change → update `docs/USAGE.md` (section, §1 status table, §8 changelog). Its examples run in CI.

Test files are named per component: `tests/test_c<N>_<name>.py`.

## Single sources of truth
Each kind of status lives in exactly one file; other files link to it instead of copying it.

| Status | File |
|---|---|
| Blind spots, assumptions, system risks | `docs/REGISTER.md` |
| Findings (F-numbers) | `results/FINDINGS.md` |
| Work items | `docs/ROADMAP.md` |
| Component gates | `docs/VALIDATION.md` §5 |
| How to use it (users) | `docs/USAGE.md` |

`tests/test_docs_consistency.py` fails CI if a document cites an undefined ID, a link is broken, or a still-failing probe is marked fixed in the register.

## Commits — Conventional Commits
`feat:` new capability · `fix:` bug · `test:` tests only · `docs:` docs only · `data:` dataset changes · `refactor:` · `chore:`

Scope = component: `feat(c2): intensity-aware relevance (W-26, fixes B-C2-3)`

## Branches
- `main` is always green.
- Work on `w<NN>-<topic>` branches (e.g. `w26-intensity-relevance`); merge via pull request once CI passes.

## Data policy
- `data/synthetic/` — fictional programs, `"synthetic": true`. Safe to commit.
- `data/reference/` — public reference tables (CIP, IPEDS). Commit with source URL and download date in `data/README.md`.
- `data/programs/` — real program attributes. Every record needs `source_url` and `retrieved_at`.
- `data/private/` — **never committed** (gitignored). Any student profile, survey answer, or consultant case lives here.
- No scraping of sites whose terms of service forbid it.

## Personal data (個資法)
Student data is personal data under Taiwan's Personal Data Protection Act. Collect only with written consent, store outside the repo, and publish only aggregated, de-identified results.
