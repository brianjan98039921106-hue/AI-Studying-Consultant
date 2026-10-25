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

## Commits — Conventional Commits
`feat:` new capability · `fix:` bug · `test:` tests only · `docs:` docs only · `data:` dataset changes · `refactor:` · `chore:`

Example: `feat(discovery): add embedding-based novelty`

## Branches
- `main` is always green.
- Work on `feat/<topic>` branches; merge via pull request once CI passes.

## Data policy
- `data/synthetic/` — fictional programs, `"synthetic": true`. Safe to commit.
- `data/reference/` — public reference tables (CIP, IPEDS). Commit with source URL and download date in `data/README.md`.
- `data/programs/` — real program attributes. Every record needs `source_url` and `retrieved_at`.
- `data/private/` — **never committed** (gitignored). Any student profile, survey answer, or consultant case lives here.
- No scraping of sites whose terms of service forbid it.

## Personal data (個資法)
Student data is personal data under Taiwan's Personal Data Protection Act. Collect only with written consent, store outside the repo, and publish only aggregated, de-identified results.
