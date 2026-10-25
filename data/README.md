# Data

| Folder | Content | Committed? |
|---|---|---|
| `synthetic/` | Fictional programs for tests and demos (`"synthetic": true`) | Yes |
| `reference/` | Public reference tables (NCES CIP 2020, IPEDS Completions extracts) | Yes, with source + date below |
| `programs/` | Real program attributes; each record has `source_url`, `retrieved_at` | Yes |
| `private/` | Student profiles, survey answers, consultant cases | **Never** (gitignored) |

## Synthetic catalogue
`synthetic/programs.jsonl` — 12 fictional programs. School names are placeholders. CIP codes are used only to exercise label-space novelty and must be checked against the official CIP 2020 list before any real use. Attribute scores and `prestige` values are hand-set for testing, not measurements. The simulator (C5) redraws prestige for its own catalogue; the prototype prestige values are used only by unit tests such as the B-C1-2 probe.

## Reference sources (to be filled in M1)
| File | Source | Retrieved |
|---|---|---|
| `reference/cip2020.csv` | NCES CIP 2020 | — |
| `reference/ipeds_completions_masters.csv` | IPEDS Completions, award level 7 | — |
