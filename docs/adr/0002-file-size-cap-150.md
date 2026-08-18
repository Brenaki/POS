# ADR 0002 — File size cap tightened from 300 to 150 lines of code

- **Status**: Accepted
- **Date**: 2026-08-17
- **Supersedes**: the ≤300 LOC rule in `AGENTS.md` (item 3)

## Context

`AGENTS.md` mandates every file ≤300 LOC (except `README.md`, `ROADMAP.md`).
The user explicitly requested a tighter cap of **<150 lines of code per file**
for the Oracle_N refactor. This is achievable because `Cpx.py` (262 LOC) and
`pool_generation.py` (657 LOC) are both being split into focused modules during
Fase 2.

Current file sizes (verified):

- `Cpx.py` — 262 LOC (OK for ≤300, **violates** <150)
- `pool_generation.py` — 657 LOC (violates both ≤300 and <150)
- `README.md` — 120 LOC (exempt)
- `AGENTS.md` — 87 LOC (exempt as docs)
- `sample.ipynb` — 83 lines JSON (exempt as notebook)

## Decision

1. **New cap: 150 LOC per `.py` file**, excluding:
   - `README.md`, `ROADMAP.md` (per original rule)
   - `AGENTS.md`, `docs/**` (documentation)
   - `*.ipynb` (notebooks)
   - Facade files re-exporting legacy API (`cpx_legacy.py`,
     `pool_generation_legacy.py`) — these are thin shims, but counted.
2. Blank lines and comments count toward the 150 LOC.
3. `pyproject.toml` is exempt (config, not code).
4. The original ≤300 rule in `AGENTS.md` is **relaxed** to <150 for code files;
   `AGENTS.md` will be updated to reflect this in a follow-up edit.

## Consequences

- **+** Forces small, focused, single-responsibility modules (SOLID-friendly).
- **+** Easier to test in isolation (TDD-friendly).
- **+** Easier to navigate and review.
- **−** More files → more imports → slightly more boilerplate.
- **−** Some natural units (e.g. the GA orchestration in `pool_generation.py`)
  need careful splitting to stay coherent.

## Enforcement

- `ruff` does not enforce LOC caps natively; a custom check
  (`tests/test_file_size.py`) will assert every `.py` under `pos/` is ≤150 LOC.
- The check runs as part of the test suite, so CI catches violations.
