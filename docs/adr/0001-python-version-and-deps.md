# ADR 0001 — Python version and pinned dependencies

- **Status**: Accepted
- **Date**: 2026-08-17
- **Supersedes**: none

## Context

`AGENTS.md` states Python 3.10 is the pinned interpreter (historical
`__pycache__/cpython-310`), and `requirements.txt` pins old versions
(scikit-learn 1.2.0, numpy 1.24.1, pandas 1.5.3, deap 1.3.3, DESlib 0.3.5,
mlxtend 0.21.0, rpy2 3.5.7). The host runs Arch Linux with system Python 3.14,
which is incompatible with those pins.

`requirements.txt` is UTF-16 with BOM (`pip install -r` works, but reading it
as text in Python requires `encoding='utf-16'`). Do not "fix" it to UTF-8
without an ADR.

## Decision

1. Use **Python 3.10.21** via `uv python install 3.10` + `uv venv --python 3.10
   .venv`. Activate with `source .venv/bin/activate`.
2. Install deps from `requirements.txt` (UTF-16) with
   `.venv/bin/python -m pip install -r requirements.txt`.
3. Keep `requirements.txt` as UTF-16 with BOM (do not convert).
4. Add `pyproject.toml` as the tooling config (pytest, ruff, mypy, coverage),
   with `requires-python = ">=3.10,<3.11"`.
5. Dev tools (`pytest`, `pytest-cov`, `ruff`, `mypy`) installed via
   `pip install -e ".[dev]"` or directly.

## Consequences

- **+** Reproducible environment matching the historical `cpython-310` artifacts.
- **+** No system Python pollution (3.14 stays clean).
- **−** `rpy2` build requires R installed first (see ADR 0003).
- **−** `pyhard` (for ADR 0003) conflicts with pinned numpy/sklearn/scipy; needs
  a separate venv or a version downgrade. To be resolved in Fase 3.

## How to reproduce

```bash
cd POS/
uv python install 3.10
uv venv --python 3.10 .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt          # UTF-16 file; pip handles it
pip install pytest pytest-cov pytest-mock ruff mypy
```
