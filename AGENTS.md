# AGENTS.md — POS (Oracle-N research, starting from Two-level Diversity pool generation)

> Canonical instruction file for OpenCode sessions in this repo. CLAUDE.md points here.

## What this project is

Research project implementing the subproject described in
`docs/SubProjeto - ANÁLISE DO IMPACTO DE ORACLES EM DIFERENTES NÍVEIS NA ACURÁCIA DE SISTEMAS DE MÚLTIPLOS CLASSIFICADORES.md`
(Oracle_N: generalizing the Oracle upper bound to "at least N classifiers correct").
The **starting codebase** is the doctoral thesis "Classifier Pool Generation based on a Two-level
Diversity Approach" (paper: https://doi.org/10.1016/j.inffus.2022.09.001). Improvements may be ported
from the sibling `ComplexityGuidedEnsemble/` repo (see "Sibling repos" below).

All steps must be **scientifically reproducible**: pin seeds, record protocols, version datasets, keep ADRs.

## Repository layout (read this first — it is non-obvious)

The working directory `/home/cbnk/Documents/github/POS/` is **NOT a git repo**. It contains two
independent git repositories side by side:

- `POS/`            — **this repo, the one we modify.** Remote `git@github.com:Brenaki/POS.git`, branch `master`. Contains `Cpx.py`, `pool_generation.py`, `sample.ipynb`, `docs/`.
- `ComplexityGuidedEnsemble/` — **reference only, do NOT push here.** Remote `github.com/matheusMrs07/ComplexityGuidedEnsemble.git`. An attempt to improve the thesis; a source of ideas/code to port, not a target of our edits.

When committing, `cd` into `POS/` first (the outer dir has no git history).

## Hard engineering constraints (enforced on every change)

These are user-mandated rules an agent will violate by default:

1. **SOLID, TDD (Kent Beck), Refactoring (Martin Fowler), DDD (Eric Evans).** Write tests first; refactor in small behavior-preserving steps; model the domain explicitly.
2. **Regression test coverage ≥ 80%** of the complete repo. No tests exist yet — adding the test harness is itself a task.
3. **Every file ≤ 300 lines of code**, except `README.md` and `ROADMAP.md`. `pool_generation.py` is currently **657 lines** and MUST be split before any feature work on it. `Cpx.py` is 262 lines (OK).
4. **Architecture Decision Records (ADRs)** in `docs/adr/` (one file per decision, numbered, `NNNN-title.md`) describing what was implemented and what is to be implemented.
4a. All design/experiment documentation lives under `docs/` (besides README and ROADMAP).
5. **`ROADMAP.md` at repo root** tracks progress against the cronograma in `docs/SubProjeto...md`
   (Set/2026 → Ago/2027: literature review → dataset selection → pool implementation → Oracle
   implementation → semestral report → comparison experiments → analysis → final report). Update
   it every session that advances a milestone.

## Setup gotchas (you will hit these on first run)

- **Python 3.10** is the pinned interpreter (`__pycache__` is `cpython-310`). `requirements.txt` is pinned to old versions (scikit-learn 1.2.0, numpy 1.24.1, pandas 1.5.3, deap 1.3.3, DESlib 0.3.5, mlxtend 0.21.0, rpy2 3.5.7).
- **`requirements.txt` is UTF-16 with BOM.** `pip install -r` works, but reading it as text in Python/scripts needs `encoding='utf-16'`. Do not "fix" it to UTF-8 without noting in an ADR.
- **R + the `ECoL` R package are required.** `Cpx.py` calls R via `rpy2` for data-complexity measures (`overlapping`, `neighborhood`, `linearity`, `dimensionality`, `balance`, `network`). Install R from https://www.r-project.org/ then `R -e 'install.packages("ECoL")'`.
- **`R_HOME` must be set.** `sample.ipynb` hardcodes a Windows path `C:/Program Files/R/R-4.2.1`; on Linux set `R_HOME=/usr/lib/R` (or wherever `R RHOME` prints) before importing `Cpx`/`pool_generation`.
- `DESlib` (DCS/DES methods: OLA, LCA, Rank) is used in `sample.ipynb` and is a direct dependency for the comparison-experiments milestone.

## Architecture

Two modules, both plain script-style (to be refactored to SOLID/DDD):

- `Cpx.py` — data-complexity (via R/ECoL), diversity (`double_fault`), min-max norm, dispersion metrics, and two classifier builders: `biuld_classifier` (Perceptron) and `biuld_classifier_tree` (DecisionTree). Note the misspelling `biuld_*` — keep until a rename ADR.
- `pool_generation.py` — `poolGeneration` class: GA-based bag generation (DEAP, NSGA-II), complexity-guided fitness, crossover/mutation on instance indices, `maxdistance` / `maxacc` stop criteria, `get_bags()` / `get_pool()`. Entry point used by `sample.ipynb`.

When splitting `pool_generation.py` to satisfy the 300-line cap, preserve the public API
(`poolGeneration`, `generate`, `get_bags`, `get_pool`) — `sample.ipynb` and downstream Oracle work depend on it.

## Testing

- No test suite, no `pytest.ini`/`tox.ini`/CI yet. **TDD**: add `tests/` with pytest, target ≥80% coverage (`pytest --cov=. --cov-fail-under=80`).
- `ComplexityGuidedEnsemble/test_*.py` are a style reference for sampler/ensemble unit tests; they are not runnable here (different deps, no R).

## Commands (verified)

```bash
# inside POS/
pip install -r requirements.txt          # note: file is UTF-16
R -e 'install.packages("ECoL")'           # required by Cpx.py
export R_HOME="$(R RHOME)"                # required before importing Cpx/pool_generation
python -m pytest --cov=. --cov-fail-under=80   # once tests exist
```

No lint/typecheck config exists yet; add one (e.g. ruff + mypy) before non-trivial code, and record the choice in an ADR.

## Workflow for every session

1. Read `ROADMAP.md` to see which cronograma milestone is current.
2. Check `docs/adr/` for prior decisions before changing architecture.
3. Write/extend tests first (TDD); keep coverage ≥80%.
4. Keep every new/edited file ≤300 lines (except README, ROADMAP).
5. Add an ADR for any non-trivial design or methodology change.
6. Update `ROADMAP.md` with the milestone advanced this session.

## Sibling repo: ComplexityGuidedEnsemble (reference, read-only for us)

- Pure-Python sampler (`complexity_sampler.py`) + ensemble (`complexity_guided_ensemble.py`), no R.
- Uses `pyhard` for complexity instead of R/ECoL — a candidate simplification to port (decide via ADR).
- Has demo/test scripts (`demo_*.py`, `test_*.py`). Do not edit or push to its remote.