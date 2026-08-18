# ADR 0004 — Split Cpx.py into pos/ package modules

- **Status**: Accepted
- **Date**: 2026-08-17

## Context

`Cpx.py` was 262 LOC in a single script-style module mixing: R/ECoL complexity
calls, min-max normalization, dispersion metrics, double-fault diversity,
hard voting, base-learner builders, and CSV bag IO. It violated the <150 LOC
cap (ADR 0002) and was hard to test in isolation because the top-level
`rpy2` import blocked any test without R installed.

## Decision

Split `Cpx.py` into focused modules under a new `pos/` package:

| Module                         | Responsibility                          | LOC |
|--------------------------------|-----------------------------------------|-----|
| `pos/normalization.py`         | `min_max_norm`                          | 39  |
| `pos/dispersion.py`            | `dispersion_linear`, `dispersion`       | 65  |
| `pos/diversity.py`             | `diversitys` (double fault)             | 32  |
| `pos/voting.py`                | `voting_classifier` (mlxtend)           | 25  |
| `pos/classifiers.py`           | `biuld_classifier[_tree]`               | 74  |
| `pos/io_csv.py`                | `save_bag`                              | 53  |
| `pos/complexity/base.py`       | `HEADER`, `GROUP_MEASURES` constants    | 27  |
| `pos/complexity/ecol_adapter.py`| `complexity_data3` (R/ECoL)           | 70  |

`Cpx.py` becomes a **facade** (33 LOC) that re-exports the same public names
(`min_max_norm`, `dispersion_linear`, `dispersion`, `diversitys`,
`voting_classifier`, `biuld_classifier`, `biuld_classifier_tree`, `save_bag`,
`header`, `complexity_data3`, `ecol`) so `sample.ipynb` and any
`from Cpx import ...` keep working unchanged.

## Behavior preserved

All characterization tests in
`tests/characterization/test_cpx_pure_functions.py` (23 tests) pass against
the refactored modules. Known bugs are preserved as documented:
- `min_max_norm` on a 2d array returns a list of 1d arrays (not floats).
- `biuld_classifier` with `X_test` as a numpy array raises `ValueError`
  (`X_test != None` ambiguity). The Perceptron path in `pool_generation` is
  therefore broken when `classifier="perc"`; users must use `classifier="tree"`.

## Consequences

- **+** Each module is independently testable and <150 LOC.
- **+** The rpy2 dependency is isolated to `pos/complexity/ecol_adapter.py`;
  the other modules can be imported without R (via the conftest mock).
- **+** `Cpx.py` facade keeps backwards compatibility.
- **−** One more level of indirection (`Cpx → pos.X`).
