# ADR 0003 — Remove R/ECoL dependency via pyhard

- **Status**: Proposed (to be Accepted after Fase 3 validation)
- **Date**: 2026-08-17
- **Supersedes**: none
- **Related**: `docs/analise-pos-vs-cge.md` (portabilidade candidata #1)

## Context

`Cpx.py` and `pool_generation.py` call R via `rpy2` for 22 data-complexity
measures (6 groups: overlapping, neighborhood, linearity, dimensionality,
balance, network) from the R package **ECoL**. This is the single biggest
infrastructure barrier:

- R must be installed + `ECoL` package + `R_HOME` set.
- `rpy2` must be compiled against that R (build-time dependency on R binary).
- Cross-platform fragility (the `sample.ipynb` hardcodes a Windows `R_HOME`).

The sibling repo **ComplexityGuidedEnsemble (CGE)** is pure-Python and uses
`pyhard` (`pyhard.measures.ClassificationMeasures`) instead of R/ECoL. CGE is
read-only for us; we port the idea, not the code directly.

## Decision

1. **Replace R/ECoL with `pyhard`** as the complexity backend in `pos/`.
2. Create `pos/complexity/pyhard_adapter.py` wrapping
   `pyhard.measures.ClassificationMeasures`.
3. The public API `complexity_data3(X, y, group, types)` signature is preserved;
   internally it dispatches to the pyhard adapter instead of `ecol.*`.
4. **Behavior changes**: pyhard exposes a different (smaller) set of measures
   than ECoL's 22. The mapping is documented in
   `pos/complexity/_ecol_to_pyhard_map.py`. Measures without a pyhard
   equivalent raise `NotImplementedError` (rather than silently returning 0).
5. **Golden values** from the current R/ECoL run (captured in Fase 1) are
   archived in `tests/_ecol_legacy_golden.json` for historical reference and
   regression auditing.
6. The characterization tests for `complexity_data3` switch from asserting
   exact ECoL values to asserting **properties**: shape, dtype, finite values,
   range ∈ [0, 1] for normalized measures, monotonicity where applicable.
7. `rpy2` and R become **optional** (dev-only, for regenerating golden values).
   They are removed from the default install path; `pyproject.toml` keeps them
   under `[project.optional-dependencies] r = ["rpy2==3.5.7"]`.

## Open issues to resolve during Fase 3

- **Version conflict**: `pyhard 2.2.4` requires `numpy~=1.23.0`,
  `scikit-learn~=1.5.0`, `scipy~=1.13.0` — incompatible with the POS pins
  (numpy 1.24.1, sklearn 1.2.0, scipy 1.10.0). Options:
  1. Use a separate `.venv-pyhard` for complexity measurement only.
  2. Downgrade pyhard to a version compatible with the pins.
  3. Upgrade the POS pins (risky — DESlib 0.3.5 and mlxtend 0.21.0 may break).
  - **Preferred**: option 2 or 1; option 3 only if 1 and 2 fail.
- **Measure coverage**: `get_best_types` in `pool_generation.py` votes over 11
  ECoL measures (5 overlapping + 6 neighborhood). pyhard may not expose all.
  Fallback: use the 3 CGE-style pure-Python measures
  (Overlap/Neighborhood/ErrorRate) as the baseline, and treat
  `get_best_types` as a no-op (return fixed types) until pyhard coverage is
  audited.

## Consequences

- **+** Removes the R/ECoL/R_HOME install barrier — biggest UX win.
- **+** Pure-Python env is portable across Linux/macOS/Windows.
- **−** Behavior change: complexity values differ from ECoL's; scientific
  reproducibility of prior thesis results requires keeping R optional.
- **−** `get_best_types` may need rework if pyhard lacks certain measures.
- **−** Adds `pyhard` as a dependency (with its own transitive deps).
