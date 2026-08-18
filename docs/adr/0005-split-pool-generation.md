# ADR 0005 — Split pool_generation.py into pos/pool/ modules via mixins

- **Status**: Accepted
- **Date**: 2026-08-17

## Context

`pool_generation.py` was 657 LOC — the largest file in the repo, violating
both the ≤300 and <150 LOC caps. It contained a single `poolGeneration` class
with: config `__init__`, bag generation, data splitting, complexity voting,
GA fitness evaluation, crossover/mutation/verification, stop criteria, bag
saving, pool building, and the `generate` entry point. The GA methods
shared heavy state via `self.*` attributes.

## Decision

Split into focused modules under `pos/pool/`, using **mixins** to preserve
the `self.*` access pattern of the original class (avoiding a risky
context-object refactor):

| Module                           | Responsibility                          | LOC |
|----------------------------------|-----------------------------------------|-----|
| `pos/pool/bag_generator.py`      | `generate_bags`, `build_bags`           | 32  |
| `pos/pool/data_splitter.py`      | `split_data`                            | 27  |
| `pos/pool/complexity_voter.py`   | `vote_complexity`, `get_best_types`     | 68  |
| `pos/pool/genetic_operators.py`  | `GeneticOperatorsMixin` (crossover etc) | 103 |
| `pos/pool/fitness_evaluator.py`  | `FitnessEvaluatorMixin` (get_complexity)| 84  |
| `pos/pool/stop_criteria.py`      | `StopCriteriaMixin` (max_distance etc)  | 94  |
| `pos/pool/pool_builder.py`       | `PoolBuilderMixin` (get_pool)           | 39  |
| `pos/pool/pool_generation.py`    | `poolGeneration` orchestrator           | 132 |

`poolGeneration` now composes four mixins:
```python
class poolGeneration(
    GeneticOperatorsMixin,
    FitnessEvaluatorMixin,
    StopCriteriaMixin,
    PoolBuilderMixin,
):
```

`pool_generation.py` becomes a **facade** (10 LOC) re-exporting
`poolGeneration` so `from pool_generation import poolGeneration` keeps working.

## Behavior preserved

All characterization tests pass:
- `TestPoolGenerationUnit` (5 tests, run without R via rpy2 mock): generate_bags,
  split_data, build_bags, verify_bag.
- `TestPoolGenerationEndToEnd` (2 tests, `@pytest.mark.requires_r`): skipped
  until R is installed, but will exercise the full `generate → get_bags →
  get_pool` flow.

The legacy `biuld_classifier` Perceptron-path bug (ADR 0004) is preserved:
`classifier="perc"` will still raise `ValueError` when `X_test` is an array.

## Why mixins and not free functions

A free-function refactor would require passing ~10 `self.*` attributes as
explicit parameters (or a context dataclass), changing every call site and
risking behavior drift. Mixins preserve the exact `self.*` attribute access
of the original, so the refactor is behavior-preserving by construction.

## Consequences

- **+** Every file <150 LOC; each concern is testable in isolation.
- **+** `pool_generation.py` facade keeps `sample.ipynb` working.
- **−** Mixins add indirection (`poolGeneration` methods spread across 4 files).
- **−** Type checkers can't follow `self.*` across mixins without explicit
  protocol classes (cosmetic, not runtime issue).
