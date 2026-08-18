# ADR 0007 — Pre-existing DEAP generation_function incompatibility

- **Status**: Accepted (documenting a pre-existing bug, not a new decision)
- **Date**: 2026-08-17

## Context

The original `pool_generation.py` calls `deap.algorithms.eaMuPlusLambda`
with a `generation_function=self.the_function` keyword argument. The DEAP
version pinned in `requirements.txt` (1.3.3) does NOT accept this parameter:

```python
# DEAP 1.3.3 signature:
def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb,
                   ngen, stats=None, halloffame=None, verbose=__debug__):
    ...
```

This means the `generate()` entry point raises `TypeError` at the GA call,
regardless of the complexity backend. The bug was latent because the only
path that exercised `generate()` end-to-end was `sample.ipynb`, which
requires R + ECoL (until Fase 3) and was never run in CI.

## Decision

1. **Document the bug** with `@pytest.mark.xfail(strict=True)` on the two
   end-to-end characterization tests that call `generate()`.
2. **Do not fix it in Fase 3** — Fase 3 is about removing R, not changing GA
   behavior. A fix would alter the optimization loop and needs its own ADR.
3. The fix is deferred to a future milestone (likely when implementing the
   Oracle_N comparison experiments, which need `generate()` to actually
   produce pools).

## Likely fix (for a future ADR)

Replace `generation_function=self.the_function` with a manual loop that
calls `the_function` after each generation, or upgrade DEAP to a version
that supports the parameter (if any). The `the_function` callback applies
the stop criteria (`maxdistance`/`maxacc`) and persists the best bags —
losing it means `get_bags()` returns an empty list.

## Consequences

- **+** Bug is documented and tracked; tests `xfail` cleanly instead of
  erroring.
- **−** `poolGeneration.generate()` is non-functional until the fix lands.
- **−** `sample.ipynb` cannot run end-to-end (it calls `generate`).
