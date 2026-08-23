"""DEAP mu+lambda loop with per-generation callback.

Replaces the broken `algorithms.eaMuPlusLambda(..., generation_function=)`
call in `pool_generation.py` (see ADR 0007). DEAP 1.3.3 does not accept a
`generation_function` kwarg; this module re-implements the same loop and
invokes the callback after each generation's selection step.

Callback contract (matches `poolGeneration.the_function`):
    generation_function(population, gen, fitness_matrix)
where `fitness_matrix` is a Nx3 numpy array of fitness values for ALL
individuals in the post-selection population. This allows computing
the global dispersion (Gdisp) across the entire population, not just
the first individual's fitness (which always gives dispersion=0).
See ADR 0008 and ADR 0013.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from deap import algorithms, base, tools


def _population_fitness_matrix(population) -> np.ndarray:
    """Extract Nx3 fitness matrix from population (all individuals)."""
    return np.array([ind.fitness.values for ind in population])


def ea_mu_plus_lambda_with_callback(
    population,
    toolbox: base.Toolbox,
    mu: int,
    lambda_: int,
    cxpb: float,
    mutpb: float,
    ngen: int,
    generation_function: Callable | None = None,
    stats=None,
    halloffame=None,
    verbose: bool = __debug__,
    n_jobs: int = 1,
):
    """mu+lambda EA with optional per-generation callback.

    Note: n_jobs is accepted for API compatibility but the DEAP evaluate
    function (evaluate_linear_dispersion) is a trivial dict lookup — the
    real parallelism happens in get_complexity via joblib.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses, strict=True):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    if generation_function is not None:
        generation_function(population, 0, _population_fitness_matrix(population))

    for gen in range(1, ngen + 1):
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses, strict=True):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = toolbox.select(population + offspring, mu)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if generation_function is not None:
            generation_function(population, gen, _population_fitness_matrix(population))

    return population, logbook
