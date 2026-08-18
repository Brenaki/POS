"""DEAP mu+lambda loop with per-generation callback.

Replaces the broken `algorithms.eaMuPlusLambda(..., generation_function=)`
call in `pool_generation.py` (see ADR 0007). DEAP 1.3.3 does not accept a
`generation_function` kwarg; this module re-implements the same loop and
invokes the callback after each generation's selection step.

Callback contract (matches `poolGeneration.the_function`):
    generation_function(population, gen, fitness)
where `fitness` is the fitness tuple of the first individual in the
post-selection population (NSGA-II first-front representative). This is an
interpretation of the original (now-lost) modified-DEAP contract, since the
upstream DEAP never shipped `generation_function`. See ADR 0008.
"""

from __future__ import annotations

from typing import Callable, Optional

from deap import algorithms, base, tools


def ea_mu_plus_lambda_with_callback(
    population,
    toolbox: base.Toolbox,
    mu: int,
    lambda_: int,
    cxpb: float,
    mutpb: float,
    ngen: int,
    generation_function: Optional[Callable] = None,
    stats=None,
    halloffame=None,
    verbose: bool = __debug__,
):
    """mu+lambda EA with optional per-generation callback.

    Identical to `deap.algorithms.eaMuPlusLambda` except that after each
    generation's selection step (and after the initial evaluation at gen 0),
    `generation_function(population, gen, population[0].fitness.values)` is
    called if `generation_function` is not None.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    if generation_function is not None:
        generation_function(population, 0, population[0].fitness.values)

    for gen in range(1, ngen + 1):
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = toolbox.select(population + offspring, mu)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if generation_function is not None:
            generation_function(population, gen, population[0].fitness.values)

    return population, logbook
