"""poolGeneration orchestrator (Fase 2 refactor).

Composes GeneticOperatorsMixin, FitnessEvaluatorMixin, StopCriteriaMixin,
and PoolBuilderMixin. The __init__, generate_bags, split_data, generate,
sequencia methods live here. All GA-operation methods come from the mixins.
"""

from __future__ import annotations

from deap import base, creator, tools

from pos.pool.bag_generator import generate_bags as _generate_bags, build_bags as _build_bags
from pos.pool.data_splitter import split_data as _split_data
from pos.pool.complexity_voter import get_best_types as _get_best_types
from pos.pool.genetic_operators import GeneticOperatorsMixin
from pos.pool.fitness_evaluator import FitnessEvaluatorMixin
from pos.pool.stop_criteria import StopCriteriaMixin
from pos.pool.pool_builder import PoolBuilderMixin
from pos.pool.ea_loop import ea_mu_plus_lambda_with_callback


class poolGeneration(
    GeneticOperatorsMixin,
    FitnessEvaluatorMixin,
    StopCriteriaMixin,
    PoolBuilderMixin,
):
    """GA-based classifier pool generation (legacy API preserved)."""

    def __init__(
        self,
        method_disperse=True,
        fit_value=[1.0, 1.0, -1.0],
        nr_generation=20,
        nr_individual=100,
        nr_pop=100,
        proba_crossover=0.9,
        proba_mutation=0.1,
        nr_child=100,
        iteration=20,
        stop_criteria="maxdistance",
        classifier="tree",
        tam_bags=0.5,
        nr_bags=100,
        group=["overlapping", "neighborhood"],
        types=None,
        jobs=8,
    ):
        self.method_disperse = method_disperse
        self.fit_value1 = fit_value[0]
        self.fit_value2 = fit_value[1]
        self.fit_value3 = fit_value[2]
        self.nr_generation = nr_generation
        self.nr_individual = nr_individual
        self.nr_pop = nr_pop
        self.proba_crossover = proba_crossover
        self.proba_mutation = proba_mutation
        self.nr_child = nr_child
        self.cont_crossover = 1
        self.iteration = iteration
        self.dist_temp = 0
        self.jobs = jobs
        self.stop_criteria = stop_criteria
        self.classifier = classifier
        self.save_info = False
        self.seq = -1
        self.base_name = "Base1"
        self.tem2 = []
        self.acc_temp = 0
        # Stop-criteria state (must be initialized here so save_bags() does
        # not crash when no generation improves over the initial 0 — a
        # pre-existing latent bug exposed once the DEAP loop actually runs).
        self.pop_temp: list = []
        self.bags_temp: dict = {}
        self.gen_temp: int = 0
        self.tam_bags = tam_bags
        self.nr_bags = nr_bags
        self.file_out = "isto_e_um_teste"
        self.local = "saida"
        self.c = []
        self.bags_saved = []
        self.pool_classificators = []
        self.group = group
        self.types = types

    def generate_bags(self, X_train, y_train):
        return _generate_bags(X_train, y_train, self.nr_bags, self.tam_bags)

    def build_bags(self, indx_bag):
        return _build_bags(indx_bag, self.X_train, self.y_train)

    def sequencia(self):
        self.seq += 1
        return self.seq

    def split_data(self, X_data, y_data):
        self.X = X_data
        self.y = y_data
        result = _split_data(X_data, y_data)
        (self.X_train, self.y_train, self.X_test, self.y_test,
         self.X_vali, self.y_vali, self.id_train, self.id_test,
         self.id_vali) = result
        return result

    def generate(self, X_train, y_train, X_val, y_val, iteration=20):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.iteration = iteration
        if self.types is None:
            self.types = _get_best_types(X_train, y_train, self.tam_bags, self.group)
        for t in range(0, self.iteration):
            print("Interation - ", t)
            self.name_individual = 100
            self.off = []
            self.seq = -1
            self.repetition = t
            self.generation = 0
            self.bags = self.generate_bags(self.X_train, self.y_train)
            creator.create("FitnessMult", base.Fitness,
                           weights=(self.fit_value1, self.fit_value2, self.fit_value3))
            creator.create("Individual", list, fitness=creator.FitnessMult)
            toolbox = base.Toolbox()
            toolbox.register("attr_item", self.sequencia)
            toolbox.register("individual", tools.initRepeat,
                             creator.Individual, toolbox.attr_item, 1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            self.pop = toolbox.population(n=self.nr_pop)
            if self.method_disperse == True:
                self.get_complexity(first_evaluate=True)
            toolbox.register("evaluate", self.evaluate_linear_dispersion)
            toolbox.register("mate", self.crossover)
            toolbox.register("mutate", self.mutation)
            toolbox.register("select", tools.selNSGA2)
            self.pop = ea_mu_plus_lambda_with_callback(
                self.pop, toolbox, self.nr_child, self.nr_individual,
                self.proba_crossover, self.proba_mutation,
                self.nr_generation, generation_function=self.the_function,
            )
