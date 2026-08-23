"""Genetic operators mixin: crossover, mutation, stratification check.

Extracted from pool_generation.py during Fase 2. Methods access `self`
attributes exactly as the original class did, preserving behavior.
"""

from __future__ import annotations

import collections
import random

import numpy as np

from pos.pool.bag_generator import build_bags
from pos.pool.errors import StratificationError


class GeneticOperatorsMixin:
    """Crossover and mutation on instance-index bags."""

    def short_cross(self, y_data, indx_bag1, indx_bag2):
        beginning = finish = 0
        ind_out1: list = []
        while y_data[beginning] == y_data[finish]:
            beginning = random.randint(0, len(y_data) - 1)
            finish = random.randint(beginning, len(y_data) - 1)
        for i in range(len(y_data)):
            if i <= beginning or i >= finish:
                ind_out1.append(indx_bag1[i])
            else:
                ind_out1.append(indx_bag2[i])
        return ind_out1

    def verify_bag(self, ind_out):
        classes = collections.Counter(self.y_train)
        _, y = build_bags(ind_out, self.X_train, self.y_train)
        counter = collections.Counter(y)
        if len(counter.values()) == len(classes) and min(counter.values()) >= 2:
            return True
        else:
            return False

    def crossover(self, ind1, ind2):
        indx = self.bags["name"].index(ind1[0])
        indx2 = self.bags["name"].index(ind2[0])
        indx_bag1 = self.bags["inst"][indx]
        indx_bag2 = self.bags["inst"][indx2]
        _, y_data = build_bags(indx_bag1, self.X_train, self.y_train)
        cont = 0
        individual = False
        ind_out1 = None
        while individual != True:
            ind_out1 = self.short_cross(y_data, indx_bag1, indx_bag2)
            individual = self.verify_bag(ind_out1)
            cont = cont + 1
            if cont == 30:
                raise StratificationError(
                    "crossover could not produce a stratified bag after 30 "
                    f"attempts (tam_bags={self.tam_bags}); the training split "
                    "is too small or too imbalanced for this protocol")
        ind1[0] = self.name_individual
        ind2[0] = self.name_individual
        self.bags["name"].append(self.name_individual)
        self.bags["inst"].append(ind_out1)
        self.name_individual += 1
        if self.method_disperse == True:
            self.cont_crossover = self.cont_crossover + 1
            if self.cont_crossover == self.nr_individual + 1:
                self.cont_crossover = 1
                self.get_complexity(first_evaluate=False, population=None)
        from deap import creator
        return creator.Individual(ind1), creator.Individual(ind2)

    def mutation(self, ind):
        ind_out: list = []
        indx = self.bags["name"].index(ind[0])
        indx_bag1 = self.bags["inst"][indx]
        _, y_data = build_bags(indx_bag1, self.X_train, self.y_train)
        inst = 0
        inst2 = len(y_data)
        if self.generation == 0 and self.off == []:
            ind2 = random.randint(0, 99)
        else:
            ind2 = random.sample(self.off, 1)
            ind2 = ind2[0]
        indx2 = self.bags["name"].index(ind2)
        indx_bag2 = self.bags["inst"][indx2]
        _, y2_data = build_bags(indx_bag2, self.X_train, self.y_train)
        tries = 0
        while y_data[inst] != y2_data[inst2 - 1]:
            inst = random.randint(0, len(y_data) - 1)
            tries += 1
            if tries == 1000:
                # The donor class is absent from this bag — without a bound
                # this loop never terminates (ADR 0014).
                raise StratificationError(
                    "mutation could not find a same-class instance to swap "
                    "after 1000 attempts; bag lost a class")
        for i in range(len(indx_bag1)):
            if i == inst:
                ind_out.append(indx_bag2[i])
            else:
                ind_out.append(indx_bag1[i])
        self.bags["name"].append(self.name_individual)
        self.bags["inst"].append(ind_out)
        ind[0] = self.name_individual
        self.name_individual += 1
        if self.method_disperse == True:
            self.cont_crossover = self.cont_crossover + 1
            if self.cont_crossover == self.nr_individual + 1:
                self.cont_crossover = 1
                self.get_complexity(first_evaluate=False, population=None)
        from deap import creator
        return (ind,)
