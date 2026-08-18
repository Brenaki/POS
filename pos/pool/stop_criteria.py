"""Stop criteria and bag persistence mixin.

Extracted from pool_generation.py during Fase 2.
"""

from __future__ import annotations

import numpy as np

from pos.dispersion import dispersion


class StopCriteriaMixin:
    """maxdistance/maxacc stop criteria + bag saving."""

    def max_distance(self, fitness, generation=None, population=None, bags=None):
        if fitness[2]:
            dist_dist_media = np.mean(
                dispersion(np.column_stack([fitness[0], fitness[1], fitness[2]]))
            )
        else:
            dist_dist_media = np.mean(
                dispersion(np.column_stack([fitness[0], fitness[1]]))
            )
        if dist_dist_media > self.dist_temp:
            self.dist_temp = dist_dist_media
            self.pop_temp = population
            self.gen_temp = generation
            self.bags_temp = bags

    def max_acc(self, acc, generation=None, population=None, bags=None):
        if acc > self.acc_temp:
            self.acc_temp = acc
            self.pop_temp = population
            self.gen_temp = generation
            self.bags_temp = bags

    def save_bags(self, pop_temp, bags_temp, gen_temp=None, base_name=None,
                  type=0, generations_escolhida="x"):
        if type == 0:
            for j in pop_temp:
                name = []
                indx = self.bags["name"].index(j)
                nm = self.bags["inst"][indx]
                name.append(self.bags["name"][indx])
                name.extend(nm)
                self.bags_saved.append(name)
        elif type == 1:
            x = open(generations_escolhida, "a")
            x.write(base_name + ";" + str(gen_temp) + "\n")
            x.close()
            for j in pop_temp:
                name = []
                indx = bags_temp["name"].index(str(j))
                nm = bags_temp["inst"][indx]
                name.append(bags_temp["name"][indx])
                name.extend(nm)
        elif type == 2:
            x = open(generations_escolhida, "a")
            x.write(base_name + ";" + str(gen_temp) + "\n")
            x.close()
            for j in pop_temp:
                name = []
                indx = bags_temp["name"].index(str(j))
                nm = bags_temp["inst"][indx]
                name.append(bags_temp["name"][indx])
                name.extend(nm)

    def the_function(self, population, gen, fitness):
        generation = gen
        self.off = []
        bags_ant = self.bags
        bags = {"name": list(), "inst": list()}
        for j in population:
            indx = bags_ant["name"].index(j[0])
            bags["name"].append(bags_ant["name"][indx])
            bags["inst"].append(bags_ant["inst"][indx])
        del bags_ant
        for i in range(len(population)):
            self.off.append(population[i][0])
        if self.stop_criteria == "maxdistance":
            self.max_distance(fitness, generation=self.generation,
                              population=self.off, bags=bags)
        elif self.stop_criteria == "maxacc":
            self.max_acc(self.dist["score_g"], generation=self.generation,
                         population=self.off, bags=bags)
        if generation == self.nr_generation:
            if self.stop_criteria in ("maxdistance", "maxacc") and len(self.pop_temp) > 0:
                self.save_bags(self.pop_temp, self.bags_temp, self.gen_temp, self.base_name)
            else:
                # Fallback: if stop_criteria never improved (e.g. dispersion
                # of a single fitness row is always 0 — a pre-existing
                # conceptual bug in max_distance), save the final population
                # so get_bags()/get_pool() return non-empty results.
                self.save_bags(self.off, bags, base_name=self.base_name)
        if self.method_disperse == True and generation != self.nr_generation:
            self.get_complexity(population=population)
        return population
