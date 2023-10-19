import itertools
import random
from typing import Callable

import numpy as np


class KnowledgeGraphModel:

    def __init__(self, n_target: int, n_source: int, p: float, alpha: float):
        self.n_target = n_target
        self.n_source = n_source
        self.alpha = alpha
        self.p = p
        self.P = np.random.choice(a=[True, False], size=(n_target, n_target), p=[p, 1 - p])
        self.ground_truth = None
        self.Mu = None
        self.all_Mu_node_pairs = list(itertools.product(range(self.n_source), range(self.n_source)))
        self.edges_in_random_order = None

    def set_ground_truth(self, ground_truth: Callable[[int], int]):
        self.ground_truth = ground_truth
        self.Mu = np.full(shape=(self.n_source, self.n_source), fill_value=False)
        self.edges_in_random_order = []
        for x1, x2 in self.all_Mu_node_pairs:
            if random.random() < self.alpha:
                self.Mu[x1, x2] = self.P[ground_truth(x1), ground_truth(x2)]
            else:
                self.Mu[x1, x2] = random.random() < self.p
            if self.Mu[x1, x2]:
                self.edges_in_random_order.append((x1, x2))
        random.shuffle(self.edges_in_random_order)

    def nodewise_error(self, translator):
        diff_count = 0
        for x in range(self.n_source):
            if self.ground_truth(x) != translator(x):
                diff_count += 1
        return float(diff_count) / self.n_source

    def edgewise_error(self, translator):
        diff_count = 0
        for x1, x2 in self.edges_in_random_order:
            if self.ground_truth(x1) != translator(x1) or self.ground_truth(x2) != translator(x2):
                diff_count += 1
        return float(diff_count) / len(self.edges_in_random_order)
