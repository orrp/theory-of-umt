import random
from src.learners.learner import Translator


class CommonNonsenseModel:

    def __init__(self, P_size, T_size, alpha):
        self.P_size, self.T_size = P_size, T_size
        self.P, self.T = set(range(P_size)), set(range(T_size))
        assert self.T.issubset(self.P)

        # Set up S, \rho, \tau
        self.S = set()
        for y in self.P:
            # keep y w.p. 1 - \alpha
            if random.uniform(0, 1) > alpha:
                self.S.add(y)
        self.Rho = self.P.intersection(self.S)  # the support of rho
        self.Tau = self.T.intersection(self.S)  # the support of tau.

        # Set up translator family (random permutations)

        self.ground_truth = None
        self.Mu = None

    def set_ground_truth(self, ground_truth: Translator):
        self.ground_truth = ground_truth
        self.Mu = self.ground_truth.get_Mu(self.Tau)

    def validation_error(self, translator, valid_samples):
        diff_count = 0
        for x in valid_samples:
            if self.ground_truth(x) != translator(x):
                diff_count += 1
        return float(diff_count) / len(valid_samples)
