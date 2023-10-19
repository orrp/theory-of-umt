import random
from typing import Optional, List, Iterable

import numpy as np
import wandb
from tqdm import tqdm

from src.envs.commonnonsense import CommonNonsenseModel
from src.learners.learner import Translator, Learner


class RandomPermutationTranslator(Translator):
    NUM_SAMPLE_ATTEMPTS = 5

    def __init__(self, seed: int, P_size: int, T_size: int):
        """

        :param seed:
        :param P_size:
        :param T_size:
        :param validation_xs: If sepcified, will map these xs to ys. More efficient than __call__(x) individually.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.sampled_ys = set()
        self.remaining_ys = None  # Will be populated if sampled_ys becomes too full.
        self.P_size = P_size
        self.T_size = T_size

        self.x_to_y = dict()
        self.Mu = None

    def initialize_validation(self, validation_xs: List[int]):
        """
        Maps each x in validation_xs to a fresh y
        :param validation_xs:
        :return:
        """
        assert len(self.x_to_y) == 0
        ys = self.rng.choice(self.P_size, size=len(validation_xs), replace=False)
        self.x_to_y = {x: y for x, y in zip(validation_xs, ys)}
        self.sampled_ys.update(ys)

    def _initialize_Mu(self, Tau):
        """
        Initializes self.Mu for this translator. Overwrites self.x_to_y computed so far!
        :param Tau:
        """
        assert self.Mu is None, "It is unlikely that you want to re-initialize self.Mu."
        # Generate the full permutation
        P_permuted = self.rng.permutation(self.P_size)
        # Store the permutation as a mapping from x to y, and build Mu as the inverse of Tau under the permutation.
        self.Mu = []
        for x, y in enumerate(P_permuted):
            self.x_to_y[x] = y
            if y in Tau:
                self.Mu.append(x)
        self.x_to_y = {
            x: y for x, y in enumerate(P_permuted)
        }

    def get_Mu(self, Tau) -> list:
        if self.Mu is None:
            self._initialize_Mu(Tau)
        return self.Mu

    def _map_x(self, x: int) -> None:
        '''
        Maps x to a fresh y (that does not have a preimage)
        :param x:
        :return:
        '''
        assert x not in self.x_to_y
        if self.remaining_ys is None:  # We are in rejection sampling
            for _ in range(self.NUM_SAMPLE_ATTEMPTS):
                y = self.rng.randint(0, self.P_size)
                if y not in self.sampled_ys:
                    self.sampled_ys.add(y)
                    self.x_to_y[x] = y
                    return
            # Attempts exhausted, so populate remaining_ys (this is slow)
            self.remaining_ys = [y for y in range(self.P_size) if y not in self.sampled_ys]
        # Sample directly from "remaining ys"
        y_idx = self.rng.randint(0, len(self.remaining_ys))
        self.x_to_y[x] = self.remaining_ys.pop(y_idx)

    def __call__(self, x: int) -> int:
        if x not in self.x_to_y:
            self._map_x(x)
        return self.x_to_y[x]


class RandomPermutationLearner(Learner):
    def __init__(self, env: CommonNonsenseModel, num_translators: int):
        self.env = env
        num_translator_seeds = 10 * num_translators
        translator_seeds = random.sample(range(num_translator_seeds), num_translators)
        self.translators = [RandomPermutationTranslator(seed, env.P_size, env.T_size) for seed in
                            tqdm(translator_seeds, desc='Generating translators', leave=True)]
        self.plausible_translators = self.translators.copy()
        # To be computed later, because ground truth is not initialized in self.env yet
        self._initial_validation_errors = None
        self.validation_errors = None

    def initialize_validation_errors(self, validation_samples):
        assert self._initial_validation_errors is None
        assert self.validation_errors is None
        self._initial_validation_errors = {
            id(t): self.env.validation_error(t, validation_samples)
            for t in tqdm(self.plausible_translators, desc='computing validation')
        }
        self.validation_errors = list(self._initial_validation_errors.values())

    def train(self, samples: Iterable[int], no_wandb: bool):
        for x in tqdm(samples, desc="Samples", position=0):
            self.train_iter(x)

    def train_iter(self, sample):
        self.plausible_translators = [t for t in self.plausible_translators if t(sample) in self.env.Rho]
        if self.validation_errors is not None:
            self.validation_errors = [self._initial_validation_errors[id(t)] for t in self.plausible_translators]
