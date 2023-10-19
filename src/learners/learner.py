import abc
import typing


class Translator(abc.ABC):

    @abc.abstractmethod
    def __call__(self, x: int) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_Mu(self, Tau) -> list:
        raise NotImplementedError


class Learner(abc.ABC):
    @abc.abstractmethod
    def train(self, samples: typing.Iterable, no_wandb: bool):
        raise NotImplementedError
