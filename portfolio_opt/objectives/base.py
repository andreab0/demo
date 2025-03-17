from abc import ABC, abstractmethod


class ObjectiveBase(ABC):

    @abstractmethod
    def add_objective(self, wrapper, instruments, weight, *args, **kwargs):
        raise NotImplementedError
