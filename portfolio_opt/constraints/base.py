from abc import ABC, abstractmethod

class ConstraintBase(ABC):
    """
    Base class for all constraints.
    """
    @abstractmethod
    def add_constraint(self, wrapper, *args, **kwargs):
        raise NotImplementedError
