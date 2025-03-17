
from .base import ObjectiveBase
from pydantic.dataclasses import dataclass
from ..cvxpy_wrapper import CvxpyWrapper

@dataclass
class LinearObjective(ObjectiveBase):
    """
    Linear objective that tries to minimize ( x dot COST ), i.e. we pass a
    'COST' or 'alpha' column to define the per-asset linear penalty or reward.
    """
    coeff_col_name: str = "COST"

    def add_objective(self, wrapper: CvxpyWrapper, instruments, weight=1.0):
        h = instruments[self.coeff_col_name].values
        x = wrapper.get_variable("weights", len(instruments))
        wrapper.add_linear_objective(x, h, weight)
