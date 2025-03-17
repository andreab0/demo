import numpy as np
import logging

from .base import ConstraintBase
from ..cvxpy_wrapper import CvxpyWrapper

logger = logging.getLogger(__name__)


class SetUpConstraint(ConstraintBase):
    """
    Basic 'setup' constraint that couples
    - final weights = (initial + trades)
    - weights_pos - weights_neg = weights
    - trades_pos - trades_neg = trades
    so we can track positive/negative parts consistently.
    """

    def add_constraint(
        self,
        wrapper: CvxpyWrapper,
        instruments,
        initial_position,
        reference_value,
        **kwargs
    ):
        """
        Actually adds the constraints that define how the 'weights' and 'trades'
        variables relate to each other and to the initial_position.
        """
        N = len(instruments)
        logger.debug("SetUpConstraint => #instruments=%d", N)

        x = wrapper.get_variable("weights", N)
        x_pos = wrapper.get_variable("weights_pos", N, pos=True)
        x_neg = wrapper.get_variable("weights_neg", N, pos=True)

        y = wrapper.get_variable("trades", N)
        y_pos = wrapper.get_variable("trades_pos", N, pos=True)
        y_neg = wrapper.get_variable("trades_neg", N, pos=True)

        x0 = initial_position.reindex(instruments.index).fillna(0.0).values

        # 1) x0 + y = x
        wrapper.add_linear_constraints(x0 + y, np.ones(N), "==", x, vectorize=True)

        # 2) x_pos - x_neg = x
        wrapper.add_linear_constraints(
            x_pos - x_neg, np.ones(N), "==", x, vectorize=True
        )

        # 3) y_pos - y_neg = y
        wrapper.add_linear_constraints(
            y_pos - y_neg, np.ones(N), "==", y, vectorize=True
        )
