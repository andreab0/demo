
import logging
import numpy as _np

from ..structured_constaint import StructuredConstraint
from ...cvxpy_wrapper import CvxpyWrapper
from ...settings import BIG_M

logger = logging.getLogger(__name__)

class TurnoverConstraint(StructuredConstraint):
    """
    Constrains total turnover = sum of absolute trades (trades_pos + trades_neg).
    
    Typically used to limit how much rebalancing or 'churn' is allowed.
    
    Usage Example:
    --------------
    >>> constraint = TurnoverConstraint(
    ...     name="MaxTurnover",
    ...     upper=100_000.0,
    ...     unit=Unit.DOLLAR  # or PERCENT if you prefer relative turnover
    ... )
    >>> # This ensures sum(|trades|) <= 100k
    """

    def _get_variables(self, wrapper: CvxpyWrapper, N):
        return wrapper.get_variable("trades_pos", N, pos=True) + wrapper.get_variable("trades_neg", N, pos=True)

    def _populate_aux_constraints(self, wrapper: CvxpyWrapper, N):
        x_pos = wrapper.get_variable("trades_pos", N, pos=True)
        x_neg = wrapper.get_variable("trades_neg", N, pos=True)
        z_pos = wrapper.get_variable("trades_pos_txn_aux", N, boolean=True)
        z_neg = wrapper.get_variable("trades_neg_txn_aux", N, boolean=True)
        z = wrapper.get_variable("trades_txn_aux", N, boolean=True)

        logger.debug("%s => adding BigM constraints with M=%.1f",
                     self.name or self.__class__.__name__, BIG_M)
        wrapper.add_linear_constraints(x_pos, _np.ones(N), "<=", BIG_M * z_pos)
        wrapper.add_linear_constraints(x_neg, _np.ones(N), "<=", BIG_M * z_neg)
        wrapper.add_linear_constraints(z_pos + z_neg, _np.ones(N), "==", z)
