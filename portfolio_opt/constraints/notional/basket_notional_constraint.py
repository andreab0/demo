
import logging
import numpy as _np

from ..structured_constaint import StructuredConstraint
from ...cvxpy_wrapper import CvxpyWrapper
from ...settings import BIG_M

logger = logging.getLogger(__name__)

class BasketNotionalConstraint(StructuredConstraint):
    """
    Constrains the net basket notional = sum_i weights_i.
    Typically something like sum_i x_i <= upper or >= lower.

    Usage Example:
    --------------
    >>> from portfolio_opt.constraints.notional.basket_notional_constraint import BasketNotionalConstraint
    >>> constraint = BasketNotionalConstraint(
    ...     name="MaxNetExposure",
    ...     upper=1_000_000.0,
    ...     unit=Unit.DOLLAR
    ... )
    >>> # This ensures total net position won't exceed $1M
    """

    def _get_variables(self, wrapper: CvxpyWrapper, N):
        return wrapper.get_variable("weights", N)


class BasketLongNotionalConstraint(StructuredConstraint):
    """
    Constrains the total LONG notional = sum_i x_pos_i.
    Typically used if you want to ensure a max total long exposure.

    Usage Example:
    --------------
    >>> constraint_long = BasketLongNotionalConstraint(
    ...     name="MaxLongNotional",
    ...     upper=2_000_000.0,
    ...     unit=Unit.DOLLAR
    ... )
    """

    def _get_variables(self, wrapper: CvxpyWrapper, N):
        return wrapper.get_variable("weights_pos", N, pos=True)

    def _populate_aux_constraints(self, wrapper: CvxpyWrapper, N):
        x_pos = wrapper.get_variable("weights_pos", N, pos=True)
        x_neg = wrapper.get_variable("weights_neg", N, pos=True)
        z_pos = wrapper.get_variable("weights_pos_txn_aux", N, boolean=True)
        z_neg = wrapper.get_variable("weights_neg_txn_aux", N, boolean=True)
        z = wrapper.get_variable("weights_txn_aux", N, boolean=True)

        logger.debug("%s => big-M linking for long notional, M=%.1f",
                     self.name or self.__class__.__name__, BIG_M)
        wrapper.add_linear_constraints(x_pos, _np.ones(N), "<=", BIG_M * z_pos)
        wrapper.add_linear_constraints(x_neg, _np.ones(N), "<=", BIG_M * z_neg)
        wrapper.add_linear_constraints(z_pos + z_neg, _np.ones(N), "==", z)


class BasketShortNotionalConstraint(StructuredConstraint):
    """
    Constrains the total SHORT notional = sum_i x_neg_i.

    Usage Example:
    --------------
    >>> constraint_short = BasketShortNotionalConstraint(
    ...     name="MaxShortNotional",
    ...     upper=500_000.0
    ... )
    """

    def _get_variables(self, wrapper: CvxpyWrapper, N):
        return wrapper.get_variable("weights_neg", N, pos=True)

    def _populate_aux_constraints(self, wrapper: CvxpyWrapper, N):
        x_pos = wrapper.get_variable("weights_pos", N, pos=True)
        x_neg = wrapper.get_variable("weights_neg", N, pos=True)
        z_pos = wrapper.get_variable("weights_pos_txn_aux", N, boolean=True)
        z_neg = wrapper.get_variable("weights_neg_txn_aux", N, boolean=True)
        z = wrapper.get_variable("weights_txn_aux", N, boolean=True)

        logger.debug("%s => big-M linking for short notional, M=%.1f",
                     self.name or self.__class__.__name__, BIG_M)
        wrapper.add_linear_constraints(x_pos, _np.ones(N), "<=", BIG_M * z_pos)
        wrapper.add_linear_constraints(x_neg, _np.ones(N), "<=", BIG_M * z_neg)
        wrapper.add_linear_constraints(z_pos + z_neg, _np.ones(N), "==", z)


class BasketGrossNotionalConstraint(StructuredConstraint):
    """
    Constrains the total GROSS notional = sum_i (|x_i|) 
    but in practice we define x_pos + x_neg = sum of absolute positions.
    Typically use if we want to ensure sum of absolute positions won't exceed a certain level.

    Usage Example:
    --------------
    >>> constraint_gross = BasketGrossNotionalConstraint(
    ...     name="MaxGrossExposure",
    ...     upper=3_000_000.0
    ... )
    """

    def _get_variables(self, wrapper: CvxpyWrapper, N):
        return (wrapper.get_variable("weights_pos", N, pos=True)
                + wrapper.get_variable("weights_neg", N, pos=True))

    def _populate_aux_constraints(self, wrapper: CvxpyWrapper, N):
        x_pos = wrapper.get_variable("weights_pos", N, pos=True)
        x_neg = wrapper.get_variable("weights_neg", N, pos=True)

        z_pos = wrapper.get_variable("weights_pos_txn_aux", N, boolean=True)
        z_neg = wrapper.get_variable("weights_neg_txn_aux", N, boolean=True)
        z = wrapper.get_variable("weights_txn_aux", N, boolean=True)

        logger.debug("%s => linking for gross notional, M=%.1f",
                     self.name or self.__class__.__name__, BIG_M)
        wrapper.add_linear_constraints(x_pos, _np.ones(N), "<=", BIG_M * z_pos)
        wrapper.add_linear_constraints(x_neg, _np.ones(N), "<=", BIG_M * z_neg)
        wrapper.add_linear_constraints(z_pos + z_neg, _np.ones(N), "==", z)
