
import numpy as _np
from pydantic.dataclasses import dataclass
from typing import Optional
from abc import abstractmethod
import logging

from enum import Enum
from ..cvxpy_wrapper import CvxpyWrapper
from .base import ConstraintBase

logger = logging.getLogger(__name__)

class Scope(Enum):
    AGGREGATE = "AGGREGATE"
    MEMBER = "MEMBER"
    ASSET = "ASSET"

class Unit(Enum):
    DOLLAR = "DOLLAR"
    SHARES = "SHARES"
    PERCENT = "PERCENT"
    NUMBER = "NUMBER"

@dataclass
class StructuredConstraint(ConstraintBase):
    """
    Generic 'structured' constraint that uses 'scope', 'unit', and possibly
    group references.
    """
    name: Optional[str] = None
    unit: Unit = Unit.DOLLAR
    scope: Scope = Scope.AGGREGATE
    upper: Optional[float] = None
    lower: Optional[float] = None
    upper_group_ref: Optional[str] = None
    lower_group_ref: Optional[str] = None
    metagroup_ref: Optional[str] = None
    weight: float = 1.0

    def add_constraint(self, wrapper, instruments, reference_value, *args, **kwargs):
        """
        Main entry point to add constraints to the CVXPY model.
        """
        cname = self.name or self.__class__.__name__
        logger.debug("Adding constraint %s => unit=%s scope=%s upper=%s lower=%s",
                     cname, self.unit, self.scope, self.upper, self.lower)

        self._fetch_bounds(instruments)
        self._normalize_bounds(instruments, reference_value)

        N = len(instruments)
        x = self._get_variables(wrapper, N)

        # Hook for constraints that rely on additional 'aux' variables
        self._populate_aux_constraints(wrapper, N)

        # Finally the main constraints
        self._populate_constraints(wrapper, x, instruments, reference_value)

    @abstractmethod
    def _get_variables(self, wrapper, N):
        """
        Which variable(s) do we actually constrain? e.g. 'weights', 'weights_pos', etc.
        """
        raise NotImplementedError

    def _fetch_bounds(self, instruments):
        # If we have group references, fetch them
        if self.upper_group_ref is not None:
            self.upper_group_ref_values = (
                instruments[self.upper_group_ref].copy() * self.weight
            )
        if self.lower_group_ref is not None:
            self.lower_group_ref_values = (
                instruments[self.lower_group_ref].copy() * self.weight
            )

    def _normalize_bounds(self, instruments, reference_value):
        """
        Convert from PERCENT or SHARES to an effective DOLLAR if needed.
        """
        if self.unit == Unit.PERCENT:
            if self.upper is not None:
                self.upper = self.upper * reference_value / 100.0
            if self.lower is not None:
                self.lower = self.lower * reference_value / 100.0

            if hasattr(self, "upper_group_ref_values"):
                self.upper_group_ref_values *= (reference_value / 100.0)
            if hasattr(self, "lower_group_ref_values"):
                self.lower_group_ref_values *= (reference_value / 100.0)

        elif self.unit == Unit.SHARES:
            # If user sets e.g. upper=100 shares => convert to notional
            # by multiplying by instruments["price"]
            if self.upper is not None:
                # create a dynamic group to represent the share->notional
                self.upper_group_ref_values = self.upper * instruments["price"].copy()
                self.upper = None
                self.upper_group_ref = "share2notional"
            if self.lower is not None:
                self.lower_group_ref_values = self.lower * instruments["price"].copy()
                self.lower = None
                self.lower_group_ref = "share2notional"

            # If user also has references, multiply them by price
            if hasattr(self, "upper_group_ref_values"):
                self.upper_group_ref_values *= instruments["price"].copy()
            if hasattr(self, "lower_group_ref_values"):
                self.lower_group_ref_values *= instruments["price"].copy()

    def _populate_aux_constraints(self, wrapper, N):
        """
        Optional method to define additional constraints or variables (like big-M constraints).
        """
        pass

    def _populate_constraints(self, wrapper: CvxpyWrapper, x, instruments, reference_value):
        """
        Actually add the linear constraints to the wrapper. 
        """
        # By default, interpret numeric 'x' in DOLLAR terms unless unit==NUMBER
        # => i.e. ref_value=1 if it's "NUMBER"
        # For everything else, we assume we want x to be scaled by reference_value if needed
        if self.unit == Unit.NUMBER:
            ref_value = 1.0
        else:
            ref_value = reference_value

        # If we have a 'metagroup_ref', it means we handle constraints group by group
        if self.metagroup_ref is not None:
            for name, group in instruments.groupby(self.metagroup_ref):
                metagroup_mask = instruments.index.isin(group.index)
                metagroup_x = x[metagroup_mask]
                N_mg = sum(metagroup_mask)
                if N_mg == 0:
                    continue

                self._apply_ub_lb(wrapper, metagroup_x, N_mg, ref_value, metagroup_mask)
        else:
            # All instruments in one group
            N_all = len(instruments)
            self._apply_ub_lb(wrapper, x, N_all, ref_value, None)

    def _apply_ub_lb(self, wrapper, x, N, ref_value, mask=None):
        """
        Helper to apply upper/lower constraints in either vectorized or aggregated manner,
        depending on self.scope.
        """
        vectorize = (self.scope in (Scope.MEMBER, Scope.ASSET))

        # If we do not have group references
        if (self.upper_group_ref is None) and (self.lower_group_ref is None):
            # If we have numeric upper/lower
            if self.upper is not None:
                wrapper.add_linear_constraints(x, _np.ones(N)*ref_value, "<=", self.upper, vectorize)
            if self.lower is not None:
                wrapper.add_linear_constraints(x, _np.ones(N)*ref_value, ">=", self.lower, vectorize)

        else:
            # We have group references for upper/lower
            if hasattr(self, "upper_group_ref_values") and self.upper_group_ref_values is not None:
                if mask is not None:
                    wrapper.add_linear_constraints(
                        x,
                        _np.ones(N)*ref_value,
                        "<=",
                        self.upper_group_ref_values[mask],
                        vectorize=True
                    )
                else:
                    wrapper.add_linear_constraints(
                        x,
                        _np.ones(N)*ref_value,
                        "<=",
                        self.upper_group_ref_values,
                        vectorize=True
                    )
            if hasattr(self, "lower_group_ref_values") and self.lower_group_ref_values is not None:
                if mask is not None:
                    wrapper.add_linear_constraints(
                        x,
                        _np.ones(N)*ref_value,
                        "<=",
                        self.lower_group_ref_values[mask],
                        vectorize=True
                    )
                else:
                    wrapper.add_linear_constraints(
                        x,
                        _np.ones(N)*ref_value,
                        "<=",
                        self.lower_group_ref_values,
                        vectorize=True
                    )
