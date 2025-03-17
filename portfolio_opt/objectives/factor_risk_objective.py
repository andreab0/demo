
from .base import ObjectiveBase
import pandas as _pd
import numpy as _np
from pydantic.dataclasses import dataclass
from factor_model.factor_model import FactorRiskModel
from ..cvxpy_wrapper import CvxpyWrapper
from pydantic import ConfigDict
import logging

logger = logging.getLogger(__name__)

class RiskObjectiveType:
    VARIANCE = "VARIANCE"
    STD_DEV = "STD_DEV"

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class FactorRiskObjective(ObjectiveBase):
    """
    Minimizes factor-based and idiosyncratic risk using the FactorRiskModel.
    risk_objective_type can be "VARIANCE" or "STD_DEV".
    """
    model: FactorRiskModel
    risk_objective_type: str = RiskObjectiveType.VARIANCE

    def add_objective(
        self,
        wrapper: CvxpyWrapper,
        instruments: _pd.DataFrame,
        factor_weight=1.0,
        idio_weight=1.0,
        *args, 
        **kwargs
    ):
        """
        Adds a quadratic or norm-based objective depending on risk_objective_type.
        If VARIANCE => x^T Q x for factor, plus sum_squares for idiosyncratic.
        If STD_DEV => we use an L2 norm objective on factor exposures + idios.
        """
        logger.debug("FactorRiskObjective => #instruments=%d, factor_weight=%.3f, idio_weight=%.3f",
                     len(instruments), factor_weight, idio_weight)

        beta = self.model.get_beta(instruments)
        idio = self.model.get_idio(instruments)
        Q = self.model.get_cov()
        Q = (Q + Q.T) / 2.0  # ensure symmetry

        x = wrapper.get_variable("weights", len(instruments))
        factors = beta.columns

        logger.debug("Beta shape=%s, Q shape=%s, #factors=%d", beta.shape, Q.shape, len(factors))

        # Factor exposures => h = Beta * x
        h = wrapper.get_variable("factor_exposures", len(factors))
        wrapper.add_linear_constraints(x, beta, "==", h, vectorize=False)

        # Attempt Cholesky, fallback to Eigen if needed
        try:
            L = _np.linalg.cholesky(Q).T
        except _np.linalg.LinAlgError:
            logger.warning("Cov matrix not PSD => adjusting with EVD fallback")
            S, U = _np.linalg.eigh(Q)
            S[S < 0.0] = 1e-6
            L = _np.diag(_np.sqrt(S)) @ U.T

        # If 'VARIANCE', do a straightforward x^T Q x + sum_squares(x * idio)
        if self.risk_objective_type == RiskObjectiveType.VARIANCE:
            if factor_weight > 0.0:
                wrapper.add_quad_objective(h, Q, weight=factor_weight)
            if idio_weight > 0.0:
                wrapper.add_sum_of_squares_objective(x, idio, weight=idio_weight)

        elif self.risk_objective_type == RiskObjectiveType.STD_DEV:
            # We'll do norm( [ L*(Beta*x), diag(idio)*x ] )
            factor_part = L @ beta.T @ x if factor_weight > 0.0 else None
            idio_part = _np.diag(idio) @ x if idio_weight > 0.0 else None
            parts = []
            if factor_part is not None:
                parts.append(factor_weight * factor_part)
            if idio_part is not None:
                parts.append(idio_weight * idio_part)
            if len(parts) > 0:
                wrapper.add_norm_objective(parts, weight=1.0)

        else:
            logger.error("Unknown risk objective type: %s", self.risk_objective_type)
