
import pandas as pd
import numpy as np
import logging
from typing import Optional
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class FactorRiskModel:
    """
    A factor risk model container compatible with both
    the 'StatisticalModelFitter' and any fundamental fitter.

    Attributes
    ----------
    factor_cov : pd.DataFrame
        Factor-factor covariance matrix, shape=(K_factors, K_factors).
    beta : pd.DataFrame
        Factor exposures (or loadings). index=assets, columns=factors.
    idio_vol : pd.Series
        Idiosyncratic volatility per asset. Usually index=assets.
    factor_returns : pd.DataFrame, optional
        Time series of factor returns (e.g. daily). If None, get_factor_returns() returns empty.
    name : Optional[str]
        Optional identifier for the risk model.
    """

    factor_cov: pd.DataFrame
    beta: pd.DataFrame
    idio_vol: pd.Series
    factor_returns: Optional[pd.DataFrame] = None
    name: Optional[str] = None

    def get_name(self)->str:
        return self.name if self.name else "Unnamed Factor Risk Model"

    def get_cov(self) -> pd.DataFrame:
        """Return a copy of the factor covariance matrix."""
        return self.factor_cov.copy()

    def get_beta(self, instruments=None) -> pd.DataFrame:
        """Return a copy of the factor exposures (beta)."""
        if instruments is None:
            return self.beta.copy()
        else:
            return self.beta.reindex(instruments.index)

    def get_idio(self,instruments = None) -> pd.Series:
        """Return a copy of the idiosyncratic vol per asset."""
        if instruments is None:
            return self.idio_vol.copy()
        else:
            return self.idio_vol.reindex(instruments.index)

    def get_factor_returns(self) -> pd.DataFrame:
        """
        Return a copy of the factor returns time series, or
        an empty DataFrame if not provided.
        """
        if self.factor_returns is None:
            return pd.DataFrame()
        return self.factor_returns.copy()

    def decompose_risk(self, portfolio: pd.Series) -> pd.DataFrame:
        """
        Decompose the portfolio's variance into factor contributions + specific risk.

        Parameters
        ----------
        portfolio : pd.Series
            Index=assets, Values=positions (can be weights or absolute holdings).

        Returns
        -------
        pd.DataFrame
            With columns:
              - 'exposure' : factor exposures for the portfolio
              - 'vol_pct' : fraction of total variance from each factor
              - 'vol_contri' : absolute volatility contribution from each factor
              - 'exposure_gmv' : factor exposure scaled by sum of absolute holdings
              - 'vol_contri_gmv' : factor volatility contribution scaled by sum of absolute holdings
        """
        if portfolio.empty:
            logger.debug("decompose_risk: empty portfolio => returning empty DataFrame.")
            return pd.DataFrame()

        cov = self.get_cov()
        beta = self.get_beta()
        idio = self.get_idio()

        # Reindex to portfolio's assets
        beta = beta.reindex(portfolio.index, fill_value=0.0)
        idio = idio.reindex(portfolio.index, fill_value=0.0)

        # Weighted factor exposures => sum of (position_i * beta_i)
        factor_expo = portfolio.dot(beta)

        # Factor variance = factor_expo * cov * factor_expo
        factor_variance = factor_expo.dot(cov).dot(factor_expo)

        # Idiosyncratic variance
        idio_variance = (portfolio**2 * idio**2).sum()
        total_variance = factor_variance + idio_variance

        if total_variance <= 1e-15:
            logger.debug("decompose_risk: total_variance is near zero => empty result.")
            return pd.DataFrame()

        # partial factor contribution
        factor_partial = (cov @ factor_expo.T) * factor_expo
        factor_share = factor_partial / total_variance

        # Create a 'Specific' row
        factor_share["Specific"] = idio_variance / total_variance

        # For each factor f, 'exposure' is factor_expo[f].
        # For 'Specific', we can store sqrt(idio_variance).
        factor_expo["Specific"] = np.sqrt(idio_variance)

        # Summaries
        gmv = float(portfolio.abs().sum())

        data = {
            "exposure": factor_expo,
            "vol_pct": factor_share,
            "vol_contri": factor_share * np.sqrt(total_variance),
        }
        if gmv > 1e-12:
            data["exposure_gmv"] = factor_expo / gmv
            data["vol_contri_gmv"] = (factor_share * np.sqrt(total_variance)) / gmv
        else:
            data["exposure_gmv"] = factor_expo
            data["vol_contri_gmv"] = factor_share

        df_risk = pd.DataFrame(data)
        return df_risk
