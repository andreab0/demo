import pandas as _pd
import numpy as _np
from .factor_model import FactorRiskModel
import logging

logger = logging.getLogger(__name__)

DAYS_OF_YEAR = 252

class StatisticalModelFitter:
    """
    A very simple statistical factor model fitter using an iterative approach:

    Steps
    -----
    1) Convert returns to shape (n_stocks x n_dates).
    2) Perform an SVD on that matrix to estimate how many factors (<= max_factors)
       are needed to explain 'variance_threshold' fraction of variance.
    3) Iteratively:
       (a) Recompute factor loadings (B) & factor matrix (F) from SVD,
       (b) Compute residual stdev per stock,
       (c) Rescale each stock's returns by its residual stdev,
       until convergence or max_iter.
    4) Final factor loadings => build covariance, idiosyncratic vol,
       and (now added) daily factor returns from projecting returns onto loadings.
    """

    def fit(
        self,
        returns: _pd.DataFrame,
        max_factors: int = 20,
        variance_threshold: float = 0.9,
        max_iter: int = 20,
        exposures: _pd.DataFrame = None,  # Not used, present for signature compatibility
        **kwargs
    ) -> FactorRiskModel:
        """
        Fit this statistical factor model.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns in wide format => index=dates, columns=stocks;
            shape = (N_dates, N_stocks).
        max_factors : int
            Max number of factors to keep (top principal components).
        variance_threshold : float
            Fraction of total variance explained to pick the # of factors
            (clamped by max_factors).
        max_iter : int
            Maximum iterations for the iterative scaling loop.
        exposures : pd.DataFrame, optional
            Not used; included only for a consistent signature with other fitters.
        kwargs : dict
            Additional ignored arguments (for future extension).

        Returns
        -------
        FactorRiskModel
            Contains:
              - factor_cov      => Covariance of factors (annualized)
              - beta            => Factor loadings (stock x factor)
              - idio_vol        => Idiosyncratic vol per stock (annualized)
              - factor_returns  => Time series of daily factor returns (date x factor)
              - name            => "Statistical Risk Model"
        """
        # 1) Convert returns to shape=(n_stocks, n_dates)
        R = returns.T.copy()
        if R.empty:
            logger.warning("StatisticalModelFitter: 'returns' is empty => returning empty FactorRiskModel.")
            return FactorRiskModel(
                factor_cov=_pd.DataFrame(),
                beta=_pd.DataFrame(),
                idio_vol=_pd.Series(dtype=float),
                factor_returns=_pd.DataFrame(),
                name="Statistical Risk Model"
            )

        # Find how many factors we need based on SVD
        _, singular_values, _ = _np.linalg.svd(R)
        total_variance = _np.sum(singular_values**2)
        cumulative_variance = _np.cumsum(singular_values**2) / total_variance
        num_factors = _np.searchsorted(cumulative_variance, variance_threshold) + 1
        num_factors = min(num_factors, max_factors)
        factor_cols = [f"StatFactor_{i+1}" for i in range(num_factors)]

        logger.debug(
            f"StatisticalModelFitter: returns shape=({returns.shape[0]}, {returns.shape[1]}), "
            f"selecting up to {num_factors} factors (variance_threshold={variance_threshold})"
        )

        # 2) Iterative scaling
        prev_S = _np.zeros(R.shape[0])  # shape=(n_stocks,)
        diff = 1e7
        i = 0

        while diff > 1e-4 and i < max_iter:
            # (a) SVD of R
            _, _, V = _np.linalg.svd(R)
            # F => (n_dates, num_factors)
            F = V.T[:, :num_factors]
            # B => (n_stocks, num_factors)
            B = R.dot(F)

            # (b) residual => shape=(n_stocks, n_dates)
            G = R.values - B.dot(F.T)
            # stdev per stock => diag of cov
            S = _np.sqrt(_np.diag(_np.cov(G, rowvar=True)))####### S = _np.sqrt(_np.diag(G.T.cov()))
            diff = _np.abs(S - prev_S).sum()

            # (c) re-scale R by these stdevs
            S_ser = _pd.Series(S, index=R.index) ###########
            R = R.divide(S_ser, axis=0) ########### R = R.divide(S, axis=0)

            prev_S = S
            i += 1
            logger.debug(f"StatFitter iteration={i}, diff={diff:.6g}")

        # 3) Final pass
        # Recompute final F, B from R
        _, _, V = _np.linalg.svd(R)
        F_final = V.T[:, :num_factors]
        B_final = R.dot(F_final)  # shape=(n_stocks, num_factors)

        # Put F_final in a DataFrame => shape=(n_dates,num_factors)
        F_df = _pd.DataFrame(F_final, index=R.columns,columns=factor_cols)

        # 4) Factor covariance => from final F
        factor_cov = F_df.cov() * DAYS_OF_YEAR
        #factor_cov.index = factor_cols
        #factor_cov.columns = factor_cols  # can be skipped if F_df.cov() already sets them

        # 5) Build final Beta => from original returns domain:
        B_df = returns.T.dot(F_df)
        B_df.columns = factor_cols
        
        # 6) Residual => shape=(n_dates,n_stocks)
        #    G2 = returns - factor prediction
        G2 = returns - F_df.dot(B_df.T)
        idio_ser = G2.std() * _np.sqrt(DAYS_OF_YEAR)

        # 7) Factor returns => project daily returns onto B
        #    shape => (n_dates, n_stocks) x (n_stocks, num_factors) => (n_dates, num_factors)
        factor_ret_array = returns.values.dot(B_df.values)
        factor_returns_df = _pd.DataFrame(
            factor_ret_array,
            index=returns.index,
            columns=factor_cols
        )

        logger.info(
            f"StatisticalModelFitter: finished after {i} iterations, final diff={diff:.6g}, "
            f"{num_factors} factors."
        )

        # 8) Return FactorRiskModel
        return FactorRiskModel(
            factor_cov=factor_cov,
            beta=B_df,
            idio_vol=idio_ser,
            factor_returns=factor_returns_df,
            name="Statistical Risk Model"
        )


