
import pandas as pd
import numpy as np
import logging
from typing import Optional, Literal, Tuple, Union, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..factor_model import FactorRiskModel

try:
    from sklearn.linear_model import HuberRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class FundamentalModelFitter:
    """
    A very simple fundamental risk model fitting class that can:
      - (optionally) chunk the processing date-by-date (chunked=True) to avoid huge memory usage.
      - (optionally) do robust regression (Huber) instead of OLS.
      - handle daily or monthly cross-sections.

    usage:
        fitter = FundamentalModelFitter(freq="daily", robust=True, chunked=True)
        risk_model = fitter.fit(returns_df, exposures)

    returns_df can be wide or stacked:
      - wide: row=date, columns=tickers => daily returns
      - stacked: (date,ticker) => "returns"

    exposures can be static (index=ticker) or time-varying (multi-index=(date,ticker)).
    """

    def __init__(self,
                 freq: Literal["daily","monthly"] = "daily",
                 ewma_half_life: Optional[float] = None,
                 robust: bool = False,
                 chunked: bool = True,
                 max_workers: int = 4,
                 force_psd: bool = False):
        """
        Parameters
        ----------
        freq : {"daily","monthly"}
            cross-sectional frequency
        ewma_half_life : float, optional
            for factor covariance => if set, do EWMA weighting
        robust : bool
            if True, try using HuberRegressor from scikit-learn for robust cross-section
        chunked : bool
            if True, do date-by-date iteration in memory-friendly manner
            if False, replicate everything into (date,ticker) => factor columns
        max_workers : int
            concurrency for cross-sectional regressions
        force_psd : bool
            if True, force factor covariance to be positive semi-definite
        """
        self.freq = freq
        self.ewma_half_life = ewma_half_life
        self.robust = robust
        self.chunked = chunked
        self.max_workers = max_workers
        self.force_psd = force_psd

    def fit(self, 
            returns: pd.DataFrame, 
            exposures: pd.DataFrame
            ) -> FactorRiskModel:
        """
        Main interface to build a FactorRiskModel.

        returns => wide or stacked
        exposures => static or time-varying, wide or stacked
        """
        if returns.empty:
            logger.warning("FundamentalModelFitter.fit: 'returns' is empty => returning degenerate model.")
            return FactorRiskModel(
                factor_cov=pd.DataFrame(),
                beta=pd.DataFrame(),
                idio_vol=pd.Series(dtype=float),
                factor_returns=pd.DataFrame(),
                name="Fundamental Risk Model"
            )

        # 1) Convert returns to stacked => (date,ticker)="returns"
        R = self._prepare_returns(returns)

        # 2) If chunked => date-by-date approach
        #    else => unify everything into one big (date,ticker) exposures matrix
        if self.chunked:
            factor_returns, residuals = self._run_chunked_regressions(R, exposures)
        else:
            X = self._prepare_exposures_full(R, exposures)
            factor_returns, residuals = self._run_cross_sectional_regressions(R, X)

        # 3) Factor covariance
        factor_cov = self._estimate_factor_cov(factor_returns)
        if self.force_psd:
            factor_cov = force_psd(factor_cov, method='auto', negativity_tol=0.01)
            factor_cov = pd.DataFrame(factor_cov, index=factor_returns.columns, columns=factor_returns.columns)
        # 4) Idiosyncratic vol
        idio_vol = self._compute_idiosyncratic_vol(residuals)

        # 5) Final exposures => pick last date if multi-index or just keep if static
        final_exposures = self._choose_final_exposures(R, exposures)

        # 6) Build model (with factor_returns to let user see daily factor returns).
        logger.info(f"FundamentalModelFitter.fit: built model with factor_cov={factor_cov.shape}, "
                    f"final_exposures={final_exposures.shape}, #factor_returns={factor_returns.shape}")
        return FactorRiskModel(
            factor_cov=factor_cov,
            beta=final_exposures,
            idio_vol=idio_vol,
            factor_returns=factor_returns,
            name="Fundamental Risk Model"
        )

    # ------------------------------------------------------------------
    # PREPARE RETURNS
    # ------------------------------------------------------------------
    def _prepare_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert 'returns' DataFrame to stacked: (date,ticker) => 'returns'.
        Handles wide or stacked input.
        """
        if df.empty:
            logger.warning("_prepare_returns: input 'returns' empty => returning empty.")
            return df

        # make a copy to avoid SettingWithCopy
        df = df.copy()

        # If already stacked => check columns
        if df.index.nlevels == 2 and "returns" in df.columns:
            if df.shape[1] == 1:
                df.columns = ["returns"]
            # ensure float
            df.loc[:, "returns"] = pd.to_numeric(df["returns"], errors="coerce").fillna(0.0)
            logger.debug(f"_prepare_returns: stacked input => shape={df.shape}.")
            return df

        # wide => row=date, col=ticker
        if df.index.nlevels == 1 and df.shape[1] > 1:
            for c in df.columns:
                df.loc[:, c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            # stack
            stacked = df.stack(future_stack=True).to_frame("returns")
            stacked.index.names = ["date","ticker"]
            if not isinstance(stacked.index.get_level_values("date"), pd.DatetimeIndex):
                # convert
                stacked = stacked.reset_index()
                stacked["date"] = pd.to_datetime(stacked["date"])
                stacked = stacked.set_index(["date","ticker"])
            logger.debug(f"_prepare_returns: wide input => stacked shape={stacked.shape}.")
            return stacked

        logger.error("Could not parse 'returns' DataFrame => not wide or stacked with 'returns'.")
        raise ValueError(
            "FundamentalModelFitter._prepare_returns: unexpected 'returns' format.\n"
            "Expected wide => row=date, columns=tickers, or stacked => (date,ticker)->returns"
        )

    # ------------------------------------------------------------------
    # NON-CHUNKED PREPARATION
    # ------------------------------------------------------------------
    def _prepare_exposures_full(self, 
                                returns_stacked: pd.DataFrame,
                                exposures: pd.DataFrame
                                ) -> pd.DataFrame:
        """
        If chunked=False, replicate/merge exposures into a single big (date,ticker) DataFrame.
        If exposures is static => replicate across all dates in returns.
        If exposures is multi-index => direct intersection.
        """
        if exposures.empty:
            logger.warning("_prepare_exposures_full: 'exposures' is empty => no factor exposures.")
            return pd.DataFrame()

        # copy exposures => avoid SettingWithCopy
        exposures = exposures.copy()

        # returns_stacked => (date,ticker)
        dates = returns_stacked.index.get_level_values("date").unique()
        tickers = returns_stacked.index.get_level_values("ticker").unique()

        if exposures.index.nlevels == 2:
            # already (date,ticker)
            common_idx = returns_stacked.index.intersection(exposures.index)
            if common_idx.empty:
                logger.warning("No overlap between returns & exposures => final empty.")
            X = exposures.loc[common_idx].copy()
            logger.debug(f"_prepare_exposures_full: time-varying exposures => shape={X.shape}")
        else:
            # static => replicate
            for c in exposures.columns:
                exposures.loc[:, c] = pd.to_numeric(exposures[c], errors="coerce").fillna(0.0)
            big_pieces = []
            for dt in dates:
                tmp = exposures.copy()
                tmp["__DATE__"] = dt
                big_pieces.append(tmp)
            all_exp = pd.concat(big_pieces)
            all_exp.set_index(["__DATE__", all_exp.index], inplace=True)
            all_exp.index.names = ["date","ticker"]
            X = all_exp

            final_idx = returns_stacked.index.intersection(X.index)
            X = X.loc[final_idx].copy()
            logger.debug(f"_prepare_exposures_full: static exposures => after replicate => shape={X.shape}")

        return X

    # ------------------------------------------------------------------
    # NON-CHUNKED CROSS-SECTION
    # ------------------------------------------------------------------
    def _run_cross_sectional_regressions(self,
                                         returns_stacked: pd.DataFrame,
                                         exposures_big: pd.DataFrame
                                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loop over each date => cross-section => factor returns & residuals.
        """
        if exposures_big.empty or returns_stacked.empty:
            logger.warning("_run_cross_sectional_regressions: empty => no factor returns.")
            return pd.DataFrame(), pd.DataFrame()

        if self.freq == "daily":
            unique_dates = returns_stacked.index.get_level_values("date").unique()
        else:
            date_periods = returns_stacked.index.get_level_values("date").to_period('M')
            unique_dates = date_periods.unique()

        factor_cols = exposures_big.columns
        factor_returns_list = []
        residuals_list = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures_map = {}
            for dval in unique_dates:
                fut = executor.submit(
                    self._regress_one_date,
                    dval, returns_stacked, exposures_big, factor_cols
                )
                futures_map[fut] = dval

            for fut in as_completed(futures_map):
                dval = futures_map[fut]
                try:
                    fr, resid = fut.result()
                    if fr is not None:
                        factor_returns_list.append(fr)
                    if resid is not None:
                        residuals_list.append(resid)
                except Exception as ex:
                    logger.exception(f"Cross-sectional regression failed for {dval}: {ex}")

        if factor_returns_list:
            factor_returns_df = pd.concat(factor_returns_list).sort_index()
        else:
            factor_returns_df = pd.DataFrame(columns=factor_cols)

        if residuals_list:
            residuals_df = pd.concat(residuals_list).sort_index()
        else:
            residuals_df = pd.DataFrame(columns=["resid"])

        return factor_returns_df, residuals_df

    def _regress_one_date(self,
                          date_val,
                          returns_stacked: pd.DataFrame,
                          exposures_big: pd.DataFrame,
                          factor_cols: pd.Index
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        For a single date => slice ret & exposures => run robust/OLS => factor returns & residuals
        """
        # pick daily or monthly mask
        if self.freq == "daily":
            mask = (returns_stacked.index.get_level_values("date") == date_val)
        else:
            row_periods = returns_stacked.index.get_level_values("date").to_period('M')
            mask = (row_periods == date_val)

        ret_slice = returns_stacked.loc[mask, "returns"]
        exp_slice = exposures_big.loc[mask, factor_cols]

        if ret_slice.empty or exp_slice.empty:
            return None, None

        common_idx = ret_slice.index.intersection(exp_slice.index)
        ret_slice = ret_slice.loc[common_idx]
        exp_slice = exp_slice.loc[common_idx]

        if ret_slice.empty or exp_slice.empty:
            return None, None

        X = exp_slice.values.astype(float)
        y = ret_slice.values.astype(float)

        if X.shape[0] < X.shape[1]:
            logger.debug(f"_regress_one_date: date={date_val}, #stocks={X.shape[0]} < #factors={X.shape[1]} => skip.")
            return None, None

        beta, residuals = self._run_regression(X, y)
        factor_ret_df = pd.DataFrame([beta], columns=factor_cols, index=[date_val])
        resid_df = pd.DataFrame(residuals, index=ret_slice.index, columns=["resid"])
        return factor_ret_df, resid_df

    # ------------------------------------------------------------------
    # CHUNKED CROSS-SECTION
    # ------------------------------------------------------------------
    def _run_chunked_regressions(self,
                                 returns_stacked: pd.DataFrame,
                                 exposures: pd.DataFrame
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Memory-friendly path: for each date, slice out returns, replicate or slice exposures,
        run cross-section, accumulate factor returns & residuals.
        """
        if returns_stacked.empty:
            return pd.DataFrame(), pd.DataFrame()

        if self.freq == "daily":
            all_dates = returns_stacked.index.get_level_values("date").unique().sort_values()
        else:
            date_periods = returns_stacked.index.get_level_values("date").to_period('M')
            all_dates = date_periods.unique().sort_values()

        is_static = (exposures.index.nlevels == 1)
        factor_cols = list(exposures.columns)

        factor_returns_list = []
        residuals_list = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {}
            for dval in all_dates:
                fut = executor.submit(
                    self._regress_one_chunk_date,
                    dval, returns_stacked, exposures, factor_cols, is_static
                )
                future_map[fut] = dval

            for fut in as_completed(future_map):
                date_val = future_map[fut]
                try:
                    fr, resid = fut.result()
                    if fr is not None:
                        factor_returns_list.append(fr)
                    if resid is not None:
                        residuals_list.append(resid)
                except Exception as ex:
                    logger.exception(f"Chunked regression failed for {date_val}: {ex}")

        # combine factor returns
        if factor_returns_list:
            factor_returns_df = pd.concat(factor_returns_list).sort_index()
        else:
            factor_returns_df = pd.DataFrame()

        # combine residuals
        if residuals_list:
            residuals_df = pd.concat(residuals_list).sort_index()
        else:
            residuals_df = pd.DataFrame()

        return factor_returns_df, residuals_df

    def _regress_one_chunk_date(self,
                                date_val,
                                returns_stacked: pd.DataFrame,
                                exposures: pd.DataFrame,
                                factor_cols: List[str],
                                is_static: bool
                                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        For a single date => slice returns => replicate/slice exposures => run robust/OLS => factor returns & residuals
        """
        if self.freq == "daily":
            mask = (returns_stacked.index.get_level_values("date") == date_val)
        else:
            row_periods = returns_stacked.index.get_level_values("date").to_period('M')
            mask = (row_periods == date_val)

        ret_slice = returns_stacked.loc[mask, "returns"]
        if ret_slice.empty:
            return None, None

        tickers = ret_slice.index.get_level_values("ticker").unique()

        # replicate or slice exposures
        if is_static:
            # static => reindex by tickers
            sub_exp = exposures.reindex(tickers).fillna(0.0)
        else:
            # time-varying => (date,ticker)
            try:
                sub_daily = exposures.xs(date_val, level="date", drop_level=False)
            except KeyError:
                return None, None
            # droplevel to get ticker index
            if "date" in sub_daily.index.names:
                sub_daily_indexed = sub_daily.droplevel("date")
            else:
                sub_daily_indexed = sub_daily

            sub_exp = sub_daily_indexed.reindex(tickers).fillna(0.0)

        if sub_exp.empty:
            return None, None

        common = tickers.intersection(sub_exp.index)
        ret_slice = ret_slice.loc[(slice(None), common)]
        sub_exp = sub_exp.loc[common, factor_cols]

        if ret_slice.empty or sub_exp.empty:
            return None, None

        X = sub_exp.values.astype(float)
        y = ret_slice.values.astype(float)

        if X.shape[0] < X.shape[1]:
            logger.debug(f"_regress_one_chunk_date: date={date_val}, #stocks={X.shape[0]} < #factors={X.shape[1]} => skip.")
            return None, None

        beta, residuals = self._run_regression(X, y)
        factor_ret_df = pd.DataFrame([beta], columns=factor_cols, index=[date_val])

        multi_idx = pd.MultiIndex.from_product([[date_val], common], names=["date","ticker"])
        resid_df = pd.DataFrame(residuals, index=multi_idx, columns=["resid"])
        return factor_ret_df, resid_df

    # ------------------------------------------------------------------
    # REGRESSION HELPER: OLS or ROBUST
    # ------------------------------------------------------------------
    def _run_regression(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        If robust=True & scikit-learn installed => HuberRegressor,
        else => ordinary least squares (np.linalg.lstsq).
        Returns (beta, residual).
        """
        if self.robust and SKLEARN_AVAILABLE:
            huber = HuberRegressor(max_iter=1000)
            huber.fit(X, y)
            beta = huber.coef_
            fitted = huber.predict(X)
            resid = y - fitted
        else:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            fitted = X @ beta
            resid = y - fitted
        return beta, resid

    # ------------------------------------------------------------------
    # FACTOR COVARIANCE
    # ------------------------------------------------------------------
    def _estimate_factor_cov(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        if factor_returns.empty:
            logger.warning("_estimate_factor_cov: No factor returns => empty cov.")
            return pd.DataFrame()

        # daily covariance
        if self.ewma_half_life is not None:
            daily_cov = self._ewma_cov(factor_returns)
        else:
            daily_cov = factor_returns.cov()

        annualized_cov = daily_cov * 252
        return annualized_cov

    def _ewma_cov(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        arr = factor_returns.values
        mean_fr = arr.mean(axis=0)
        arr_centered = arr - mean_fr
        n = arr.shape[0]
        alpha = np.log(2) / self.ewma_half_life
        w_sum = 0.0
        cov_mat = np.zeros((arr.shape[1], arr.shape[1]), dtype=float)

        for t in range(n):
            weight = np.exp(-alpha * (n - 1 - t))
            row = arr_centered[t:t+1]
            cov_mat += weight * (row.T @ row)
            w_sum += weight

        cov_mat /= w_sum
        return pd.DataFrame(cov_mat, columns=factor_returns.columns, index=factor_returns.columns)

    # ------------------------------------------------------------------
    # IDIOSYNCRATIC RISK
    # ------------------------------------------------------------------
    def _compute_idiosyncratic_vol(self, residuals: pd.DataFrame) -> pd.Series:
        if residuals.empty:
            logger.warning("_compute_idiosyncratic_vol: empty => returning empty Series.")
            return pd.Series(dtype=float)

        ticker_idx = residuals.index.get_level_values("ticker")
        daily_std = residuals.groupby(ticker_idx)["resid"].std()
        ann_vol = daily_std * np.sqrt(252)
        ann_vol.name = "idio_vol"
        return ann_vol

    # ------------------------------------------------------------------
    # FINAL EXPOSURES
    # ------------------------------------------------------------------
    def _choose_final_exposures(self,
                                returns_stacked: pd.DataFrame,
                                exposures: pd.DataFrame
                                ) -> pd.DataFrame:
        """
        If single-level => static => just return it
        If multi-index => pick last date from returns
        """
        if exposures.empty:
            return exposures

        if exposures.index.nlevels == 1:
            # static
            return exposures.copy()

        # multi-index => (date,ticker)
        last_date = returns_stacked.index.get_level_values("date").max()
        try:
            sub = exposures.xs(last_date, level="date")
            return sub.copy()
        except KeyError:
            logger.warning(f"_choose_final_exposures: no exposures for last_date={last_date} => empty.")
            return pd.DataFrame()



def nearest_posdef_clip(A, eps=1e-12):
    """
    Simple eigenvalue-clipping approach.
    1) Symmetrize A.
    2) Eigen-decompose.
    3) Clip negative eigenvalues to eps (or 0).
    4) Reconstruct & symmetrize.
    """
    A_sym = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A_sym)
    vals_clipped = np.maximum(vals, eps)
    A_pd = (vecs * vals_clipped).dot(vecs.T)
    A_pd = 0.5 * (A_pd + A_pd.T)
    return A_pd

def nearest_posdef_higham(A, max_iter=10, tol=1e-14):
    """
    Higham's nearest positive semidefinite matrix approach.
    Reference:
      Nicholas Higham (1988), "Computing a nearest symmetric positive
      semidefinite matrix," Linear Algebra and its Applications.
    """
    A_sym = 0.5 * (A + A.T)
    X = A_sym.copy()
    Y = np.zeros_like(A_sym)
    
    for _ in range(max_iter):
        R = X - Y
        R_sym = 0.5 * (R + R.T)
        
        eigvals, eigvecs = np.linalg.eigh(R_sym)
        eigvals_clipped = np.clip(eigvals, 0, None)
        
        R_psd = (eigvecs * eigvals_clipped) @ eigvecs.T
        
        Y = R_psd - R
        X = R_psd
        
        diff_norm = np.linalg.norm(X - A_sym, ord='fro')
        if diff_norm < tol:
            break
    
    X = 0.5 * (X + X.T)
    return X

def force_psd(A, method="auto", negativity_tol=0.01, eps=1e-12):
    """
    Force matrix A to be PSD using either:
      - "clip": eigenvalue clipping
      - "higham": Higham's iterative approach
      - "auto":  measure negativity ratio & pick the best approach

    negativity_tol => threshold ratio to decide which approach for 'auto'
    eps => small shift to ensure positive eigenvalues in clipping.

    Returns PSD matrix (closest in some sense).
    """
    # If method is forced:
    if method.lower() == "clip":
        return nearest_posdef_clip(A, eps=eps)
    elif method.lower() == "higham":
        return nearest_posdef_higham(A)
    
    # Otherwise, "auto"
    # 1) Symmetrize
    A_sym = 0.5*(A + A.T)
    # 2) Eigs
    vals, _ = np.linalg.eigh(A_sym)
    
    negative_sum = np.abs(vals[vals<0]).sum()
    positive_sum = np.abs(vals).sum()  # sum of absolute, including negative
    if positive_sum < 1e-15:
        # degenerate: basically all near zero
        return nearest_posdef_clip(A, eps=eps)
    negativity_ratio = negative_sum / positive_sum
    
    # If negativity is small, do the faster clip. Otherwise do Higham.
    if negativity_ratio <= negativity_tol:
        return nearest_posdef_clip(A, eps=eps)
    else:
        return nearest_posdef_higham(A)