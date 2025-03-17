
import pandas as pd, numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Any,Callable
import logging

logger = logging.getLogger(__name__)


class BaseFactorDefinition(ABC):
    """
    Abstract base class that all factor definitions must extend.
    Ensures a standard interface for computing factor exposures.
    """

    @property
    @abstractmethod
    def factor_names(self) -> List[str]:
        """
        Return a list of factor names produced by this definition.
        e.g. ["Value"] for a single factor, or ["sector_Energy", "sector_Materials"] for many.
        """
        pass

    @abstractmethod
    def compute_exposures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Core method to calculate factor exposures from a DataFrame of fundamental or metadata.

        Parameters
        ----------
        df : pd.DataFrame
            Typically includes columns needed by this factor definition
            (e.g., "Sector", "MarketCap").

        Returns
        -------
        pd.DataFrame
            Index aligned with `df`. Columns = self.factor_names.
            Values = numeric factor exposures.
        """
        pass


class CategoricalFactorDefinition(BaseFactorDefinition):
    """
    Builds one-hot or dummy exposures from a given categorical column.
    Each category becomes a separate factor (column).
    """

    def __init__(self,
                 col_name: str,
                 prefix: str,
                 drop_first: bool = False,
                 fill_value: Optional[str] = "Unknown"):
        """
        Parameters
        ----------
        col_name : str
            Name of the column containing the categorical variable.
        prefix : str
            Prefix for the factor columns. e.g. "sector" => columns like sector_Energy
        drop_first : bool
            If True, drop one category column to avoid the full-dummy trap.
        fill_value : Optional[str]
            If not None, fill NA with this value before one-hot encoding.
        """
        self.col_name = col_name
        self.prefix = prefix
        self.drop_first = drop_first
        self.fill_value = fill_value
        self._factor_names: List[str] = []  # We'll populate dynamically

    @property
    def factor_names(self) -> List[str]:
        """
        After compute_exposures() has run once, we store the newly created dummy columns.
        If not yet run, returns an empty list or you can guess the factor names from the dataset you expect.
        """
        return self._factor_names

    def compute_exposures(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.col_name not in df.columns:
            msg = f"CategoricalFactorDefinition expects column '{self.col_name}' not found in DataFrame."
            logger.error(msg)
            # Could raise an exception or return an empty DataFrame
            raise KeyError(msg)

        data = df[self.col_name].copy()

        if self.fill_value is not None:
            data = data.fillna(self.fill_value)

        dummies = pd.get_dummies(data, prefix=self.prefix, drop_first=self.drop_first)
        dummies.index = df.index  # align index

        # Record factor names
        self._factor_names = list(dummies.columns)
        return dummies


class NumericFactorDefinition(BaseFactorDefinition):
    """
    A numeric factor that can:
      1) read from a source column (like MarketCap),
      2) apply a transform_func (like np.log),
      3) optionally do winsorization,
      4) optionally do z-score standardization,
      5) handle date-by-date if exposures have multi-index (date,ticker).
    """

    def __init__(self,
                 factor_name: str,
                 source_col: str,
                 transform_func: Optional[Callable[[pd.Series], pd.Series]] = None,
                 winsor_percentile: float = 0.01,
                 zscore: bool = True,
                 apply_cross_sectional: bool = True
                 ):
        """
        Parameters
        ----------
        factor_name : str
            The name for the factor, e.g. "Size".
        source_col : str
            Column name in df that contains the raw numeric data (like "MarketCap").
        transform_func : callable, optional
            A function that takes a pd.Series -> pd.Series, e.g. np.log.
        winsor_percentile : float
            If > 0, apply top/bottom winsorization at that percentile.
        zscore : bool
            If True, do standard deviation scaling (mean=0, std=1).
        apply_cross_sectional : bool
            If True, we apply winsor & zscore cross-section by date if df is multi-index (date,ticker).
            If False, we just do it once globally. 
        """
        self._factor_name = factor_name
        self.source_col = source_col
        self.transform_func = transform_func
        self.winsor_percentile = winsor_percentile
        self.zscore = zscore
        self.apply_cross_sectional = apply_cross_sectional

    @property
    def factor_names(self) -> List[str]:
        return [self._factor_name]

    def compute_exposures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1) If 'source_col' not in df, raise an error.
        2) Take that column => transform => winsor => zscore => return as [factor_name].
        3) If apply_cross_sectional and df index has date level => do per date.
        4) Else do once globally.
        """
        if self.source_col not in df.columns:
            raise KeyError(f"NumericFactorDefinition expects '{self.source_col}' in df columns.")

        s = df[self.source_col].astype(float).copy()
        if self.transform_func:
            s = self.transform_func(s)  # e.g. np.log
            
        if isinstance(s, np.ndarray):
            s = pd.Series(s.squeeze())
            s.name = self.source_col
            s.index = df.index
            
        if self.apply_cross_sectional and self._df_has_date_ticker(df):
            # do per date
            # groupby date level => apply winsor, zscore
            grouped = s.groupby(level=self._date_level(df))
            s = grouped.apply(self._winsor_and_zscore)

        else:
            # do once globally
            s = self._winsor_and_zscore(s)
        
        return pd.DataFrame({self._factor_name: s}, index=df.index)




    def _winsor_and_zscore(self, series: pd.Series) -> pd.Series:
        # step 1) winsor
        if isinstance(series, np.ndarray): series = pd.Series(series.squeeze())
        if self.winsor_percentile > 0.0:
            lower = series.quantile(self.winsor_percentile)
            upper = series.quantile(1 - self.winsor_percentile)

            series = series.clip(lower, upper)

        # step 2) zscore
        if self.zscore:
            mean_ = series.mean()
            std_ = series.std()
            if std_ > 0:
                series = (series - mean_) / std_
            else:
                # if zero std, everything is same => set to 0
                series = series - mean_  # => all zero
        return series

    def _df_has_date_ticker(self, df: pd.DataFrame) -> bool:
        """Check if df index is multi-index with 'date' in level names."""
        if df.index.nlevels < 2:
            return False
        return "date" in df.index.names

    def _date_level(self, df: pd.DataFrame) -> Union[str, int]:
        """Return the index name or level number for 'date' if it exists."""
        for i, name in enumerate(df.index.names):
            if name == "date":
                return name
        # fallback
        return 0
