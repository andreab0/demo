
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from .factor_definitions import BaseFactorDefinition

logger = logging.getLogger(__name__)

class FundamentalExposureBuilder:
    """
    Builds a combined DataFrame of factor exposures from a list of BaseFactorDefinition objects.
    """

    def __init__(self, factor_definitions: List[BaseFactorDefinition], max_workers: int = 4):
        self.factor_definitions = factor_definitions
        self.max_workers = max_workers

    def build_exposures(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("build_exposures expects a pd.DataFrame")

        factor_results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {executor.submit(self._safe_compute, fdef, df): fdef for fdef in self.factor_definitions}
            for fut in as_completed(future_map):
                fdef = future_map[fut]
                try:
                    result_df = fut.result()
                    factor_results[fdef] = result_df
                except Exception as ex:
                    logger.exception(f"Error computing factor {fdef}: {ex}")

        # combine horizontally
        if not factor_results:
            return pd.DataFrame()

        all_factor_dfs = list(factor_results.values())
        combined = pd.concat(all_factor_dfs, axis=1)
        return combined

    def _safe_compute(self, factor_def: BaseFactorDefinition, df: pd.DataFrame) -> pd.DataFrame:
        exposures = factor_def.compute_exposures(df)
        # ensure float columns
        for col in exposures.columns:
            exposures[col] = pd.to_numeric(exposures[col], errors="coerce").fillna(0.0).astype(float)
        return exposures
