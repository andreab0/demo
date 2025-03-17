
#a basic signal for demonstration purposes
import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple, Dict
import logging
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # or INFO
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

def build_momentum_signal(returns_df: pd.DataFrame, window: int = 20, signflip: float = -1.0) -> pd.DataFrame:
    """
    Construct a momentum  signal:
      - For each date T, compute rolling(20) average of returns up to T-1
      - For each date T, compute min & max of that momentum up to T-1
      - Normalize => 0..1 => then transform => -1..+1 => then apply logistic & signflip
    Returns a DataFrame signal_df with the same shape as returns_df.
    """

    # 1) Basic momentum => day T uses returns up to T-1
    #    rolling(20).mean() for e.g. => day T is average of T-20..T-1
    #    Do an extra shift(1) so day T doesn't see day T's return
    raw_momentum = returns_df.shift(1).rolling(window).mean()

    # Prepare an output DataFrame
    signal_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)

    # 2) We'll walk forward date by date, ensuring at each date T
    #    we only use historical min/max from [start..T-1].
    for i, date in enumerate(raw_momentum.index):
        if i < window:
            # not enough history to compute an signal yet
            continue

        # current momentum row
        mom_row = raw_momentum.iloc[i]

        # min & max from all historical momentum up to (and including) i-1
        # i.e. raw_momentum.iloc[:i] => up to row index i-1
        hist_mom = raw_momentum.iloc[:i]  # slice up to but not including i
        mom_min = hist_mom.min(axis=0)    # vector of per-column min
        mom_max = hist_mom.max(axis=0)

        # safely handle any zero range
        denom = (mom_max - mom_min).replace(0.0, np.nan)  # avoid div by zero

        # 3) Normalize each instrument => 0..1
        normed = (mom_row - mom_min) / denom

        # clip out-of-range
        normed = normed.clip(lower=0.0, upper=1.0)

        # 4) Transform => -1..+1
        final = normed * 2 - 1

        # 6) apply signflip
        final *= signflip

        # store in signal_df
        signal_df.iloc[i] = final.values

    # fill any early rows with 0 or something
    signal_df = signal_df.fillna(0.0)
    return signal_df

