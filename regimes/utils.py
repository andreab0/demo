import pandas as pd
import numpy as _np
from dal.config import defaultCacheDir, dataCategoryInfo_bdh, dataCategoryInfo_bdp, dataCategoryInfo_bds
import os 
from cryptography.fernet import Fernet
import json
import logging
import re


all_data_types = {'bdh':dataCategoryInfo_bdh,'bdp':dataCategoryInfo_bdp,'bds':dataCategoryInfo_bds}

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def log_change(sindexData, periods=1, dropNA = False,concat = False, shiftbyNeg = False, useAbs = False, cliptoZero = False):
    if shiftbyNeg:
        minVal = sindexData.min().min()
        if minVal < 0:
            sindexData = sindexData - minVal
            
    if useAbs:
        logChg = lambda df: {str(periods)+"D_chg_"+str(sec): _np.log(_np.abs(df[sec])) - _np.log(_np.abs(df[sec].shift(periods))) for sec in df.columns}
    else:
        if cliptoZero:      
            logChg = lambda df: {str(periods)+"D_chg_"+str(sec): _np.log(df[sec].clip(lower=1e-10)) - _np.log(df[sec].clip(lower=1e-10).shift(periods)) for sec in df.columns}
        else:
            logChg = lambda df: {str(periods)+"D_chg_"+str(sec): _np.log(df[sec]) - _np.log(df[sec].shift(periods)) for sec in df.columns}
    t = sindexData.assign(**logChg(sindexData)) if concat else pd.DataFrame(logChg(sindexData))
    return t.dropna() if dropNA else t.fillna(0.0)


def noise(cleandf:pd.DataFrame,noisestd:float,inplace = False)->pd.DataFrame:
    if inplace:
        return cleandf +pd.DataFrame(_np.random.normal(0, noisestd, size=cleandf.shape), columns=cleandf.columns, index=cleandf.index)
    else:
        return pd.DataFrame(_np.random.normal(0, noisestd, size=cleandf.shape), columns=cleandf.columns, index=cleandf.index)

def arithmetic_change(
    sindexData,
    periods=1,
    dropNA=False,
    concat=False,
    shiftbyNeg=False,
    useAbs=False,
    cliptoZero=False
):
    if shiftbyNeg:
        minVal = sindexData.min().min()
        if minVal < 0:
            sindexData = sindexData - minVal
    
    if useAbs:
        diffFunc = lambda df: {
            f"{periods}D_chg_{sec}": df[sec].abs() - df[sec].abs().shift(periods)
            for sec in df.columns
        }
    else:
        if cliptoZero:
            diffFunc = lambda df: {
                f"{periods}D_chg_{sec}": df[sec].clip(lower=1e-10) 
                                        - df[sec].clip(lower=1e-10).shift(periods)
                for sec in df.columns
            }
        else:
            diffFunc = lambda df: {
                f"{periods}D_chg_{sec}": df[sec] - df[sec].shift(periods)
                for sec in df.columns
            }
    
    # either assign to the original DataFrame (concat=True) or build a new DataFrame
    t = (sindexData.assign(**diffFunc(sindexData))
         if concat else pd.DataFrame(diffFunc(sindexData)))
    
    return t.dropna() if dropNA else t.fillna(0.0)

