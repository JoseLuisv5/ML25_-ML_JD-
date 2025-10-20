# utils.py
import re
import numpy as np
import pandas as pd

def to_dt(s):
    return pd.to_datetime(s, errors="coerce")

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def id_to_int(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else np.nan
