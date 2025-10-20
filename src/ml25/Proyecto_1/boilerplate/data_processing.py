# data_processing.py — simple y funcional (con fix de timezone)

import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ---------------- Config ----------------
DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
T0 = pd.Timestamp(DATA_COLLECTED_AT).tz_localize("UTC")      # <- T0 en UTC
T0_PLUS_30 = T0 + pd.Timedelta(days=30)

#DATA_DIR = Path(r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\datasets\customer_purchases")
DATA_DIR = Path(r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\datasets\customer_purchases")
PREPROC_PATH = DATA_DIR / "preprocessor.pkl"
FEATURE_NAMES_PATH = DATA_DIR / "feature_names.npy"
# ---------------------------------------


# -------------- Utils ------------------
def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv")

def _to_int_safe(x):
    try: return int(x)
    except Exception:
        if isinstance(x, str):
            m = re.search(r"(\d+)", x)
            return int(m.group(1)) if m else np.nan
        return np.nan

def _choose(cols, *names):
    low = {c.lower(): c for c in cols}
    for n in names:
        n = n.lower()
        if n in low: return low[n]
        for k, v in low.items():
            if n in k: return v
    return None

def _mode_or_unknown(s: pd.Series):
    s = s.dropna().astype(str)
    return s.mode().iloc[0] if len(s) else "Unknown"
# ---------------------------------------


# --------- Feature engineering ---------
def extract_customer_features(df: pd.DataFrame, training: bool = True):
    data = df.copy()

    col_id     = _choose(data.columns, "customer_id")
    col_ts     = _choose(data.columns, "purchase_timestamp", "timestamp", "purchase_date")
    col_amt    = _choose(data.columns, "item_price", "price", "amount")
    col_cat    = _choose(data.columns, "item_category", "category")
    col_dev    = _choose(data.columns, "purchase_device", "device")
    col_gender = _choose(data.columns, "customer_gender", "gender", "sex")
    col_color  = _choose(data.columns, "item_color", "color")

    if col_ts is None:
        data["__ts__"] = pd.NaT
        col_ts = "__ts__"
    data[col_ts] = pd.to_datetime(data[col_ts], errors="coerce", utc=True)  # tz-aware

    past = data[(data[col_ts].isna()) | (data[col_ts] <= T0)]
    future = data[(data[col_ts] > T0) & (data[col_ts] <= T0_PLUS_30)] if training else pd.DataFrame(columns=data.columns)

    if training:
        buyers = set(
            future[col_id].dropna().apply(_to_int_safe).dropna().astype(int).unique()
        )

    rows = []
    for cust, g in past.groupby(col_id):
        cid = _to_int_safe(cust)
        if pd.isna(cid): continue

        ts = g[col_ts].dropna().sort_values()
        last_dt = ts.max() if len(ts) else pd.NaT
        first_dt = ts.min() if len(ts) else pd.NaT

        recency_days = (T0 - last_dt).days if pd.notna(last_dt) else np.nan
        tenure_days  = (T0 - first_dt).days if pd.notna(first_dt) else np.nan

        def cnt(days):
            if not len(ts): return 0
            return int((ts > (T0 - pd.Timedelta(days=days))).sum())

        amt = pd.to_numeric(g[col_amt], errors="coerce") if col_amt else pd.Series(dtype=float)

        if len(ts) >= 2:
            diffs = np.diff(ts.values) / np.timedelta64(1, "D")
            avg_gap = float(np.mean(diffs)); std_gap = float(np.std(diffs))
        else:
            avg_gap = 0.0; std_gap = 0.0

        rows.append({
            "customer_id": int(cid),
            "n_purchases": int(len(g)),
            "recency_days": 9999 if np.isnan(recency_days) else recency_days,
            "tenure_days": 0 if np.isnan(tenure_days) else tenure_days,
            "purchases_last_30d": cnt(30),
            "purchases_last_60d": cnt(60),
            "purchases_last_90d": cnt(90),
            "total_amount": float(np.nansum(amt)) if len(amt) else 0.0,
            "avg_amount": float(np.nanmean(amt)) if len(amt) else 0.0,
            "max_amount": float(np.nanmax(amt)) if len(amt) else 0.0,
            "min_amount": float(np.nanmin(amt)) if len(amt) else 0.0,
            "avg_days_between": avg_gap,
            "std_days_between": std_gap,
            "top_category": _mode_or_unknown(g[col_cat]) if col_cat else "Unknown",
            "top_device": _mode_or_unknown(g[col_dev]) if col_dev else "Unknown",
            "top_color": _mode_or_unknown(g[col_color]) if col_color else "Unknown",
            "gender": _mode_or_unknown(g[col_gender]) if col_gender else "Unknown",
        })

    feats = pd.DataFrame(rows)
    if training:
        feats["label"] = feats["customer_id"].astype(int).isin(buyers).astype(int)
        return feats, feats["label"].astype(int).values
    return feats
# ---------------------------------------


# ------------- Preprocesamiento -------------
def process_df(df: pd.DataFrame, training: bool = True):
    data = df.copy()
    id_col = "customer_id"
    has_label = "label" in data.columns

    X = data.drop(columns=[id_col, "label"]) if has_label else data.drop(columns=[id_col])
    X = X.loc[:, X.notna().any(axis=0)]

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")

    if training:
        pipe = Pipeline([("pre", pre)])
        Z = pipe.fit_transform(X)

        num_names = num_cols
        cat_names = []
        if cat_cols:
            ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
            cat_names = ohe.get_feature_names_out(cat_cols).tolist()
        feat_names = num_names + cat_names

        joblib.dump(pipe, PREPROC_PATH)
        np.save(FEATURE_NAMES_PATH, np.array(feat_names, dtype=object))

        out = pd.DataFrame(Z, columns=feat_names)
        if has_label: out["label"] = data["label"].values
        out["customer_id"] = data[id_col].values
        return out

    pipe = joblib.load(PREPROC_PATH)
    Z = pipe.transform(X)
    feat_names = np.load(FEATURE_NAMES_PATH, allow_pickle=True).tolist()
    out = pd.DataFrame(Z, columns=feat_names)
    out["customer_id"] = data[id_col].values
    return out
# --------------------------------------------


# ----------------- API pública -----------------
def preprocess(raw_df: pd.DataFrame, training: bool = False):
    if training:
        feats, _ = extract_customer_features(raw_df, training=True)
        proc = process_df(feats, training=True)
        X = proc.drop(columns=["customer_id", "label"])
        y = proc["label"].astype(int).values
        return X, y, proc[["customer_id", "label"]]
    else:
        feats = extract_customer_features(raw_df, training=False)
        proc = process_df(feats, training=False)
        X = proc.drop(columns=["customer_id"])
        return X, proc[["customer_id"]]

def read_train_data():
    df = read_csv("customer_purchases_train")
    X, y, _ = preprocess(df, training=True)
    return X, y

def read_test_data():
    df = read_csv("customer_purchases_test")
    X, _ = preprocess(df, training=False)
    return X
# ----------------------------------------------
