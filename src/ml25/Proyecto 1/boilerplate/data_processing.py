import os
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------- Configuración base ----------------------
DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
T0 = pd.Timestamp(DATA_COLLECTED_AT)
T0_PLUS_30 = T0 + pd.Timedelta(days=30)

DATA_DIR = Path(r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\datasets\customer_purchases")

PREPROC_PATH = DATA_DIR / "preprocessor.pkl"
FEATURE_NAMES_PATH = DATA_DIR / "feature_names.npy"
# ---------------------------------------------------------------


def read_csv(filename: str):
    file = DATA_DIR / f"{filename}.csv"
    return pd.read_csv(str(file))


def save_df(df, filename: str):
    save_path = DATA_DIR / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"df saved to {save_path}")


def _to_int_safe(x):
    try:
        return int(x)
    except Exception:
        if isinstance(x, str):
            m = re.search(r"(\d+)", x)
            return int(m.group(1)) if m else np.nan
        return np.nan


def _mode_or_nan(series: pd.Series):
    vc = series.dropna().astype(str).value_counts()
    return vc.index[0] if len(vc) else np.nan


def extract_customer_features(df: pd.DataFrame, training: bool = True):
    """
    Agregación por cliente usando SOLO historial <= T0 (sin fuga).
    Si training=True, retorna (features_df, y); en test, solo features_df.
    """
    data = df.copy()

    # ------------------ mapeo robusto de columnas ------------------
    cols_lower = {c.lower(): c for c in data.columns}

    def choose(*candidates):
        for cand in candidates:
            key = cand.lower()
            if key in cols_lower:
                return cols_lower[key]
            for k, v in cols_lower.items():
                if key in k:
                    return v
        return None

    col_id     = choose("customer_id")
    col_ts     = choose("purchase_timestamp", "purchase_time", "timestamp", "purchase_date")
    col_amt    = choose("amount", "price", "item_price", "purchase_amount")
    col_cat    = choose("product_category", "category", "item_category")
    col_dev    = choose("device_type", "purchase_device", "device")
    col_gender = choose("gender", "customer_gender", "sex")
    col_color  = choose("color", "item_color")
    col_prod   = choose("product_id", "item_id", "sku")
    # ---------------------------------------------------------------

    # Timestamp normalizado
    if col_ts in data.columns and col_ts is not None:
        data[col_ts] = pd.to_datetime(data[col_ts], errors="coerce")
    else:
        data[col_ts] = pd.NaT

    # Particiones temporales
    past_df = data[(data[col_ts].isna()) | (data[col_ts] <= T0)]
    future_df = data[(data[col_ts] > T0) & (data[col_ts] <= T0_PLUS_30)] if training else pd.DataFrame(columns=data.columns)

    # Etiquetas (solo training)
    if training:
        buyers = future_df[col_id].dropna().apply(_to_int_safe).dropna().astype(int).unique().tolist()
        buyers_set = set(buyers)
        y_series = past_df[[col_id]].drop_duplicates().copy()
        y_series["label"] = y_series[col_id].apply(_to_int_safe).astype("Int64").isin(buyers_set).astype(int)
    else:
        y_series = None

    def count_in_window(g, days):
        if not g[col_ts].notna().any():
            return 0
        cutoff = T0 - pd.Timedelta(days=days)
        return int((g[col_ts] > cutoff).sum())

    # Agregación por cliente
    agg_rows = []
    for cust_id, g in past_df.groupby(col_id):
        g = g.copy()

        n_purchases = len(g)
        first_dt = g[col_ts].min() if g[col_ts].notna().any() else pd.NaT
        last_dt  = g[col_ts].max() if g[col_ts].notna().any() else pd.NaT
        recency_days = (T0 - last_dt).days if pd.notna(last_dt) else np.nan
        tenure_days  = (T0 - first_dt).days if pd.notna(first_dt) else np.nan

        purchases_30 = count_in_window(g, 30)
        purchases_60 = count_in_window(g, 60)
        purchases_90 = count_in_window(g, 90)

        amt_series = g[col_amt] if (col_amt and col_amt in g.columns) else pd.Series(dtype=float)
        amt_series = pd.to_numeric(amt_series, errors="coerce")
        total_amt = float(np.nansum(amt_series)) if len(amt_series) else np.nan
        avg_amt   = float(np.nanmean(amt_series)) if len(amt_series) else np.nan
        max_amt   = float(np.nanmax(amt_series)) if len(amt_series) else np.nan
        min_amt   = float(np.nanmin(amt_series)) if len(amt_series) else np.nan

        n_prod = g[col_prod].nunique(dropna=True) if col_prod in g.columns else np.nan
        n_cat  = g[col_cat].nunique(dropna=True)  if (col_cat and col_cat in g.columns) else np.nan

        top_cat   = _mode_or_nan(g[col_cat])   if (col_cat   and col_cat   in g.columns) else np.nan
        top_dev   = _mode_or_nan(g[col_dev])   if (col_dev   and col_dev   in g.columns) else np.nan
        top_color = _mode_or_nan(g[col_color]) if (col_color and col_color in g.columns) else np.nan
        gender_md = _mode_or_nan(g[col_gender])if (col_gender and col_gender in g.columns) else np.nan

        if g[col_ts].notna().sum() >= 2:
            ord_ts = g[col_ts].dropna().sort_values().values
            diffs = np.diff(ord_ts) / np.timedelta64(1, "D")
            avg_days_between = float(np.mean(diffs))
            std_days_between = float(np.std(diffs))
        else:
            avg_days_between = np.nan
            std_days_between = np.nan

        agg_rows.append({
            "customer_id": _to_int_safe(cust_id),
            "n_purchases": n_purchases,
            "recency_days": recency_days,
            "tenure_days": tenure_days,
            "purchases_last_30d": purchases_30,
            "purchases_last_60d": purchases_60,
            "purchases_last_90d": purchases_90,
            "total_amount": total_amt,
            "avg_amount": avg_amt,
            "max_amount": max_amt,
            "min_amount": min_amt,
            "n_unique_products": n_prod,
            "n_unique_categories": n_cat,
            "top_category": top_cat,
            "top_device": top_dev,
            "top_color": top_color,
            "gender": gender_md,
            "avg_days_between": avg_days_between,
            "std_days_between": std_days_between,
        })

    features = pd.DataFrame(agg_rows).dropna(subset=["customer_id"])
    features["customer_id"] = features["customer_id"].astype(int)

    if training and y_series is not None:
        y_series[col_id] = y_series[col_id].apply(_to_int_safe)
        y_series = y_series.dropna(subset=[col_id])
        y_series[col_id] = y_series[col_id].astype(int)
        y_series = y_series.rename(columns={col_id: "customer_id"})
        features = features.merge(y_series, on="customer_id", how="left")
        features["label"] = features["label"].fillna(0).astype(int)
        return features, features["label"].astype(int).values
    else:
        return features


def process_df(df: pd.DataFrame, training: bool = True):
    """
    OneHotEncoder + StandardScaler en ColumnTransformer.
    Guarda y reutiliza el preprocesador para test/validación.
    """
    data = df.copy()
    id_col = "customer_id"
    label_col = "label" if "label" in data.columns else None

    X = data.drop(columns=[id_col, label_col]) if label_col else data.drop(columns=[id_col])
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
        ],
        remainder="drop",
    )

    if training:
        pipe = Pipeline([("pre", preprocessor)])
        Z = pipe.fit_transform(X)

        num_names = num_cols
        cat_names = []
        if cat_cols:
            ohe = pipe.named_steps["pre"].named_transformers_["cat"]
            cat_names = ohe.get_feature_names_out(cat_cols).tolist()

        feat_names = num_names + cat_names
        joblib.dump(pipe, PREPROC_PATH)
        np.save(FEATURE_NAMES_PATH, np.array(feat_names, dtype=object))

        processed_df = pd.DataFrame(Z, columns=feat_names)
        if label_col:
            processed_df[label_col] = data[label_col].values
        processed_df["customer_id"] = data[id_col].values
        return processed_df
    else:
        pipe = joblib.load(PREPROC_PATH)
        Z = pipe.transform(X)
        feat_names = np.load(FEATURE_NAMES_PATH, allow_pickle=True).tolist()
        processed_df = pd.DataFrame(Z, columns=feat_names)
        processed_df["customer_id"] = data[id_col].values
        return processed_df


def preprocess(raw_df: pd.DataFrame, training: bool = False):
    """
    Orquesta: agrega por cliente y aplica preprocesador.
    """
    if training:
        feats, _y = extract_customer_features(raw_df, training=True)
        processed = process_df(feats, training=True)
        X = processed.drop(columns=["customer_id", "label"])
        y = processed["label"].astype(int).values
        return X, y, processed[["customer_id", "label"]]
    else:
        feats = extract_customer_features(raw_df, training=False)
        processed = process_df(feats, training=False)
        X = processed.drop(columns=["customer_id"])
        return X, processed[["customer_id"]]


def df_to_numeric(df):
    data = df.copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data


def read_train_data():
    train_df = read_csv("customer_purchases_train")
    X, y, meta = preprocess(train_df, training=True)
    return X, y


def read_test_data():
    test_df = read_csv("customer_purchases_test")
    X_test, meta = preprocess(test_df, training=False)
    return X_test


if __name__ == "__main__":
    train_df = read_csv("customer_purchases_train")
    print("Train shape:", train_df.shape)
    test_df = read_csv("customer_purchases_test")
    print("Test columns:", list(test_df.columns))
