import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

DATA_DIR = Path(r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\datasets\customer_purchases")
OUT_DIR  = Path(r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto 1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CUST_FEAT_PATH = OUT_DIR / "customer_features.csv"
PER_CUST_PATH = OUT_DIR / "train_features_per_customer.csv"
ADJ_LIST = ["exclusive","style","casual","stylish","elegant","durable","classic","lightweight","modern","premium","cozy","sporty","trendy","soft"]

def read_csv(n: str): return pd.read_csv(DATA_DIR / f"{n}.csv")

def _age_years(s):
    dob = pd.to_datetime(s, errors="coerce")
    return ((pd.Timestamp(DATA_COLLECTED_AT) - dob).dt.days // 365).astype("Int64")

def _age_range(a):
    bins = [-1,17,24,34,44,54,64,150]
    labs = ["<=18","18-24","25-34","35-44","45-54","55-64","65+"]
    return pd.cut(a, bins=bins, labels=labs)

def _img_color(f):
    f = str(f).lower()
    m = {"imgb.jpg":"blue","imgbl.jpg":"black","imgg.jpg":"green","imgr.jpg":"red","imgp.jpg":"pink","imgo.jpg":"orange","imgy.jpg":"yellow","imgw.jpg":"white","imgpr.jpg":"purple"}
    return m.get(f,"unknown")

def _base_row_features(df):
    d = df.copy()
    d["customer_age_years"] = _age_years(d["customer_date_of_birth"])
    d["customer_age_range"] = _age_range(d["customer_age_years"])
    d["customer_tenure_years"] = ((pd.Timestamp(DATA_COLLECTED_AT) - pd.to_datetime(d["customer_signup_date"], errors="coerce")).dt.days // 365).astype("Int64")
    d["item_color"] = d["item_img_filename"].map(_img_color)
    if "purchase_device" not in d.columns: d["purchase_device"] = "unknown"
    if "purchase_timestamp" in d.columns: d["purchase_timestamp"] = pd.to_datetime(d["purchase_timestamp"], errors="coerce")
    return d

def _mode_or_unknown(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else "unknown"

def extract_customer_features(train_df):
    d = _base_row_features(train_df)
    g = d.groupby("customer_id")
    out = pd.DataFrame({
        "customer_id": g["customer_id"].first(),
        "customer_age_years": g["customer_age_years"].median(),
        "customer_age_range": g["customer_age_range"].first(),
        "customer_tenure_years": g["customer_tenure_years"].median(),
        "customer_purchases": g["purchase_id"].count(),
        "customer_spend_total": g["item_price"].sum(),
        "customer_spend_avg": g["item_price"].mean(),
        "customer_price_max": g["item_price"].max(),
        "customer_price_min": g["item_price"].min(),
        "customer_views_sum": g["customer_item_views"].sum() if "customer_item_views" in d.columns else 0,
        "customer_ratings_avg": g["item_avg_rating"].mean() if "item_avg_rating" in d.columns else 0,
        "customer_num_ratings_avg": g["item_num_ratings"].mean() if "item_num_ratings" in d.columns else 0,
        "customer_top_category": g["item_category"].apply(_mode_or_unknown),
        "customer_top_device": g["purchase_device"].apply(_mode_or_unknown),
        "customer_top_color": g["item_color"].apply(_mode_or_unknown),
        "customer_last_purchase": g["purchase_timestamp"].max() if "purchase_timestamp" in d.columns else pd.NaT,
    }).reset_index(drop=True)
    out.to_csv(CUST_FEAT_PATH, index=False)
    return out

def _one_hot_from_values(prefix, s, top=None):
    vc = s.value_counts().head(top) if top else s.value_counts()
    cols = sorted(vc.index.astype(str).tolist())
    oh = pd.get_dummies(s.astype(str), prefix=prefix)
    keep = [f"{prefix}_{c}" for c in cols]
    for c in keep:
        if c not in oh.columns: oh[c] = 0
    return oh[keep].astype(int)

def _adj_binary(series):
    texts = series.fillna("").astype(str).str.lower()
    data = {}
    for w in ADJ_LIST:
        data[f"adj_{w}"] = texts.str.contains(rf"\b{w}\b", regex=True).astype(int)
    return pd.DataFrame(data).astype(int)

def build_per_customer_matrix(train_df):
    d = _base_row_features(train_df)
    d["genero"] = d["customer_gender"].fillna("unk") if "customer_gender" in d.columns else "unk"
    genero = _one_hot_from_values("genero", d.groupby("customer_id")["genero"].agg(_mode_or_unknown))
    medio = _one_hot_from_values("medio", d.groupby("customer_id")["purchase_device"].agg(_mode_or_unknown)) if "purchase_device" in d.columns else pd.DataFrame()
    color = _one_hot_from_values("color", d.groupby("customer_id")["item_color"].agg(_mode_or_unknown))
    cats = _one_hot_from_values("categoria", d.groupby("customer_id")["item_category"].agg(_mode_or_unknown))
    adjs = _adj_binary(d.groupby("customer_id")["item_title"].agg(lambda x: " ".join([str(t) for t in x]) if "item_title" in d.columns else "")) if "item_title" in d.columns else pd.DataFrame()
    cust = d.groupby("customer_id").agg(
        antiguedad_dias=("customer_tenure_years", lambda x: int(np.nanmedian(x)*365) if len(x) else 0),
        edad_anios=("customer_age_years", lambda x: int(np.nanmedian(x)) if len(x) else 0),
        dias_desde_ultima_compra=("purchase_timestamp", lambda x: int((pd.Timestamp(DATA_COLLECTED_AT) - pd.to_datetime(x).max()).days) if x.notna().any() else 0),
        visitas=("customer_item_views", "sum") if "customer_item_views" in d.columns else ("purchase_id","count"),
        compras=("purchase_id","count"),
        gasto=("item_price","sum")
    ).reset_index()
    total = cust["gasto"].sum() if len(cust) else 1
    cust["gasto_pct"] = (cust["gasto"]/total)*100
    cust = cust.drop(columns=["gasto"])
    parts = [cust.set_index("customer_id")]
    for blk in [genero, medio, color, cats, adjs]:
        if blk is not None and not blk.empty: parts.append(blk)
    mat = pd.concat(parts, axis=1).fillna(0).reset_index()
    mat = mat.astype(int, errors="ignore")
    mat.to_csv(PER_CUST_PATH, index=False)
    return mat

def _build_preprocessor(num_cols, cat_cols, text_col):
    trs = [("num", StandardScaler(), num_cols),
           ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
    if text_col: trs.append((text_col, CountVectorizer(max_features=200, lowercase=True, binary=True), text_col))
    return ColumnTransformer(trs, remainder="drop", verbose_feature_names_out=False)

def _get_feature_names(pre, num_cols, cat_cols, text_col):
    n = list(num_cols)
    if "cat" in pre.named_transformers_: n += list(pre.named_transformers_["cat"].get_feature_names_out(cat_cols))
    if text_col and text_col in pre.named_transformers_: n += [f"{text_col}_bow_{t}" for t in pre.named_transformers_[text_col].get_feature_names_out()]
    return n

def _coerce_types(d, num_cols, cat_cols):
    for c in num_cols:
        if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    for c in cat_cols:
        if c in d.columns: d[c] = d[c].astype(str).fillna("unknown")
    return d

def preprocess_fit(df):
    d = _base_row_features(df)
    num_cols = ["item_price","item_avg_rating","item_num_ratings","customer_age_years","customer_tenure_years"]
    cat_cols = ["customer_gender","customer_age_range","customer_top_category","customer_top_device","customer_top_color","item_category","item_color","purchase_device"]
    num_cols = [c for c in num_cols if c in d.columns]
    cat_cols = [c for c in cat_cols if c in d.columns]
    text_col = "item_title" if "item_title" in d.columns else None
    d = _coerce_types(d, num_cols, cat_cols)
    pre = _build_preprocessor(num_cols, cat_cols, text_col)
    X = pre.fit_transform(d)
    cols = _get_feature_names(pre, num_cols, cat_cols, text_col)
    arr = X.toarray() if hasattr(X,"toarray") else X
    return pd.DataFrame(arr, columns=cols), pre

def preprocess_transform(df, pre):
    d = _base_row_features(df)
    num_cols = ["item_price","item_avg_rating","item_num_ratings","customer_age_years","customer_tenure_years"]
    cat_cols = ["customer_gender","customer_age_range","customer_top_category","customer_top_device","customer_top_color","item_category","item_color","purchase_device"]
    num_cols = [c for c in num_cols if c in d.columns]
    cat_cols = [c for c in cat_cols if c in d.columns]
    d = _coerce_types(d, num_cols, cat_cols)
    text_col = "item_title" if "item_title" in d.columns else None
    X = pre.transform(d)
    cols = _get_feature_names(pre, num_cols, cat_cols, text_col)
    arr = X.toarray() if hasattr(X,"toarray") else X
    return pd.DataFrame(arr, columns=cols)

def read_train_data():
    train = read_csv("customer_purchases_train")
    cf = extract_customer_features(train)
    _ = build_per_customer_matrix(train)
    train = train.merge(cf, on="customer_id", how="left")
    X, pre = preprocess_fit(train.drop(columns=[c for c in ["purchase_id"] if c in train.columns]))
    y = train["label"].astype(int) if "label" in train.columns else pd.Series(dtype=int)
    return X, y, pre

def read_test_data(pre):
    test = read_csv("customer_purchases_test")
    cf = pd.read_csv(CUST_FEAT_PATH)
    test = test.merge(cf, on="customer_id", how="left")
    X = preprocess_transform(test.drop(columns=[c for c in ["purchase_id"] if c in test.columns]), pre)
    return X

if __name__ == "__main__":
    X, y, pre = read_train_data()
    _ = read_test_data(pre)
