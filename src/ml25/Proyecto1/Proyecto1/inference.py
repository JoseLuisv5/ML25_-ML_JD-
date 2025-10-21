import argparse
import os
import json
import numpy as np
import pandas as pd

DEF_RAW   = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Proyecto1\Archivos base\customer_purchases_test.csv"
DEF_MODEL = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\model_lr.pkl"
DEF_META  = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\model_lr_meta.json"
DEF_OUT   = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\preds.csv"
DEF_T0    = "2025-09-21"

def to_dt(s):
    return pd.to_datetime(s, errors="coerce")

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def id_to_int(x):
    if pd.isna(x):
        return np.nan
    try:
        return int(x)
    except:
        import re
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else np.nan

class PurchaseModel:
    @staticmethod
    def load(path):
        import joblib
        m = PurchaseModel()
        m.model = joblib.load(path)
        return m
    def predict_proba(self, X):
        return self.model.predict_proba(X)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw",   type=str, default=DEF_RAW)
    p.add_argument("--model", type=str, default=DEF_MODEL)
    p.add_argument("--meta",  type=str, default=DEF_META)
    p.add_argument("--out",   type=str, default=DEF_OUT)
    p.add_argument("--t0",    type=str, default=DEF_T0)
    p.add_argument("--no_date_filter", action="store_true", default=False)
    return p.parse_args()

def _multi_hot(df, col, prefix):
    if col not in df.columns:
        return None
    tmp = df[["customer_id", col]].copy()
    tmp["customer_id"] = tmp["customer_id"].apply(id_to_int).astype("Int64")
    dmy = pd.get_dummies(tmp[col].astype(str).str.lower(), prefix=prefix, dtype=np.int8)
    dmy = pd.concat([tmp[["customer_id"]], dmy], axis=1).groupby("customer_id").max().reset_index()
    dmy["customer_id"] = dmy["customer_id"].astype(np.int32)
    return dmy

def build_customer_features(df_raw: pd.DataFrame, t0_ts: pd.Timestamp, no_date_filter: bool) -> pd.DataFrame:
    df = df_raw.copy()
    df["purchase_timestamp"]     = to_dt(df.get("purchase_timestamp"))
    df["customer_signup_date"]   = to_dt(df.get("customer_signup_date"))
    df["customer_date_of_birth"] = to_dt(df.get("customer_date_of_birth"))
    df["item_price"]             = to_num(df.get("item_price"))
    df["customer_item_views"]    = to_num(df.get("customer_item_views")).fillna(0)
    df["customer_id"] = df["customer_id"].apply(id_to_int).astype(np.int32)
    has_ts = bool(df["purchase_timestamp"].notna().any())
    if (not no_date_filter) and has_ts:
        df_f = df[df["purchase_timestamp"].notna() & (df["purchase_timestamp"] <= t0_ts)].copy()
        if len(df_f) == 0:
            df_f = df[df["purchase_timestamp"].notna()].copy()
        if len(df_f) == 0:
            df_f = df.copy()
    else:
        df_f = df.copy()
    g = df_f.groupby("customer_id", dropna=True)
    base = g.agg(
        signup_min    = ("customer_signup_date", "min"),
        dob_min       = ("customer_date_of_birth", "min"),
        last_purchase = ("purchase_timestamp", "max"),
        visitas       = ("customer_item_views", "sum"),
        gasto_total   = ("item_price", "sum"),
        compras       = ("purchase_timestamp", "size"),
    ).reset_index()
    if base["last_purchase"].isna().all():
        base["last_purchase"] = base["signup_min"]
    base["antiguedad_dias"] = (t0_ts - base["signup_min"]).dt.days
    base["edad_anios"] = ((t0_ts - base["dob_min"]).dt.days / 365.25).round(2)
    base["dias_desde_ultima_compra"] = (t0_ts - base["last_purchase"]).dt.days
    if not has_ts:
        base["compras"] = 0
    base["gasto_pct"] = base["gasto_total"].rank(pct=True) * 100.0
    genero_oh = _multi_hot(df_f, "customer_gender", "genero")
    device_oh = _multi_hot(df_f, "purchase_device", "device")
    cat_oh    = _multi_hot(df_f, "item_category",   "cat")
    feats = base[[
        "customer_id", "antiguedad_dias", "edad_anios",
        "dias_desde_ultima_compra", "visitas", "gasto_pct", "compras"
    ]].copy()
    for blk in (genero_oh, device_oh, cat_oh):
        if blk is not None:
            blk["customer_id"] = blk["customer_id"].astype(np.int32)
            feats = feats.merge(blk, on="customer_id", how="left")
    feats = feats.fillna(0)
    feats["customer_id"] = feats["customer_id"].astype(int)
    return feats

def align_to_training(feats: pd.DataFrame, feature_names: list):
    ids = feats["customer_id"].values
    X = feats.drop(columns=["customer_id"]).copy()
    for c in feature_names:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_names]
    return ids, X

if __name__ == "__main__":
    args = parse_args()
    T0 = pd.Timestamp(args.t0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_test = pd.read_csv(args.raw).copy()
    n_test = len(df_test)
    print(f"[INFO] filas test: {n_test}")
    row_id_candidates = ["ID", "Id", "id", "row_id", "Row_ID"]
    row_id_col = next((c for c in row_id_candidates if c in df_test.columns), None)
    if row_id_col is None:
        df_test["ID"] = np.arange(1, n_test + 1, dtype=int)
        row_id_col = "ID"
        print("[WARN] No se encontró columna de ID en el test. Se usará un índice 1..N como ID.")
    if "customer_id" not in df_test.columns:
        print("[WARN] El test no contiene 'customer_id'. Se emitirá pred=0 para todas las filas.")
        sub = pd.DataFrame({"ID": df_test[row_id_col].values, "pred": np.zeros(n_test, dtype=int)})
        sub.to_csv(args.out, index=False)
        print(f"[OK] Predicciones -> {args.out} (filas={len(sub)})")
        raise SystemExit(0)
    df_test["customer_id"] = df_test["customer_id"].apply(id_to_int).astype(np.int32)
    feats_cust = build_customer_features(df_test, T0, no_date_filter=args.no_date_filter)
    print(f"[INFO] clientes con features: {len(feats_cust)}")
    clf = PurchaseModel.load(args.model)
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feat_names = meta.get("feature_names", [])
    thr = float(meta.get("threshold", 0.5))
    if len(feats_cust) == 0:
        print("[WARN] No se pudieron construir features por cliente. Se emitirá pred=0 para todas las filas.")
        df_out = df_test[[row_id_col]].copy()
        df_out["pred"] = 0
        sub = df_out.rename(columns={row_id_col: "ID"})
        assert len(sub) == n_test
        assert sub["ID"].is_unique
        sub.to_csv(args.out, index=False)
        print(f"[OK] Predicciones -> {args.out} (filas={len(sub)})")
        raise SystemExit(0)
    cust_ids, Xcust = align_to_training(feats_cust, feat_names)
    if Xcust.shape[0] == 0:
        print("[WARN] Features alineadas vacías. Se emitirá pred=0 para todas las filas.")
        df_out = df_test[[row_id_col]].copy()
        df_out["pred"] = 0
        sub = df_out.rename(columns={row_id_col: "ID"})
        assert len(sub) == n_test
        assert sub["ID"].is_unique
        sub.to_csv(args.out, index=False)
        print(f"[OK] Predicciones -> {args.out} (filas={len(sub)})")
        raise SystemExit(0)
    probs = clf.predict_proba(Xcust)[:, 1]
    preds = (probs >= thr).astype(int)
    preds_cust = pd.DataFrame({
        "customer_id": cust_ids.astype(int),
        "pred": preds.astype(int)
    })
    df_join = df_test[[row_id_col, "customer_id"]].merge(
        preds_cust, on="customer_id", how="left"
    )
    df_join["pred"] = df_join["pred"].fillna(0).astype(int)
    df_join = df_join.reset_index(drop=True)
    n_test = len(df_test)
    sub = pd.DataFrame({
        "ID": np.arange(n_test, dtype=int),
        "pred": df_join["pred"].astype(int).values
    })
    assert len(sub) == n_test
    assert sub["ID"].min() == 0 and sub["ID"].max() == n_test - 1
    assert sub["ID"].is_unique
    sub.to_csv(args.out, index=False)
    print(f"[OK] Predicciones -> {args.out} (filas={len(sub)}) | ID 0..{n_test-1}")
