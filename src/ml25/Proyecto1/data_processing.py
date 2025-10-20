# data_processing.py (fix dtype antes de agrupar y mergear)
import argparse, os
import numpy as np
import pandas as pd
from utils import to_dt, to_num, id_to_int

DEF_RAW    = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Archivos base\customer_purchases_train.csv"
DEF_OUTDIR = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1"
DEF_T0     = "2025-09-21"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=str, default=DEF_RAW)
    p.add_argument("--outdir", type=str, default=DEF_OUTDIR)
    p.add_argument("--t0", type=str, default=DEF_T0)
    return p.parse_args()

def _multi_hot(df, col, prefix):
    if col not in df.columns:
        return None
    tmp = df[["customer_id", col]].copy()
    # Normaliza customer_id ANTES de crear dummies y agrupar
    tmp["customer_id"] = tmp["customer_id"].apply(id_to_int).astype("Int64")
    dmy = pd.get_dummies(tmp[col].astype(str).str.lower(), prefix=prefix, dtype=np.int8)
    dmy = pd.concat([tmp[["customer_id"]], dmy], axis=1).groupby("customer_id").max().reset_index()
    # Devuelve como int32 para que coincida con el base
    dmy["customer_id"] = dmy["customer_id"].astype(np.int32)
    return dmy

def build_features(raw_path: str, t0_ts: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    df["purchase_timestamp"]   = to_dt(df.get("purchase_timestamp"))
    df["customer_signup_date"] = to_dt(df.get("customer_signup_date"))
    df["customer_date_of_birth"] = to_dt(df.get("customer_date_of_birth"))
    df["item_price"] = to_num(df.get("item_price"))
    df["customer_item_views"] = to_num(df.get("customer_item_views")).fillna(0)

    # Normaliza customer_id en TODO el DF desde el inicio
    df["customer_id"] = df["customer_id"].apply(id_to_int).astype(np.int32)

    # Solo historial hasta T0
    df = df[df["purchase_timestamp"].notna() & (df["purchase_timestamp"] <= t0_ts)].copy()

    g = df.groupby("customer_id")
    base = g.agg(
        signup_min=("customer_signup_date","min"),
        dob_min=("customer_date_of_birth","min"),
        last_purchase=("purchase_timestamp","max"),
        visitas=("customer_item_views","sum"),
        gasto_total=("item_price","sum"),
        compras=("purchase_timestamp","size"),
    ).reset_index()

    # base ya viene con customer_id int32
    base["antiguedad_dias"] = (t0_ts - base["signup_min"]).dt.days
    base["edad_anios"] = ((t0_ts - base["dob_min"]).dt.days/365.25).round(2)
    base["dias_desde_ultima_compra"] = (t0_ts - base["last_purchase"]).dt.days
    base["gasto_pct"] = base["gasto_total"].rank(pct=True)*100.0

    genero_oh = _multi_hot(df, "customer_gender", "genero")
    device_oh = _multi_hot(df, "purchase_device", "device")
    cat_oh    = _multi_hot(df, "item_category",   "cat")

    feats = base[["customer_id","antiguedad_dias","edad_anios",
                  "dias_desde_ultima_compra","visitas","gasto_pct","compras"]].copy()

    for blk in (genero_oh, device_oh, cat_oh):
        if blk is not None:
            blk["customer_id"] = blk["customer_id"].astype(np.int32)
            feats = feats.merge(blk, on="customer_id", how="left")

    feats = feats.fillna(0)
    feats["customer_id"] = feats["customer_id"].astype(np.int32)
    return feats

if __name__ == "__main__":
    args = parse_args()
    T0 = pd.Timestamp(args.t0)
    out_dir = os.path.join(args.outdir, "out_features_agg", "train")
    os.makedirs(out_dir, exist_ok=True)

    feats = build_features(args.raw, T0)
    path = os.path.join(out_dir, "train_features_per_customer.csv")
    feats.to_csv(path, index=False)
    print(f"[OK] Features -> {path}  (clientes={len(feats)}, cols={feats.shape[1]})")
