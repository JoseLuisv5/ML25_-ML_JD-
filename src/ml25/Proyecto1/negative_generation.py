import pandas as pd, numpy as np, re, json

RAW       = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Archivos base\customer_purchases_train.csv"
FEAT      = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\train_features_per_customer.csv"
META_FEAT = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\timing_meta.json"
OUT       = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\train_features_labeled.csv"

def to_id_int(x):
    if pd.isna(x): return np.nan
    try: return int(x)
    except:
        m = re.search(r"(\d+)", str(x)); return int(m.group(1)) if m else np.nan

if __name__ == "__main__":
    np.random.seed(42)

    meta = json.load(open(META_FEAT,"r",encoding="utf-8"))
    T0      = pd.Timestamp(meta["T0"])
    max_ts  = pd.Timestamp(meta["max_ts"])

    raw = pd.read_csv(RAW)
    raw["customer_id"] = raw["customer_id"].apply(to_id_int).astype("Int64")
    raw["purchase_timestamp"] = pd.to_datetime(raw["purchase_timestamp"], errors="coerce")

    pos_mask = raw["purchase_timestamp"].notna() & (raw["purchase_timestamp"] > T0) & (raw["purchase_timestamp"] <= max_ts)
    pre_mask = raw["purchase_timestamp"].notna() & (raw["purchase_timestamp"] <= T0)

    buyers = set(raw.loc[pos_mask, "customer_id"].dropna().unique().tolist())
    active_pre = set(raw.loc[pre_mask, "customer_id"].dropna().unique().tolist())
    neg_candidates = list(active_pre - buyers)

    feats = pd.read_csv(FEAT)
    feats["customer_id"] = feats["customer_id"].apply(to_id_int).astype("Int64")

    buyers_in = set(feats.loc[feats["customer_id"].isin(buyers), "customer_id"].unique().tolist())
    neg_pool  = list(set(feats["customer_id"].dropna().unique().tolist()) & set(neg_candidates))

    if len(buyers_in) == 0: raise RuntimeError("Sin positivos en la ventana final (T0, max_ts].")
    if len(neg_pool)  == 0: raise RuntimeError("Sin negativos candidatos cruzados con features.")

    take = min(len(buyers_in), len(neg_pool))
    neg_pick = np.random.choice(neg_pool, size=take, replace=False)

    pos_df = feats[feats["customer_id"].isin(buyers_in)].copy(); pos_df["label"] = 1
    neg_df = feats[feats["customer_id"].isin(neg_pick)].copy();  neg_df["label"] = 0

    out = pd.concat([pos_df,neg_df],axis=0).drop_duplicates("customer_id").sample(frac=1,random_state=42).reset_index(drop=True)
    out.to_csv(OUT, index=False)
    print("Pos:",len(pos_df),"Neg:",len(neg_df),"Total:",len(out), f"| T0={T0} max_ts={max_ts}")
    print("Guardado en", OUT)
