# negative_generation.py (labels: 1 si compra en (T0, T0+30])
import argparse, os
import numpy as np, pandas as pd

<<<<<<< HEAD:src/ml25/Proyecto1/Proyecto1/negative_generation.py
DEF_RAW  = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Proyecto1\Archivos base\customer_purchases_train.csv"
DEF_FEAT = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Proyecto1\out_features_agg\train\train_features_per_customer.csv"
=======
DEF_RAW  = r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\Proyecto1\Archivos base\customer_purchases_train.csv"
DEF_FEAT = r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\Proyecto1\out_features_agg\train\train_features_per_customer.csv"
>>>>>>> 73d113488c0b0635b3f2f45661ba81aaad07b490:src/ml25/Proyecto1/negative_generation.py
DEF_T0   = "2025-09-21"
DEF_OUT  = r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\Proyecto1\out_features_agg\train\train_features_labeled.csv"

# Funciones auxiliares
def to_dt(s):  
    return pd.to_datetime(s, errors="coerce")

def id_to_int(x):
    if pd.isna(x): 
        return np.nan
    try: 
        return int(x)
    except:
        import re
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else np.nan

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw",  type=str, default=DEF_RAW)
    p.add_argument("--feat", type=str, default=DEF_FEAT)
    p.add_argument("--t0",   type=str, default=DEF_T0)
    p.add_argument("--out",  type=str, default=DEF_OUT)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    T0 = pd.Timestamp(args.t0)

    raw = pd.read_csv(args.raw)
    raw["purchase_timestamp"] = to_dt(raw.get("purchase_timestamp"))

    # 1 si compra en (T0, T0+30]; si no 0
    mask = (raw["purchase_timestamp"] > T0) & (raw["purchase_timestamp"] <= T0 + pd.Timedelta(days=30))
    buyers = raw.loc[mask, "customer_id"].apply(id_to_int).dropna().astype(int).unique().tolist()
    buyers_set = set(buyers)

    feat = pd.read_csv(args.feat)
    feat["customer_id"] = feat["customer_id"].astype(int)
    feat["label"] = feat["customer_id"].isin(buyers_set).astype(int)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    feat.to_csv(args.out, index=False)
    print(f"[OK] Labeled -> {args.out} (pos={int((feat.label==1).sum())}, neg={int((feat.label==0).sum())}, total={len(feat)})")