# negative_generation.py
# Etiqueta por cliente (1 si compra en (T0, T0+30]) y además genera
# pares cliente×item balanceados 1:1 con negativos "NO compró Y".
import argparse, os, re
import numpy as np, pandas as pd
from utils import to_dt, id_to_int

DEF_RAW  = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Archivos base\customer_purchases_train.csv"
DEF_FEAT = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\out_features_agg\train\train_features_per_customer.csv"
DEF_T0   = "2025-09-21"
DEF_OUT  = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\out_features_agg\train\train_features_labeled.csv"

rng = np.random.default_rng(42)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw",  type=str, default=DEF_RAW)
    p.add_argument("--feat", type=str, default=DEF_FEAT)
    p.add_argument("--t0",   type=str, default=DEF_T0)
    p.add_argument("--out",  type=str, default=DEF_OUT)
    return p.parse_args()

# ------- helpers mínimos para item_key -------
def canon(x):
    if isinstance(x, str):
        t = x.strip().lower()
        return "unk" if t in ("", "nan", "none", "unknown", "unspecified") else t
    return "unk"

def color_from_img(fname):
    if not isinstance(fname, str): return "unk"
    m = re.search(r"img([a-z]+)\.", fname.lower())
    code = (m.group(1) if m else "")
    mapp = {"r":"red","g":"green","y":"yellow","w":"white","o":"orange","p":"purple",
            "bl":"blue","b":"blue","bk":"black","gr":"green"}
    return mapp.get(code, "unk")

def make_item_key(row):
    title = str(row.get("item_title", "") or "").strip().lower()
    if title:
        return title
    cat = canon(row.get("item_category"))
    col = color_from_img(row.get("item_img_filename"))
    return f"{cat}|{col}"

if __name__ == "__main__":
    args = parse_args()
    T0 = pd.Timestamp(args.t0)
    T1 = T0 + pd.Timedelta(days=30)

    # ============= 1) LABEL POR CLIENTE (compatible con training.py) =============
    raw = pd.read_csv(args.raw)
    raw["purchase_timestamp"] = to_dt(raw.get("purchase_timestamp"))
    raw["customer_id"] = raw["customer_id"].apply(id_to_int).astype("Int64")

    # compradores en (T0, T1]
    mask = (raw["purchase_timestamp"] > T0) & (raw["purchase_timestamp"] <= T1)
    buyers = (
        raw.loc[mask, "customer_id"]
           .dropna().astype(int).unique().tolist()
    )
    buyers_set = set(buyers)

    feat = pd.read_csv(args.feat)
    feat["customer_id"] = feat["customer_id"].astype(int)
    feat["label"] = feat["customer_id"].isin(buyers_set).astype(int)

    # balance sencillo ~1:1 (downsample de negativos)
    pos_df = feat[feat.label == 1]
    neg_df = feat[feat.label == 0]
    if len(pos_df) > 0 and len(neg_df) > 0:
        n_target = min(len(neg_df), len(pos_df))  # 1:1
        neg_sample = neg_df.sample(n=n_target, random_state=42) if n_target < len(neg_df) else neg_df
        feat_bal = pd.concat([pos_df, neg_sample], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    else:
        feat_bal = feat.copy()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    feat_bal.to_csv(args.out, index=False)
    print(f"[OK] Labeled (cliente) -> {args.out}  | pos={int((feat_bal.label==1).sum())}  neg={int((feat_bal.label==0).sum())}  total={len(feat_bal)}")

    # ============= 2) PARES CLIENTE×ITEM (negativos “NO compró Y”) =============
    # Esto NO rompe tu pipeline: se guarda además un CSV con pares balanceados 1:1.
    # Útil si luego quieres entrenar por par (cliente, producto).
    # Construye item_key y genera negativos por cliente en base a lo que sí compró.
    try:
        df = raw.copy()
        df["item_category"]     = df.get("item_category").map(canon)
        df["item_img_filename"] = df.get("item_img_filename")
        df["item_title"]        = df.get("item_title")
        df["color"]             = df["item_img_filename"].apply(color_from_img)
        df["item_key"]          = df.apply(make_item_key, axis=1)

        # catálogo histórico <= T0 (si no hay fechas, usa todo)
        hist = df[df["purchase_timestamp"].isna() | (df["purchase_timestamp"] <= T0)]
        global_items = hist["item_key"].dropna().unique().tolist()
        if not global_items:
            global_items = df["item_key"].dropna().unique().tolist()

        # positivos: compras en (T0, T1]
        win = df[df["purchase_timestamp"].notna() & (df["purchase_timestamp"] > T0) & (df["purchase_timestamp"] <= T1)]
        pos_pairs = win[["customer_id","item_key"]].dropna().drop_duplicates()
        if len(pos_pairs) == 0:
            # si no hay positivos, deja un archivo vacío de pares y termina
            pairs_path = os.path.join(os.path.dirname(args.out), "train_pairs_labeled.csv")
            pd.DataFrame(columns=["customer_id","item_key","label"]).to_csv(pairs_path, index=False)
            print("[WARN] No hubo pares positivos en la ventana. Se dejó train_pairs_labeled.csv vacío.")
        else:
            pos_by_cust = pos_pairs.groupby("customer_id")["item_key"].apply(set).to_dict()

            neg_rows = []
            for cust, pos_set in pos_by_cust.items():
                need = len(pos_set)  # 1:1 por cliente
                cand = [it for it in global_items if it not in pos_set]
                if not cand:
                    continue
                replace = len(cand) < need
                sample = rng.choice(cand, size=need, replace=replace)
                for it in sample:
                    neg_rows.append((int(cust), it, 0))

            pos_rows = [(int(r.customer_id), r.item_key, 1) for r in pos_pairs.itertuples(index=False)]
            pairs = pd.DataFrame(pos_rows + neg_rows, columns=["customer_id","item_key","label"]).drop_duplicates()

            pairs_path = os.path.join(os.path.dirname(args.out), "train_pairs_labeled.csv")
            pairs.to_csv(pairs_path, index=False)
            vc = pairs["label"].value_counts()
            print(f"[OK] Pairs (cliente×item) -> {pairs_path}  | pos={int(vc.get(1,0))}  neg={int(vc.get(0,0))}  total={len(pairs)}")
    except Exception as e:
        # no falla el pipeline si falta alguna columna de producto
        print(f"[WARN] No se generaron pares cliente×item (detalle: {e})")
