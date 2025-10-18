import numpy as np
import pandas as pd
import os
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../../datasets/customer_purchases/"


def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    df = pd.read_csv(fullfilename)
    return df


def get_negatives(df):
    unique_customers = df["customer_id"].unique()
    unique_items = set(df["item_id"].unique())

    negatives = {}
    for customer in unique_customers:
        purchased_items = df[df["customer_id"] == customer]["item_id"].unique()
        non_purchased = unique_items - set(purchased_items)
        negatives[customer] = non_purchased
    return negatives


def gen_all_negatives(df):
    negatives = get_negatives(df)
    negative_lst = []
    for customer_id, item_set in negatives.items():
        negatives_for_customer = [
            {"customer_id": customer_id, "item_id": item_id, "label": 0}
            for item_id in item_set
        ]
        negative_lst.extend(negatives_for_customer)
    return pd.DataFrame(negative_lst)


def gen_random_negatives(df, n_per_positive=2, random_state=42):
    rng = np.random.default_rng(random_state)
    negatives = get_negatives(df)
    negative_lst = []
    # aproximamos #positivos por cliente con su conteo de filas en df
    pos_count = df.groupby("customer_id")["item_id"].count()

    for customer_id, item_set in negatives.items():
        k = int(pos_count.get(customer_id, 1) * n_per_positive)
        pool = list(item_set)
        if len(pool) == 0:
            continue
        k = min(k, len(pool))
        rand_items = rng.choice(pool, size=k, replace=False)
        negatives_for_customer = [
            {"customer_id": customer_id, "item_id": item_id, "label": 0}
            for item_id in rand_items
        ]
        negative_lst.extend(negatives_for_customer)
    neg_df = pd.DataFrame(negative_lst)
    return neg_df


def gen_smart_negatives(df, n_per_positive=2, price_tol=0.2, random_state=42):
    """
    Hard negatives:
      - No comprados por el cliente
      - Misma(s) categoría(s) que sí compró
      - Precio similar (± price_tol, p.ej. 20%)
    Completa al azar si no alcanza.
    """
    rng = np.random.default_rng(random_state)

    # cat y precio por item
    item_meta = df[["item_id", "item_category", "item_price"]].drop_duplicates("item_id")
    item_meta["item_price"] = pd.to_numeric(item_meta["item_price"], errors="coerce")

    # por cliente: categorías y rango de precios de sus compras
    cust_pos = df.groupby("customer_id").agg(
        bought_cats=("item_category", lambda s: set(s.dropna().astype(str))),
        price_med=("item_price", lambda s: pd.to_numeric(s, errors="coerce").median()),
        pos_cnt=("item_id", "count"),
    ).reset_index()

    all_items = set(item_meta["item_id"].unique())
    negatives_map = get_negatives(df)  # items no comprados por cliente

    smart_rows = []
    for _, row in cust_pos.iterrows():
        cid = row["customer_id"]
        cats = row["bought_cats"] or set()
        med_price = row["price_med"]
        pos_cnt = int(row["pos_cnt"])
        quota = max(1, pos_cnt * n_per_positive)

        candidate_items = list(negatives_map.get(cid, set()))
        if not candidate_items:
            continue

        cand_meta = item_meta[item_meta["item_id"].isin(candidate_items)].copy()

        # Filtro por categoría
        if len(cats) > 0:
            cand_meta = cand_meta[cand_meta["item_category"].astype(str).isin(cats)]

        # Filtro por precio similar si tenemos mediana
        if pd.notna(med_price) and np.isfinite(med_price) and med_price > 0:
            low = med_price * (1 - price_tol)
            high = med_price * (1 + price_tol)
            cand_meta = cand_meta[
                (cand_meta["item_price"].astype(float).between(low, high, inclusive="both"))
            ]

        pool = cand_meta["item_id"].unique().tolist()

        # si no alcanzan, rellena del resto no comprado
        if len(pool) < quota:
            remainder = list(set(candidate_items) - set(pool))
            rng.shuffle(remainder)
            need = quota - len(pool)
            pool = pool + remainder[:need]

        if len(pool) == 0:
            continue

        k = min(quota, len(pool))
        chosen = rng.choice(pool, size=k, replace=False).tolist()
        for iid in chosen:
            smart_rows.append({"customer_id": cid, "item_id": iid, "label": 0})

    return pd.DataFrame(smart_rows, columns=["customer_id", "item_id", "label"])


def gen_final_dataset(train_df, negatives):
    """
    Devuelve un DF con la MISMA estructura que train_df (más 'label'):
      - Positivos = train_df con label=1
      - Negativos = mergea metadatos de cliente e ítem; campos de compra se rellenan neutro
      - Concat y shuffle
    """
    # Columnas por grupo (según tu comentario)
    customer_columns = [
        "customer_date_of_birth",
        "customer_gender",
        "customer_signup_date",
    ]
    item_columns = [
        "item_title",
        "item_category",
        "item_price",
        "item_img_filename",
        "item_avg_rating",
        "item_num_ratings",
        "item_release_date",
    ]
    purchase_columns = [
        "purchase_id",
        "purchase_timestamp",
        "customer_item_views",
        "purchase_item_rating",
        "purchase_device",
    ]

    id_cols = ["customer_id", "item_id"]

    # Positivos con label=1
    pos_df = train_df.copy()
    pos_df["label"] = 1

    # Metadatos únicos para mergear
    cust_meta = train_df[id_cols[:1] + customer_columns].drop_duplicates("customer_id")
    item_meta = train_df[id_cols[1:] + item_columns].drop_duplicates("item_id")

    # Negativos: unir metadatos de cliente e ítem
    neg_df = negatives.copy()
    neg_df = neg_df.merge(cust_meta, on="customer_id", how="left")
    neg_df = neg_df.merge(item_meta, on="item_id", how="left")

    # Rellenar columnas de "purchase" para los negativos
    for col in purchase_columns:
        if col not in neg_df.columns:
            neg_df[col] = np.nan

    # Valores neutrales razonables
    if "purchase_id" in neg_df.columns:
        neg_df["purchase_id"] = -1
    if "purchase_timestamp" in neg_df.columns:
        # timestamps nulos para negativos
        neg_df["purchase_timestamp"] = pd.NaT
    for col in ["customer_item_views", "purchase_item_rating"]:
        if col in neg_df.columns:
            neg_df[col] = 0
    if "purchase_device" in neg_df.columns:
        neg_df["purchase_device"] = neg_df["purchase_device"].fillna("unknown")

    # Asegurar columnas finales en el mismo orden que train_df + label
    base_cols = id_cols + customer_columns + item_columns + purchase_columns
    # Algunas columnas podrían no existir en el train original;
    # construimos el orden tomando columnas reales del train.
    ordered = []
    for c in base_cols:
        if c in train_df.columns:
            ordered.append(c)

    # Reordenar y concatenar
    pos_df = pos_df[ordered + [c for c in pos_df.columns if c not in ordered]]
    neg_df = neg_df.reindex(columns=pos_df.columns, fill_value=np.nan)

    full_df = pd.concat([pos_df, neg_df], axis=0, ignore_index=True)

    # Shuffle
    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    return full_df


if __name__ == "__main__":
    train_df = read_csv("customer_purchases_train")
    allnegatives = gen_all_negatives(train_df)
    print("All negatives:", allnegatives.shape)
    randnegatives = gen_random_negatives(train_df, n_per_positive=3)
    print("Random negatives:", randnegatives.shape)
    smartnegs = gen_smart_negatives(train_df, n_per_positive=2, price_tol=0.2)
    print("Smart negatives:", smartnegs.shape)

    final_ds = gen_final_dataset(train_df, smartnegs)
    print("Final dataset shape:", final_ds.shape)
    print("Label balance:\n", final_ds["label"].value_counts(dropna=False))
