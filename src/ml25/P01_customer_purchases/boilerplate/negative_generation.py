import numpy as np
import pandas as pd


def gen_random_negatives(train_df, n_per_positive=2, random_state=42):
    rng = np.random.default_rng(random_state)

    # All unique customers and items
    customers = train_df["customer_id"].unique()
    items = train_df["item_id"].unique()

    negatives = []
    seen_pairs = set()

    for _, row in train_df.iterrows():
        cust = row["customer_id"]

        # Sample until we have n unique negative pairs for this customer
        samples = 0
        while samples < n_per_positive:
            item = rng.choice(items)

            # Skip if pair already exists in positives or in negatives
            if (cust, item) in seen_pairs or (
                (train_df["customer_id"] == cust) & (train_df["item_id"] == item)
            ).any():
                continue

            negatives.append({"customer_id": cust, "item_id": item, "label": 0})
            seen_pairs.add((cust, item))
            samples += 1

    neg_df = pd.DataFrame(negatives)
    return neg_df


def gen_all_negatives(df):
    """
    Genera todos los pares cliente-producto que no existen como positivos.
    Puede ser muy grande.
    """
    all_items = df["item_id"].unique()
    all_customers = df["customer_id"].unique()
    positives = set(zip(df["customer_id"], df["item_id"]))
    negatives = []

    for customer in all_customers:
        for item in all_items:
            if (customer, item) not in positives:
                negatives.append({"customer_id": customer, "item_id": item, "label": 0})

    df_neg = pd.DataFrame(negatives)
    return df_neg


def gen_smart_negatives(df):
    pass
