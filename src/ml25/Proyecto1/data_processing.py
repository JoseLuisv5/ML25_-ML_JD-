import os
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Negativos (usa tus utilidades existentes)
from ml25.P01_customer_purchases.boilerplate.negative_generation import (
    gen_random_negatives,
)

# ---------- Paths base ----------
DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
# OJO: usar el DIRECTORIO del archivo
DATA_DIR = (CURRENT_FILE.parent / "../../datasets/customer_purchases").resolve()


# ---------- Helpers I/O ----------
def read_csv(filename: str) -> pd.DataFrame:
    path = (DATA_DIR / f"{filename}.csv").resolve()
    return pd.read_csv(os.path.abspath(path))


def save_df(df: pd.DataFrame, filename: str) -> None:
    path = (DATA_DIR / filename).resolve()
    df.to_csv(os.path.abspath(path), index=False)
    print(f"df saved to {path}")


def df_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ---------- Features de cliente (agregadas) ----------
def extract_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    train_df = df.copy()
    # Fechas
    for col in ["customer_date_of_birth", "customer_signup_date"]:
        train_df[col] = pd.to_datetime(train_df[col], errors="coerce")

    # Punto de referencia (formato YA correcto: año-día-mes)
    today = datetime.strptime("2025-21-09", "%Y-%d-%m")

    # Edad y antigüedad (años enteros)
    train_df["customer_age_years"] = (
        (today - train_df["customer_date_of_birth"]).dt.days // 365
    ).astype("Int64")
    train_df["customer_tenure_years"] = (
        (today - train_df["customer_signup_date"]).dt.days // 365
    ).astype("Int64")

    g = train_df.groupby("customer_id", dropna=False)

    # Categoría más comprada (modo)
    most_purchased_category = g["item_category"].agg(
        lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else np.nan
    )

    customer_feat = pd.DataFrame(
        {
            "customer_id": g["customer_id"].first(),
            "customer_age_years": g["customer_age_years"].first(),
            "customer_tenure_years": g["customer_tenure_years"].first(),
            "customer_prefered_cat": most_purchased_category,
        }
    ).reset_index(drop=True)

    save_df(customer_feat, "customer_features.csv")
    return customer_feat


# ---------- Preprocesamiento ----------
def build_processor(
    df: pd.DataFrame,
    numerical_features,
    categorical_features,
    free_text_features,
    training: bool = True,
) -> pd.DataFrame:
    """
    Ajusta o carga un ColumnTransformer y devuelve un DataFrame transformado.
    Mantiene el resto de columnas por 'passthrough' (incluye purchase_id).
    """
    savepath = DATA_DIR / "preprocessor.pkl"

    if training:
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        free_text_blocks = []
        for col in free_text_features:
            free_text_blocks.append(
                (
                    col,
                    CountVectorizer(
                        stop_words=[
                            "the",
                            "you",
                            "your",
                            "that",
                            "for",
                            "with",
                            "have",
                            "must",
                            "need",
                            "every",
                            "out",
                            "up",
                            "occasion",
                            "stand",
                            "step",
                            "style",
                        ],
                        max_features=30,
                        lowercase=True,
                    ),
                    col,
                )
            )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
                *free_text_blocks,
            ],
            remainder="passthrough",  # <-- conserva purchase_id y demás
        )

        df_fit = df.drop(columns=["label"], errors="ignore")
        processed_array = preprocessor.fit_transform(df_fit)
        joblib.dump(preprocessor, savepath)
    else:
        preprocessor = joblib.load(savepath)
        processed_array = preprocessor.transform(df)

    # Columnas finales
    num_cols = list(numerical_features)
    cat_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(
        categorical_features
    )

    bow_cols = []
    for col in free_text_features:
        vectorizer = preprocessor.named_transformers_[col]
        bow_cols.extend([f"{col}_bow_{t}" for t in vectorizer.get_feature_names_out()])

    other_cols = [
        c
        for c in df.columns
        if c not in list(numerical_features) + list(categorical_features) + list(free_text_features)
    ]

    final_cols = list(num_cols) + list(cat_cols) + bow_cols + other_cols
    processed_df = pd.DataFrame(processed_array, columns=final_cols)
    return processed_df


def preprocess(raw_df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    """
    Preprocesa y conserva 'purchase_id' gracias a remainder='passthrough'.
    """
    # NO borrar 'purchase_id' aquí:
    dropcols = [
        # "purchase_id",  # <- NO la borres
        "customer_date_of_birth",
        "customer_signup_date",
        "purchase_item_rating",
        "purchase_device",
        "purchase_timestamp",
        "customer_item_views",
        "item_release_date",
        "item_avg_rating",
        "item_num_ratings",
        "customer_prefered_cat",
    ]

    # Numéricas
    numerical_feat = ["item_price", "customer_age_years", "customer_tenure_years"]

    # Categóricas (one-hot)
    raw_df = raw_df.copy()
    raw_df["customer_cat_is_prefered"] = (
        raw_df["item_category"] == raw_df["customer_prefered_cat"]
    ).astype(int)
    categorical_features = ["customer_gender", "item_category", "item_img_filename"]

    # Texto
    free_text_features = ["item_title"]

    processed_df = build_processor(
        raw_df,
        numerical_feat,
        categorical_features,
        free_text_features,
        training=training,
    )

    # Borrar columnas no útiles (pero NO purchase_id)
    processed_df = processed_df.drop(columns=dropcols, errors="ignore")
    return processed_df


# ---------- Construcción de dataset de entrenamiento ----------
def read_train_data():
    train_df = read_csv("customer_purchases_train")
    cust_feat = extract_customer_features(train_df)

    # Negativos (1 por positivo) y únicos
    train_df_neg = gen_random_negatives(train_df, n_per_positive=1).drop_duplicates(
        subset=["customer_id", "item_id"]
    )

    # Merge features de cliente
    train_df_cust = pd.merge(train_df, cust_feat, on="customer_id", how="left")

    # Positivos preprocesados (conserva purchase_id)
    processed_pos = preprocess(train_df_cust, training=True)
    processed_pos["label"] = 1

    # Columnas de items/clientes detectadas
    all_cols = processed_pos.columns.tolist()

    # Atributos de ítem (incluye 'item_id')
    item_cols = [c for c in all_cols if "item" in c] + ["item_id"]
    item_cols = list(dict.fromkeys(item_cols))  # quitar duplicados conservando orden
    unique_items = processed_pos[item_cols].drop_duplicates(subset=["item_id"])

    # Atributos de cliente (incluye 'customer_id')
    cust_cols = [c for c in all_cols if "customer" in c] + ["customer_id"]
    cust_cols = list(dict.fromkeys(cust_cols))
    unique_customers = processed_pos[cust_cols].drop_duplicates(subset=["customer_id"])

    # Negativos con atributos (purchase_id quedará NaN y está bien)
    processed_neg = pd.merge(train_df_neg, unique_items, on="item_id", how="left")
    processed_neg = pd.merge(processed_neg, unique_customers, on="customer_id", how="left")
    processed_neg["label"] = 0

    # Dataset completo (mezclado)
    processed_full = (
        pd.concat([processed_pos, processed_neg], axis=0)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # Tipos numéricos; 'purchase_id' se conserva (si no es numérico, quedará NaN)
    full_numeric = df_to_numeric(processed_full)

    y = full_numeric["label"].astype(int)
    # Para entrenar NO uses IDs:
    X = full_numeric.drop(
        columns=["label", "customer_id", "item_id", "purchase_id"], errors="ignore"
    )

    return X, y, full_numeric  # <- devolvemos también el DF completo (con purchase_id)


# ---------- Dataset de prueba ----------
def read_test_data():
    test_df = read_csv("customer_purchases_test")
    cust_feat = read_csv("customer_features")

    merged = pd.merge(test_df, cust_feat, on="customer_id", how="left")

    processed = preprocess(merged, training=False)
    processed = processed.drop(columns=[], errors="ignore")  # por si quieres borrar algo extra

    return df_to_numeric(processed)


# ---------- Main ----------
if __name__ == "__main__":
    # Train (X, y) + DF completo (incluye purchase_id)
    X_train, y_train, df_train_full = read_train_data()

    # Guardar dataset preprocesado COMPLETO (con purchase_id)
    out_train = DATA_DIR / "customer_purchases_train_final_preprocessed.csv"
    df_train_full.to_csv(out_train, index=False)
    print(f"\nDataset final preprocesado (incluye purchase_id) guardado en: {out_train}")

    # Test (incluye purchase_id por passthrough si viene en el raw de test)
    X_test = read_test_data()
    out_test = DATA_DIR / "customer_purchases_test_final_preprocessed.csv"
    X_test.to_csv(out_test, index=False)
    print(f"Dataset de prueba preprocesado guardado en: {out_test}")

    print(f"\nColumnas de test: {X_test.columns.tolist()[:12]} ... (total={len(X_test.columns)})")
    print(f"Forma de test: {X_test.shape}")
