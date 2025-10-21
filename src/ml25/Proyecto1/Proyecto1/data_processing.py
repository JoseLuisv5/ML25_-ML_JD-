import pandas as pd
import numpy as np
import os

RAW_PATH = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Proyecto1\Archivos base\customer_purchases_train.csv"
OUT_PATH = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\train_features.csv"
T0_DATE = "2025-09-21"

print("Procesando datos...")

df = pd.read_csv(RAW_PATH)

df["purchase_timestamp"] = pd.to_datetime(df["purchase_timestamp"])
df["customer_signup_date"] = pd.to_datetime(df["customer_signup_date"])
T0 = pd.Timestamp(T0_DATE)

df_hist = df[df["purchase_timestamp"] <= T0].copy()

df_hist["item_price"] = pd.to_numeric(df_hist["item_price"], errors="coerce").fillna(0)

features = df_hist.groupby("customer_id").agg(
    total_compras=("purchase_timestamp", "count"),
    total_gasto=("item_price", "sum"),
    promedio_gasto=("item_price", "mean"),
    max_gasto=("item_price", "max"),
    antiguedad_dias=("customer_signup_date", lambda x: (T0 - x.min()).days),
    dias_desde_ultima_compra=("purchase_timestamp", lambda x: (T0 - x.max()).days)
).reset_index()

features["frecuencia_compra"] = features["total_compras"] / features["antiguedad_dias"].clip(1)
features["gasto_por_compra"] = features["total_gasto"] / features["total_compras"].replace(0, 1)

features = features.fillna(0)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

features.to_csv(OUT_PATH, index=False)
print(f"Features guardadas: {OUT_PATH}")
print(f"Clientes: {len(features)}, Features: {features.shape[1]-1}")
