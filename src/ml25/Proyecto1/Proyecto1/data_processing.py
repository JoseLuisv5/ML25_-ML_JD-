# data_processing.py (Corregido)
import pandas as pd
import numpy as np
import os

RAW_PATH = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Proyecto1\Archivos base\customer_purchases_train.csv"
OUT_PATH = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\train_features.csv"
T0_DATE = "2025-09-21"

print("Procesando datos...")

# Leer datos
df = pd.read_csv(RAW_PATH)

# Convertir fechas
df["purchase_timestamp"] = pd.to_datetime(df["purchase_timestamp"])
df["customer_signup_date"] = pd.to_datetime(df["customer_signup_date"])
T0 = pd.Timestamp(T0_DATE)

# Filtrar solo datos históricos (hasta T0)
df_hist = df[df["purchase_timestamp"] <= T0].copy()

# Convertir precios a numérico
df_hist["item_price"] = pd.to_numeric(df_hist["item_price"], errors="coerce").fillna(0)

# Agrupar por cliente
features = df_hist.groupby("customer_id").agg(
    total_compras=("purchase_timestamp", "count"),
    total_gasto=("item_price", "sum"),
    promedio_gasto=("item_price", "mean"),
    max_gasto=("item_price", "max"),
    antiguedad_dias=("customer_signup_date", lambda x: (T0 - x.min()).days),
    dias_desde_ultima_compra=("purchase_timestamp", lambda x: (T0 - x.max()).days)
).reset_index()

# Calcular algunas features adicionales
features["frecuencia_compra"] = features["total_compras"] / features["antiguedad_dias"].clip(1)
features["gasto_por_compra"] = features["total_gasto"] / features["total_compras"].replace(0, 1)

# Llenar NaN
features = features.fillna(0)

# Crear directorio de salida si no existe
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Guardar
features.to_csv(OUT_PATH, index=False)
print(f"Features guardadas: {OUT_PATH}")
print(f"Clientes: {len(features)}, Features: {features.shape[1]-1}")