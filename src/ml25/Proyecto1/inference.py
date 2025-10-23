# inference_final.py
import argparse
import os
import json
import numpy as np
import pandas as pd

from utils import to_dt, to_num, id_to_int
from model import PurchaseModel

# --- Paths por defecto ---
DEF_RAW   = r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\Proyecto1\Archivos base\customer_purchases_test.csv"
DEF_MODEL = r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\Proyecto1\model_lr_balanced.pkl"
DEF_META  = r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\Proyecto1\model_lr_meta_balanced.json"
DEF_OUT   = r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\Proyecto1\preds.csv"
DEF_T0    = "2025-09-21"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw",   type=str, default=DEF_RAW)
    p.add_argument("--model", type=str, default=DEF_MODEL)
    p.add_argument("--meta",  type=str, default=DEF_META)
    p.add_argument("--out",   type=str, default=DEF_OUT)
    p.add_argument("--t0",    type=str, default=DEF_T0)
    p.add_argument("--threshold", type=float, default=None, help="Threshold personalizado (si no se usa, se calcula automático)")
    p.add_argument("--no_date_filter", action="store_true", default=False)
    return p.parse_args()

def _multi_hot(df, col, prefix):
    if col not in df.columns:
        return None
    tmp = df[["customer_id", col]].copy()
    tmp["customer_id"] = tmp["customer_id"].apply(id_to_int).astype("Int64")
    dmy = pd.get_dummies(tmp[col], prefix=prefix, dtype=np.int8)
    dmy = pd.concat([tmp[["customer_id"]], dmy], axis=1)
    dmy = dmy.groupby("customer_id").max().reset_index()
    dmy["customer_id"] = dmy["customer_id"].astype(np.int32)
    return dmy

def build_customer_features_compatible(df_raw: pd.DataFrame, t0_ts: pd.Timestamp, no_date_filter: bool) -> pd.DataFrame:
    df = df_raw.copy()
    df["customer_id"] = df["customer_id"].apply(id_to_int).astype(np.int32)
    
    available_features = []
    numeric_features = ['item_price']
    for feat in numeric_features:
        if feat in df.columns:
            df[feat] = to_num(df[feat]).fillna(0)
            available_features.append(feat)
    
    categorical_mappings = [
        ('customer_gender', 'customer_gender'),
        ('item_category', 'item_category'), 
        ('item_img_filename', 'item_img_filename'),
    ]
    
    g = df.groupby("customer_id", dropna=True)
    base_features = g[available_features].first().reset_index()
    
    for col, prefix in categorical_mappings:
        if col in df.columns:
            oh_df = _multi_hot(df, col, prefix)
            if oh_df is not None:
                base_features = base_features.merge(oh_df, on="customer_id", how="left")
    
    base_features = base_features.fillna(0)
    base_features["customer_id"] = base_features["customer_id"].astype(int)
    
    return base_features

def align_to_training(feats: pd.DataFrame, feature_names: list):
    ids = feats["customer_id"].values
    X = feats.drop(columns=["customer_id"]).copy()
    for c in feature_names:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_names]
    return ids, X

def calculate_optimal_threshold(probs, target_positive_ratio=0.1):
    """Calcula threshold para obtener aproximadamente target_positive_ratio de predicciones positivas"""
    if len(probs) == 0:
        return 0.5
    
    # Ordenar probabilidades
    sorted_probs = np.sort(probs)[::-1]
    
    # Encontrar threshold que dé aproximadamente target_positive_ratio
    n_positive = int(len(probs) * target_positive_ratio)
    n_positive = max(1, n_positive)  # Al menos 1 positiva
    
    if n_positive < len(sorted_probs):
        optimal_threshold = sorted_probs[n_positive]
    else:
        optimal_threshold = sorted_probs[-1]
    
    # Asegurar que no sea 0
    optimal_threshold = max(optimal_threshold, 0.001)
    
    return optimal_threshold

if __name__ == "__main__":
    args = parse_args()
    T0 = pd.Timestamp(args.t0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Cargar test
    df_test = pd.read_csv(args.raw).copy()
    n_test = len(df_test)
    
    # Detectar ID
    row_id_candidates = ["ID", "Id", "id", "row_id", "Row_ID"]
    row_id_col = next((c for c in row_id_candidates if c in df_test.columns), None)
    if row_id_col is None:
        df_test["ID"] = np.arange(1, n_test + 1, dtype=int)
        row_id_col = "ID"

    if "customer_id" not in df_test.columns:
        sub = pd.DataFrame({"ID": df_test[row_id_col].values, "pred": np.zeros(n_test, dtype=int)})
        sub.to_csv(args.out, index=False)
        print(f"[OK] Predicciones -> {args.out} (filas={len(sub)})")
        raise SystemExit(0)

    df_test["customer_id"] = df_test["customer_id"].apply(id_to_int).astype(np.int32)

    # Cargar modelo
    clf = PurchaseModel.load(args.model)
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feat_names = meta.get("feature_names", [])
    default_thr = float(meta.get("threshold", 0.5))

    # Construir features
    feats_cust = build_customer_features_compatible(df_test, T0, no_date_filter=args.no_date_filter)
    
    if len(feats_cust) == 0:
        sub = pd.DataFrame({"ID": np.arange(n_test), "pred": np.zeros(n_test, dtype=int)})
        sub.to_csv(args.out, index=False)
        print(f"[OK] Predicciones -> {args.out} (filas={len(sub)})")
        raise SystemExit(0)

    cust_ids, Xcust = align_to_training(feats_cust, feat_names)

    # PREDICCIÓN CON THRESHOLD DINÁMICO
    probs = clf.predict_proba(Xcust)[:, 1]
    
    # Determinar threshold
    if args.threshold is not None:
        # Usar threshold explícito del usuario
        final_threshold = args.threshold
        print(f"Usando threshold explícito: {final_threshold:.4f}")
    else:
        # Threshold automático
        final_threshold = calculate_optimal_threshold(probs, target_positive_ratio=0.1)
        print(f"Threshold automático calculado: {final_threshold:.4f}")
        print(f"  (Max prob: {probs.max():.4f}, Target: ~10% positivas)")

    preds = (probs >= final_threshold).astype(int)
    
    # Estadísticas
    print(f"\n=== RESULTADOS ===")
    print(f"Threshold usado: {final_threshold:.4f}")
    print(f"Probabilidades: Min={probs.min():.4f}, Max={probs.max():.4f}, Mean={probs.mean():.4f}")
    
    pred_counts = pd.Series(preds).value_counts().sort_index()
    print(f"Predicciones por cliente:")
    for val, count in pred_counts.items():
        print(f"  Clase {val}: {count} clientes ({count/len(preds)*100:.1f}%)")

    # Propagación
    preds_cust = pd.DataFrame({"customer_id": cust_ids.astype(int), "pred": preds.astype(int)})
    df_join = df_test[[row_id_col, "customer_id"]].merge(preds_cust, on="customer_id", how="left")
    df_join["pred"] = df_join["pred"].fillna(0).astype(int)

    # Exportar
    sub = pd.DataFrame({
        "ID": np.arange(n_test, dtype=int),
        "pred": df_join["pred"].astype(int).values
    })

    final_counts = pd.Series(sub["pred"]).value_counts().sort_index()
    print(f"Predicciones finales:")
    for val, count in final_counts.items():
        print(f"  Clase {val}: {count} filas ({count/len(sub)*100:.1f}%)")

    sub.to_csv(args.out, index=False)
    print(f"[OK] Predicciones -> {args.out}")