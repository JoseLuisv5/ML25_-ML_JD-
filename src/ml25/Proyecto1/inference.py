# inference.py (robusto y compatible con training)
import argparse
import os
import json
import numpy as np
import pandas as pd

from utils import to_dt, to_num, id_to_int
from model import PurchaseModel


# --- Paths por defecto ---
DEF_RAW   = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Archivos base\customer_purchases_test.csv"
DEF_MODEL = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\model_lr.pkl"
DEF_META  = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\model_lr_meta.json"
DEF_OUT   = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\preds_lr_scoring.csv"
DEF_T0    = "2025-09-21"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw",   type=str, default=DEF_RAW)
    p.add_argument("--model", type=str, default=DEF_MODEL)
    p.add_argument("--meta",  type=str, default=DEF_META)
    p.add_argument("--out",   type=str, default=DEF_OUT)
    p.add_argument("--t0",    type=str, default=DEF_T0)
    # si True, no filtra por fecha (solo si tu test no tiene timestamps)
    p.add_argument("--no_date_filter", action="store_true", default=False)
    return p.parse_args()


def _multi_hot(df, col, prefix):
    """One-hot por cliente con prefijo dado (misma lógica que training)."""
    if col not in df.columns:
        return None
    tmp = df[["customer_id", col]].copy()
    tmp["customer_id"] = tmp["customer_id"].apply(id_to_int).astype("Int64")
    dmy = pd.get_dummies(tmp[col].astype(str).str.lower(), prefix=prefix, dtype=np.int8)
    dmy = pd.concat([tmp[["customer_id"]], dmy], axis=1).groupby("customer_id").max().reset_index()
    dmy["customer_id"] = dmy["customer_id"].astype(np.int32)
    return dmy


def build_features_for_inference(raw_path: str, t0_ts: pd.Timestamp, no_date_filter: bool) -> pd.DataFrame:
    """Construye la misma matriz numérica usada en training, pero desde el CSV crudo."""
    df_raw = pd.read_csv(raw_path)

    # Casts
    df_raw["purchase_timestamp"]     = to_dt(df_raw.get("purchase_timestamp"))
    df_raw["customer_signup_date"]   = to_dt(df_raw.get("customer_signup_date"))
    df_raw["customer_date_of_birth"] = to_dt(df_raw.get("customer_date_of_birth"))
    df_raw["item_price"]             = to_num(df_raw.get("item_price"))
    df_raw["customer_item_views"]    = to_num(df_raw.get("customer_item_views")).fillna(0)
    df_raw["customer_id"]            = df_raw["customer_id"].apply(id_to_int).astype(np.int32)

    # Info de depuración
    n_raw = len(df_raw)
    has_ts = bool(df_raw["purchase_timestamp"].notna().any())
    print(f"[INFO] raw filas: {n_raw} | con timestamp válido: {int(df_raw['purchase_timestamp'].notna().sum())}")

    # Filtro temporal
    if no_date_filter or not has_ts:
        if not has_ts:
            print("[WARN] No hay purchase_timestamp parseables; se usará TODO el archivo sin filtrar por fecha.")
        else:
            print("[INFO] no_date_filter=True -> no se filtra por T0.")
        df = df_raw.copy()
    else:
        df = df_raw[df_raw["purchase_timestamp"].notna() & (df_raw["purchase_timestamp"] <= t0_ts)].copy()
        print(f"[INFO] <=T0 filas: {len(df)}")
        if len(df) == 0:
            print("[WARN] No hay historial <=T0; se usará TODO el historial con timestamp válido.")
            df = df_raw[df_raw["purchase_timestamp"].notna()].copy()
            print(f"[INFO] fallback filas: {len(df)}")

    if len(df) == 0:
        # último fallback: usa todo el archivo (aunque no sirva temporalmente), para no dejar vacío
        print("[WARN] DataFrame quedó vacío tras filtros. Usando df_raw completo.")
        df = df_raw.copy()

    # Agregación por cliente (tolerante a NaT)
    g = df.groupby("customer_id", dropna=True)
    base = g.agg(
        signup_min   = ("customer_signup_date", "min"),
        dob_min      = ("customer_date_of_birth", "min"),
        last_purchase= ("purchase_timestamp", "max"),
        visitas      = ("customer_item_views", "sum"),
        gasto_total  = ("item_price", "sum"),
        compras      = ("purchase_timestamp", "size"),
    ).reset_index()

    # Si no hay last_purchase (todo NaT), usa signup_min para calcular días desde "última compra"
    if base["last_purchase"].isna().all():
        base["last_purchase"] = base["signup_min"]

    base["antiguedad_dias"] = (t0_ts - base["signup_min"]).dt.days
    base["edad_anios"] = ((t0_ts - base["dob_min"]).dt.days / 365.25).round(2)
    base["dias_desde_ultima_compra"] = (t0_ts - base["last_purchase"]).dt.days

    # Si no hay timestamps válidos, fuerza compras=0 (para que la semántica sea consistente)
    if not has_ts:
        base["compras"] = 0

    base["gasto_pct"] = base["gasto_total"].rank(pct=True) * 100.0

    # One-hots (mismos prefijos que en training)
    genero_oh = _multi_hot(df, "customer_gender", "genero")
    device_oh = _multi_hot(df, "purchase_device", "device")
    cat_oh    = _multi_hot(df, "item_category",   "cat")

    feats = base[[
        "customer_id", "antiguedad_dias", "edad_anios",
        "dias_desde_ultima_compra", "visitas", "gasto_pct", "compras"
    ]].copy()

    for blk in (genero_oh, device_oh, cat_oh):
        if blk is not None:
            blk["customer_id"] = blk["customer_id"].astype(np.int32)
            feats = feats.merge(blk, on="customer_id", how="left")

    feats = feats.fillna(0)
    feats["customer_id"] = feats["customer_id"].astype(int)

    print(f"[INFO] clientes en base: {len(base)} | clientes con feats: {len(feats)}")
    return feats


def align_to_training(feats: pd.DataFrame, feature_names: list):
    """Alinea columnas a las usadas en training, creando faltantes con 0 y respetando orden."""
    if feats is None or len(feats) == 0:
        return np.array([]), pd.DataFrame(columns=feature_names)
    ids = feats["customer_id"].values
    X = feats.drop(columns=["customer_id"]).copy()
    for c in feature_names:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_names]
    return ids, X


if __name__ == "__main__":
    args = parse_args()
    T0 = pd.Timestamp(args.t0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    feats = build_features_for_inference(args.raw, T0, no_date_filter=args.no_date_filter)
    if feats is None or len(feats) == 0:
        print("[WARN] No se generaron features. Saliendo sin predicciones.")
        pd.DataFrame(columns=["customer_id", "prob", "pred"]).to_csv(args.out, index=False)
        raise SystemExit(0)

    clf: PurchaseModel = PurchaseModel.load(args.model)
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    ids, X = align_to_training(feats, meta.get("feature_names", []))
    if X.shape[0] == 0:
        print("[WARN] No hay clientes alineados a las columnas de entrenamiento. Saliendo sin predicciones.")
        pd.DataFrame(columns=["customer_id", "prob", "pred"]).to_csv(args.out, index=False)
        raise SystemExit(0)

    probs = clf.predict_proba(X)[:, 1]
    thr = float(meta.get("threshold", 0.5))
    preds = (probs >= thr).astype(int)

# SOLO las columnas que quieres:
    out = pd.DataFrame({"ID": ids, "pred": preds})
    out.to_csv(args.out, index=False)
    print(f"[OK] Predicciones -> {args.out} (filas={len(out)})")
