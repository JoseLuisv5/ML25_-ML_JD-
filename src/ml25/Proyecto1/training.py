# training.py (imprime Accuracy en terminal + chequeo de overfitting)
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from model import PurchaseModel

# --------------------------
# Chequeo de overfitting
# --------------------------
def _simple_score(model, X, y):
    """Accuracy usando el umbral del modelo (o 0.5). Soporta modelos sin .score()."""
    thr = getattr(model, "threshold", 0.5)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
        yhat = (p >= thr).astype(int)
    elif hasattr(model, "predict"):
        yhat = model.predict(X)
    else:
        raise AttributeError("El modelo no tiene predict_proba ni predict.")
    return accuracy_score(y, yhat)

def check_and_prevent_overfitting(model, X_train, X_val, y_train, y_val, X_test=None):
    print("\n" + "="*60)
    print("ANALIZANDO OVERFITTING")
    print("="*60)
    train_score = _simple_score(model, X_train, y_train)
    val_score   = _simple_score(model, X_val,   y_val)
    difference  = train_score - val_score

    print(f"   -Score en train: {train_score:.4f}")
    print(f"   -Score en validation: {val_score:.4f}")
    print(f"   -Diferencia: {difference:.4f}")

    if difference > 0.05:
        print("  POSIBLE OVERFITTING DETECTADO :O")
        return True
    elif difference > 0.02:
        print("  Pequeña diferencia, monitorear :/")
        return False
    else:
        print("  No se detecta overfitting significativo :D")
        return False

# --------------------------
# Paths por defecto
# --------------------------
DEF_LABELED = r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\Proyecto1\out_features_agg\train\train_features_labeled.csv"
DEF_OUTDIR  = r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\Proyecto1"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labeled", type=str, default=DEF_LABELED)
    p.add_argument("--outdir",  type=str, default=DEF_OUTDIR)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Carga dataset etiquetado
    df = pd.read_csv(args.labeled)
    y  = df["label"].astype(int).values
    # Solo features (si hay texto/categorías, conviértelas antes o elimínalas aquí)
    X  = df.drop(columns=["label","customer_id"])

    # 2) Split
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # 3) Modelo (simple .fit)
    thr = 0.5
    model = PurchaseModel(threshold=thr)
    model.fit(Xtr, ytr)

    # 4) Predicción en validación
    if hasattr(model, "predict_proba"):
        pva  = model.predict_proba(Xva)[:, 1]
        yhat = (pva >= thr).astype(int)
    else:
        yhat = model.predict(Xva)

    # 5) Métricas
    acc  = accuracy_score(yva, yhat)
    bacc = balanced_accuracy_score(yva, yhat)
    cm   = confusion_matrix(yva, yhat)

    # >>>> IMPRESIONES EN TERMINAL <<<<
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    # 6) Chequeo de overfitting (usando el mismo umbral)
    _ = check_and_prevent_overfitting(model, Xtr, Xva, ytr, yva)

    # 7) Guardado de modelo y metadatos
    model_path = os.path.join(args.outdir, "model_lr.pkl")
    meta_path  = os.path.join(args.outdir, "model_lr_meta.json")
    model.save(model_path)
    meta = {
        "feature_names": X.columns.tolist(),
        "threshold": float(thr),
        "val_accuracy": float(acc),
        "val_balanced_accuracy": float(bacc)
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Modelo -> {model_path}")
    print(f"[OK] Meta   -> {meta_path}")
