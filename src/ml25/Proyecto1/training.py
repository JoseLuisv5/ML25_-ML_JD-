# training.py (imprime Accuracy en terminal)
import argparse, os, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from model import PurchaseModel

# Paths por defecto
DEF_LABELED = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\out_features_agg\train\train_features_labeled.csv"
DEF_OUTDIR  = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labeled", type=str, default=DEF_LABELED)
    p.add_argument("--outdir",  type=str, default=DEF_OUTDIR)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Carga dataset etiquetado
    df = pd.read_csv(args.labeled)
    y  = df["label"].astype(int).values
    X  = df.drop(columns=["label","customer_id"])  # solo features numéricas

    # Split
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # Modelo (simple .fit)
    model = PurchaseModel(threshold=0.5)
    model.fit(Xtr, ytr)

    # Predicción en validación
    pva  = model.predict_proba(Xva)[:, 1]
    yhat = (pva >= 0.5).astype(int)

    # Métricas
    acc  = accuracy_score(yva, yhat)
    bacc = balanced_accuracy_score(yva, yhat)
    cm   = confusion_matrix(yva, yhat)

    # >>>> IMPRESIONES EN TERMINAL <<<<
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    # Guardado de modelo y metadatos
    model_path = os.path.join(args.outdir, "model_lr.pkl")
    meta_path  = os.path.join(args.outdir, "model_lr_meta.json")
    model.save(model_path)
    meta = {"feature_names": X.columns.tolist(), "threshold": 0.5}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Modelo -> {model_path}")
    print(f"[OK] Meta   -> {meta_path}")
