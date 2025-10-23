# training_fixed_v2.py
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
import joblib

class PurchaseModel:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        # ESTRATEGIA MÁS AGRESIVA PARA DESBALANCE EXTREMO
        self.model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=2000,
            solver='liblinear',
            C=0.1,  # Regularización más fuerte
            penalty='l2'
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict(self, X):
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)
    
    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
    
    def save(self, path):
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path):
        return joblib.load(path)

def find_optimal_threshold(model, X_val, y_val):
    """Encuentra el threshold óptimo para maximizar balanced accuracy"""
    probas = model.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.5
    best_bacc = 0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (probas >= threshold).astype(int)
        bacc = balanced_accuracy_score(y_val, y_pred)
        if bacc > best_bacc:
            best_bacc = bacc
            best_threshold = threshold
    
    print(f"Threshold óptimo encontrado: {best_threshold:.3f} (Balanced Acc: {best_bacc:.4f})")
    return best_threshold

def prepare_features(df):
    """Prepara características eliminando columnas problemáticas"""
    
    # 1. Eliminar columnas siempre vacías
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    if empty_cols:
        print(f"Eliminando columnas vacías: {empty_cols}")
        df = df.drop(columns=empty_cols)
    
    # 2. POSIBLE FUGA: Eliminar customer_cat_is_prefered
    if 'customer_cat_is_prefered' in df.columns:
        print("⚠️  Eliminando customer_cat_is_prefered (posible fuga de datos)")
        df = df.drop(columns=['customer_cat_is_prefered'])
    
    # 3. Eliminar IDs
    id_cols = [c for c in ["purchase_id", "customer_id", "item_id"] if c in df.columns]
    if id_cols:
        df = df.drop(columns=id_cols)
    
    return df

# Paths por defecto
DEF_TRAIN = r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\datasets\customer_purchases\customer_purchases_train_with_ids.csv"
DEF_OUT   = r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\Proyecto1"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default=DEF_TRAIN)
    p.add_argument("--outdir", type=str, default=DEF_OUT)
    p.add_argument("--threshold", type=float, default=None)  # None para auto-detect
    p.add_argument("--folds", type=int, default=5)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Cargar y preparar datos
    df = pd.read_csv(args.train_csv)
    if "label" not in df.columns:
        raise ValueError("El CSV de entrenamiento debe traer columna 'label'.")

    print("=== DISTRIBUCIÓN DE CLASES ORIGINAL ===")
    print(df['label'].value_counts())
    print("Proporciones:", df['label'].value_counts(normalize=True))

    # Preparar features
    y = df["label"].astype(int).values
    X = prepare_features(df.drop(columns=['label']))

    # Limpiar datos
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)

    print(f"\nFeatures finales: {X.shape[1]}")
    print(f"Muestras: {X.shape[0]}")

    # 2) Validación cruzada con threshold automático
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    
    accs, baccs, roc_aucs, optimal_thresholds = [], [], [], []
    cm_total = np.array([[0, 0], [0, 0]], dtype=int)
    fold_idx = 0
    
    for tr_idx, va_idx in skf.split(X, y):
        fold_idx += 1
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        # Modelo
        model = PurchaseModel(threshold=0.5)  # Temporal
        model.fit(Xtr, ytr)

        # Encontrar threshold óptimo para este fold
        optimal_threshold = find_optimal_threshold(model, Xva, yva)
        optimal_thresholds.append(optimal_threshold)
        
        # Usar threshold óptimo para predicciones
        model.threshold = optimal_threshold
        pva = model.predict_proba(Xva)[:, 1]
        yhat = (pva >= optimal_threshold).astype(int)

        # Métricas
        acc = accuracy_score(yva, yhat)
        bacc = balanced_accuracy_score(yva, yhat)
        roc_auc = roc_auc_score(yva, pva)
        cm = confusion_matrix(yva, yhat, labels=[0,1])

        accs.append(acc); baccs.append(bacc); roc_aucs.append(roc_auc)
        cm_total += cm

        print(f"\n--- Fold {fold_idx}/{args.folds} ---")
        print(f"Accuracy:          {acc:.4f}")
        print(f"Balanced Accuracy: {bacc:.4f}")
        print(f"ROC AUC:           {roc_auc:.4f}")
        print(f"Threshold usado:   {optimal_threshold:.3f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(yva, yhat))

    # Threshold final (promedio de los óptimos)
    final_threshold = np.mean(optimal_thresholds) if args.threshold is None else args.threshold
    print(f"\nThreshold final seleccionado: {final_threshold:.3f}")

    # Resumen CV
    print("\n" + "="*60)
    print("RESUMEN VALIDACIÓN CRUZADA")
    print("="*60)
    print(f"Accuracy (mean ± std):          {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Balanced Accuracy (mean ± std): {np.mean(baccs):.4f} ± {np.std(baccs):.4f}")
    print(f"ROC AUC (mean ± std):           {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    print(f"Optimal thresholds: {[f'{t:.3f}' for t in optimal_thresholds]}")
    print("Confusion Matrix ACUMULADA:")
    print(cm_total)

    # 3) Modelo final con threshold óptimo
    final_model = PurchaseModel(threshold=final_threshold)
    final_model.fit(X, y)

    model_path = os.path.join(args.outdir, "model_lr_balanced.pkl")
    meta_path = os.path.join(args.outdir, "model_lr_meta_balanced.json")
    final_model.save(model_path)

    # Meta data (arreglando el error de serialización)
    meta = {
        "feature_names": X.columns.tolist(),
        "threshold": float(final_threshold),
        "train_shape": [int(X.shape[0]), int(X.shape[1])],
        "class_distribution": {int(k): int(v) for k, v in pd.Series(y).value_counts().items()},
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs)),
        "cv_bal_acc_mean": float(np.mean(baccs)),
        "cv_bal_acc_std": float(np.std(baccs)),
        "cv_roc_auc_mean": float(np.mean(roc_aucs)),
        "cv_roc_auc_std": float(np.std(roc_aucs)),
        "optimal_thresholds": [float(t) for t in optimal_thresholds],
        "cm_total": cm_total.astype(int).tolist(),
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Modelo final guardado -> {model_path}")
    print(f"[OK] Meta guardada -> {meta_path}")
    print(f"[OK] Threshold final: {final_threshold:.3f}")