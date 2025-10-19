# training.py — mini tuner para subir accuracy sin tocar otros archivos
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\Proyecto_1\boilerplate")

from data_processing import read_train_data
from model import PurchaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = read_train_data()
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# espacio MUY pequeño de búsqueda (rápido)
candidates = [
    {"n_estimators": 400, "min_samples_leaf": 5, "max_depth": None, "max_features": "sqrt", "class_weight":"balanced"},
    {"n_estimators": 800, "min_samples_leaf": 3, "max_depth": 20,   "max_features": "sqrt", "class_weight":"balanced"},
    {"n_estimators":1200, "min_samples_leaf": 3, "max_depth": None, "max_features": 0.5,    "class_weight":"balanced"},
    {"n_estimators": 800, "min_samples_leaf": 1, "max_depth": 16,   "max_features": "sqrt", "class_weight":"balanced"},
]

best_acc, best_t, best_params, best_model = -1.0, 0.5, None, None
thr_grid = np.linspace(0.05, 0.95, 19)

for params in candidates:
    clf = PurchaseModel(threshold=0.5, params=params).fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]
    # umbral que maximiza accuracy
    accs = [accuracy_score(yva, (p >= t).astype(int)) for t in thr_grid]
    t = float(thr_grid[int(np.argmax(accs))])
    acc = float(max(accs))
    if acc > best_acc:
        best_acc, best_t, best_params, best_model = acc, t, params, clf

best_model.threshold = best_t
print(f"Best ACC: {best_acc:.4f}  @thr={best_t:.2f}  params={best_params}")
print("Modelo guardado en:", best_model.save(prefix="purchase_rf"))
