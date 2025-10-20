# training.py - Logistic Regression simple
import sys, numpy as np
sys.path.insert(0, r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto_1\boilerplate")

from data_processing import read_train_data
from model import PurchaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = read_train_data()
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

candidates = [
    {"C": 0.5, "solver": "lbfgs"},
    {"C": 1.0, "solver": "lbfgs"},
    {"C": 2.0, "solver": "lbfgs"},
]

best_acc, best_t, best_params, best_model = -1.0, 0.5, None, None
thr_grid = np.linspace(0.05, 0.95, 19)

for params in candidates:
    clf = PurchaseModel(threshold=0.5, params=params).fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]
    accs = [accuracy_score(yva, (p >= t).astype(int)) for t in thr_grid]
    i = int(np.argmax(accs))
    acc, t = float(accs[i]), float(thr_grid[i])
    if acc > best_acc:
        best_acc, best_t, best_params, best_model = acc, t, params, clf

best_model.threshold = best_t
print(f"Best ACC: {best_acc:.4f}  @thr={best_t:.2f}  params={best_params}")
print("Modelo guardado en:", best_model.save(prefix="purchase_lr"))
