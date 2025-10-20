import numpy as np
from data_processing import read_train_data
from model import PurchaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score

# Carga
X, y = read_train_data()

# Split estratificado
Xtr, Xva, ytr, yva = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Probar unos C simples (puedes ampliar si quieres)
candidates = [
    {"C": 0.5, "solver": "lbfgs"},
    {"C": 1.0, "solver": "lbfgs"},
    {"C": 2.0, "solver": "lbfgs"},
]

best_balacc, best_t, best_params, best_model = -1.0, 0.5, None, None
thr_grid = np.linspace(0.05, 0.95, 19)

for params in candidates:
    # Nota: PurchaseModel ya trae class_weight="balanced"
    clf = PurchaseModel(threshold=0.5, params=params).fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]

    # Elegimos el umbral por BALANCED ACCURACY
    balaccs = [balanced_accuracy_score(yva, (p >= t).astype(int)) for t in thr_grid]
    i = int(np.argmax(balaccs))
    balacc, t = float(balaccs[i]), float(thr_grid[i])

    if balacc > best_balacc:
        best_balacc, best_t, best_params, best_model = balacc, t, params, clf

# Seteamos el mejor umbral
best_model.threshold = best_t

# Logs útiles para evitar "todo 1"
p = best_model.predict_proba(Xva)[:, 1]
pos_rate = float((p >= best_t).mean())
print(f"Valid prevalence (y==1) = {float(yva.mean()):.4f}")
print(f"Probs: min={float(p.min()):.4f}  mean={float(p.mean()):.4f}  max={float(p.max()):.4f}")
print(f"pos_rate@thr={best_t:.2f} = {pos_rate:.4f}")

# Métricas extra informativas (no influyen en la selección)
print(f"Best BAL_ACC: {best_balacc:.4f}  @thr={best_t:.2f}  params={best_params}")
try:
    print(f"ROC AUC: {roc_auc_score(yva, p):.4f}")
    print(f"PR  AUC: {average_precision_score(yva, p):.4f}")
except Exception:
    pass

print("Modelo guardado en:", best_model.save(prefix="purchase_lr"))
