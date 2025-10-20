import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score

from data_processing import read_train_data
from model import PurchaseModel

# 1) Carga datos preprocesados
X, y = read_train_data()

# 2) Split estratificado
Xtr, Xva, ytr, yva = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Modelos candidatos (puedes ampliar el grid de C)
candidates = [
    {"C": 0.5, "solver": "lbfgs"},
    {"C": 1.0, "solver": "lbfgs"},
    {"C": 2.0, "solver": "lbfgs"},
]

thr_grid = np.linspace(0.05, 0.95, 19)
best_balacc, best_t, best_params, best_model = -1.0, 0.5, None, None

for params in candidates:
    clf = PurchaseModel(threshold=0.5, params=params).fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]

    # 4) Elegimos umbral por Balanced Accuracy
    balaccs = np.array([balanced_accuracy_score(yva, (p >= t).astype(int)) for t in thr_grid])
    i_max = int(np.argmax(balaccs))
    balacc_max = float(balaccs[i_max])

    # 4b) Tie-breaker: acercar pos_rate a la prevalencia real
    ties = np.where(balaccs == balacc_max)[0]
    if ties.size > 1:
        rates = np.array([(p >= thr_grid[i]).mean() for i in ties])
        tgt = float(yva.mean())
        i_best = int(ties[np.argmin(np.abs(rates - tgt))])
    else:
        i_best = i_max

    t = float(thr_grid[i_best])

    if balacc_max > best_balacc:
        best_balacc, best_t, best_params, best_model = balacc_max, t, params, clf

# 5) Guardamos umbral y prevalencia de validación en el modelo
best_model.threshold = best_t
best_model.val_prevalence_ = float(yva.mean())

# 6) Logs de diagnóstico
p = best_model.predict_proba(Xva)[:, 1]
pos_rate = float((p >= best_t).mean())
print(f"Valid prevalence (y==1) = {float(yva.mean()):.4f}")
print(f"Probs: min={float(p.min()):.4f}  mean={float(p.mean()):.4f}  max={float(p.max()):.4f}")
print(f"pos_rate@thr={best_t:.2f} = {pos_rate:.4f}")
print(f"Best BAL_ACC: {best_balacc:.4f}  @thr={best_t:.2f}  params={best_params}")

# 7) Métricas extra informativas
try:
    print(f"ROC AUC: {roc_auc_score(yva, p):.4f}")
    print(f"PR  AUC: {average_precision_score(yva, p):.4f}")
except Exception:
    pass

# 8) Guardado
print("Modelo guardado en:", best_model.save(prefix="purchase_lr"))