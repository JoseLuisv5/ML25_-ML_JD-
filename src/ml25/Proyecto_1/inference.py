import sys
import pandas as pd
import numpy as np
from pathlib import Path

CURRENT = Path(_file_).resolve()
ROOT = CURRENT.parent
BOILER = ROOT / "boilerplate"
MODELS_DIR = BOILER / "trained_models"
RESULTS_DIR = ROOT / "test_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Para runtime y para que Pylance encuentre los módulos
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BOILER))

from data_processing import read_test_data
from model import PurchaseModel

def _latest_model_name():
    fs = sorted(MODELS_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return fs[0].name if fs else ""

def _align_columns(X: pd.DataFrame, model):
    names = getattr(model, "feature_names_", None)
    if names is None:
        core = getattr(model, "model", model)
        names = getattr(core, "feature_names_in_", None)
    if names is None:
        return X
    return X.reindex(columns=list(names), fill_value=0)

def _safe_rate_target(model) -> float:
    """Tasa objetivo de positivos basada en la prevalencia de validación, acotada [0.05, 0.95]."""
    val_prev = getattr(model, "val_prevalence_", None)
    if val_prev is None or not (0.0 < float(val_prev) < 1.0):
        return 0.5
    return float(np.clip(val_prev, 0.05, 0.95))

def _retune_threshold_if_extreme(probs: np.ndarray, thr: float, model) -> float:
    """Si todo sale 1/0, recalibra el umbral para aproximar la tasa objetivo (cuantil)."""
    preds = (probs >= thr).astype(int)
    rate = float(preds.mean())
    if rate >= 0.995 or rate <= 0.005:
        tgt = _safe_rate_target(model)
        # umbral = cuantil (1 - tgt) de las probabilidades
        thr_new = float(np.quantile(probs, 1.0 - tgt))
        return thr_new
    return thr

if _name_ == "_main_":
    name = _latest_model_name()
    if not name:
        raise FileNotFoundError(f"No hay modelos en {MODELS_DIR}")

    model = PurchaseModel.load(name)

    Xte = read_test_data()
    Xte = _align_columns(Xte, model)

    # Diagnóstico rápido del input
    try:
        nz_cols = int((Xte != 0).any(axis=0).sum())
        print(f"Xte shape = {Xte.shape}, columnas_no_constantes = {nz_cols}")
    except Exception:
        pass

    probs = model.predict_proba(Xte)[:, 1]
    thr = float(getattr(model, "threshold", 0.5))
    if not (0.0 <= thr <= 1.0) or np.isnan(thr):
        thr = 0.5

    # Si las probs son constantes (p. ej., todas 1.0), avisamos y hacemos fallback por ranking
    if float(probs.min()) == float(probs.max()):
        print(f"[WARN] Todas las probabilidades son constantes = {float(probs[0]):.4f}. "
              f"Revisa alineación de columnas/preprocesamiento. Aplicando fallback por ranking.")
        tgt = _safe_rate_target(model)
        k = int(round(tgt * len(probs)))
        order = np.argsort(-probs)  # mayor prob primero
        preds = np.zeros_like(probs, dtype=int)
        preds[order[:k]] = 1
    else:
        # Guard-rail si todo sale 1/0 con el umbral actual
        thr2 = _retune_threshold_if_extreme(probs, thr, model)
        if thr2 != thr:
            print(f"[WARN] Ajuste de threshold: {thr:.3f} -> {thr2:.3f} para evitar todo 1/0.")
            thr = thr2
        preds = (probs >= thr).astype(int)

    pos_rate = float((preds == 1).mean())
    print(f"Modelo usado: {name}")
    print(f"thr={thr:.3f}  pos_rate={pos_rate:.4f}  prob_min={float(probs.min()):.4f}  prob_mean={float(probs.mean()):.4f}  prob_max={float(probs.max()):.4f}")

    # Mantengo tus IDs como el índice actual (si después quieres customer_id real lo cambiamos)
    pd.DataFrame({"ID": Xte.index, "pred": probs}).to_csv(RESULTS_DIR / "sub_prob.csv", index=False, float_format="%.6f")
    pd.DataFrame({"ID": Xte.index, "pred": preds}).to_csv(RESULTS_DIR / "sub_bin.csv", index=False)

    print("Guardado:", RESULTS_DIR / "sub_prob.csv")
    print("Guardado:", RESULTS_DIR / "sub_bin.csv")