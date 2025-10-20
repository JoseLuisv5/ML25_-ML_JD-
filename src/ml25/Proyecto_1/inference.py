import sys, joblib, pandas as pd, numpy as np
from pathlib import Path

PROYECTO_1 = Path(r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto_1")
BOILER     = PROYECTO_1 / "boilerplate"
MODELS_DIR = BOILER / "trained_models"
RESULTS_DIR = PROYECTO_1 / "test_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BOILER))
from data_processing import read_test_data, read_train_data

def _latest_model_name():
    fs = sorted(MODELS_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return fs[0].name if fs else ""

def load_model(filename: str):
    p = MODELS_DIR / filename if not Path(filename).exists() else Path(filename)
    return joblib.load(p)

def _align_columns(X: pd.DataFrame, model):
    core = getattr(model, "model", model)
    names = getattr(core, "feature_names_in_", None)
    if names is None:
        return X
    return X.reindex(columns=list(names), fill_value=0)

if __name__ == "__main__":
    name = _latest_model_name()
    if not name:
        raise FileNotFoundError(f"No hay modelos en {MODELS_DIR}")
    m = load_model(name)

    Xtr, ytr = read_train_data()
    pos_rate_tr = float((ytr.values if hasattr(ytr,"values") else ytr).mean())

    Xte = read_test_data()
    Xte = _align_columns(Xte, m)

    probs = m.predict_proba(Xte)[:, 1]

    thr = float(getattr(m, "threshold", 0.5))
    thr = 0.5 if not (0.0 <= thr <= 1.0) or np.isnan(thr) else thr
    preds = (probs >= thr).astype(int)

    # recalibra solo si está casi todo en 1
    rate = float(np.mean(preds))
    if rate > 0.95 and 0.0 < pos_rate_tr < 1.0:
        thr = float(np.quantile(probs, 1.0 - pos_rate_tr))
        preds = (probs >= thr).astype(int)

    pd.DataFrame({"ID": Xte.index, "pred": probs}).to_csv(RESULTS_DIR / "sub_prob.csv",
                                                          index=False, float_format="%.6f")
    pd.DataFrame({"ID": Xte.index, "pred": preds}).to_csv(RESULTS_DIR / "sub_bin.csv",
                                                          index=False)

    print(f"Modelo: {name}")
    print(f"thr_final={thr:.4f}  prob_min={float(np.min(probs)):.4f}  prob_max={float(np.max(probs)):.4f}")
    print(f"pred_rate_ones={float(np.mean(preds)):.4f}")
    print("Guardado: test_results/sub_prob.csv y sub_bin.csv")
