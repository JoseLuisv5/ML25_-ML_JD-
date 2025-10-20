import sys, joblib, pandas as pd, numpy as np
from pathlib import Path

CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parent
BOILER = ROOT / "boilerplate"
MODELS_DIR = BOILER / "trained_models"
RESULTS_DIR = ROOT / "test_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BOILER))
from data_processing import read_test_data

def _latest_model_name():
    fs = sorted(MODELS_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return fs[0].name if fs else ""

def load_model(filename: str):
    p = MODELS_DIR / filename if not Path(filename).exists() else Path(filename)
    return joblib.load(p)

def _align_columns(X: pd.DataFrame, model):
    names = getattr(model, "feature_names_", None)
    if names is None:
        core = getattr(model, "model", model)
        names = getattr(core, "feature_names_in_", None)
    if names is None:
        return X
    return X.reindex(columns=list(names), fill_value=0)

if __name__ == "__main__":
    name = _latest_model_name()
    if not name: raise FileNotFoundError(f"No hay modelos en {MODELS_DIR}")
    m = load_model(name)
    Xte = read_test_data()
    Xte = _align_columns(Xte, m)
    probs = m.predict_proba(Xte)[:, 1]
    thr = float(getattr(m, "threshold", 0.5))
    if not (0.0 <= thr <= 1.0) or np.isnan(thr): thr = 0.5
    preds = (probs >= thr).astype(int)
    pd.DataFrame({"ID": Xte.index, "pred": probs}).to_csv(RESULTS_DIR / "sub_prob.csv", index=False, float_format="%.6f")
    pd.DataFrame({"ID": Xte.index, "pred": preds}).to_csv(RESULTS_DIR / "sub_bin.csv", index=False)
    print("Guardado:", RESULTS_DIR / "sub_prob.csv")
    print("Guardado:", RESULTS_DIR / "sub_bin.csv")
    print("Modelo usado:", name)
