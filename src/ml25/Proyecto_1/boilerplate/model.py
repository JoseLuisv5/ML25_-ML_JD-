# model.py
from pathlib import Path
from datetime import datetime
import os, joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

MODELS_DIR = (Path(__file__).resolve().parent / "trained_models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class PurchaseModel:
    def __init__(self, threshold: float = 0.5, params: dict | None = None):
        self.threshold = float(threshold)
        default = dict(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
        if params: default.update(params)
        self.model = RandomForestClassifier(**default)
        self._is_fitted = False

    def fit(self, X, y):
        X = self._to_numpy(X); y = np.asarray(y, dtype=int)
        self.model.fit(X, y); self._is_fitted = True
        return self

    def predict_proba(self, X):
        self._require_fitted()
        return self.model.predict_proba(self._to_numpy(X))

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= self.threshold).astype(int)

    def save(self, prefix: str = "purchase_rf"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.abspath(MODELS_DIR / f"{prefix}_{ts}.pkl")
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(filename: str):
        p = Path(filename)
        if not p.exists(): p = MODELS_DIR / filename
        return joblib.load(p)

    @staticmethod
    def _to_numpy(X):
        return X.values if hasattr(X, "values") else np.asarray(X)

    def _require_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Entrena primero con fit(X, y).")
