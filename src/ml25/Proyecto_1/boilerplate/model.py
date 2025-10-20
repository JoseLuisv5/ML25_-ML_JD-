# model.py — Logistic Regression simple
from pathlib import Path
from datetime import datetime
import os, joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

MODELS_DIR = (Path(__file__).resolve().parent / "trained_models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class PurchaseModel:
    def __init__(self, threshold: float = 0.5, params: dict | None = None):
        self.threshold = float(threshold)
        lr_params = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "random_state": 42}
        if params: lr_params.update(params)
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**lr_params)),
        ])
        self._is_fitted = False
        self.feature_names_ = None

    def fit(self, X, y):
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        y_np = np.asarray(y, dtype=int)
        if hasattr(X, "columns"):
            try: self.feature_names_ = list(X.columns)
            except: self.feature_names_ = None
        self.model.fit(X_np, y_np)
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        self._require_fitted()
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        return self.model.predict_proba(X_np)

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= self.threshold).astype(int)

    def save(self, prefix: str = "purchase_lr"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.abspath(MODELS_DIR / f"{prefix}_{ts}.pkl")  # ← sin la '}' extra
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(filename: str):
        p = Path(filename)
        if not p.exists():
            p = MODELS_DIR / filename
        return joblib.load(p)

    def _require_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Entrena primero con fit(X, y).")
