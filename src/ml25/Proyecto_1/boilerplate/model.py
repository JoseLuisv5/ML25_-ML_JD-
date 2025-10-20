from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Carpeta donde se guardan los modelos .pkl
MODELS_DIR = (Path(__file__).resolve().parent / "trained_models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class PurchaseModel:
    def _init_(self, threshold: float = 0.5, params: Optional[Dict] = None):
        self.threshold = float(threshold)
        # Sin doble escalado; el preprocesador ya estandariza. Balanceo por clase.
        p = {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
            "class_weight": "balanced",
        }
        if params:
            p.update(params)
        self.model = LogisticRegression(**p)
        self.feature_names_ = None
        self.val_prevalence_ = None  # se setea en training

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"f{i}" for i in range(X.shape[1])]
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= self.threshold).astype(int)

    def save(self, prefix: str = "purchase_lr"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.abspath(MODELS_DIR / f"{prefix}_{ts}.pkl")
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(filename: str):
        p = Path(filename)
        if not p.exists():
            p = MODELS_DIR / filename
        return joblib.load(p)