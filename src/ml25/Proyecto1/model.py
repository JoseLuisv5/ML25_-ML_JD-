# model.py
import joblib
from typing import Optional, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class PurchaseModel:
    def __init__(self, threshold: float = 0.5, params: Optional[dict] = None):
        self.threshold = float(threshold)
        cfg = {"max_iter": 1000}
        if params:
            cfg.update(params)
        self.model = LogisticRegression(**cfg)
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X, y):
        # Guarda nombres de columnas para trazabilidad (si X es DataFrame)
        try:
            self.feature_names_ = list(X.columns)
        except Exception:
            self.feature_names_ = None
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= self.threshold).astype(int)

    def score(self, X, y):
        """
        Accuracy usando el UMBRAL del modelo (self.threshold),
        para ser compatible con check_and_prevent_overfitting(model, ...).
        """
        yhat = self.predict(X)
        if isinstance(y, (list, tuple)):
            y = np.asarray(y)
        return accuracy_score(y, yhat)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)
