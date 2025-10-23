import joblib
from typing import Optional, List
from sklearn.linear_model import LogisticRegression

class PurchaseModel:
    def __init__(self, threshold: float = 0.5, params: Optional[dict] = None):
        self.threshold = float(threshold)
        cfg = {"max_iter": 1000}
        if params: cfg.update(params)
        self.model = LogisticRegression(**cfg)
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X, y):
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

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)
