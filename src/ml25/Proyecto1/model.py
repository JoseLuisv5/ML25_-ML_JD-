# model.py  (Random Forest simple y robusto)
import joblib
from sklearn.ensemble import RandomForestClassifier

class PurchaseModel:
    def __init__(self, threshold=0.5, params=None):
        self.threshold = threshold
        base_params = dict(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
            class_weight="balanced_subsample",  # ayuda si hay desbalance
        )
        if params:
            base_params.update(params)
        self.model = RandomForestClassifier(**base_params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)

    @staticmethod
    def load(path):
        m = PurchaseModel()
        m.model = joblib.load(path)
        return m
