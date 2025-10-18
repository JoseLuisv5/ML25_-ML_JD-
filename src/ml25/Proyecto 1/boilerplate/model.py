# model.py

# Data management
from pathlib import Path
import joblib
from datetime import datetime
import os

# ML
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# XGBoost es opcional
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


CURRENT_FILE = Path(__file__).resolve()
MODELS_DIR = CURRENT_FILE.parent / "trained_models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


class PurchaseModel:
    def __init__(self, *args, **kwargs):
        """
        Mantiene la firma original. Puedes pasar opcionalmente:
          - model_type: "rf" (default) o "xgb"
          - threshold: umbral de decisión para predict() (default 0.5)
          - params: dict de hiperparámetros del estimador
        """
        self.model_type = str(kwargs.get("model_type", "rf")).lower()
        self.threshold = float(kwargs.get("threshold", 0.5))
        self.params = dict(kwargs.get("params", {}))

        if self.model_type == "xgb":
            if not _HAS_XGB:
                raise ImportError("xgboost no está disponible en el entorno.")
            default = dict(
                objective="binary:logistic",
                tree_method="hist",
                eval_metric="aucpr",
                n_estimators=600,
                max_depth=6,
                learning_rate=0.06,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            default.update(self.params)
            self.model = XGBClassifier(**default)
        else:
            # Random Forest por defecto (robusto y simple)
            default = dict(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=10,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42
            )
            default.update(self.params)
            self.model = RandomForestClassifier(**default)

        self._is_fitted = False

    def fit(self, X, y):
        X = self._to_numpy(X)
        y = np.asarray(y).astype(int)

        # Balanceo rápido para XGB si no se pasó scale_pos_weight
        if self.model_type == "xgb" and "scale_pos_weight" not in self.params:
            pos = y.sum()
            neg = len(y) - pos
            spw = (neg / pos) if pos > 0 else 1.0
            try:
                self.model.set_params(scale_pos_weight=spw)
            except Exception:
                pass

        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= self.threshold).astype(int)

    def predict_proba(self, X):
        self._require_fitted()
        X = self._to_numpy(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        # Fallback teórico (RF/XGB sí tienen predict_proba)
        p1 = np.asarray(self.model.predict(X), dtype=float)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def get_config(self):
        """
        Devuelve hiperparámetros clave y metadatos para logging.
        """
        cfg = {"model_type": self.model_type, "threshold": self.threshold}
        try:
            cfg.update(self.model.get_params(deep=False))
        except Exception:
            pass
        return cfg

    def save(self, prefix: str):
        """
        Guarda el wrapper completo (modelo + config) en:
        trained_models/<prefix>_<timestamp>.pkl
        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{now}.pkl"
        filepath = Path(MODELS_DIR) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath = os.path.abspath(filepath)

        joblib.dump(self, filepath)
        print(f"{repr(self)} || Model saved to {filepath}")
        return filepath

    def load(self, filename: str):
        """
        Carga un PurchaseModel desde trained_models/filename y lo retorna.
        """
        filepath = Path(MODELS_DIR) / filename
        model = joblib.load(filepath)
        print(f"{repr(model)} || Model loaded from {filepath}")
        return model

    # -------- utilidades internas --------
    def _require_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("El modelo aún no ha sido entrenado. Llama a fit(X, y) primero.")

    @staticmethod
    def _to_numpy(X):
        if hasattr(X, "values"):  # pandas DataFrame/Series
            return X.values
        return np.asarray(X)

    def __repr__(self):
        name = "XGBClassifier" if self.model_type == "xgb" else "RandomForestClassifier"
        return f"<PurchaseModel {name} thr={self.threshold}>"
