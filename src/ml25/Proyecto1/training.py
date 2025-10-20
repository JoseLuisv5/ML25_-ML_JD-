# training.py
import pandas as pd, numpy as np, json, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

RAW    = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Archivos base\customer_purchases_train.csv"
FEAT   = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\train_features_per_customer.csv"
META_T = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\timing_meta.json"
OUTM   = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\model_rf.pkl"
OUTJ   = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\model_meta.json"

def load_T0_from_meta_or_raw(raw_df: pd.DataFrame) -> str:
    if os.path.exists(META_T):
        try:
            meta = json.load(open(META_T, "r", encoding="utf-8"))
            if "T0" in meta and meta["T0"]:
                return str(meta["T0"])
        except:
            pass
    tmp = raw_df.copy()
    tmp["purchase_timestamp"] = pd.to_datetime(tmp["purchase_timestamp"], errors="coerce")
    return str(tmp["purchase_timestamp"].max())

def make_labeled(feats: pd.DataFrame, raw: pd.DataFrame, T0_str: str) -> pd.DataFrame:
    T0 = pd.Timestamp(T0_str)
    r = raw.copy()
    r["purchase_timestamp"] = pd.to_datetime(r["purchase_timestamp"], errors="coerce")
    buyers = r[
        (r["purchase_timestamp"].notna()) &
        (r["purchase_timestamp"] > T0) &
        (r["purchase_timestamp"] <= T0 + pd.Timedelta(days=30))
    ]["customer_id"].astype(str).unique().tolist()
    f = feats.copy()
    f["customer_id"] = f["customer_id"].astype(str)
    f["label"] = f["customer_id"].isin(buyers).astype(int)
    return f

if __name__ == "__main__":
    feats = pd.read_csv(FEAT)
    raw   = pd.read_csv(RAW)
    T0_str = load_T0_from_meta_or_raw(raw)

    lab = make_labeled(feats, raw, T0_str)
    X = lab.drop(columns=["customer_id", "label"])
    y = lab["label"].astype(int).values

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(Xtr, ytr)

    pva = clf.predict_proba(Xva)[:, 1]
    preval = float(yva.mean())
    thr = np.quantile(pva, 1.0 - preval) if preval > 0 else 1.0

    acc = accuracy_score(yva, (pva >= thr).astype(int))
    print(f"Valid ACC={acc:.4f}  thr*={thr:.2f}")

    joblib.dump(clf, OUTM)
    json.dump({"thr": float(thr), "T0": T0_str}, open(OUTJ, "w"), ensure_ascii=False, indent=2)
    print(f"Modelo guardado: {OUTM} | Meta: {OUTJ}")
# training.py
import pandas as pd, numpy as np, json, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

RAW    = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Archivos base\customer_purchases_train.csv"
FEAT   = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\train_features_per_customer.csv"
META_T = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\timing_meta.json"
OUTM   = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\model_rf.pkl"
OUTJ   = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\model_meta.json"

def load_T0_from_meta_or_raw(raw_df: pd.DataFrame) -> str:
    if os.path.exists(META_T):
        try:
            meta = json.load(open(META_T, "r", encoding="utf-8"))
            if "T0" in meta and meta["T0"]:
                return str(meta["T0"])
        except:
            pass
    tmp = raw_df.copy()
    tmp["purchase_timestamp"] = pd.to_datetime(tmp["purchase_timestamp"], errors="coerce")
    return str(tmp["purchase_timestamp"].max())

def make_labeled(feats: pd.DataFrame, raw: pd.DataFrame, T0_str: str) -> pd.DataFrame:
    T0 = pd.Timestamp(T0_str)
    r = raw.copy()
    r["purchase_timestamp"] = pd.to_datetime(r["purchase_timestamp"], errors="coerce")
    buyers = r[
        (r["purchase_timestamp"].notna()) &
        (r["purchase_timestamp"] > T0) &
        (r["purchase_timestamp"] <= T0 + pd.Timedelta(days=30))
    ]["customer_id"].astype(str).unique().tolist()
    f = feats.copy()
    f["customer_id"] = f["customer_id"].astype(str)
    f["label"] = f["customer_id"].isin(buyers).astype(int)
    return f

if __name__ == "__main__":
    feats = pd.read_csv(FEAT)
    raw   = pd.read_csv(RAW)
    T0_str = load_T0_from_meta_or_raw(raw)

    lab = make_labeled(feats, raw, T0_str)
    X = lab.drop(columns=["customer_id", "label"])
    y = lab["label"].astype(int).values

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(Xtr, ytr)

    pva = clf.predict_proba(Xva)[:, 1]
    preval = float(yva.mean())
    thr = np.quantile(pva, 1.0 - preval) if preval > 0 else 1.0

    acc = accuracy_score(yva, (pva >= thr).astype(int))
    print(f"Valid ACC={acc:.4f}  thr*={thr:.2f}")

    joblib.dump(clf, OUTM)
    json.dump({"thr": float(thr), "T0": T0_str}, open(OUTJ, "w"), ensure_ascii=False, indent=2)
    print(f"Modelo guardado: {OUTM} | Meta: {OUTJ}")
