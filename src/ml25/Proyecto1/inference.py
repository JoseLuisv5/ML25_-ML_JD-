# inference.py
import pandas as pd, numpy as np, re, json, joblib, os

TEST  = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Archivos base\customer_purchases_test.csv"
MODEL = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\model_rf.pkl"
META  = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\model_meta.json"
META_T= r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\timing_meta.json"
OUT   = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\preds.csv"

def to_dt(s):  return pd.to_datetime(s, errors="coerce")
def to_num(s): return pd.to_numeric(s, errors="coerce")
def canon(s):  return str(s).strip().lower() if pd.notna(s) else "unk"
def color_from_img(fname):
    if not isinstance(fname,str): return "unk"
    m = re.search(r"img([a-z]+)\.", fname.lower()); code = m.group(1) if m else ""
    mapp = {"r":"red","g":"green","y":"yellow","w":"white","o":"orange","p":"purple","bl":"blue","b":"blue","bk":"black","gr":"green"}
    return mapp.get(code,"unk")

ADJ = ["luxury","deluxe","exclusive","elegant","casual","modern","classic","stylish",
       "durable","lightweight","premium","sport","comfort","trend"]

def resolve_T0(df_test: pd.DataFrame):
    t0 = None
    if os.path.exists(META):
        try:
            meta = json.load(open(META,"r",encoding="utf-8"))
            if "T0" in meta and meta["T0"]:
                t0 = pd.Timestamp(meta["T0"])
        except:
            pass
    if t0 is None and os.path.exists(META_T):
        try:
            meta_t = json.load(open(META_T,"r",encoding="utf-8"))
            if "T0" in meta_t and meta_t["T0"]:
                t0 = pd.Timestamp(meta_t["T0"])
        except:
            pass
    if t0 is None:
        ts = to_dt(df_test.get("purchase_timestamp"))
        t0 = ts.max() if ts.notna().any() else pd.Timestamp.today()
    return t0

def build_features(df, T0):
    d = df.copy()
    if "customer_id" not in d.columns:
        d["customer_id"] = np.arange(len(d), dtype=int)
    d["customer_id"]            = d["customer_id"].astype(str)
    d["purchase_timestamp"]     = to_dt(d.get("purchase_timestamp"))
    d["customer_signup_date"]   = to_dt(d.get("customer_signup_date"))
    d["customer_date_of_birth"] = to_dt(d.get("customer_date_of_birth"))
    d["item_price"]             = to_num(d.get("item_price"))
    d["customer_item_views"]    = to_num(d.get("customer_item_views")).fillna(0)
    d["item_category"]          = d.get("item_category","unk").map(canon)
    d["purchase_device"]        = d.get("purchase_device","unk").map(canon)
    d["customer_gender"]        = d.get("customer_gender","unk").map(canon)
    d["color"]                  = d.get("item_img_filename","").apply(color_from_img)
    title = d.get("item_title","").astype(str).str.lower()

    def multi_hot(col,prefix):
        x = pd.get_dummies(d[col], prefix=prefix, dtype=np.int8)
        x["customer_id"] = d["customer_id"].values
        return x.groupby("customer_id").max().reset_index()

    genero_oh = multi_hot("customer_gender","gen")
    cat_oh    = multi_hot("item_category","cat")
    dev_oh    = multi_hot("purchase_device","dev")
    color_oh  = multi_hot("color","color")

    g = d.groupby("customer_id")
    base = g.agg(
        compras=("purchase_timestamp","count"),
        gasto_total=("item_price","sum"),
        visitas=("customer_item_views","sum"),
        price_mean=("item_price","mean"),
        price_max=("item_price","max"),
        signup_min=("customer_signup_date","min"),
        dob_min=("customer_date_of_birth","min")
    ).reset_index()

    base["gasto_total"] = base["gasto_total"].fillna(0)
    base["visitas"]     = base["visitas"].fillna(0)
    base["price_mean"]  = base["price_mean"].fillna(0)
    base["price_max"]   = base["price_max"].fillna(0)
    base["signup_min"]  = base["signup_min"].fillna(T0)
    base["dob_min"]     = base["dob_min"].fillna(T0)

    base["antiguedad_dias"] = (T0 - base["signup_min"]).dt.days.clip(lower=1)
    base["edad_anios"]      = ((T0 - base["dob_min"]).dt.days/365.25).replace([np.inf,-np.inf],0).fillna(0)
    base["views_per_day"]   = base["visitas"]/base["antiguedad_dias"]
    base["buys_per_100d"]   = base["compras"]/(base["antiguedad_dias"]/100.0 + 1.0)
    base["view_to_buy"]     = base["visitas"]/(base["compras"] + 1.0)

    base["log_visitas"]     = np.log1p(base["visitas"])
    base["log_compras"]     = np.log1p(base["compras"])
    base["log_gasto_total"] = np.log1p(base["gasto_total"])
    base["log_price_mean"]  = np.log1p(base["price_mean"])
    base["log_price_max"]   = np.log1p(base["price_max"])

    feats = base[[
        "customer_id","compras","gasto_total","visitas","price_mean","price_max",
        "antiguedad_dias","edad_anios",
        "views_per_day","buys_per_100d","view_to_buy",
        "log_visitas","log_compras","log_gasto_total","log_price_mean","log_price_max"
    ]]

    adj_df = pd.DataFrame({"customer_id": d["customer_id"].values})
    for w in ADJ:
        adj_df[f"adj_{w}"] = title.str.contains(rf"\b{re.escape(w)}\b", na=False).astype(np.int8)
    adj_oh = adj_df.groupby("customer_id").max().reset_index()

    for blk in (genero_oh,cat_oh,dev_oh,color_oh,adj_oh):
        feats = feats.merge(blk, on="customer_id", how="left")

    return feats.fillna(0)

if __name__ == "__main__":
    clf = joblib.load(MODEL)
    meta = {}
    if os.path.exists(META):
        try:
            meta = json.load(open(META,"r",encoding="utf-8"))
        except:
            meta = {}
    thr = float(meta.get("thr", 0.50))

    df_test = pd.read_csv(TEST)
    T0 = resolve_T0(df_test)

    feats = build_features(df_test, T0)
    X = feats.drop(columns=["customer_id"]).copy()

    if hasattr(clf, "feature_names_in_"):
        cols = list(clf.feature_names_in_)
        for c in cols:
            if c not in X.columns: X[c] = 0
        X = X[cols]

    probs = clf.predict_proba(X)[:,1]
    preds = (probs >= thr).astype(int)

    sub = pd.DataFrame({"ID": np.arange(len(df_test), dtype=int), "Pred": preds})
    sub.to_csv(OUT, index=False)
    print(f"Preds: {OUT} | rows={len(sub)} | thr={thr:.2f} | pos_rate={preds.mean():.3f}")
