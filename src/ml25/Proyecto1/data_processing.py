import pandas as pd, numpy as np, re, argparse, json

RAW = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Archivos base\customer_purchases_train.csv"
OUT = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\train_features_per_customer.csv"
META_FEAT = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\timing_meta.json"

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

def build_features(df, T0):
    df = df.copy()
    df = df[df["purchase_timestamp"].notna() & (df["purchase_timestamp"] <= T0)]

    df["customer_signup_date"]   = to_dt(df.get("customer_signup_date"))
    df["customer_date_of_birth"] = to_dt(df.get("customer_date_of_birth"))
    df["item_price"]             = to_num(df.get("item_price"))
    df["customer_item_views"]    = to_num(df.get("customer_item_views")).fillna(0)
    df["item_category"]          = df.get("item_category","unk").map(canon)
    df["purchase_device"]        = df.get("purchase_device","unk").map(canon)
    df["customer_gender"]        = df.get("customer_gender","unk").map(canon)
    df["color"]                  = df.get("item_img_filename","").apply(color_from_img)
    title = df.get("item_title","").astype(str).str.lower()

    def multi_hot(col,prefix):
        x = pd.get_dummies(df[col], prefix=prefix, dtype=np.int8)
        x["customer_id"] = df["customer_id"].values
        return x.groupby("customer_id").max().reset_index()
    genero_oh = multi_hot("customer_gender","gen")
    cat_oh    = multi_hot("item_category","cat")
    dev_oh    = multi_hot("purchase_device","dev")
    color_oh  = multi_hot("color","color")

    g = df.groupby("customer_id")
    base = g.agg(
        compras=("purchase_timestamp","count"),
        gasto_total=("item_price","sum"),
        visitas=("customer_item_views","sum"),
        price_mean=("item_price","mean"),
        price_max=("item_price","max"),
        signup_min=("customer_signup_date","min"),
        dob_min=("customer_date_of_birth","min")
        # OJO: sin last_purchase
    ).reset_index()

    base["gasto_total"] = base["gasto_total"].fillna(0)
    base["visitas"]     = base["visitas"].fillna(0)
    base["price_mean"]  = base["price_mean"].fillna(0)
    base["price_max"]   = base["price_max"].fillna(0)

    base["antiguedad_dias"] = (T0 - base["signup_min"]).dt.days.clip(lower=1)
    base["edad_anios"]      = ((T0 - base["dob_min"]).dt.days/365.25).replace([np.inf,-np.inf],0).fillna(0)

    base["gasto_pct"]       = base["gasto_total"].rank(pct=True)*100.0
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
        "antiguedad_dias","edad_anios","gasto_pct",
        "views_per_day","buys_per_100d","view_to_buy",
        "log_visitas","log_compras","log_gasto_total","log_price_mean","log_price_max"
    ]]

    adj_df = pd.DataFrame({"customer_id": df["customer_id"].values})
    for w in ADJ:
        adj_df[f"adj_{w}"] = title.str.contains(rf"\b{re.escape(w)}\b", na=False).astype(np.int8)
    adj_oh = adj_df.groupby("customer_id").max().reset_index()

    for blk in (genero_oh,cat_oh,dev_oh,color_oh,adj_oh):
        feats = feats.merge(blk, on="customer_id", how="left")

    feats = feats.fillna(0)
    return feats

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default=RAW)
    ap.add_argument("--out", type=str, default=OUT)
    ap.add_argument("--meta", type=str, default=META_FEAT)
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    raw["purchase_timestamp"] = to_dt(raw.get("purchase_timestamp"))
    max_ts = raw["purchase_timestamp"].max()
    T0 = max_ts - pd.Timedelta(days=30)

    feats = build_features(raw, T0)
    feats.to_csv(args.out, index=False)

    with open(args.meta,"w",encoding="utf-8") as f:
        json.dump({"T0": str(T0), "max_ts": str(max_ts)}, f, ensure_ascii=False, indent=2)

    print("Features:", args.out, "| cols:", feats.shape[1], "| T0:", T0, "| max_ts:", max_ts)
import pandas as pd, numpy as np, re, argparse, json

RAW = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Archivos base\customer_purchases_train.csv"
OUT = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\train_features_per_customer.csv"
META_FEAT = r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\timing_meta.json"

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

def build_features(df, T0):
    df = df.copy()
    df = df[df["purchase_timestamp"].notna() & (df["purchase_timestamp"] <= T0)]

    df["customer_signup_date"]   = to_dt(df.get("customer_signup_date"))
    df["customer_date_of_birth"] = to_dt(df.get("customer_date_of_birth"))
    df["item_price"]             = to_num(df.get("item_price"))
    df["customer_item_views"]    = to_num(df.get("customer_item_views")).fillna(0)
    df["item_category"]          = df.get("item_category","unk").map(canon)
    df["purchase_device"]        = df.get("purchase_device","unk").map(canon)
    df["customer_gender"]        = df.get("customer_gender","unk").map(canon)
    df["color"]                  = df.get("item_img_filename","").apply(color_from_img)
    title = df.get("item_title","").astype(str).str.lower()

    def multi_hot(col,prefix):
        x = pd.get_dummies(df[col], prefix=prefix, dtype=np.int8)
        x["customer_id"] = df["customer_id"].values
        return x.groupby("customer_id").max().reset_index()
    genero_oh = multi_hot("customer_gender","gen")
    cat_oh    = multi_hot("item_category","cat")
    dev_oh    = multi_hot("purchase_device","dev")
    color_oh  = multi_hot("color","color")

    g = df.groupby("customer_id")
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

    adj_df = pd.DataFrame({"customer_id": df["customer_id"].values})
    for w in ADJ:
        adj_df[f"adj_{w}"] = title.str.contains(rf"\b{re.escape(w)}\b", na=False).astype(np.int8)
    adj_oh = adj_df.groupby("customer_id").max().reset_index()

    for blk in (genero_oh,cat_oh,dev_oh,color_oh,adj_oh):
        feats = feats.merge(blk, on="customer_id", how="left")

    feats = feats.fillna(0)
    return feats

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default=RAW)
    ap.add_argument("--out", type=str, default=OUT)
    ap.add_argument("--meta", type=str, default=META_FEAT)
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    raw["purchase_timestamp"] = to_dt(raw.get("purchase_timestamp"))
    max_ts = raw["purchase_timestamp"].max()
    T0 = max_ts - pd.Timedelta(days=30)

    feats = build_features(raw, T0)
    feats.to_csv(args.out, index=False)

    with open(args.meta,"w",encoding="utf-8") as f:
        json.dump({"T0": str(T0), "max_ts": str(max_ts)}, f, ensure_ascii=False, indent=2)

    print("Features:", args.out, "| cols:", feats.shape[1], "| T0:", T0, "| max_ts:", max_ts)
