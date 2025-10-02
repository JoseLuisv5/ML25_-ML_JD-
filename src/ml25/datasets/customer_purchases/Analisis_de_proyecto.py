# eda_processing_project1.py
# Uso:
#   python eda_processing_project1.py
#   python eda_processing_project1.py --train "C:\ruta\train.csv" --outdir "C:\ruta\salida"

import argparse, os, warnings, re, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DEF_TRAIN = r"c:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\datasets\customer_purchases\customer_purchases_train.csv"
DEF_OUT   = r"c:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\datasets\customer_purchases\out_features_agg"
CUTOFF = pd.Timestamp("2025-09-21")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, default=DEF_TRAIN)
    p.add_argument("--outdir", type=str, default=DEF_OUT)
    return p.parse_args()

def to_dt(s): return pd.to_datetime(s, errors="coerce")
def num(s):   return pd.to_numeric(s, errors="coerce")

def canon(x):
    if isinstance(x, str):
        t = x.strip().lower()
        return "unk" if t in ("", "nan", "none", "unknown", "unspecified") else t
    return "unk"

def color_from_img(fname):
    if not isinstance(fname, str): return "unk"
    m = re.search(r"img([a-z]+)\.", fname.lower())
    code = (m.group(1) if m else "")
    mapp = {"r":"red","g":"green","y":"yellow","w":"white","o":"orange","p":"purple",
            "bl":"blue","b":"blue","bk":"black","gr":"green"}
    return mapp.get(code, "unk")

def id_to_num(s):
    if not isinstance(s, str): return np.nan
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else np.nan

def make_multi_hot(df, col, prefix):
    tmp = df[["customer_id", col]].copy()
    tmp["customer_id"] = tmp["customer_id"].apply(id_to_num).astype("Int64")
    tmp[col] = tmp[col].map(canon)
    dmy = pd.get_dummies(tmp[col], prefix=prefix, dtype=np.int8)
    dmy = pd.concat([tmp[["customer_id"]], dmy], axis=1).groupby("customer_id").max().reset_index()
    return dmy

def drop_all_zero_columns(df):
    keep = ["customer_id","antiguedad_dias","edad_anios","dias_desde_ultima_compra","visitas","gasto_pct"]
    drop = [c for c in df.columns if c not in keep and np.isfinite(df[c]).all() and df[c].sum()==0]
    return df.drop(columns=drop)

def build_matrix(df):
    df = df.copy()
    df["purchase_timestamp"]     = to_dt(df.get("purchase_timestamp"))
    df = df[df["purchase_timestamp"].notna() & (df["purchase_timestamp"] <= CUTOFF)]
    df["customer_signup_date"]   = to_dt(df.get("customer_signup_date"))
    df["customer_date_of_birth"] = to_dt(df.get("customer_date_of_birth"))
    df["item_price"]             = num(df.get("item_price"))
    df["customer_item_views"]    = num(df.get("customer_item_views")).fillna(0)
    df["purchase_device"]        = df.get("purchase_device").map(canon)
    df["item_category"]          = df.get("item_category").map(canon)
    df["customer_gender"]        = df.get("customer_gender").map(canon)
    df["color"]                  = df.get("item_img_filename").apply(color_from_img)

    base = df.groupby("customer_id").agg(
        signup_min=("customer_signup_date","min"),
        dob_min=("customer_date_of_birth","min"),
        last_purchase=("purchase_timestamp","max"),
        visitas=("customer_item_views","sum"),
        gasto_total=("item_price","sum")
    ).reset_index()

    base["customer_id"] = base["customer_id"].apply(id_to_num).astype("Int64")
    base["antiguedad_dias"] = (CUTOFF - base["signup_min"]).dt.days
    base["edad_anios"] = ((CUTOFF - base["dob_min"]).dt.days/365.25).round(2)
    base["dias_desde_ultima_compra"] = (CUTOFF - base["last_purchase"]).dt.days
    base["gasto_pct"] = base["gasto_total"].rank(pct=True)*100.0

    genero_oh      = make_multi_hot(df, "customer_gender", "genero")
    categoria_oh   = make_multi_hot(df, "item_category",   "categoria")
    dispositivo_oh = make_multi_hot(df, "purchase_device", "dispositivo")
    color_oh       = make_multi_hot(df, "color",           "color")

    out = base[["customer_id","antiguedad_dias","edad_anios",
                "dias_desde_ultima_compra","visitas","gasto_pct"]]
    for d in (genero_oh, categoria_oh, dispositivo_oh, color_oh):
        d["customer_id"] = d["customer_id"].astype("Int64")
        out = out.merge(d, on="customer_id", how="left")

    out = out.fillna(0)
    out["customer_id"] = out["customer_id"].astype(np.int32)

    for c in out.columns:
        if c == "customer_id": continue
        if out[c].dtype == np.float64: out[c] = out[c].astype(np.float32)
        if out[c].dtype == np.int64:   out[c] = out[c].astype(np.int32)

    out = drop_all_zero_columns(out)
    return out

def _strip_prefix(idx, prefix):
    return idx.str.replace(f"{prefix}_","", regex=False)\
              .str.replace("unspecified","unk").str.replace("unknown","unk").str.replace("nan","unk")

def plots_all_in_one(df_feat, outdir):
    os.makedirs(outdir, exist_ok=True)
    plots = []

    core = ["antiguedad_dias","edad_anios","dias_desde_ultima_compra","visitas","gasto_pct"]
    for c in core:
        if c in df_feat.columns: plots.append(("numeric", c, df_feat[c].astype(float)))

    for pref, titulo in [("genero","Genero"), ("categoria","Categoria"),
                         ("dispositivo","Dispositivo"), ("color","Color")]:
        cols = [c for c in df_feat.columns if c.startswith(pref+"_")]
        if not cols: continue
        counts = df_feat[cols].sum().sort_values(ascending=False)
        counts.index = _strip_prefix(counts.index.to_series(), pref)
        counts = counts[counts > 0]  # quitar categorías en cero (incluye unk si no hubo)
        if counts.empty: continue
        plots.append(("categorical", titulo, counts))

    n = len(plots); ncols = 3; nrows = math.ceil(n/ncols) if n>0 else 1
    fig = plt.figure(figsize=(ncols*5.5, nrows*3.4))
    for i,(kind,name,data) in enumerate(plots, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        if kind=="numeric":
            x = pd.Series(data).replace([np.inf,-np.inf], np.nan).dropna()
            if name=="gasto_pct":
                y = x.values
                idx = np.arange(1, len(y)+1)
                ax.scatter(idx, y, s=12, alpha=0.8)
                ax.axhline(float(np.nanmean(y)), linestyle="--", linewidth=1)
                ax.set_title("gasto_pct"); ax.set_xlabel("cliente"); ax.set_ylabel("percentil")
            else:
                x.plot(kind="hist", bins=30, ax=ax)
                ax.set_title(name); ax.set_xlabel(name); ax.set_ylabel("conteo")
        else:
            data.plot(kind="bar", ax=ax)
            ax.set_title(name); ax.set_xlabel("categoria"); ax.set_ylabel("clientes")
    plt.tight_layout()
    png = os.path.join(outdir, "all_plots.png")
    fig.savefig(png, dpi=160)
    plt.close(fig)
    return png

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    out_dir = os.path.join(args.outdir, "train"); os.makedirs(out_dir, exist_ok=True)

    raw = pd.read_csv(args.train)
    feat = build_matrix(raw)

    csv_path = os.path.join(out_dir, "train_features_per_customer.csv")
    feat.to_csv(csv_path, index=False)

    png_path = plots_all_in_one(feat, out_dir)

    print("OK.")
    print(f"Clientes (filtrados y agregados): {len(feat)}")
    print(f"Matriz numérica (incluye customer_id numérico): {csv_path}")
    print(f"PNG: {png_path}")

if __name__ == "__main__":
    main()
