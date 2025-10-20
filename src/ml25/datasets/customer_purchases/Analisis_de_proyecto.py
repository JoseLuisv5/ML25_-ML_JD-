# eda_processing_project1.py
# Uso:
#   python eda_processing_project1.py
#   python eda_processing_project1.py --train "C:\ruta\train.csv" --outdir "C:\ruta\salida"

import argparse, os, warnings, re, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

#DEF_TRAIN = r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\datasets\customer_purchases\customer_purchases_train.csv"
#DEF_OUT   = r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\datasets\customer_purchases\out_features_agg"

DEF_TRAIN = r"c:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\datasets\customer_purchases\customer_purchases_train.csv"
DEF_OUT   = r"c:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\datasets\customer_purchases\out_features_agg"
CUTOFF = pd.Timestamp("2025-09-21")
ADJ_WORDS = ["exclusive","style","casual","stylish","elegant",
             "durable","classic","lightweight","modern","premium"]

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

def make_adj_hot(df, words):
    tmp = df[["customer_id", "item_title"]].copy()
    tmp["customer_id"] = tmp["customer_id"].apply(id_to_num).astype("Int64")
    title = tmp["item_title"].fillna("").str.lower()
    for w in words:
        tmp[f"adj_{w}"] = title.str.contains(rf"\b{re.escape(w)}\b", na=False).astype(np.int8)
    keep = ["customer_id"] + [f"adj_{w}" for w in words]
    return tmp[keep].groupby("customer_id").max().reset_index()

def drop_all_zero_columns(df):
    keep = ["customer_id","antiguedad_dias","edad_anios",
            "dias_desde_ultima_compra","visitas","gasto_pct","compras"]
    drop = [c for c in df.columns if c not in keep and np.isfinite(df[c]).all() and df[c].sum()==0]
    return df.drop(columns=drop)

def filter_valid_purchases(df):
    d = df.copy()
    d["purchase_timestamp"] = to_dt(d.get("purchase_timestamp"))
    return d[d["purchase_timestamp"].notna() & (d["purchase_timestamp"] <= CUTOFF)]

def build_matrix(df):
    df = filter_valid_purchases(df)
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

    compras_df = df.groupby("customer_id").size().rename("compras").reset_index()
    base = base.merge(compras_df, on="customer_id", how="left")
    base["compras"] = base["compras"].fillna(0).astype(np.int32)

    base["customer_id"] = base["customer_id"].apply(id_to_num).astype("Int64")
    base["antiguedad_dias"] = (CUTOFF - base["signup_min"]).dt.days
    base["edad_anios"] = ((CUTOFF - base["dob_min"]).dt.days/365.25).round(2)
    base["dias_desde_ultima_compra"] = (CUTOFF - base["last_purchase"]).dt.days
    base["gasto_pct"] = base["gasto_total"].rank(pct=True)*100.0

    genero_oh      = make_multi_hot(df, "customer_gender", "genero")
    categoria_oh   = make_multi_hot(df, "item_category",   "categoria")
    dispositivo_oh = make_multi_hot(df, "purchase_device", "dispositivo")
    color_oh       = make_multi_hot(df, "color",           "color")
    adj_oh         = make_adj_hot(df, ADJ_WORDS)

    out = base[["customer_id","antiguedad_dias","edad_anios",
                "dias_desde_ultima_compra","visitas","gasto_pct","compras"]]
    for d in (genero_oh, categoria_oh, dispositivo_oh, color_oh, adj_oh):
        d["customer_id"] = d["customer_id"].astype("Int64")
        out = out.merge(d, on="customer_id", how="left")

    out = out.fillna(0)
    out["customer_id"] = out["customer_id"].astype(np.int32)
    for c in out.columns:
        if c == "customer_id": continue
        if out[c].dtype == np.float64: out[c] = out[c].astype(np.float32)
        if out[c].dtype == np.int64:   out[c] = out[c].astype(np.int32)

    out = drop_all_zero_columns(out)
    return out, df

def _strip_prefix(idx, prefix):
    return idx.str.replace(f"{prefix}_","", regex=False)\
              .str.replace("unspecified","unk").str.replace("unknown","unk").str.replace("nan","unk")

def _counts_from_prefix(df_feat, pref):
    cols = [c for c in df_feat.columns if c.startswith(pref+"_")]
    if not cols: return pd.Series(dtype=float)
    counts = df_feat[cols].sum().sort_values(ascending=False)
    if pref != "adj":
        counts.index = _strip_prefix(counts.index.to_series(), pref)
    else:
        counts.index = counts.index.str.replace("adj_","", regex=False)
    return counts[counts > 0]

def _age_bins_series(dob, when):
    edad = ((when - dob).dt.days/365.25)
    return pd.cut(
        edad, bins=[0,18,25,35,45,55,65,120],
        labels=["<18","18-24","25-34","35-44","45-54","55-64","65+"],
        right=False
    )

def plots_all_in_one(df_feat, df_raw_valid, outdir):
    os.makedirs(outdir, exist_ok=True)
    plots = []

    # SCATTERS básicos
    if "gasto_pct" in df_feat.columns:
        plots.append(("scatter", "gasto_pct", df_feat["gasto_pct"].astype(float)))
    if "compras" in df_feat.columns:
        plots.append(("scatter", "compras", df_feat["compras"].astype(float)))

    # BARRAS básicas (cuentas por categoría)
    for pref, titulo in [("genero","Genero"), ("dispositivo","Dispositivo"),
                         ("color","Color"), ("adj","Adjetivos")]:
        counts = _counts_from_prefix(df_feat, pref)
        if counts.empty: continue
        plots.append(("bar", titulo, counts))

    # PIE de categorías (top 8 + otros)
    cat_counts = _counts_from_prefix(df_feat, "categoria")
    if not cat_counts.empty:
        top = cat_counts.head(8)
        otros = cat_counts.iloc[8:].sum()
        if otros > 0:
            top = pd.concat([top, pd.Series({"otros": otros})])
        plots.append(("pie", "Categoria", top))
        # además una barra simple de categorías
        plots.append(("bar", "Categoria (bar)", cat_counts.head(15)))

    # Top ropa por rangos de edad (barras)
    dob  = to_dt(df_raw_valid.get("customer_date_of_birth"))
    ageb = _age_bins_series(dob, df_raw_valid["purchase_timestamp"])
    cats = df_raw_valid["item_category"].map(canon)
    edad_cat = pd.DataFrame({"agebin": ageb, "cat": cats}).dropna()
    if not edad_cat.empty:
        for label in ["<18","18-24","25-34","35-44","45-54","55-64","65+"]:
            sub = edad_cat[edad_cat["agebin"] == label]["cat"].value_counts()
            sub = sub[sub > 0].head(5)
            if not sub.empty:
                plots.append(("bar", f"Top ropa {label}", sub))

    # Render en una sola imagen
    n = len(plots); ncols = 3; nrows = math.ceil(n/ncols) if n>0 else 1
    fig = plt.figure(figsize=(ncols*5.8, nrows*3.8))
    for i,(kind,name,data) in enumerate(plots, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        if kind == "scatter":
            y = pd.Series(data).replace([np.inf,-np.inf], np.nan).dropna().values
            idx = np.arange(1, len(y)+1)
            ax.scatter(idx, y, s=12, alpha=0.8)
            if len(y) > 0:
                ax.axhline(float(np.nanmean(y)), linestyle="--", linewidth=1)
            ax.set_title(name); ax.set_xlabel("cliente"); ax.set_ylabel("valor" if name!="gasto_pct" else "percentil")
        elif kind == "bar":
            data.plot(kind="bar", ax=ax)
            xlabel = "categoria" if not name.startswith("Top ropa") and "Categoria" not in name else ("ropa" if name.startswith("Top ropa") else "categoria")
            ax.set_title(name); ax.set_xlabel(xlabel); ax.set_ylabel("clientes")
        elif kind == "pie":
            vals = data.values; labels = data.index.tolist()
            ax.pie(vals, labels=labels, autopct=lambda p: f"{p:.0f}%" if p >= 2 else "")
            ax.set_title(name)
        else:
            ax.axis("off")

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
    feat, raw_valid = build_matrix(raw)

    csv_path = os.path.join(out_dir, "train_features_per_customer.csv")
    feat.to_csv(csv_path, index=False)

    png_path = plots_all_in_one(feat, raw_valid, out_dir)

    print("OK.")
    print(f"Clientes (filtrados y agregados): {len(feat)}")
    print(f"Matriz numérica (incluye customer_id numérico): {csv_path}")
    print(f"PNG: {png_path}")

if __name__ == "__main__":
    main()
