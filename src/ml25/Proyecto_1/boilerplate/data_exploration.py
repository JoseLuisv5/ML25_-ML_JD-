import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import Counter

#DATA_DIR = Path(r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\datasets\customer_purchases")
#OUT_DIR  = Path(r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto 1")
DATA_DIR = Path(r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\datasets\customer_purchases")
OUT_DIR = Path(r"C:\Users\busta\Desktop\CETYS\Profesional\5to Semestre\Aprendizaje de Maquina\ML25_-ML_JD-\src\ml25\Proyecto 1")
DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
ADJ_LIST = ["exclusive","style","casual","stylish","elegant","durable","classic","lightweight","modern","premium","cozy","sporty","trendy","soft"]

def read_csv(name: str): 
    return pd.read_csv(DATA_DIR / f"{name}.csv")

# === NUEVO: helper para asegurar "Unknown" en categóricas ===
def cat_unknown(s, unknown_label="Unknown"):
    s = s.astype("string")
    s = s.str.strip()
    s = s.fillna(unknown_label)
    s = s.replace({"": unknown_label, "nan": unknown_label, "None": unknown_label})
    return s

def _age_years(dob):
    dob = pd.to_datetime(dob, errors="coerce")
    return ((pd.Timestamp(DATA_COLLECTED_AT) - dob).dt.days // 365).astype("Int64")

def _age_range(age):
    bins = [-1,17,24,34,44,54,64,150]
    labs = ["<=18","18-24","25-34","35-44","45-54","55-64","65+"]
    return pd.cut(age, bins=bins, labels=labs)

# === MOD: color seguro, devuelve "Unknown" si no matchea ===
def _img_color(fname):
    f = (str(fname).lower().strip() if isinstance(fname, str) else "")
    m = {
        "imgb.jpg":"blue","imgbl.jpg":"black","imgg.jpg":"green","imgr.jpg":"red",
        "imgp.jpg":"pink","imgo.jpg":"orange","imgy.jpg":"yellow","imgw.jpg":"white","imgpr.jpg":"purple"
    }
    return m.get(f, "Unknown")

def _adj_counts(series):
    from collections import Counter
    c = Counter()
    if series is not None:
        for t in series.fillna("").astype(str).str.lower():
            for w in t.replace(",", " ").replace(".", " ").split():
                if w in ADJ_LIST: c[w] += 1
    return pd.Series(c).sort_values(ascending=False)

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = read_csv("customer_purchases_train")

    # Derivadas
    train["customer_age_years"] = _age_years(train["customer_date_of_birth"])
    train["customer_age_range"] = _age_range(train["customer_age_years"]).astype("string").fillna("Unknown")  # <- MOD: Unknown
    train["customer_tenure_days"] = (pd.to_datetime(DATA_COLLECTED_AT) - pd.to_datetime(train["customer_signup_date"], errors="coerce")).dt.days
    train["item_color"] = train["item_img_filename"].map(_img_color)

    if "purchase_timestamp" in train.columns:
        train["purchase_timestamp"] = pd.to_datetime(train["purchase_timestamp"], errors="coerce")

    # Normaliza categóricas a "Unknown" (si existen)
    if "customer_gender" in train.columns:
        train["customer_gender"] = cat_unknown(train["customer_gender"])
    if "purchase_device" in train.columns:
        train["purchase_device"] = cat_unknown(train["purchase_device"])
    if "item_category" in train.columns:
        train["item_category"] = cat_unknown(train["item_category"])
    train["item_color"] = cat_unknown(train["item_color"])  # siempre existe ya

    # Agregados por cliente
    cust = train.groupby("customer_id").agg(
        compras=("purchase_id","count"),
        gasto=("item_price","sum"),
        last_purchase=("purchase_timestamp","max") if "purchase_timestamp" in train.columns else ("purchase_id","count"),
        visitas=("customer_item_views","sum") if "customer_item_views" in train.columns else ("purchase_id","count")
    ).reset_index()
    total = cust["gasto"].sum() if len(cust) else 1
    cust["gasto_pct"] = (cust["gasto"]/total)*100

    fig, axes = plt.subplots(4, 4, figsize=(18, 18)); ax = axes.ravel()
    ax[0].scatter(cust.index, cust["gasto_pct"], s=8); ax[0].axhline(cust["gasto_pct"].mean() if len(cust) else 0, ls="--"); ax[0].set_title("gasto_pct")
    ax[1].scatter(cust.index, cust["compras"], s=8); ax[1].set_title("compras")

    # === MOD: usa series normalizadas con Unknown ===
    if "customer_gender" in train.columns: 
        train["customer_gender"].value_counts(dropna=False).plot(kind="bar", ax=ax[2], title="Género")
    else: 
        ax[2].set_visible(False)

    if "purchase_device" in train.columns: 
        train["purchase_device"].value_counts(dropna=False).plot(kind="bar", ax=ax[3], title="Medio")
    else: 
        ax[3].set_visible(False)

    train["item_color"].value_counts(dropna=False).plot(kind="bar", ax=ax[4], title="Color")

    adjs = _adj_counts(train["item_title"] if "item_title" in train.columns else None).head(10)
    if not adjs.empty: 
        adjs.plot(kind="bar", ax=ax[5], title="Adjetivos")
    else: 
        ax[5].set_visible(False)

    if "item_category" in train.columns:
        train["item_category"].value_counts(dropna=False).head(8).plot(kind="pie", ax=ax[6], autopct="%1.0f%%", ylabel="", title="Categoría")
        train["item_category"].value_counts(dropna=False).head(10).plot(kind="bar", ax=ax[7], title="Categoría (bar)")
    else:
        ax[6].set_visible(False); ax[7].set_visible(False)

    train["customer_age_range"].value_counts(dropna=False).sort_index().plot(kind="bar", ax=ax[8], title="Rangos de edad")

    if "item_price" in train.columns and train["item_price"].notna().any(): 
        train["item_price"].dropna().plot(kind="hist", bins=20, ax=ax[9], title="Precio")
    else: 
        ax[9].set_visible(False)

    if "item_avg_rating" in train.columns and train["item_avg_rating"].notna().any(): 
        train["item_avg_rating"].dropna().plot(kind="hist", bins=10, ax=ax[10], title="Rating")
    else: 
        ax[10].set_visible(False)

    buckets = ["<=18","18-24","25-34","35-44","45-54","55-64","65+"]
    start = 8
    for j, b in enumerate(buckets):
        idx = start + j
        if idx >= len(ax): break
        sub = train[train["customer_age_range"] == b]
        if len(sub) > 0 and "item_category" in sub.columns:
            cat_unknown(sub["item_category"]).value_counts(dropna=False).head(4).plot(kind="bar", ax=ax[idx], title=f"Top ropa {b}")
        else:
            ax[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "plots.png", dpi=150)
    plt.close()
