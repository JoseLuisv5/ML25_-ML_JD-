import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import Counter

DATA_DIR = Path(r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Proyecto1\Archivos base")
OUT_DIR  = Path(r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Proyecto1")

DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
ADJ_LIST = ["exclusive","style","casual","stylish","elegant","durable","classic",
            "lightweight","modern","premium","cozy","sporty","trendy","soft"]

def read_csv(name: str): 
    return pd.read_csv(DATA_DIR / f"{name}.csv")

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
    return pd.cut(age, bins=bins, labels=labs, ordered=True)

def _img_color(fname):
    f = (str(fname).lower().strip() if isinstance(fname, str) else "")
    m = {
        "imgb.jpg":"blue","imgbl.jpg":"black","imgg.jpg":"green","imgr.jpg":"red",
        "imgp.jpg":"pink","imgo.jpg":"orange","imgy.jpg":"yellow","imgw.jpg":"white",
        "imgpr.jpg":"purple"
    }
    return m.get(f, "Unknown")

def _adj_counts(series):
    c = Counter()
    if series is not None:
        for t in series.fillna("").astype(str).str.lower():
            for w in t.replace(",", " ").replace(".", " ").split():
                if w in ADJ_LIST: 
                    c[w] += 1
    return pd.Series(c).sort_values(ascending=False)

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = read_csv("customer_purchases_train")

    train["customer_age_years"]  = _age_years(train.get("customer_date_of_birth"))
    train["customer_age_range"]  = _age_range(train["customer_age_years"]).astype("string").fillna("Unknown")
    train["customer_tenure_days"]= (pd.to_datetime(DATA_COLLECTED_AT) - pd.to_datetime(train.get("customer_signup_date"), errors="coerce")).dt.days
    train["item_color"]          = train.get("item_img_filename").map(_img_color)

    if "purchase_timestamp" in train.columns:
        train["purchase_timestamp"] = pd.to_datetime(train["purchase_timestamp"], errors="coerce")

    for col in ["customer_gender", "purchase_device", "item_category"]:
        if col in train.columns:
            train[col] = cat_unknown(train[col])
    train["item_color"] = cat_unknown(train["item_color"])

    cust = train.groupby("customer_id").agg(
        compras=("purchase_id","count") if "purchase_id" in train.columns else ("item_price","size"),
        gasto=("item_price","sum"),
        last_purchase=("purchase_timestamp","max") if "purchase_timestamp" in train.columns else ("customer_id","count"),
        visitas=("customer_item_views","sum") if "customer_item_views" in train.columns else ("customer_id","count")
    ).reset_index()
    total_gasto = cust["gasto"].sum() if len(cust) else 1.0
    cust["gasto_pct"] = (cust["gasto"] / total_gasto) * 100.0

    fig1, axes1 = plt.subplots(3, 4, figsize=(18, 12))
    ax = axes1.ravel()

    ax[0].scatter(cust.index, cust["gasto_pct"], s=8)
    if len(cust):
        ax[0].axhline(cust["gasto_pct"].mean(), ls="--", linewidth=1)
    ax[0].set_title("gasto_pct"); ax[0].set_xlabel("cliente"); ax[0].set_ylabel("% del gasto total")

    ax[1].scatter(cust.index, cust["compras"], s=8)
    ax[1].set_title("compras"); ax[1].set_xlabel("cliente"); ax[1].set_ylabel("n° compras")

    if "customer_gender" in train.columns:
        vc = train["customer_gender"].value_counts(dropna=False)
        vc.plot(kind="bar", ax=ax[2], title="Género")
        ax[2].set_xlabel(""); ax[2].tick_params(axis='x', rotation=0)
    else:
        ax[2].axis("off")

    if "purchase_device" in train.columns:
        vc = train["purchase_device"].value_counts(dropna=False)
        vc.plot(kind="bar", ax=ax[3], title="Medio")
        ax[3].set_xlabel(""); ax[3].tick_params(axis='x', rotation=0)
    else:
        ax[3].axis("off")

    train["item_color"].value_counts(dropna=False).plot(kind="bar", ax=ax[4], title="Color")
    ax[4].set_xlabel(""); ax[4].tick_params(axis='x', rotation=0)

    adjs = _adj_counts(train["item_title"] if "item_title" in train.columns else None).head(10)
    if not adjs.empty:
        adjs.plot(kind="bar", ax=ax[5], title="Adjetivos")
        ax[5].set_xlabel(""); ax[5].tick_params(axis='x', rotation=45)
    else:
        ax[5].axis("off")

    if "item_category" in train.columns:
        train["item_category"].value_counts(dropna=False).head(8).plot(
            kind="pie", ax=ax[6], autopct="%1.0f%%", ylabel="", title="Categoría"
        )
    else:
        ax[6].axis("off")

    if "item_category" in train.columns:
        train["item_category"].value_counts(dropna=False).head(10).plot(kind="bar", ax=ax[7], title="Categoría (bar)")
        ax[7].set_xlabel(""); ax[7].tick_params(axis='x', rotation=45)
    else:
        ax[7].axis("off")

    train["customer_age_range"].value_counts(dropna=False).sort_index().plot(kind="bar", ax=ax[8], title="Rangos de edad")
    ax[8].set_xlabel(""); ax[8].tick_params(axis='x', rotation=0)

    if "item_price" in train.columns and train["item_price"].notna().any():
        train["item_price"].dropna().plot(kind="hist", bins=20, ax=ax[9], title="Precio")
        ax[9].set_xlabel("precio")
    else:
        ax[9].axis("off")

    if "item_avg_rating" in train.columns and train["item_avg_rating"].notna().any():
        train["item_avg_rating"].dropna().plot(kind="hist", bins=10, ax=ax[10], title="Rating")
        ax[10].set_xlabel("rating")
    else:
        ax[10].axis("off")

    ax[11].axis("off")

    plt.tight_layout()
    fig1_path = OUT_DIR / "plots_overview.png"
    plt.savefig(fig1_path, dpi=150)
    plt.close(fig1)
    print("[OK] Guardado:", fig1_path)

    buckets = ["<=18","18-24","25-34","35-44","45-54","55-64","65+"]
    buckets = [b for b in buckets if (train["customer_age_range"] == b).any()]
    if len(buckets) == 0 or "item_category" not in train.columns:
        print("[INFO] Sin buckets de edad o sin item_category; no se genera plots_top_by_age.png")
    else:
        n = len(buckets)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5.6*ncols, 4*nrows))
        if nrows == 1 and ncols == 1:
            axes2 = [[axes2]]
        elif nrows == 1:
            axes2 = [axes2]
        ax2 = [a for row in axes2 for a in row]

        for i, b in enumerate(buckets):
            sub = train[train["customer_age_range"] == b]
            vc = cat_unknown(sub["item_category"]).value_counts(dropna=False).head(5)
            vc.plot(kind="bar", ax=ax2[i], title=f"Top ropa {b}")
            ax2[i].set_xlabel("item_category"); ax2[i].set_ylabel("Frequency")
            ax2[i].tick_params(axis='x', rotation=45)

        for j in range(len(buckets), len(ax2)):
            ax2[j].axis("off")

        plt.tight_layout()
        fig2_path = OUT_DIR / "plots_top_by_age.png"
        plt.savefig(fig2_path, dpi=150)
        plt.close(fig2)
        print("[OK] Guardado:", fig2_path)
