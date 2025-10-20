import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import Counter

DATA_DIR = Path(r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1\Archivos base")
OUT_DIR  = Path(r"C:\Users\jlvh0\Documents\ML25_-ML_JD-\src\ml25\Proyecto1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
ADJ_LIST = ["exclusive","style","casual","stylish","elegant","durable","classic",
            "lightweight","modern","premium","cozy","sporty","trendy","soft"]
COLOR_MAP = {
    "imgb.jpg":"blue","imgbl.jpg":"black","imgg.jpg":"green","imgr.jpg":"red",
    "imgp.jpg":"pink","imgo.jpg":"orange","imgy.jpg":"yellow","imgw.jpg":"white","imgpr.jpg":"purple"
}
AGE_BINS = [-1,17,24,34,44,54,64,150]
AGE_LABS = ["<=18","18-24","25-34","35-44","45-54","55-64","65+"]

df = pd.read_csv(DATA_DIR / "customer_purchases_train.csv")

df["purchase_timestamp"]   = pd.to_datetime(df["purchase_timestamp"], errors="coerce")
df["customer_signup_date"] = pd.to_datetime(df["customer_signup_date"], errors="coerce")
dob = pd.to_datetime(df["customer_date_of_birth"], errors="coerce")
df["customer_age_years"]   = ((pd.Timestamp(DATA_COLLECTED_AT) - dob).dt.days // 365).astype("Int64")
df["customer_age_range"]   = pd.cut(df["customer_age_years"], bins=AGE_BINS, labels=AGE_LABS).astype("string")

for col in ["customer_gender","purchase_device","item_category"]:
    s = df[col].astype("string").str.strip()
    df[col] = s.fillna("Unknown").replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})

fn = df["item_img_filename"].astype("string").str.lower().str.strip().fillna("")
df["item_color"] = fn.apply(lambda x: COLOR_MAP.get(x, "Unknown"))
df["customer_tenure_days"] = (pd.to_datetime(DATA_COLLECTED_AT) - df["customer_signup_date"]).dt.days

cust = df.groupby("customer_id").agg(
    compras=("purchase_id","count"),
    gasto=("item_price","sum"),
    last_purchase=("purchase_timestamp","max")
).reset_index()
total = max(cust["gasto"].sum(), 1)
cust["gasto_pct"] = (cust["gasto"]/total)*100

adjs = Counter()
for t in df["item_title"].fillna("").astype(str).str.lower():
    for w in t.replace(",", " ").replace(".", " ").split():
        if w in ADJ_LIST: adjs[w] += 1
adjs = pd.Series(adjs).sort_values(ascending=False).head(10)

fig, axes = plt.subplots(4, 4, figsize=(18, 18))
ax = axes.ravel()

ax[0].scatter(cust.index, cust["gasto_pct"], s=8); ax[0].axhline(cust["gasto_pct"].mean(), ls="--"); ax[0].set_title("gasto_pct")
ax[1].scatter(cust.index, cust["compras"], s=8); ax[1].set_title("compras")
df["customer_gender"].value_counts(dropna=False).plot(kind="bar", ax=ax[2], title="Género")
df["purchase_device"].value_counts(dropna=False).plot(kind="bar", ax=ax[3], title="Medio")
df["item_color"].value_counts(dropna=False).plot(kind="bar", ax=ax[4], title="Color")
adjs.plot(kind="bar", ax=ax[5], title="Adjetivos")
df["item_category"].value_counts(dropna=False).head(8).plot(kind="pie", ax=ax[6], autopct="%1.0f%%", ylabel="", title="Categoría")
df["item_category"].value_counts(dropna=False).head(10).plot(kind="bar", ax=ax[7], title="Categoría (bar)")
df["customer_age_range"].value_counts(dropna=False).sort_index().plot(kind="bar", ax=ax[8], title="Rangos de edad")
df["item_price"].dropna().plot(kind="hist", bins=20, ax=ax[9], title="Precio")
df["item_avg_rating"].dropna().plot(kind="hist", bins=10, ax=ax[10], title="Rating")

buckets = ["<=18","18-24","25-34","35-44","45-54","55-64","65+"]
start = 11
for j, b in enumerate(buckets):
    idx = start + j
    if idx >= len(ax): break
    sub = df[df["customer_age_range"] == b]
    s = sub["item_category"].value_counts(dropna=False).head(4)
    if s.empty:
        ax[idx].set_title(f"Top ropa {b} (sin datos)")
        ax[idx].axis("off")
    else:
        s.plot(kind="bar", ax=ax[idx], title=f"Top ropa {b}")

# Colocar todos los subplots y luego añadir las gráficas extra (evita el warning)
plt.tight_layout()

# Extra 1: Ingresos por sexo
ax_sexo = fig.add_axes([0.06, 0.03, 0.38, 0.22])  # [left, bottom, width, height]
df.groupby("customer_gender")["item_price"].sum().sort_values(ascending=False).plot(
    kind="bar", ax=ax_sexo, title="Ingresos por sexo"
)

# Extra 2: Ingresos por categoría (Top 10)
ax_cat = fig.add_axes([0.56, 0.03, 0.38, 0.22])
df.groupby("item_category")["item_price"].sum().sort_values(ascending=False).head(10).plot(
    kind="bar", ax=ax_cat, title="Ingresos por categoría (Top 10)"
)

plt.savefig(OUT_DIR / "plots.png", dpi=150)
plt.close()
