
from pathlib import Path

import pandas as pd


# 1. Configuration
DATA_PATH = Path("data/raw")
OUTPUT_PATH = Path("data/processed/churn_dataset_v3.csv")

CUTOFF = pd.Timestamp("2018-01-01")
OBSERVATION_DAYS = 180
PREDICTION_DAYS = 180

OBSERVATION_START = CUTOFF - pd.Timedelta(days=OBSERVATION_DAYS)
PREDICTION_END = CUTOFF + pd.Timedelta(days=PREDICTION_DAYS)


# 2. Charger les donnees
customers = pd.read_csv(
    DATA_PATH / "olist_customers_dataset.csv",
    usecols=["customer_id", "customer_unique_id"],
)

orders = pd.read_csv(
    DATA_PATH / "olist_orders_dataset.csv",
    usecols=[
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
    ],
)

items = pd.read_csv(
    DATA_PATH / "olist_order_items_dataset.csv",
    usecols=["order_id", "price"],
)

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"], errors="coerce"
)
orders["order_delivered_customer_date"] = pd.to_datetime(
    orders["order_delivered_customer_date"], errors="coerce"
)

orders = orders.merge(
    customers,
    on="customer_id",
    how="left",
    validate="many_to_one",
)

order_amounts = (
    items.groupby("order_id", as_index=False)["price"]
    .sum()
    .rename(columns={"price": "montant_articles"})
)


# 3. Construire les caracteristiques historiques
observation = orders[
    (orders["order_status"] == "delivered")
    & orders["order_delivered_customer_date"].notna()
    & (orders["order_delivered_customer_date"] < CUTOFF)
    & (orders["order_purchase_timestamp"] >= OBSERVATION_START)
    & (orders["order_purchase_timestamp"] < CUTOFF)
    & orders["customer_unique_id"].notna()
].copy()

observation = observation.merge(
    order_amounts,
    on="order_id",
    how="left",
    validate="one_to_one",
)

missing_amount_orders = observation["montant_articles"].isna().sum()
observation["montant_articles"] = observation["montant_articles"].fillna(0)

features = (
    observation.groupby("customer_unique_id")
    .agg(
        frequence_180j=("order_id", "nunique"),
        montant_articles_180j=("montant_articles", "sum"),
        derniere_commande=("order_purchase_timestamp", "max"),
    )
    .reset_index()
)

features["recence_jours"] = (
    CUTOFF - features["derniere_commande"]
).dt.days

features["panier_moyen_180j"] = (
    features["montant_articles_180j"] / features["frequence_180j"]
)

features = features.drop(columns=["derniere_commande"])


# 4. Construire la cible sur les 180 jours futurs
future_orders = orders[
    (orders["order_status"] == "delivered")
    & (orders["order_purchase_timestamp"] >= CUTOFF)
    & (orders["order_purchase_timestamp"] < PREDICTION_END)
    & orders["customer_unique_id"].notna()
]

returning_ids = set(future_orders["customer_unique_id"])

features["churn"] = (
    ~features["customer_unique_id"].isin(returning_ids)
).astype("int8")


# 5. Verifications et sauvegarde
assert features["customer_unique_id"].is_unique
assert features.notna().all().all()
assert features["frequence_180j"].ge(1).all()
assert features["recence_jours"].ge(0).all()
assert features["churn"].isin([0, 1]).all()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(OUTPUT_PATH, index=False)

print("=== DATASET CHURNGUARD V3 - HORIZON 180 JOURS ===")
print("Debut observation :", OBSERVATION_START.date())
print("Fin observation exclusive :", CUTOFF.date())
print("Fin prediction exclusive :", PREDICTION_END.date())
print("Clients :", len(features))
print("Colonnes :", features.columns.tolist())
print("Commandes sans montant d'articles :", missing_amount_orders)

print("\nRepartition de la cible (0 = retour, 1 = sans retour) :")
print(features["churn"].value_counts().sort_index())

print("\nPourcentages :")
print(
    (
        features["churn"].value_counts(normalize=True).sort_index() * 100
    ).round(2)
)

print("\nValeurs manquantes :")
print(features.isna().sum())

print("\nDataset enregistre dans :", OUTPUT_PATH)
