
import pandas as pd

path = "data/raw/"

customers = pd.read_csv(path + "olist_customers_dataset.csv")
orders = pd.read_csv(path + "olist_orders_dataset.csv")
items = pd.read_csv(path + "olist_order_items_dataset.csv")

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

items_by_order = (
    items.groupby("order_id")
    .agg(montant_articles=("price", "sum"))
    .reset_index()
)

delivered = orders[orders["order_status"] == "delivered"].copy()

delivered = delivered.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="left",
    validate="one_to_one"
)

delivered = delivered.merge(
    items_by_order,
    on="order_id",
    how="left",
    validate="one_to_one"
)

cutoff = pd.Timestamp("2017-12-31")
observation_start = cutoff - pd.Timedelta(days=180)

observation = delivered[
    (delivered["order_purchase_timestamp"] > observation_start)
    & (delivered["order_purchase_timestamp"] <= cutoff)
].copy()

features = (
    observation.groupby("customer_unique_id")
    .agg(
        frequence=("order_id", "nunique"),
        derniere_commande=("order_purchase_timestamp", "max"),
        montant_total=("montant_articles", "sum"),
    )
    .reset_index()
)

features = features[features["frequence"] >= 2].copy()

features["recence_jours"] = (
    cutoff - features["derniere_commande"]
).dt.days

print("Nombre de clients récurrents :", len(features))

print("\n=== STATISTIQUES DES VARIABLES ===")
print(
    features[["frequence", "recence_jours", "montant_total"]]
    .describe()
    .round(2)
    .to_string()
)

print("\n=== RECENCE PAR GROUPE ===")
print(
    features["recence_jours"]
    .value_counts(bins=4, sort=False)
    .to_string()
)
