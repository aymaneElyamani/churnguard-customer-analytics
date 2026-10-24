
import pandas as pd

data_path = "data/raw/"

customers = pd.read_csv(
    data_path + "olist_customers_dataset.csv"
)

orders = pd.read_csv(
    data_path + "olist_orders_dataset.csv"
)

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

delivered = orders[
    (orders["order_status"] == "delivered")
    & (orders["order_purchase_timestamp"] < "2018-01-01")
].copy()

delivered = delivered.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="left"
)

frequency = (
    delivered.groupby("customer_unique_id")
    .size()
)

print("=== CLIENTS AVANT LE 1ER JANVIER 2018 ===")
print("Clients avec au moins une commande :", len(frequency))
print("Clients avec au moins deux commandes :", (frequency >= 2).sum())
print("Clients avec au moins trois commandes :", (frequency >= 3).sum())

print("\n=== DISTRIBUTION DES COMMANDES ===")
print(frequency.value_counts().sort_index().head(10))