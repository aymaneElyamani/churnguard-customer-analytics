
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
    orders["order_status"] == "delivered"
].copy()

delivered = delivered.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="left"
)

cutoff_date = pd.Timestamp("2018-01-01")
prediction_end = pd.Timestamp("2018-06-30")

# Commandes livrées avant la date de coupure
past_orders = delivered[
    delivered["order_purchase_timestamp"] < cutoff_date
]

frequency = past_orders.groupby("customer_unique_id").size()

# Population : clients ayant déjà commandé au moins deux fois
repeat_customers = set(frequency[frequency >= 2].index)

# Commandes livrées pendant la période future
future_orders = delivered[
    (delivered["order_purchase_timestamp"] >= cutoff_date)
    & (delivered["order_purchase_timestamp"] < prediction_end)
]

returning_customers = set(
    future_orders["customer_unique_id"].dropna()
)

returned_count = len(repeat_customers & returning_customers)
no_return_count = len(repeat_customers) - returned_count


# Comparer les retours selon le statut de la commande
future_all_orders = orders[
    (orders["order_purchase_timestamp"] >= cutoff_date)
    & (orders["order_purchase_timestamp"] < prediction_end)
].merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="left"
)

returning_any_status = set(
    future_all_orders["customer_unique_id"].dropna()
)

returned_any_status = len(
    repeat_customers & returning_any_status
)

print("\n=== COMPARAISON DES DÉFINITIONS ===")
print("Clients avec une commande future, tous statuts :",
      returned_any_status)
print("Clients avec une commande future livrée :",
      returned_count)
print("Commandes futures de tous statuts :",
      len(future_all_orders))
print("Répartition des statuts futurs :")
print(future_all_orders["order_status"].value_counts())

print("=== CHURN DES CLIENTS RÉCURRENTS ===")
print("Clients récurrents :", len(repeat_customers))
print("Clients revenus :", returned_count)
print("Clients sans retour observé :", no_return_count)

if repeat_customers:
    print(
        "Taux sans retour observé :",
        round(no_return_count / len(repeat_customers) * 100, 2),
        "%"
    )