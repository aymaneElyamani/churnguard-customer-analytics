
import pandas as pd

# 1. Charger les clients et les commandes
path = "data/raw/"

customers = pd.read_csv(path + "olist_customers_dataset.csv")

orders = pd.read_csv(
    path + "olist_orders_dataset.csv",
    parse_dates=["order_purchase_timestamp"]
)

# 2. Garder uniquement les commandes livrées
delivered = orders[orders["order_status"] == "delivered"].copy()

# 3. Associer chaque commande au véritable identifiant client
delivered = delivered.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    validate="one_to_one"
)

# 4. Tester plusieurs dates de référence
reference_dates = [
    "2017-06-30",
    "2017-09-30",
    "2017-12-31",
    "2018-03-31",
    "2018-05-31",
]

print("Date reference | Clients eligibles | Churn 1 | Churn 0 | Churn %")
print("-" * 75)

for date in reference_dates:
    cutoff = pd.Timestamp(date)
    observation_start = cutoff - pd.Timedelta(days=180)
    prediction_end = cutoff + pd.Timedelta(days=90)

    # Clients ayant au moins une commande livrée
    # pendant les 180 jours précédant la date de référence
    observation = delivered[
        (delivered["order_purchase_timestamp"] > observation_start)
        & (delivered["order_purchase_timestamp"] <= cutoff)
    ]

    # Clients ayant au moins une commande livrée
    # pendant les 90 jours suivants
    future = delivered[
        (delivered["order_purchase_timestamp"] > cutoff)
        & (delivered["order_purchase_timestamp"] <= prediction_end)
    ]

    eligible_clients = set(observation["customer_unique_id"])
    future_clients = set(future["customer_unique_id"])

    total = len(eligible_clients)
    churn_1 = len(eligible_clients - future_clients)
    churn_0 = len(eligible_clients & future_clients)

    churn_pct = churn_1 / total * 100 if total else 0

    print(
        f"{date} | {total} | {churn_1} | "
        f"{churn_0} | {churn_pct:.1f}%"
    )