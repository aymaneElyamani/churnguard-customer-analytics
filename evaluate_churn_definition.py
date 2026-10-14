
import pandas as pd

path = "data/raw/"

customers = pd.read_csv(path + "olist_customers_dataset.csv")
orders = pd.read_csv(path + "olist_orders_dataset.csv")

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

# Ne conserver que les commandes livrées
delivered = orders[orders["order_status"] == "delivered"].copy()

# Relier les commandes à l'identifiant client unique
delivered = delivered.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="left",
    validate="one_to_one"
)

cutoffs = [
    "2017-06-30",
    "2017-09-30",
    "2017-12-31",
]

observation_days = 180
prediction_days = 90

print("=== EVALUATION DE LA DEFINITION DU CHURN ===")

for cutoff_text in cutoffs:
    cutoff = pd.Timestamp(cutoff_text)
    observation_start = cutoff - pd.Timedelta(days=observation_days)
    prediction_end = cutoff + pd.Timedelta(days=prediction_days)

    observation = delivered[
        (delivered["order_purchase_timestamp"] > observation_start)
        & (delivered["order_purchase_timestamp"] <= cutoff)
    ]

    future = delivered[
        (delivered["order_purchase_timestamp"] > cutoff)
        & (delivered["order_purchase_timestamp"] <= prediction_end)
    ]

    # Clients ayant au moins deux commandes dans l'observation
    counts = observation.groupby("customer_unique_id").size()
    eligible_ids = counts[counts >= 2].index

    # Parmi ces clients, ceux qui rachètent dans les 90 jours
    future_ids = set(future["customer_unique_id"])
    repeat_ids = set(eligible_ids) & future_ids

    eligible_count = len(eligible_ids)
    repeat_count = len(repeat_ids)
    no_repeat_count = eligible_count - repeat_count

    print(f"\nDate de coupure : {cutoff_text}")
    print("Clients éligibles :", eligible_count)
    print("Nouvelle commande dans les 90 jours :", repeat_count)
    print("Aucune nouvelle commande dans les 90 jours :", no_repeat_count)

    if eligible_count > 0:
        print(
            "Taux sans nouvelle commande :",
            round(no_repeat_count / eligible_count * 100, 2),
            "%"
        )