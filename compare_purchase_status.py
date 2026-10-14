
import pandas as pd

path = "data/raw/"

customers = pd.read_csv(path + "olist_customers_dataset.csv")
orders = pd.read_csv(
    path + "olist_orders_dataset.csv",
    parse_dates=["order_purchase_timestamp"]
)

orders = orders.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    validate="one_to_one"
)

cutoff = pd.Timestamp("2017-12-31")
start = cutoff - pd.Timedelta(days=180)
end = cutoff + pd.Timedelta(days=90)

observation = orders[
    (orders["order_purchase_timestamp"] > start)
    & (orders["order_purchase_timestamp"] <= cutoff)
    & (orders["order_status"] == "delivered")
]

eligible = set(observation["customer_unique_id"])

future = orders[
    (orders["order_purchase_timestamp"] > cutoff)
    & (orders["order_purchase_timestamp"] <= end)
]

future_delivered = set(
    future.loc[
        future["order_status"] == "delivered",
        "customer_unique_id"
    ]
)

# Toutes les nouvelles commandes, quel que soit leur statut
future_any = set(future["customer_unique_id"])

print("Date de reference : 2017-12-31")
print("Clients eligibles :", len(eligible))
print()

for label, future_clients in [
    ("Commande future livree", future_delivered),
    ("Commande future tous statuts", future_any),
]:
    bought_again = len(eligible & future_clients)
    no_purchase = len(eligible) - bought_again
    rate = no_purchase / len(eligible) * 100 if eligible else 0

    print(label)
    print("  Nouvelle commande :", bought_again)
    print("  Sans nouvelle commande :", no_purchase)
    print(f"  Sans nouvelle commande (%) : {rate:.1f}%")