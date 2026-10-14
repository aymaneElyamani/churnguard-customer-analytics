
import pandas as pd

path = "data/raw/"

customers = pd.read_csv(path + "olist_customers_dataset.csv")
orders = pd.read_csv(
    path + "olist_orders_dataset.csv",
    parse_dates=["order_purchase_timestamp"]
)

delivered = orders[orders["order_status"] == "delivered"].copy()

delivered = delivered.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    validate="one_to_one"
)

tests = [
    ("2017-06-30", 90),
    ("2017-09-30", 90),
    ("2017-12-31", 90),
    ("2018-03-31", 90),
    ("2017-06-30", 180),
    ("2017-09-30", 180),
]

for date, horizon in tests:
    cutoff = pd.Timestamp(date)
    start = cutoff - pd.Timedelta(days=180)
    end = cutoff + pd.Timedelta(days=horizon)

    last_date = delivered["order_purchase_timestamp"].max()

    if end > last_date:
        continue

    observation = delivered[
        (delivered["order_purchase_timestamp"] > start)
        & (delivered["order_purchase_timestamp"] <= cutoff)
    ]

    future = delivered[
        (delivered["order_purchase_timestamp"] > cutoff)
        & (delivered["order_purchase_timestamp"] <= end)
    ]

    # Nombre de commandes par client pendant l'observation
    order_counts = observation.groupby("customer_unique_id")[
        "order_id"
    ].nunique()

    # Population ciblée : clients ayant déjà commandé au moins 2 fois
    repeat_buyers = set(order_counts[order_counts >= 2].index)
    future_buyers = set(future["customer_unique_id"])

    total = len(repeat_buyers)
    repeat_again = len(repeat_buyers & future_buyers)
    no_repeat = total - repeat_again

    rate = no_repeat / total * 100 if total else 0

    print(f"\nDate de reference : {date}")
    print(f"Fenetre future : {horizon} jours")
    print(f"Clients avec au moins 2 commandes avant la reference : {total}")
    print(f"Avec une commande future : {repeat_again}")
    print(f"Sans commande future : {no_repeat}")
    print(f"Sans commande future (%) : {rate:.1f}%" if total else "Aucun client eligible")