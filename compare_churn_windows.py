
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

last_purchase_date = delivered["order_purchase_timestamp"].max()

observation_days = 180
prediction_windows = [90, 180, 365]
reference_dates = [
    "2017-06-30",
    "2017-09-30",
    "2017-12-31",
    "2018-03-31",
    "2018-05-31",
]

print("Dernier achat livre dans le dataset :", last_purchase_date)
print()

for horizon in prediction_windows:
    print(f"===== Fenetre future : {horizon} jours =====")

    for date in reference_dates:
        cutoff = pd.Timestamp(date)
        observation_start = cutoff - pd.Timedelta(days=observation_days)
        prediction_end = cutoff + pd.Timedelta(days=horizon)

        if prediction_end > last_purchase_date:
            print(f"{date} : periode future incomplete, test ignore")
            continue

        observation = delivered[
            (delivered["order_purchase_timestamp"] > observation_start)
            & (delivered["order_purchase_timestamp"] <= cutoff)
        ]

        future = delivered[
            (delivered["order_purchase_timestamp"] > cutoff)
            & (delivered["order_purchase_timestamp"] <= prediction_end)
        ]

        eligible = set(observation["customer_unique_id"])
        future_clients = set(future["customer_unique_id"])

        total = len(eligible)
        repeat = len(eligible & future_clients)
        inactive = total - repeat

        if total:
            print(
                f"{date} | Clients: {total} | "
                f"Reachat: {repeat} | Sans reachat: {inactive} | "
                f"Sans reachat (%): {inactive / total * 100:.1f}%"
            )

    print()