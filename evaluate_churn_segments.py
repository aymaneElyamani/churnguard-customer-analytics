
import pandas as pd

DATA_PATH = "data/raw/"

customers = pd.read_csv(DATA_PATH + "olist_customers_dataset.csv")
orders = pd.read_csv(DATA_PATH + "olist_orders_dataset.csv")

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

# Conserver uniquement les commandes livrees
delivered = orders[
    orders["order_status"] == "delivered"
].copy()

# Associer chaque commande a l'identifiant client unique
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
prediction_horizons = [90, 180]

print("=" * 70)
print("CHURNGUARD - COMPARAISON DES CIBLES DE CHURN")
print("=" * 70)

for cutoff_text in cutoffs:
    cutoff = pd.Timestamp(cutoff_text)
    observation_start = cutoff - pd.Timedelta(
        days=observation_days
    )

    observation = delivered[
        (delivered["order_purchase_timestamp"] > observation_start)
        & (delivered["order_purchase_timestamp"] <= cutoff)
        & (delivered["customer_unique_id"].notna())
    ]

    # Nombre de commandes par client pendant l'observation
    counts = observation.groupby("customer_unique_id").size()

    # Tous les clients ayant au moins une commande
    all_ids = set(counts.index)

    # Clients ayant au moins deux commandes
    recurrent_ids = set(counts[counts >= 2].index)

    print(f"\n{'-' * 70}")
    print(f"DATE DE COUPURE : {cutoff_text}")
    print(f"{'-' * 70}")
    print(f"Tous les clients eligibles : {len(all_ids)}")
    print(f"Clients recurrents (>= 2 commandes) : {len(recurrent_ids)}")

    if all_ids:
        recurrent_pct = len(recurrent_ids) / len(all_ids) * 100
        print(f"Part de clients recurrents : {recurrent_pct:.2f}%")

    for horizon in prediction_horizons:
        prediction_end = cutoff + pd.Timedelta(days=horizon)

        future = delivered[
            (delivered["order_purchase_timestamp"] > cutoff)
            & (delivered["order_purchase_timestamp"] <= prediction_end)
            & (delivered["customer_unique_id"].notna())
        ]

        future_ids = set(future["customer_unique_id"])

        print(f"\nHORIZON : {horizon} jours")

        for segment_name, eligible_ids in [
            ("Tous les clients", all_ids),
            ("Clients recurrents", recurrent_ids),
        ]:
            returned_ids = eligible_ids & future_ids
            returned_count = len(returned_ids)
            eligible_count = len(eligible_ids)
            no_return_count = eligible_count - returned_count

            return_rate = (
                returned_count / eligible_count * 100
                if eligible_count else 0
            )

            no_return_rate = (
                no_return_count / eligible_count * 100
                if eligible_count else 0
            )

            print(f"\n  Segment : {segment_name}")
            print(f"  Clients eligibles : {eligible_count}")
            print(f"  Avec une commande future : {returned_count}")
            print(f"  Sans commande future : {no_return_count}")
            print(f"  Taux de retour : {return_rate:.2f}%")
            print(f"  Taux sans retour : {no_return_rate:.2f}%")

print("\nEvaluation terminee. Aucun fichier de donnees modifie.")
