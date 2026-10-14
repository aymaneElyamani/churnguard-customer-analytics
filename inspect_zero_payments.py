
import pandas as pd

path = "data/raw/"

payments = pd.read_csv(path + "olist_order_payments_dataset.csv")

orders = pd.read_csv(
    path + "olist_orders_dataset.csv",
    parse_dates=["order_purchase_timestamp"]
)

print("===== 1. PAIEMENTS NULS =====")

zero_payments = payments[payments["payment_value"] == 0]

print("Nombre de lignes :", len(zero_payments))

if not zero_payments.empty:
    details = zero_payments.merge(
        orders[
            [
                "order_id",
                "order_status",
                "order_purchase_timestamp"
            ]
        ],
        on="order_id",
        how="left",
        validate="many_to_one"
    )

    print("\nDetails des paiements nuls :")
    print(
        details[
            [
                "order_id",
                "payment_sequential",
                "payment_type",
                "payment_installments",
                "payment_value",
                "order_status",
                "order_purchase_timestamp"
            ]
        ].to_string(index=False)
    )

    print("\nStatuts des commandes concernees :")
    print(details["order_status"].value_counts(dropna=False).to_string())

print("\n===== 2. TYPES DE PAIEMENT =====")
print(payments["payment_type"].value_counts(dropna=False).to_string())

print("\n===== 3. DOUBLONS DE SEQUENCE DE PAIEMENT =====")
duplicates = payments.duplicated(
    subset=["order_id", "payment_sequential"]
)

print("Couples order_id/payment_sequential dupliques :", duplicates.sum())

print("\n===== 4. REPARTITION DES VALEURS DE PAIEMENT =====")
print(payments["payment_value"].describe().round(2).to_string())