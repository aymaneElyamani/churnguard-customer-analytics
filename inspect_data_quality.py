
import pandas as pd

path = "data/raw/"

files = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
}

for name, filename in files.items():
    df = pd.read_csv(path + filename)

    print(f"\n{'=' * 50}")
    print(f"TABLE : {name}")
    print(f"{'=' * 50}")

    print("Nombre de lignes :", len(df))
    print("Nombre de colonnes :", len(df.columns))
    print("Doublons complets :", df.duplicated().sum())

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print("\nValeurs manquantes :")
    if missing.empty:
        print("Aucune")
    else:
        for column, count in missing.items():
            percentage = count / len(df) * 100
            print(f"{column}: {count} ({percentage:.2f}%)")

    print("\nTypes de données :")
    print(df.dtypes.to_string())