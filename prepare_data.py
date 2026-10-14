
import pandas as pd

# Chemin vers les données brutes
path = "data/raw/"

# Charger les trois fichiers CSV
customers = pd.read_csv(path + "olist_customers_dataset.csv")
orders = pd.read_csv(path + "olist_orders_dataset.csv")
order_items = pd.read_csv(path + "olist_order_items_dataset.csv")

# Afficher les dimensions des tables
print("Customers :", customers.shape)
print("Orders :", orders.shape)
print("Order items :", order_items.shape)

# Afficher les premières lignes
print("\nAperçu des commandes :")
print(orders.head())


# Colonnes contenant des dates dans la table orders
date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

# Convertir les colonnes en dates
for column in date_columns:
    orders[column] = pd.to_datetime(orders[column], errors="coerce")

# Vérifier les types après conversion
print("\n=== TYPES DES COLONNES DE DATES ===")
print(orders[date_columns].dtypes)

# Vérifier les dates manquantes
print("\n=== DATES MANQUANTES ===")
print(orders[date_columns].isna().sum())


print("\n=== REPARTITION DES STATUTS ===")
print(orders["order_status"].value_counts())

print("\n=== NOMBRE DE CLIENTS UNIQUES ===")
print(customers["customer_unique_id"].nunique())

print("\n=== COMMANDES LIVREES ===")
delivered_orders = orders[orders["order_status"] == "delivered"]
print("Nombre de commandes livrées :", len(delivered_orders))
print(
    "Nombre de customer_id distincts :",
    delivered_orders["customer_id"].nunique()
)