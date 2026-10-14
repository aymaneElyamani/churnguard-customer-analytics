
import pandas as pd

path = "data/raw/"

customers = pd.read_csv(path + "olist_customers_dataset.csv")
orders = pd.read_csv(path + "olist_orders_dataset.csv")
items = pd.read_csv(path + "olist_order_items_dataset.csv")
payments = pd.read_csv(path + "olist_order_payments_dataset.csv")

print("===== 1. UNICITE DES IDENTIFIANTS =====")

print("customer_id duplique :", customers["customer_id"].duplicated().sum())
print("order_id duplique dans orders :", orders["order_id"].duplicated().sum())

item_key_duplicates = items.duplicated(
    subset=["order_id", "order_item_id"]
).sum()
print("Couples order_id/order_item_id dupliques :", item_key_duplicates)

print("\n===== 2. RELATIONS ENTRE LES TABLES =====")

customer_ids = set(customers["customer_id"])
order_customer_ids = set(orders["customer_id"])

print(
    "Commandes avec customer_id absent de customers :",
    len(order_customer_ids - customer_ids)
)

order_ids = set(orders["order_id"])

print(
    "Lignes order_items avec order_id absent de orders :",
    (~items["order_id"].isin(order_ids)).sum()
)

print(
    "Lignes payments avec order_id absent de orders :",
    (~payments["order_id"].isin(order_ids)).sum()
)

print("\n===== 3. NOMBRE D'ARTICLES ET DE PAIEMENTS =====")

items_per_order = items.groupby("order_id").size()
payments_per_order = payments.groupby("order_id").size()

print("Commandes avec plusieurs lignes d'articles :", (items_per_order > 1).sum())
print("Maximum de lignes d'articles par commande :", items_per_order.max())

print("Commandes avec plusieurs lignes de paiement :", (payments_per_order > 1).sum())
print("Maximum de lignes de paiement par commande :", payments_per_order.max())

print("\n===== 4. CONTROLES DES MONTANTS =====")

print("Prix négatifs :", (items["price"] < 0).sum())
print("Prix nuls :", (items["price"] == 0).sum())
print("Frais de livraison négatifs :", (items["freight_value"] < 0).sum())

print("Paiements négatifs :", (payments["payment_value"] < 0).sum())
print("Paiements nuls :", (payments["payment_value"] == 0).sum())

print("\n===== 5. STATUTS DES COMMANDES =====")
print(orders["order_status"].value_counts().to_string())