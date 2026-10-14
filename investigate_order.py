import pandas as pd

path = "data/raw/"

orders = pd.read_csv(path + "olist_orders_dataset.csv")
items = pd.read_csv(path + "olist_order_items_dataset.csv")
payments = pd.read_csv(path + "olist_order_payments_dataset.csv")

# Commande présentant le plus grand écart
order_id = "ce6d150fb29ada17d2082f4847107665"

print("=== COMMANDE ===")
print(
    orders.loc[orders["order_id"] == order_id].to_string(index=False)
)

print("\n=== ARTICLES ===")
order_items = items[items["order_id"] == order_id]
print(order_items.to_string(index=False))

print("\n=== TOTAL DES ARTICLES ET LIVRAISON ===")
print("Articles :", round(order_items["price"].sum(), 2))
print("Livraison :", round(order_items["freight_value"].sum(), 2))

print("\n=== PAIEMENTS ===")
order_payments = payments[payments["order_id"] == order_id]
print(order_payments.to_string(index=False))

print("\nTotal payé :", round(order_payments["payment_value"].sum(), 2))
print("Nombre de lignes de paiement :", len(order_payments))
