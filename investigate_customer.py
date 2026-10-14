
import pandas as pd

path = "data/raw/"
order_id = "ce6d150fb29ada17d2082f4847107665"

orders = pd.read_csv(path + "olist_orders_dataset.csv")
customers = pd.read_csv(path + "olist_customers_dataset.csv")
items = pd.read_csv(path + "olist_order_items_dataset.csv")
payments = pd.read_csv(path + "olist_order_payments_dataset.csv")

# Identifier le client de la commande étudiée
customer_id = orders.loc[
    orders["order_id"] == order_id, "customer_id"
].iloc[0]

customer_unique_id = customers.loc[
    customers["customer_id"] == customer_id, "customer_unique_id"
].iloc[0]

# Retrouver tous les identifiants associés à ce client
customer_ids = customers.loc[
    customers["customer_unique_id"] == customer_unique_id,
    "customer_id"
]

customer_orders = orders[orders["customer_id"].isin(customer_ids)].copy()

# Calculer les montants par commande, séparément
item_totals = (
    items.groupby("order_id")
    .agg(
        total_articles=("price", "sum"),
        frais_livraison=("freight_value", "sum")
    )
)

payment_totals = (
    payments.groupby("order_id")
    .agg(total_paye=("payment_value", "sum"))
)

customer_orders = customer_orders.join(item_totals, on="order_id")
customer_orders = customer_orders.join(payment_totals, on="order_id")

customer_orders["total_attendu"] = (
    customer_orders["total_articles"]
    + customer_orders["frais_livraison"]
)

customer_orders["ecart"] = (
    customer_orders["total_paye"]
    - customer_orders["total_attendu"]
)

print("Nombre de customer_id associés :", len(customer_ids))
print("Nombre de commandes du client :", len(customer_orders))
print("\n=== HISTORIQUE DES COMMANDES ===")

columns = [
    "order_id",
    "order_status",
    "order_purchase_timestamp",
    "total_articles",
    "frais_livraison",
    "total_paye",
    "ecart"
]

print(
    customer_orders[columns]
    .sort_values("order_purchase_timestamp")
    .round(2)
    .to_string(index=False)
)
print("\n=== VERIFICATION DES IDENTIFIANTS ===")

print(
    "Nombre total de customer_id :",
    customers["customer_id"].nunique()
)

print(
    "Nombre total de customer_unique_id :",
    customers["customer_unique_id"].nunique()
)

print(
    "Nombre de customer_unique_id associés à plusieurs customer_id :",
    (customers.groupby("customer_unique_id")["customer_id"].nunique() > 1).sum()
)