
import pandas as pd

path = "data/raw/"

# Charger les tables nécessaires
customers = pd.read_csv(path + "olist_customers_dataset.csv")
orders = pd.read_csv(path + "olist_orders_dataset.csv")
items = pd.read_csv(path + "olist_order_items_dataset.csv")

# Convertir la date d'achat
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

# Garder uniquement les commandes livrées
delivered_orders = orders[
    orders["order_status"] == "delivered"
].copy()

# Associer chaque commande à son client unique
delivered_orders = delivered_orders.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="left",
    validate="one_to_one"
)

# Calculer le montant des articles par commande
item_totals = (
    items.groupby("order_id")
    .agg(montant_articles=("price", "sum"))
    .reset_index()
)

# Ajouter le montant à chaque commande
delivered_orders = delivered_orders.merge(
    item_totals,
    on="order_id",
    how="left",
    validate="one_to_one"
)

# Regrouper les commandes par client unique
customer_table = (
    delivered_orders
    .groupby("customer_unique_id")
    .agg(
        nombre_commandes=("order_id", "nunique"),
        montant_total_articles=("montant_articles", "sum"),
        premiere_commande=("order_purchase_timestamp", "min"),
        derniere_commande=("order_purchase_timestamp", "max"),
    )
    .reset_index()
)

# Afficher les premiers résultats
print("Nombre de clients :", len(customer_table))
print("\nAperçu de la table clients :")
print(customer_table.head().to_string(index=False))

print("\nStatistiques du nombre de commandes par client :")
print(customer_table["nombre_commandes"].describe())


print("\n=== CONTROLES QUALITE ===")

print(
    "Nombre de lignes :",
    len(customer_table)
)

print(
    "Nombre de clients uniques :",
    customer_table["customer_unique_id"].nunique()
)

print(
    "Identifiants clients manquants :",
    customer_table["customer_unique_id"].isna().sum()
)

print(
    "Montants totaux manquants :",
    customer_table["montant_total_articles"].isna().sum()
)

print(
    "Montants totaux négatifs :",
    (customer_table["montant_total_articles"] < 0).sum()
)

print(
    "Dates de dernière commande manquantes :",
    customer_table["derniere_commande"].isna().sum()
)


from pathlib import Path

# Créer le dossier de sortie s'il n'existe pas
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Sauvegarder la table préparée
output_file = output_dir / "customer_features.csv"

customer_table.to_csv(output_file, index=False)

print("\n=== SAUVEGARDE ===")
print("Fichier créé :", output_file)
print("Nombre de lignes sauvegardées :", len(customer_table))