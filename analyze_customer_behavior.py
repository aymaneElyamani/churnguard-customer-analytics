
import pandas as pd

# Charger la table préparée
customers = pd.read_csv("data/processed/customer_features.csv")

print("=== REPARTITION DES CLIENTS ===")

distribution = (
    customers["nombre_commandes"]
    .value_counts()
    .sort_index()
)

print(distribution.to_string())

print("\n=== CLIENTS SELON LE NOMBRE DE COMMANDES ===")

total = len(customers)

one_order = (customers["nombre_commandes"] == 1).sum()
repeat_customers = (customers["nombre_commandes"] >= 2).sum()

print("Total clients :", total)
print("Clients avec une seule commande :", one_order)
print("Clients avec plusieurs commandes :", repeat_customers)

print(
    "Pourcentage avec une seule commande :",
    round(one_order / total * 100, 2),
    "%"
)

print(
    "Pourcentage avec plusieurs commandes :",
    round(repeat_customers / total * 100, 2),
    "%"
)