
import pandas as pd

path = "data/raw/"

orders = pd.read_csv(path + "olist_orders_dataset.csv")
items = pd.read_csv(path + "olist_order_items_dataset.csv")
payments = pd.read_csv(path + "olist_order_payments_dataset.csv")

# Calculer les totaux séparément pour éviter les multiplications de lignes
item_totals = (
    items.groupby("order_id")
    .agg(
        total_articles=("price", "sum"),
        total_frais_livraison=("freight_value", "sum"),
        nombre_articles=("order_item_id", "count"),
    )
    .reset_index()
)

payment_totals = (
    payments.groupby("order_id")
    .agg(
        total_paiements=("payment_value", "sum"),
        nombre_paiements=("payment_sequential", "count"),
    )
    .reset_index()
)

comparison = (
    orders[["order_id", "order_status"]]
    .merge(item_totals, on="order_id", how="left", validate="one_to_one")
    .merge(payment_totals, on="order_id", how="left", validate="one_to_one")
)

print("Nombre de commandes :", len(comparison))
print("Commandes sans lignes d'articles :", comparison["total_articles"].isna().sum())
print("Commandes sans lignes de paiement :", comparison["total_paiements"].isna().sum())

# Comparer les montants pour les commandes livrées
delivered = comparison[comparison["order_status"] == "delivered"].copy()

delivered["total_attendu"] = (
    delivered["total_articles"] + delivered["total_frais_livraison"]
)

delivered["ecart"] = (
    delivered["total_paiements"] - delivered["total_attendu"]
).abs()

print("\n--- Ecarts entre paiements et articles + livraison ---")
print("Commandes livrées :", len(delivered))
print("Ecart median :", round(delivered["ecart"].median(), 2))
print("Ecart moyen :", round(delivered["ecart"].mean(), 2))
print("Ecart maximum :", round(delivered["ecart"].max(), 2))
print("Commandes avec ecart > 1 :", (delivered["ecart"] > 1).sum())

print("\n--- Exemples avec ecart > 1 ---")
columns = [
    "order_id",
    "total_articles",
    "total_frais_livraison",
    "total_paiements",
    "ecart",
]

print(
    delivered.loc[delivered["ecart"] > 1, columns]
    .sort_values("ecart", ascending=False)
    .head(10)
    .round(2)
    .to_string(index=False)
)