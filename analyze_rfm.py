
import pandas as pd

# Réutiliser la table de caractéristiques construite précédemment
path = "data/raw/"

customers = pd.read_csv(path + "olist_customers_dataset.csv")
orders = pd.read_csv(path + "olist_orders_dataset.csv")
items = pd.read_csv(path + "olist_order_items_dataset.csv")

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

delivered = orders[orders["order_status"] == "delivered"].merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="left",
    validate="one_to_one"
)

item_totals = (
    items.groupby("order_id")
    .agg(montant_articles=("price", "sum"))
    .reset_index()
)

delivered = delivered.merge(
    item_totals,
    on="order_id",
    how="left",
    validate="one_to_one"
)

cutoff = pd.Timestamp("2017-12-31")
start = cutoff - pd.Timedelta(days=180)

observation = delivered[
    (delivered["order_purchase_timestamp"] > start)
    & (delivered["order_purchase_timestamp"] <= cutoff)
]

rfm = (
    observation.groupby("customer_unique_id")
    .agg(
        frequence=("order_id", "nunique"),
        montant=("montant_articles", "sum"),
        derniere_commande=("order_purchase_timestamp", "max"),
    )
    .reset_index()
)

# Garder les clients récurrents pour cette première analyse
rfm = rfm[rfm["frequence"] >= 2].copy()

rfm["recence_jours"] = (
    cutoff - rfm["derniere_commande"]
).dt.days

# Décrire les profils sans imposer de seuil métier arbitraire
rfm["groupe_recence"] = pd.cut(
    rfm["recence_jours"],
    bins=[0, 45, 90, 135, 180],
    labels=["0-45 jours", "46-90 jours", "91-135 jours", "136-180 jours"],
    include_lowest=True
)

print("=== REPARTITION PAR RECENCE ===")
print(rfm["groupe_recence"].value_counts(sort=False))

print("\n=== FREQUENCE DES COMMANDES ===")
print(rfm["frequence"].value_counts().sort_index())

print("\n=== PROFIL GLOBAL ===")
print(
    rfm[["recence_jours", "frequence", "montant"]]
    .describe()
    .round(2)
    .to_string()
)


import numpy as np

# Calculer le rang de chaque client en pourcentage
rang_recence = rfm["recence_jours"].rank(
    method="average", pct=True
)
rang_frequence = rfm["frequence"].rank(
    method="average", pct=True
)
rang_montant = rfm["montant"].rank(
    method="average", pct=True
)

# Transformer les rangs en scores de 1 à 5
# Pour la récence, moins de jours = meilleur score
rfm["R_score"] = 6 - np.ceil(rang_recence * 5).astype(int)

# Pour la fréquence et le montant, plus = meilleur score
rfm["F_score"] = np.ceil(rang_frequence * 5).astype(int)
rfm["M_score"] = np.ceil(rang_montant * 5).astype(int)

# Score RFM combiné
rfm["score_RFM"] = (
    rfm["R_score"].astype(str)
    + rfm["F_score"].astype(str)
    + rfm["M_score"].astype(str)
)

print("\n=== EXEMPLE DE SCORES RFM ===")
print(
    rfm[
        [
            "recence_jours",
            "frequence",
            "montant",
            "R_score",
            "F_score",
            "M_score",
            "score_RFM",
        ]
    ].head(10)
)

print("\n=== DISTRIBUTION DES SCORES ===")
print(rfm[["R_score", "F_score", "M_score"]].describe())


def segmenter_client(client):
    # Récence élevée = commande récente
    if client["R_score"] >= 4 and client["M_score"] >= 4:
        return "Clients récents à forte valeur"

    elif client["R_score"] >= 4:
        return "Clients récents"

    elif client["R_score"] <= 2 and client["M_score"] >= 4:
        return "Clients à forte valeur à réactiver"

    elif client["R_score"] <= 2:
        return "Clients à surveiller"

    else:
        return "Clients intermédiaires"


rfm["segment"] = rfm.apply(segmenter_client, axis=1)

print("\n=== NOMBRE DE CLIENTS PAR SEGMENT ===")
print(rfm["segment"].value_counts())

print("\n=== PROFIL MOYEN PAR SEGMENT ===")
print(
    rfm.groupby("segment")[
        ["recence_jours", "frequence", "montant"]
    ].mean().round(2)
)


import matplotlib.pyplot as plt

# Compter les clients dans chaque segment
nombre_par_segment = rfm["segment"].value_counts()

# Créer le diagramme en barres
plt.figure(figsize=(11, 6))

nombre_par_segment.plot(kind="bar")

# Afficher le nombre de clients au-dessus de chaque barre
ax = plt.gca()
ax.bar_label(ax.containers[0], fmt="%.0f", padding=3)

plt.title("Répartition des clients par segment RFM")
plt.xlabel("Segment client")
plt.ylabel("Nombre de clients")
plt.xticks(rotation=25, ha="right")

plt.tight_layout()

# Créer le dossier des graphiques s'il n'existe pas
from pathlib import Path

output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Enregistrer le graphique
plt.savefig(
    output_dir / "rfm_segments.png",
    dpi=300,
    bbox_inches="tight"
)

print("Graphique sauvegardé dans reports/figures/rfm_segments.png")

plt.show()