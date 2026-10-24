
import pandas as pd

data_path = "data/raw/"

# Charger les données nécessaires
customers = pd.read_csv(
    data_path + "olist_customers_dataset.csv"
)

orders = pd.read_csv(
    data_path + "olist_orders_dataset.csv"
)

# Convertir la date de commande
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

# Garder uniquement les commandes livrées
delivered_orders = orders[
    orders["order_status"] == "delivered"
].copy()

# Ajouter l'identifiant unique du client
delivered_orders = delivered_orders.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="left"
)

# Extraire le mois de chaque commande
delivered_orders["order_month"] = (
    delivered_orders["order_purchase_timestamp"]
    .dt.to_period("M")
)

print("=== PRÉPARATION DES COHORTES ===")
print("Commandes livrées :", len(delivered_orders))
print(
    "Clients uniques :",
    delivered_orders["customer_unique_id"].nunique()
)
print(
    "Première commande :",
    delivered_orders["order_purchase_timestamp"].min()
)
print(
    "Dernière commande :",
    delivered_orders["order_purchase_timestamp"].max()
)
print("\n=== EXEMPLE DE DONNÉES ===")
print(
    delivered_orders[
        [
            "customer_unique_id",
            "order_purchase_timestamp",
            "order_month"
        ]
    ].head(5)
)


# Première commande livrée de chaque client
first_orders = (
    delivered_orders
    .groupby("customer_unique_id")["order_purchase_timestamp"]
    .min()
    .reset_index()
)

# Mois de la première commande = cohorte du client
first_orders["cohort_month"] = (
    first_orders["order_purchase_timestamp"]
    .dt.to_period("M")
)

print("\n=== PREMIÈRES COHORTES ===")
print(
    first_orders[
        ["customer_unique_id", "order_purchase_timestamp", "cohort_month"]
    ].head(10)
)

print("\n=== NOMBRE DE CLIENTS PAR COHORTE ===")
print(
    first_orders["cohort_month"]
    .value_counts()
    .sort_index()
)

print("\nNombre total de clients dans les cohortes :", len(first_orders))


# Associer chaque commande à la cohorte du client
cohort_orders = delivered_orders.merge(
    first_orders[["customer_unique_id", "cohort_month"]],
    on="customer_unique_id",
    how="left"
)

# Mois de la commande
cohort_orders["order_month"] = (
    cohort_orders["order_purchase_timestamp"].dt.to_period("M")
)

# Calculer le nombre de mois écoulés depuis la première commande
cohort_orders["cohort_index"] = (
    (cohort_orders["order_month"].dt.year
     - cohort_orders["cohort_month"].dt.year) * 12
    + (cohort_orders["order_month"].dt.month
       - cohort_orders["cohort_month"].dt.month)
)


# Dernier mois réellement présent dans les données
last_month = delivered_orders["order_month"].max()

# Nombre maximal de mois écoulés depuis chaque cohorte
max_index = (
    (last_month.year - first_orders["cohort_month"].min().year) * 12
    + last_month.month - first_orders["cohort_month"].min().month
)

# Nombre de clients actifs par cohorte et mois écoulé
cohort_counts = (
    cohort_orders
    .groupby(["cohort_month", "cohort_index"])["customer_unique_id"]
    .nunique()
    .unstack(fill_value=0)
)

# Ajouter les colonnes des mois sans aucune commande
cohort_counts = cohort_counts.reindex(
    columns=range(max_index + 1),
    fill_value=0
)

# Masquer les mois qui n'étaient pas encore observables
for cohort_month in cohort_counts.index:
    last_observed_index = (
        (last_month.year - cohort_month.year) * 12
        + last_month.month - cohort_month.month
    )

    cohort_counts.loc[
        cohort_month,
        cohort_counts.columns > last_observed_index
    ] = float("nan")

# Calculer la rétention en pourcentage
retention = cohort_counts.div(cohort_counts[0], axis=0) * 100

print("\n=== RÉTENTION (%) : PREMIÈRES COHORTES ===")
print(retention.head(5).round(1).to_string())

print("\n=== DERNIÈRES COHORTES ===")
print(retention.tail(5).round(1).to_string())


print("\n=== CONTRÔLE DE LA RÉTENTION ===")

# Exemple : cohorte de janvier 2017
example_cohort = pd.Period("2017-01", freq="M")

example_data = cohort_counts.loc[example_cohort]

print("Cohorte :", example_cohort)
print("Clients initiaux :", int(example_data[0]))

for month_index in range(1, 7):
    value = example_data.get(month_index)

    if pd.notna(value):
        print(
            f"Mois {month_index} : "
            f"{int(value)} clients revenus "
            f"({value / example_data[0] * 100:.2f} %)"
        )
    else:
        print(f"Mois {month_index} : période non observable")


import matplotlib.pyplot as plt
import numpy as np

# Préparer les données pour la visualisation
retention_values = retention.to_numpy()

# Créer la figure
fig, ax = plt.subplots(figsize=(16, 10))


# Ignorer les périodes non observables
retention_values = retention.to_numpy(dtype=float)
masked_values = np.ma.masked_invalid(retention_values)

fig, ax = plt.subplots(figsize=(16, 10))

image = ax.imshow(
    masked_values,
    aspect="auto",
    interpolation="nearest",
    cmap="YlGnBu",
    vmin=0,
    vmax=5
)


# Ajouter les étiquettes des axes
ax.set_title("Analyse de la rétention client par cohorte", fontsize=16)
ax.set_xlabel("Mois depuis la première commande")
ax.set_ylabel("Cohorte d'acquisition")

ax.set_xticks(range(len(retention.columns)))
ax.set_xticklabels(retention.columns)

ax.set_yticks(range(len(retention.index)))
ax.set_yticklabels([str(month) for month in retention.index])

# Ajouter une barre de couleur
fig.colorbar(image, ax=ax, label="Rétention (%)")

plt.tight_layout()

# Enregistrer le graphique
plt.savefig(
    "reports/figures/cohort_retention.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Heatmap enregistrée dans reports/figures/cohort_retention.png")
