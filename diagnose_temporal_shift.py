
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

FEATURES = [
    "frequence_180j",
    "montant_articles_180j",
    "recence_jours",
    "panier_moyen_180j",
]

v1 = pd.read_csv("data/processed/churn_dataset_v1.csv")
v2 = pd.read_csv("data/processed/churn_dataset_v2.csv")

# Reprendre la même partition train V1 que dans les expériences précédentes.
X1 = v1[FEATURES]
y1 = v1["churn"]

X_train, _, y_train, _ = train_test_split(
    X1,
    y1,
    test_size=0.40,
    random_state=42,
    stratify=y1,
)

train = X_train.copy()
future = v2[FEATURES].copy()

print("=" * 75)
print("CHURNGUARD - DIAGNOSTIC DU CHANGEMENT TEMPOREL")
print("=" * 75)

print("\n1. Comparaison des caractéristiques")
print("SMD = différence des moyennes divisée par l'écart-type combiné.")
print("Une valeur absolue élevée signale un changement à examiner.\n")

for feature in FEATURES:
    mean_train = train[feature].mean()
    mean_future = future[feature].mean()

    std_train = train[feature].std()
    std_future = future[feature].std()

    pooled_std = np.sqrt(
        (std_train**2 + std_future**2) / 2
    )

    smd = (
        (mean_future - mean_train) / pooled_std
        if pooled_std > 0 else 0
    )

    print(f"--- {feature} ---")
    print(f"Moyenne V1 train : {mean_train:.2f}")
    print(f"Moyenne V2       : {mean_future:.2f}")
    print(f"Médiane V1 train : {train[feature].median():.2f}")
    print(f"Médiane V2       : {future[feature].median():.2f}")
    print(f"SMD              : {smd:.3f}")
    print()

print("\n2. Répartition de la fréquence d'achat")
print("Pourcentage de clients par fréquence observée :\n")

freq_train = (
    train["frequence_180j"]
    .value_counts(normalize=True)
    .mul(100)
    .rename("V1_train_pct")
)

freq_future = (
    future["frequence_180j"]
    .value_counts(normalize=True)
    .mul(100)
    .rename("V2_pct")
)

comparison = pd.concat([freq_train, freq_future], axis=1).fillna(0)
comparison.index.name = "frequence_180j"

print(comparison.sort_index().round(2).to_string())

print("\n3. Taux de retour observé")
print(f"V1 train : {(y_train == 0).mean():.2%}")
print(f"V2       : {(v2['churn'] == 0).mean():.2%}")

print("\nDiagnostic terminé. Aucun fichier n'a été modifié.")
