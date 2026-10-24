
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CHURNGUARD - DIAGNOSTIC DES CARACTERISTIQUES V3
# Lecture seule : aucun dataset existant n'est modifie.
# ============================================================

DATA_PATH = Path("data/processed/churn_dataset_v3.csv")

FEATURES = [
    "frequence_180j",
    "montant_articles_180j",
    "recence_jours",
    "panier_moyen_180j",
]

LABELS = {
    0: "Retour observe",
    1: "Sans retour observe",
}


def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


# 1. Charger et verifier les donnees
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset introuvable : {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

required = FEATURES + ["customer_unique_id", "churn"]
missing_columns = set(required) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Colonnes manquantes : {missing_columns}"
    )

if df[required].isna().any().any():
    raise ValueError(
        "Des valeurs manquantes ont ete detectees."
    )

if not df["churn"].isin([0, 1]).all():
    raise ValueError("La cible churn doit contenir 0 et 1.")

if not df["customer_unique_id"].is_unique:
    raise ValueError("Des identifiants clients sont dupliques.")

if not df[FEATURES].apply(
    lambda column: pd.api.types.is_numeric_dtype(column)
).all():
    raise TypeError("Toutes les caracteristiques doivent etre numeriques.")


# 2. Vue generale
print_section("1. VUE GENERALE")

print(f"Nombre de clients : {len(df):,}")
print(f"Nombre de caracteristiques : {len(FEATURES)}")

print("\nRepartition de la cible :")

for value in [0, 1]:
    count = int((df["churn"] == value).sum())
    percentage = count / len(df) * 100

    print(
        f"- {LABELS[value]} (churn={value}) : "
        f"{count:,} clients ({percentage:.2f} %)"
    )


# 3. Statistiques descriptives globales et par classe
print_section("2. STATISTIQUES DESCRIPTIVES")

for feature in FEATURES:
    print(f"\n--- {feature} ---")

    print("\nTous les clients :")
    print(
        df[feature]
        .describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        .round(2)
        .to_string()
    )

    print("\nPar classe :")
    grouped = df.groupby("churn")[feature].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )

    grouped["q25"] = df.groupby("churn")[feature].quantile(0.25)
    grouped["q75"] = df.groupby("churn")[feature].quantile(0.75)
    grouped["q90"] = df.groupby("churn")[feature].quantile(0.90)

    grouped.index = [
        LABELS[value] for value in grouped.index
    ]

    print(grouped.round(2).to_string())


# 4. Comparer les moyennes et medianes entre les classes
print_section("3. DIFFERENCES ENTRE LES DEUX CLASSES")

for feature in FEATURES:
    returned = df.loc[df["churn"] == 0, feature]
    no_return = df.loc[df["churn"] == 1, feature]

    mean_return = returned.mean()
    mean_no_return = no_return.mean()

    median_return = returned.median()
    median_no_return = no_return.median()

    print(f"\n{feature}")
    print(
        f"  Moyenne - retour observe       : {mean_return:.2f}"
    )
    print(
        f"  Moyenne - sans retour observe  : {mean_no_return:.2f}"
    )
    print(
        f"  Mediane - retour observe      : {median_return:.2f}"
    )
    print(
        f"  Mediane - sans retour observe : {median_no_return:.2f}"
    )

    if mean_no_return != 0:
        difference_pct = (
            (mean_return - mean_no_return)
            / abs(mean_no_return)
        ) * 100

        print(
            f"  Ecart des moyennes             : "
            f"{difference_pct:+.2f} %"
        )


# 5. Distribution des valeurs
print_section("4. DISTRIBUTIONS PAR TRANCHES")

for feature in FEATURES:
    print(f"\n--- {feature} ---")

    try:
        bins = pd.qcut(
            df[feature],
            q=5,
            duplicates="drop",
        )

        distribution = pd.DataFrame({
            "tranche": bins,
            "churn": df["churn"],
        })

        summary = distribution.groupby(
            "tranche",
            observed=True,
        ).agg(
            clients=("churn", "size"),
            retours=("churn", lambda s: (s == 0).sum()),
            sans_retour=("churn", lambda s: (s == 1).sum()),
        )

        summary["taux_retour_pct"] = (
            summary["retours"] / summary["clients"] * 100
        )

        print(summary.round(2).to_string())

    except ValueError as error:
        print(
            "Distribution par tranches impossible :",
            error,
        )

# Distribution exacte de la frequence
print("\nFrequence exacte des commandes :")
frequency_table = pd.crosstab(
    df["frequence_180j"],
    df["churn"],
)

frequency_table = frequency_table.rename(
    columns=LABELS
)

print(frequency_table.to_string())


# 6. Valeurs extremes avec la methode IQR
print_section("5. DETECTION DES VALEURS EXTREMES")

print(
    "Regle IQR : valeur inferieure a Q1 - 1,5*IQR "
    "ou superieure a Q3 + 1,5*IQR."
)

for feature in FEATURES:
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (
        (df[feature] < lower_bound)
        | (df[feature] > upper_bound)
    )

    count = int(mask.sum())
    percentage = count / len(df) * 100

    print(f"\n{feature}")
    print(f"  Q1 : {q1:.2f}")
    print(f"  Q3 : {q3:.2f}")
    print(f"  IQR : {iqr:.2f}")
    print(
        f"  Limites statistiques : "
        f"[{lower_bound:.2f}, {upper_bound:.2f}]"
    )
    print(
        f"  Valeurs extremes : {count:,} "
        f"({percentage:.2f} %)"
    )

    if count > 0:
        print("  Plus grandes valeurs observees :")
        print(
            df.loc[mask, feature]
            .nlargest(5)
            .to_list()
        )


# 7. Valeurs impossibles ou incoherentes
print_section("6. CONTROLES DE COHERENCE")

checks = {
    "Frequence strictement positive": (
        df["frequence_180j"] > 0
    ).all(),
    "Montant articles non negatif": (
        df["montant_articles_180j"] >= 0
    ).all(),
    "Recence non negative": (
        df["recence_jours"] >= 0
    ).all(),
    "Panier moyen non negatif": (
        df["panier_moyen_180j"] >= 0
    ).all(),
    "Frequence entiere": (
        np.allclose(
            df["frequence_180j"],
            df["frequence_180j"].round(),
        )
    ),
}

for name, passed in checks.items():
    status = "OK" if passed else "A VERIFIER"
    print(f"[{status}] {name}")


# 8. Correlations avec la cible
print_section("7. CORRELATIONS AVEC LA CIBLE")

print(
    "Attention : churn=1 signifie absence de retour observe. "
    "La correlation ne prouve pas une relation causale."
)

correlations = (
    df[FEATURES + ["churn"]]
    .corr(numeric_only=True)["churn"]
    .drop("churn")
    .sort_values()
)

print(correlations.round(4).to_string())


# 9. Conclusion automatique prudente
print_section("8. RAPPEL POUR L'INTERPRETATION")

print("- Compare les medianes, pas seulement les moyennes.")
print("- Verifie si les distributions des deux classes se chevauchent.")
print("- Une valeur extreme n'est pas automatiquement une erreur.")
print("- Une correlation faible n'exclut pas une relation non lineaire.")
print("- Le taux de retour est faible : les statistiques sur cette classe")
print("  reposent sur beaucoup moins de clients.")
print("- Ce diagnostic est descriptif : il ne prouve pas qu'une")
print("  caracteristique cause le retour ou le non-retour.")

print("\nDiagnostic termine. Aucun fichier n'a ete modifie.")
