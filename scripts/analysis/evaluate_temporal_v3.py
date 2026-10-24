
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "frequence_180j",
    "montant_articles_180j",
    "recence_jours",
    "panier_moyen_180j",
]

V3_PATH = Path("data/processed/churn_dataset_v3.csv")
V4_PATH = Path("data/processed/churn_dataset_v4.csv")


# 1. Charger et verifier les datasets
v3 = pd.read_csv(V3_PATH)
v4 = pd.read_csv(V4_PATH)

required = FEATURES + ["customer_unique_id", "churn"]

for name, df in [("V3", v3), ("V4", v4)]:
    missing_columns = set(required) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{name} : colonnes manquantes {missing_columns}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            f"{name} : valeurs manquantes detectees"
        )

    if not df["churn"].isin([0, 1]).all():
        raise ValueError(
            f"{name} : la cible doit contenir 0 et 1"
        )

    if not df["customer_unique_id"].is_unique:
        raise ValueError(
            f"{name} : identifiants clients dupliques"
        )


# 2. Separ­er V3 en train, validation et test interne
X3 = v3[FEATURES]
y3 = v3["churn"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X3,
    y3,
    test_size=0.40,
    random_state=42,
    stratify=y3,
)

X_val, _, y_val, _ = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

# Classe positive pour les metriques : retour observe (churn == 0)
y_val_return = (y_val == 0).astype(int)


# 3. Entrainer le modele uniquement sur le train V3
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )),
])

model.fit(X_train, y_train)

classes = list(model.named_steps["classifier"].classes_)
return_column = classes.index(0)


# 4. Choisir le seuil sur la validation V3 uniquement
val_scores = model.predict_proba(X_val)[:, return_column]
threshold_results = []

for threshold_candidate in np.arange(0.05, 0.51, 0.05):
    val_predictions = (
        val_scores >= threshold_candidate
    ).astype(int)

    f1 = f1_score(
        y_val_return,
        val_predictions,
        zero_division=0,
    )
    precision = precision_score(
        y_val_return,
        val_predictions,
        zero_division=0,
    )

    threshold_results.append({
        "threshold": float(threshold_candidate),
        "f1": f1,
        "precision": precision,
    })

best = max(
    threshold_results,
    key=lambda result: (result["f1"], result["precision"]),
)
threshold = best["threshold"]


# 5. Evaluer V4 sans ajuster le modele ni le seuil
X4 = v4[FEATURES]
y4_return = (v4["churn"] == 0).astype(int)

v4_scores = model.predict_proba(X4)[:, return_column]
v4_predictions = (v4_scores >= threshold).astype(int)


# 6. Afficher les metriques
print("=" * 68)
print("CHURNGUARD - VALIDATION TEMPORELLE V3 VERS V4")
print("=" * 68)

print("\nProtocole :")
print(f"V3 : {len(v3)} clients")
print(f"Train V3 : {len(X_train)} clients")
print(f"Validation V3 : {len(X_val)} clients")
print(f"V4 : {len(v4)} clients")
print(f"Seuil choisi sur validation V3 : {threshold:.2f}")
print(f"F1 sur validation V3 : {best['f1']:.4f}")

print("\nResultats sur V4 :")
print(f"Clients V4 : {len(v4)}")
print(f"Retours reels : {int(y4_return.sum())}")
print(f"Taux de retour reel : {y4_return.mean():.2%}")
print(
    "Average Precision : "
    f"{average_precision_score(y4_return, v4_scores):.4f}"
)
print(
    "Precision retour : "
    f"{precision_score(y4_return, v4_predictions, zero_division=0):.4f}"
)
print(
    "Recall retour : "
    f"{recall_score(y4_return, v4_predictions, zero_division=0):.4f}"
)
print(
    "F1 retour : "
    f"{f1_score(y4_return, v4_predictions, zero_division=0):.4f}"
)

print("\nMatrice de confusion")
print("Lignes = realite [sans retour, retour]")
print("Colonnes = prediction [sans retour, retour]")
print(confusion_matrix(y4_return, v4_predictions, labels=[0, 1]))


# 7. Evaluer le classement des clients
ranking = pd.DataFrame({
    "score_retour": v4_scores,
    "retour_reel": y4_return.to_numpy(),
}).sort_values("score_retour", ascending=False)

total_returns = int(ranking["retour_reel"].sum())

print("\nClassement des clients sur V4")
print(
    "Part ciblee | Clients cibles | Retours detectes "
    "| Precision | Recall | Lift"
)

for percentage in [0.01, 0.02, 0.05, 0.10]:
    k = max(1, int(len(ranking) * percentage))
    top_clients = ranking.head(k)
    detected = int(top_clients["retour_reel"].sum())

    precision = detected / k
    recall = detected / total_returns if total_returns else 0
    lift = (
        precision / y4_return.mean()
        if y4_return.mean() > 0
        else 0
    )

    print(
        f"{percentage:>6.0%}      | "
        f"{k:>6}         | "
        f"{detected:>6}          | "
        f"{precision:>7.2%}  | "
        f"{recall:>7.2%} | "
        f"{lift:>5.2f}x"
    )

print(
    "\nNote : V4 est utilise uniquement pour l'evaluation, "
    "pas pour entrainer le modele ni choisir le seuil."
)
print("Aucun fichier de dataset n'a ete modifie.")
