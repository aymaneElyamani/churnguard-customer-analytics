
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

# 1. Charger les deux périodes
v1 = pd.read_csv("data/processed/churn_dataset_v1.csv")
v2 = pd.read_csv("data/processed/churn_dataset_v2.csv")

for name, df in [("V1", v1), ("V2", v2)]:
    required = FEATURES + ["customer_unique_id", "churn"]
    missing_columns = set(required) - set(df.columns)

    if missing_columns:
        raise ValueError(f"{name} : colonnes manquantes {missing_columns}")

    if df[required].isna().any().any():
        raise ValueError(f"{name} : valeurs manquantes détectées")

    if not df["churn"].isin([0, 1]).all():
        raise ValueError(f"{name} : la cible doit contenir 0 et 1")

# 2. Séparer V1 comme dans l'expérience précédente
X1 = v1[FEATURES]
y1 = v1["churn"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X1,
    y1,
    test_size=0.40,
    random_state=42,
    stratify=y1,
)

X_val, _, y_val, _ = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

# Classe positive pour nos métriques : client revenu (churn == 0)
y_val_return = (y_val == 0).astype(int)

# 3. Entraîner sur le train de V1 uniquement
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

# 4. Sélectionner le seuil sur la validation V1 uniquement
val_scores = model.predict_proba(X_val)[:, return_column]
threshold_results = []

for threshold in np.arange(0.05, 0.51, 0.05):
    val_predictions = (val_scores >= threshold).astype(int)
    f1 = f1_score(y_val_return, val_predictions, zero_division=0)
    precision = precision_score(
        y_val_return, val_predictions, zero_division=0
    )

    threshold_results.append({
        "threshold": float(threshold),
        "f1": f1,
        "precision": precision,
    })

best = max(
    threshold_results,
    key=lambda result: (result["f1"], result["precision"]),
)
threshold = best["threshold"]

# 5. Évaluer sur V2 sans ajuster le modèle ni le seuil
X2 = v2[FEATURES]
y2_return = (v2["churn"] == 0).astype(int)

v2_scores = model.predict_proba(X2)[:, return_column]
v2_predictions = (v2_scores >= threshold).astype(int)

print("=" * 65)
print("CHURNGUARD - VALIDATION TEMPORELLE")
print("=" * 65)

print("\nProtocole :")
print(f"V1 totale : {len(v1)} clients")
print(f"Train V1 : {len(X_train)} clients")
print(f"Validation V1 : {len(X_val)} clients")
print(f"V2 : {len(v2)} clients")
print(f"Seuil choisi sur validation V1 : {threshold:.2f}")
print(f"F1 sur validation V1 : {best['f1']:.4f}")

print("\nRésultats sur V2 :")
print(f"Clients V2 : {len(v2)}")
print(f"Retours réels : {int(y2_return.sum())}")
print(f"Taux de retour réel : {y2_return.mean():.2%}")
print(f"Average Precision : {average_precision_score(y2_return, v2_scores):.4f}")
print(f"Précision retour : {precision_score(y2_return, v2_predictions, zero_division=0):.4f}")
print(f"Recall retour : {recall_score(y2_return, v2_predictions, zero_division=0):.4f}")
print(f"F1 retour : {f1_score(y2_return, v2_predictions, zero_division=0):.4f}")

print("\nMatrice de confusion")
print("Lignes = réalité [sans retour, retour]")
print("Colonnes = prédiction [sans retour, retour]")
print(confusion_matrix(y2_return, v2_predictions, labels=[0, 1]))

print("\nClassement des clients sur V2")
ranking = pd.DataFrame({
    "score_retour": v2_scores,
    "retour_reel": y2_return.to_numpy(),
}).sort_values("score_retour", ascending=False)

total_returns = int(ranking["retour_reel"].sum())

print("Part ciblée | Clients ciblés | Retours détectés | Precision | Recall")

for percentage in [0.01, 0.02, 0.05, 0.10]:
    k = max(1, int(len(ranking) * percentage))
    top_clients = ranking.head(k)
    detected = int(top_clients["retour_reel"].sum())

    precision = detected / k
    recall = detected / total_returns if total_returns else 0

    print(
        f"{percentage:>6.0%}      | "
        f"{k:>6}         | "
        f"{detected:>6}          | "
        f"{precision:>7.2%}  | "
        f"{recall:>7.2%}"
    )

print("\nNote : V2 sert uniquement à l'évaluation, pas au choix du seuil.")
